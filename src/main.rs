use std::{cmp::Ordering, collections::{BTreeMap, BinaryHeap, HashMap}, fs::File, io::BufReader};

use serde::Deserialize;

#[derive(PartialEq, PartialOrd, Debug, Eq, Ord, Hash, Clone, Deserialize)]
enum Item {
    IronOre,
    CopperOre,
    GoldOre,
    Coal,
    Redstone,
    Sand,
    Stone,
    Wood,
    // intermediates
    IronIngot,
    CopperIngot,
    GoldIngot,
    SteelIngot,
    Glass,
    Plank,
    Stick,
    IronPlate,
    IronGear,
    Pipe,
    CopperWire,
    GoldWire,
    Circuit,
    AdvancedCircuit,
    Casing,
    Motor,
    Battery,
    Lamp,
    Furnace,
    Conveyor,
    Machine,
    // end products
    AdvancedMachine,
    Generator,
    Pump,
    Drill,
    Pickaxe,
    SolarPanel,
    Assembler,
}

type Inventory = BTreeMap<Item, u32>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct State {
    inventory: Inventory
}

#[derive(Deserialize)]
struct Recipe {
    name: String,
    inputs: Vec<(Item, u32)>,
    outputs: Vec<(Item, u32)>,
}

#[derive(Clone, Debug)]
struct Node {
    state: State,
    cost: u32,
    priority: u32
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        other.priority.cmp(&self.priority)
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for Node {}

fn check_is_goal(state: &State, goal: &State) -> bool {
    goal.inventory.iter().all(|(item, requred)| {
        let current = state.inventory.get(item).copied().unwrap_or(0);

        current >= *requred
    }) 
}

fn can_craft(state: &State, recipe: &Recipe) -> bool {
    recipe.inputs.iter().all(|(item, required)| {
        let available = state.inventory.get(item).copied().unwrap_or(0);

        available >= *required
    })
}

fn perform_craft(state: &State, recipe: &Recipe) -> State {
    let mut inventory = state.inventory.clone();

    for (item, amount) in &recipe.inputs {
        let left = inventory[item] - amount;
        if left == 0 {
            inventory.remove(item);
        } else {
            inventory.insert(item.clone(), left);
        }
    }

    for (item, amount) in &recipe.outputs {
        *inventory.entry(item.clone()).or_insert(0) += amount;
    }

    State { inventory }
}

/// The index of the recipe that produced the state, along with the state.
fn neighbors(
    state: &State,
    recipes: &[Recipe],
) -> Vec<(usize, State)> {
    let mut result = Vec::new();

    for (recipe_index, recipe) in recipes.iter().enumerate() {
        if can_craft(state, recipe) {
            let next_state = perform_craft(state, recipe);

            result.push((recipe_index, next_state));
        }
    }

    result
}

//create the maximum amount of an item that can be produced by any recipe, as a lookup table.
fn build_max_outputs(
    recipes: &[Recipe],
) -> HashMap<Item, u32> {
    let mut max_outputs = HashMap::new();

    for recipe in recipes {
        for (item, amount) in &recipe.outputs {
            let current = max_outputs.entry(item.clone()).or_insert(0);

            if *amount > *current {
                *current = *amount;
            }
        }
    }

    max_outputs
}

//ignores counts and measures the longest recipe chain that we can make over all the items we have.
//downside is it doesn't track quantity
fn stronger_heuristic(state: &State, goal: &State, recipes: &[Recipe]) -> u32 {
    //basucally how many crafts until i can have this item. anything we have is initialized to 0. we dont care about quantity
    let mut cost: HashMap<&Item, u32> = state.inventory.keys().map(|i| (i, 0)).collect();
    loop {
        let mut changed = false; //keep sweeping over the recipes until a pass improves nothing

        for r in recipes {

            //check the depth of each input of the recipe, collects into Vec<u32> or gives none if any input isn't in the table yet.
            //if none then skip the recipe
            let Some(ins) = r.inputs.iter().map(|(i, _)| cost.get(i).copied()).collect::<Option<Vec<_>>>() else { continue };
            
            //take the heighest cost
            let c = 1 + ins.into_iter().max().unwrap_or(0);

            //for each output that isn't in the table, or this reaches in fewer crafts than the one in the table, record the new depth.
            //this is what accounts for cheaper recipes
            //make it run another pass with changed = true, since this can unlock new recipes
            for (o, _) in &r.outputs {
                if cost.get(o).map_or(true, |&old| c < old) {
                    cost.insert(o, c);
                    changed = true;
                }
            }
        }
        if !changed { break; }
    }
    //take max depth seen of all the goal items in the cost table. If it is u32::Max then it was never craftable.
    goal.inventory.keys().map(|i| cost.get(i).copied().unwrap_or(u32::MAX)).max().unwrap_or(0)
}

//only looks at the goal itself. For each goal item that we don't have it checks how many recipes it would at take at least, then takes the max across all items
//it is weak because it never looks at the inputs, so for single item goals it returns just 1.
fn very_weak_heuristic(
    state: &State,
    goal: &State,
    max_outputs: &HashMap<Item, u32>,
) -> u32 {
    let mut estimate = 0;

    for (item, required_amount) in &goal.inventory {
        let current_amount =
            state.inventory.get(&item).copied().unwrap_or(0);

        if current_amount >= *required_amount {
            continue;
        }

        let missing = required_amount - current_amount;

        let Some(&max_per_craft) = max_outputs.get(&item) else {
            return u32::MAX;
        };

        // ceil(missing / max_per_craft)
        let crafts_needed = missing.div_ceil(max_per_craft);

        estimate = estimate.max(crafts_needed);
    }

    estimate
}

//index of recipe used to get there
type Parent = (State, usize);

//max_supply and goal_possible aren't required for the actual implementation, but can tell when it is impossible to reach the goal without getting stuck looping forever
fn max_supply(state: &State, recipes: &[Recipe]) -> HashMap<Item, u64> {
    let held: HashMap<Item, u64> = state.inventory.iter().map(|(i, n)| (i.clone(), *n as u64)).collect();
    let mut avail = held.clone();
    for _ in 0..64 {
        let mut next = held.clone();
        for r in recipes {
            let runs = r.inputs.iter()
                .map(|(i, n)| avail.get(i).copied().unwrap_or(0) / *n as u64)
                .min().unwrap_or(0);
            for (o, n) in &r.outputs {
                *next.entry(o.clone()).or_insert(0) += runs * *n as u64;
            }
        }
        if next == avail { break; }
        avail = next;
    }
    avail
}

fn goal_possible(state: &State, goal: &State, recipes: &[Recipe]) -> bool {
    let avail = max_supply(state, recipes);
    goal.inventory.iter().all(|(i, n)| avail.get(i).copied().unwrap_or(0) >= *n as u64)
}

//https://en.wikipedia.org/wiki/A*_search_algorithm
fn astar(
    start: State,
    goal: State,
    recipes: &[Recipe]
) -> Option<Vec<usize>> {
    let mut priority_queue: BinaryHeap<Node> = BinaryHeap::new();

    let max_outputs = build_max_outputs(recipes);

    let h = stronger_heuristic(
        &start,
        &goal,
        &recipes,
    );

    if h == u32::MAX || !goal_possible(&start, &goal, recipes) {
        println!("IMPOSSIBLE");
        //the heuristic returns max if it is unreachable, the second function checks if it is possible given starting items
        return None;
    }

    let mut came_from_list: HashMap<State, Parent> = HashMap::new();

    priority_queue.push(Node {
        state: start.clone(),
        cost: 0,
        priority: h
    });

    let mut best_cost: HashMap<State, u32> =
        HashMap::new();

    best_cost.insert(start, 0);

    while let Some(node) = priority_queue.pop() {
        if check_is_goal(&node.state, &goal) {
            println!("YiPPEE {:?}", node);

            //walk it back now
            let mut path = vec![];

            let mut current = node.state;

            while let Some((prev, i)) = came_from_list.get(&current) {
                path.push(*i);
                current = prev.clone();
            }

            path.reverse();

            return Some(path);
        }
        if node.cost > best_cost[&node.state] { continue; }

        for (index, next) in neighbors(&node.state, recipes) {
            let newcost = node.cost + 1;

            let oldcost = best_cost
                .get(&next)
                .copied()
                .unwrap_or(u32::MAX);

            if newcost < oldcost {
                best_cost.insert(
                    next.clone(),
                    newcost,
                );

                came_from_list.insert(
                    next.clone(),
                    (
                        node.state.clone(),
                        index,
                    ),
                );

                let h = stronger_heuristic(
                    &next,
                    &goal,
                    &recipes,
                );

                if h == u32::MAX {
                    continue;
                }

                let priority = newcost.saturating_add(h);

                priority_queue.push(Node {
                    state: next,
                    cost: newcost,
                    priority,
                });
            }
        }
    }
    None
}

pub fn main() {
    let recipes: Vec<Recipe> = serde_json::from_reader(BufReader::new(File::open("testrecipe.json").unwrap())).unwrap();

    let path = astar(
        State {
            inventory: BTreeMap::from([(Item::CopperOre, 114), (Item::IronOre, 124), (Item::Wood, 100), (Item::GoldOre, 1000), (Item::Sand, 3), (Item::Redstone, 100)])
        },
        State {
            inventory: BTreeMap::from([(Item::SolarPanel, 1)])
        },
        &recipes
    ).unwrap();

    println!("{:?}", path.iter().map(|v| &recipes[*v].name).collect::<Vec<_>>());
}