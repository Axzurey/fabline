use std::{cmp::Ordering, collections::{BTreeMap, BinaryHeap, HashMap}, fs::{self, File}, io::BufReader, ops::Sub, path::Path, time::Instant};

use serde::{Deserialize, Serialize, ser::SerializeMap};

#[derive(PartialEq, PartialOrd, Debug, Eq, Ord, Hash, Clone, Deserialize, Serialize)]
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

impl Serialize for State {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer
    {
        let len = self.inventory.len();
        let mut map = serializer.serialize_map(Some(len))?;

        for (k, v) in &self.inventory {
            map.serialize_entry(k, v)?;
        }
        map.end()
    }
}

#[derive(Deserialize, Debug, Serialize, Clone)]
struct Recipe {
    name: String,
    inputs: Vec<(Item, u32)>,
    outputs: Vec<(Item, u32)>,
    energy_cost: u32
}

#[derive(Clone, Debug)]
struct Node {
    state: State,
    cost: u32,
    priority: u32,
    tie_break_key: u32,
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        other.priority.cmp(&self.priority)
            .then(self.tie_break_key.cmp(&other.tie_break_key))
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
            && self.tie_break_key == other.tie_break_key
    }
}

impl Eq for Node {}

#[derive(Debug, Serialize)]
enum OptimizeMode {
    //or any other generic parameter
    Energy,
    //all edge weights are 1
    Nodes
}

impl OptimizeMode {
    pub fn weight(&self, r: &Recipe) -> u32 {
        match self {
            OptimizeMode::Nodes => 1,
            OptimizeMode::Energy => r.energy_cost,
        }
    }
}

#[derive(Debug, Serialize)]
enum Heuristic {
    //the maximum of both
    Max,
    Strong,
    Weak,
}

impl Heuristic {
    pub fn heuristic(
        &self, 
        config: &TraversalConfig, 
        state: &State, 
        goal: &State, 
        recipes: &[Recipe], 
        max_outputs: &HashMap<Item, u32>, 
        min_weights: &HashMap<Item, u32>, 
        cache: &mut HashMap<u128, u32>
    ) -> (u32, f32) {
        if config.algorithm == Algorithm::Ucs {return (0, 0.0)}
        let instant = Instant::now();
        (match self {
            Self::Strong => {
                stronger_heuristic_cached(config, state, goal, recipes, cache)
            },
            Self::Weak => {
                very_weak_heuristic(min_weights, state, goal, max_outputs)
            },
            Self::Max => {
                stronger_heuristic_cached(config, state, goal, recipes, cache).max(very_weak_heuristic(min_weights, state, goal, max_outputs))
            }
        }, Instant::now().sub(instant).as_secs_f32())
    }
}

#[derive(Debug, Serialize, PartialEq)]
enum Algorithm {
    Ucs,
    Greedy,
    AStar
}

#[derive(Debug, Serialize)]
struct TraversalConfig {
    pub optimize_mode: OptimizeMode,
    pub origin_state: State,
    pub target_state: State,
    pub heuristic: Heuristic,
    pub algorithm: Algorithm,
    //only for the strong heuristic rn
    pub memoize_heuristic: bool,
    //has a significant impact on the amount of nodes expanded by greedy vs a*
    pub tie_breaker: TieBreaker
}

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

fn stronger_heuristic_cached(
    config: &TraversalConfig,
    state: &State,
    goal: &State,
    recipes: &[Recipe],
    cache: &mut HashMap<u128, u32>,
) -> u32 {
    if !config.memoize_heuristic {
        return stronger_heuristic(config, state, goal, recipes);
    }
    let key = held_mask(state);
    if let Some(&h) = cache.get(&key) {
        return h;
    }
    let h = stronger_heuristic(config, state, goal, recipes);
    cache.insert(key, h);
    h
}

//ignores counts and measures the longest recipe chain that we can make over all the items we have.
//downside is it doesn't track quantity
fn stronger_heuristic(config: &TraversalConfig, state: &State, goal: &State, recipes: &[Recipe]) -> u32 {
    //basucally how many crafts until i can have this item. anything we have is initialized to 0. we dont care about quantity
    let mut cost: HashMap<&Item, u32> = state.inventory.keys().map(|i| (i, 0)).collect();
    loop {
        let mut changed = false; //keep sweeping over the recipes until a pass improves nothing

        for r in recipes {

            //check the depth of each input of the recipe, collects into Vec<u32> or gives none if any input isn't in the table yet.
            //if none then skip the recipe
            let Some(ins) = r.inputs.iter().map(|(i, _)| cost.get(i).copied()).collect::<Option<Vec<_>>>() else { continue };
            
            //take the heighest cost + the cost of the recipe, either as 1 or the energy usage
            let c = config.optimize_mode.weight(r).saturating_add(ins.into_iter().max().unwrap_or(0));

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
fn very_weak_heuristic(
    min_weights: &HashMap<Item, u32>,
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

        estimate = estimate.max(crafts_needed.saturating_mul(min_weights[item])); //multiplied by the lowest weight of this item
    }

    estimate
}

fn build_min_weights(recipes: &[Recipe], mode: &OptimizeMode) -> HashMap<Item, u32> {
    let mut m: HashMap<Item, u32> = HashMap::new();
    for r in recipes {
        let w = mode.weight(r);
        for (o, _) in &r.outputs {
            m.entry(o.clone()).and_modify(|c| *c = (*c).min(w)).or_insert(w);
        }
    }
    m
}

//index of recipe used to get there
type Parent = (State, usize);

//max_supply and goal_possible aren't required for the actual implementation, but can tell when it is impossible to reach the goal without getting stuck looping forever
fn max_supply(state: &State, recipes: &[Recipe]) -> Option<HashMap<Item, u64>> {
    let held: HashMap<Item, u64> = state.inventory.iter().map(|(i, n)| (i.clone(), *n as u64)).collect();
    let mut avail = held.clone();
    for _ in 0..128 {
        let mut next = held.clone();
        for r in recipes {
            let runs = r.inputs.iter()
                .map(|(i, n)| avail.get(i).copied().unwrap_or(0) / *n as u64)
                .min().unwrap_or(0);
            for (o, n) in &r.outputs {
                let e = next.entry(o.clone()).or_insert(0);
                *e = e.saturating_add(runs.saturating_mul(*n as u64));
            }
        }
        if next == avail { return Some(avail); }
        avail = next;
    }
    None // didn't converge: unbounded growth somewhere, can't prove anything
}

//returns true if it can't prove that the search doesn't terminate, or if it can prove it will terminate
fn goal_possible(state: &State, goal: &State, recipes: &[Recipe]) -> bool {
    let Some(avail) = max_supply(state, recipes) else { return true };
    goal.inventory.iter().all(|(i, n)| avail.get(i).copied().unwrap_or(0) >= *n as u64)
}

//https://en.wikipedia.org/wiki/A*_search_algorithm
fn run_algo(
    config: TraversalConfig,
    recipes: &[Recipe]
) -> Option<Metrics> {
    let mut heuristic_cache = HashMap::new();

    let mut priority_queue: BinaryHeap<Node> = BinaryHeap::new();

    let max_outputs = build_max_outputs(recipes);
    let min_weights = build_min_weights(recipes, &config.optimize_mode);

    let mut nodes_expanded = 0;
    let mut nodes_generated = 0;
    let mut max_open_list_size = 0;
    let clock_start = Instant::now();
    let mut time_spend_in_heuristic = 0.0;

    let (h, he) = config.heuristic.heuristic(
        &config,
        &config.origin_state,
        &config.target_state,
        &recipes,
        &max_outputs,
        &min_weights,
        &mut heuristic_cache
    );
    time_spend_in_heuristic += he;

    let initial_heuristic_cost = h;

    if h == u32::MAX || !goal_possible(&config.origin_state, &config.target_state, recipes) {
        println!("IMPOSSIBLE");
        //the heuristic returns max if it is unreachable, the second function checks if it is possible given starting items
        return None;
    }

    let mut came_from_list: HashMap<State, Parent> = HashMap::new();

    priority_queue.push(Node {
        state: config.origin_state.clone(),
        cost: 0,
        priority: h,
        tie_break_key: config.tie_breaker.key(0)
    });
    nodes_generated += 1;

    let mut best_cost: HashMap<State, u32> =
        HashMap::new();

    best_cost.insert(config.origin_state.clone(), 0);

    while let Some(node) = priority_queue.pop() {

        if node.cost > best_cost[&node.state] { continue; }

        if check_is_goal(&node.state, &config.target_state) {

            //walk it back now
            let mut path = vec![];

            let mut current = node.state;

            while let Some((prev, i)) = came_from_list.get(&current) {
                path.push(*i);
                current = prev.clone();
            }

            path.reverse();

            let metrics = Metrics {
                nodes_expanded,
                nodes_generated,
                max_open_list_size,
                elapsed_time: Instant::now().sub(clock_start).as_secs_f32(),
                solution_cost: path.iter().map(|&i| config.optimize_mode.weight(&recipes[i])).sum(),
                path_length: path.len() as u32,
                result_path: path.iter().map(|v| recipes[*v].clone()).collect(),
                initial_heuristic_cost,
                time_spend_in_heuristic,
                config
            };

            return Some(metrics);
        }

        nodes_expanded += 1;

        for (index, next) in neighbors(&node.state, recipes) {
            let newcost = node.cost.saturating_add(config.optimize_mode.weight(&recipes[index]));

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

                let (h, he) = config.heuristic.heuristic(
                    &config,
                    &next,
                    &config.target_state,
                    &recipes,
                    &max_outputs,
                    &min_weights,
                    &mut heuristic_cache
                );

                time_spend_in_heuristic += he;

                if h == u32::MAX {
                    continue;
                }

                let priority = match config.algorithm {
                    Algorithm::Ucs => newcost,
                    Algorithm::Greedy => h,
                    Algorithm::AStar => newcost.saturating_add(h),
                };

                priority_queue.push(Node {
                    state: next,
                    cost: newcost,
                    priority,
                    tie_break_key: config.tie_breaker.key(newcost),
                });

                if priority_queue.len() as u32 > max_open_list_size {
                    max_open_list_size = priority_queue.len() as u32
                }
                nodes_generated += 1;
            }
        }
    }
    None
}

fn held_mask(state: &State) -> u128 {
    state.inventory.keys().fold(0u128, |m, i| m | (1u128 << (i.clone() as u32)))
}

#[derive(Debug, Serialize)]
//changes the node that the algorithm explores first when 2 nodes have the same heuristic value
enum TieBreaker {
    // Prefer smaller g(n), may explore more nodes but can find better solution
    LowerCost,
    // Prefer larger g(n), may get to goal faster but may be a more expensive solution
    HigherCost,
}

impl TieBreaker {
    fn key(&self, cost: u32) -> u32 {
        match self {
            TieBreaker::LowerCost => u32::MAX - cost,
            TieBreaker::HigherCost => cost,
        }
    }
}

#[derive(Debug, Serialize)]
struct Metrics {
    nodes_expanded: u32,
    nodes_generated: u32,
    max_open_list_size: u32,
    elapsed_time: f32,
    solution_cost: u32,
    path_length: u32,
    result_path: Vec<Recipe>,

    initial_heuristic_cost: u32,
    time_spend_in_heuristic: f32,
    config: TraversalConfig,
}

impl Metrics {
    pub fn write_to_file<T: AsRef<Path>>(&self, out: T) {
        let json = serde_json::to_string(&self).unwrap();

        fs::write(out, json).unwrap();
    }
}

pub fn main() {
    let recipes: Vec<Recipe> = serde_json::from_reader(BufReader::new(File::open("testrecipe.json").unwrap())).unwrap();

    let path = run_algo(
    TraversalConfig {
            optimize_mode: OptimizeMode::Energy,
            origin_state: State {
                inventory: BTreeMap::from([
                    (Item::CopperOre, 14),
                    (Item::IronOre, 12),
                    (Item::Wood, 10),
                    (Item::GoldOre, 10),
                    (Item::Sand, 3),
                    (Item::Redstone, 10),
                ]),
            },
            target_state: State {
                inventory: BTreeMap::from([(Item::SolarPanel, 1)]),
            },
            heuristic: Heuristic::Weak,
            memoize_heuristic: true,
            algorithm: Algorithm::AStar,
            tie_breaker: TieBreaker::LowerCost
        },
        &recipes,
    ).unwrap();

    println!("{:#?}", path);
    
    path.write_to_file("./astar-weak.json");
}