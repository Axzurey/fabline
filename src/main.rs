use std::{cmp::Ordering, collections::{BTreeMap, BinaryHeap, HashMap, HashSet}, fs::{self, File}, io::BufReader, ops::Sub, path::Path, time::Instant};

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
    //the strongest one
    Demand
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
        cache: &mut HashMap<State, u32>,
        potentials: &Potentials
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
            },
            Self::Demand => {
                potential_heuristic(potentials, state, goal)
            }
        }, Instant::now().sub(instant).as_secs_f32())
    }
}

type Potentials = HashMap<Item, f64>;

/// Items with no recipe (raw) are worth 0.
fn build_potentials(recipes: &[Recipe], mode: &OptimizeMode) -> Potentials {
    let mut c: Potentials = HashMap::new();

    for recipe in recipes {
        for (output, _) in &recipe.outputs {
            c.insert(output.clone(), f64::INFINITY);
        }
    }

    loop {
        let mut changed = false;

        for recipe in recipes {
            let budget = mode.weight(recipe) as f64
                + recipe
                    .inputs
                    .iter()
                    .map(|(item, n)| c.get(item).copied().unwrap_or(0.0) * *n as f64)
                    .sum::<f64>();

            // Split the budget across output kinds so the sum of outputs
            // can never exceed it.
            let kinds = recipe.outputs.len() as f64;

            for (output, n) in &recipe.outputs {
                let share = budget / (kinds * *n as f64);

                if share < c[output] - 1e-9 {
                    c.insert(output.clone(), share);
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }

    // Items that can never be produced stay infinite; zero keeps the bound valid.
    for v in c.values_mut() {
        if v.is_infinite() {
            *v = 0.0;
        }
    }

    c
}

fn potential_of(inventory: &Inventory, c: &Potentials) -> f64 {
    inventory
        .iter()
        .map(|(item, n)| c.get(item).copied().unwrap_or(0.0) * *n as f64)
        .sum()
}

fn potential_heuristic(c: &Potentials, state: &State, goal: &State) -> u32 {
    let h = potential_of(&goal.inventory, c) - potential_of(&state.inventory, c);

    if h <= 1e-9 {
        0
    } else {
        (h - 1e-9).ceil() as u32
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
    cache: &mut HashMap<State, u32>,
) -> u32 {
    if !config.memoize_heuristic {
        return stronger_heuristic(config, state, goal, recipes);
    }
    let key = state.clone();
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

use std::collections::BTreeSet;

fn prune_recipes(recipes: &[Recipe], goal: &Inventory) -> Vec<Recipe> {
    // Start with the items needed
    let mut needed: BTreeSet<Item> = goal.keys().cloned().collect();

    loop {
        let mut changed = false;

        // Find every recipe that can produce a needed item
        for recipe in recipes {
            let produces_needed = recipe.outputs.iter()
                .any(|(item, _)| needed.contains(item));

            if produces_needed {
                // Its ingredients must also be obtainable
                for (item, _) in &recipe.inputs {
                    if needed.insert(item.clone()) {
                        changed = true;
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }

    // Keep all recipes that produce any item
    recipes.iter()
        .filter(|recipe| {
            recipe.outputs.iter()
                .any(|(item, _)| needed.contains(item))
        })
        .cloned()
        .collect()
}

const MAX_EXPANDED: usize = 10_000_000;
const MAX_STATES: usize = 20_000_000;

fn expand_demand(
    item: &Item,
    amount: u32,
    recipes: &[Recipe],
    caps: &mut HashMap<Item, u32>,
    visiting: &mut HashSet<Item>,
) -> bool {
    if amount == 0 {
        return true;
    }

    // If we encounter a cycle, these bounds aren't safe to use.
    if !visiting.insert(item.clone()) {
        return false;
    }

    let entry = caps.entry(item.clone()).or_insert(0);
    *entry = entry.saturating_add(amount);

    for recipe in recipes {
        for (output, output_amount) in &recipe.outputs {
            if output != item || *output_amount == 0 {
                continue;
            }

            let runs = amount.div_ceil(*output_amount);

            for (input, input_amount) in &recipe.inputs {
                let needed = runs.saturating_mul(*input_amount);

                if !expand_demand(
                    input,
                    needed,
                    recipes,
                    caps,
                    visiting,
                ) {
                    visiting.remove(item);
                    return false;
                }
            }
        }
    }

    visiting.remove(item);
    true
}

fn build_item_caps(
    goal: &Inventory,
    recipes: &[Recipe],
) -> Option<HashMap<Item, u32>> {
    let mut caps = HashMap::new();

    for (item, amount) in goal {
        if !expand_demand(
            item,
            *amount,
            recipes,
            &mut caps,
            &mut HashSet::new(),
        ) {
            // Do not apply these bounds if a dependency cycle exists.
            return None;
        }
    }

    Some(caps)
}

fn cap_state(
    state: &State,
    caps: &HashMap<Item, u32>,
) -> State {
    let inventory = state.inventory.iter()
        .filter_map(|(item, amount)| {
            let cap = caps.get(item)?;

            let useful_amount = (*amount).min(*cap);

            if useful_amount == 0 {
                None
            } else {
                Some((item.clone(), useful_amount))
            }
        })
        .collect();

    State { inventory }
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
    
    let recipes = &prune_recipes(recipes, &config.target_state.inventory);

    let mut nodes_expanded = 0;
    let mut nodes_generated = 0;
    let mut max_open_list_size = 0;
    let clock_start = Instant::now();
    let mut time_spend_in_heuristic = 0.0;

    let Some(item_caps) = build_item_caps(
        &config.target_state.inventory,
        &recipes,
    ) else {
        println!("Dependency cycle detected; skipping item caps");
        return None;
    };

    let origin = cap_state(&config.origin_state, &item_caps);
    let potentials = build_potentials(recipes, &config.optimize_mode);

    let (h, he) = config.heuristic.heuristic(
        &config,
        &origin,
        &config.target_state,
        &recipes,
        &max_outputs,
        &min_weights,
        &mut heuristic_cache,
        &potentials
    );
    
    time_spend_in_heuristic += he;

    let initial_heuristic_cost = h;

    if h == u32::MAX || !goal_possible(&origin, &config.target_state, recipes) {
        println!("IMPOSSIBLE");
        //the heuristic returns max if it is unreachable, the second function checks if it is possible given starting items
        return None;
    }

    let mut came_from_list: HashMap<State, Parent> = HashMap::new();

    priority_queue.push(Node {
        state: origin.clone(),
        cost: 0,
        priority: h,
        tie_break_key: config.tie_breaker.key(0)
    });
    nodes_generated += 1;

    let mut best_cost: HashMap<State, u32> =
        HashMap::new();

    best_cost.insert(origin.clone(), 0);

    while let Some(node) = priority_queue.pop() {

        if nodes_expanded as usize >= MAX_EXPANDED || best_cost.len() >= MAX_STATES {
            println!("Search limit reached");
            break;
        }

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
        // if nodes_expanded % 10_000 == 0 {
        //     println!(
        //         "expanded={}, unique={}, open={}, g={}, h={}, f={}",
        //         nodes_expanded,
        //         best_cost.len(),
        //         priority_queue.len(),
        //         node.cost,
        //         node.priority.saturating_sub(node.cost),
        //         node.priority,
        //     );
        // }

        for (index, next) in neighbors(&node.state, recipes) {
            let next = cap_state(&next, &item_caps);

            let newcost = node.cost.saturating_add(
                config.optimize_mode.weight(&recipes[index])
            );

            let oldcost = best_cost
                .get(&next)
                .copied()
                .unwrap_or(u32::MAX);

            if newcost < oldcost {
                let (h, he) = config.heuristic.heuristic(
                    &config,
                    &next,
                    &config.target_state,
                    &recipes,
                    &max_outputs,
                    &min_weights,
                    &mut heuristic_cache,
                    &potentials
                );

                time_spend_in_heuristic += he;

                if h == u32::MAX {
                    continue;
                }

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

#[derive(Debug, Serialize)]
//changes the node that the algorithm explores first when 2 nodes have the same heuristic value
enum TieBreaker {
    // Prefer smaller g(n), may explore more nodes but can find better solution. ONLY really reasonable for greedy
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
            optimize_mode: OptimizeMode::Nodes,
            origin_state: State {
                inventory: BTreeMap::from([
                    (Item::CopperOre, 142),
                    (Item::IronOre, 165),
                    (Item::Wood, 142),
                    (Item::GoldOre, 142),
                    (Item::Sand, 150),
                    (Item::Redstone, 170),
                ]),
            },
            target_state: State {
                inventory: BTreeMap::from([(Item::SolarPanel, 6)]),
            },
            heuristic: Heuristic::Strong,
            memoize_heuristic: false,
            algorithm: Algorithm::AStar,
            tie_breaker: TieBreaker::HigherCost
        },
        &recipes,
    ).unwrap();

    println!("{} {}", path.elapsed_time, path.time_spend_in_heuristic);
    
    path.write_to_file("./astar-weak.json");
}