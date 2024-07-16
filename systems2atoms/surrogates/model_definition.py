import warnings
import pathlib
this_file = pathlib.Path(__file__).parent.resolve()

def model_initializer():
    try:
        import pyomo.environ as pyo
        from pyomo.gdp import Disjunction
        from omlt import OmltBlock
        from omlt.linear_tree import LinearTreeGDPFormulation, LinearTreeDefinition
    except ImportError:
        warnings.warn('Could not import Linear Tree modules from OMLT. Ensure you have the latest version of OMLT.')

    try:
        from lineartree.lineartree import tree_from_json
    except ImportError:
        warnings.warn('linear-tree has no "tree_from_json(). You will not be able to load the surrogate LMDTs!')

    # Initialize Pyomo model
    model = pyo.ConcreteModel()

    # Initialize OMLT blocks
    model.component_block = OmltBlock()
    model.costing_block = OmltBlock()
    model.nanoparticle_block = OmltBlock()

    # Load surrogate models into Python
    component_surrogate = tree_from_json(this_file/pathlib.Path('models/component_surrogate.json'))
    costing_surrogate = tree_from_json(this_file/pathlib.Path('models/costing_surrogate.json'))
    nanoparticle_surrogate = tree_from_json(this_file/pathlib.Path('models/nanoparticle_surrogate.json'))

    # Load surrogate models into OMLT
    component_surrogate = LinearTreeDefinition(component_surrogate, unscaled_input_bounds = {
        0: (0.1, 2), # CSTR Volume (m^3)
        1: (0.3, 1), # Pellet Effectiveness Factor (fraction)
        2: (0.1, 0.5), # Dehdrogenation Reaction Order 
        3: (1e-3, 1e3), # Turnover Frequency (1/s)
        4: (10, 1000), # Catalyst metal amount (mol)
        })

    costing_surrogate = LinearTreeDefinition(costing_surrogate, unscaled_input_bounds ={
        0: (0.1, 2), # CSTR Volume (m^3)
        1: (100, 1500), # Station Capacity (kg/day)
        2: (25, 550), # Dehydrogenation Catalyst Amount (kg)
        3: (0.7, 1), # Dehydrogenation Reaction Yield (fraction)
        4: (300, 400), # Dehydrogenation Reaction Temperature (K)
        5: (1, 100), # Dehydrogenation Reaction Pressure (bar)
        6: (5, 40000), # Dehydrogenation Catalyst Cost
        })

    nanoparticle_surrogate = LinearTreeDefinition(nanoparticle_surrogate, unscaled_input_bounds={
        0: (0, 2), # Catalyst metal choice (0 = Pd, 1 = Cu, 2 = Pt)
        1: (300, 400), # Dehydrogenation Reaction Temperature (K)
        2: (1, 100), # Dehydrogenation Reaction Pressure (bar)
        3: (3, 5) # Nanoparticle diameter (nm)
        })

    # Define equation formulations for each surrogate model                                       
    component_formulation = LinearTreeGDPFormulation(component_surrogate)
    costing_formulation = LinearTreeGDPFormulation(costing_surrogate)
    nanoparticle_formulation = LinearTreeGDPFormulation(nanoparticle_surrogate)

    # Convert equations into Pyomo blocks
    model.component_block.build_formulation(component_formulation)
    model.costing_block.build_formulation(costing_formulation)
    model.nanoparticle_block.build_formulation(nanoparticle_formulation)

    # Define free variables inside of the model
    model.catalyst_type = pyo.Var(bounds = (0, 2), domain = pyo.Integers) # 0 = Pd, 1 = Cu, 2 = Pt
    model.nanoparticle_diameter = pyo.Var(bounds = (3, 5)) #nm
    model.temperature = pyo.Var(bounds = (300, 400)) #K
    model.pressure = pyo.Var(bounds = (1, 100)) #atm

    model.turnover_frequency = pyo.Var(bounds = (1e-3, 1e3))

    model.cstr_volume = pyo.Var(bounds = (0.1, 2))
    model.pellet_effectiveness_factor = pyo.Var(bounds = (0.3, 1))
    model.reaction_order = pyo.Var(bounds = (0.1, 0.5))
    model.catalyst_amount = pyo.Var(bounds = (10, 1000)) # mol of metal

    model.H2Yield = pyo.Var(bounds = (0.70, 1))

    model.station_capacity = pyo.Var(bounds = (100, 1500))
    model.catalyst_mass = pyo.Var(bounds = (1, 550)) # kg of metal
    model.catalyst_price = pyo.Var(bounds = (5, 40000))

    model.cost = pyo.Var()

    # Add constraints to connect the variables to each surroage
    model.connections_constraints = pyo.ConstraintList()

    # Constraints for the nanoparticle surrogate
    model.connections_constraints.add(model.catalyst_type == model.nanoparticle_block.inputs[0])
    model.connections_constraints.add(model.temperature == model.nanoparticle_block.inputs[1])
    model.connections_constraints.add(model.pressure == model.nanoparticle_block.inputs[2])
    model.connections_constraints.add(model.nanoparticle_diameter == model.nanoparticle_block.inputs[3])

    model.connections_constraints.add(model.turnover_frequency == model.nanoparticle_block.outputs[0])

    # Constraints for the component surrogate
    model.connections_constraints.add(model.cstr_volume == model.component_block.inputs[0])
    model.connections_constraints.add(model.pellet_effectiveness_factor == model.component_block.inputs[1])
    model.connections_constraints.add(model.reaction_order == model.component_block.inputs[2])
    model.connections_constraints.add(model.turnover_frequency == model.component_block.inputs[3])
    model.connections_constraints.add(model.catalyst_amount == model.component_block.inputs[4])

    model.connections_constraints.add(model.H2Yield <= model.component_block.outputs[0])

    # Constraints for the costing surrogate
    model.connections_constraints.add(model.cstr_volume == model.costing_block.inputs[0])
    model.connections_constraints.add(model.station_capacity == model.costing_block.inputs[1])
    model.connections_constraints.add(model.catalyst_mass == model.costing_block.inputs[2])
    model.connections_constraints.add(model.H2Yield == model.costing_block.inputs[3])
    model.connections_constraints.add(model.temperature == model.costing_block.inputs[4])
    model.connections_constraints.add(model.pressure == model.costing_block.inputs[5])
    model.connections_constraints.add(model.catalyst_price == model.costing_block.inputs[6])

    model.connections_constraints.add(model.cost == model.costing_block.outputs[0])

    # Add discrete choice of catalyst
    model.catalyst_choice = Disjunction(expr = [
        [ # Pd
            model.catalyst_type == 0,
            model.catalyst_price == 38000,
            model.catalyst_mass == model.catalyst_amount * 106.42 / 1000, # molar mass of Pd
            model.catalyst_mass <= model.cstr_volume * 12.02 * 1000 * 0.003 # density of Pd, 0.3% volume loading
        ], 
        [ # Cu
            model.catalyst_type == 1,
            model.catalyst_price == 8,
            model.catalyst_mass == model.catalyst_amount * 63.546 / 1000, # molar mass of Cu
            model.catalyst_mass <= model.cstr_volume * 8.96 * 1000 * 0.003 # density of Cu, 0.3% volume loading
        ],
        [ # Pt
            model.catalyst_type == 2,
            model.catalyst_price == 28000,
            model.catalyst_mass == model.catalyst_amount * 195.08 / 1000, # molar mass of Pt
            model.catalyst_mass  <= model.cstr_volume * 21.4 * 1000 * 0.003 # density of Pt, 0.3% volume loading
        ],
    ])

    model.economic_constraints = pyo.ConstraintList()
    model.economic_constraints.add(
        model.cstr_volume <= model.station_capacity / 500
    )

    # Transform disjunctions
    pyo.TransformationFactory('gdp.hull').apply_to(model)

    # Define optimzation objective
    model.obj = pyo.Objective(expr = model.cost, sense = pyo.minimize)
    return model


def solve(model, tee = True, solver = 'glpk'):
    import pyomo.environ as pyo
    solver = pyo.SolverFactory(solver)
    solution = solver.solve(model, tee = tee)
    return solution

def pprint(model):
    import pyomo.environ as pyo
    for key, value in model.__dict__.items():
     value.pprint() if isinstance(value, pyo.Var) else None