from energyscope.models import Model
from energyscope.energyscope import Energyscope
from energyscope.result import postprocessing

path_model = './data/esm/' # Path to the energy system model
path_model_lca = './data/esm/lca/'

def run_esm(
        objective_function: str = 'TotalCost',
        returns: str = 'results'
):

    with open(path_model + 'objective_function.mod', 'w') as f:
        f.write(f'minimize obj: {objective_function};')

    # Initialize the model with .mod and .dat files
    model = Model(
        mod_files=[
            path_model + 'main.mod',
            path_model_lca + 'objectives_lca.mod',
            path_model_lca + 'objectives_lca_direct.mod',
            path_model + 'objective_function.mod',
        ],
        dat_files=[
            path_model + 'data.dat',
            path_model + 'techs.dat',
            path_model_lca + 'techs_lca.dat',
            path_model_lca + 'techs_lca_direct.dat',
        ],
    )

    # Define the solver options
    solver_options = {
        'solver': 'gurobi',
        'solver_msg': 0,
    }

    # Initialize the model
    es = Energyscope(model=model, solver_options=solver_options)

    if returns == 'model':
        # Return the model object
        return es

    # Solve the model and get results
    results = postprocessing(es.calc())

    if returns == 'results':
        return results