import energyscope
import pandas as pd
import ast
from energyscope.models import Model
from energyscope.energyscope import Energyscope
from energyscope.result import postprocessing
import plotly.express as px
import plotly.io as pio

pio.templates["custom"] = pio.templates["plotly_white"]
pio.templates["custom"].layout.font.family = "Arial"
pio.templates["custom"].layout.font.color = "black"
pio.templates["custom"].layout.xaxis.color = "black"
pio.templates["custom"].layout.xaxis.showline = True
pio.templates["custom"].layout.xaxis.linecolor = "black"
pio.templates["custom"].layout.xaxis.ticks = "outside"
pio.templates["custom"].layout.xaxis.tickcolor = "black"
pio.templates["custom"].layout.yaxis.color = "black"
pio.templates["custom"].layout.yaxis.showline = True
pio.templates["custom"].layout.yaxis.linecolor = "black"
pio.templates["custom"].layout.yaxis.ticks = "outside"
pio.templates["custom"].layout.yaxis.tickcolor = "black"
pio.templates["custom"].layout.xaxis.mirror = True
pio.templates["custom"].layout.yaxis.mirror = True
pio.templates.default = "custom"

path_model = './data/esm/' # Path to the energy system model
path_model_lca = './data/esm/lca/'

max_per_cat = pd.read_csv(path_model_lca + 'techs_lca_max.csv')
max_ccs = max_per_cat[max_per_cat['Abbrev'] == 'm_CCS']['max_AoP'].values[0]
max_tthh = max_per_cat[max_per_cat['Abbrev'] == 'TTHH']['max_AoP'].values[0]
max_tteq = max_per_cat[max_per_cat['Abbrev'] == 'TTEQ']['max_AoP'].values[0]

max_ind_dict = {
    'TotalCost': 1,
    'TotalLCIA_m_CCS': max_ccs,
    'TotalLCIA_TTHH': max_tthh,
    'TotalLCIA_TTEQ': max_tteq,
}

N_cap = 2e5  # number of people on Tatooine

obj_name_dict = {
    'TotalCost': 'Total cost',
    'TotalLCIA_m_CCS': 'Climate change, short-term',
    'TotalLCIA_TTHH': 'Total human health',
    'TotalLCIA_TTEQ': 'Total ecosystem quality',
}

obj_code_dict = {
    'TotalCost': 'TC',
    'TotalLCIA_m_CCS': 'CCST',
    'TotalLCIA_TTHH': 'TTHH',
    'TotalLCIA_TTEQ': 'TTEQ',
}

unit_ind_dict = {
    'Total cost': 'credits',
    'Climate change, short term': 't CO<sub>2</sub>-eq',
    'Total human health': 'DALY',
    'Total ecosystem quality': 'PDF.m<sup>2</sup>.yr',
}

tech_name_dict = {
    'CCGT': 'CCGT',
    'CCGT_CC': 'CCGT with CCS',
    'COAL_IGCC': 'IGCC',
    'COAL_IGCC_CC': 'IGCC with CCS',
    'NUCLEAR': 'Nuclear',
    'PV': 'Photovoltaic',
    'WIND_ONSHORE': 'Onshore wind',
    'COAL': 'Coal',
    'NG': 'Natural gas',
    'URANIUM': 'Uranium',
    'BATTERY': 'Battery',
    'GRID': 'Grid',
}

color_dict = {
    'CCGT': '#808080',  # Medium Grey
    'CCGT with CCS': '#dcdcdc',  # Very Light Grey (Gainsboro)
    'Coal': '#8c564b',                      # Brownish Red
    'IGCC': '#d62728',  # Vivid Red
    'IGCC with CCS': '#ff9896',  # Light Coral
    'Natural gas': '#505050',               # Charcoal Grey
    'Nuclear': '#2ca02c',                   # Green
    'Uranium': '#98df8a',                   # Light Green
    'Photovoltaic': '#ff7f0e',              # Orange
    'Onshore wind': '#1f77b4',              # Blue
    'Battery': '#9467bd',                   # Dark Purple
    'Grid': '#17becf'                       # Steel Blue
}

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


def get_impact_scores(
        impact_category: tuple or list[tuple],
        df_impact_scores: pd.DataFrame,
        df_results: energyscope.result.Result,
        assessment_type: str = 'esm',
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] or pd.DataFrame:

    if isinstance(impact_category, tuple):
        impact_category = [impact_category]

    if type(df_impact_scores.Impact_category.iloc[0]) is tuple:
        pass
    elif type(df_impact_scores.Impact_category.iloc[0]) is str:
        df_impact_scores.Impact_category = df_impact_scores.Impact_category.apply(lambda x: ast.literal_eval(x))

    df_lifetime = df_results.parameters['lifetime'].reset_index()
    df_f_mult = df_results.variables['F_Mult'].reset_index()
    df_f_mult = df_f_mult.merge(df_lifetime[['index', 'lifetime']], on='index', how='left')
    df_annual_prod = df_results.variables['Annual_Prod'].reset_index()
    df_annual_res = df_results.variables['Annual_Res'].reset_index()
    df_tech_cost = df_results.postprocessing['df_annual'].reset_index()[['level_0', 'C_maint', 'C_inv_an']]
    df_res_cost = df_results.variables['C_op'].reset_index()

    df_f_mult = df_f_mult.merge(df_tech_cost, left_on='index', right_on='level_0', how='left').drop(columns=['level_0'])
    df_annual_res = df_annual_res.merge(df_res_cost, on='index', how='left')

    df_sectors = df_results.postprocessing['df_annual'].reset_index()[['level_0', 'Category']]
    df_f_mult = df_f_mult.merge(df_sectors, left_on='index', right_on='level_0', how='left')
    df_annual_prod = df_annual_prod.merge(df_sectors, left_on='index', right_on='level_0', how='left')

    df_f_mult.drop(columns=['level_0'], inplace=True)
    df_annual_prod.drop(columns=['level_0'], inplace=True)

    for cat in impact_category:
        impact_scores_cat = df_impact_scores[df_impact_scores.Impact_category == cat]

        if assessment_type == 'esm':
            df_f_mult = df_f_mult.merge(impact_scores_cat[impact_scores_cat.Type == 'Construction'][['Name', 'Value']],
                                        left_on='index', right_on='Name', how='left')
            df_f_mult[cat[-1]] = df_f_mult.F_Mult * df_f_mult.Value / df_f_mult.lifetime
            df_f_mult.drop(columns=['Name', 'Value'], inplace=True)

            df_annual_res = df_annual_res.merge(impact_scores_cat[impact_scores_cat.Type == 'Resource'][['Name', 'Value']],
                                                left_on='index', right_on='Name', how='left')
            df_annual_res[cat[-1]] = df_annual_res.Annual_Res * df_annual_res.Value
            df_annual_res.drop(columns=['Name', 'Value'], inplace=True)

        df_annual_prod = df_annual_prod.merge(impact_scores_cat[impact_scores_cat.Type == 'Operation'][['Name', 'Value']],
                                              left_on='index', right_on='Name', how='left')
        df_annual_prod[cat[-1]] = df_annual_prod.Annual_Prod * df_annual_prod.Value
        df_annual_prod.drop(columns=['Name', 'Value'], inplace=True)

    if assessment_type == 'esm':
        return df_f_mult, df_annual_prod, df_annual_res
    else:
        return df_annual_prod


def plot_technologies_contribution(
        cat: str,
        esm_results_annual_prod: pd.DataFrame,
        esm_results_annual_res: pd.DataFrame,
        esm_results_f_mult: pd.DataFrame,
        tech_to_show_list: list,
        save_fig: bool = False,
        show_legend: bool = False,
):

    if cat == 'Total cost':
        esm_results_f_mult['Total cost'] = esm_results_f_mult['C_inv_an'] + esm_results_f_mult['C_maint']
        esm_results_annual_res['Total cost'] = esm_results_annual_res['C_op']
        esm_results_annual_prod['Total cost'] = 0

    esm_results_total = pd.concat([
        esm_results_annual_prod[['Name', 'Run', cat]],
        esm_results_annual_res[['Name', 'Run', cat]],
        esm_results_f_mult[['Name', 'Run', cat]],
    ]).groupby(['Name', 'Run']).sum().reset_index()

    esm_results_total['Run'] = esm_results_total['Run'].apply(lambda x: obj_code_dict[x])
    esm_results_total.drop(esm_results_total[~esm_results_total['Name'].isin(tech_to_show_list)].index, inplace=True)
    esm_results_total['Name'] = esm_results_total['Name'].apply(lambda x: tech_name_dict[x])

    esm_results_total[cat] = esm_results_total[cat] / N_cap
    esm_results_total[cat] *= 1e6
    if cat == 'Climate change, short term':  # from kg CO2 eq to t CO2 eq
        esm_results_total[cat] = esm_results_total[cat] * 1e-3

    fig = px.bar(
        esm_results_total,
        x='Run',
        y=cat,
        color='Name',
        barmode='stack',
        labels={'Run': 'Objective function', 'Name': 'Energy technology or resource', cat: f'{cat} [{unit_ind_dict[cat]}/cap]'},
        height=370,
        width=390,
    )

    fig.for_each_trace(lambda t: t.update(marker_color=color_dict.get(t.name, '#000000')))
    # fig.update_layout(template='plotly_white')

    if not show_legend:
        fig.update_layout(showlegend=False)
    else:
        fig.update_layout(
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="bottom",
                y=-0.2,  # Adjusts the vertical position
                xanchor="center",
                x=0.5  # Centers the legend horizontally
            )
        )

    if save_fig:  # save as pdf
        fig.write_image(f"./figures/soo_tech_contrib_{cat.lower().replace(' ', '_').replace(',','')}.pdf")

    fig.show()