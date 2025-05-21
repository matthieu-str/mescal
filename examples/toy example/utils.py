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

full_name_ind = {
    'Total cost': 'Total cost',
    'Climate change, short term': 'Climate change, short term',
    'Total human health': 'Human health damage',
    'Total ecosystem quality': 'Ecosystem quality damage',
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

impact_category_hh_colors = {
    'Climate change, human health, long term': '#0072B2',  # Dark blue
    'Climate change, human health, short term': '#56B4E9',  # Light blue
    'Human toxicity cancer, long term': '#D55E00',  # Burnt orange
    'Human toxicity cancer, short term': '#E69F00',  # Orange
    'Human toxicity non-cancer, long term': '#CC79A7',  # Magenta
    'Human toxicity non-cancer, short term': '#F7CAE0',  # Light pink
    'Ionizing radiations, human health': '#999933',  # Olive green
    'Ozone layer depletion': '#00CED1',  # Dark turquoise
    'Particulate matter formation': '#7F7F7F',  # Medium grey
    'Photochemical ozone formation, human health': '#9E5BBA',  # Soft violet
    'Water availability, human health': '#009E73',  # Teal green
}

impact_category_eq_colors = {
    'Climate change, ecosystem quality, long term': '#0072B2',  # Dark blue
    'Climate change, ecosystem quality, short term': '#56B4E9',  # Light blue
    'Fisheries impact': '#1B9E77',  # Sea green
    'Freshwater acidification': '#8DA0CB',  # Periwinkle
    'Freshwater ecotoxicity, long term': '#984EA3',  # Dark purple
    'Freshwater ecotoxicity, short term': '#DDA0DD',  # Plum
    'Freshwater eutrophication': '#A6CEE3',  # Light cyan
    'Ionizing radiations, ecosystem quality': '#999933',  # Olive green
    'Land occupation, biodiversity': '#A65628',  # Rust brown
    'Land transformation, biodiversity': '#E6AB02',  # Mustard yellow
    'Marine acidification, long term': '#377EB8',  # Blue
    'Marine acidification, short term': '#ADD8E6',  # Light blue
    'Marine ecotoxicity, long term': '#984EA3',  # Dark purple
    'Marine ecotoxicity, short term': '#DDA0DD',  # Plum
    'Marine eutrophication': '#1B9E77',  # Sea green
    'Photochemical ozone formation, ecosystem quality': '#9E5BBA',  # Soft violet
    'Terrestrial acidification': '#FDB462',  # Light orange
    'Terrestrial ecotoxicity, long term': '#BC80BD',  # Lavender
    'Terrestrial ecotoxicity, short term': '#F7CAE0',  # Light pink
    'Thermally polluted water': '#E377C2',  # Pink
    'Water availability, freshwater ecosystem': '#009E73',  # Teal green
    'Water availability, terrestrial ecosystem': '#66C2A5',  # Light teal
}

elementary_flow_colors = {
    # Greenhouse gases
    'Carbon dioxide, fossil': '#0072B2',  # Dark blue
    'Methane, fossil': '#56B4E9',         # Light blue
    'Tetrafluoromethane': '#4682B4',      # Steel blue

    # Air pollutants
    'Sulfur dioxide': '#999933',          # Olive green
    'Nitrogen oxides': '#D55E00',         # Burnt orange
    'Particulate Matter, < 2.5 um': '#7F7F7F',  # Medium grey

    # Metals and ions
    'Aluminium III': '#A9A9A9',           # Dark grey
    'Chromium III': '#696969',            # Dim grey
    'Chromium VI': '#8B8B8B',             # Slightly lighter grey
    'Copper ion': '#B87333',              # Copper
    'Iron ion': '#8B0000',                # Dark red
    'Lead II': '#6E6E6E',                 # Graphite grey
    'Mercury II': '#5E5E5E',              # Charcoal grey
    'Cadmium II': '#B0C4DE',              # Light steel blue
    'Silver I': '#C0C0C0',                # Silver

    # Land/water
    'Land occupation': '#A65628',         # Rust brown
    'Land transformation': '#E6AB02',     # Mustard yellow
    'Water': '#009E73',                   # Teal green

    # Catch-all
    'Other': '#D9B382',                   # Light tan
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
    df_f_mult = df_results.variables['F_Mult'].reset_index().drop(columns=['Run'])
    df_f_mult = df_f_mult.merge(df_lifetime[['index', 'lifetime']], on='index', how='left')
    df_annual_prod = df_results.variables['Annual_Prod'].reset_index().drop(columns=['Run'])
    df_annual_res = df_results.variables['Annual_Res'].reset_index().drop(columns=['Run'])
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
        labels={'Run': 'Objective function', 'Name': 'Energy technology or resource', cat: f'{full_name_ind[cat]} [{unit_ind_dict[cat]}/cap]'},
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

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),  # left, right, top, bottom
    )

    if save_fig:  # save as pdf
        fig.write_image(f"./figures/soo_tech_contrib_{cat.lower().replace(' ', '_').replace(',','')}.pdf")

    fig.show()


def plot_ef_contributions(
        df_contrib_analysis_ef: pd.DataFrame,
        esm_results_f_mult: pd.DataFrame,
        esm_results_annual_prod: pd.DataFrame,
        esm_results_annual_res: pd.DataFrame,
        main_variables_results: pd.DataFrame,
        aop: str,
        cutoff: float = 0.01,
        save_fig: bool = False,
):
    if aop == 'Total ecosystem quality':
        impact_category = ('IMPACT World+ Damage 2.1 for ecoinvent v3.10', 'Ecosystem quality', 'Total ecosystem quality')
        short_name = 'TotalLCIA_TTEQ'
    elif aop == 'Total human health':
        impact_category = ('IMPACT World+ Damage 2.1 for ecoinvent v3.10', 'Human health', 'Total human health')
        short_name = 'TotalLCIA_TTHH'
    else:
        raise ValueError("aop must be 'Total ecosystem quality' or 'Total human health'")

    contrib_analysis_ef = df_contrib_analysis_ef.copy(deep=True)

    if type(contrib_analysis_ef.impact_category.iloc[0]) is str:
        contrib_analysis_ef['impact_category'] = contrib_analysis_ef['impact_category'].apply(lambda x: ast.literal_eval(x))
    
    contrib_analysis_ef['score'] = contrib_analysis_ef['score'] * 1e6 / N_cap
    contrib_analysis_ef['amount'] = contrib_analysis_ef['amount'] * 1e6 / N_cap
    
    contrib_analysis_ef = contrib_analysis_ef[
        contrib_analysis_ef['impact_category'] == impact_category
    ].groupby(['act_name', 'act_type', 'ef_name']).sum(['score', 'amount']).reset_index()
    
    contrib_analysis_ef_constr = pd.merge(
        contrib_analysis_ef[contrib_analysis_ef['act_type'] == 'Construction'][['act_name', 'ef_name', 'score', 'amount']],
        esm_results_f_mult[['Name', 'Run', 'F_Mult', 'lifetime']],
        how='right',
        left_on=['act_name'],
        right_on=['Name'],
    ).drop(columns=['act_name'])
    
    contrib_analysis_ef_op = pd.merge(
        contrib_analysis_ef[contrib_analysis_ef['act_type'] == 'Operation'][['act_name', 'ef_name', 'score', 'amount']],
        esm_results_annual_prod[['Name', 'Run', 'Annual_Prod']],
        how='right',
        left_on=['act_name'],
        right_on=['Name'],
    ).drop(columns=['act_name'])
    
    contrib_analysis_ef_res = pd.merge(
        contrib_analysis_ef[contrib_analysis_ef['act_type'] == 'Resource'][['act_name', 'ef_name', 'score', 'amount']],
        esm_results_annual_res[['Name', 'Run', 'Annual_Res']],
        how='right',
        left_on=['act_name'],
        right_on=['Name'],
    ).drop(columns=['act_name'])
    
    contrib_analysis_ef_constr['scaled_impact'] = (
            contrib_analysis_ef_constr['score'] * contrib_analysis_ef_constr['F_Mult'] / contrib_analysis_ef_constr['lifetime']
    )
    contrib_analysis_ef_constr = contrib_analysis_ef_constr[contrib_analysis_ef_constr['scaled_impact'] != 0]

    contrib_analysis_ef_op['scaled_impact'] = contrib_analysis_ef_op['score'] * contrib_analysis_ef_op['Annual_Prod']
    contrib_analysis_ef_op = contrib_analysis_ef_op[contrib_analysis_ef_op['scaled_impact'] != 0]

    contrib_analysis_ef_res['scaled_impact'] = contrib_analysis_ef_res['score'] * contrib_analysis_ef_res['Annual_Res']
    contrib_analysis_ef_res = contrib_analysis_ef_res[contrib_analysis_ef_res['scaled_impact'] != 0]
    
    contrib_analysis_ef_full = pd.concat([
        contrib_analysis_ef_constr[['Run', 'ef_name', 'scaled_impact']],
        contrib_analysis_ef_op[['Run', 'ef_name', 'scaled_impact']],
        contrib_analysis_ef_res[['Run', 'ef_name', 'scaled_impact']]
    ]).groupby(['Run', 'ef_name']).sum().reset_index()
    
    # Concatenate the contributions from land occupation, land transformation and water
    contrib_analysis_ef_full['ef_name'] = contrib_analysis_ef_full['ef_name'].apply(
        lambda x: 'Land ' + x.split(', ')[0].lower() if x.startswith('Transformation') or x.startswith('Occupation') else x
    )
    contrib_analysis_ef_full['ef_name'] = contrib_analysis_ef_full['ef_name'].apply(
        lambda x: x.split(', ')[0] if x.startswith('Water') else x
    )

    contrib_analysis_ef_full = contrib_analysis_ef_full.groupby(['Run', 'ef_name']).sum().reset_index()
    
    df_total = main_variables_results[['Objective', short_name]]
    df_total[short_name] = df_total[short_name] * max_ind_dict[short_name] * 1e6 / N_cap
    df_total.rename(columns={'Objective': 'Run', short_name: 'total_impact'}, inplace=True)
    
    contrib_analysis_ef_full = pd.merge(contrib_analysis_ef_full, df_total, on='Run', how='left')

    # cut-off criteria
    contrib_analysis_ef_full['scaled_impact_perc'] = contrib_analysis_ef_full['scaled_impact'] / contrib_analysis_ef_full['total_impact']
    contrib_analysis_ef_full = contrib_analysis_ef_full[abs(contrib_analysis_ef_full['scaled_impact_perc']) > cutoff]
    
    # add an 'other' row for the difference between sum of contributions and total
    contrib_analysis_ef_rest = pd.merge(
        contrib_analysis_ef_full.groupby(['Run']).sum().reset_index()[['Run', 'scaled_impact']],
        df_total,
        on='Run',
        how='left',
    )

    contrib_analysis_ef_rest['scaled_impact'] = contrib_analysis_ef_rest['total_impact'] - contrib_analysis_ef_rest['scaled_impact']
    contrib_analysis_ef_rest['ef_name'] = 'Other'
    
    contrib_analysis_ef_full = pd.concat([contrib_analysis_ef_full, contrib_analysis_ef_rest])
    
    contrib_analysis_ef_full['Run'] = contrib_analysis_ef_full['Run'].apply(lambda x: obj_code_dict[x])
    
    fig = px.bar(
        contrib_analysis_ef_full.sort_values('scaled_impact', ascending=False),
        x='Run',
        y='scaled_impact',
        color='ef_name',
        barmode='stack',
        labels={
            'ef_name': 'Elementary flow',
            'scaled_impact': f'{full_name_ind[obj_name_dict[short_name]]} [{unit_ind_dict[obj_name_dict[short_name]]}/cap]',
            'Run': 'Objective function',
        },
        width=550,
        height=350,
    )

    fig.update_layout(legend_traceorder='reversed')

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),  # left, right, top, bottom
    )

    fig.for_each_trace(lambda t: t.update(marker_color=elementary_flow_colors.get(t.name, '#000000')))

    if save_fig:
        fig.write_image(f"./figures/soo_ef_contrib_{aop.replace('Total ', '').replace(' ', '_')}.pdf")

    fig.show()