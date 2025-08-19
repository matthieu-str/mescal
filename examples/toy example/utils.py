import energyscope
import pandas as pd
import ast
from energyscope.models import Model
from energyscope.energyscope import Energyscope
from energyscope.result import postprocessing
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

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
max_ccs = max_per_cat[max_per_cat['Abbrev'] == 'm_CCS']['max_unit'].values[0]
max_tthh = max_per_cat[max_per_cat['Abbrev'] == 'TTHH']['max_unit'].values[0]
max_tteq = max_per_cat[max_per_cat['Abbrev'] == 'TTEQ']['max_unit'].values[0]

# Characterization factors from IMPACT World+ (version 2.1)
cf_dict = {
    'Climate change, short term': 1.0,
    'Total human health': 7.197288788407747e-06,
    'Total ecosystem quality': 0.7025434445756051,
}

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
    'TC': 'Total cost',
    'CCST': 'Climate change, short term',
    'TTHH': 'Total human health',
    'TTEQ': 'Total ecosystem quality',
}

unit_ind_dict = {
    'Total cost': 'credits',
    'Climate change, short term': 't CO<sub>2</sub>-eq',
    'Total human health': 'DALY',
    'Total ecosystem quality': 'PDF.m<sup>2</sup>.yr',
}

unit_ind_txt_dict = {
    'Total cost': '$',
    'Climate change, short term': 't CO2-eq',
    'Total human health': 'DALY',
    'Total ecosystem quality': 'PDF.m2.yr',
}

unit_ind_mpl_dict = {
    'Total cost': 'credits',
    'Climate change, short term': 't CO$_2$-eq',
    'Total human health': 'DALY',
    'Total ecosystem quality': 'PDF.m$^2$.yr',
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
    'N/A': 'N/A',
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
        scenario: bool = False,
        lca_metrics_background: str = 'base',
        returns: str = 'results',
):

    with open(path_model + 'objective_function.mod', 'w') as f:
        f.write(f'minimize obj: {objective_function};')

    mod_files = [
        path_model + 'main.mod',
        path_model_lca + 'objectives_lca.mod',
        path_model + 'objective_function.mod',
    ]

    dat_files = [
        path_model + 'data.dat',
        path_model + 'techs.dat',
    ]

    if lca_metrics_background == 'base':
        dat_files.append(path_model_lca + 'techs_lca.dat')
    elif lca_metrics_background == 'esm_not_harmonized':
        run = objective_function.replace("Total", "").replace("LCIA_", "").replace('[2050]', '').lower()
        dat_files.append(path_model_lca + f'esm_results/techs_lca_{run.replace("[2060]", "")}_wo_h.dat')
    elif lca_metrics_background == 'esm_harmonized':
        run = objective_function.replace("Total", "").replace("LCIA_", "").replace('[2050]', '').lower()
        dat_files.append(path_model_lca + f'esm_results/techs_lca_{run.replace("[2060]", "")}.dat')
    else:
        raise ValueError("lca_metrics_background must be 'base', 'esm_not_harmonized' or 'esm_harmonized'")

    if scenario:
        dat_files.append(path_model + 'scenario.dat')

    # Initialize the model with .mod and .dat files
    model = Model(
        mod_files=mod_files,
        dat_files=dat_files,
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


def update_scenario_file(
        run: str,
        df_f_mult: pd.DataFrame,
):

    with open(path_model + 'scenario.dat', 'r') as f:
        lines = f.readlines()

    existing_capacities = df_f_mult[df_f_mult['Run'] == run][['Name', 'F_Mult']]
    for i in range(len(existing_capacities)):
        tech = existing_capacities.iloc[i]['Name']
        cap = existing_capacities.iloc[i]['F_Mult']
        lines[i] = f"let f_min['{tech}'] := {cap} ;\n"

    with open(path_model + 'scenario.dat', 'w') as f:
        f.writelines(lines)


def get_impact_scores(
        impact_category: tuple or list[tuple],
        df_impact_scores: pd.DataFrame,
        df_results: energyscope.result.Result,
        assessment_type: str = 'esm',
        specific_year: int = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] or pd.DataFrame:

    if isinstance(impact_category, tuple):
        impact_category = [impact_category]

    if type(df_impact_scores.Impact_category.iloc[0]) is tuple:
        pass
    elif type(df_impact_scores.Impact_category.iloc[0]) is str:
        df_impact_scores.Impact_category = df_impact_scores.Impact_category.apply(lambda x: ast.literal_eval(x))

    df_lifetime = df_results.parameters['lifetime'].reset_index()
    df_f_mult = df_results.variables['F_Mult'].reset_index().drop(columns=['Run'])
    df_f_mult = df_f_mult.merge(df_lifetime[['index0', 'index1', 'lifetime']], on=['index0', 'index1'], how='left')
    df_annual_prod = df_results.variables['Annual_Prod'].reset_index().drop(columns=['Run'])
    df_annual_res = df_results.variables['Annual_Res'].reset_index().drop(columns=['Run'])
    df_tech_cost = pd.merge(df_results.variables['C_maint']['C_maint'].reset_index(), df_results.variables['C_inv']['C_in'].reset_index(), on=['index0', 'index1'], how='outer')
    df_tech_cost = df_tech_cost.merge(df_results.parameters['tau'].reset_index(), on=['index0', 'index1'], how='left')
    df_tech_cost['C_inv_an'] = df_tech_cost['C_in'] * df_tech_cost['tau']
    df_res_cost = df_results.variables['C_op'].reset_index()

    df_f_mult = df_f_mult.merge(df_tech_cost, on=['index0', 'index1'], how='left')
    df_annual_res = df_annual_res.merge(df_res_cost, on=['index0', 'index1'], how='left')

    if specific_year is not None:
        df_f_mult = df_f_mult[df_f_mult['index1'] == specific_year]
        df_annual_prod = df_annual_prod[df_annual_prod['index1'] == specific_year]
        df_annual_res = df_annual_res[df_annual_res['index1'] == specific_year]

    for cat in impact_category:
        impact_scores_cat = df_impact_scores[df_impact_scores.Impact_category == cat]

        if assessment_type == 'esm':
            df_f_mult = df_f_mult.merge(impact_scores_cat[impact_scores_cat.Type == 'Construction'][['Name', 'Value', 'Year']],
                                        left_on=['index0', 'index1'], right_on=['Name', 'Year'], how='left')
            df_f_mult[cat[-1]] = df_f_mult.F_Mult * df_f_mult.Value / df_f_mult.lifetime
            df_f_mult.drop(columns=['Name', 'Value', 'Year'], inplace=True)

            df_annual_res = df_annual_res.merge(impact_scores_cat[impact_scores_cat.Type == 'Resource'][['Name', 'Value', 'Year']],
                                                left_on=['index0', 'index1'], right_on=['Name', 'Year'], how='left')
            df_annual_res[cat[-1]] = df_annual_res.Annual_Res * df_annual_res.Value
            df_annual_res.drop(columns=['Name', 'Value', 'Year'], inplace=True)

        df_annual_prod = df_annual_prod.merge(impact_scores_cat[impact_scores_cat.Type == 'Operation'][['Name', 'Value', 'Year']],
                                              left_on=['index0', 'index1'], right_on=['Name', 'Year'], how='left')
        df_annual_prod[cat[-1]] = df_annual_prod.Annual_Prod * df_annual_prod.Value
        df_annual_prod.drop(columns=['Name', 'Value', 'Year'], inplace=True)

    if assessment_type == 'esm':
        return df_f_mult, df_annual_prod, df_annual_res
    else:
        return df_annual_prod


def aggregate_phases_results(
        cat: str,
        esm_results_annual_prod: pd.DataFrame,
        esm_results_annual_res: pd.DataFrame,
        esm_results_f_mult: pd.DataFrame,
        tech_to_show_list: list,
):
    if cat == 'Total cost':
        esm_results_f_mult['Total cost'] = esm_results_f_mult['C_inv_an'] + esm_results_f_mult['C_maint']
        esm_results_annual_res['Total cost'] = esm_results_annual_res['C_op']
        esm_results_annual_prod['Total cost'] = 0

    col_to_keep = ['Name', 'Run', cat]
    if cat != 'Total cost':
        col_to_keep.append(f'{cat} (direct)')

    if 'Background database' in esm_results_f_mult.columns:
        col_to_keep.append('Background database')
        esm_results_total = pd.concat([
            esm_results_annual_prod[col_to_keep],
            esm_results_annual_res[col_to_keep],
            esm_results_f_mult[col_to_keep],
        ]).groupby(['Name', 'Run', 'Background database']).sum().reset_index()
    else:
        esm_results_total = pd.concat([
            esm_results_annual_prod[col_to_keep],
            esm_results_annual_res[col_to_keep],
            esm_results_f_mult[col_to_keep],
        ]).groupby(['Name', 'Run']).sum().reset_index()

    esm_results_total['Run'] = esm_results_total['Run'].apply(lambda x: obj_code_dict[x])
    esm_results_total.drop(esm_results_total[~esm_results_total['Name'].isin(tech_to_show_list)].index, inplace=True)
    esm_results_total['Name'] = esm_results_total['Name'].apply(lambda x: tech_name_dict[x])

    esm_results_total[cat] = esm_results_total[cat] / N_cap
    esm_results_total[cat] *= 1e6
    if cat == 'Climate change, short term':  # from kg CO2 eq to t CO2 eq
        esm_results_total[cat] = esm_results_total[cat] * 1e-3

    if cat != 'Total cost':
        esm_results_total[f'{cat} (direct)'] = esm_results_total[f'{cat} (direct)'] / N_cap
        esm_results_total[f'{cat} (direct)'] *= 1e6
        if cat == 'Climate change, short term':  # from kg CO2 eq to t CO2 eq
            esm_results_total[f'{cat} (direct)'] = esm_results_total[f'{cat} (direct)'] * 1e-3

    return esm_results_total


plt.rcParams['hatch.linewidth'] = 0.5  # Set hatch line width to 0.5
plt.rcParams['font.family'] = 'arial'

def plot_technologies_contribution(
        cat: str,
        esm_results_annual_prod: pd.DataFrame,
        esm_results_annual_res: pd.DataFrame,
        esm_results_f_mult: pd.DataFrame,
        tech_to_show_list: list,
        save_fig: bool = False,
        show_legend: bool = False,
):

    esm_results_total = aggregate_phases_results(
        cat,
        esm_results_annual_prod,
        esm_results_annual_res,
        esm_results_f_mult,
        tech_to_show_list,
    )

    esm_results_total = esm_results_total[esm_results_total[cat] != 0]

    # Pivot for stacking
    data_pivot = esm_results_total.pivot(index='Run', columns='Name', values=cat).fillna(0)

    tech_order = ['CCGT', 'CCGT with CCS', 'Natural gas', 'Photovoltaic', 'Onshore wind', 'Battery']
    data_pivot = data_pivot[tech_order]

    if cat != 'Total cost':
        direct_pivot = esm_results_total.pivot(index='Run', columns='Name', values=f'{cat} (direct)').fillna(0)
        direct_pivot = direct_pivot[tech_order]

    # Desired order
    run_order = ['CCST', 'TTHH', 'TTEQ', 'TC']

    # Reorder the index
    data_pivot = data_pivot.reindex(run_order)

    if cat != 'Total cost':
        direct_pivot = direct_pivot.reindex(run_order)

    fig, ax = plt.subplots(figsize=(5, 2.5))
    left = np.zeros(len(data_pivot))
    y = np.arange(len(data_pivot.index))

    for i, tech in enumerate(data_pivot.columns):
        values = data_pivot[tech].values
        if cat != 'Total cost':
            direct_values = direct_pivot[tech].values
            remainder = values - direct_values
            # Plot direct (hatch)
            bars_direct = ax.barh(
                y, direct_values, left=left,
                color=color_dict.get(tech, '#000000'),
                label=None,
                edgecolor='black',
                linewidth=0.5,
                hatch='\\\\',
            )
            # Plot remainder (no hatch)
            bars_remainder = ax.barh(
                y, remainder, left=left + direct_values,
                color=color_dict.get(tech, '#000000'),
                label=tech,
                edgecolor='black',
                linewidth=0.5,
            )
            left += values
        else:
            bars = ax.barh(
                y, values, left=left,
                color=color_dict.get(tech, '#000000'),
                label=tech,
                edgecolor='black',
                linewidth=0.5,
            )
            left += values

    if cat != 'Total cost':
        hatch_proxy = mpatches.Patch(
            facecolor='white', edgecolor='black', hatch='\\\\', linewidth=0.5, label='Covered in ESM'
        )
        legend = ax.legend([hatch_proxy], ['Covered in ESM'], loc='center right', frameon=True)
        legend.get_frame().set_edgecolor('white')
    elif show_legend:
        legend = ax.legend(title='Energy technology or resource', loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=6)
        legend.get_frame().set_edgecolor('white')

    ax.set_yticks(y)
    ax.set_yticklabels(data_pivot.index)
    ax.set_ylabel('Objective function')
    ax.set_xlabel(f"{full_name_ind[cat]} [{unit_ind_mpl_dict[cat]}/(cap.yr)]")

    # Increase x-axis range with margin (e.g., 5% above max)
    xmax = left.max() * 1.05
    ax.set_xlim(0, xmax)

    plt.tight_layout()
    if save_fig:
        plt.savefig(f'./figures/soo_tech_contrib_{cat.lower().replace(" ", "_").replace(",", "")}.pdf')
    plt.show()

    if cat != 'Total cost':
        df = pd.merge(
            data_pivot.sum(axis=1).rename('Life-cycle'),
            direct_pivot.sum(axis=1).rename('Direct CO2'),
            left_index=True,
            right_index=True,
        )
        df['Covered in ESM'] = df['Direct CO2'] / df['Life-cycle']

        return df


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
            'scaled_impact': f'{full_name_ind[obj_name_dict[short_name]]} [{unit_ind_dict[obj_name_dict[short_name]]}/(cap.yr)]',
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

    return contrib_analysis_ef_full


def plot_energy_system_configuration(
        type: str,
        df_res: pd.DataFrame,
        save_fig: bool = False,
        show_legend: bool = False,
) -> pd.DataFrame:
    df = df_res[~df_res['Name'].isin(['GRID', 'BATTERY'])]
    df["Run"] = df["Run"].apply(lambda x: obj_code_dict[x])
    df["Name"] = df["Name"].apply(
        lambda x: tech_name_dict[x] if x in tech_name_dict.keys() else x)

    if type == 'capacity':
        y = 'F_Mult'
        file_name = 'soo_installed_capacities'
        legend = 'Installed capacity [kW/(cap.yr)]'
        df[y] *= 1e6 / N_cap
    elif type == 'production':
        y = 'Annual_Prod'
        file_name = 'soo_annual_production'
        legend = 'Annual production [MWh/(cap.yr)]'
        df[y] *= 1e3 / N_cap
    else:
        raise ValueError("type must be 'capacity' or 'production'")

    df = df[df[y] != 0]  # remove unused technos from legend

    fig = px.bar(
        df,
        color='Name',
        y=y,
        x='Run',
        barmode='stack',
        labels={
            y: legend,
            'Name': 'Technology',
            'Run': 'Objective function',
        },
        width=350,
        height=310,
        category_orders={'Run': ['TC', 'TTEQ', 'TTHH', 'CCST']},  # desired order
    )

    fig.update_layout(legend_traceorder='reversed')
    fig.update_layout(showlegend=show_legend)
    if show_legend:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),  # left, right, top, bottom
    )
    fig.update_traces(marker_line_color='black', marker_line_width=0.5)

    fig.for_each_trace(lambda t: t.update(marker_color=color_dict.get(t.name, '#000000')))

    if save_fig:
        fig.write_image(f"./figures/{file_name}.pdf")

    fig.show()
    if type == 'capacity':
        return df[df.F_Mult != 0][['Name', 'Run', 'F_Mult']]
    elif type == 'production':
        return df[df.Annual_Prod != 0][['Name', 'Run', 'Annual_Prod']]


def harmonization_level(run):
    if 'esm_harmonized' in run:
        return 'ESM-H'
    elif 'esm_not_harmonized' in run:
        return 'ESM-NH'
    elif 'base' in run:
        return 'Base'
    else:
        raise ValueError(f'Unknown run {run}')
    

def plot_technologies_contribution_second_iteration(
        df_res: pd.DataFrame,
        type: str = 'capacity',
        save_fig: bool = False,
        cat: str = None,
):

    if type == 'capacity':
        type_var = 'F_Mult'
        show_var = 'F_Mult'
        y_label_name = 'New installed capacity [GW]'
        file_name = 'soo_installed_capacities_second_iteration'
    elif type == 'production':
        type_var = 'Annual_Prod'
        show_var = 'Annual_Prod'
        y_label_name = 'Total annual production [GWh]'
        file_name = 'soo_annual_production_second_iteration'
    elif type == 'impact':
        if cat is None:
            raise ValueError("cat must be defined for impact type")
        else:
            type_var = None
            show_var = cat
            y_label_name = f'{full_name_ind[cat]} [{unit_ind_mpl_dict[cat]}/(cap.yr)]'
            file_name = f"soo_{cat.replace(' ', '_').replace(',', '').lower()}_second_iteration"
    else:
        raise ValueError("type must be 'impact', 'capacity' or 'production'")

    run_first_it = ['TotalCost', 'TotalLCIA_m_CCS', 'TotalLCIA_TTHH', 'TotalLCIA_TTEQ']
    run_second_it = [i for i in df_res.Run.unique() if i not in run_first_it]

    if type != 'impact':
        df_res = df_res[~df_res.Name.isin(['BATTERY', 'GRID'])]

        df_res_sec_iter = df_res[df_res.Run.isin(run_second_it)]
        df_res_sec_iter['Background database'] = df_res_sec_iter.Run.apply(harmonization_level)

        df_res_sec_iter['Run'] = df_res_sec_iter['Run'].apply(lambda x: x.replace('_base', ''))
        df_res_sec_iter['Run'] = df_res_sec_iter['Run'].apply(lambda x: x.replace('_esm_harmonized', ''))
        df_res_sec_iter['Run'] = df_res_sec_iter['Run'].apply(lambda x: x.replace('_esm_not_harmonized', ''))

        df_res_sec_iter = df_res_sec_iter.merge(
            df_res[df_res.Run.isin(run_first_it)][['Run', 'Name', type_var]],
            on=['Run', 'Name'],
            how='left',
            suffixes=('', '_existing')
        )
        df_res_sec_iter[f'{type_var}_new'] = df_res_sec_iter[type_var] - df_res_sec_iter[f'{type_var}_existing']

        df_res_sec_iter['Run'] = df_res_sec_iter['Run'].apply(lambda x: obj_code_dict[x] if x in obj_code_dict else x)
        df_res_sec_iter['Name'] = df_res_sec_iter['Name'].apply(lambda x: tech_name_dict[x] if x in tech_name_dict else x)

    else:
        df_res_sec_iter = df_res.copy()
    
    names = df_res_sec_iter['Name'].unique()
    backgrounds = df_res_sec_iter['Background database'].unique()
    runs = df_res_sec_iter['Run'].unique()

    desired_order_runs = ['TC', 'TTEQ', 'TTHH', 'CCST']
    runs = [run for run in desired_order_runs if run in runs]
    desired_order_backgrounds = ['Base', 'ESM-NH', 'ESM-H']
    backgrounds = [bg for bg in desired_order_backgrounds if bg in backgrounds]

    data = np.zeros((len(runs), len(backgrounds), len(names)))
    for i, run in enumerate(runs):
        for j, bg in enumerate(backgrounds):
            for k, name in enumerate(names):
                val = df_res_sec_iter[
                    (df_res_sec_iter['Run'] == run) &
                    (df_res_sec_iter['Background database'] == bg) &
                    (df_res_sec_iter['Name'] == name)
                    ][show_var]
                data[i, j, k] = val.values[0] if not val.empty else 0

    fig, ax = plt.subplots(figsize=(4.3, 4.5))
    bar_width = 0.2
    x = np.arange(len(runs))
    ax.set_xticks(x)
    ax.set_xticklabels(runs)

    for j, bg in enumerate(backgrounds):
        bottom = np.zeros(len(runs))
        for k, name in enumerate(names):
            ax.bar(
                x + (j - 1) * bar_width,
                data[:, j, k],
                bar_width,
                bottom=bottom,
                color=color_dict.get(name, '#000000'),
                label=name if j == 0 else None,
                edgecolor='black',
                linewidth=0.5,
            )
            bottom += data[:, j, k]
        # Add background label below bars
        for i, run in enumerate(runs):
            xpos = x[i] + (j - 1) * bar_width
            ax.text(
                xpos,
                -max(data.flatten()) * 0.05,
                bg,
                ha='center',
                va='top',
                fontsize=9,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.tick_params(axis='x', length=0)
    ax.set_xticklabels(runs)
    ax.set_xlabel('Objective function', labelpad=10)
    ax.set_ylabel(y_label_name)
    y_max = ax.get_ylim()[1]
    ax.set_ylim(0, y_max * 1.05)

    ax.tick_params(axis='x', pad=55)

    plt.tight_layout()
    if save_fig:
        plt.savefig(f'./figures/{file_name}.pdf')
    plt.show()

    if cat is not None:
        df = df_res_sec_iter[['Name', 'Run', 'Background database', cat]]
        df = df.pivot_table(
            index='Run',
            columns='Background database',
            values=cat,
            aggfunc='sum',
        )
        df['Harmonization effect'] = (df['ESM-H'] - df['ESM-NH']) / df['ESM-NH']
        return df['Harmonization effect']
    else:
        return df_res_sec_iter

def plot_moo_config(
        results_pareto,
        obj1: str,
        y_axis: str = 'F_Mult',
        plot_type: str = 'line_plot',
        save_fig: bool = False
):
    normalized_limit_list = list(results_pareto.parameters['limit_lcia'].loc[obj1.replace("TotalLCIA_", "")]['limit_lcia'])
    df_y_pareto = results_pareto.variables[y_axis].reset_index()
    plt.figure(figsize=(4.5, 3.5))

    if obj1 == 'TotalLCIA_m_CCS':
        x = [i * 1e3 * max_ccs / N_cap for i in normalized_limit_list]
        plt.xlabel(f'Upper limit for {obj_name_dict[obj1].lower()} [t CO$_2$-eq/(cap.yr)]')

    elif obj1 == 'TotalLCIA_TTHH':
        x = [i * 1e6 * max_tthh / N_cap for i in normalized_limit_list]
        plt.xlabel(f'Upper limit for {obj_name_dict[obj1].replace("Total ", "").lower()} damage [DALY/(cap.yr)]')

    elif obj1 == 'TotalLCIA_TTEQ':
        x = [i * 1e6 * max_tteq / N_cap for i in normalized_limit_list]
        plt.xlabel(f'Upper limit for {obj_name_dict[obj1].replace("Total ", "").lower()} damage [PDF.m$^2$.yr/(cap.yr)]')

    else:
        raise ValueError(f"Unknown objective: {obj1}")

    y_1 = list(df_y_pareto[(df_y_pareto['index0'] == 'CCGT') & (df_y_pareto['index1'] == 2050)][y_axis])
    y_2 = list(df_y_pareto[(df_y_pareto['index0'] == 'CCGT_CC') & (df_y_pareto['index1'] == 2050)][y_axis])
    y_3 = list(df_y_pareto[(df_y_pareto['index0'] == 'PV') & (df_y_pareto['index1'] == 2050)][y_axis])
    y_4 = list(df_y_pareto[(df_y_pareto['index0'] == 'WIND_ONSHORE') & (df_y_pareto['index1'] == 2050)][y_axis])

    if y_axis == 'Annual_Prod':
        # from GWh to MWh / cap
        y_1 = [i * 1e3 / N_cap for i in y_1]
        y_2 = [i * 1e3 / N_cap for i in y_2]
        y_3 = [i * 1e3 / N_cap for i in y_3]
        y_4 = [i * 1e3 / N_cap for i in y_4]

        plt.ylabel('Annual production [MWh/(cap.yr)]')

    elif y_axis == 'F_Mult':
        # from GW to kW / cap
        y_1 = [i * 1e6 / N_cap for i in y_1]
        y_2 = [i * 1e6 / N_cap for i in y_2]
        y_3 = [i * 1e6 / N_cap for i in y_3]
        y_4 = [i * 1e6 / N_cap for i in y_4]

        plt.ylabel('Installed capacity [kW/(cap.yr)]')

    if plot_type == 'line_plot':

        plt.plot(x, y_1, label=tech_name_dict['CCGT'], c=color_dict[tech_name_dict['CCGT']])
        plt.plot(x, y_2, label=tech_name_dict['CCGT_CC'], c=color_dict[tech_name_dict['CCGT']], ls='-.')
        plt.plot(x, y_3, label=tech_name_dict['PV'], c=color_dict[tech_name_dict['PV']])
        plt.plot(x, y_4, label=tech_name_dict['WIND_ONSHORE'], c=color_dict[tech_name_dict['WIND_ONSHORE']], ls='--')

    elif plot_type == 'stack_plot':

        plt.stackplot(
            x,
            [y_1, y_2, y_3, y_4],
            labels=[tech_name_dict['CCGT'], tech_name_dict['CCGT_CC'], tech_name_dict['PV'], tech_name_dict['WIND_ONSHORE']],
            colors=[color_dict[tech_name_dict['CCGT']], color_dict[tech_name_dict['CCGT_CC']], color_dict[tech_name_dict['PV']], color_dict[tech_name_dict['WIND_ONSHORE']]],
        )

    if plot_type == 'line_plot':
        plt.legend()
    plt.tight_layout()

    if save_fig:
        if y_axis == 'F_Mult':
            plt.savefig(f'./figures/pareto_front_capacities_{obj1.replace("TotalLCIA_", "").lower()}.pdf')
        elif y_axis == 'Annual_Prod':
            plt.savefig(f'./figures/pareto_front_production_{obj1.replace("TotalLCIA_", "").lower()}.pdf')

    plt.show()

def plot_moo_indicators(
        results_pareto,
        obj1: str,
        direct_co2_emissions_dict: dict,
        save_fig: bool = False
):
    normalized_limit_list = list(results_pareto.parameters['limit_lcia'].loc[obj1.replace("TotalLCIA_", "")]['limit_lcia'])

    pareto_direct = results_pareto.variables['Annual_Prod'].reset_index().rename(columns={'index0': 'Name'})
    pareto_direct['Direct CO2 emissions'] = pareto_direct['Annual_Prod'] * pareto_direct.Name.apply(lambda x: direct_co2_emissions_dict[x])

    for imp_cat in ['TotalLCIA_TTHH', 'TotalLCIA_TTEQ']:
        # Adding direct CO2 emissions columns to the operation impacts dataframe
        pareto_direct[f'{imp_cat} (direct)'] = pareto_direct['Direct CO2 emissions'] * cf_dict[obj_name_dict[imp_cat]]
    
    fig, ax1 = plt.subplots(figsize=(4.5, 3.5))
    ax2 = ax1.twinx()
    ax1.set_zorder(1)
    ax2.set_zorder(0)
    ax1.patch.set_visible(False)

    color_cost_axis = 'red'

    ax1.tick_params(axis="y", colors=color_cost_axis)
    ax1.spines["left"].set_color(color_cost_axis)

    if obj1 == 'TotalLCIA_TTHH':
        max_x = max_tthh
        max_y = max_tteq
        y_metric = 'TTEQ'
        ax1.set_xlabel('Upper limit for human health damage [DALY/(cap.yr)]')
        ax2.set_ylabel('Ecosystem quality damage [PDF.m$^2$.yr/(cap.yr)]')

    elif obj1 == 'TotalLCIA_TTEQ':
        max_x = max_tteq
        max_y = max_tthh
        y_metric = 'TTHH'
        ax1.set_xlabel('Upper limit for ecosystem quality damage [PDF.m$^2$.yr/(cap.yr)]')
        ax2.set_ylabel('Human health damage [DALY/(cap.yr)]')

    else:
        raise ValueError(f"Unknown objective: {obj1}")

    ax1.set_ylabel('Total cost [credits/(cap.yr)]', c=color_cost_axis)
    x = [i * 1e6 * max_x / N_cap for i in normalized_limit_list]
    y_1 = [i * 1e6 / N_cap for i in list(results_pareto.variables['TotalCost'].loc[2050]['TotalCost'])]
    ax1.set_ylim(0, 1.05*max(y_1))

    tech_impact = {}
    for tech in ['CCGT', 'CCGT_CC', 'PV', 'WIND_ONSHORE', 'BATTERY']:
        tech_impact[tech] = {}
        tech_impact[tech]['Construction'] = [(1e6 * max_y / N_cap) * i for i in list(results_pareto.variables['LCIA_constr'].loc[y_metric].loc[tech]['LCIA_constr'])]
        tech_impact[tech]['Operation'] = [(1e6 * max_y / N_cap) * i for i in list(results_pareto.variables['LCIA_op'].loc[y_metric].loc[tech]['LCIA_op'])]
        tech_impact[tech]['Total'] = [i + j for i, j in zip(tech_impact[tech]['Construction'], tech_impact[tech]['Operation'])]
        tech_impact[tech]['Direct CO2'] = [(1e6 / N_cap) * i for i in list(pareto_direct[pareto_direct.Name == tech][f'TotalLCIA_{y_metric} (direct)'])]

    for res in ['NG']:
        tech_impact[res] = {}
        tech_impact[res]['Operation'] = [(1e6 * max_y / N_cap) * i for i in list(results_pareto.variables['LCIA_res'].loc[y_metric].loc[res]['LCIA_res'])]
        tech_impact[res]['Total'] = tech_impact[res]['Operation']
        tech_impact[res]['Direct CO2'] = [0] * len(tech_impact[res]['Total'])

    ax1.plot(x, y_1, label='Total cost', c=color_cost_axis, linewidth=2)

    techs = ['CCGT', 'CCGT_CC', 'NG', 'PV', 'WIND_ONSHORE', 'BATTERY']
    colors = [color_dict[tech_name_dict[tech]] for tech in techs]

    cumulated_total = np.zeros(len(x))
    for i, tech in enumerate(techs):
        direct = np.array(tech_impact[tech]['Direct CO2'])
        total = np.array(tech_impact[tech]['Total'])

        # Set linewidth to 0 if direct is all zeros
        lw = 0 if np.all(direct == 0) else 0.5

        # Plot direct (hatched)
        ax2.fill_between(
            x, cumulated_total, cumulated_total + direct,
            color=colors[i], edgecolor='black', linewidth=0, hatch='\\\\', label=None
        )
        # Plot remainder (plain)
        ax2.fill_between(
            x, cumulated_total + direct, cumulated_total + total,
            color=colors[i], edgecolor='black', linewidth=lw, label=tech_name_dict[tech]
        )
        cumulated_total += total

    ax2.set_ylim(0, 1.1 * max(cumulated_total))

    # Legend for hatching
    hatch_proxy = mpatches.Patch(
        facecolor='white', edgecolor='black', hatch='\\\\', linewidth=0.5, label='Covered in ESM'
    )
    ax2.legend([hatch_proxy], ['Covered in ESM'], loc='upper center').get_frame().set_edgecolor('white')
    
    plt.tight_layout()

    if save_fig:
        plt.savefig(f"./figures/pareto_front_indicators_{obj1.replace('TotalLCIA_', '').replace(' ', '_').lower()}.pdf")

    plt.show()

    records = []
    for tech, v1 in tech_impact.items():
        for phase, v2 in v1.items():
            for id, value in enumerate(v2):
                records.append({
                    'Run': id+1,
                    f'Upper limit on {obj_name_dict[obj1].lower()} [{unit_ind_txt_dict[obj_name_dict[obj1]]}/(cap.yr)]': x[id],
                    'Technology': tech,
                    'Phase': phase,
                    'Impact': value,
                })
    df = pd.DataFrame(records)
    df.Technology = df.Technology.apply(lambda x: tech_name_dict[x] if x in tech_name_dict else x)
    df = df[df.Impact != 0].sort_values('Run').reset_index(drop=True)
    df = df.pivot_table('Impact', ['Run', f'Upper limit on {obj_name_dict[obj1].lower()} [{unit_ind_txt_dict[obj_name_dict[obj1]]}/(cap.yr)]', 'Technology'], 'Phase')
    df.fillna(0, inplace=True)
    if y_metric == 'TTEQ':
        df['Unit'] = 'PDF.m2.yr/(cap.yr)'
    elif y_metric == 'TTHH':
        df['Unit'] = 'DALY/(cap.yr)'
    return df.reset_index()

def plot_pareto_front(
        results_pareto,
        main_variables_results: pd.DataFrame,
        obj1: str,
        obj2: str,
        colors_var: str,
        add_soo_point: bool = False,
):
    normalized_limit_list = list(results_pareto.parameters['limit_lcia'].loc[obj1.replace("TotalLCIA_", "")]['limit_lcia'])

    if obj1 == 'TotalLCIA_m_CCS':
        x = [i * 1e3 * max_ccs / N_cap for i in normalized_limit_list]
    elif obj1 == 'TotalLCIA_TTHH':
        x = [i * 1e6 * max_tthh / N_cap for i in normalized_limit_list]
    elif obj1 == 'TotalLCIA_TTEQ':
        x = [i * 1e6 * max_tteq / N_cap for i in normalized_limit_list]
    else:
        raise ValueError(f"Unknown objective: {obj1}")

    y = list(results_pareto.variables[obj2].loc[2050][obj2])
    colors = list(results_pareto.variables[colors_var].loc[2050][colors_var])

    # adding data point from colors_var SOO
    if add_soo_point:
        x = x + [main_variables_results[main_variables_results['Objective'] == colors_var][obj1].values[0]]
        y = y + [main_variables_results[main_variables_results['Objective'] == colors_var][obj2].values[0]]
        colors = colors + [main_variables_results[main_variables_results['Objective'] == colors_var][colors_var].values[0]]

    # Create the scatter plot
    scatter = plt.scatter(x, y, c=colors, cmap='jet')

    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label(colors_var)

    # Add labels and title
    plt.xlabel(obj1)
    plt.ylabel(obj2)

    # Show the plot
    plt.show()