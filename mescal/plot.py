from plotly.subplots import make_subplots
import plotly.graph_objects as go
import bw2data as bd
import pandas as pd
from pathlib import Path


def plot_indicators_of_technologies_for_one_impact_category(
        self,
        R: pd.DataFrame,
        technologies_list: list[str],
        impact_category: tuple,
        metadata: dict = None,
        filename: str = None,
        saving_format: str = None,
        saving_path: str = None,
        show_plot: bool = True,
):
    """
    Plot operation and infrastructure LCA indicators for a set of technologies for a given impact category.

    :param R: dataframe of LCA indicators
    :param technologies_list: list of technologies to plot. They should have the same operation/infrastructure units
        to have a meaningful comparison
    :param impact_category: impact category to plot (brightway format)
    :param metadata: dictionary with metadata to include in the plot. It can include 'technologies_type',
        'operation_unit', 'construction_unit'.
    :param filename: name of the file to save the plot. If None, the plot is named with the impact category.
    :param saving_format: format to save the plot, can be 'png', 'jpeg', 'pdf', 'html', etc.
    :param saving_path: path to save the plot under the form 'path/to/folder/'. If None, the plot is saved in the
        current directory.
    :param show_plot: if True, the plot is shown in the notebook.
    :return: None (plot is shown and/or saved)
    """

    if saving_path is None:
        saving_path = ''
    if metadata is None:
        metadata = {}

    for tech in technologies_list:
        if tech not in R['Name'].unique():
            raise ValueError(f'Technology {tech} not found in the LCA indicators dataframe')

    unit = bd.Method(impact_category).metadata['unit']
    impact_category_name = impact_category[-1]

    if filename is None:
        filename = impact_category_name

    if 'technologies_type' in metadata:
        graph_title = f"LCA Indicators of {metadata['technologies_type']} technologies for {impact_category[-1]}"
    else:
        graph_title = f"LCA Indicators for {impact_category[-1]}"

    if 'operation_unit' in metadata and 'construction_unit' in metadata:
        operation_unit = f'/{metadata["operation_unit"]}'
        construction_unit = f'/{metadata["construction_unit"]}'
    else:
        operation_unit = ''
        construction_unit = ''

    df = R[(R['Name'].isin(technologies_list)) & (R['Impact_category'] == str(impact_category))]
    df_op = df[df['Type'] == 'Operation']
    df_constr = df[df['Type'] == 'Construction']

    if self.lifetime is not None:
        df_constr = df_constr.merge(self.lifetime, on='Name')
        df_constr['Value'] = df_constr['Value'] / df_constr['ESM']
        construction_unit += '.year'

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Operation {operation_unit}", f"Construction {construction_unit}"),
        )

    # Add bar chart for operation
    fig.add_trace(
        go.Bar(
            x=df_op['Name'],
            y=df_op['Value'],
            name='Operation'
        ),
        row=1, col=1
    )

    # Add bar chart for construction
    fig.add_trace(
        go.Bar(
            x=df_constr['Name'],
            y=df_constr['Value'],
            name='Construction'
        ),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title_text=graph_title,
        template='plotly_white',
        showlegend=False
    )

    # Update axis labels
    fig.update_yaxes(title_text=f'{impact_category_name} [{unit}]', row=1, col=1)
    fig.update_yaxes(title_text=f'{impact_category_name} [{unit}]', row=1, col=2)

    # Show plot
    if show_plot:
        fig.show()

    if saving_format is None:
        pass
    elif saving_format == 'html':
        Path(saving_path).mkdir(parents=True, exist_ok=True)  # Create the folder if it does not exist
        fig.write_html(f'{saving_path}{filename}.{saving_format}')
    else:
        Path(saving_path).mkdir(parents=True, exist_ok=True)  # Create the folder if it does not exist
        fig.write_image(f'{saving_path}{filename}.{saving_format}')


def plot_indicators_of_resources_for_one_impact_category(
        self,
        R: pd.DataFrame,
        resources_list: list[str],
        impact_category: tuple,
        metadata: dict = None,
        filename: str = None,
        saving_format: str = None,
        saving_path: str = None,
        show_plot: bool = True,
):
    """
    Plot operation LCA indicators for a set of resources for a given impact category.

    :param R: dataframe of LCA indicators
    :param resources_list: list of technologies to plot. They should have the same operation/infrastructure units
        to have a meaningful comparison
    :param impact_category: impact category to plot (brightway format)
    :param metadata: dictionary with metadata to include in the plot. It can include 'resources_type', 'unit'.
    :param filename: name of the file to save the plot. If None, the plot is named with the impact category.
    :param saving_format: format to save the plot, can be 'png', 'jpeg', 'pdf', 'html', etc.
    :param saving_path: path to save the plot under the form 'path/to/folder/'. If None, the plot is saved in the
        current directory.
    :param show_plot: if True, the plot is shown in the notebook.
    :return: None (plot is shown and/or saved)
    """

    if saving_path is None:
        saving_path = ''
    if metadata is None:
        metadata = {}

    for res in resources_list:
        if res not in R['Name'].unique():
            raise ValueError(f'Resource {res} not found in the LCA indicators dataframe')

    unit = bd.Method(impact_category).metadata['unit']
    impact_category_name = impact_category[-1]

    if filename is None:
        filename = impact_category_name

    if 'resources_type' in metadata:
        graph_title = f"LCA Indicators of {metadata['resources_type']} resources for {impact_category[-1]})"
    else:
        graph_title = f"LCA Indicators for {impact_category[-1]}"

    if 'unit' in metadata:
        graph_title += f' (/{metadata["unit"]})'

    df = R[(R['Name'].isin(resources_list)) & (R['Impact_category'] == str(impact_category))]

    # Add bar chart
    fig = go.Figure(
        go.Bar(
            x=df['Name'],
            y=df['Value'],
        ),
    )

    # Update layout
    fig.update_layout(
        title_text=graph_title,
        template='plotly_white',
        showlegend=False
    )

    # Update axis labels
    fig.update_yaxes(title_text=f'{impact_category_name} [{unit}]')

    # Show plot
    if show_plot:
        fig.show()

    if saving_format is None:
        pass
    elif saving_format == 'html':
        Path(saving_path).mkdir(parents=True, exist_ok=True)  # Create the folder if it does not exist
        fig.write_html(f'{saving_path}{filename}.{saving_format}')
    else:
        Path(saving_path).mkdir(parents=True, exist_ok=True)  # Create the folder if it does not exist
        fig.write_image(f'{saving_path}{filename}.{saving_format}')


def plot_indicators_of_technologies_for_several_impact_category():
    pass
    # TODO
    # plot different technologies for different impact categories (with relative values [%] w.r.t the max impact)
