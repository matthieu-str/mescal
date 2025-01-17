from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import bw2data as bd
import pandas as pd
from pathlib import Path
from ast import literal_eval


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
        contributions_total_score: bool = False,
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
    :param contributions_total_score: if True, the contributions of all categories to the total score are shown.
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
        operation_unit = f'{unit}/{metadata["operation_unit"]}'
        construction_unit = f'{unit}/{metadata["construction_unit"]}'
    else:
        operation_unit = unit
        construction_unit = unit

    if contributions_total_score:
        impact_category_set = str(impact_category).split(impact_category_name)[0]
        df = R[
            (R['Name'].isin(technologies_list))
            & (R['Impact_category'].str.startswith(impact_category_set))  # Disaggregate total endpoint indicator
            & (R['Impact_category'] != str(impact_category))  # Exclude total endpoint indicator
            ]
    else:
        df = R[(R['Name'].isin(technologies_list)) & (R['Impact_category'] == str(impact_category))]
    df_op = df[df['Type'] == 'Operation']
    df_constr = df[df['Type'] == 'Construction']

    if self.lifetime is not None:
        df_constr = df_constr.merge(self.lifetime, on='Name')
        df_constr['Value'] = df_constr['Value'] / df_constr['ESM']
        construction_unit += '/year'

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Operation", "Construction"),
        )

    if contributions_total_score:

        color_scale = px.colors.qualitative.Plotly
        data_op = []
        data_constr = []

        for i, disaggregated_impact_category in enumerate(df_op['Impact_category'].unique()):
            df_op_disaggregated = df_op[df_op['Impact_category'] == str(disaggregated_impact_category)]
            df_constr_disaggregated = df_constr[df_constr['Impact_category'] == str(disaggregated_impact_category)]

            data_op.append(go.Bar(
                name=literal_eval(disaggregated_impact_category)[-1],
                x=df_op_disaggregated['Name'],
                y=df_op_disaggregated['Value'],
                marker=dict(color=color_scale[i % len(color_scale)]),
                hovertemplate=
                '<br><b>Technology</b>: %{x}</br>' +
                '<b>Value</b>: %{y:.2e}' + f' {unit}</br>',
            ))

            data_constr.append(go.Bar(
                name=literal_eval(disaggregated_impact_category)[-1],
                x=df_constr_disaggregated['Name'],
                y=df_constr_disaggregated['Value'],
                marker=dict(color=color_scale[i % len(color_scale)]),
                hovertemplate=
                '<br><b>Technology</b>: %{x}</br>' +
                '<b>Value</b>: %{y:.2e}' + f' {unit}</br>',
            ))

        # Add bar chart for operation
        for trace in data_op:
            fig.add_trace(trace, row=1, col=1)

        # Add bar chart for construction
        for trace in data_constr:
            trace.showlegend = False  # Hide legend for the second subplot
            fig.add_trace(trace, row=1, col=2)

        fig.update_layout(barmode='stack', showlegend=True)

    else:
        # Add bar chart for operation
        fig.add_trace(
            go.Bar(
                x=df_op['Name'],
                y=df_op['Value'],
                name='Operation',
                hovertemplate=
                '<br><b>Technology</b>: %{x}</br>' +
                '<b>Value</b>: %{y:.2e}' + f' {unit}</br>',
            ),
            row=1, col=1
        )

        # Add bar chart for construction
        fig.add_trace(
            go.Bar(
                x=df_constr['Name'],
                y=df_constr['Value'],
                name='Construction',
                hovertemplate=
                '<br><b>Technology</b>: %{x}</br>' +
                '<b>Value</b>: %{y:.2e}' + f' {unit}</br>',
            ),
            row=1, col=2
        )

        fig.update_layout(showlegend=False)

    # Update layout
    fig.update_layout(
        title_text=graph_title,
        template='plotly_white',
    )

    # Update axis labels
    fig.update_yaxes(title_text=f'{impact_category_name} [{operation_unit}]', row=1, col=1)
    fig.update_yaxes(title_text=f'{impact_category_name} [{construction_unit}]', row=1, col=2)

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


@staticmethod
def plot_indicators_of_resources_for_one_impact_category(
        R: pd.DataFrame,
        resources_list: list[str],
        impact_category: tuple,
        metadata: dict = None,
        filename: str = None,
        saving_format: str = None,
        saving_path: str = None,
        show_plot: bool = True,
        contributions_total_score: bool = False,
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
    :param contributions_total_score: if True, the contributions of all categories to the total score are shown.
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
        unit += f'/{metadata["unit"]}'

    if contributions_total_score:

        impact_category_set = str(impact_category).split(impact_category_name)[0]

        df = R[
            (R['Name'].isin(resources_list))
            & (R['Impact_category'].str.startswith(impact_category_set))  # Disaggregate total endpoint indicator
            & (R['Impact_category'] != str(impact_category))  # Exclude total endpoint indicator
            ]

        data = []

        for disaggregated_impact_category in df['Impact_category'].unique():

            df_disaggregated = df[df['Impact_category'] == str(disaggregated_impact_category)]

            data.append(go.Bar(
                name=literal_eval(disaggregated_impact_category)[-1],
                x=df_disaggregated['Name'],
                y=df_disaggregated['Value'],
                hovertemplate=
                '<br><b>Resource</b>: %{x}</br>' +
                '<b>Value</b>: %{y:.2e}' + f' {unit}</br>',
            ))

        fig = go.Figure(data=data)
        fig.update_layout(barmode='stack', showlegend=True)

    else:
        df = R[(R['Name'].isin(resources_list)) & (R['Impact_category'] == str(impact_category))]

        # Add bar chart
        fig = go.Figure(
            go.Bar(
                x=df['Name'],
                y=df['Value'],
                name='Resource',
                hovertemplate=
                '<br><b>Resource</b>: %{x}</br>' +
                '<b>Value</b>: %{y:.2e}' + f' {unit}</br>',
            ),
        )

        fig.update_layout(showlegend=False)

    # Update layout
    fig.update_layout(
        title_text=graph_title,
        template='plotly_white',
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


@staticmethod
def plot_indicators_of_technologies_for_several_impact_categories(
        R: pd.DataFrame,
        technologies_list: list[str],
        impact_categories_list: list[tuple],
        filename: str = None,
        saving_format: str = None,
        saving_path: str = None,
        show_plot: bool = True,
):
    """
    Plot operation and infrastructure LCA indicators for a set of technologies for a set of impact categories.

    :param R: dataframe of LCA indicators
    :param technologies_list: list of technologies to plot. They should have the same operation/infrastructure units
        to have a meaningful comparison
    :param impact_categories_list: list of impact category to plot (brightway format)
    :param filename: name of the file to save the plot. If None, the plot is named with the impact category.
    :param saving_format: format to save the plot, can be 'png', 'jpeg', 'pdf', 'html', etc.
    :param saving_path: path to save the plot under the form 'path/to/folder/'. If None, the plot is saved in the
        current directory.
    :param show_plot: if True, the plot is shown in the notebook.
    :return: None (plot is shown and/or saved)
    """

    if saving_path is None:
        saving_path = ''

    for tech in technologies_list:
        if tech not in R['Name'].unique():
            raise ValueError(f'Technology {tech} not found in the LCA indicators dataframe')

    for impact_category in impact_categories_list:
        if str(impact_category) not in R['Impact_category'].unique():
            raise ValueError(f'Impact category {impact_category} not found in the LCA indicators dataframe')

    data_op = []
    data_constr = []

    color_scale = px.colors.qualitative.Plotly

    for i, impact_category in enumerate(impact_categories_list):

        unit = bd.Method(impact_category).metadata['unit']

        df_op = R[
            (R['Name'].isin(technologies_list))
            & (R['Impact_category'] == str(impact_category))
            & (R['Type'] == 'Operation')
            ]

        data_op.append(go.Bar(
            name=impact_category[-1],
            x=df_op['Name'],
            y=100*df_op['Value']/df_op['Value'].max(),
            marker=dict(color=color_scale[i % len(color_scale)]),
            hovertemplate=
            '<br><b>Technology</b>: %{x}</br>' +
            '<b>Relative value</b>: %{y:.2f}%</br>' +
            '<b>Physical value</b>: %{customdata:.2e}' + f' {unit}</br>',
            customdata=df_op['Value'],
        ))

        df_constr = R[
            (R['Name'].isin(technologies_list))
            & (R['Impact_category'] == str(impact_category))
            & (R['Type'] == 'Construction')
            ]

        data_constr.append(go.Bar(
            name=impact_category[-1],
            x=df_constr['Name'],
            y=100*df_constr['Value']/df_constr['Value'].max(),
            marker=dict(color=color_scale[i % len(color_scale)]),
            hovertemplate=
            '<br><b>Technology</b>: %{x}</br>' +
            '<b>Relative value</b>: %{y:.2f}%</br>' +
            '<b>Physical value</b>: %{customdata:.2e}' + f' {unit}</br>',
            customdata=df_constr['Value'],
            ))

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Operation", "Construction"),
        )

    # Add bar chart for operation
    for trace in data_op:
        fig.add_trace(trace, row=1, col=1)

    # Add bar chart for construction
    for trace in data_constr:
        trace.showlegend = False  # Hide legend for the second subplot
        fig.add_trace(trace, row=1, col=2)

    fig.update_layout(barmode='group')

    # Update axis labels
    fig.update_yaxes(title_text=f'Relative impacts [%]', row=1, col=1)

    # Update layout
    fig.update_layout(
        template='plotly_white',
        showlegend=True,
        legend_title_text='Impact categories',
    )

    # Show plot
    if show_plot:
        fig.show()

    if saving_format is None or filename is None:
        pass
    elif saving_format == 'html':
        Path(saving_path).mkdir(parents=True, exist_ok=True)  # Create the folder if it does not exist
        fig.write_html(f'{saving_path}{filename}.{saving_format}')
    else:
        Path(saving_path).mkdir(parents=True, exist_ok=True)  # Create the folder if it does not exist
        fig.write_image(f'{saving_path}{filename}.{saving_format}')


@staticmethod
def plot_indicators_of_resources_for_several_impact_categories(
        R: pd.DataFrame,
        resources_list: list[str],
        impact_categories_list: list[tuple],
        filename: str = None,
        saving_format: str = None,
        saving_path: str = None,
        show_plot: bool = True,
):
    """
    Plot LCA indicators for a set of resources for a set of impact categories.

    :param R: dataframe of LCA indicators
    :param resources_list: list of technologies to plot. They should have the same operation/infrastructure units
        to have a meaningful comparison
    :param impact_categories_list: list of impact category to plot (brightway format)
    :param filename: name of the file to save the plot. If None, the plot is named with the impact category.
    :param saving_format: format to save the plot, can be 'png', 'jpeg', 'pdf', 'html', etc.
    :param saving_path: path to save the plot under the form 'path/to/folder/'. If None, the plot is saved in the
        current directory.
    :param show_plot: if True, the plot is shown in the notebook.
    :return: None (plot is shown and/or saved)
    """

    if saving_path is None:
        saving_path = ''

    for res in resources_list:
        if res not in R['Name'].unique():
            raise ValueError(f'Resource {res} not found in the LCA indicators dataframe')

    for impact_category in impact_categories_list:
        if str(impact_category) not in R['Impact_category'].unique():
            raise ValueError(f'Impact category {impact_category} not found in the LCA indicators dataframe')

    data = []

    for impact_category in impact_categories_list:

        unit = bd.Method(impact_category).metadata['unit']
        df = R[(R['Name'].isin(resources_list)) & (R['Impact_category'] == str(impact_category))]

        data.append(go.Bar(
            name=impact_category[-1],
            x=df['Name'],
            y=100*df['Value']/df['Value'].max(),
            hovertemplate=
            '<br><b>Resource</b>: %{x}</br>' +
            '<b>Relative value</b>: %{y:.2f}%</br>' +
            '<b>Physical value</b>: %{customdata:.2e}' + f' {unit}</br>',
            customdata=df['Value'],
        ))

    fig = go.Figure(data=data)
    fig.update_layout(barmode='group')

    # Update axis labels
    fig.update_yaxes(title_text=f'Relative impacts [%]')

    # Update layout
    fig.update_layout(
        template='plotly_white',
        showlegend=True,
        legend_title_text='Impact categories',
    )

    # Show plot
    if show_plot:
        fig.show()

    if saving_format is None or filename is None:
        pass
    elif saving_format == 'html':
        Path(saving_path).mkdir(parents=True, exist_ok=True)  # Create the folder if it does not exist
        fig.write_html(f'{saving_path}{filename}.{saving_format}')
    else:
        Path(saving_path).mkdir(parents=True, exist_ok=True)  # Create the folder if it does not exist
        fig.write_image(f'{saving_path}{filename}.{saving_format}')
