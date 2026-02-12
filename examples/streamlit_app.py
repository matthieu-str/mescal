import argparse
from pathlib import Path
import shutil
import subprocess

import pandas as pd
import streamlit as st

from mescal.contribution_analysis import process_contribution_data, _export_comprehensive_excel
from mescal.plot import plot_contribution_analysis

import sys
sys.path.insert(1, '../')


st.set_page_config(
    page_title="MESCAL Contribution Analysis",
    page_icon="📊",
    layout="wide",
)

st.title("🔬 MESCAL Contribution Analysis Visualization")
st.markdown("---")


def _parse_cli_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--contrib_data_path")
    parser.add_argument("--contrib_data_path_processes")
    parser.add_argument("--contrib_data_path_emissions")
    parser.add_argument("--impact_scores_path")
    parser.add_argument("--unit_conversion_path")
    return parser.parse_known_args()[0]


def launch_streamlit_app(
    impact_scores_df: pd.DataFrame,
    unit_conversion_df: pd.DataFrame,
    contrib_df: pd.DataFrame = None,
    contrib_processes_df: pd.DataFrame = None,
    contrib_emissions_df: pd.DataFrame = None,
    app_path: str = "streamlit_app.py",
):
    """
    Launch the Streamlit app from Python with DataFrames.

    Since Streamlit runs as a subprocess, DataFrames are saved to temporary files
    and paths are passed via CLI arguments.

    :param impact_scores_df: DataFrame with impact scores
    :param unit_conversion_df: DataFrame with unit conversions
    :param contrib_df: Single contribution DataFrame (legacy, use processes/emissions instead)
    :param contrib_processes_df: Contribution DataFrame for processes
    :param contrib_emissions_df: Contribution DataFrame for emissions
    :param app_path: Path to the Streamlit app file
    """
    temp_dir = Path("temp_streamlit")
    temp_dir.mkdir(exist_ok=True)

    cmd = [
        "streamlit",
        "run",
        str(Path(app_path)),
        "--",
    ]

    # Save DataFrames to temp files and build CLI args
    if contrib_df is not None:
        contrib_path = temp_dir / "contrib_data.csv"
        contrib_df.to_csv(contrib_path, index=False)
        cmd.extend(["--contrib_data_path", str(contrib_path)])
    if contrib_processes_df is not None:
        contrib_processes_path = temp_dir / "contrib_data_processes.csv"
        contrib_processes_df.to_csv(contrib_processes_path, index=False)
        cmd.extend(["--contrib_data_path_processes", str(contrib_processes_path)])
    if contrib_emissions_df is not None:
        contrib_emissions_path = temp_dir / "contrib_data_emissions.csv"
        contrib_emissions_df.to_csv(contrib_emissions_path, index=False)
        cmd.extend(["--contrib_data_path_emissions", str(contrib_emissions_path)])

    impact_path = temp_dir / "impact_scores.csv"
    unit_path = temp_dir / "unit_conversion.csv"
    impact_scores_df.to_csv(impact_path, index=False)
    unit_conversion_df.to_csv(unit_path, index=False)

    cmd.extend([
        "--impact_scores_path",
        str(impact_path),
        "--unit_conversion_path",
        str(unit_path),
    ])
    subprocess.run(cmd, check=True)


def run_contribution_analysis(
        contrib_df: pd.DataFrame,
        impact_scores_df: pd.DataFrame,
        unit_conversion_df: pd.DataFrame,
        contribution_type: str,
        saving_path: str,
        export_excel: bool = True,
        impact_categories_list: list[str] = None,
        annot_fmt: str = ".1%",
        threshold: float = None,
        cell_size: float = None,
        dpi: int = None,
) -> tuple[pd.DataFrame, dict]:
    grouped_df, unit_type_groups_dict = process_contribution_data(
        contrib_df=contrib_df,
        impact_scores_df=impact_scores_df,
        unit_conversion_df=unit_conversion_df,
        contribution_type=contribution_type,
        saving_path=saving_path,
        export_excel=export_excel,
    )

    plot_kwargs = {
        "df": grouped_df,
        "unit_type_groups_dict": unit_type_groups_dict,
        "saving_path": saving_path,
        "contribution_type": contribution_type,
        "annot_fmt": annot_fmt,
    }

    if impact_categories_list:
        plot_kwargs["impact_categories_list"] = impact_categories_list
    if threshold is not None:
        plot_kwargs["threshold"] = threshold
    if cell_size is not None:
        plot_kwargs["cell_size"] = cell_size
    if dpi is not None:
        plot_kwargs["dpi"] = dpi

    plot_contribution_analysis(**plot_kwargs)

    return grouped_df, unit_type_groups_dict


cli_args = _parse_cli_args()
cli_paths_provided = all(
    [
        cli_args.impact_scores_path,
        cli_args.unit_conversion_path,
    ]
) and any(
    [
        cli_args.contrib_data_path,
        cli_args.contrib_data_path_processes,
        cli_args.contrib_data_path_emissions,
    ]
)


with st.sidebar:
    contrib_file = None
    impact_scores_file = None
    unit_conversion_file = None

    st.header("⚙️ Settings")

    cli_has_processes = bool(cli_args.contrib_data_path_processes)
    cli_has_emissions = bool(cli_args.contrib_data_path_emissions)
    if cli_paths_provided and (cli_has_processes or cli_has_emissions):
        available_types = []
        if cli_has_processes:
            available_types.append("processes")
        if cli_has_emissions:
            available_types.append("emissions")
    else:
        available_types = ["processes", "emissions"]

    contribution_type = st.radio(
        "Contribution Type",
        options=available_types,
        index=0,
        format_func=str.capitalize,
    )

    st.markdown("---")
    st.subheader("Plot Parameters")

    annot_fmt = st.text_input("Annotation Format", ".1%")

    use_threshold = st.checkbox("Set custom threshold", value=False)
    threshold = st.slider("Threshold", 0.0, 0.2, 0.03, 0.01) if use_threshold else None

    use_cell_size = st.checkbox("Set custom cell size", value=False)
    cell_size = st.slider("Cell Size", 0.4, 1.5, 0.6, 0.1) if use_cell_size else 0.6

    use_min_fig = st.checkbox("Set minimum figure size", value=False)
    min_fig_width = st.slider("Min Fig Width", 6.0, 12.0, 8.0, 0.5) if use_min_fig else 8.0
    min_fig_height = st.slider("Min Fig Height", 4.0, 10.0, 5.0, 0.5) if use_min_fig else 5.0

    use_dpi = st.checkbox("Set custom DPI", value=False)
    dpi = st.slider("DPI (Resolution)", 100, 300, 150, 50) if use_dpi else None


if cli_paths_provided or (contrib_file and impact_scores_file and unit_conversion_file):
    try:
        if cli_paths_provided:
            if cli_args.contrib_data_path_processes or cli_args.contrib_data_path_emissions:
                if contribution_type == "processes" and cli_args.contrib_data_path_processes:
                    contrib_path = Path(cli_args.contrib_data_path_processes)
                elif contribution_type == "emissions" and cli_args.contrib_data_path_emissions:
                    contrib_path = Path(cli_args.contrib_data_path_emissions)
                else:
                    raise ValueError("Selected contribution type doesn't have a provided CSV path.")
            else:
                contrib_path = Path(cli_args.contrib_data_path)
            impact_path = Path(cli_args.impact_scores_path)
            unit_path = Path(cli_args.unit_conversion_path)
            saving_path = Path("temp_streamlit/output")
            saving_path.mkdir(parents=True, exist_ok=True)
        else:
            # Create temporary directory for uploaded files
            # Note: This folder will be deleted once the app is closed
            temp_dir = Path("temp_streamlit")
            temp_dir.mkdir(exist_ok=True)

            contrib_path = temp_dir / "contrib_data.csv"
            impact_path = temp_dir / "impact_scores.csv"
            unit_path = temp_dir / "unit_conversion.csv"
            saving_path = temp_dir / "output"

            with open(contrib_path, "wb") as f:
                f.write(contrib_file.getbuffer())
            with open(impact_path, "wb") as f:
                f.write(impact_scores_file.getbuffer())
            with open(unit_path, "wb") as f:
                f.write(unit_conversion_file.getbuffer())

        with st.spinner("Loading data..."):
            # Load CSVs into DataFrames
            contrib_df = pd.read_csv(contrib_path)
            impact_scores_df = pd.read_csv(impact_path)
            unit_conversion_df = pd.read_csv(unit_path)

            columns = set(contrib_df.columns)

            processes_cols = {"process_name"}
            emissions_cols = {"ef_name"}

            effective_contribution_type = contribution_type
            if contribution_type == "processes" and not processes_cols.issubset(columns):
                if emissions_cols.issubset(columns):
                    effective_contribution_type = "emissions"
                    st.warning("Selected type is 'processes' but CSV looks like emissions. Switching to 'emissions'.")
                else:
                    missing = sorted(processes_cols - columns)
                    raise KeyError(f"Missing required columns for processes: {missing}")
            elif contribution_type == "emissions" and not emissions_cols.issubset(columns):
                if processes_cols.issubset(columns):
                    effective_contribution_type = "processes"
                    st.warning("Selected type is 'emissions' but CSV looks like processes. Switching to 'processes'.")
                else:
                    missing = sorted(emissions_cols - columns)
                    raise KeyError(f"Missing required columns for emissions: {missing}")

            grouped_df, unit_type_groups_dict = process_contribution_data(
                contrib_df=contrib_df,
                impact_scores_df=impact_scores_df,
                unit_conversion_df=unit_conversion_df,
                contribution_type=effective_contribution_type,
                saving_path=str(saving_path),
                export_excel=False,
            )

        st.success("Data loaded successfully!")

        st.subheader("Select Impact Categories")
        impact_categories_list = (
            grouped_df["impact_category"].dropna().unique().tolist()
            if "impact_category" in grouped_df.columns
            else []
        )
        selected_impact = st.multiselect(
            "Impact Categories (leave empty for all)",
            options=impact_categories_list,
            default=[],
        )

        st.subheader("Select Activity Types")
        act_type_options = (
            grouped_df["act_type"].dropna().unique().tolist()
            if "act_type" in grouped_df.columns
            else []
        )
        selected_act_types = st.multiselect(
            "Activity Types (leave empty for all)",
            options=act_type_options,
            default=[],
        )

        st.subheader("Select ESM Units")
        esm_options = sorted({esm for esm, _ in unit_type_groups_dict.keys()})
        selected_esm = st.multiselect(
            "ESM Units (leave empty for all)",
            options=esm_options,
            default=[],
        )

        impact_categories_param = selected_impact if selected_impact else None
        act_types_param = selected_act_types if selected_act_types else None
        esm_units_param = selected_esm if selected_esm else None

        filtered_df = grouped_df.copy()
        if impact_categories_param:
            filtered_df = filtered_df[filtered_df["impact_category"].isin(impact_categories_param)]
        if act_types_param:
            filtered_df = filtered_df[filtered_df["act_type"].isin(act_types_param)]
        if esm_units_param:
            allowed_act_types = act_types_param or filtered_df["act_type"].dropna().unique().tolist()
            allowed_techs = set()
            for esm in esm_units_param:
                for at in allowed_act_types:
                    allowed_techs.update(unit_type_groups_dict.get((esm, at), []))
            if allowed_techs:
                filtered_df = filtered_df[filtered_df["act_name"].isin(allowed_techs)]
            else:
                filtered_df = filtered_df.iloc[0:0]

        st.markdown("---")
        st.subheader("Data Preview")
        st.dataframe(filtered_df.head(20), width='stretch')

        export_clicked = st.button("Export data to Excel", width='stretch')
        if export_clicked:
            export_dir = saving_path / "export"
            export_dir.mkdir(parents=True, exist_ok=True)

            detail_col = "process_name" if effective_contribution_type == "processes" else "ef_name"
            export_act_types = act_types_param or ["Construction", "Operation", "Resource"]

            export_unit_type_groups_dict = {
                key: val
                for key, val in unit_type_groups_dict.items()
                if key[1] in export_act_types
                and (esm_units_param is None or key[0] in esm_units_param)
            }

            _export_comprehensive_excel(
                filtered_df,
                export_unit_type_groups_dict,
                str(export_dir),
                export_act_types,
                effective_contribution_type,
                detail_col,
            )

            if effective_contribution_type == "processes":
                export_name = "contribution_analysis_processes_results.xlsx"
            else:
                export_name = "contribution_analysis_emissions_results.xlsx"
            export_path = export_dir / export_name

            if export_path.exists():
                with open(export_path, "rb") as f:
                    st.download_button(
                        label="Download Excel",
                        data=f,
                        file_name=export_name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        width='stretch',
                    )

        st.markdown("---")
        plot_files = []
        if st.button("🎨 Plotting contribution analysis", type="primary", width='stretch'):
            with st.spinner("Running contribution analysis..."):
                if saving_path.exists():
                    shutil.rmtree(saving_path)
                saving_path.mkdir(parents=True)

                plot_kwargs = {
                    "df": grouped_df,
                    "unit_type_groups_dict": unit_type_groups_dict,
                    "saving_path": str(saving_path),
                    "contribution_type": effective_contribution_type,
                    "annot_fmt": annot_fmt,
                }

                if impact_categories_param:
                    plot_kwargs["impact_categories_list"] = impact_categories_param
                if act_types_param:
                    plot_kwargs["act_types"] = act_types_param
                if esm_units_param:
                    plot_kwargs["esm_units"] = esm_units_param
                if threshold is not None:
                    plot_kwargs["threshold"] = threshold
                if cell_size is not None:
                    plot_kwargs["cell_size"] = cell_size
                if min_fig_width is not None:
                    plot_kwargs["min_fig_width"] = min_fig_width
                if min_fig_height is not None:
                    plot_kwargs["min_fig_height"] = min_fig_height
                if dpi is not None:
                    plot_kwargs["dpi"] = dpi

                plot_contribution_analysis(**plot_kwargs)
                plot_files = sorted(saving_path.rglob("*.png"))

            st.success("Analysis complete!")

        st.markdown("---")
        st.subheader("Generated Plots")
        if plot_files:
            for plot_file in plot_files:
                st.image(str(plot_file), caption=plot_file.name, width='stretch')
        else:
            st.info("Run the plotting to display figures here.")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)
else:
    st.info("👆 Please upload all three required CSV files in the sidebar to begin.")
