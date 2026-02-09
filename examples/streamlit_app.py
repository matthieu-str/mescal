import argparse
from io import BytesIO
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
    impact_scores_path,
    unit_conversion_path,
    contrib_data_path=None,
    contrib_data_path_processes=None,
    contrib_data_path_emissions=None,
    app_path="streamlit_app.py",
):
    cmd = [
        "streamlit",
        "run",
        str(Path(app_path)),
        "--",
    ]
    if contrib_data_path:
        cmd.extend(["--contrib_data_path", str(contrib_data_path)])
    if contrib_data_path_processes:
        cmd.extend(["--contrib_data_path_processes", str(contrib_data_path_processes)])
    if contrib_data_path_emissions:
        cmd.extend(["--contrib_data_path_emissions", str(contrib_data_path_emissions)])
    cmd.extend([
        "--impact_scores_path",
        str(impact_scores_path),
        "--unit_conversion_path",
        str(unit_conversion_path),
    ])
    subprocess.run(cmd, check=True)


def run_contribution_analysis(
    contrib_data_path,
    impact_scores_path,
    unit_conversion_path,
    contribution_type,
    output_dir,
    export_excel=True,
    impact_categories=None,
    annot_fmt=".1%",
    threshold=None,
    cell_size=None,
    dpi=None,
):
    grouped_df, unit_type_groups_dict = process_contribution_data(
        contrib_data_path=contrib_data_path,
        impact_scores_path=impact_scores_path,
        unit_conversion_path=unit_conversion_path,
        contribution_type=contribution_type,
        output_dir=output_dir,
        export_excel=export_excel,
    )

    plot_kwargs = {
        "df": grouped_df,
        "unit_type_groups_dict": unit_type_groups_dict,
        "output_dir": output_dir,
        "contribution_type": contribution_type,
        "annot_fmt": annot_fmt,
    }

    if impact_categories:
        plot_kwargs["impact_categories"] = impact_categories
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
            output_dir = Path("temp_streamlit/output")
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Create temporary directory for uploaded files
            # Note: This folder will be deleted once the app is closed
            temp_dir = Path("temp_streamlit")
            temp_dir.mkdir(exist_ok=True)

            contrib_path = temp_dir / "contrib_data.csv"
            impact_path = temp_dir / "impact_scores.csv"
            unit_path = temp_dir / "unit_conversion.csv"
            output_dir = temp_dir / "output"

            with open(contrib_path, "wb") as f:
                f.write(contrib_file.getbuffer())
            with open(impact_path, "wb") as f:
                f.write(impact_scores_file.getbuffer())
            with open(unit_path, "wb") as f:
                f.write(unit_conversion_file.getbuffer())

        with st.spinner("Loading data..."):
            header_df = pd.read_csv(contrib_path, nrows=1)
            columns = set(header_df.columns)

            processes_cols = {"process_name", "process_reference_product"}
            emissions_cols = {"ef_name", "ef_categories"}

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
                contrib_data_path=str(contrib_path),
                impact_scores_path=str(impact_path),
                unit_conversion_path=str(unit_path),
                contribution_type=effective_contribution_type,
                output_dir=str(output_dir),
                export_excel=False,
            )

        st.success("Data loaded successfully!")

        col_left, col_right = st.columns([1, 2], gap="large")
        plot_files = []

        with col_left:
            st.subheader("Select Impact Categories")
            impact_categories = (
                grouped_df["impact_category"].dropna().unique().tolist()
                if "impact_category" in grouped_df.columns
                else []
            )
            selected_impact = st.multiselect(
                "Impact Categories (leave empty for all)",
                options=impact_categories,
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
            esm_keys_param = selected_esm if selected_esm else None

            filtered_df = grouped_df.copy()
            if impact_categories_param:
                filtered_df = filtered_df[filtered_df["impact_category"].isin(impact_categories_param)]
            if act_types_param:
                filtered_df = filtered_df[filtered_df["act_type"].isin(act_types_param)]
            if esm_keys_param:
                allowed_act_types = act_types_param or filtered_df["act_type"].dropna().unique().tolist()
                allowed_techs = set()
                for esm in esm_keys_param:
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
                export_dir = output_dir / "export"
                export_dir.mkdir(parents=True, exist_ok=True)

                detail_col = "process_name" if effective_contribution_type == "processes" else "ef_name"
                export_act_types = act_types_param or ["Construction", "Operation", "Resource"]

                export_unit_type_groups_dict = {
                    key: val
                    for key, val in unit_type_groups_dict.items()
                    if key[1] in export_act_types
                    and (esm_keys_param is None or key[0] in esm_keys_param)
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
            if st.button("🎨 Plotting contribution analysis", type="primary", width='stretch'):
                with st.spinner("Running contribution analysis..."):
                    if output_dir.exists():
                        shutil.rmtree(output_dir)
                    output_dir.mkdir(parents=True)

                    plot_kwargs = {
                        "df": grouped_df,
                        "unit_type_groups_dict": unit_type_groups_dict,
                        "output_dir": str(output_dir),
                        "contribution_type": effective_contribution_type,
                        "annot_fmt": annot_fmt,
                    }

                    if impact_categories_param:
                        plot_kwargs["impact_categories"] = impact_categories_param
                    if act_types_param:
                        plot_kwargs["act_types"] = act_types_param
                    if esm_keys_param:
                        plot_kwargs["esm_keys"] = esm_keys_param
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
                    plot_files = sorted(output_dir.rglob("*.png"))

                st.success("Analysis complete!")

        with col_right:
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
