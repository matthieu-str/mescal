from bw2analyzer import ContributionAnalysis

from typing import Optional
import numpy as np
import os
import pandas as pd
import re

class ABContributionAnalysis(ContributionAnalysis):
    """Activity Browser version of bw2analyzer.ContributionAnalysis"""
    def sort_array(self, data: np.array, limit: float = 25, limit_type: str = "number", total: Optional[float] = None) -> np.array:
        """Activity Browser version of bw2analyzer.ContributionAnalysis.sort_array.

        Should be removed once https://github.com/brightway-lca/brightway2-analyzer/pull/32 is merged.
        See PR above on why we overwrite this function.
        """
        if not total:
            total = np.abs(data).sum()

        if total == 0 and limit_type == "cum_percent":
            raise ValueError(
                "Cumulative percentage cannot be calculated to a total of 0, use a different limit type or total")

        if limit_type not in ("number", "percent", "cum_percent"):
            raise ValueError(f"limit_type must be either 'number', 'percent' or 'cum_percent' not '{limit_type}'.")
        if limit_type in ("percent", "cum_percent"):
            if not 0 < limit <= 1:
                raise ValueError("Percentage limits > 0 and <= 1.")
        if limit_type == "number":
            if not int(limit) == limit:
                raise ValueError("Number limit must a whole number.")
            if not 0 < limit:
                raise ValueError("Number limit must be < 0.")

        results = np.hstack(
            (data.reshape((-1, 1)), np.arange(data.shape[0]).reshape((-1, 1)))
        )

        if limit_type == "number":
            # sort and cut off at limit
            return results[np.argsort(np.abs(data))[::-1]][:limit, :]
        elif limit_type == "percent":
            # identify good values, drop rest and sort
            limit = (np.abs(data) >= (abs(total) * limit))
            results = results[limit, :]
            return results[np.argsort(np.abs(results[:, 0]))[::-1]]
        elif limit_type == "cum_percent":
            # if we would apply this on the 'correct' order, this would stop just before the limit,
            # we want to be on or the first step over the limit.
            results = results[np.argsort(np.abs(data))]  # sort low to high impact
            cumsum = np.cumsum(np.abs(results[:, 0])) / abs(total)
            limit = (cumsum >= (1 - limit))  # find items under limit
            return results[limit, :][::-1]  # drop items under limit and set correct order
        
def process_contribution_data(
        contrib_df: pd.DataFrame,
        impact_scores_df: pd.DataFrame,
        unit_conversion_df: pd.DataFrame,
        contribution_type: str = 'processes',
        saving_path: str = None,
        export_excel: bool = False,
        act_types: list[str] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Process contribution analysis data for environmental impacts.

    :param contrib_df: contribution analysis dataframe (processes or emissions)
    :param impact_scores_df: impact scores dataframe with total impacts
    :param unit_conversion_df: unit conversion dataframe
    :param contribution_type: Type of contribution analysis: 'processes' or 'emissions'
    :param saving_path: Output directory for Excel file (required if export_excel=True)
    :param export_excel: Whether to export comprehensive Excel file
    :param act_types: List of activity types for Excel export
    :return: Processed DataFrame with impact_share column, Unit type groups dictionary
    """
    # Define column mappings based on contribution type
    detail_col = 'process_name' if contribution_type == 'processes' else 'ef_name'
    
    if contribution_type not in ['processes', 'emissions']:
        raise ValueError("contribution_type must be 'processes' or 'emissions'")
    
    # Filter and prepare contribution data
    contrib_df = contrib_df[['act_name', 'impact_category', 'score', 'act_type', detail_col]]
    
    # Rename columns for merge
    impact_scores_df = impact_scores_df.rename(columns={
        'Name': 'act_name',
        'Impact_category': 'impact_category',
        'Type': 'act_type',
        'Value': 'total_impact'
    })
    
    # Merge dataframes
    merged_df = pd.merge(
        impact_scores_df[['act_name', 'impact_category', 'act_type', 'total_impact']],
        contrib_df,
        on=['act_name', 'impact_category', 'act_type'],
        how='inner'
    )
    
    # Clean detail column names (process_name or ef_name)
    def split_name(s):
        parts = re.split(r',(?!\d)', s)
        return parts[0] if parts else s
    
    merged_df[detail_col] = merged_df[detail_col].apply(split_name)
    
    # Group and aggregate
    grouped_df = merged_df.groupby(
        ['act_name', 'act_type', 'impact_category', detail_col]
    ).agg({'score': 'sum', 'total_impact': 'first'}).reset_index()
    
    # Calculate impact share
    grouped_df['total_impact'] = grouped_df.groupby(
        ['act_name', 'act_type', 'impact_category']
    )['score'].transform('sum')
    grouped_df['impact_share'] = grouped_df['score'] / grouped_df['total_impact']
    
    # Add 'Others' category
    Others_rows = []
    for keys, group in grouped_df.groupby(['act_name', 'impact_category', 'act_type']):
        total_share = group['impact_share'].sum()
        Others_share = 1 - total_share
        if Others_share > 0.01:
            Others_rows.append({
                'act_name': keys[0],
                'impact_category': keys[1],
                'act_type': keys[2],
                detail_col: 'Others',
                'score': None,
                'total_impact': group['total_impact'].iloc[0],
                'impact_share': Others_share
            })
    
    if Others_rows:
        grouped_df = pd.concat([grouped_df, pd.DataFrame(Others_rows)], ignore_index=True)
    
    # Load unit conversion mapping
    unit_conversion_df = unit_conversion_df[
        (unit_conversion_df['ESM'] != 'unit') &
        (unit_conversion_df['Type'] != 'Other') &
        (unit_conversion_df['Type'] != 'Flow')
    ]
    
    unit_type_groups_dict = {}
    for _, row in unit_conversion_df.groupby(['ESM', 'Type'])['Name'].apply(list).reset_index().iterrows():
        key = (row['ESM'], row['Type'])
        unit_type_groups_dict[key] = row['Name']
    
    # Export to Excel if requested
    if export_excel:
        if saving_path is None:
            raise ValueError("saving_path must be provided when export_excel=True")
        
        if act_types is None:
            act_types = ['Construction', 'Decommission', 'Operation', 'Resource']
        
        _export_comprehensive_excel(
            grouped_df, 
            unit_type_groups_dict, 
            saving_path,
            act_types,
            contribution_type,
            detail_col
        )
    
    return grouped_df, unit_type_groups_dict

def _export_comprehensive_excel(
        df: pd.DataFrame,
        unit_type_groups_dict: dict,
        saving_path: str,
        act_types: list[str],
        contribution_type: str,
        detail_col: str,
) -> None:
    """
    Internal function to export comprehensive Excel file.

    :param df: DataFrame with contribution analysis results
    :param unit_type_groups_dict: Dictionary mapping (ESM, Type) to list of technology names
    :param saving_path: Output directory for Excel file
    :param act_types: List of activity types to include in Excel export
    :param contribution_type: Type of contribution analysis: 'processes' or 'emissions'
    :param detail_col: Column name for process or emission details ('process_name' or 'ef_name')
    :return: None
    """
    os.makedirs(saving_path, exist_ok=True)
    
    # Set filename based on contribution type
    if contribution_type == 'processes':
        filename = 'contribution_analysis_processes_results.xlsx'
    else:
        filename = 'contribution_analysis_emissions_results.xlsx'
    
    output_path = os.path.join(saving_path, filename)
    
    impact_categories = df['impact_category'].unique().tolist()
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for impact_category in impact_categories:
            df_cat = df[df['impact_category'] == impact_category].copy()
            
            # Calculate 'Others' share for small contributions
            Others_share = df_cat[df_cat['impact_share'] <= 0.05].groupby(
                ['act_name', 'act_type', 'impact_category']
            )['impact_share'].sum().reset_index()
            Others_share[detail_col] = 'Others'
            
            df_cat = df_cat[df_cat['impact_share'] > 0.05]
            if not Others_share.empty:
                df_cat = pd.concat([df_cat, Others_share], ignore_index=True)
            
            if df_cat.empty:
                continue
            
            # Get unique ESM keys
            esm_keys = sorted(set(esm for esm, typ in unit_type_groups_dict.keys() if typ in act_types))
            
            sheet_data = []
            
            for at in act_types:
                for esm in esm_keys:
                    tech_names = unit_type_groups_dict.get((esm, at), [])
                    sub = df_cat[(df_cat['act_type'] == at) & (df_cat['act_name'].isin(tech_names))]
                    
                    if sub.empty:
                        continue
                    
                    # Add metadata columns
                    sub = sub.copy()
                    sub['esm_group'] = esm
                    
                    # Convert impact_share to percentage format
                    sub['impact_share_pct'] = sub['impact_share'] * 100
                    
                    # Reorder columns for clarity
                    sub = sub[['act_type', 'esm_group', 'act_name', 
                              detail_col, 'impact_share_pct']]
                    
                    sheet_data.append(sub)
                    
                    # Add blank row separator between groups
                    blank_row = pd.DataFrame([{
                        'act_type': '',
                        'esm_group': '',
                        'act_name': '',
                        detail_col: '',
                        'impact_share_pct': None
                    }])
                    sheet_data.append(blank_row)
            
            if sheet_data:
                # Combine all data for this impact category
                sheet_df = pd.concat(sheet_data, ignore_index=True)
                
                # Rename column for clarity
                sheet_df = sheet_df.rename(columns={'impact_share_pct': 'Impact Share (%)'})
                
                # Create safe sheet name (Excel has 31 char limit)
                safe_sheet_name = str(impact_category).replace('/', '_').replace(':', '_').replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_').replace("'", "")[:31]
                
                # Write to Excel
                sheet_df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
                
                # Get worksheet to format
                worksheet = writer.sheets[safe_sheet_name]
                
                # Format the Impact Share column as percentage with 1 decimal
                for row in range(2, len(sheet_df) + 2):
                    cell = worksheet.cell(row=row, column=5)
                    if cell.value is not None and isinstance(cell.value, (int, float)):
                        cell.number_format = '0.0"%"'
    
    print(f"Comprehensive Excel saved to: {output_path}")
    print(f"Created {len(impact_categories)} sheets")