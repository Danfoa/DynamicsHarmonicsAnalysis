import copy
import json
import os
from typing import Iterable, Optional, Union

import pandas as pd
import numpy as np
from pathlib import Path
import itertools

from plotly.graph_objs import Figure
import plotly.graph_objs as go
import plotly.express as px
from tqdm import tqdm
from wandb.apis.public import RunArtifacts
from wandb.sdk.wandb_summary import SummarySubDict


def plot_aggregated_lines(df: pd.DataFrame,
                          x: str,
                          y: str,
                          group_variables: Union[str, Iterable[str]],
                          color_sequence=px.colors.qualitative.Dark2,
                          area_alpha=0.3,
                          line_width=2,
                          line_styles: Optional[dict] = None,
                          color_group: Optional[dict] = None,
                          area_metric="min-max",
                          label_replace: Optional[dict] = None) -> Figure:
    """ Plot aggregated lines showing mean/min/max of y for each value of x for every combination of group_variables

    Args:
        df: Dataframe containing the data to plot
        x: Name of the column in df to use as the x-axis
        y: Name of the column in df to use as the y-axis
        group_variables: List of column names in df to use for grouping
        color_sequence: List of colors to use for the lines
        area_alpha: Opacity of the shaded area
        line_styles: Dictionary mapping (potentially substrings of) groups labels to line styles
        color_group: Dictionary mapping (potentially substrings of) groups labels to line styles
    Returns:
        fig: Figure object
    """
    assert x in df.columns, f"x={x} not in df.columns: {df.columns}"
    assert y in df.columns, f"y={y} not in df.columns: {df.columns}"

    fig = go.Figure()

    # Get a sorted array of the unique values of x
    x_vals = df[x].unique()

    # In case group_labels is a single string, convert to list then compute the direct product
    if isinstance(group_variables, str):
        group_variables = [group_variables]

    # Compute all the possible combinations of the grouping variables
    unique_group_vals = {}
    for group_var in group_variables:
        assert group_var in df.columns, f"group_var={group_var} not in df.columns: {df.columns}"
        unique_group_vals[group_var] = set(df[group_var].unique())
    direct_product_group = list(itertools.product(*unique_group_vals.values()))

    # Get a view of the data corresponding to each group
    group_df = {}
    for group_var_values in direct_product_group:
        assert len(group_var_values) == len(group_variables)
        mask = pd.Series([True] * len(df), index=df.index)
        for col, val in zip(group_variables, group_var_values):
            mask = mask & (df[col] == val)
        group_df[group_var_values] = df[mask]
        if group_df[group_var_values].shape[0] <= 1:
            print(f"Warning: Group {group_var_values} has only {group_df[group_var_values].shape[0]} samples")

    line_count = 0
    show_true_legend = line_styles is None or color_group is None

    for group_var_values in sorted(direct_product_group):
        df_subgroup = group_df[group_var_values]
        group_hps = {k: v for k, v in zip(group_variables, group_var_values)}
        if df_subgroup.shape[0] == 0:
            continue
        group_label = "-".join([f"{val}" for group_var, val in zip(group_variables, group_var_values)])
        # Get the mean and std of y for each value of x
        y_mean, y_bottom, y_upper = [], [], []
        series_x_vals = []
        for x_val in x_vals:
            y_vals = df_subgroup[df_subgroup[x] == x_val][y].values
            if len(y_vals) == 0:
                continue
                # print(f"Warning: No data for {group_label} at {x_val}")
            y_mean.append(np.mean(y_vals))
            y_bottom.append(np.min(y_vals) if area_metric == "min-max" else y_mean[-1] - np.std(y_vals))
            y_upper.append(np.max(y_vals) if area_metric == "min-max" else y_mean[-1] + np.std(y_vals))
            series_x_vals.append(x_val)

        # Select the defualt color and linestyle
        color = color_sequence[line_count % len(color_sequence)]
        line_style = dict()
        for hp, val in group_hps.items():
            if line_styles is not None:
                ref_styles = line_styles.get(hp, None)
                line_style = ref_styles.get(val, line_style) if ref_styles is not None else line_style
            if color_group is not None:
                ref_colors = color_group.get(hp, None)
                color = ref_colors.get(val, color) if ref_colors is not None else color

        color_area = color.replace("rgb", "rgba").replace(")", f",{area_alpha})")
        common_kwargs = dict(line_shape='spline')

        group_label = group_label if label_replace is None else label_replace.get(group_label, group_label)
        # Mean
        fig.add_trace(
            go.Scatter(x=series_x_vals, y=y_mean, mode='lines', name=group_label, legendgroup=group_label,
                       showlegend=show_true_legend,
                       line=dict(color=color, width=line_width, **line_style), **common_kwargs, ))
        # Bottom and upper bounds
        fig.add_trace(
            go.Scatter(x=series_x_vals, y=y_bottom, mode='lines', line=dict(width=0),
                       name=group_label + ' Lower', legendgroup=group_label, showlegend=False,
                       fill=None, **common_kwargs))
        fig.add_trace(
            go.Scatter(x=series_x_vals, y=y_upper, mode='lines', line=dict(width=0),
                       name=group_label + ' Upper', legendgroup=group_label, showlegend=False,
                       fill='tonexty', fillcolor=color_area, **common_kwargs))

        line_count += 1

    if not show_true_legend:
        for hp, trace_styles in line_styles.items():
            hp_name = hp if label_replace is None else label_replace.get(hp, hp)
            # Fake Legend Titles
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                                     name=f'{hp_name}',
                                     line=dict(color='rgba(255,255,255,0)'),
                                     showlegend=True,
                                     legendrank=3))
            for name, style in trace_styles.items():
                name = name if label_replace is None else label_replace.get(name, name)
                fig.add_trace(
                    go.Scatter(x=[None], y=[None], mode='lines', name=name,
                               line=dict(color='black', width=line_width, **style),
                               legendrank=2))

        for hp, trace_colors in color_group.items():
            hp_name = hp if label_replace is None else label_replace.get(hp, hp)
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                                     name=f'{hp_name}',
                                     line=dict(color='rgba(255,255,255,0)'),
                                     showlegend=True,
                                     legendrank=1))
            for name, color in trace_colors.items():
                name = name if label_replace is None else label_replace.get(name, name)
                fig.add_trace(
                    go.Scatter(x=[None], y=[None], mode='lines', name=name, line=dict(color=color, width=line_width),
                               legendrank=0))

    fig.update_layout(plot_bgcolor='white',
                      paper_bgcolor='white',
                      # Draw mild grid lines with an alpha of 0.1
                      yaxis=dict(gridcolor='rgba(0,0,0,0.1)',
                                 showline=True,
                                 linewidth=1.0,
                                 linecolor='rgba(0,0,0,0.3)'),
                      xaxis=dict(gridcolor='rgba(0,0,0,0.1)',
                                 showline=True,
                                 linewidth=1.0,
                                 linecolor='rgba(0,0,0,0.3)'),
                      autosize=True,
                      ),
    return fig


if __name__ == "__main__":
    # Generic layout config
    desired_font_size = 28
    layout_config = dict(width=500, height=400,
                         margin=dict(l=0, r=0, t=0, b=0),
                         # Set legend box in the upper right corner inside the plot
                         legend=dict(x=1, y=1,
                                     xanchor='right',
                                     yanchor='top',
                                     font=dict(size=desired_font_size - 8),
                                     # itemsizing='constant'
                                     ),
                         xaxis=dict(
                             titlefont=dict(size=desired_font_size),  # Set desired font size for x-axis title
                             tickfont=dict(size=desired_font_size - 8),  # Set desired font size for x-axis
                             tickformat='s'
                             ),
                         yaxis=dict(
                             type="log",
                             titlefont=dict(size=desired_font_size),  # Set desired font size for y-axis title
                             tickfont=dict(size=desired_font_size - 8),  # Set desired font size for y-axis
                             ),
                         autosize=False,
                         )
    color_pallet = px.colors.qualitative.Prism
    line_styles = {'model.name': {"E-DAE":   dict(),
                                  "DAE-AUG": dict(dash='dot'),
                                  "DAE":     dict(dash='dash')}
                   }
    color_group = {'system.group': {"K4xC2": color_pallet[1],
                                    "K4":    color_pallet[7]}
                   }

    data_path = Path("mini_cheetah_sample_eff_uneven_easy_terrain.csv")
    print(data_path)
    df = pd.read_csv(data_path)
    # Ignore all records of group = C2
    df = df[df["system.group"] != "C2"]
    fig = plot_aggregated_lines(df,
                                x="system.train_ratio",
                                y="state_pred_loss/test",
                                group_variables=['model.name', 'system.group'],
                                line_styles=line_styles,
                                color_group=color_group,
                                area_metric="std",
                                label_replace={"model.name":   "Model",
                                               "system.group": "Group",
                                               "C2":           r"$\mathbb{G}=\mathbb{C}_2$",
                                               "K4":           r"$\mathbb{G}=\mathbb{K}_4$",
                                               "K4xC2":        r"$\mathbb{G}=\mathbb{K}_4 \times \mathbb{"
                                                               r"C}_2$", }
                                )
    # Set the figure size to a quarter of an A4 page
    layout_config['legend'].update(bgcolor="rgba(1.0,1.0,1.0,0.7)")
    fig.update_layout(**layout_config)
    fig.update_xaxes(title_text="training samples")
    fig.update_yaxes(title_text="state prediction MSE", type="log")
    # fig.show()
    fig.write_html(data_path.with_suffix(".html"))
    fig.write_image(data_path.with_suffix(".svg"))
    fig.write_image(data_path.with_suffix(".png"))

    # Obs state ratio =================================================
    data_path = Path("mini_cheetah_obs_state_ratio_uneven_easy_terrain.csv")
    print(data_path)
    df = pd.read_csv(data_path)
    STATE_DIM = 42
    df['system.obs_state_ratio'] = df['system.obs_state_ratio'] * STATE_DIM
    # Ignore all records of group = C2
    df = df[df["system.group"] != "C2"]
    fig = plot_aggregated_lines(df,
                                x="system.obs_state_ratio",
                                y="state_pred_loss/test",
                                group_variables=['model.name', 'system.group'],
                                line_styles=line_styles,
                                color_group=color_group,
                                area_metric="std",
                                label_replace={"model.name":   "Model",
                                               "system.group": "Group",
                                               "C2":           r"$\mathbb{G}=\mathbb{C}_2$",
                                               "K4":           r"$\mathbb{G}=\mathbb{K}_4$",
                                               "K4xC2":        r"$\mathbb{G}=\mathbb{K}_4 \times \mathbb{"
                                                               r"C}_2$", }
                                )
    # Set the figure size to a quarter of an A4 page
    layout_config['legend'].update(bgcolor="rgba(1.0,1.0,1.0,0.7)")
    fig.update_layout(**layout_config, showlegend=False)
    fig.update_xaxes(title_text="obs state dimension")
    fig.update_yaxes(title_text="state prediction MSE", type="linear")
    # fig.show()
    fig.write_html(data_path.with_suffix(".html"))
    fig.write_image(data_path.with_suffix(".svg"))
    fig.write_image(data_path.with_suffix(".png"))

    # # MSE vs Time =================================================
    # print(f"Mini Cheetah MSE vs Time")
    # import wandb
    # wandb.login()
    # api = wandb.Api()
    # project_path = "dls-csml/mini_cheetah"
    # group_name = "mse_vs_time_final"
    # metric_name = "state_pred_loss_t"
    # runs = api.runs(project_path, {"$and": [{"group": group_name}] })
    # print(f"Found {len(runs)} runs for group {group_name}")
    # df = pd.DataFrame()
    # download_path = Path("./artifacts")
    # # Iterate over each run
    # for i, run in tqdm(enumerate(list(runs))):
    #     # Access the list of artifacts for the run
    #     artifacts = run.logged_artifacts()
    #     print(f"Run {i}")
    #     df_run = None
    #     for artifact in artifacts:
    #         if metric_name in artifact.name:
    #             # Construct the unique path for this run's artifact
    #             artifact_file_path = download_path / f"{artifact.name}.json"
    #             # Check if the file already exists
    #             if os.path.exists(artifact_file_path):
    #                 print(f"Artifact already downloaded: {artifact_file_path}")
    #             else:
    #                 print(f"Downloading artifact to : {artifact_file_path}")
    #                 # Download the artifact
    #                 table_dir = artifact.download()
    #                 table_files = list(Path(table_dir).rglob('*test.table.json'))
    #                 if len(table_files) == 0:
    #                     print(f"Run {run.id} {run.name} did not save the state_pred_loss_t table")
    #                     continue
    #                 table_path = table_files[0]
    #                 # Move the file to the specified base directory
    #                 os.rename(table_path, artifact_file_path)
    #
    #             with artifact_file_path.open('r') as file:
    #                 json_dict = json.load(file)
    #             df_run = pd.DataFrame(json_dict["data"], columns=json_dict["columns"])
    #             break
    #     if df_run is not None:
    #         # Search in this run config values for model.name and system.group values and append to df
    #         config = run.config
    #         model_name = config["model"]["name"]
    #         system_group = config["system"]["group"]
    #         df_run = df_run.assign(**{"model.name": model_name, "system.group": system_group, "Name": run.name})
    #
    #         df = pd.concat([df, df_run], axis=0)
    #     else:
    #         print(f"Run {run.id} {run.name} did not save the state_pred_loss_t table")
    #
    # # Ignore all records of group = C2
    # df = df[df["system.group"] != "C2"]
    # fig = plot_aggregated_lines(df,
    #                             x="time",
    #                             y="state_pred_loss_t/test",
    #                             group_variables=['model.name', 'system.group'],
    #                             line_styles=line_styles,
    #                             color_group=color_group,
    #                             area_metric="std",
    #                             label_replace={"model.name":   "Model",
    #                                            "system.group": "Group",
    #                                            "K4":           r"$\mathbb{G}=\mathbb{K}_4$",
    #                                            "K4xC2":        r"$\mathbb{G}=\mathbb{K}_4 \times \mathbb{"
    #                                                            r"C}_2$", }
    #                             )
    # # Set the figure size to a quarter of an A4 page
    # fig.update_layout(**layout_config)
    # fig.update_xaxes(title_text="Prediction horizon [s]")
    # fig.update_yaxes(title_text="state prediction MSE",
    #                  type="log",
    #                  )
    # # fig.show()
    # data_path = Path("mini_cheetah_mse_vs_time.csv")
    # fig.write_html(data_path.with_suffix(".html"))
    # fig.write_image(data_path.with_suffix(".svg"))
    # fig.write_image(data_path.with_suffix(".png"))
    # # fig.show()
    # ==========================================================================
    # ========================LINEAR EXPERIMENT ================================
    # ==========================================================================
    # ==========================================================================
    # Linear sample efficiency =================================================
    data_path = Path("linear_C10_C5_sample_eff.csv")
    print(data_path)
    df = pd.read_csv(data_path)
    LINEAR_NUM_SAMPLES = 23600
    df['system.train_ratio'] = df['system.train_ratio'] * LINEAR_NUM_SAMPLES

    color_pallet = px.colors.qualitative.Prism

    line_styles = {'model.name': {"E-DAE": dict(),
                                  "DAE":   dict(dash='dash')}
                   }
    color_group = {'system.group': {"C10": color_pallet[0],
                                    "C5":  color_pallet[10],
                                    }
                   }

    fig = plot_aggregated_lines(df,
                                x="system.train_ratio",
                                y="state_pred_loss/test",
                                group_variables=['model.name', 'system.group'],
                                line_styles=line_styles,
                                color_group=color_group,
                                area_metric="min-max",
                                line_width=3,
                                label_replace={"model.name":   "Model",
                                               "system.group": "Group",
                                               "C5":           r"$\large \mathbb{G}=\mathbb{C}_5$",
                                               "C10":          r"$\large \mathbb{G}=\mathbb{C}_{10}$",
                                               }

                                )
    fig.update_layout(**layout_config)
    fig.update_xaxes(title_text="training samples")
    fig.update_yaxes(title_text="state prediction MSE")

    fig.write_html(data_path.with_suffix(".html"))
    fig.write_image(data_path.with_suffix(".svg"))
    fig.write_image(data_path.with_suffix(".png"))

    # Linear obs state dimension =================================================
    data_path = Path("linear_C10_C5_obs_state_dim.csv")
    print(data_path)
    df = pd.read_csv(data_path)
    STATE_DIM = 30
    df['system.obs_state_ratio'] = df['system.obs_state_ratio'] * STATE_DIM

    fig = plot_aggregated_lines(df,
                                x="system.obs_state_ratio",
                                y="state_pred_loss/test",
                                group_variables=['model.name', 'system.group'],
                                line_styles=line_styles,
                                color_group=color_group,
                                area_metric="min-max",
                                line_width=3,
                                label_replace={"model.name":   "Model",
                                               "system.group": "Group",
                                               "C5":           r"$\large \mathbb{G}=\mathbb{C}_5$",
                                               "C10":          r"$\large \mathbb{G}=\mathbb{C}_{10}$",
                                               }

                                )
    fig.update_layout(**layout_config, showlegend=False)
    fig.update_xaxes(title_text=r"$|\mathcal{X}|$ model state dimension")
    fig.update_yaxes(title_text="state prediction MSE", type="linear")

    fig.write_html(data_path.with_suffix(".html"))
    fig.write_image(data_path.with_suffix(".svg"))
    fig.write_image(data_path.with_suffix(".png"))

    # Linear obs state dimension =================================================
    data_path = Path("linear_C10_C5_state_dim.csv")
    print(data_path)
    df = pd.read_csv(data_path)

    fig = plot_aggregated_lines(df,
                                x="system.state_dim",
                                y="state_pred_loss/test",
                                group_variables=['model.name', 'system.group'],
                                line_styles=line_styles,
                                color_group=color_group,
                                area_metric="min-max",
                                line_width=3,
                                label_replace={"model.name":   "Model",
                                               "system.group": "Group",
                                               "C5":           r"$\large \mathbb{G}=\mathbb{C}_5$",
                                               "C10":          r"$\large \mathbb{G}=\mathbb{C}_{10}$",
                                               }

                                )
    fig.update_layout(**layout_config, showlegend=False)
    fig.update_xaxes(title_text="state dimension")
    fig.update_yaxes(title_text="state prediction MSE", type="linear")

    fig.write_html(data_path.with_suffix(".html"))
    fig.write_image(data_path.with_suffix(".svg"))
    fig.write_image(data_path.with_suffix(".png"))

    # Linear NOISE =================================================
    data_path = Path("linear_C10_C5_noise.csv")
    print(data_path)
    df = pd.read_csv(data_path)
    BASE_SIGMA = 0.1
    df['system.noise_level'] = df['system.noise_level'] * BASE_SIGMA
    fig = plot_aggregated_lines(df,
                                x="system.noise_level",
                                y="state_pred_loss/test",
                                group_variables=['model.name', 'system.group'],
                                line_styles=line_styles,
                                color_group=color_group,
                                area_metric="min-max",
                                line_width=3,
                                label_replace={"model.name":   "Model",
                                               "system.group": "Group",
                                               "C5":           r"$\large \mathbb{G}=\mathbb{C}_5$",
                                               "C10":          r"$\large \mathbb{G}=\mathbb{C}_{10}$",
                                               }

                                )
    fig.update_layout(**layout_config, showlegend=False)
    fig.update_xaxes(title_text=r"Noise variance")
    fig.update_yaxes(title_text="state prediction MSE", type="linear")

    fig.write_html(data_path.with_suffix(".html"))
    fig.write_image(data_path.with_suffix(".svg"))
    fig.write_image(data_path.with_suffix(".png"))

    # MSE vs Time =================================================
    data_path = Path("linear_mse_vs_time.csv")
    print(data_path)
    df = pd.read_csv(data_path)
    # MSE vs time needs a bit of reformat
    df_reformatted = pd.DataFrame()
    for col in df.columns:
        if "time" in col or "step" in col: continue
        run_name, var_name = col.split(" - ")
        print(var_name)
        # var_name = var_name.split("__")[0]
        if "_DAE" in run_name: print(run_name)
        if not var_name == "state_pred_loss_t/test": continue
        model_name = "E-DAE" if "E-DAE" in run_name else "DAE"
        system_group = "C5" if "C5" in run_name else "C10"
        df_run = pd.DataFrame({"Name": run_name, "model.name": model_name, "system.group": system_group,
                               "time":     df["time"], var_name: df[col]})
        df_run = df_run[np.logical_not(np.isnan(df_run[var_name]))]
        df_reformatted = pd.concat([df_reformatted, df_run], axis=0)

    fig = plot_aggregated_lines(df_reformatted,
                                x="time",
                                y="state_pred_loss_t/test",
                                group_variables=['model.name', 'system.group'],
                                line_styles=line_styles,
                                color_group=color_group,
                                area_metric="std",
                                label_replace={"model.name":   "Model",
                                               "system.group": "Group",
                                               "C2":           r"$\mathbb{G}=\mathbb{C}_2$",
                                               "K4":           r"$\mathbb{G}=\mathbb{K}_4$",
                                               "K4xC2":        r"$\mathbb{G}=\mathbb{K}_4 \times \mathbb{"
                                                               r"C}_2$", }
                                )

    # Set the figure size to a quarter of an A4 page
    fig.update_layout(**layout_config)
    fig.update_xaxes(title_text="Prediction horizon [s]")
    fig.update_yaxes(title_text="state prediction MSE",
                     type="linear",
                     )
    # fig.show()
    fig.write_html(data_path.with_suffix(".html"))
    fig.write_image(data_path.with_suffix(".svg"))
    fig.write_image(data_path.with_suffix(".png"))
    # fig.show()