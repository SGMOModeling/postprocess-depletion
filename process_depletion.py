#!/usr/bin/env python
# coding: utf-8

import os
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pywfm import IWFMModel
from pywfm import IWFMBudget

# Matplotlib 3.5.1 - https://github.com/matplotlib/matplotlib/issues/21950
# Resolves: Fail to create pixmap with Tk_GetPixmap in TkImgPhotoInstanceSetSize
matplotlib.use('Agg')

def get_stream_budget(model_path, results_dir, stream_budget_file):
    """
    Retrieve surface water groundwater interaction for all stream nodes
    """
    with IWFMBudget(
        os.path.join(model_path, results_dir, stream_budget_file)
    ) as sr_bud:
        dfs_list = []
        for sr in range(1, sr_bud.get_n_locations() + 1):
            df = sr_bud.get_values_for_a_column(sr, "Gain from GW_Inside Model (+)")
            df["ReachID"] = sr
            dfs_list.append(df[["ReachID", "Time", "Gain from GW_Inside Model (+)"]])

    return pd.concat(dfs_list)


def get_streamnode_budget(model_path, results_dir, streamnode_budget_file):
    """
    Retrieve surface water groundwater interaction for all stream nodes
    """
    with IWFMBudget(
        os.path.join(model_path, results_dir, streamnode_budget_file)
    ) as sn_bud:
        dfs_list = []
        for sn in range(1, sn_bud.get_n_locations() + 1):
            df = sn_bud.get_values_for_a_column(sn, "Gain from GW_Inside Model (+)")
            df["IRV"] = sn
            dfs_list.append(df[["IRV", "Time", "Gain from GW_Inside Model (+)"]])

    return pd.concat(dfs_list)


def get_interconnected_surface_water(model_path, pp_dir, pp_file, sim_dir, sim_file):
    with IWFMModel(
        os.path.join(model_path, pp_dir, pp_file),
        os.path.join(model_path, sim_dir, sim_file),
    ) as model:
        stream_network = model.get_stream_network()
        nodes = model.get_node_info()
        streambed = model.get_stream_bottom_elevations()

        stream_geometry = pd.merge(
            stream_network, nodes, left_on="GroundwaterNodes", right_on="NodeID"
        )
        stream_geometry["BedElev"] = streambed

        dates, heads = model.get_gwheads_foralayer(1)
        gw_levels = (
            pd.DataFrame(
                data=heads, index=dates, columns=np.arange(1, heads.shape[-1] + 1)
            )
            .stack()
            .reset_index()
        )
        gw_levels.rename(
            columns={"level_0": "Date", "level_1": "NodeID", 0: "GWL"}, inplace=True
        )

        swgw = pd.merge(stream_geometry, gw_levels, on="NodeID")
        swgw["ISW"] = np.where(
            swgw["GWL"].to_numpy() >= swgw["BedElev"].to_numpy(), 1, 0
        )

    return swgw


def convert_coordinates(x):
    return float(x) * fact_xy


def convert_radius(r):
    return float(r) * fact_rw


def convert_depths(d):
    return float(d) * fact_lt


def convert_date(dt):
    return dt.split("_")[0]


if __name__ == "__main__":

    basemodel_path = "C2VSimFG_V1_05a"
    scenario_dir = "1980_V1_05a"
    output_path = "PostProc_{}".format(scenario_dir)
    scenario_paths = [os.path.join(scenario_dir, d) for d in os.listdir(scenario_dir) if "VS" in d]

    pp_dir = "Preprocessor"
    sim_dir = "Simulation"
    results_dir = "Results"
    
    pp_file = "C2VSimFG_Preprocessor.in"
    sim_file = "C2VSimFG.in"
    
    sn_bud_file = "C2VSimFG_Stream_NodeBudget.hdf"
    sr_bud_file = "C2VSimFG_Stream_Budget.hdf"
    gw_bud_file = "C2VSimFG_GW_Budget.hdf"

    for scenario_path in scenario_paths:
        print('Plotting {}'.format(os.path.basename(scenario_path)))

        if not os.path.exists(output_path):
            os.mkdir(output_path)
            os.mkdir(os.path.join(output_path, os.path.basename(scenario_path)))
            os.mkdir(
                os.path.join(
                    output_path, os.path.basename(scenario_path), "DepletionMaps"
                )
            )
            os.mkdir(os.path.join(output_path, os.path.basename(scenario_path), "ISW"))

        elif not os.path.exists(
            os.path.join(output_path, os.path.basename(scenario_path))
        ):
            os.mkdir(os.path.join(output_path, os.path.basename(scenario_path)))
            os.mkdir(
                os.path.join(
                    output_path, os.path.basename(scenario_path), "DepletionMaps"
                )
            )
            os.mkdir(os.path.join(output_path, os.path.basename(scenario_path), "ISW"))

        elif not os.path.exists(
            os.path.join(output_path, os.path.basename(scenario_path), "DepletionMaps")
        ):
            os.mkdir(
                os.path.join(
                    output_path, os.path.basename(scenario_path), "DepletionMaps"
                )
            )

        elif not os.path.exists(
            os.path.join(output_path, os.path.basename(scenario_path), "ISW")
        ):
            os.mkdir(os.path.join(output_path, os.path.basename(scenario_path), "ISW"))

        # Calculate Depletion at each Stream Node from the Stream Node Budget

        base_swgw_sn = get_streamnode_budget(basemodel_path, results_dir, sn_bud_file)
        base_swgw_sr = get_stream_budget(basemodel_path, results_dir, sr_bud_file)

        scenario_swgw_sn = get_streamnode_budget(
            scenario_path, results_dir, sn_bud_file
        )
        scenario_swgw_sr = get_stream_budget(scenario_path, results_dir, sr_bud_file)

        depletion_sr = pd.merge(
            scenario_swgw_sr,
            base_swgw_sr,
            on=["ReachID", "Time"],
            suffixes=("_scenario", "_base"),
        )
        depletion_sn = pd.merge(
            scenario_swgw_sn,
            base_swgw_sn,
            on=["IRV", "Time"],
            suffixes=("_scenario", "_base"),
        )

        depletion_sr["Depletion"] = (
            depletion_sr["Gain from GW_Inside Model (+)_scenario"]
            - depletion_sr["Gain from GW_Inside Model (+)_base"]
        )
        depletion_sn["Depletion"] = (
            depletion_sn["Gain from GW_Inside Model (+)_scenario"]
            - depletion_sn["Gain from GW_Inside Model (+)_base"]
        )

        cumulative_depletion_sr = depletion_sr.groupby("ReachID")["Depletion"].sum()
        cumulative_depletion_sn = depletion_sn.groupby("IRV")["Depletion"].sum()

        total_depletion_sr = depletion_sr.groupby("Time")["Depletion"].sum().to_frame()
        total_depletion_sn = depletion_sn.groupby("Time")["Depletion"].sum().to_frame()

        total_depletion_sr["CumulativeDepletion"] = total_depletion_sr[
            "Depletion"
        ].cumsum()
        total_depletion_sn["CumulativeDepletion"] = total_depletion_sn[
            "Depletion"
        ].cumsum()

        # Calculate Total Depletion from the Groundwater Budget

        with IWFMBudget(
            os.path.join(basemodel_path, results_dir, gw_bud_file)
        ) as gw_bud:
            base_gw_df = gw_bud.get_values(22)

        with IWFMBudget(
            os.path.join(scenario_path, results_dir, gw_bud_file)
        ) as gw_bud:
            scenario_gw_df = gw_bud.get_values(22)

        gw_df = pd.merge(
            scenario_gw_df[["Time", "Gain from Stream (+)"]],
            base_gw_df[["Time", "Gain from Stream (+)"]],
            on="Time",
            suffixes=("_scenario", "_base"),
        )

        gw_df["Depletion"] = (
            gw_df["Gain from Stream (+)_scenario"] - gw_df["Gain from Stream (+)_base"]
        )

        # Plot Cumulative Depletion during the Simulation
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(11, 12))
        ax[0].plot(
            total_depletion_sr.index,
            total_depletion_sr["Depletion"].cumsum(),
            label="Stream Reaches",
        )
        ax[0].plot(gw_df["Time"], gw_df["Depletion"].cumsum(), label="Groundwater")
        ax[0].legend()
        ax[1].plot(
            total_depletion_sr.index,
            -1 * total_depletion_sr["Depletion"].cumsum(),
            label="Stream Reaches",
        )
        ax[1].plot(gw_df["Time"], gw_df["Depletion"].cumsum(), label="Groundwater")
        ax[1].legend()
        plt.suptitle("Global Check of Depletion between Stream Reaches and Groundwater")
        ax[0].set_title("Visual Check of Symmetry")
        ax[1].set_title("Visual Check of Equality")
        plt.savefig(
            os.path.join(
                output_path,
                os.path.basename(scenario_path),
                '{}_check.png'.format(os.path.basename(scenario_path))
            )
        )
        plt.close()

        # Read Well Locations and Time Series Information
        #
        # Read location information only for wells in transfer project
        # Calculate total pumping for transfer project during 6 month period
        # Rescale depletion as a percent of pumping
        # Plot Pumping and incremental depletion

        with open(
            os.path.join(scenario_path, sim_dir, "Groundwater/C2VSimFG_WellSpec.dat"),
            "r",
        ) as f:
            count = 0
            while True:
                line = f.readline()
                if line[0] not in ["C", "c", "*"]:
                    if count == 0:
                        n_well = int(line.split("/")[0].strip())
                    elif count == 1:
                        fact_xy = float(line.split("/")[0].strip())
                    elif count == 2:
                        fact_rw = float(line.split("/")[0].strip())
                    elif count == 3:
                        fact_lt = float(line.split("/")[0].strip())
                    count += 1

                if count == 4:
                    break

        ws_col = "ID       XWELL     YWELL       RWELL      PERFT      PERFB"
        ws = pd.read_csv(
            os.path.join(scenario_path, sim_dir, "Groundwater/C2VSimFG_WellSpec.dat"),
            header=None,
            names=ws_col.split(),
            skiprows=94,
            converters={
                "XWELL": convert_coordinates,
                "YWELL": convert_coordinates,
                "RWELL": convert_radius,
                "PERFT": convert_depths,
                "PERFB": convert_depths,
            },
            nrows=n_well,
            comment="/",
            delim_whitespace=True,
        )

        wc_col = "ID      ICOLWL   FRACWL    IOPTWL   TYPDSTWL    DSTWL   ICFIRIGWL   ICADJWL  ICWLMAX   FWLMAX"
        wc = pd.read_csv(
            os.path.join(scenario_path, sim_dir, "Groundwater/C2VSimFG_WellSpec.dat"),
            header=None,
            names=wc_col.split(),
            skiprows=94 + n_well + 49,
            nrows=n_well,
            comment="/",
            delim_whitespace=True,
        )

        well_spec = pd.merge(ws, wc, on="ID")

        well_ts_cols = well_spec[well_spec["ID"] > 610]["ICOLWL"].tolist()

        pump_rates = pd.read_csv(
            os.path.join(scenario_path, sim_dir, "Groundwater/C2VSimFG_PumpRates.dat"),
            header=None,
            skiprows=5,
            converters={0: convert_date},
            parse_dates={"Date": [0]},
            delim_whitespace=True,
        )

        test_pumping = pump_rates[["Date"] + well_ts_cols].copy()
        test_pumping["total"] = test_pumping[well_ts_cols].sum(axis=1)

        total_volume = test_pumping["total"].sum()

        # Plot pumping by well and total pumping
        fig, ax = plt.subplots(figsize=(11, 6))
        for col in well_ts_cols:
            ax.plot(pump_rates["Date"], pump_rates[col])
        ax.plot(test_pumping["Date"], test_pumping["total"])
        plt.savefig(
            os.path.join(
                output_path,
                os.path.basename(scenario_path),
                '{}_pumping.png'.format(os.path.basename(scenario_path))
            )
        )
        plt.close()

        # Get Stream Network Information and Groundwater Levels for Model Layer 1
        base_swgw = get_interconnected_surface_water(
            basemodel_path, pp_dir, pp_file, sim_dir, sim_file
        )
        scenario_swgw = get_interconnected_surface_water(
            scenario_path, pp_dir, pp_file, sim_dir, sim_file
        )

        sim_dates = base_swgw["Date"].dt.strftime("%Y-%m-%d").unique()

        # Plot maps of interconnected surface water
        for dt in sim_dates:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 11))
            ax[0].scatter(
                base_swgw[(base_swgw["Date"] == dt) & (base_swgw["ISW"] == 0)]["X"],
                base_swgw[(base_swgw["Date"] == dt) & (base_swgw["ISW"] == 0)]["Y"],
                s=2,
                c="red",
                label="disconnected",
            )
            ax[0].scatter(
                base_swgw[(base_swgw["Date"] == dt) & (base_swgw["ISW"] == 1)]["X"],
                base_swgw[(base_swgw["Date"] == dt) & (base_swgw["ISW"] == 1)]["Y"],
                s=2,
                c="green",
                label="connected",
            )
            ax[0].set_title("ISW for Base".format())
            ax[0].legend()
            ax[0].grid()
            ax[0].set_aspect("equal")

            ax[1].scatter(
                scenario_swgw[
                    (scenario_swgw["Date"] == dt) & (scenario_swgw["ISW"] == 0)
                ]["X"],
                scenario_swgw[
                    (scenario_swgw["Date"] == dt) & (scenario_swgw["ISW"] == 0)
                ]["Y"],
                s=2,
                c="red",
                label="disconnected",
            )
            ax[1].scatter(
                scenario_swgw[
                    (scenario_swgw["Date"] == dt) & (scenario_swgw["ISW"] == 1)
                ]["X"],
                scenario_swgw[
                    (scenario_swgw["Date"] == dt) & (scenario_swgw["ISW"] == 1)
                ]["Y"],
                s=2,
                c="green",
                label="connected",
            )
            ax[1].set_title("ISW for Scenario".format())
            ax[1].legend()
            ax[1].grid()
            ax[1].set_aspect("equal")

            plt.suptitle("{}".format(dt))
            plt.savefig(
                os.path.join(
                    output_path,
                    os.path.basename(scenario_path),
                    "ISW/{}_ISW_{}.png".format(os.path.basename(scenario_path), dt),
                )
            )
            plt.close()

        # Plot incremental and cumulative stream depletion by stream reach
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(11, 12))
        for rch in depletion_sr["ReachID"].unique():
            ax[0].plot(
                depletion_sr[depletion_sr["ReachID"] == rch]["Time"],
                -1 * depletion_sr[depletion_sr["ReachID"] == rch]["Depletion"] / 43560,
            )
            ax[1].plot(
                depletion_sr[depletion_sr["ReachID"] == rch]["Time"],
                -1
                * depletion_sr[depletion_sr["ReachID"] == rch]["Depletion"].cumsum()
                / 43560,
            )

        ax[0].set_ylabel("Stream Depletion (AF)")
        ax[1].set_ylabel("Cumulative Stream Depletion (AF)")
        ax[0].set_title("Stream Depletion for each Stream Reach")
        ax[1].set_title("Cumulative Stream Depletion for each Stream Reach")
        plt.savefig(
            os.path.join(
                output_path,
                os.path.basename(scenario_path),
                "{}_StreamDepletion.png".format(os.path.basename(scenario_path)),
            )
        )
        plt.close()

        cumulative_depletion_by_streamnode = pd.merge(
            base_swgw[base_swgw["Date"] == base_swgw["Date"].min()][
                ["StreamNodes", "X", "Y"]
            ],
            cumulative_depletion_sn,
            left_on="StreamNodes",
            right_on="IRV",
        )

        # Plot cumulative stream depletion map
        cutoff = 0.01
        fig, ax = plt.subplots(figsize=(8, 11))
        ax.scatter(
            cumulative_depletion_by_streamnode["X"],
            cumulative_depletion_by_streamnode["Y"],
            s=1,
            c="#D3D3D3",
        )
        sp = ax.scatter(
            cumulative_depletion_by_streamnode[
                cumulative_depletion_by_streamnode["Depletion"]
                < total_volume * 43560 * cutoff
            ]["X"],
            cumulative_depletion_by_streamnode[
                cumulative_depletion_by_streamnode["Depletion"]
                < total_volume * 43560 * cutoff
            ]["Y"],
            s=2,
            c=cumulative_depletion_by_streamnode[
                cumulative_depletion_by_streamnode["Depletion"]
                < total_volume * 43560 * cutoff
            ]["Depletion"]
            / 43560
            / total_volume
            * 100,
            cmap="viridis",
        )
        ax.scatter(
            well_spec[well_spec["ID"] > 610]["XWELL"],
            well_spec[well_spec["ID"] > 610]["YWELL"],
            s=5,
            c="red",
        )
        ax.set_aspect("equal")
        fig.colorbar(sp, ax=ax)
        plt.savefig(
            os.path.join(
                output_path,
                os.path.basename(scenario_path),
                "{}_CumulativeDepletion_ByStreamNode.png".format(
                    os.path.basename(scenario_path)
                ),
            )
        )
        plt.close()
        
        # Plot Depletion Maps for each time step
        threshold = 0.0001
        for dt in sim_dates:

            fig, ax = plt.subplots(figsize=(8, 11))

            temp = pd.merge(
                scenario_swgw[scenario_swgw["Date"] == dt][
                    ["Date", "StreamReach", "ReachName", "StreamNodes", "X", "Y"]
                ],
                depletion_sn[depletion_sn["Time"] == dt][["Time", "IRV", "Depletion"]],
                left_on="StreamNodes",
                right_on="IRV",
            )
            cum_depletion = total_depletion_sn[total_depletion_sn.index == dt][
                "CumulativeDepletion"
            ].to_numpy()[0]

            ax.scatter(temp["X"], temp["Y"], s=1, c="#D3D3D3")

            dpl = ax.scatter(
                temp[temp["Depletion"] < cum_depletion * threshold]["X"],
                temp[temp["Depletion"] < cum_depletion * threshold]["Y"],
                s=2,
                c=temp[temp["Depletion"] < cum_depletion * threshold]["Depletion"]
                / 43560
                / total_volume
                * 100,
            )

            ax.scatter(
                well_spec[well_spec["ID"] > 610]["XWELL"],
                well_spec[well_spec["ID"] > 610]["YWELL"],
                s=5,
                c="red",
            )

            inset_ax = fig.add_axes([0.5, 0.7, 0.2, 0.15])
            inset_ax.plot(
                total_depletion_sn.index,
                total_depletion_sn["CumulativeDepletion"] / 43560 / total_volume,
            )
            inset_ax.plot(
                total_depletion_sn[total_depletion_sn.index == dt].index,
                total_depletion_sn[total_depletion_sn.index == dt][
                    "CumulativeDepletion"
                ]
                / 43560
                / total_volume,
                "ro",
                markersize=10,
                mfc="None",
                ls="",
            )
            inset_ax.grid()

            labels = inset_ax.get_xticklabels()
            for label in labels:
                label.set(rotation=45, horizontalalignment="right")

            cbar = fig.colorbar(dpl, ax=ax)
            cbar.set_label("% Depletion by Stream Node")

            ax.grid()
            ax.set_aspect("equal")
            ax.set_title("Stream Depletion\nAs of: {}".format(dt))

            s = "{:<15s}{:>10.2f} AF\n{:<15s}{:>10.2f} AF".format(
                "Total Pumping:",
                total_volume,
                "Total Depletion:",
                cum_depletion / 43560,
            )

            ax.text(
                0.1,
                0.1,
                s,
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )

            plt.savefig(
                os.path.join(
                    output_path,
                    os.path.basename(scenario_path),
                    "DepletionMaps/{}_Depletion_{}.png".format(
                        os.path.basename(scenario_path), dt
                    ),
                )
            )
            plt.close()

        # Plot comparison between pumping hydrograph and depletion hydrograph
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(11, 12))
        ax[0].plot(
            total_depletion_sr.index,
            -1 * total_depletion_sr["Depletion"] / 43560,
            label="Depletion",
        )
        ax[0].plot(test_pumping["Date"], -1 * test_pumping["total"], label="Pumping")
        ax[1].plot(
            total_depletion_sr.index,
            -1 * total_depletion_sr["Depletion"].cumsum() / 43560,
            label="Depletion",
        )
        ax[1].plot(
            test_pumping["Date"], -1 * test_pumping["total"].cumsum(), label="Pumping"
        )
        ax[0].grid()
        ax[1].grid()
        ax[0].legend()
        ax[1].legend()
        ax[0].set_ylabel("Volume (AF)")
        ax[1].set_ylabel("Cumulative Volume (AF)")
        ax[0].set_title("Comparison of Pumping and Stream Depletion")
        ax[0].set_title(
            "Comparison of Cumulative Pumping and Cumulative Stream Depletion"
        )
        plt.savefig(
            os.path.join(
                output_path,
                os.path.basename(scenario_path),
                "{}_Pumping_SFD_Response.png".format(os.path.basename(scenario_path)),
            )
        )
        plt.close()

        ranked_depletion = (
            depletion_sr.groupby("ReachID")["Depletion"]
            .sum()
            .sort_values()
            .reset_index()
        )
        ranked_depletion["PercentDepletion"] = (
            ranked_depletion["Depletion"] / 43560 / total_volume * 100
        )

        percent_depletion = pd.merge(
            scenario_swgw[scenario_swgw["Date"] == "2015-09-30"][
                ["StreamReach", "StreamNodes", "ReachName"]
            ],
            depletion_sn[["Time", "IRV", "Depletion"]],
            left_on="StreamNodes",
            right_on="IRV",
        )

        percent_depletion_by_sr = (
            percent_depletion.groupby(["StreamReach", "Time"])["Depletion"]
            .sum()
            .groupby(level=0)
            .cumsum()
            .reset_index()
        )

        percent_depletion_by_sr["PercentSFD"] = (
            percent_depletion_by_sr["Depletion"] / 43560 / total_volume * 100
        )

        # Plot Cumulative Stream Depletion Curve for each stream reach
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.stackplot(
            percent_depletion_by_sr.pivot(
                index="Time", columns="StreamReach", values="PercentSFD"
            ).index,
            percent_depletion_by_sr.pivot(
                index="Time", columns="StreamReach", values="PercentSFD"
            )[np.flip(ranked_depletion["ReachID"].to_numpy())].values.T,
        )
        ax.grid()
        ax.set_ylabel("Stream Depletion (%)")
        ax.set_title("Percent of Stream Depletion by Stream Reach")
        plt.savefig(
            os.path.join(
                output_path,
                os.path.basename(scenario_path),
                "{}_CumulativePercentDepletionByReach.png".format(
                    os.path.basename(scenario_path)
                ),
            )
        )
        plt.close()