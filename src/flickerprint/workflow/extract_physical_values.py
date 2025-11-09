#!/usr/bin/env python

""" Generate and store the best fit values and metadata for an experiment.

Quick Start
-----------

Once the Fourier terms have been calculated by ``process_images.py`` this script fits
the data to the experimental spectrum and creates a ``fitting_vals.csv`` table for the
experiment directory. After this use ``create_plots.py`` to merge the results from
several experiments and create summaries of the data.

Outline
-------

For each Fourier terms file we fit every granule to a given theoretical spectrum. From
this we calculate the best estimates for the physical values, typically the surface
tension and bending rigidity.

This is saved in a summary table, with other metadata about the granules, this
includes properties of the granule such as its size or the time since treatment,
    input_path = Path(frame_info["input_path"])
but also abstract parameters, such as the quality of fitting to the expected
spectrum.

Once the summary table is created, we add a number of secondary columns for
convenience, typically physical values re-scaled by some factor or on a log
scale. We also bin the time values into 5 minute intervals that can be used for
aggregation/averaging.

"""

import logging
from pathlib import Path

import argh
import subprocess
import platform
import h5py
import os
import warnings
import numpy as np
import pandas as pd
import pickle as pkl
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann as kB
from tqdm import tqdm
from functools import partial
from time import sleep
from matplotlib.ticker import MaxNLocator

from flickerprint.common.utilities import strtobool
import flickerprint.fluctuation.spectra as sf
import flickerprint.version as version
import flickerprint.tools.plot_tools as pt
from flickerprint.common.configuration import config


def process_fourier_file(fourier_path: Path, output: Path, plotting: bool=False, _pbar_pos: int = 0):
    """
    Perform the spectrum fitting on one .h5 file corresponding to one time series image.
    ====================================================================================

    Returns:
      - ``property_df``: one line per granule including sigma/kappa estimates
      - ``magnitude_df``: one line per order per granule, contains the fluctuation/fixed spectrum
    """

    print(f"#{_pbar_pos+1} Working on file: {fourier_path}")
    fourier_terms, frame_info = load_fourier_terms(fourier_path)

    pixel_size = frame_info["pixel_size"]

    temperature = float(config("spectrum_fitting", "temperature")) + 273.15
    max_order = int(config("spectrum_fitting", "fitting_orders"))
    spectrum_type = str(config("spectrum_fitting", "experimental_spectrum"))

    # Take ownership of the filtered array to stop warnings
    fourier_terms = fourier_terms.query(f"order <= {max_order}").copy()
    fourier_terms["mag_abs"] = np.abs(fourier_terms["magnitude"])
    fourier_terms["mag_squared"] = fourier_terms["mag_abs"] ** 2

    grouped_by_granule = fourier_terms.groupby("granule_id")

    spectrum_builder = sf.SpectrumFitterBuilder(q_max=max_order, l_max=75)
    ST_only_builder = sf.SpectrumFitterBuilder_ST_Only(q_max=max_order, l_max=75)

    property_df = []
    magnitude_df = []

    for granule_id, granule in tqdm(grouped_by_granule, position=_pbar_pos, unit="condensates", desc=f"#{_pbar_pos+1}"):
        metadata = gather_granule_metadata(granule)

        # Create a DF of the time averaged terms
        # This is the experimental spectrum that we compare against
        # We have to do the latter term using λ as otherwise it uses a cython version
        # without complex support.
        mag_df = granule.groupby(by="order").agg(
            mag_squ_mean=("mag_squared", "mean"),
            mag_mean=("magnitude", lambda x: np.mean(x)),
        )
        mag_df.reset_index(inplace=True)

        # Supplementary columns used in Pécréaux 2004
        # These terms differ from the definition in the paper as we take
        # |〈mag〉|**2 rather than 〈|mag|〉**2
        mag_df["fixed_squ"] = np.abs(mag_df["mag_mean"]) ** 2
        mag_df["fluct_squ"] = mag_df["mag_squ_mean"] - mag_df["fixed_squ"]
        if spectrum_type == 'direct':
            mag_df['experiment_spectrum'] = mag_df["mag_squ_mean"]
        elif spectrum_type == 'corrected':
            mag_df["experiment_spectrum"] = mag_df["fluct_squ"]
        else:
            raise ValueError(f"Invalid spectrum type: {spectrum_type}. Choose either 'direct' or 'corrected'.")
        experimental_spectrum = mag_df["experiment_spectrum"].values

        spectrum_total = (experimental_spectrum**2).sum()
        if spectrum_total < 1e-20:
            logging.debug(f"Skipping spectrum as all values zero: {spectrum_total}")
            continue

        mag_df["granule_id"] = granule_id
        mag_df["figure_path"] = frame_info["input_path"]
        magnitude_df.append(mag_df)

        # try:
        residuals, minimisation_function = spectrum_builder.create_fitting_function(experimental_spectrum) # We need to do this one separately as it is needed for plotting.
        fitting_result = spectrum_builder.minimiser(residuals, minimisation_function)
        if fitting_result is None:
            continue
        ST_only_fitting_result = ST_only_builder.minimiser(
            *ST_only_builder.create_fitting_function(experimental_spectrum))
        # except ValueError:
        #     continue

        mag_df['best_fit'] = spectrum_builder.get_spectra(
            fitting_result['sigma_bar'], fitting_result['kappa_scale']
        )
        durbin_watson = sf.calculate_durbin_watson(
            experimental_spectrum, mag_df['best_fit']
        )

        fitting_result["granule_id"] = granule_id
        fitting_result["durbin_watson"] = durbin_watson
        fitting_result["q_2_mag"] = mag_df["fixed_squ"][0]
        fitting_result["experiment"] = config("workflow", "experiment_name")
        fitting_result |= metadata
        fitting_result["image_path"] = str(granule["im_path"].iloc[0])

        # Caluclate whether the spectrum is above the pixel threshold
        pixel_threshold = (pixel_size/15)**2/fitting_result["mean_radius"]**2
        fitting_result["above_res_threshold"] = (mag_df["experiment_spectrum"] > pixel_threshold).sum() > (len(mag_df["experiment_spectrum"]) / 2)
        fitting_result["pass_count"] = granule[granule['order'] == 2]['valid'].sum()
        fitting_result["pass_rate"] = fitting_result["pass_count"] / len(granule[granule['order'] == 2])


        fitting_result["sigma"] = (
            fitting_result["sigma_bar"]
            / (fitting_result["mean_radius"]*1e-6) ** 2
            * fitting_result["kappa_scale"]
            * kB
            * temperature
        )
        fitting_result["sigma_st"] = ST_only_fitting_result["sigma_ST_bar"] / (fitting_result["mean_radius"]*1e-6) ** 2 * kB * temperature

        fitting_result["sigma_err"] = fitting_result['sigma'] * np.sqrt((fitting_result['sigma_bar_err']/fitting_result['sigma_bar'])**2 + (fitting_result['kappa_scale_err']/fitting_result['kappa_scale'])**2)
        fitting_result["sigma_err_st"] = ST_only_fitting_result['sigma_ST_bar_err']/ST_only_fitting_result['sigma_ST_bar'] * fitting_result['sigma_st']
        fitting_result["fitting_diff"] = ST_only_fitting_result["fitting_error_ST"] - fitting_result["fitting_error"]

        save_name = Path(frame_info["input_path"]).stem + f"--G{granule_id:02d}.png"
        fitting_result['figure_path'] = save_name
        if plotting :
            
            n_kappa, n_sigma = 100, 100
            sigma_mid, kappa_mid  = 10e1,10e-1
            width = 1000000.0
            line_func = np.linspace if width <= 5 else np.geomspace
            sigma_bars = line_func(sigma_mid / width, sigma_mid * width, num=n_sigma)
            kappa_scales = line_func(kappa_mid / width, kappa_mid * width, num=n_kappa)
            plot_spectrum(
                granule_mag_df=mag_df,
                granule_fit_df=fitting_result,
                resolution_threshold=pixel_threshold,
                ax=None,
                save_path=output / Path(f"fitting/spectra")/save_name,
            )
            plot_heatmap(save_path=output / Path(f"fitting/heatmaps")/save_name,
                            mag_df=mag_df,
                            error_function=minimisation_function,
                            sigma_bars=sigma_bars,
                            kappa_scales=kappa_scales,
                            mean_radius=fitting_result['mean_radius'] * 1e-6,
                            temperature=temperature)

        property_df.append(fitting_result)


    property_df = pd.DataFrame(property_df)
    if property_df.empty:
        raise ValueError(f"No valid granules found in {fourier_path}")
    property_df.drop(columns=["sigma_bar", "sigma_bar_err"], inplace=True)
    # Just reorder the columns so that they're the same as the documentation
    property_df = property_df.loc[:, ["granule_id", 
                                      "sigma", 
                                      "sigma_err", 
                                      "kappa_scale", 
                                      "kappa_scale_err", 
                                      "mean_radius", 
                                      "figure_path", 
                                      "image_path",
                                      "pass_rate",
                                      "pass_count",
                                      "fitting_error",
                                      "durbin_watson",
                                      "mean_intensity",
                                      "x",
                                      "y",
                                      "bbox_left",
                                      "bbox_bottom",
                                      "bbox_right",
                                      "bbox_top",
                                      "q_2_mag",
                                      "experiment",
                                      "timestamp",
                                      "fitting_diff",
                                      "sigma_st",
                                      "sigma_err_st",
                                      "above_res_threshold"
                              ]]
    magnitude_df = pd.concat(magnitude_df, ignore_index=True)
    return property_df, magnitude_df


def gather_granule_metadata(granule_df: pd.DataFrame) -> dict:
    props = {}

    # Properties averaged across all frames
    props["mean_radius"] = granule_df["mean_radius"].mean()
    props["mean_intensity"] = granule_df["mean_intensity"].mean()

    # (Mostly) Unchanging parameters where we only need to consider the first frame
    first_frame = granule_df.iloc[0]
    keyword_list = [
        "x",
        "y",
        "timestamp",
        "bbox_left",
        "bbox_bottom",
        "bbox_right",
        "bbox_top",
    ]
    for keyword_ in keyword_list:
        props[keyword_] = first_frame[keyword_]

    return props


def main(working_dir: Path, plotting=False, cores=1):
    """ Merge multiple Fourier terms into a single file. """
    print(f"\n================\nSpectrum Fitting\n================\n")
    working_dir = Path(working_dir)
    config.refresh(working_dir / "config.yaml")
    print(f"\nConfiguration file location: {working_dir}/config.yaml")
    input_paths = list(working_dir.glob("fourier/*.h5")) + list(working_dir.glob("fourier/*.pkl"))# This lets us search for either .h5 or .pkl files.
    print(f"Current working directory: {working_dir}")
    if input_paths == []:
            raise FileNotFoundError(f"\nNo images found in {working_dir}/fourier.\nCheck that you are in the correct directory.")
    
    plotting = bool(strtobool(config("spectrum_fitting", "plot_spectra_and_heatmaps")))

    if cores > os.cpu_count():
        cores = os.cpu_count()
        warnings.warn(f"Number of cores requested exceeds available cores. Only {os.cpu_count()} cores are available.", UserWarning)
    if cores > len(input_paths):
        cores = len(input_paths)
    if cores == 1:
        print(f"Using 1 core")
    else:
        print(f"Using {cores} cores")
    
    print(f"----------\n")

    if len(input_paths) > 1:
        with mp.Pool(processes=cores, maxtasksperchild=1) as pool:
            args = []
            for pbar_pos, file in enumerate(input_paths):
                args.append((Path(file), Path(working_dir), plotting, pbar_pos))

            frame_info = pool.starmap(process_fourier_file, args)
    else:
        if cores != 1:
            print("Using 1 core as only a single image to be analysed.")
        print(f"\n")
        frame_info = [process_fourier_file(input_paths[0], working_dir, plotting, 0)]

    aggregate_data, fourier_terms = zip(*frame_info)
    aggregate_data = pd.concat(aggregate_data, ignore_index=True,)
    fourier_terms = pd.concat(fourier_terms, ignore_index=True,)
    if str(config("workflow", "experiment_name")) != "experiment_name":
        save_path = working_dir / f"aggregate_fittings.h5"
    else:
        save_path = working_dir / "aggregate_fittings.h5"
    _write_hdf(save_path, aggregate_data, fourier_terms)
    sleep(2)
    print("\n\n")
    if bool(strtobool(config("spectrum_fitting", "plot_spectra_and_heatmaps"))):
        try:
            heatmaps_return = subprocess.call(f"zip -r heatmaps.zip heatmaps", shell =True, cwd=f"{working_dir}/fitting", stdout=subprocess.DEVNULL)
            spectra_return = subprocess.call(f"zip -r spectra.zip spectra", shell =True, cwd=f"{working_dir}/fitting", stdout=subprocess.DEVNULL)
            if heatmaps_return == 0 and spectra_return == 0:
                subprocess.call(f"rm -rf spectra", shell =True, cwd=f"{working_dir}/fitting", stdout=subprocess.DEVNULL)
                subprocess.call(f"rm -rf heatmaps", shell =True, cwd=f"{working_dir}/fitting", stdout=subprocess.DEVNULL)
                subprocess.call(f"mkdir fitting/spectra fitting/heatmaps", shell=True)
            else:
                print("Zipping spectra and heatmaps images unsuccessful. Images will be available as separate files instead.")
            subprocess.call(f"cd {working_dir}", shell=True)
        except:
            subprocess.call(f"cd {working_dir}", shell=True)
            print("Zipping spectra and heatmaps images unsuccessful. Images will be available as separate files instead.")
    print(f"\nSpectrum fitting analysis complete\n----------------------------------\n")


def _write_hdf(
    save_path: Path, aggregate_data: pd.DataFrame, fourier_terms: pd.DataFrame
):
    """ Write the dataframe to HDF5 along with metadata. """
    if platform.system()=="Darwin" and "ARM64" in platform.version():
        # Doing it this way will ensure we still catch Apple Silicon Macs even when using Rosetta 2.
        # The 'else' case below should catch all other platforms where writing to hdf5 should work normally.
        try:
            aggregate_data.to_hdf(save_path, key="aggregate_data", mode="w")
            fourier_terms.to_hdf(save_path, key="fourier_terms", mode="a")
            print(f"\nAggregate fittings file location: aggregate_fittings.h5")

            with h5py.File(save_path, "a") as f:
                aggregate_hdf = f["aggregate_data"]
                config_yaml, _ = config._aggregate_all()
                aggregate_hdf.attrs['config'] = config_yaml
                aggregate_hdf.attrs['version'] = version.__version__
        except:
            config_yaml, config_summary = config._aggregate_all()
            with open(f'{str(save_path)[:-3]}.pkl', 'wb') as file:
                pkl.dump({'fourier_terms': fourier_terms, "aggregate_data": aggregate_data, "configuration": config_yaml, "version": version.__version__}, file=file)
            print(f"\nAggregate fittings file location: aggregate_fittings.pkl")

    else:
        aggregate_data.to_hdf(save_path, key="aggregate_data", mode="w")
        fourier_terms.to_hdf(save_path, key="fourier_terms", mode="a")
        print(f"\nAggregate fittings file location: aggregate_fittings.h5")

        with h5py.File(save_path, "a") as f:
            aggregate_hdf = f["aggregate_data"]
            config_yaml, _ = config._aggregate_all()
            aggregate_hdf.attrs['config'] = config_yaml
            aggregate_hdf.attrs['version'] = version.__version__



def load_fourier_terms(fourier_path: Path) -> pd.DataFrame:
    """ Read the Fourier terms from file. """

    if fourier_path.name.endswith(".h5"):
        fourier_terms = pd.read_hdf(fourier_path, key="fourier", mode="r")

        with h5py.File(fourier_path, "r") as f:
            attrs = f["fourier"].attrs
            frame_info = dict(
                input_path=attrs["input_path"], pixel_size=attrs["pixel_size"]
            )
            config_old = attrs["config"]
            version_old = attrs["version"]
    elif fourier_path.name.endswith(".pkl"):
        file = open(f'{str(fourier_path)}', 'rb')
        f = pkl.load(file=file)
        fourier_terms = f['fourier']
        frame_info = dict(
                input_path=f['frame_data']["input_path"], pixel_size=f['frame_data']["pixel_size"]
            )
        config_old = f['config']
        version_old = f['version']
    else:
        raise IOError("We can only load data from HDF5 and pkl files currently.")

    config_current, _ = config._aggregate_all()
    if config_old != config_current:
        warnings.warn("Warning: Config file has changed since process-image was run")
    if version_old != version.__version__:
        warnings.warn("Warning: Version number has changed since process-image was run")

    return fourier_terms, frame_info

def plot_spectrum(granule_mag_df: dict,
                  granule_fit_df: dict, 
                  resolution_threshold: float = None, 
                  ax = None, 
                  save_path: str = '/tmp/spectrum_plot.png'
                  ):
    
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    standalone_plot = False
    if ax is None:
        fig, ax = pt.create_axes(1, axes_height= 3)
        standalone_plot = True

    plot_fmt = dict(data=granule_mag_df, mew=0.6, ms=4, lw=0.6)
    ax.plot("order", "fluct_squ", "ob", label="Fluct. only", **plot_fmt)
    ax.plot(
                "order", "mag_squ_mean", "1r", label="Total Perturbation", **plot_fmt
            )
    if granule_mag_df["best_fit"] is not None:
        ax.plot(
            "order",
            "best_fit",
            "k-",
            label="Best fit",
            **plot_fmt,
        )
    if standalone_plot:
        # We only want this for the spectrum plots, not the heatmaps
        sigma_si = pt.format_si(float(granule_fit_df['sigma']))
        sigma_si_err = pt.format_si(float(granule_fit_df['sigma_err']))
        title_str = f"σ = {sigma_si} ± {sigma_si_err}N/m, κ = {float(granule_fit_df['kappa_scale']):0.2f} ± {float(granule_fit_df['kappa_scale_err']):0.2f} kT"
        ax.set_title(title_str)

    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())

    ax.fill_between(ax.get_xlim(), y1=0, y2=resolution_threshold, color="lightgrey", alpha=0.5)

    ax.legend(title="Spectrum", fancybox=False, fontsize=8)
    ax.set_ylabel("Mean Perturbation Mag. Squ.")
    ax.set_xlabel("$q$")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if standalone_plot:
        pt.save_figure_and_trim(save_path.with_suffix(".png"))

def plot_heatmap(
        ax=None,
        save_path: Path = "/tmp/heatmap.png",
        sigma_bars=None,
        kappa_scales=None,
        mean_radius = None,
        mag_df: pd.DataFrame = None,
        error_function: callable = None,
        temperature: float = None,
):
    _create_plot = ax is None
    if _create_plot:
        fig, ax = pt.create_axes(
            3,
            # axes_height=2.5,
            axes_height=6.5,
            col_wrap=3,
            sharex=False,
            sharey=False,
            aspect=1,
            # aspect=1.13
        )

    if sigma_bars is None:
        n_sigma = 60
        sigma_bars = np.logspace(1, 5, num=n_sigma)
        x_log = True
    else:
        n_sigma = len(sigma_bars)
        x_log = (sigma_bars.max() / sigma_bars.min()) > 25
        #print(f"alt_width = {sigma_bars.max() / sigma_bars.min()}")

    if kappa_scales is None:
        n_kappa = 60
        kappa_scales = np.logspace(-3, 1, num=n_kappa)
        y_log = True
    else:
        n_kappa = len(kappa_scales)
        y_log = (kappa_scales.max() / kappa_scales.min()) > 25

    rms_grid = np.zeros((n_kappa, n_sigma), dtype=float)
    sigma_grid, kappa_grid = np.meshgrid(sigma_bars, kappa_scales)

    rms_grid = error_function((sigma_grid, kappa_grid))

    rms_invert = np.log(rms_grid / rms_grid.min()) #so no log(0)

    ax[0].pcolormesh(
        sigma_bars, kappa_scales, rms_grid, cmap="inferno_r", shading="nearest",
    )
    ax[1].pcolormesh(
        sigma_bars, kappa_scales, rms_invert, cmap="inferno_r", shading="nearest",
    )
    data = {"sigma_bars": sigma_bars, "kappa_scales": kappa_scales, "rms_grid": rms_grid, "rms_invert": rms_invert}

    min_error = np.min(rms_grid)
    min_args = np.unravel_index(np.argmin(rms_grid, axis=None), rms_grid.shape)
    sigma_bar_min = sigma_bars[min_args[1]]
    kappa_min = kappa_scales[min_args[0]]
    sigma_min = sigma_bar_min / mean_radius ** 2 * kappa_min * kB * temperature
    sigma_min_si = pt.format_si(sigma_min)

    ax[1].set_title(f"σ = {sigma_min_si}N/m, κ = {kappa_min:0.2f} kT fitting = {min_error:0.2f}", fontsize=8)
    ax[1].plot([sigma_bar_min],[kappa_min], "ro")

    plot_spectrum(granule_mag_df=mag_df, granule_fit_df=data, resolution_threshold=0.01, ax=ax[2])

    ax[1].plot([sigma_bar_min],[kappa_min], "ko")

    # if self.fit_para_err != None:
    #     sigma_err, kappa_err = self.physical_errors(mean_radius)
    #     sigma_err_si = pt.format_si(sigma_err)
    #     title_str = f"σ = {sigma_si} +- {sigma_err_si} N/m, κ = {kappa_scale:0.2f} +- {kappa_err:0.2f} kT fitting = {self.fitting_error:0.2f}"
    # else:
    #     title_str = f"σ = {sigma_si} N/m, κ = {kappa_scale:0.2f} kT fitting = {self.fitting_error:0.2f}"
    title_str = ""
    ax[2].set_title(title_str, fontsize=8)

    #line fit
    x = sigma_bars[int(len(sigma_bars)/2):]
    y = [1.8e2 / (sigma) for sigma in x]
    ax[1].plot(x,y,"b-")

    ax[0].contour(sigma_grid, kappa_grid, rms_grid, cmap="Blues_r")
    labels = [f"({i})" for i in "ab"]
    label_colors = ["black", "white"]
    for label, color, ax in zip(labels, label_colors, ax):
        if x_log:
            ax.set_yscale("log")
        if y_log:
            ax.set_xscale("log")
        pt.annotate_axis(ax, label, color=color, fontsize=8)
    pt.set_labels(
        ax,
        ylabels="Bending Rigidity",
        xlabels="Reduced Surface Tension",
        fontsize=8,
    )

    if _create_plot:
        pt.save(save_path)









if __name__ == "__main__":
    argh.dispatch_command(main)
