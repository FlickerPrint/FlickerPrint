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
but also abstract parameters, such as the quality of fitting to the expected
spectrum.

Once the summary table is created, we add a number of secondary columns for
convenience, typically physical values re-scaled by some factor or on a log
scale. We also bin the time values into 5 minute intervals that can be used for
aggregation/averaging.

As the fitting is somewhat expensive, we provide an option for reusing the
summary table . This can be cleared by using the ``--clear`` flag, and new data
will be generated.

Merge Caches
------------

Once we have gathered all of the data in an experiment directory, we can then analyse
these files together using the ``merge_caches`` function. We provide means for dealing
with mismatched or missing columns.

Simply provide a list of ``Path`` objects pointing to the ``fittingVals.csv`` files to
merge.

Flags
-----

``--all-frames``
    Do not remove frames that have failed the boundary check when fitting
    for a granule. The pass rate is still recorded with the granule and may be used to
    filter the entire granule from later analysis.

``-j``, ``--processes`` : 1
    Number of cores to run the fittings on; with each core running a different time
    series. Defaults to single core for debugging, but multiple cores are recommended.

``--clear``
    Recreate the ``fittingVals.csv`` file.

"""

from functools import partial
from multiprocessing import Pool
from pathlib import Path

import h5py
import subprocess
import platform
import numpy as np
import pandas as pd
import pickle as pkl
from scipy.constants import Boltzmann as kB
import warnings
import os

from flickerprint.common.utilities import strtobool
from flickerprint.common.configuration import config
from flickerprint.fluctuation import spectra
import flickerprint.version as version


def process_fourier_file(input_path: Path, output: Path, plotting=True):
    """ Perform spectrum fitting on one h5 file. """
    fourier_terms, frame_info = load_fourier_terms(input_path)

    pd.set_option("min_rows", 8)
    pd.set_option("display.width", 300)
    groups = fourier_terms.groupby("granule_id")
    # Provide an empty dataframe so that this won't fail when no granules are found
    aggregate_data = [
        pd.DataFrame(),
    ]
    fourier_terms = [
        pd.DataFrame(),
    ]

    max_order = int(config("spectrum_fitting", "fitting_orders"))
    fitting_method = "least_squares"
    plot_heatmaps = bool(strtobool(config("spectrum_fitting", "plot_spectra_and_heatmaps")))
    pixel_size = frame_info["pixel_size"]

    for granule_id, granule in groups:

        # if granule_id > 100:
        #     break

        mean_radius = granule["mean_radius"].mean()
        mean_intensity = granule["mean_intensity"].mean()
        im_path = Path(granule["im_path"].iloc[0])

        #find first frame
        first_frame_index = granule["frame"].idxmin()
        x = granule["x"][first_frame_index]
        y = granule["y"][first_frame_index]
        timestamp = granule["timestamp"][first_frame_index]
        bbox_left = granule["bbox_left"][first_frame_index]
        bbox_bottom = granule["bbox_bottom"][first_frame_index]
        bbox_right = granule["bbox_right"][first_frame_index]
        bbox_top = granule["bbox_top"][first_frame_index]

        # Add additional terms required for fitting
        granule["mag_abs"] = np.abs(granule["magnitude"])
        granule["mag_squared"] = granule["mag_abs"] ** 2

        # Create a DF of the time averaged terms
        # This is the experimental spectrum that we compare against
        # We have to do the latter term using λ as otherwise it uses a cython version
        # without complex support.
        
        mag_df = granule.groupby(by="order").agg(
            mag_squ_mean=("mag_squared", "mean"),
            mag_mean=("magnitude", lambda x: np.mean(x)),
        )
        mag_df.reset_index(inplace=True)
        mag_df["fixed_squ"] = np.abs(mag_df["mag_mean"]) ** 2
        q_2_mag = mag_df["fixed_squ"][0]
        pixel_threshold = (pixel_size/15)**2/(mean_radius*1e6)**2

        # TODO: Implement a truncated fitting method
        # For now fit only to the first few orders
        granule = granule.query(f"order <= {max_order}")

        # Fit the spectrum for this granule
        fittingST = None
        if (fitting_method == "least_squares"):
            fitting = spectra.FittingLeastSquares(granule.copy(), frame_info)
            fittingST = spectra.FittingSurfaceTensionOnlyLeastSquares(granule.copy(), frame_info)
        elif (fitting_method == "least_squares_bending_only"):
            fitting = spectra.FittingBendingRigidityOnlyLeastSquares(granule.copy(), frame_info)
        elif (fitting_method == "least_squares_surface_only"):
            fitting = spectra.FittingSurfaceTensionOnlyLeastSquares(granule.copy(), frame_info)
        else:
            raise ValueError("fitting method not found")


        try:
            fitting.get_best_fit()
            if fittingST:
                fittingST.get_best_fit()
        except ValueError:
            print(f"Error fitting granule {granule_id}")
            continue
        #fitting.get_best_fit()
        if not np.all(np.isfinite(fitting.best_fit_vals)):
            print(f"Invalid fitting parameters {granule_id}")
            continue

        #heatmap !jl
        if (plot_heatmaps) :
            fitting.plot(fixed=False, mean_radius=mean_radius * 1e-6, outdir=output)
            n_kappa, n_sigma = 100, 100
            sigma_mid, kappa_mid  = 10e1,10e-1
            width = 1000000.0
            line_func = np.linspace if width <= 5 else np.geomspace
            sigma_bars = line_func(sigma_mid / width, sigma_mid * width, num=n_sigma)
            kappa_scales = line_func(kappa_mid / width, kappa_mid * width, num=n_kappa)
            save_name = Path(frame_info["input_path"]).stem + f"--G{granule_id:02d}.png"
            save_path = output / Path("fitting/heatmaps/") / save_name

            fitting.plot_heatmap(
                sigma_bars=sigma_bars, kappa_scales=kappa_scales, save_name=save_path, mean_radius=mean_radius * 1e-6
            )

        sigma, kappa_scale = fitting.physical_vals(mean_radius * 1e-6)
        physical_errors = fitting.physical_errors(mean_radius * 1e-6)
        if physical_errors != None:
            sigma_err, kappa_scale_err = physical_errors
        else:
            sigma_err, kappa_scale_err = (None,None)

        if fittingST:
            sigmaST , kappa_scaleST = fittingST.physical_vals(mean_radius * 1e-6)
            physical_errorsST = fittingST.physical_errors(mean_radius * 1e-6)
            if physical_errors != None:
                sigma_errST, kappa_scale_errST = physical_errorsST
            else:
                sigma_errST, kappa_scale_errST = (None,None)
        
        fourier_term = fitting.mag_df.copy()
        fourier_term["granule_id"] = granule_id
        fourier_term["figure_path"] = fitting.save_name
        above_res_threshold = (fourier_term["experiment_spectrum"] > pixel_threshold).sum() > (len(fourier_term["experiment_spectrum"]) / 2)

        data = dict(
            granule_id=granule_id,
            sigma=sigma,
            sigma_err=sigma_err,
            kappa_scale=kappa_scale,
            kappa_scale_err=kappa_scale_err,
            mean_radius=mean_radius,
            figure_path=fitting.save_name,
            pass_rate=fitting.pass_rate,
            pass_count=fitting.pass_count,
            fitting_error=fitting.fitting_error,
            durbin_watson=fitting.durbin_watson,
            mean_intensity=mean_intensity,
            image_path=str(im_path),
            x=x,
            y=y,
            bbox_left=bbox_left,
            bbox_bottom=bbox_bottom,
            bbox_right=bbox_right,
            bbox_top=bbox_top,
            q_2_mag=q_2_mag,
            experiment = config("workflow", "experiment_name"),
            timestamp = timestamp,
            above_res_threshold= above_res_threshold,
            timestamp=float(timestamp),
        )
        if fittingST:
            data["fitting_diff"] = fittingST.fitting_error - fitting.fitting_error
            data["sigma_st"] = sigmaST
            data["sigma_errST"] = sigma_errST
        aggregate_data.append(pd.DataFrame(data=data, index=(0,)))
        fourier_terms.append(fourier_term)

    aggregate_data = pd.concat(aggregate_data, ignore_index=True,)
    fourier_terms = pd.concat(fourier_terms, ignore_index=True,)
    return aggregate_data, fourier_terms


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
    input_paths = iter(input_paths)

    # Multiprocessing map only accepts one argument, so we use ``partial`` to remove
    # the constant arguments. ``imap`` would also work.
    process_frame = partial(process_fourier_file, output=working_dir, plotting=plotting)
    print("Working.....")

    if cores > 1:
        with Pool(cores) as p:
            frame_info = p.map(process_frame, input_paths, chunksize=1)
    else:
        frame_info = map(process_frame, input_paths)
    #frame_info=process_fourier_file(input_paths,working_dir)

    aggregate_data, fourier_terms = zip(*frame_info)
    aggregate_data = pd.concat(aggregate_data, ignore_index=True,)
    fourier_terms = pd.concat(fourier_terms, ignore_index=True,)
    _write_hdf(working_dir / "aggregate_fittings.h5", aggregate_data, fourier_terms)
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
            print(f"\nAggregate fittings sile location: aggregate_fittings.h5")

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
        print(f"\nAggregate fittings sile location: aggregate_fittings.h5")

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


if __name__ == "__main__":
    main()
