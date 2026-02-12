# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""NDVI anomaly detection operator.

Compares NDVI values between a baseline period and a comparison period to detect
vegetation anomalies. Outputs per-pixel change maps and anomaly flags.
"""

import os
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from vibe_core.data import AssetVibe, CategoricalRaster, DataVibe, Raster, gen_guid
from vibe_lib.raster import (
    get_categorical_cmap,
    get_cmap,
    json_to_asset,
    load_raster,
    save_raster_to_asset,
)


def compute_temporal_mean(rasters: List[Raster]) -> Tuple[xr.DataArray, NDArray[np.float32]]:
    """Compute temporal mean of rasters at pixel level.

    Args:
        rasters: List of rasters to average.

    Returns:
        Tuple of (reference xarray for metadata, mean values array).
    """
    if not rasters:
        raise ValueError("Cannot compute temporal mean of empty raster list")

    # Sort by time and load all rasters
    rasters = sorted(rasters, key=lambda x: x.time_range[0])

    # Load first raster as reference for geometry/metadata
    ref_data = load_raster(rasters[0], use_geometry=True)
    ref_ma = ref_data.to_masked_array()

    # Stack all rasters
    stack = [ref_ma]
    for r in rasters[1:]:
        data = load_raster(r, use_geometry=True).to_masked_array()
        stack.append(data)

    # Compute mean along time axis, ignoring NaN/masked values
    stacked = np.ma.stack(stack, axis=0)
    mean_values = np.ma.mean(stacked, axis=0).astype(np.float32)

    return ref_data, mean_values


def compute_anomaly(
    baseline_mean: NDArray[np.float32],
    comparison_mean: NDArray[np.float32],
    threshold: float,
) -> Tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Compute NDVI difference and anomaly flags.

    Args:
        baseline_mean: Baseline period mean NDVI.
        comparison_mean: Comparison period mean NDVI.
        threshold: Threshold for flagging anomalies (negative = decline).

    Returns:
        Tuple of (difference array, anomaly flags).
    """
    # Difference: comparison - baseline
    # Negative values indicate decline in vegetation health
    diff = comparison_mean - baseline_mean

    # Flag anomalies where difference is below threshold
    # threshold is typically negative (e.g., -0.1 for 10% decline)
    anomaly = cast(NDArray[np.int32], (diff < threshold).astype(np.int32))

    return diff.astype(np.float32), anomaly


def compute_summary_stats(
    diff: NDArray[np.float32], anomaly: NDArray[np.int32]
) -> Dict[str, float]:
    """Compute summary statistics for the anomaly detection.

    Args:
        diff: Difference array.
        anomaly: Anomaly flag array.

    Returns:
        Dictionary of summary statistics.
    """
    valid_diff = diff[~np.ma.getmaskarray(diff) & ~np.isnan(diff)]
    valid_anomaly = anomaly[~np.ma.getmaskarray(anomaly)]

    return {
        "mean_change": float(np.nanmean(valid_diff)) if len(valid_diff) > 0 else 0.0,
        "std_change": float(np.nanstd(valid_diff)) if len(valid_diff) > 0 else 0.0,
        "min_change": float(np.nanmin(valid_diff)) if len(valid_diff) > 0 else 0.0,
        "max_change": float(np.nanmax(valid_diff)) if len(valid_diff) > 0 else 0.0,
        "anomaly_fraction": (
            float(np.sum(valid_anomaly) / len(valid_anomaly))
            if len(valid_anomaly) > 0
            else 0.0
        ),
        "total_pixels": int(len(valid_diff)),
        "anomaly_pixels": int(np.sum(valid_anomaly)) if len(valid_anomaly) > 0 else 0,
    }


class CallbackBuilder:
    """Callback builder for NDVI anomaly detection."""

    def __init__(self, threshold: Optional[float]):
        """Initialize the callback builder.

        Args:
            threshold: NDVI difference threshold for flagging anomalies.
                       Negative values indicate vegetation decline threshold.
                       Default is -0.1 (10% decline in NDVI).
        """
        self.tmp_dir = TemporaryDirectory()
        if threshold is None:
            threshold = -0.1  # Default: flag 10% decline as anomaly
        self.threshold = threshold

    def __call__(self):
        """Return the callback function."""

        def callback(
            baseline_rasters: List[Raster],
            comparison_rasters: List[Raster],
            input_geometry: DataVibe,
        ) -> Dict[str, List[Any]]:
            """Compute NDVI anomaly between baseline and comparison periods.

            Args:
                baseline_rasters: NDVI rasters from the baseline period.
                comparison_rasters: NDVI rasters from the comparison period.
                input_geometry: Geometry of interest.

            Returns:
                Dictionary with change_raster, anomaly_raster, and summary outputs.
            """
            if not baseline_rasters:
                raise ValueError("No baseline rasters provided")
            if not comparison_rasters:
                raise ValueError("No comparison rasters provided")

            # Compute temporal means for both periods
            ref_baseline, baseline_mean = compute_temporal_mean(baseline_rasters)
            _, comparison_mean = compute_temporal_mean(comparison_rasters)

            # Compute difference and anomaly flags
            diff, anomaly_flags = compute_anomaly(
                baseline_mean.data if hasattr(baseline_mean, "data") else baseline_mean,
                comparison_mean.data if hasattr(comparison_mean, "data") else comparison_mean,
                self.threshold,
            )

            # Compute summary statistics
            stats = compute_summary_stats(diff, anomaly_flags)

            # Determine time range spanning both periods
            all_rasters = baseline_rasters + comparison_rasters
            min_time = min(r.time_range[0] for r in all_rasters)
            max_time = max(r.time_range[1] for r in all_rasters)
            time_range = (min_time, max_time)
            geom = input_geometry.geometry

            # Helper function to reshape arrays back to raster format
            def to_xarray(values: NDArray[Any]) -> xr.DataArray:
                data = np.ma.masked_array(values, mask=ref_baseline.isnull())
                data = ref_baseline.copy(data=data.filled(np.nan))
                data.rio.update_encoding({"dtype": str(values.dtype)}, inplace=True)
                return data

            # Save change raster (difference)
            diff_vis = {
                "bands": [0],
                "colormap": get_cmap("RdYlGn"),  # Red=negative, Green=positive
                "range": (-0.5, 0.5),  # NDVI difference range
            }
            change_raster = Raster(
                id=gen_guid(),
                geometry=geom,
                time_range=time_range,
                assets=[
                    save_raster_to_asset(to_xarray(diff), self.tmp_dir.name),
                    json_to_asset(diff_vis, self.tmp_dir.name),
                ],
                bands={"ndvi_change": 0},
            )

            # Save anomaly raster (categorical)
            anomaly_vis = {
                "bands": [0],
                "colormap": get_categorical_cmap("tab10", 2),
                "range": (0, 1),
            }
            anomaly_raster = CategoricalRaster(
                id=gen_guid(),
                geometry=geom,
                time_range=time_range,
                assets=[
                    save_raster_to_asset(to_xarray(anomaly_flags), self.tmp_dir.name),
                    json_to_asset(anomaly_vis, self.tmp_dir.name),
                ],
                bands={"anomaly": 0},
                categories=["normal", "anomaly"],
            )

            # Save summary as a simple JSON asset
            import json

            summary_id = gen_guid()
            summary_path = os.path.join(self.tmp_dir.name, f"{summary_id}_summary.json")
            with open(summary_path, "w") as f:
                json.dump(stats, f, indent=2)

            summary_vibe = DataVibe(
                id=gen_guid(),
                geometry=geom,
                time_range=time_range,
                assets=[AssetVibe(reference=summary_path, type="application/json", id=summary_id)],
            )

            return {
                "change_raster": [change_raster],
                "anomaly_raster": [anomaly_raster],
                "summary": [summary_vibe],
            }

        return callback

    def __del__(self):
        """Clean up temporary directory."""
        self.tmp_dir.cleanup()
