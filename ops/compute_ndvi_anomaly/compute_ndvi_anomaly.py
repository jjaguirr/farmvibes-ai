# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
from tempfile import TemporaryDirectory
from typing import Dict, List

import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr

from vibe_core.data import CategoricalRaster, Raster, TimeSeries, gen_guid
from vibe_lib.raster import (
    get_categorical_cmap,
    get_cmap,
    json_to_asset,
    load_raster,
    load_raster_match,
    save_raster_to_asset,
)
from vibe_lib.timeseries import save_timeseries_to_asset

LOGGER = logging.getLogger(__name__)


def _load_as_stack(rasters: List[Raster], ref: Raster) -> ma.MaskedArray:
    """Load all rasters onto the reference grid and return a (T, 1, H, W) masked array."""
    sorted_rasters = sorted(rasters, key=lambda r: r.time_range[0])
    arrays = []
    for r in sorted_rasters:
        if r is ref:
            da = load_raster(r, use_geometry=True)
        else:
            da = load_raster_match(r, ref, use_geometry=True)
        arrays.append(da.to_masked_array().astype(np.float32))
    return ma.stack(arrays, axis=0)  # shape: (T, bands, H, W)


def _temporal_mean(stack: ma.MaskedArray) -> ma.MaskedArray:
    """Pixel-wise temporal mean over axis 0, honouring the masked array."""
    return stack.mean(axis=0)  # shape: (bands, H, W)


def _to_xr(data: ma.MaskedArray, ref_da: xr.DataArray, dtype: str) -> xr.DataArray:
    """Copy the spatial metadata of ref_da onto a masked array, filling masked pixels with NaN."""
    filled = data.filled(np.nan if "float" in dtype else 0).astype(dtype)
    result = ref_da.copy(data=filled)
    result.rio.update_encoding({"dtype": dtype}, inplace=True)
    return result


class CallbackBuilder:
    def __init__(self, threshold: float):
        self.tmp_dir = TemporaryDirectory()
        self.threshold = float(threshold)

    def __call__(self):
        def callback(
            baseline_rasters: List[Raster],
            comparison_rasters: List[Raster],
        ) -> Dict[str, object]:
            n_base = len(baseline_rasters)
            n_comp = len(comparison_rasters)
            LOGGER.info(
                "Computing NDVI anomaly: %d baseline rasters, %d comparison rasters, "
                "threshold=%.3f",
                n_base,
                n_comp,
                self.threshold,
            )

            # Sort to establish time ranges
            baseline_sorted = sorted(baseline_rasters, key=lambda r: r.time_range[0])
            comparison_sorted = sorted(comparison_rasters, key=lambda r: r.time_range[0])
            ref = baseline_sorted[0]

            # Load stacks and compute temporal means
            baseline_stack = _load_as_stack(baseline_rasters, ref)
            comparison_stack = _load_as_stack(comparison_rasters, ref)
            baseline_mean = _temporal_mean(baseline_stack)  # (bands, H, W)
            comparison_mean = _temporal_mean(comparison_stack)  # (bands, H, W)

            # Per-pixel change: negative values indicate vegetation decline
            change = comparison_mean - baseline_mean

            # Reference DataArray for spatial metadata (shape and CRS)
            ref_da = load_raster(ref, use_geometry=True)

            # --- Change map ---
            change_da = _to_xr(change, ref_da, "float32")
            change_asset = save_raster_to_asset(change_da, self.tmp_dir.name)
            vis_change = {
                "bands": [0],
                "colormap": get_cmap("RdYlGn"),
                "range": (-0.5, 0.5),
            }
            time_range = (baseline_sorted[0].time_range[0], comparison_sorted[-1].time_range[1])
            change_map = Raster(
                id=gen_guid(),
                geometry=ref.geometry,
                time_range=time_range,
                assets=[change_asset, json_to_asset(vis_change, self.tmp_dir.name)],
                bands={"ndvi_change": 0},
            )

            # --- Anomaly mask (1 where change < threshold, else 0) ---
            change_mask = ma.getmaskarray(change)
            anomaly_data = np.where(change_mask, 0, (change.data < self.threshold).astype(np.int16))
            anomaly_ma = ma.array(anomaly_data, mask=change_mask)
            anomaly_da = _to_xr(anomaly_ma, ref_da, "float32")
            anomaly_asset = save_raster_to_asset(anomaly_da, self.tmp_dir.name)
            vis_anomaly = {
                "bands": [0],
                "colormap": get_categorical_cmap("tab10", 2),
                "range": (0, 1),
            }
            anomaly_mask = CategoricalRaster(
                id=gen_guid(),
                geometry=ref.geometry,
                time_range=time_range,
                assets=[anomaly_asset, json_to_asset(vis_anomaly, self.tmp_dir.name)],
                bands={"anomaly": 0},
                categories=["normal", "anomaly"],
            )

            # --- Summary statistics ---
            valid_change = change.compressed()
            valid_pixels = max(int((~change_mask).sum()), 1)
            anomaly_pixels = int(anomaly_data[~change_mask].sum())
            stats = {
                "change_mean": float(valid_change.mean()) if len(valid_change) else float("nan"),
                "change_std": float(valid_change.std()) if len(valid_change) else float("nan"),
                "change_min": float(valid_change.min()) if len(valid_change) else float("nan"),
                "change_max": float(valid_change.max()) if len(valid_change) else float("nan"),
                "anomaly_fraction": anomaly_pixels / valid_pixels,
                "baseline_ndvi_mean": float(baseline_mean.compressed().mean())
                if len(baseline_mean.compressed())
                else float("nan"),
                "comparison_ndvi_mean": float(comparison_mean.compressed().mean())
                if len(comparison_mean.compressed())
                else float("nan"),
            }
            stats_df = pd.DataFrame(
                [stats], index=pd.Index([time_range[1]], name="date")
            )
            summary = TimeSeries(
                id=gen_guid(),
                geometry=ref.geometry,
                time_range=time_range,
                assets=[save_timeseries_to_asset(stats_df, self.tmp_dir.name)],
            )

            LOGGER.info(
                "NDVI anomaly complete: change_mean=%.4f, anomaly_fraction=%.3f",
                stats["change_mean"],
                stats["anomaly_fraction"],
            )

            return {
                "change_map": change_map,
                "anomaly_mask": anomaly_mask,
                "summary": summary,
            }

        return callback

    def __del__(self):
        self.tmp_dir.cleanup()
