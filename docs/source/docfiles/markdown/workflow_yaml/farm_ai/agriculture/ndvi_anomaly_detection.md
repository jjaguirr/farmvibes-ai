# farm_ai/agriculture/ndvi_anomaly_detection

Detects NDVI anomalies by comparing vegetation indices between a baseline period and a comparison period. This workflow compares vegetation health across two time periods to identify areas of significant change. It downloads and preprocesses Sentinel-2 imagery for both a baseline period (e.g., same season last year) and a comparison period (e.g., current season), computes NDVI for each period, then calculates the per-pixel difference. Areas where the NDVI decline exceeds the configurable threshold are flagged as anomalies.

Use cases include detecting drought stress, disease outbreaks, pest damage, or management changes by comparing current vegetation health to historical baselines. The workflow accounts for cloud cover using improved masks and aggregates multiple dates within each period to reduce noise from individual observations.

```{mermaid}
    graph TD
    inp1>baseline_input]
    inp2>comparison_input]
    out1>change_raster]
    out2>anomaly_raster]
    out3>summary]
    out4>baseline_ndvi]
    out5>comparison_ndvi]
    tsk1{{baseline_s2}}
    tsk2{{baseline_ndvi}}
    tsk3{{comparison_s2}}
    tsk4{{comparison_ndvi}}
    tsk5{{anomaly}}
    tsk1{{baseline_s2}} -- raster --> tsk2{{baseline_ndvi}}
    tsk3{{comparison_s2}} -- raster --> tsk4{{comparison_ndvi}}
    tsk2{{baseline_ndvi}} -- index_raster/baseline_rasters --> tsk5{{anomaly}}
    tsk4{{comparison_ndvi}} -- index_raster/comparison_rasters --> tsk5{{anomaly}}
    inp1>baseline_input] -- user_input --> tsk1{{baseline_s2}}
    inp1>baseline_input] -- input_geometry --> tsk5{{anomaly}}
    inp2>comparison_input] -- user_input --> tsk3{{comparison_s2}}
    tsk5{{anomaly}} -- change_raster --> out1>change_raster]
    tsk5{{anomaly}} -- anomaly_raster --> out2>anomaly_raster]
    tsk5{{anomaly}} -- summary --> out3>summary]
    tsk2{{baseline_ndvi}} -- index_raster --> out4>baseline_ndvi]
    tsk4{{comparison_ndvi}} -- index_raster --> out5>comparison_ndvi]
```

## Sources

- **baseline_input**: Time range and geometry for the baseline (reference) period. This should typically be the same season from a previous year or a known healthy period.

- **comparison_input**: Time range and geometry for the comparison (current) period. Should cover the same geometry as baseline_input but a different time range.

## Sinks

- **change_raster**: Per-pixel NDVI change raster (GeoTIFF). Values are comparison - baseline, so negative values indicate vegetation decline.

- **anomaly_raster**: Binary categorical raster flagging anomaly zones. Pixels with NDVI decline exceeding the threshold are marked as anomalies.

- **summary**: Summary statistics including mean change, standard deviation, anomaly fraction, and pixel counts.

- **baseline_ndvi**: NDVI rasters computed for the baseline period.

- **comparison_ndvi**: NDVI rasters computed for the comparison period.

## Parameters

- **pc_key**: Optional Planetary Computer API key.

- **threshold**: NDVI difference threshold for flagging anomalies. Default is -0.1 (flag pixels with more than 0.1 NDVI decline). Use more negative values for stricter detection (e.g., -0.2 for 20% decline).

## Tasks

- **baseline_s2**: Downloads and preprocesses Sentinel-2 imagery that covers the input geometry and time range, and computes improved cloud masks using cloud and shadow segmentation models.

- **baseline_ndvi**: Computes NDVI index from the bands of the baseline Sentinel-2 rasters.

- **comparison_s2**: Downloads and preprocesses Sentinel-2 imagery that covers the input geometry and time range, and computes improved cloud masks using cloud and shadow segmentation models.

- **comparison_ndvi**: Computes NDVI index from the bands of the comparison Sentinel-2 rasters.

- **anomaly**: Computes NDVI anomalies between baseline and comparison periods by comparing temporal mean NDVI values at each pixel.

## Workflow Yaml

```yaml

name: ndvi_anomaly_detection
sources:
  baseline_input:
  - baseline_s2.user_input
  - anomaly.input_geometry
  comparison_input:
  - comparison_s2.user_input
sinks:
  change_raster: anomaly.change_raster
  anomaly_raster: anomaly.anomaly_raster
  summary: anomaly.summary
  baseline_ndvi: baseline_ndvi.index_raster
  comparison_ndvi: comparison_ndvi.index_raster
parameters:
  pc_key: null
  threshold: null
tasks:
  baseline_s2:
    workflow: data_ingestion/sentinel2/preprocess_s2_improved_masks
    parameters:
      max_tiles_per_time: 1
      pc_key: '@from(pc_key)'
  baseline_ndvi:
    workflow: data_processing/index/index
    parameters:
      index: ndvi
  comparison_s2:
    workflow: data_ingestion/sentinel2/preprocess_s2_improved_masks
    parameters:
      max_tiles_per_time: 1
      pc_key: '@from(pc_key)'
  comparison_ndvi:
    workflow: data_processing/index/index
    parameters:
      index: ndvi
  anomaly:
    op: compute_ndvi_anomaly
    parameters:
      threshold: '@from(threshold)'
edges:
- origin: baseline_s2.raster
  destination:
  - baseline_ndvi.raster
- origin: comparison_s2.raster
  destination:
  - comparison_ndvi.raster
- origin: baseline_ndvi.index_raster
  destination:
  - anomaly.baseline_rasters
- origin: comparison_ndvi.index_raster
  destination:
  - anomaly.comparison_rasters
description:
  short_description: Detects NDVI anomalies by comparing vegetation indices between
    a baseline period and a comparison period.
  long_description: This workflow compares vegetation health across two time periods
    to identify areas of significant change. It downloads and preprocesses Sentinel-2
    imagery for both a baseline period (e.g., same season last year) and a comparison
    period (e.g., current season), computes NDVI for each period, then calculates
    the per-pixel difference. Areas where the NDVI decline exceeds the configurable
    threshold are flagged as anomalies.
  sources:
    baseline_input: Time range and geometry for the baseline (reference) period.
    comparison_input: Time range and geometry for the comparison (current) period.
  sinks:
    change_raster: Per-pixel NDVI change raster (GeoTIFF).
    anomaly_raster: Binary categorical raster flagging anomaly zones.
    summary: Summary statistics including mean change, anomaly fraction, etc.
    baseline_ndvi: NDVI rasters computed for the baseline period.
    comparison_ndvi: NDVI rasters computed for the comparison period.
  parameters:
    pc_key: Optional Planetary Computer API key.
    threshold: NDVI difference threshold for flagging anomalies.


```
