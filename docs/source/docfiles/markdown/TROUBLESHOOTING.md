# Troubleshooting

This document compiles the most common issues encountered when installing and running FarmVibes.AI platform, grouped into broad categories.
Besides the issues listed here, we also collect a list of [known issues on our GitHub repository](https://github.com/microsoft/farmvibes-ai/labels/known%20issues)
that are currently being addressed by the development team.

- **Package installation:**

    <details>
    <summary> Permission denied when installing `vibe_core`</summary>

    Old versions of `pip` might fail to install the `vibe_core` library because
    it erroneously tries to write the library to the system's `site-packages`
    directory.

    An excerpt of the error follows:

    ```
    × python setup.py develop did not run successfully.
    │ exit code: 1
    ╰─> [32 lines of output]
        running develop
        /usr/lib/python3/dist-packages/setuptools/command/easy_install.py:158:
            EasyInstallDeprecationWarning: easy_install command is deprecated. Use
            build and pip and other standards-based tools.
          warnings.warn(
        WARNING: The user site-packages directory is disabled.
        /usr/lib/python3/dist-packages/setuptools/command/install.py:34:
            SetuptoolsDeprecationWarning: setup.py install is deprecated. Use build
            and pip and other standards-based tools.
          warnings.warn(
        error: can't create or remove files in install directory
    ```

    If that happens, you might have to upgrade `pip` itself. Please run `pip
    install --upgrade pip` if you have write access to the directory where `pip`
    is installed, or `sudo pip install --upgrade pip` if you need root
    privileges.

    </details>

<br>

- **Cluster setup:**

    <details>
    <summary> How to change the storage location during cluster creation</summary>

    You may change the storage location by defining the environment variable
    `FARMVIBES_AI_STORAGE_PATH` prior to installation with the *farmvibes-ai*
    command. Additionally, you may use the flag `--storage-path` when running
    the `farmvibes-ai local setup` command. For more information, please refer
    to the help message of the *farmvibes-ai* command.

    </details>

    <details>
    <summary> Missing secrets</summary>

    Running a workflow while missing a required secret will yield the following error message:

    ```bash
    Could not retrieve secret {secret_name} from Dapr.
    ```

    Add the missing secrets to the Kubernetes cluster. [Learn more about secrets here](SECRETS.md).

    </details>

    <details>
    <summary> No route to the Rest-API </summary>

    Building a cluster with the *farmvibes-ai* command will set up a Rest-API
    service with an address visible only within the cluster. In case the client
    cannot reach the Rest-API, make sure to restart the cluster with:

    ```bash
    farmvibes-ai local restart
    ```

    </details>

    <details>
    <summary> Running out of space even after changing storage location</summary>

    If, even after setting the `FARMVIBES_AI_STORAGE_PATH` env var to point to
    another location you are still running out of space with FarmVibes.AI, you
    might have to change the storage location of the docker daemon.

    That happens because even though asset storage goes into
    `FARMVIBES_AI_STORAGE_PATH`, we still use temporary space in our worker
    pods. If your operating system's disk is limited in space (especially when
    running multiple workers), you might run out of space. If that's the case,
    you can change the [docker daemon data directory
    location](https://docs.docker.com/config/daemon/#daemon-data-directory) to
    another disk with more space.

    For example, to instruct the docker daemon to save data in
    `/mnt/docker-data`, you would have to define the contents of `/etc/docker/daemon.json`
    as

    ```json
    {
      "data-root": "/mnt/docker-data"
    }
    ```

    As an alternative you might also want to delete data from previous workflow runs
    to free some space. For more information on how to do that and other data
    management operations, please refer to the [Data Management user guide](CACHE.md).

    </details>

    <details>
    <summary> Unable to run workflows after machine rebooted </summary>

    After a reboot, make sure to start the cluster with:

    ```bash
    farmvibes-ai local start
    ```

    </details>

    <details>
    <summary> Updating cluster in the `dev` branch after pulling files with Git LFS </summary>

    If you did not have Git LFS installed when cloning the repository and checking out to `dev`,
    you will be missing some of the large files in the repository (e.g., ONNX models). Make sure
    to install and setup Git LFS as described in the [Quickstart guide](QUICKSTART.md#restore-files-with-git-lfs).
    You will also need to update your cluster with `make local`.
    </details>

<br>

- **Composing and running workflows:**

    <details>
    <summary> Calling an unknown workflow</summary>

    Calling `client.run()` with a wrong workflow name will yield the following error message:

    ```HTTPError: 400 Client Error: Bad Request for url: http://192.168.49.2:30000/v0/runs. Unable to run workflow with provided parameters. Workflow "WORKFLOW_NAME" unknown```

    Solutions:

  - Double check the workflow name and parameters;
  - Verify that your cluster and repo are up-to-date;

    </details>

    <details>
    <summary> Tasks fail with "subprocess terminated by SIGKILL"</summary>

    Some workflows (notably SpaceEye and Segment Anything Model) can use a large
    amount of memory depending on the input area and time range. When the
    container exceeds its cgroup memory limit, the kernel OOM-killer sends
    SIGKILL to the offending subprocess and the workflow fails with an error
    like:

    ```
    Op '<name>' subprocess terminated by SIGKILL (exit code -9). Memory at termination: 4096MB / 4096MB (100%).
    ```

    **What to do:**

  - **Reduce input size** — smaller region of interest, shorter time range, fewer bands.
  - **Scale down worker replicas** so each gets more of the node's memory:
    ```bash
    ~/.config/farmvibes-ai/kubectl scale deployment terravibes-worker --replicas=1
    ```
  - **Increase the worker memory request** via the `worker_memory_request`
    Terraform variable (default `8Gi`), then re-apply.
  - If the machine itself lacks RAM, migrate the cluster to a larger node.

    **Earlier versions** showed `ProcessExpired: Abnormal termination` for the
    same cause — the worker now reports the actual signal and memory stats so
    you don't have to guess.

    </details>

    <details>
    <summary> Unable to find ONNX model when running workflows </summary>

    Make sure the ONNX model was added to the FarmVibes.AI cluster:

    ```bash
    farmvibes-ai local add-onnx <onnx-model>
    ```

    If no output is generated, then your model was successfully added.

    </details>

    <details>
    <summary> Verifying why a workflow run failed </summary>

    In case a workflow run fails, you might see a similar status table when monitoring a run with `run.monitor()` (please refer to the [client documentation](CLIENT.md) for more information on `monitor`):

    ```bash
    >>> run.monitor()
                                  🌎 FarmVibes.AI 🌍 dataset_generation/datagren_crop_segmentation 🌏
                                          Run name: Generating dataset for crop segmentation                                    
                                            Run id: dd541f5b-4f03-46e2-b017-8e88a518dfe6                              
                                                          Run status: failed                                           
                                                        Run duration: 00:00:16
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ Task Name                          ┃ Status   ┃ Start Time          ┃ End Time            ┃ Duration ┃ Progress                    ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │ spaceeye.preprocess.s2.s2.download │ failed   │ 2022/10/03 22:22:16 │ 2022/10/03 22:22:20 │ 00:00:00 │  ━━━━━━━━━━━━━━━━━━━━  0/1  │
    │ cdl.download_cdl                   │ done     │ 2022/10/03 22:22:12 │ 2022/10/03 22:22:15 │ 00:00:05 │  ━━━━━━━━━━━━━━━━━━━━  1/1  │
    │ spaceeye.preprocess.s2.s2.filter   │ done     │ 2022/10/03 22:22:10 │ 2022/10/03 22:22:12 │ 00:00:02 │  ━━━━━━━━━━━━━━━━━━━━  1/1  │
    │ spaceeye.preprocess.s2.s2.list     │ done     │ 2022/10/03 22:22:09 │ 2022/10/03 22:22:10 │ 00:00:01 │  ━━━━━━━━━━━━━━━━━━━━  1/1  │
    │ cdl.list_cdl                       │ done     │ 2022/10/03 22:22:04 │ 2022/10/03 22:22:09 │ 00:00:04 │  ━━━━━━━━━━━━━━━━━━━━  1/1  │
    └────────────────────────────────────┴──────────┴─────────────────────┴─────────────────────┴──────────┴─────────────────────────────┘
                                                   Last update: 2022/10/03 22:23:59
    ```

    The platform logs the possible reason why a task failed, which might be recovered with `run.reason` and `run.task_details`.

    </details>

    <details>
    <summary> Workflow run with 'pending' status indefinitally</summary>

    If the status of a workflow run remains in 'pending', make sure to restart the cluster with:

    ```bash
    farmvibes-ai local restart
    ```

    </details>

<br>

- **Example notebooks:**

  <details>
  <summary> Unable to import modules when running a notebook</summary>

  Make sure you have installed and activated the [micromamba environment](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) provided with the notebook.

  </details>

  <details>
  <summary> Workflow run monitor table not rendering inside notebook</summary>

  Make sure to have the `ipywidgets` [package](https://pypi.org/project/ipywidgets/) installed in your environment.

  </details>

<br>

- **Segment Anything Model (SAM):**

  <details>
  <summary> Adding SAM's ONNX models to the cluster</summary>

  Running workflows based on SAM requires the image encoder and prompt encoder/mask decoder to be
  exported as ONNX models, and added to the cluster. To do so, run the following command in the repository root:
  
  ```bash
  python scripts/export_sam_models.py --models <model_types>
  ```

  where `<model_types>` is a list of model types to be exported (`vit_b`, `vit_l`, `vit_h`).
  For example, to export all three ViT backbones, run:
  
  ```bash
  python scripts/export_sam_models.py --models vit_b vit_l vit_h
  ```

  The script will download the models from the [SAM repository](https://github.com/facebookresearch/segment-anything),
  export each component as a separate ONNX file, and add them to the cluster with the
  `farmvibes-ai local add-onnx` command. If you are using a different storage location,
  make sure to pass the `--storage-path` flag to the `add-onnx` command.

  Before running the script, make sure you have a micromamba environment set up with the required
  packages. You can use the environments defined by `env_cpu.yaml` or `env_gpu.yaml` files in the
  `notebooks/segment_anything` directory.

  </details>

  <details>
  <summary> Unreliable segmentation mask when using bounding box as prompt</summary>

  As the input Sentinel-2 rasters may be considerably larger than the images expected by SAM, we split the rasters
  into 1024 x 1024 chips (with an overlap defined by the `spatial_overlap` parameter of the workflow). This may lead to
  corner cases that yield unreliable segmentation masks, especially when using a bounding box as prompt. To avoid
  such cases, consider the following:

  - Only a single bounding box is supported per prompt group (i.e., all points with the same `prompt_id`).
  - We recommend providing at least one foreground point within the bounding box. Even though the model supports segmentating rasters solely with a bounding box, the results may be unreliable.
  - If the prompt contains a foreground point outside the provided bounding box, the workflow will adjust the bounding box to include all foreground points in that prompt group.
  - Background points outside the bounding box are ignored.
  - Regions outside the bounding box will be masked out in the final segmentation mask.

  </details>

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
```
