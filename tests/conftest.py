"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import pickle
from datetime import datetime
from typing import Generator, Tuple, TypedDict

import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from aurora import Batch, Metadata
from aurora.batch import interpolate_numpy


class SavedMetadata(TypedDict):
    """Type of metadata of a saved test batch."""

    lat: np.ndarray
    lon: np.ndarray
    time: list[datetime]
    atmos_levels: list[int | float]


class SavedBatch(TypedDict):
    """Type of a saved test batch."""

    surf_vars: dict[str, np.ndarray]
    static_vars: dict[str, np.ndarray]
    atmos_vars: dict[str, np.ndarray]
    metadata: SavedMetadata


@pytest.fixture()
def test_input_output() -> Generator[tuple[Batch, SavedBatch], None, None]:
    # Load test input.
    path = hf_hub_download(
        repo_id="microsoft/aurora",
        filename="aurora-0.25-small-pretrained-test-input.pickle",
    )
    with open(path, "rb") as f:
        test_input: SavedBatch = pickle.load(f)

    # Load static variables.
    path = hf_hub_download(
        repo_id="microsoft/aurora",
        filename="aurora-0.25-static.pickle",
    )
    with open(path, "rb") as f:
        static_vars: dict[str, np.ndarray] = pickle.load(f)

    static_vars = {
        k: interpolate_numpy(
            v,
            np.linspace(90, -90, v.shape[0]),
            np.linspace(0, 360, v.shape[1], endpoint=False),
            test_input["metadata"]["lat"],
            test_input["metadata"]["lon"],
        )
        for k, v in static_vars.items()
    }

    # Construct a proper batch from the test input.
    batch = Batch(
        surf_vars={k: torch.from_numpy(v) for k, v in test_input["surf_vars"].items()},
        static_vars={k: torch.from_numpy(v) for k, v in static_vars.items()},
        atmos_vars={
            k: torch.from_numpy(v) for k, v in test_input["atmos_vars"].items()
        },
        metadata=Metadata(
            lat=torch.from_numpy(test_input["metadata"]["lat"]),
            lon=torch.from_numpy(test_input["metadata"]["lon"]),
            atmos_levels=tuple(test_input["metadata"]["atmos_levels"]),
            time=tuple(test_input["metadata"]["time"]),
        ),
    )

    # Load test output.
    path = hf_hub_download(
        repo_id="microsoft/aurora",
        filename="aurora-0.25-small-pretrained-test-output.pickle",
    )
    with open(path, "rb") as f:
        test_output: SavedBatch = pickle.load(f)

    yield batch, test_output


def _expand_to_num_atmos_levels(arr: np.ndarray, num_levels: int) -> np.ndarray:
    arr_moved = np.moveaxis(arr, 2, -1)
    x_original = np.arange(arr_moved.shape[-1])
    x_new = np.linspace(0, arr_moved.shape[-1] - 1, num_levels)
    f = interp1d(x_original, arr_moved, axis=-1, kind="linear")
    arr_moved_interpolated = f(x_new)
    arr_interpolated = np.moveaxis(arr_moved_interpolated, -1, 2)
    return arr_interpolated


def _get_lats_lons_for_resolution(res: float) -> Tuple[NDArray, NDArray]:
    # Designed to exactly match gridding of Aurora
    num_lats = int(180 / res) + 1
    num_lons = int(360 / res)

    new_lats = np.linspace(90, -90, num_lats)
    new_lons = np.linspace(0, 360, num_lons, endpoint=False)
    return new_lats, new_lons


@pytest.fixture()
def test_full_sized_input() -> Generator[Batch, None, None]:
    resolution = 0.25
    atmos_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

    test_input_path = hf_hub_download(
        repo_id="microsoft/aurora",
        filename="aurora-0.25-small-pretrained-test-input.pickle",
    )
    with open(test_input_path, "rb") as f:
        test_input: SavedBatch = pickle.load(f)

    static_vars_path = hf_hub_download(
        repo_id="microsoft/aurora",
        filename="aurora-0.25-static.pickle",
    )
    with open(static_vars_path, "rb") as f:
        static_vars: dict[str, np.ndarray] = pickle.load(f)

    new_lats, new_lons = _get_lats_lons_for_resolution(res=resolution)
    num_levels = len(atmos_levels)

    static_vars = {
        k: interpolate_numpy(
            v,
            np.linspace(90, -90, v.shape[0]),
            np.linspace(0, 360, v.shape[1], endpoint=False),
            new_lats,
            new_lons,
        )
        for k, v in static_vars.items()
    }

    surf_vars = {
        k: interpolate_numpy(
            v,
            test_input["metadata"]["lat"],
            test_input["metadata"]["lon"],
            new_lats,
            new_lons,
        )
        for k, v in test_input["surf_vars"].items()
    }

    atmos_vars = {
        k: interpolate_numpy(
            _expand_to_num_atmos_levels(v, num_levels),
            test_input["metadata"]["lat"],
            test_input["metadata"]["lon"],
            new_lats,
            new_lons,
        )
        for k, v in test_input["atmos_vars"].items()
    }

    # Construct a proper batch from the test input.
    batch = Batch(
        surf_vars={k: torch.from_numpy(v) for k, v in surf_vars.items()},
        static_vars={k: torch.from_numpy(v) for k, v in static_vars.items()},
        atmos_vars={k: torch.from_numpy(v) for k, v in atmos_vars.items()},
        metadata=Metadata(
            lat=torch.from_numpy(new_lats),
            lon=torch.from_numpy(new_lons),
            atmos_levels=tuple(atmos_levels),
            time=tuple(test_input["metadata"]["time"]),
        ),
    )

    yield batch
