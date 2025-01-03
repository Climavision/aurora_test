"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import os
import time
from datetime import timedelta

import numpy as np
import pytest
import torch
import torch.distributed as dist

from tests.conftest import SavedBatch

from aurora import Aurora, AuroraSmall, Batch


def _prepare_model_for_inference(model: Aurora) -> Aurora:
    # Putting encoder and decoder on a separate device from the backbone makes inference
    # on 24GB GPUs feasible for full 0.25 resolution and 13 atmos levels. You can
    # do either decoder on cpu and all else on gpu, or more optimally
    # decoder/encoder on 1 gpu, and backbone on the other.
    model.encoder = model.encoder.to("cuda:0")
    # backbone was trained on bfloat16
    model.backbone = model.backbone.to(torch.bfloat16).to("cuda:1")
    model.decoder = model.decoder.to("cuda:0")
    model.autocast = True
    model.eval()
    return model


@pytest.fixture(scope="session")
def aurora_full() -> Aurora:
    model = Aurora(use_lora=True)
    model.load_checkpoint(
        "microsoft/aurora",
        "aurora-0.25-finetuned.ckpt",
    )
    return _prepare_model_for_inference(model)


# testing at full target inference resolution/num levels
def test_aurora_full(aurora_full: Aurora, test_full_sized_input: Batch) -> None:
    batch = test_full_sized_input

    start_time = time.time()
    print("Starting forecast generation.")

    with torch.inference_mode():
        pred = aurora_full.forward(batch).to("cpu")  # noqa: F841
    print("Finished generating forecasts.")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Inference took {elapsed_time} seconds.")
    # assert False


@pytest.fixture(scope="session")
def aurora_small() -> Aurora:
    model = AuroraSmall(use_lora=True)
    model.load_checkpoint(
        "microsoft/aurora",
        "aurora-0.25-small-pretrained.ckpt",
        strict=False,  # LoRA parameters not available.
    )
    return _prepare_model_for_inference(model)


def test_aurora_small(
    aurora_small: Aurora, test_input_output: tuple[Batch, SavedBatch]
) -> None:
    batch, test_output = test_input_output

    with torch.inference_mode():
        pred = aurora_small.forward(batch).to("cpu")

    def assert_approx_equality(
        v_out: np.ndarray, v_ref: np.ndarray, tol: float
    ) -> None:
        err = np.abs(v_out - v_ref).mean()
        mag = np.abs(v_ref).mean()
        assert err / mag <= tol

    # For some reason, wind speed and specific humidity are more numerically unstable, so we use a
    # higher tolerance of 0.5% there.
    tolerances = {
        "2t": 1e-4,
        "10u": 5e-3,
        "10v": 5e-3,
        "msl": 1e-4,
        "u": 5e-3,
        "v": 5e-3,
        "t": 1e-4,
        "q": 5e-3,
    }

    # Check the outputs.
    for k in pred.surf_vars:
        assert_approx_equality(
            pred.surf_vars[k].numpy(),
            test_output["surf_vars"][k],
            tolerances[k] * 5e4,
        )
    for k in pred.static_vars:
        assert_approx_equality(
            pred.static_vars[k].numpy(),
            batch.static_vars[k].numpy(),
            1e-7,  # These should be exactly equal.
        )
    for k in pred.atmos_vars:
        assert_approx_equality(
            pred.atmos_vars[k].numpy(),
            test_output["atmos_vars"][k],
            tolerances[k],
        )

    np.testing.assert_allclose(pred.metadata.lon, test_output["metadata"]["lon"])
    np.testing.assert_allclose(pred.metadata.lat, test_output["metadata"]["lat"])
    assert pred.metadata.atmos_levels == tuple(test_output["metadata"]["atmos_levels"])
    assert pred.metadata.time == tuple(test_output["metadata"]["time"])


def test_aurora_small_ddp(
    aurora_small: Aurora, test_input_output: tuple[Batch, SavedBatch]
) -> None:
    batch, test_output = test_input_output

    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("gloo", rank=0, world_size=1)

    aurora_small = torch.nn.parallel.DistributedDataParallel(aurora_small)  # type: ignore

    # Just test that it runs.
    with torch.inference_mode():
        aurora_small.forward(batch)


def test_aurora_small_decoder_init() -> None:
    aurora_small = AuroraSmall(use_lora=True)

    # Check that the decoder heads are properly initialised. The biases should be zero, but the
    # weights shouldn't.
    for layer in [
        *aurora_small.decoder.surf_heads.values(),
        *aurora_small.decoder.atmos_heads.values(),
    ]:
        assert not torch.all(layer.weight == 0)
        assert torch.all(layer.bias == 0)


def test_aurora_small_lat_lon_matrices(
    aurora_small: Aurora, test_input_output: tuple[Batch, SavedBatch]
) -> None:
    batch, test_output = test_input_output

    with torch.inference_mode():
        pred = aurora_small.forward(batch).to("cpu")

        # Modify the batch to have a latitude and longitude matrices.
        n_lat = len(batch.metadata.lat)
        n_lon = len(batch.metadata.lon)
        batch.metadata.lat = batch.metadata.lat[:, None].expand(n_lat, n_lon)
        batch.metadata.lon = batch.metadata.lon[None, :].expand(n_lat, n_lon)

        pred_matrix = aurora_small.forward(batch).to("cpu")

    # Check the outputs.
    for k in pred.surf_vars:
        np.testing.assert_allclose(
            pred.surf_vars[k],
            pred_matrix.surf_vars[k],
            rtol=1e-5,
        )
    for k in pred.static_vars:
        np.testing.assert_allclose(
            pred.static_vars[k],
            pred_matrix.static_vars[k],
            rtol=1e-5,
        )
    for k in pred.atmos_vars:
        np.testing.assert_allclose(
            pred.atmos_vars[k],
            pred_matrix.atmos_vars[k],
            rtol=1e-5,
        )


def test_aurora_small_flags(test_input_output: tuple[Batch, SavedBatch]) -> None:
    batch, test_output = test_input_output

    flag_collections: list[dict] = [
        {},
        {"stabilise_level_agg": True},
        {"timestep": timedelta(hours=12)},
    ]

    preds = []
    for flags in flag_collections:
        model = AuroraSmall(use_lora=True, **flags)
        model.load_checkpoint(
            "microsoft/aurora",
            "aurora-0.25-small-pretrained.ckpt",
            strict=False,  # LoRA parameters not available.
        )
        model = _prepare_model_for_inference(model)
        with torch.inference_mode():
            preds.append(model.forward(batch).normalise(model.surf_stats).to("cpu"))

    # Check that all predictions are different.
    for i, pred1 in enumerate(preds):
        for pred2 in preds[i + 1 :]:
            for k in pred1.surf_vars:
                assert not np.allclose(
                    pred1.surf_vars[k],
                    pred2.surf_vars[k],
                    rtol=5e-2,
                )
            for k in pred1.static_vars:
                np.testing.assert_allclose(
                    pred1.static_vars[k],
                    pred2.static_vars[k],
                    rtol=1e-5,
                )
            for k in pred1.atmos_vars:
                assert not np.allclose(
                    pred1.atmos_vars[k],
                    pred2.atmos_vars[k],
                    rtol=5e-2,
                )
