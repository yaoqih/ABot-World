"""Protocol/packing regressions plus real MP4 I/O using a synthetic test backend.

No model checkpoint or CUDA is needed. Synthetic frames verify orchestration,
not world-model quality; they are created only in pytest temporary directories.
"""
import ast
import copy
import json
import math
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest
import torch

from scripts import control_frequency_sweep as sweep
from utils.action_conditioning import pack_frame_actions
from utils.control_signals import KEY_ORDER, build_schedule, sampling_info


def schedule(**overrides):
    params = dict(fps=12, latent_frames=3, frequency=1, waveform="square", pair="AD",
                  control_mode="frame", duration=2, warmup=0, cooldown=0)
    return build_schedule(**(params | overrides))


def test_square_has_twelve_samples_per_cycle_and_reverses_keys():
    rows = schedule()["rows"]
    assert [r["u_applied"] for r in rows[:12]] == [1] * 6 + [-1] * 6
    assert rows[0]["applied"][KEY_ORDER.index("A")] == 1
    assert rows[6]["applied"][KEY_ORDER.index("D")] == 1
    assert all(not (r["applied"][1] and r["applied"][3]) for r in rows)


def test_sine_preserves_fractional_amplitudes_and_negative_half_cycle():
    rows = schedule(waveform="sine", amplitude=0.5)["rows"]
    assert rows[0]["u_applied"] == 0
    assert rows[1]["applied"][1] == pytest.approx(0.25)
    assert rows[3]["applied"][1] == 0.5
    assert rows[9]["applied"][3] == 0.5


def test_stop_go_is_not_forward_reverse():
    rows = schedule(waveform="stop-go", pair="WS")["rows"]
    assert [r["u_applied"] for r in rows[:12]] == [1] * 6 + [0] * 6
    assert all(r["applied"][2] == 0 for r in rows)


def test_first_frame_alignment_and_final_block_padding():
    result = schedule(duration=1, warmup=1, cooldown=1)
    assert result["video_frames"] == 36
    assert result["generated_frames"] == 45
    assert result["blocks"][0]["packed_frame_indices"] == [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert result["blocks"][1]["packed_frame_indices"] == list(range(9, 21))
    assert result["rows"][11]["segment"] == "warmup"
    assert result["rows"][12]["segment"] == "active"
    assert result["rows"][24]["segment"] == "cooldown"
    assert result["rows"][36]["segment"] == "padding"
    assert sum(r["written_to_video"] for r in result["rows"]) == 36


@pytest.mark.parametrize("mode,expected,rate", [
    ("frame", list(range(13)), 12),
    ("latent", [0] + [1]*4 + [5]*4 + [9]*4, 3),
    ("block", [0]*9 + [9]*4, 1),
])
def test_hold_modes_use_the_causal_video_alignment(mode, expected, rate):
    result = schedule(control_mode=mode)
    assert [r["applied_from_frame"] for r in result["rows"][:13]] == expected
    assert result["sampling"]["control_sample_rate_hz"] == rate
    for row in result["rows"]:
        assert row["applied"] == result["rows"][row["applied_from_frame"]]["command"]


def test_nyquist_boundary_is_not_accepted_and_aliases_are_explicit():
    assert sampling_info(5, 12, 3, "frame")["fundamental_below_nyquist"]
    assert not sampling_info(6, 12, 3, "frame")["fundamental_below_nyquist"]
    assert sampling_info(8, 12, 3, "frame")["folded_fundamental_hz"] == 4
    assert sampling_info(10, 12, 3, "frame")["folded_fundamental_hz"] == 2


@pytest.mark.parametrize("overrides", [
    {"frequency": math.nan}, {"frequency": -1}, {"frequency": math.inf},
    {"duration": 0}, {"warmup": -1}, {"amplitude": 2},
    {"base_keys": ("A",)}, {"base_keys": ("W", "S")}, {"base_keys": ("X",)},
])
def test_invalid_inputs_fail_before_gpu_loading(overrides):
    with pytest.raises(ValueError):
        schedule(**overrides)


@pytest.mark.parametrize("first,count", [(True, 9), (False, 12)])
def test_packing_matches_original_set_act_for_constant_inputs(first, count):
    # Execute the actual legacy method without importing the CUDA-only model stack.
    tree = ast.parse((sweep.ROOT / "pipeline/causal_inference.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CausalInferencePipeline")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "set_act")
    namespace = {"torch": torch, "Optional": Optional}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "legacy_set_act", "exec"), namespace)
    for key in KEY_ORDER:
        values = [int(k == key) for k in KEY_ORDER]
        stub = SimpleNamespace(conditional_dict={})
        namespace["set_act"](stub, dict(zip(KEY_ORDER, values)), height=2, width=3, num_frames=3, device="cpu")
        packed = pack_frame_actions([values] * count, num_latent_frames=3, first_block=first,
                                    height=2, width=3, device="cpu")
        torch.testing.assert_close(packed, stub.conditional_dict["act_context"], rtol=0, atol=0)


def test_temporal_and_key_channels_do_not_get_swapped():
    values = torch.zeros(9, 8)
    values[0, 1] = 1  # Reference action A in all four first-latent slots.
    values[1, 3] = 1  # D in slot 0 of latent 1.
    values[2, 5] = 1  # J in slot 1 of latent 1.
    values[8, 7] = 1  # L in slot 3 of latent 2.
    packed = pack_frame_actions(values, num_latent_frames=3, first_block=True, height=1, width=1)
    assert tuple(packed.shape) == (1, 32, 3, 1, 1)
    nonzero = {tuple(x) for x in packed[0, :, :, 0, 0].nonzero().tolist()}
    assert nonzero == {(4, 0), (5, 0), (6, 0), (7, 0), (12, 1), (21, 1), (31, 2)}


@pytest.mark.parametrize("bad", [torch.zeros(12, 8), torch.full((9, 8), math.nan),
                                   torch.full((9, 8), 1.1), torch.full((9, 8), -0.1)])
def test_packing_rejects_misalignment_and_invalid_values(bad):
    with pytest.raises(ValueError):
        pack_frame_actions(bad, num_latent_frames=3, first_block=True, height=2, width=2)


def fixture_cases(tmp_path, extra=()):
    args = sweep.make_parser().parse_args([
        "--scenes", "scene00", "--frequencies", "1", "--duration", "1",
        "--warmup", "0", "--cooldown", "0", "--output-dir", str(tmp_path), *extra])
    settings = dict(fps=12, height=80, width=320, latent_frames_per_block=3,
                    vae_type="synthetic-test-only", quant_type="none", model_config={})
    scene = dict(id="scene00", label="test", reference_cache_complete=True)
    return args, settings, sweep.make_cases(args, [scene], settings)


def test_alias_policies_and_scene_prompt_semantics(tmp_path):
    args, settings, _ = fixture_cases(tmp_path, ["--frequencies", "6", "8", "10", "--alias-policy", "skip"])
    cases = sweep.make_cases(args, [dict(id="s", label="s", reference_cache_complete=True)], settings)
    assert [c["skip"] for c in cases] == [False, True, True, True]
    args.alias_policy = "error"
    with pytest.raises(ValueError, match="Nyquist"):
        sweep.make_cases(args, [dict(id="s", label="s", reference_cache_complete=True)], settings)
    scenes = sweep.load_scenes(sweep.ROOT / "web_client/scene_presets.yaml")
    assert len(scenes) == 7
    assert scenes[0]["prompt"].startswith("| unknown | A realistic")
    assert scenes[2]["prompt"] == "| unknown |"


class SyntheticBackend:
    def __init__(self, settings):
        import imageio.v2 as imageio
        pytest.importorskip("imageio_ffmpeg")
        self.imageio = imageio
        self.settings = settings
        self.runtime = {"backend": "synthetic-test-only"}
        self.calls = []
        self.fail = False

    def frames(self, case):
        self.calls.append(case["spec"]["id"])
        if self.fail:
            raise RuntimeError("injected inference failure")
        rng = np.random.default_rng(case["spec"]["seed"])
        for row in case["schedule"]["rows"]:
            if row["written_to_video"]:
                frame = rng.integers(0, 256, (self.settings["height"], self.settings["width"], 3), dtype=np.uint8)
                yield frame, row

    def cleanup(self):
        pass


def test_real_mp4_roundtrip_resume_and_corruption_recovery(tmp_path):
    args, settings, cases = fixture_cases(tmp_path, ["--annotate"])
    backend = SyntheticBackend(settings)
    assert sweep.execute(args, cases, settings, lambda _: backend) == 0
    assert len(backend.calls) == 2
    results = json.loads((tmp_path / "results.json").read_text())
    assert all(r["frame_count"] == 12 and r["status"] == "ok" for r in results)
    for case in cases:
        directory = tmp_path / case["spec"]["id"]
        with backend.imageio.get_reader(str(directory / "video.mp4")) as video:
            assert video.count_frames() == 12
            assert video.get_data(0).shape == (80, 320, 3)
        assert (directory / "annotated.mp4").is_file()
        assert (directory / "actions.csv").is_file()
    args.resume = True
    assert sweep.execute(args, cases, settings, lambda _: backend) == 0
    assert len(backend.calls) == 2
    (tmp_path / cases[1]["spec"]["id"] / "video.mp4").write_bytes(b"corrupted")
    assert sweep.execute(args, cases, settings, lambda _: backend) == 0
    assert len(backend.calls) == 3
    assert backend.calls[-1] == cases[1]["spec"]["id"]
    assert not list(tmp_path.rglob("*.partial.mp4"))


def test_failed_videos_are_not_marked_complete_and_nonzero_exit(tmp_path):
    args, settings, cases = fixture_cases(tmp_path)
    backend = SyntheticBackend(settings)
    backend.fail = True
    assert sweep.execute(args, cases, settings, lambda _: backend) == 1
    assert len(backend.calls) == 2  # A per-case failure doesn't lose the remaining plan.
    results = json.loads((tmp_path / "results.json").read_text())
    assert all(r["status"] == "failed" for r in results)
    assert not list(tmp_path.rglob("*.mp4"))


def test_runtime_mismatch_aborts_before_generating_any_new_case(tmp_path):
    args, settings, cases = fixture_cases(tmp_path)
    backend = SyntheticBackend(settings)
    assert sweep.execute(args, cases, settings, lambda _: backend) == 0
    args.resume = True
    # Force a resume that needs to regenerate a case.
    (tmp_path / cases[0]["spec"]["id"] / "video.mp4").unlink()
    backend.runtime = {"backend": "changed-checkpoint"}
    assert sweep.execute(args, cases, settings, lambda _: backend) == 1
    assert len(backend.calls) == 2
    results = json.loads((tmp_path / "results.json").read_text())
    assert len(results) == 1
    assert "identity changed" in results[0]["error"]


def test_dry_run_never_loads_the_gpu_backend_and_resume_rejects_changes(tmp_path):
    args, settings, cases = fixture_cases(tmp_path, ["--dry-run"])
    def forbidden(_):
        raise AssertionError("Dry run must not load a backend")
    assert sweep.execute(args, cases, settings, forbidden) == 0
    assert not list(tmp_path.rglob("*.mp4"))
    assert len(list(tmp_path.rglob("actions.csv"))) == 2
    args.resume = True
    changed = copy.deepcopy(cases)
    changed[0]["spec"]["seed"] = 43
    with pytest.raises(ValueError, match="differ"):
        sweep.execute(args, changed, settings, forbidden)


def test_unavailable_backend_is_attempted_once_and_recorded(tmp_path):
    args, settings, cases = fixture_cases(tmp_path)
    calls = []
    def missing(_):
        calls.append(1)
        raise RuntimeError("CUDA unavailable in test")
    assert sweep.execute(args, cases, settings, missing) == 1
    assert len(calls) == 1
    result = json.loads((tmp_path / "results.json").read_text())[0]
    assert result["status"] == "failed" and "CUDA" in result["error"]


def test_production_stream_loop_resets_history_and_pairs_noise(tmp_path):
    """Exercise the real stream loop/prepare with a small CPU pipeline double."""
    args, settings, cases = fixture_cases(tmp_path, ["--no-baseline", "--frequencies", "1", "2"])
    backend = object.__new__(sweep.InferenceBackend)
    backend.torch, backend.device, backend.settings = torch, torch.device("cpu"), settings
    backend.scene_id, backend.condition = None, None
    tree = ast.parse((sweep.ROOT / "pipeline/causal_inference.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CausalInferencePipeline")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "set_act_sequence")
    namespace = {"torch": torch, "Optional": Optional}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "set_act_sequence", "exec"), namespace)

    class PipelineDouble:
        set_act_sequence = namespace["set_act_sequence"]

        def __init__(self):
            self.device = torch.device("cpu")
            self.vae = SimpleNamespace(model=SimpleNamespace(clear_cache=lambda: None))
            self.encoder = SimpleNamespace(model=SimpleNamespace(clear_cache=lambda: None),
                                           z_dim=2, upsampling_factor=16)
            self.prompts_calls = self.resets = self.clears = 0
            self.noises, self.actions = [], []
            self.conditional_dict = None

        def _clear_generation_caches(self):
            self.clears += 1

        def set_prompts(self, *args, **kwargs):
            self.prompts_calls += 1
            self.conditional_dict = {"prompt_embeds": torch.ones(1)}

        def set_ref_latent_mask_from_exists_paths(self, *args, **kwargs):
            self.conditional_dict["ref_mask"] = torch.ones(1)

        def set_first_frame_latent(self, *args, **kwargs):
            self.conditional_dict["first_frame_latents"] = torch.ones(1)

        def reset_stream(self, *args, **kwargs):
            self.current_start_frame = 0
            self.resets += 1

        def generate_next_block(self, noise):
            self.noises.append(noise.clone())
            self.actions.append(self.conditional_dict["act_context"].clone())
            self.current_start_frame += 3
            return noise + torch.randn_like(noise)  # Simulate scheduler RNG consumption.

    backend.pipeline = PipelineDouble()
    def decode(pipeline, latent):
        count = 9 if pipeline.current_start_frame == 3 else 12
        return [np.zeros((80, 320, 3), dtype=np.uint8)] * count
    backend.decode = decode
    for case in cases:
        case["spec"]["scene"].update(image="fixture", ref_cache_dir="fixture", prompt="fixture")
        rows = list(backend.frames(case))
        assert len(rows) == 12
        assert [row["frame_index"] for _, row in rows] == list(range(12))
    pipeline = backend.pipeline
    assert pipeline.prompts_calls == 1  # Static scene conditions are reusable.
    assert pipeline.resets == pipeline.clears == 2
    assert len(pipeline.noises) == 4
    torch.testing.assert_close(pipeline.noises[0], pipeline.noises[2], rtol=0, atol=0)
    torch.testing.assert_close(pipeline.noises[1], pipeline.noises[3], rtol=0, atol=0)
    assert not torch.equal(pipeline.actions[0], pipeline.actions[2])
    backend.decode = lambda *_: []
    with pytest.raises(RuntimeError, match="decoded frames"):
        list(backend.frames(cases[0]))
