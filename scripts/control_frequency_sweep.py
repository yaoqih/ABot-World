#!/usr/bin/env python3
"""Multi-scene control stress tests. --dry-run does not import the GPU stack."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import itertools
import json
import math
import os
import sys
import time
import traceback
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.control_signals import KEY_ORDER, PAIRS, build_schedule  # noqa: E402


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_scenes(path: Path) -> list[dict]:
    from omegaconf import OmegaConf

    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)

    def prompt_text(value) -> str:
        value = str(value or "").strip()
        if value.endswith(".json"):
            caption = Path(value)
            if not caption.is_absolute():
                caption = path.parent / caption
            value = json.loads(caption.read_text(encoding="utf-8"))["scene_static"]
            if not isinstance(value, str):
                raise ValueError(f"scene_static must be a string: {caption}")
        return value.strip()

    scenes = []
    for group in data.get("groups", []):
        for item in group.get("items", []):
            image = Path(item["image"])
            if not image.is_absolute():
                image = path.parent / image
            image = image.resolve()
            raw_value = item["prompt"] if "prompt" in item else data.get("default_prompt", "")
            raw_prompt = prompt_text(raw_value)
            # Match the existing Gradio caption convention, including explicit "".
            prompt = "| unknown |" + (" " + raw_prompt if raw_prompt else "")
            scenes.append({"id": f"scene{len(scenes):02d}",
                           "label": str(item.get("label", image.stem)),
                           "group": str(group.get("name", "")),
                           "image": str(image), "prompt": prompt})
    if not scenes:
        raise ValueError(f"No scenes found in {path}; expected groups[].items[].")
    return scenes


def scene_resources(scene: dict) -> dict:
    image = Path(scene["image"])
    if not image.is_file():
        raise ValueError(f"Scene image does not exist: {image}")
    # Same cache lookup as the existing application; MD5 is only a cache key.
    cache_key = hashlib.md5(image.read_bytes()).hexdigest()[:16]
    cache = ROOT / "outputs/ref_image_cache" / cache_key
    extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif")
    refs = {}
    for name in ("head", "left", "right", "front", "back"):
        candidates = [cache / (name + variant) for ext in extensions
                      for variant in (ext, ext.upper())]
        found = next((p for p in candidates if p.is_file()), None)
        refs[name] = {"path": str(found), "sha256": file_digest(found)} if found else None
    return {**scene, "image_sha256": file_digest(image), "ref_cache_dir": str(cache),
            "reference_images": refs,
            "reference_cache_complete": all(v is not None for v in refs.values())}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--scenes-file", type=Path, default=ROOT / "web_client/scene_presets.yaml")
    parser.add_argument("--scenes", nargs="+", default=["all"],
                        help="Scene IDs (scene00), indices (0), exact labels, or all")
    parser.add_argument("--list-scenes", action="store_true")
    parser.add_argument("--frequencies", nargs="+", type=float,
                        default=[0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
    parser.add_argument("--waveforms", nargs="+", choices=["square", "sine", "stop-go"],
                        default=["square"])
    parser.add_argument("--pairs", nargs="+", choices=list(PAIRS), default=["AD"])
    parser.add_argument("--control-mode", choices=["frame", "latent", "block"], default="frame",
                        help="frame=experimental 4-slot packing; latent=hold for 4 frames; block=original UI")
    parser.add_argument("--duration", type=float, default=12.0, help="Active oscillation duration, seconds")
    parser.add_argument("--warmup", type=float, default=2.0, help="Nonoscillating prefix, seconds")
    parser.add_argument("--cooldown", type=float, default=2.0, help="Nonoscillating suffix, seconds")
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--phase-deg", type=float, default=0.0)
    parser.add_argument("--base-keys", nargs="*", choices=KEY_ORDER, default=[])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--alias-policy", choices=["skip", "error", "allow"], default="allow",
                        help="Handle requested fundamentals >= control sampling Nyquist limit")
    parser.add_argument("--no-baseline", action="store_true", help="Omit constant-base-keys control videos")
    parser.add_argument("--quant-type", default=None, help="Same quantizer as the demo; none disables it")
    parser.add_argument("--vae-type", choices=["wan2.2", "taew2_2", "mg_lightvae", "mg_lightvae_v2"])
    parser.add_argument("--annotate", action="store_true", help="Also write a HUD video; clean video is always kept")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/control_frequency_sweep" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--dry-run", action="store_true", help="Write plans and action traces, without models/video")
    parser.add_argument("--resume", action="store_true", help="Skip successful matching cases in this output directory")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def experiment_settings(args) -> dict:
    from omegaconf import OmegaConf
    from web_client.config import STREAM_HEIGHT, STREAM_WIDTH, VIDEO_FPS

    config = OmegaConf.merge(OmegaConf.load(ROOT / "configs/default_config.yaml"),
                             OmegaConf.load(ROOT / "configs/long_forcing_dmd.yaml"))
    model_config = OmegaConf.to_container(config, resolve=True)
    vae = args.vae_type or str(config.get("vae_type", "taew2_2"))
    quant = args.quant_type or (str(config.get("quant_type", "fp8-per-token"))
                                if config.get("use_fp8_gemm", False) else "none")
    # Include the actual decoder override in the reproducibility record.
    decoder_override = os.environ.get("TAEW2_2_CHECKPOINT") if vae == "taew2_2" else None
    return {"fps": VIDEO_FPS, "height": STREAM_HEIGHT, "width": STREAM_WIDTH,
            "latent_frames_per_block": int(config.num_frame_per_block),
            "vae_type": vae, "quant_type": quant, "decoder_override": decoder_override,
            "model_config": model_config}


def make_cases(args, scenes: list[dict], settings: dict) -> list[dict]:
    cases = []
    for scene, seed in itertools.product(scenes, dict.fromkeys(args.seeds)):
        conditions = [] if args.no_baseline else [("hold", args.pairs[0], 0.0)]
        conditions += list(itertools.product(dict.fromkeys(args.waveforms),
                                             dict.fromkeys(args.pairs),
                                             dict.fromkeys(args.frequencies)))
        for waveform, pair, frequency in conditions:
            schedule = build_schedule(
                fps=settings["fps"], latent_frames=settings["latent_frames_per_block"],
                frequency=frequency, waveform=waveform, pair=pair, control_mode=args.control_mode,
                duration=args.duration, warmup=args.warmup, cooldown=args.cooldown,
                amplitude=args.amplitude, phase_deg=args.phase_deg, base_keys=tuple(args.base_keys))
            sampling = schedule["sampling"]
            limited = not sampling["fundamental_below_nyquist"]
            if limited and args.alias_policy == "error":
                raise ValueError(f"{frequency:g} Hz is at/above {sampling['nyquist_hz']:g} Hz "
                                 f"Nyquist for {args.control_mode} controls; use --alias-policy skip/allow.")
            name = "baseline" if waveform == "hold" else f"{pair}_{waveform}_{frequency:g}Hz"
            # The hash prevents close float frequencies colliding after filename formatting.
            if waveform != "hold":
                name += "_" + digest(frequency)[:8]
            if limited:
                name += "_SAMPLING_LIMITED"
            case_id = f"{scene['id']}/{args.control_mode}/seed{seed}/{name}"
            warnings = []
            if limited:
                warnings.append("Sampling-limited stress input; NOT an identifiable response at requested Hz.")
            if waveform != "hold" and sampling["samples_per_requested_cycle"] < 8:
                warnings.append("Fewer than 8 control samples/cycle; waveform is coarsely sampled.")
            if waveform != "hold" and schedule["active_cycles"] < 4:
                warnings.append("Fewer than 4 active cycles; increase --duration for response analysis.")
            if waveform == "sine" or args.amplitude != 1:
                warnings.append("Fractional key amplitudes are experimental/OOD for the binary-key interface.")
            if args.control_mode == "frame":
                warnings.append("Within-latent temporal slot semantics require validation against training data.")
            if not scene["reference_cache_complete"]:
                warnings.append("Incomplete multi-view cache: model uses zero reference mask, as in the demo.")
            spec = {"id": case_id, "scene": scene, "seed": seed, "waveform": waveform,
                    "pair": pair, "requested_frequency_hz": frequency,
                    "control_mode": args.control_mode, "amplitude": args.amplitude,
                    "phase_deg": args.phase_deg, "base_keys": args.base_keys,
                    "duration": args.duration, "warmup": args.warmup, "cooldown": args.cooldown,
                    "annotate": args.annotate, "settings": settings}
            cases.append({"spec": spec, "schedule": schedule, "warnings": warnings,
                          "skip": limited and args.alias_policy == "skip"})
    return cases


def write_actions(directory: Path, case: dict) -> None:
    schedule = case["schedule"]
    columns = ["frame_index", "time_s", "segment", "written_to_video", "block_index",
               "applied_from_frame", "u_command", "u_applied"]
    with (directory / "actions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns + [f"{p}_{k}" for p in ("command", "applied") for k in KEY_ORDER])
        writer.writeheader()
        for row in schedule["rows"]:
            item = {k: row[k] for k in columns}
            for prefix in ("command", "applied"):
                item.update({f"{prefix}_{k}": v for k, v in zip(KEY_ORDER, row[prefix])})
            writer.writerow(item)
    write_json(directory / "actions.json", {
        "key_order": KEY_ORDER, "fps": case["spec"]["settings"]["fps"],
        "action_tensor_dtype": "bfloat16",
        "values": "Scheduled adapter inputs before bfloat16 conversion; sine amplitudes are rounded by the model dtype.",
        "channel_layout": "key_index * 4 + temporal_slot",
        "initial_alignment": "reference action replicated four times, then groups of four",
        "video_frames": schedule["video_frames"], "generated_frames": schedule["generated_frames"],
        "blocks": schedule["blocks"],
        "frames": [{"frame_id": f"{row['frame_index']:06d}",
                    "written_to_video": row["written_to_video"],
                    "keys": dict(zip(KEY_ORDER, row["applied"]))} for row in schedule["rows"]],
    })


class InferenceBackend:
    def __init__(self, settings: dict):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU unavailable. Use --dry-run to prepare inputs; run inference "
                               "in the repository's NVIDIA/CUDA environment with downloaded checkpoints.")
        self.torch = torch
        self.settings = settings
        config = settings["model_config"]
        required = [Path(config["model_kwargs"]["model_name"]) / "diffusion_pytorch_model.safetensors",
                    Path(config["text_encoder_kwargs"]["encoder_pth_path"]),
                    Path(config["text_encoder_kwargs"]["tokenizer_path"]),
                    Path(config["vae_kwargs"]["pretrained_path"])]
        if settings["vae_type"] == "taew2_2":
            required.append(Path(settings["decoder_override"] or config["taew2_2_checkpoint"]))
        elif settings["vae_type"].startswith("mg_lightvae"):
            key = "lightvae_v2_checkpoint" if settings["vae_type"].endswith("v2") else "lightvae_checkpoint"
            if not config.get(key):
                raise ValueError(f"Missing decoder configuration: {key}")
            required.append(Path(config[key]))
            if config.get("lightvae_encoder_checkpoint"):
                required.append(Path(config["lightvae_encoder_checkpoint"]))
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing checkpoint resources: " + ", ".join(missing))
        import imageio.v2 as imageio
        import imageio_ffmpeg
        imageio_ffmpeg.get_ffmpeg_exe()
        from web_client.pipeline_loader import get_pipeline, decode_block_to_frames

        self.imageio = imageio
        self.decode = decode_block_to_frames
        self.pipeline, _, self.device = get_pipeline(
            vae_type=settings["vae_type"], use_fp8_gemm=settings["quant_type"] != "none",
            quant_type=settings["quant_type"])
        if self.pipeline.num_frame_per_block != settings["latent_frames_per_block"]:
            raise RuntimeError("Loaded pipeline block size differs from the experiment plan.")
        self.pipeline.eval()
        self.scene_id = None
        self.condition = None
        self.runtime = {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(self.device),
                        "checkpoints": [{"path": str(p.resolve()), "size": p.stat().st_size,
                                         "mtime_ns": p.stat().st_mtime_ns} for p in required]}

    def prepare(self, scene: dict):
        pipeline = self.pipeline
        # Discard all history, including model-level cached reference tokens.
        pipeline._clear_generation_caches()
        if self.scene_id != scene["id"]:
            self.condition = None
            pipeline.vae.model.clear_cache()
            pipeline.set_prompts([scene["prompt"]], device=self.device)
            pipeline.set_ref_latent_mask_from_exists_paths(scene["ref_cache_dir"], device=self.device)
            pipeline.set_first_frame_latent(scene["image"], height=self.settings["height"],
                                            width=self.settings["width"], device=self.device)
            self.condition = dict(pipeline.conditional_dict)
            self.scene_id = scene["id"]
        pipeline.conditional_dict = dict(self.condition)
        pipeline.vae.model.clear_cache()
        if pipeline.encoder is not None:
            pipeline.encoder.model.clear_cache()
        pipeline.reset_stream(1, dtype=self.torch.bfloat16, device=self.device, initial_latent=None)

    def frames(self, case: dict):
        import random
        import numpy as np

        torch = self.torch
        spec, schedule = case["spec"], case["schedule"]
        with torch.inference_mode():
            self.prepare(spec["scene"])
            # Reseed AFTER conditioning; also covers randn_like inside the scheduler.
            random.seed(spec["seed"])
            np.random.seed(spec["seed"])
            torch.manual_seed(spec["seed"])
            torch.cuda.manual_seed_all(spec["seed"])
            vae = self.pipeline.encoder if self.pipeline.encoder is not None else self.pipeline.vae
            shape = (1, self.settings["latent_frames_per_block"], vae.z_dim,
                     self.settings["height"] // vae.upsampling_factor,
                     self.settings["width"] // vae.upsampling_factor)
            for block in schedule["blocks"]:
                begin, count = block["start_frame"], block["frame_count"]
                actions = [r["applied"] for r in schedule["rows"][begin:begin + count]]
                if spec["control_mode"] == "block" and all(v in (0, 1) for v in actions[0]):
                    self.pipeline.set_act(dict(zip(KEY_ORDER, actions[0])),
                                          height=self.settings["height"], width=self.settings["width"],
                                          num_frames=shape[1], device=self.device)
                else:
                    self.pipeline.set_act_sequence(actions, height=self.settings["height"],
                                                   width=self.settings["width"], num_frames=shape[1],
                                                   device=self.device)
                noise = torch.randn(shape, device=self.device, dtype=torch.bfloat16)
                latent = self.pipeline.generate_next_block(noise)
                if latent is None:
                    raise RuntimeError("Pipeline returned no latent block.")
                decoded = self.decode(self.pipeline, latent)
                if len(decoded) != count:
                    raise RuntimeError(f"Block {block['block_index']}: expected {count} decoded frames, "
                                       f"got {len(decoded)}. Refusing misaligned action timestamps.")
                print(f"  block {block['block_index'] + 1}/{len(schedule['blocks'])}", flush=True)
                for offset, frame in enumerate(decoded):
                    if begin + offset < schedule["video_frames"]:
                        yield frame, schedule["rows"][begin + offset]
                del decoded, latent, noise

    def cleanup(self):
        self.pipeline._clear_generation_caches()
        self.pipeline.vae.model.clear_cache()
        self.torch.cuda.empty_cache()


def hud_frame(frame, row: dict, case: dict):
    import numpy as np
    from PIL import Image, ImageDraw

    spec = case["spec"]
    canvas = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 61), fill=(0, 0, 0))
    suffix = " | SAMPLING LIMITED" if not case["schedule"]["sampling"]["fundamental_below_nyquist"] else ""
    keys = " ".join(f"{k}={v:.2f}" for k, v in zip(KEY_ORDER, row["applied"]) if v > 0) or "none"
    draw.text((8, 5), f"{spec['scene']['id']}  {spec['pair']} {spec['waveform']}  "
              f"requested={spec['requested_frequency_hz']:g} Hz  seed={spec['seed']}{suffix}", fill="white")
    draw.text((8, 23), f"t={row['time_s']:.3f}s  {row['segment']}  "
              f"command={row['u_command']:+.3f} applied={row['u_applied']:+.3f}", fill="white")
    draw.text((8, 41), f"applied keys: {keys}", fill="white")
    return np.asarray(canvas)


def render_case(backend, case: dict, directory: Path) -> dict:
    settings = case["spec"]["settings"]
    paths = [directory / "video.mp4"]
    if case["spec"]["annotate"]:
        paths.append(directory / "annotated.mp4")
    partials = [p.with_name(p.stem + ".partial.mp4") for p in paths]
    count = 0
    started = time.perf_counter()
    try:
        with ExitStack() as stack:
            writers = [stack.enter_context(backend.imageio.get_writer(
                str(p), format="FFMPEG", fps=settings["fps"], codec="libx264",
                macro_block_size=1, ffmpeg_params=["-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p"]))
                for p in partials]
            for frame, row in backend.frames(case):
                if tuple(frame.shape) != (settings["height"], settings["width"], 3):
                    raise RuntimeError(f"Unexpected decoded frame shape: {frame.shape}")
                writers[0].append_data(frame)
                if len(writers) > 1:
                    writers[1].append_data(hud_frame(frame, row, case))
                count += 1
        if count != case["schedule"]["video_frames"]:
            raise RuntimeError(f"Expected {case['schedule']['video_frames']} frames, got {count}.")
        for partial in partials:
            # Inspect the encoded artifact, not merely the number of append calls.
            with backend.imageio.get_reader(str(partial), format="FFMPEG") as reader:
                meta = reader.get_meta_data()
                encoded_count = reader.count_frames()
                if encoded_count != count or not math.isclose(meta["fps"], settings["fps"], abs_tol=0.01):
                    raise RuntimeError(f"Encoded video timing mismatch: {partial}")
                if tuple(meta["size"]) != (settings["width"], settings["height"]):
                    raise RuntimeError(f"Encoded video dimensions mismatch: {partial}")
        for partial, final in zip(partials, paths):
            partial.replace(final)
        return {"frame_count": count, "elapsed_s": time.perf_counter() - started,
                "artifacts": {p.name: {"bytes": p.stat().st_size, "sha256": file_digest(p)} for p in paths}}
    finally:
        for partial in partials:
            partial.unlink(missing_ok=True)


def reusable(directory: Path, fingerprint: str) -> dict | None:
    path = directory / "result.json"
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "ok" or result.get("fingerprint") != fingerprint:
            return None
        artifacts = result.get("artifacts", {})
        if "video.mp4" not in artifacts:
            return None
        for name, info in artifacts.items():
            artifact = directory / name
            if not artifact.is_file() or artifact.stat().st_size != info["bytes"] or file_digest(artifact) != info["sha256"]:
                return None
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return None


def write_summary(output: Path, results: list[dict]):
    write_json(output / "results.json", results)
    columns = ["case", "scene", "waveform", "pair", "frequency_hz", "seed", "status", "video",
               "nyquist_hz", "fundamental_below_nyquist", "folded_fundamental_hz", "error"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    cards = []
    for result in results:
        title = html.escape(f"{result['scene']} | {result['pair']} {result['waveform']} | "
                            f"{result['frequency_hz']:g} Hz | seed {result['seed']} | {result['status']}")
        video = html.escape(result.get("video", ""), quote=True)
        player = f'<video controls preload="none" src="{video}"></video>' if video else ""
        trace = html.escape(result["case"] + "/actions.csv", quote=True)
        warning = "Sampling limited: requested frequency is not identifiable." if not result["fundamental_below_nyquist"] else ""
        cards.append(f'<article><h3>{title}</h3><p>{warning}</p>{player}<p><a href="{trace}">Action CSV</a></p></article>')
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Control frequency sweep</title>'
        '<style>body{font:15px system-ui;margin:24px;background:#f6f7f9;color:#18212b}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}'
        'article{background:white;padding:16px;border:1px solid #ddd;border-radius:8px}'
        'video{width:100%}h3{font-size:16px}</style><h1>Control frequency sweep</h1>'
        '<p>Frequency uses decoded-video time. Frame packing and fractional controls are experimental. '
        'These videos alone do not establish a Bode response, geometric-collapse score, or safety boundary.</p>'
        '<main>' + "".join(cards) + '</main></html>', encoding="utf-8")


def execute(args, cases: list[dict], settings: dict, backend_factory=InferenceBackend) -> int:
    output = args.output_dir
    code_paths = [Path(__file__), ROOT / "web_client/pipeline_loader.py", ROOT / "web_client/config.py"]
    for package in ("pipeline", "utils", "wan", "quantizer"):
        code_paths.extend(sorted((ROOT / package).rglob("*.py")))
    protocol = {"version": 1, "cases": [c["spec"] for c in cases], "alias_policy": args.alias_policy,
                "source_sha256": {str(p.relative_to(ROOT)): file_digest(p) for p in code_paths}}
    fingerprint = digest(protocol)
    manifest_path = output / "manifest.json"
    if output.exists() and any(output.iterdir()):
        if not args.resume:
            raise ValueError(f"Output directory is not empty: {output}. Use --resume with the same settings.")
        if not manifest_path.is_file():
            raise ValueError("Cannot resume a nonempty directory without manifest.json.")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Resume settings, source code, or scene resources differ. Use a new output directory.")
    else:
        output.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, {"fingerprint": fingerprint, "created_at": datetime.now(timezone.utc).isoformat(),
                                   "protocol": protocol})
    backend = None
    results = []
    failed = False
    for index, case in enumerate(cases):
        spec, schedule = case["spec"], case["schedule"]
        directory = output / spec["id"]
        directory.mkdir(parents=True, exist_ok=True)
        case_fingerprint = digest({"run": fingerprint, "case": spec})
        previous = reusable(directory, case_fingerprint) if args.resume else None
        item = {"case": spec["id"], "scene": spec["scene"]["label"], "waveform": spec["waveform"],
                "pair": spec["pair"], "frequency_hz": spec["requested_frequency_hz"], "seed": spec["seed"],
                "status": "planned", "video": "", "fingerprint": case_fingerprint, "error": "",
                **schedule["sampling"], "warnings": case["warnings"]}
        write_actions(directory, case)
        write_json(directory / "case.json", {"spec": spec, "warnings": case["warnings"],
                   "schedule": {k: v for k, v in schedule.items() if k != "rows"}})
        print(f"[{index + 1}/{len(cases)}] {spec['id']}", flush=True)
        interrupted = False
        initialization_failed = False
        try:
            if previous:
                item = previous
                print("  resume: verified existing videos", flush=True)
            elif case["skip"]:
                item["status"] = "skipped_sampling_limit"
            elif args.dry_run:
                item["status"] = "dry_run"
            else:
                if backend is None:
                    try:
                        backend = backend_factory(settings)
                        runtime_path = output / "runtime.json"
                        if args.resume and runtime_path.is_file():
                            old_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
                            if old_runtime != backend.runtime:
                                raise RuntimeError("Runtime/checkpoint identity changed; use a new output directory.")
                        write_json(runtime_path, backend.runtime)
                    except Exception:
                        initialization_failed = True
                        raise
                item["status"] = "running"
                write_json(directory / "result.json", item)
                item.update(render_case(backend, case, directory))
                item["status"] = "ok"
                item["video"] = spec["id"] + "/video.mp4"
        except KeyboardInterrupt:
            item.update(status="interrupted", error="Interrupted by user")
            interrupted = True
        except Exception as exc:
            item.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            print(f"  ERROR: {item['error']}", file=sys.stderr, flush=True)
            failed = True
        finally:
            if backend is not None and item["status"] in ("ok", "failed", "interrupted"):
                try:
                    backend.cleanup()
                except Exception as exc:
                    item.setdefault("warnings", []).append(f"Cleanup failed: {exc}")
            write_json(directory / "result.json", item)
            results.append(item)
            write_summary(output, results)
        if interrupted:
            return 130
        # A loader failure applies to the entire sweep; do not retry it dozens of times.
        if failed and (args.fail_fast or initialization_failed):
            break
    print(f"Output: {output}\nSummary: {output / 'summary.csv'}", flush=True)
    return 1 if failed else 0


def main(argv=None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    args.scenes_file = args.scenes_file.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    try:
        scenes = load_scenes(args.scenes_file)
        if args.list_scenes:
            for index, scene in enumerate(scenes):
                print(f"{index:2d}  {scene['id']}  {scene['label']}  {scene['image']}")
            return 0
        selected = []
        if args.scenes == ["all"]:
            selected = scenes
        else:
            for selector in args.scenes:
                matches = [s for i, s in enumerate(scenes) if selector in (str(i), s["id"], s["label"])]
                if len(matches) != 1:
                    raise ValueError(f"Scene selector {selector!r} has {len(matches)} matches. Use --list-scenes.")
                if matches[0] not in selected:
                    selected.append(matches[0])
        if any(not math.isfinite(f) or f <= 0 for f in args.frequencies):
            raise ValueError("Frequencies must be finite and positive.")
        if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
            raise ValueError("Seeds must be in [0, 2**32).")
        settings = experiment_settings(args)
        selected = [scene_resources(s) for s in selected]
        cases = make_cases(args, selected, settings)
        skipped = sum(c["skip"] for c in cases)
        limited = sum(not c["schedule"]["sampling"]["fundamental_below_nyquist"] for c in cases)
        print(f"{len(selected)} scenes, {len(cases)} cases, {len(cases) - skipped} runnable, "
              f"{skipped} sampling-limited cases skipped, {limited - skipped} included as aliasing stress only.\n"
              f"Video clock: {settings['fps']} FPS; controls: {args.control_mode}; "
              f"Nyquist: {cases[0]['schedule']['sampling']['nyquist_hz']:g} Hz.", flush=True)
        if args.control_mode == "frame":
            print("NOTE: within-latent action packing is experimental; constant-key equivalence does not "
                  "prove training-time temporal slot semantics.", flush=True)
        # Existing loader resolves checkpoints and config paths relative to the repository.
        os.chdir(ROOT)
        os.environ.setdefault("PROJECT_ROOT", str(ROOT))
        return execute(args, cases, settings)
    except (ValueError, OSError, ImportError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
