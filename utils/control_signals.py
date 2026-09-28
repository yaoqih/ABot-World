"""Deterministic control schedules on the decoded-video clock (no GPU imports)."""
from __future__ import annotations

import math

KEY_ORDER = ("W", "A", "S", "D", "I", "J", "K", "L")
PAIRS = {"AD": ("A", "D"), "JL": ("J", "L"),
         "WS": ("W", "S"), "IK": ("I", "K")}
TEMPORAL_FACTOR = 4


def block_layout(total_frames: int, latent_frames: int = 3) -> list[dict]:
    """Cover the requested output length, retaining final block padding."""
    if total_frames < 1 or latent_frames < 1:
        raise ValueError("Frame counts must be positive.")
    blocks = []
    start = 0
    while start < total_frames:
        first = not blocks
        count = TEMPORAL_FACTOR * latent_frames - (3 if first else 0)
        indices = list(range(start, start + count))
        packed = [0] * 3 + indices if first else indices
        blocks.append({"block_index": len(blocks), "start_frame": start,
                       "frame_count": count, "packed_frame_indices": packed})
        start += count
    return blocks


def sampling_info(frequency: float, fps: float, latent_frames: int,
                  control_mode: str) -> dict:
    if control_mode not in ("frame", "latent", "block"):
        raise ValueError(f"Unknown control mode: {control_mode}")
    divisor = {"frame": 1, "latent": 4, "block": 4 * latent_frames}[control_mode]
    sample_rate = fps / divisor
    nyquist = sample_rate / 2
    folded = abs((frequency + nyquist) % sample_rate - nyquist)
    return {
        "control_sample_rate_hz": sample_rate,
        "nyquist_hz": nyquist,
        "fundamental_below_nyquist": frequency < nyquist,
        "folded_fundamental_hz": folded,
        "samples_per_requested_cycle": sample_rate / frequency if frequency else None,
        "frequency_clock": "decoded_video_time_not_wall_clock",
    }


def _signal(t: float, frequency: float, waveform: str, phase_deg: float) -> float:
    cycles = frequency * t + phase_deg / 360
    if waveform in ("square", "stop-go"):
        # Stable, right-continuous switch convention at exact half periods.
        positive = math.floor(2 * cycles + 1e-10) % 2 == 0
        return 1.0 if positive else (0.0 if waveform == "stop-go" else -1.0)
    if waveform == "sine":
        value = math.sin(2 * math.pi * cycles)
        return 0.0 if abs(value) < 1e-12 else value
    if waveform == "hold":
        return 0.0
    raise ValueError(f"Unknown waveform: {waveform}")


def build_schedule(*, fps: int, latent_frames: int, frequency: float,
                   waveform: str, pair: str, control_mode: str,
                   duration: float, warmup: float, cooldown: float,
                   amplitude: float = 1.0, phase_deg: float = 0.0,
                   base_keys: tuple[str, ...] = ()) -> dict:
    if pair not in PAIRS:
        raise ValueError(f"Unknown key pair: {pair}")
    for name, value in (("frequency", frequency), ("duration", duration),
                        ("warmup", warmup), ("cooldown", cooldown),
                        ("amplitude", amplitude), ("phase", phase_deg)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if fps <= 0 or duration <= 0 or min(warmup, cooldown) < 0:
        raise ValueError("FPS/duration must be positive; warmup/cooldown nonnegative.")
    if waveform != "hold" and frequency <= 0:
        raise ValueError("Oscillation frequency must be positive.")
    if not 0 < amplitude <= 1:
        raise ValueError("Amplitude must be in (0, 1].")
    if any(k not in KEY_ORDER for k in base_keys):
        raise ValueError("Unknown base key.")
    if any(set(keys) <= set(base_keys) for keys in PAIRS.values()):
        raise ValueError("Opposing base keys cannot be held together.")
    if waveform != "hold" and set(base_keys) & set(PAIRS[pair]):
        raise ValueError(f"Base keys must not overlap the oscillating pair {pair}.")
    sample = sampling_info(frequency, fps, latent_frames, control_mode)
    warmup_frames = math.ceil(warmup * fps)
    active_frames = math.ceil(duration * fps)
    cooldown_frames = math.ceil(cooldown * fps)
    total = warmup_frames + active_frames + cooldown_frames
    blocks = block_layout(total, latent_frames)
    generated = sum(b["frame_count"] for b in blocks)
    positive_key, negative_key = PAIRS[pair]
    positive_idx, negative_idx = KEY_ORDER.index(positive_key), KEY_ORDER.index(negative_key)
    commands, scalars, segments = [], [], []
    for i in range(generated):
        if i < warmup_frames:
            segment = "warmup"
        elif i < warmup_frames + active_frames:
            segment = "active"
        elif i < total:
            segment = "cooldown"
        else:
            segment = "padding"
        u = (amplitude * _signal((i - warmup_frames) / fps, frequency, waveform, phase_deg)
             if segment == "active" else 0.0)
        action = [float(k in base_keys) for k in KEY_ORDER]
        if waveform != "hold":
            action[positive_idx], action[negative_idx] = max(u, 0.0), max(-u, 0.0)
        commands.append(action)
        scalars.append(u)
        segments.append(segment)
    rows = []
    for block in blocks:
        start = block["start_frame"]
        for i in range(start, start + block["frame_count"]):
            if control_mode == "block":
                sampled_at = start
            elif control_mode == "latent":
                sampled_at = 0 if i == 0 else 1 + ((i - 1) // 4) * 4
            else:
                sampled_at = i
            rows.append({
                "frame_index": i, "time_s": i / fps, "segment": segments[i],
                "written_to_video": i < total, "block_index": block["block_index"],
                "applied_from_frame": sampled_at,
                "u_command": scalars[i], "u_applied": scalars[sampled_at],
                "command": commands[i], "applied": commands[sampled_at],
            })
    return {"video_frames": total, "generated_frames": generated,
            "video_duration_s": total / fps, "warmup_frames": warmup_frames,
            "active_frames": active_frames, "cooldown_frames": cooldown_frames,
            "active_cycles": active_frames * frequency / fps,
            "sampling": sample, "blocks": blocks, "rows": rows}
