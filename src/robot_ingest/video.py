"""Cut one episode out of a source MP4 (local or remote) and re-encode it to the canonical format.

Output contract, relied on by training loaders and the viewer:
  * exactly `expected_frames` frames, constant frame rate, frame i <-> frames row i
  * H.264, yuv420p, no B-frames, a keyframe every `keyframe_interval` frames, moov atom first
    (plays in every browser, and a random seek never decodes more than keyframe_interval frames)

Remote sources are opened by URL. FFmpeg then issues HTTP range requests: it reads the MP4
index, seeks to the keyframe before the episode, and downloads only that episode's bytes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from av.video.frame import PictureType

from robot_ingest.sources.lerobot import VideoSegment

log = logging.getLogger(__name__)

# Bump when a change in this module alters the files it writes: stored videos are then re-encoded on the next run.
ENCODER_VERSION = "1"
# FFmpeg HTTP options: reconnect on drops, give up on a stalled read after 30 s (microseconds).
HTTP_OPTIONS = {"reconnect": "1", "reconnect_on_network_error": "1", "reconnect_delay_max": "8", "rw_timeout": "30000000"}
THUMBNAIL_MAX_SIDE = 320
LUMA_SAMPLES = 20


class VideoError(Exception):
    """The source video does not hold the frames the episode needs."""


@dataclass(frozen=True)
class TranscodeResult:
    num_frames: int
    width: int
    height: int
    mean_luma: float  # 0-255 over sampled frames; ~0 means the stream is black


def video_fingerprint(segment: VideoSegment, *, fps: float, expected_frames: int, crf: int, keyframe_interval: int) -> str:
    """Everything an output video is made from, hashed. Same fingerprint, same file, so re-encoding can be skipped.

    For a Hugging Face source the URL contains the pinned revision, so a new revision counts as new input.
    """
    inputs = {"encoder": ENCODER_VERSION, "url": segment.url, "start_s": segment.start_s, "end_s": segment.end_s,
              "fps": fps, "frames": expected_frames, "crf": crf, "keyframe_interval": keyframe_interval}
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:16]


def transcode_segment(
    segment: VideoSegment,
    out_path: Path,
    thumbnail_path: Path,
    *,
    fps: float,
    expected_frames: int,
    crf: int,
    keyframe_interval: int,
    attempts: int = 3,
) -> TranscodeResult:
    """Retry wrapper: remote reads fail transiently; a short or corrupt result is also retried once fetched again."""
    for attempt in range(1, attempts + 1):
        try:
            return _transcode(segment, out_path, thumbnail_path, fps, expected_frames, crf, keyframe_interval)
        except Exception as exc:  # noqa: BLE001 - network, demux and decode errors all look different
            out_path.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            log.warning("video %s attempt %d/%d failed (%s); retrying", segment.url, attempt, attempts, exc)
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _transcode(segment, out_path, thumbnail_path, fps, expected_frames, crf, keyframe_interval) -> TranscodeResult:
    remote = segment.url.startswith(("http://", "https://"))
    rate = Fraction(fps).limit_denominator(1001)
    half_frame = 0.5 / fps  # select frames by the nearest frame slot, robust to timestamp rounding
    sample_every = max(1, expected_frames // LUMA_SAMPLES)
    lumas: list[float] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)

    with av.open(segment.url, options=HTTP_OPTIONS if remote else {}) as src:
        stream = src.streams.video[0]
        stream.thread_type = "AUTO"
        width, height = stream.codec_context.width, stream.codec_context.height
        if width % 2 or height % 2:
            raise VideoError(f"odd frame size {width}x{height} is not encodable as yuv420p")
        if segment.start_s > 0:  # lands on the keyframe at or before the start; we decode forward from there
            src.seek(int(segment.start_s / stream.time_base), stream=stream, backward=True)

        with av.open(str(out_path), "w", container_options={"movflags": "+faststart"}) as dst:
            out = dst.add_stream("libx264", rate=rate, options={
                "crf": str(crf),
                "preset": "medium",
                "x264-params": f"keyint={keyframe_interval}:min-keyint={keyframe_interval}:scenecut=0:bframes=0",
            })
            out.width, out.height, out.pix_fmt = width, height, "yuv420p"

            n = 0
            for frame in src.decode(stream):
                if frame.time is None:
                    raise VideoError("decoded frame has no timestamp")
                if frame.time < segment.start_s - half_frame:
                    continue  # between the keyframe we seeked to and the episode start
                if frame.time >= segment.end_s - half_frame or n == expected_frames:
                    break
                if n == expected_frames // 2:
                    image = frame.to_image()
                    image.thumbnail((THUMBNAIL_MAX_SIDE, THUMBNAIL_MAX_SIDE))
                    image.save(thumbnail_path, quality=85)
                if n % sample_every == 0:
                    lumas.append(float(frame.to_ndarray(format="gray").mean()))
                frame = frame.reformat(format="yuv420p")
                frame.pts, frame.time_base = n, 1 / rate
                frame.pict_type = PictureType.NONE  # let x264 place keyframes, not the source's GOP
                dst.mux(out.encode(frame))
                n += 1
            dst.mux(out.encode(None))  # flush

    if n != expected_frames:
        raise VideoError(f"source segment holds {n} frames, episode has {expected_frames}")
    return TranscodeResult(num_frames=n, width=width, height=height, mean_luma=float(np.mean(lumas)) if lumas else 0.0)
