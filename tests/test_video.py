"""Frame-accurate cutting: the property training depends on (video frame i == frames row i)."""

from fractions import Fraction

import av
import numpy as np
import pytest

from robot_ingest.sources.lerobot import VideoSegment
from robot_ingest.video import VideoError, transcode_segment

FPS = 10
W, H = 64, 48


def gray_level(i: int) -> int:
    return 8 * i + 4  # frame i is a flat gray image of this brightness, so we can identify it after decoding


@pytest.fixture
def concatenated_video(tmp_path):
    """30 frames at 10 fps, like a LeRobot v3 file holding three 1-second episodes back to back."""
    path = tmp_path / "source.mp4"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=FPS, options={"crf": "0", "x264-params": "keyint=2:bframes=0"})
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        for i in range(30):
            frame = av.VideoFrame.from_ndarray(np.full((H, W, 3), gray_level(i), np.uint8), format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, FPS)
            out.mux(stream.encode(frame))
        out.mux(stream.encode(None))
    return path


def decoded_levels(path):
    with av.open(str(path)) as c:
        return [int(round(f.to_ndarray(format="gray").mean())) for f in c.decode(video=0)]


def test_cuts_exactly_one_episode(concatenated_video, tmp_path):
    segment = VideoSegment(str(concatenated_video), start_s=1.0, end_s=2.0, codec="h264")
    result = transcode_segment(segment, tmp_path / "ep.mp4", tmp_path / "ep.jpg", fps=FPS, expected_frames=10,
                               crf=0, keyframe_interval=10)

    assert result.num_frames == 10
    levels = decoded_levels(tmp_path / "ep.mp4")
    assert len(levels) == 10
    np.testing.assert_allclose(levels, [gray_level(i) for i in range(10, 20)], atol=2)  # frames 10..19, in order
    assert (tmp_path / "ep.jpg").exists()


def test_short_source_is_an_error(concatenated_video, tmp_path):
    segment = VideoSegment(str(concatenated_video), start_s=2.0, end_s=3.0, codec="h264")
    with pytest.raises(VideoError):
        transcode_segment(segment, tmp_path / "ep.mp4", tmp_path / "ep.jpg", fps=FPS, expected_frames=15,
                          crf=0, keyframe_interval=10, attempts=1)
