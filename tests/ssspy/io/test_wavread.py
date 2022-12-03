import numpy as np
import pytest
from scipy.io import wavfile

from ssspy import wavread

parameters_frame_offset = [0, 10]
parameters_num_frames = [-1, 100]


@pytest.mark.parametrize("frame_offset", parameters_frame_offset)
@pytest.mark.parametrize("num_frames", parameters_num_frames)
def test_wavread(frame_offset: int, num_frames: int):
    mono_filename = "./tests/mock/audio/monoral_16k_5sec.wav"

    # load file using scipy
    sample_rate_scipy, waveform_scipy = wavfile.read(mono_filename)
    waveform_scipy = waveform_scipy / 2**15

    # load file using ssspy
    waveform_ssspy, sample_rate_ssspy = wavread(
        mono_filename, frame_offset=frame_offset, num_frames=num_frames
    )

    assert sample_rate_scipy == sample_rate_ssspy

    if num_frames > 0:
        assert np.all(waveform_scipy[frame_offset : frame_offset + num_frames] == waveform_ssspy)
    else:
        assert np.all(waveform_scipy[frame_offset:] == waveform_ssspy)
