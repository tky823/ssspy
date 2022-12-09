import numpy as np
import pytest
from scipy.io import wavfile

from ssspy import wavread

parameters_frame_offset = [0, 10]
parameters_num_frames = [None, 100]
parameters_channels_first = [True, False]


@pytest.mark.parametrize("frame_offset", parameters_frame_offset)
@pytest.mark.parametrize("num_frames", parameters_num_frames)
def test_wavread_monoral(frame_offset: int, num_frames: int):
    filename = "./tests/mock/audio/monoral_16k_5sec.wav"

    # load file using scipy
    sample_rate_scipy, waveform_scipy = wavfile.read(filename)
    waveform_scipy = waveform_scipy / 2**15

    # load file using ssspy
    waveform_ssspy, sample_rate_ssspy = wavread(
        filename, frame_offset=frame_offset, num_frames=num_frames
    )

    assert sample_rate_scipy == sample_rate_ssspy

    if num_frames is not None:
        assert np.all(waveform_scipy[frame_offset : frame_offset + num_frames] == waveform_ssspy)
    else:
        assert np.all(waveform_scipy[frame_offset:] == waveform_ssspy)


@pytest.mark.parametrize("frame_offset", parameters_frame_offset)
@pytest.mark.parametrize("num_frames", parameters_num_frames)
@pytest.mark.parametrize("channels_first", parameters_channels_first)
def test_wavread_stereo(frame_offset: int, num_frames: int, channels_first: bool):
    filename = "./tests/mock/audio/stereo_16k_5sec.wav"

    # load file using scipy
    sample_rate_scipy, waveform_scipy = wavfile.read(filename)
    waveform_scipy = waveform_scipy / 2**15

    # load file using ssspy
    waveform_ssspy, sample_rate_ssspy = wavread(
        filename, frame_offset=frame_offset, num_frames=num_frames, channels_first=channels_first
    )

    assert sample_rate_scipy == sample_rate_ssspy

    if channels_first:
        # same order as that of scipy
        waveform_ssspy = waveform_ssspy.transpose(1, 0)

    if num_frames is not None:
        assert np.all(waveform_scipy[frame_offset : frame_offset + num_frames] == waveform_ssspy)
    else:
        assert np.all(waveform_scipy[frame_offset:] == waveform_ssspy)


@pytest.mark.parametrize("frame_offset", parameters_frame_offset)
def test_wavread_invalid_monoral(frame_offset: int):
    # invalid memory size
    filename = "./tests/mock/audio/monoral_16k_5sec.wav"
    max_frame = 5 * 16000
    valid_num_frames = max_frame - frame_offset
    invalid_num_frames = valid_num_frames + 1

    wavread(filename, frame_offset=frame_offset, num_frames=valid_num_frames)

    with pytest.raises(ValueError) as e:
        wavread(filename, frame_offset=frame_offset, num_frames=invalid_num_frames)

    assert str(e.value) == f"num_frames={invalid_num_frames} exceeds maximum frame {max_frame}."


@pytest.mark.parametrize("frame_offset", parameters_frame_offset)
def test_wavread_invalid_stereo(frame_offset: int):
    # invalid memory size
    filename = "./tests/mock/audio/stereo_16k_5sec.wav"
    max_frame = 5 * 16000
    valid_num_frames = max_frame - frame_offset
    invalid_num_frames = valid_num_frames + 1

    wavread(filename, frame_offset=frame_offset, num_frames=valid_num_frames)

    with pytest.raises(ValueError) as e:
        wavread(filename, frame_offset=frame_offset, num_frames=invalid_num_frames)

    assert str(e.value) == f"num_frames={invalid_num_frames} exceeds maximum frame {max_frame}."
