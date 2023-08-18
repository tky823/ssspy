import os
import tempfile

import numpy as np
import pytest
from dummy.io import save_invalid_wavfile
from scipy.io import wavfile

from ssspy import wavread, wavwrite

parameters_frame_offset = [0, 10]
parameters_num_frames = [None, 100]
parameters_channels_first = [True, False, None]
parameters_float = [True, False]
parameters_channels = [1, 2]


@pytest.mark.parametrize("frame_offset", parameters_frame_offset)
@pytest.mark.parametrize("num_frames", parameters_num_frames)
@pytest.mark.parametrize("channels_first", parameters_channels_first)
def test_wavread_monoral(frame_offset: int, num_frames: int, channels_first: bool):
    filename = "./tests/mock/audio/monoral_16k_5sec.wav"

    if channels_first is not None:
        return_2d = True
    else:
        return_2d = False

    # load file using scipy
    sample_rate_scipy, waveform_scipy = wavfile.read(filename)
    waveform_scipy = waveform_scipy / 2**15

    # load file using ssspy
    waveform_ssspy, sample_rate_ssspy = wavread(
        filename,
        frame_offset=frame_offset,
        num_frames=num_frames,
        return_2d=return_2d,
        channels_first=channels_first,
    )

    assert sample_rate_scipy == sample_rate_ssspy

    if return_2d:
        if channels_first:
            waveform_ssspy = waveform_ssspy.squeeze(axis=0)
        else:
            waveform_ssspy = waveform_ssspy.squeeze(axis=1)

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
    filename = "./tests/mock/audio/monoral_16k_5sec.wav"
    max_frame = 5 * 16000
    valid_num_frames = max_frame - frame_offset

    # valid data
    wavread(filename, frame_offset=frame_offset, num_frames=valid_num_frames)

    # invalid memory size
    invalid_num_frames = valid_num_frames + 1

    with pytest.raises(ValueError) as e:
        wavread(filename, frame_offset=frame_offset, num_frames=invalid_num_frames)

    assert str(e.value) == f"num_frames={invalid_num_frames} exceeds maximum frame {max_frame}."


@pytest.mark.parametrize("frame_offset", parameters_frame_offset)
def test_wavread_invalid_stereo(frame_offset: int):
    filename = "./tests/mock/audio/stereo_16k_5sec.wav"
    max_frame = 5 * 16000
    valid_num_frames = max_frame - frame_offset

    # valid data
    wavread(filename, frame_offset=frame_offset, num_frames=valid_num_frames)

    # invalid memory size
    invalid_num_frames = valid_num_frames + 1

    with pytest.raises(ValueError) as e:
        wavread(filename, frame_offset=frame_offset, num_frames=invalid_num_frames)

    assert str(e.value) == f"num_frames={invalid_num_frames} exceeds maximum frame {max_frame}."


@pytest.mark.parametrize("is_float", parameters_float)
def test_wavio_1d(is_float: np.dtype):
    rng = np.random.default_rng(0)

    filename = "valid.wav"
    sample_rate = 16000
    duration = 5
    bits_per_sample = 16
    bytes_per_sample = bits_per_sample // 8
    num_frames = sample_rate * duration
    vmax = 2 ** (bits_per_sample - 1)

    waveform = rng.integers(-vmax, vmax, size=(num_frames,), dtype=f"<i{bytes_per_sample}")

    if is_float:
        waveform = waveform / vmax
        waveform_in = waveform.copy()
    else:
        waveform_in = waveform / vmax

    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, filename)
        wavwrite(path, waveform, sample_rate=sample_rate)
        waveform_out, _ = wavread(path)

    assert np.all(waveform_in == waveform_out)


@pytest.mark.parametrize("is_float", parameters_float)
@pytest.mark.parametrize("n_channels", parameters_channels)
@pytest.mark.parametrize("channels_first", parameters_channels_first)
def test_wavio_2d(is_float: np.dtype, n_channels: int, channels_first: bool):
    rng = np.random.default_rng(0)

    filename = "valid.wav"
    sample_rate = 16000
    duration = 5
    bits_per_sample = 16
    bytes_per_sample = bits_per_sample // 8
    num_frames = sample_rate * duration
    vmax = 2 ** (bits_per_sample - 1)

    if channels_first:
        shape = (n_channels, num_frames)
    else:
        shape = (num_frames, n_channels)

    waveform = rng.integers(
        -vmax,
        vmax,
        size=shape,
        dtype=f"<i{bytes_per_sample}",
    )

    if is_float:
        waveform = waveform / vmax
        waveform_in = waveform.copy()
    else:
        waveform_in = waveform / vmax

    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, filename)
        wavwrite(path, waveform, sample_rate=sample_rate, channels_first=channels_first)
        waveform_out, _ = wavread(path, return_2d=True, channels_first=channels_first)

    assert np.all(waveform_in == waveform_out)


def test_waveread_invalid_metadata():
    filename = "./tests/mock/audio/monoral_16k_5sec_invalid.wav"

    # invalid riff
    save_invalid_wavfile(filename, invalid_riff=True)

    with pytest.raises(NotImplementedError) as e:
        wavread(filename)

    assert str(e.value) == f"Not support {b'RIFX'}."

    # invalid ftype
    save_invalid_wavfile(filename, invalid_ftype=True)

    with pytest.raises(NotImplementedError) as e:
        wavread(filename)

    assert str(e.value) == f"Not support {b'wave'}."

    # invalid fmt chunk marker
    save_invalid_wavfile(filename, invalid_fmt_chunk_marker=True)

    with pytest.raises(NotImplementedError) as e:
        wavread(filename)

    assert str(e.value) == f"Not support {b'FMT '}."

    # invalid fmt chunk size
    save_invalid_wavfile(filename, invalid_fmt_chunk_size=True)

    with pytest.raises(NotImplementedError) as e:
        wavread(filename)

    assert str(e.value) == "Invalid header is detected."

    # invalid fmt
    save_invalid_wavfile(filename, invalid_fmt=True)

    with pytest.raises(NotImplementedError) as e:
        wavread(filename)

    assert str(e.value) == "Invalid header 0 is detected."

    # invalid fmt byte rate
    save_invalid_wavfile(filename, invalid_byte_rate=True)

    with pytest.raises(ValueError) as e:
        wavread(filename)

    assert str(e.value) == "Invalid header is detected."

    # invalid data chunk marker
    save_invalid_wavfile(filename, invalid_data_chunk_marker=True)

    with pytest.raises(NotImplementedError) as e:
        wavread(filename)

    assert str(e.value) == f"Not support {b'DATA'}."

    os.remove(filename)
