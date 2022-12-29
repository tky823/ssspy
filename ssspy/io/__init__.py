import struct
from io import BufferedReader, BufferedWriter
from typing import Optional, Tuple

import numpy as np


def wavread(
    path: str,
    frame_offset: int = 0,
    num_frames: Optional[int] = None,
    return_2d: Optional[bool] = None,
    channels_first: Optional[bool] = None,
) -> Tuple[np.ndarray, int]:
    with open(path, mode="rb") as f:
        riff = f.read(4)

        # ensure byte order is little endian
        if riff != b"RIFF":
            raise NotImplementedError(f"Not support {repr(riff)}.")

        # total file size
        _ = struct.unpack("<I", f.read(4))[0] + 4 + 4

        ftype = f.read(4)

        # ensure file type is WAV
        if ftype != b"WAVE":
            raise NotImplementedError(f"Not support {repr(ftype)}.")

        chunk_marker = f.read(4)

        if chunk_marker != b"fmt ":
            raise NotImplementedError(f"Not support {repr(chunk_marker)}.")

        n_channels, sample_rate, block_align = _read_fmt_chunk(f)

        chunk_marker = f.read(4)

        if chunk_marker != b"data":
            raise NotImplementedError(f"Not support {repr(chunk_marker)}.")

        data = _read_data_chunk(
            f,
            n_channels,
            block_align,
            frame_offset=frame_offset,
            num_frames=num_frames,
            return_2d=return_2d,
            channels_first=channels_first,
        )

    return data, sample_rate


def wavwrite(
    path: str,
    waveform: np.ndarray,
    sample_rate: int,
    channels_first: Optional[bool] = None,
) -> None:
    assert path[-4:] == ".wav", "Only wav file is supported."

    if waveform.ndim == 1:
        _waveform = waveform
        n_channels = 1
    elif waveform.ndim == 2:
        if channels_first:
            _waveform = waveform.transpose(1, 0)
        else:
            _waveform = waveform

        n_channels = _waveform.shape[1]

        if n_channels < 1 or 2 < n_channels:
            raise ValueError(f"{n_channels}channel-input is not supported.")
    else:
        raise ValueError(f"waveform.ndim should be less or equal to 2, but given {waveform.ndim}.")

    if _waveform.dtype in ["f2", "f4", "f8", "f16"]:
        bits_per_sample = 16

        # float to int
        _waveform = _waveform * 2 ** (bits_per_sample - 1)
        _waveform = _waveform.astype("<i2")
    elif _waveform.dtype == "i1":
        bits_per_sample = 8
    elif _waveform.dtype == "i2":
        bits_per_sample = 16
    else:
        raise ValueError(f"Invalid dtype={_waveform.dtype} is detected.")

    assert (
        bits_per_sample % 8 == 0
    ), f"bits_per_sample should be divisible by 8, but given {bits_per_sample}."

    byte_rate = (bits_per_sample * sample_rate * n_channels) // 8
    block_align = byte_rate // sample_rate

    with open(path, mode="wb") as f:
        valid_file_size = 0

        data = b"RIFF"
        f.write(data)
        valid_file_size += 4

        filesize_position = f.tell()
        data = struct.pack("<I", 0)  # calculate file size at last
        f.write(data)

        data = b"WAVE"
        f.write(data)

        _write_fmt_chunk(f, n_channels, sample_rate, byte_rate, block_align, bits_per_sample)

        _write_data_chunk(f, _waveform)

        total_file_size = f.tell()
        data = struct.pack("<I", total_file_size - 8)
        f.seek(filesize_position)
        f.write(data)


def _read_fmt_chunk(
    f: BufferedReader,
) -> Tuple[int, int, int]:
    fmt_chunk_size = struct.unpack("<I", f.read(4))[0]

    if fmt_chunk_size != 16:
        raise NotImplementedError("Invalid header is detected.")

    fmt = struct.unpack("<H", f.read(2))[0]

    # ensure format is PCM
    if fmt != 1:
        raise NotImplementedError(f"Invalid header {fmt} is detected.")

    n_channels, sample_rate, byte_rate, block_align, bits_per_sample = struct.unpack(
        "<HIIHH", f.read(2 + 4 + 4 + 2 + 2)
    )

    if bits_per_sample * sample_rate * n_channels != 8 * byte_rate:
        raise ValueError("Invalid header is detected.")

    return n_channels, sample_rate, block_align


def _read_data_chunk(
    f: BufferedReader,
    n_channels: int,
    block_align: int,
    frame_offset: int = 0,
    num_frames: Optional[int] = None,
    return_2d: Optional[bool] = None,
    channels_first: Optional[bool] = None,
) -> np.ndarray:
    data_chunk_size = struct.unpack("<I", f.read(4))[0]
    bytes_per_sample = block_align // n_channels
    n_full_samples = data_chunk_size // bytes_per_sample

    start = f.tell() + block_align * frame_offset
    max_frame = data_chunk_size // block_align

    if num_frames is None:
        shape = (n_full_samples - n_channels * frame_offset,)
        end_frame = data_chunk_size // block_align
    elif num_frames >= 0:
        shape = (n_channels * num_frames,)
        end_frame = frame_offset + num_frames
    else:
        raise ValueError(f"Invalid num_frames={num_frames} is given. Set nonnegative integer.")

    if end_frame > max_frame:
        raise ValueError(f"num_frames={num_frames} exceeds maximum frame {max_frame}.")

    data = np.memmap(f, dtype=f"<i{bytes_per_sample}", mode="c", offset=start, shape=shape)

    if n_channels > 1:
        data = data.reshape(-1, n_channels)

        if channels_first:
            data = data.transpose(1, 0)
    else:
        if return_2d:
            data = data.reshape(-1, n_channels)

            if channels_first:
                data = data.transpose(1, 0)

    vmax = 2 ** (8 * bytes_per_sample - 1)

    return data / vmax


def _write_fmt_chunk(
    f: BufferedWriter,
    n_channels: int,
    sample_rate: int,
    byte_rate: int,
    block_align: int,
    bits_per_sample: int,
) -> None:
    data = b"fmt "
    f.write(data)

    data = struct.pack("<I", 16)
    f.write(data)

    data = struct.pack("<H", 1)
    f.write(data)

    data = struct.pack("<HIIHH", n_channels, sample_rate, byte_rate, block_align, bits_per_sample)
    f.write(data)


def _write_data_chunk(f: BufferedWriter, waveform: np.ndarray) -> None:
    data = b"data"
    f.write(data)

    data_chunk_size = waveform.nbytes
    data = struct.pack("<I", data_chunk_size)
    f.write(data)

    _waveform = waveform.flatten()
    data = _waveform.view("b").data
    f.write(data)
