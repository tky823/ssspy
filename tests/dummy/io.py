import os
import struct

import numpy as np


def save_invalid_wavfile(
    path: str,
    invalid_riff: bool = False,
    invalid_ftype: bool = False,
    invalid_fmt_chunk_marker: bool = False,
    invalid_fmt_chunk_size: bool = False,
    invalid_fmt: bool = False,
    invalid_byte_rate: bool = False,
    invalid_data_chunk_marker: bool = False,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    n_channels = 1
    sample_rate = 16000
    bits_per_sample = 16
    duration = 5
    byte_rate = (bits_per_sample * sample_rate * n_channels) // 8
    block_align = byte_rate // sample_rate
    total_file_size = byte_rate * duration + 44

    rng = np.random.default_rng(42)
    num_frames = sample_rate * duration
    bytes_per_sample = block_align // n_channels
    vmax = 2 ** (bits_per_sample - 1)

    valid_file_size = 0

    with open(path, mode="wb") as f:
        if invalid_riff:
            data = b"RIFX"
        else:
            data = b"RIFF"

        f.write(data)
        valid_file_size += 4

        data = struct.pack("<I", total_file_size - 4 - 4)
        f.write(data)
        valid_file_size += 4

        if invalid_ftype:
            data = b"wave"
        else:
            data = b"WAVE"

        f.write(data)
        valid_file_size += 4

        if invalid_fmt_chunk_marker:
            data = b"FMT "
        else:
            data = b"fmt "

        f.write(data)
        valid_file_size += 4

        if invalid_fmt_chunk_size:
            data = struct.pack("<I", 15)
        else:
            data = struct.pack("<I", 16)

        f.write(data)
        valid_file_size += 4

        if invalid_fmt:
            data = struct.pack("<H", 0)
        else:
            data = struct.pack("<H", 1)

        f.write(data)
        valid_file_size += 2

        if invalid_byte_rate:
            data = struct.pack(
                "<HIIHH", n_channels, sample_rate, byte_rate + 1, block_align, bits_per_sample
            )
        else:
            data = struct.pack(
                "<HIIHH", n_channels, sample_rate, byte_rate, block_align, bits_per_sample
            )

        f.write(data)
        valid_file_size += 2 + 4 + 4 + 2 + 2

        if invalid_data_chunk_marker:
            data = b"DATA"
        else:
            data = b"data"

        f.write(data)
        valid_file_size += 4

        data_chunk_size = num_frames * n_channels * block_align
        data = struct.pack("<I", data_chunk_size)
        f.write(data)
        valid_file_size += 4

        waveform = rng.integers(-vmax, vmax, size=(num_frames,), dtype=f"<i{bytes_per_sample}")
        data = waveform.view("b").data
        f.write(data)
        valid_file_size += byte_rate * duration

        assert valid_file_size == total_file_size
