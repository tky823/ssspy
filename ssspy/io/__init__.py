import struct
from typing import Optional, Tuple

import numpy as np


def wavread(
    filename: str,
    frame_offset: int = 0,
    num_frames: Optional[int] = None,
    return_2d: Optional[bool] = None,
    channels_first: Optional[bool] = None,
) -> Tuple[np.ndarray, int]:
    with open(filename, mode="rb") as f:
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

        fmt_chunk_size = struct.unpack("<I", f.read(4))[0]

        if fmt_chunk_size != 16:
            raise NotImplementedError("Invalid header is detected.")

        fmt = struct.unpack("<H", f.read(2))[0]

        # ensure format is PCM
        if fmt != 1:
            raise NotImplementedError(f"Invalid header {fmt} is detected.")

        n_channels = struct.unpack("<H", f.read(2))[0]
        sample_rate = struct.unpack("<I", f.read(4))[0]
        byte_rate = struct.unpack("<I", f.read(4))[0]
        block_align = struct.unpack("<H", f.read(2))[0]
        bits_per_sample = struct.unpack("<H", f.read(2))[0]

        if bits_per_sample * sample_rate * n_channels != 8 * byte_rate:
            raise ValueError("Invalid header is detected.")

        chunk_marker = f.read(4)

        if chunk_marker != b"data":
            raise NotImplementedError(f"Not support {repr(chunk_marker)}.")

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

    return data / vmax, sample_rate
