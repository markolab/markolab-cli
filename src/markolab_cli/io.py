import os
import toml
import numpy as np
from tqdm.auto import tqdm
from markovids.vid import io


def convert_raw_to_avi(input_filename, output_filename, chunk_size, delete, threads, codec):
    base_filename = os.path.splitext(os.path.basename(input_filename))[0]
    tstamp_filename = f"{base_filename}.txt"
    metadata_filename = os.path.join(os.path.dirname(input_filename), "metadata.toml")

    if output_filename is None:
        output_filename = os.path.join(
            os.path.dirname(input_filename), "{}.avi".format(base_filename)
        )

    metadata = toml.load(metadata_filename)["camera_metadata"][base_filename]
    frame_size = metadata["Width"], metadata["Height"]

    try:
        fps = int(np.round(metadata["AcquisitionFrameRate"]))
    except KeyError:
        timestamps = io.read_timestamps(f"{tstamp_filename}.txt")
        fps = int(np.round((1 / timestamps["device_timestamp"].diff()).median()))

    dtype = io.pixel_format_to_np_dtype(metadata["PixelFormat"])
    if dtype.itemsize == 2:
        pixel_format = "gray16le"
    elif dtype.itemsize == 1:
        pixel_format = "gray"
    else:
        raise RuntimeError(f"Can't map {dtype.itemsize} bytes to pixel_format")

    input_obj = io.AutoReader(filepath=input_filename, frame_size=frame_size, dtype=dtype)
    all_frames = list(range(input_obj.nframes))
    frame_batches = [
        all_frames[idx : idx + chunk_size] for idx in range(0, input_obj.nframes, chunk_size)
    ]

    if not os.path.exists(output_filename):
        output_obj = io.AviWriter(
            filepath=output_filename,
            frame_size=frame_size,
            dtype=dtype,  # check this...
            pixel_format=pixel_format,
            threads=threads,
            codec=codec,
            fps=fps,
        )

        # bail if output exists? maybe not for now...
        for batch in tqdm(frame_batches, desc="Encoding batches"):
            frames = input_obj.get_frames(batch)
            output_obj.write_frames(frames, progress_bar=False)
        output_obj.close()
    else:
        print("Skipping encoding step, file exists...")

    new_input_obj = io.AutoReader(output_filename)

    for batch in tqdm(frame_batches, desc="Checking data integrity"):
        raw_frames = input_obj.get_frames(batch)
        encoded_frames = new_input_obj.get_frames(batch)

        if not np.array_equal(raw_frames, encoded_frames):
            raise RuntimeError(
                "Raw frames and encoded frames not equal from {} to {}".format(batch[0], batch[-1])
            )

    print("Encoding successful")
    if delete:
        print("Deleting {}".format(input_filename))
        os.remove(input_filename)