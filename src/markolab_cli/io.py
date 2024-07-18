from typing import Optional

def convert_dat_to_avi(
    input_filename: str,
    output_filename: Optional[str] = None,
    chunk_size: int = 3000,
    delete: bool = False,
    threads: int = 8,
    codec: str = "ffv1",
    force: bool = False,
) -> None:
    
    import os
    import toml
    import numpy as np
    from tqdm.auto import tqdm
    from markovids.vid import io
    
    base_filename = os.path.splitext(input_filename)[0]
    camera_name = os.path.basename(base_filename)
    tstamp_filename = f"{base_filename}.txt"
    metadata_filename = os.path.join(os.path.dirname(input_filename), "metadata.toml")

    if output_filename is None:
        output_filename = os.path.join(
            os.path.dirname(input_filename), "{}.avi".format(base_filename)
        )

    try:
        metadata = toml.load(metadata_filename)["camera_metadata"][camera_name]
    except KeyError:
        print(f"Could not locate metadata for {camera_name}")
        return None
    frame_size = metadata["Width"], metadata["Height"]

    try:
        fps = int(np.round(metadata["AcquisitionFrameRate"]))
    except KeyError:
        timestamps = io.read_timestamps(tstamp_filename)
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
    # convert to ranges for faster reading/writing...
    frame_batches = [range(batch[0], batch[-1] + 1) for batch in frame_batches]

    if not os.path.exists(output_filename) or force:
        output_obj = io.AviWriter(
            filepath=output_filename,
            frame_size=frame_size,
            dtype=dtype,  # type: ignore
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
    if new_input_obj.nframes != input_obj.nframes:
        raise RuntimeError(
            f"Original data has {input_obj.nframes} new file has {new_input_obj.nframes}"
        )

    for batch in tqdm(frame_batches, desc="Checking data integrity"):
        raw_frames = input_obj.get_frames(batch)
        encoded_frames = new_input_obj.get_frames(batch)

        if not np.array_equal(raw_frames, encoded_frames):  # type: ignore
            raise RuntimeError(
                "Raw frames and encoded frames not equal from {} to {}".format(batch[0], batch[-1])
            )

    input_obj.close()
    new_input_obj.close()

    print("Encoding successful")
    if delete:
        print("Deleting {}".format(input_filename))
        os.remove(input_filename)
