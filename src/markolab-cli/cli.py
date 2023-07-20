import os
import click

@click.group()
def cli():
    pass


@cli.command(name="convert-dat-to-avi")
@click.argument('input-file', type=click.Path(exists=True, resolve_path=True))
@click.option('-o', '--output-file', type=click.Path(), default=None, help='Path to output file')
@click.option('-b', '--chunk-size', type=int, default=3000, help='Chunk size')
@click.option('--fps', type=float, default=30, help='Video FPS')
@click.option('--delete', type=bool, is_flag=True, help='Delete raw file if encoding is sucessful')
@click.option('-t', '--threads', type=int, default=3, help='Number of threads for encoding')
def convert_raw_to_avi(input_file, output_file, chunk_size, fps, delete, threads):

    if output_file is None:
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(os.path.dirname(input_file),
                                   '{}.avi'.format(base_filename))

    vid_info = get_movie_info(input_file)
    frame_batches = list(gen_batch_sequence(vid_info['nframes'], chunk_size, 0))
    video_pipe = None

    for batch in tqdm.tqdm(frame_batches, desc='Encoding batches'):
        frames = load_movie_data(input_file, batch)
        video_pipe = write_frames(output_file,
                                  frames,
                                  pipe=video_pipe,
                                  close_pipe=False,
                                  threads=threads,
                                  fps=fps)

    if video_pipe:
        video_pipe.stdin.close()
        video_pipe.wait()

    for batch in tqdm.tqdm(frame_batches, desc='Checking data integrity'):
        raw_frames = load_movie_data(input_file, batch)
        encoded_frames = load_movie_data(output_file, batch)

        if not np.array_equal(raw_frames, encoded_frames):
            raise RuntimeError('Raw frames and encoded frames not equal from {} to {}'.format(batch[0], batch[-1]))

    print('Encoding successful')

    if delete:
        print('Deleting {}'.format(input_file))
        os.remove(input_file)