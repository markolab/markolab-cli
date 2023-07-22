import click
from markolab_cli.io import convert_raw_to_avi


@click.group()
def cli():
    pass


@cli.command(name="convert-raw-to-avi", context_settings={"show_default": True})
@click.argument("input-filename", type=click.Path(exists=True, resolve_path=True))
@click.option(
    "-o", "--output-filename", type=click.Path(), default=None, help="Path to output file"
)
@click.option("-b", "--chunk-size", type=int, default=3000, help="Chunk size")
@click.option("--delete", type=bool, is_flag=True, help="Delete raw file if encoding is sucessful")
@click.option("-t", "--threads", type=int, default=6, help="Number of threads for encoding")
@click.option("-c", type=str, default="ffv1", help="ffmpeg codec")
def convert_raw_to_avi_cli(input_filename, output_filename, chunk_size, delete, threads, codec):
    convert_raw_to_avi(input_filename, output_filename, chunk_size, delete, threads, codec)


if __name__ == '__main__':
    cli()