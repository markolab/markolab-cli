import click
import functools
from tqdm.auto import tqdm
from markolab_cli.io import convert_dat_to_avi


@click.group()
def cli():
    pass


def convert_params(func):
    @click.option("-b", "--chunk-size", type=int, default=3000, help="Chunk size")
    @click.option(
        "--delete", type=bool, is_flag=True, help="Delete raw file if encoding is sucessful"
    )
    @click.option("-t", "--threads", type=int, default=6, help="Number of threads for encoding")
    @click.option("-c", "--codec", type=str, default="ffv1", help="ffmpeg codec")
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def slurm_params(func):
    @click.option("--ncpus", "-n", type=int, default=2, help="Number of CPUs", show_envvar=True)
    @click.option("--memory", "-m", type=str, default="10GB", help="RAM string", show_envvar=True)
    @click.option("--wall-time", "-w", type=str, default="3:00:00", help="Wall time", show_envvar=True)
    @click.option("--qos", type=str, default="inferno", help="QOS name", show_envvar=True)
    @click.option("--prefix", type=str, default=None, help="Command prefix", show_envvar=True)
    @click.option("--account", type=str, default=None, help="Account name", show_envvar=True)
    @click.option("--ngpus", type=int, default=0, help="Number of GPUs to include in request", show_envvar=True)
    @click.option("--gpu-type", type=str, default=None, help="GPU type (e.g. A100, V100)", show_envvar=True)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@cli.command(name="convert-dat-to-avi", context_settings={"show_default": True})
@click.argument("input-filename", type=click.Path(exists=True, resolve_path=True))
@click.option(
    "-o", "--output-filename", type=click.Path(), default=None, help="Path to output file"
)
@convert_params
def convert_dat_to_avi_cli(input_filename, output_filename, chunk_size, delete, threads, codec):
    convert_dat_to_avi(
        input_filename=input_filename,
        output_filename=output_filename,
        chunk_size=chunk_size,
        delete=delete,
        threads=threads,
        codec=codec,
    )


@cli.command(name="convert-dat-to-avi-batch", context_settings={"show_default": True})
@click.option("-d", "--chk-dir", type=click.Path(), default=None, help="Directory to check")
@click.option("-f", "--file-filter", type=str, default="*.dat", help="File filter")
@convert_params
def convert_dat_to_avi_cli_batch(chk_dir, file_filter, chunk_size, delete, threads, codec):
    import os
    import glob

    if chk_dir is None:
        chk_dir = os.getcwd()

    files_proc = glob.glob(os.path.join(chk_dir, "**", file_filter), recursive=True)
    for f in tqdm(files_proc, desc="Total files"):
        convert_dat_to_avi(
            input_filename=f,
            output_filename=None,
            chunk_size=chunk_size,
            delete=delete,
            threads=threads,
            codec=codec,
        )


@cli.command(
    name="create-slurm-batch",
    context_settings={"show_default": True, "auto_envvar_prefix": "MARKOLABCLI_BATCH"},
)
@click.argument("command", type=str)
@click.option("-d", "--chk-dir", type=click.Path(), default=None, help="Directory to check")
@click.option(
    "-f", "--file-filter", type=str, default="*.dat", help="File filter", show_envvar=True
)
@slurm_params
def create_slurm_batch_cli(
    command, chk_dir, file_filter, ncpus, memory, wall_time, qos, prefix, account, ngpus, gpu_type
):
    import os
    import glob

    if chk_dir is None:
        chk_dir = os.getcwd()

    files_proc = glob.glob(os.path.join(chk_dir, "**", file_filter), recursive=True)

    if prefix is not None:
        base_command = f"{prefix};"
    else:
        base_command = ""

    if gpu_type is not None:
        gpu_cmd = f"{gpu_type}:{ngpus}"
    else:
        gpu_cmd = f"{ngpus}"

    cluster_prefix = f'sbatch --gpu-per-node={gpu_cmd} --nodes 1 --ntasks-per-node 1 --cpus-per-task {ncpus:d} --mem={memory} -q {qos} -t {wall_time} -A {account} --wrap "'
    issue_command = f"{cluster_prefix}{base_command}"
    for f in files_proc:
        run_command = f'{issue_command}{command} \\"{f}\\""'
        print(run_command)




@cli.command(
    name="create-slurm",
    context_settings={"show_default": True, "auto_envvar_prefix": "MARKOLABCLI_SLURM"},
)
@click.argument("command", type=str)
@slurm_params
def create_slurm_cli(
    command, ncpus, memory, wall_time, qos, prefix, account, ngpus
):
    
    if prefix is not None:
        base_command = f"{prefix};"
    else:
        base_command = ""

    cluster_prefix = f'sbatch --gres=gpu:{ngpus} --nodes 1 --ntasks-per-node 1 --cpus-per-task {ncpus:d} --mem={memory} -q {qos} -t {wall_time} -A {account} --wrap "'
    issue_command = f"{cluster_prefix}{base_command}"
    print(issue_command)


if __name__ == "__main__":
    cli()
