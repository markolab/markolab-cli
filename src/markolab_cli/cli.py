import click
import functools
from copy import deepcopy
from tqdm.auto import tqdm


@click.group()
def cli():
    pass


def convert_params(func):
    # fmt: off
    @click.option("-b", "--chunk-size", type=int, default=3000, help="Chunk size")
    @click.option( "--delete", type=bool, is_flag=True, help="Delete raw file if encoding is sucessful", )
    @click.option( "-t", "--threads", type=int, default=6, help="Number of threads for encoding" )
    @click.option("-c", "--codec", type=str, default="ffv1", help="ffmpeg codec")
    @functools.wraps(func)
    # fmt: on
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def slurm_params(func):
    # fmt: off
    @click.option( "--ncpus", "-n", type=int, default=2, help="Number of CPUs", envvar="MARKOLABCLI_SLURM_NCPUS", show_envvar=True, )
    @click.option( "--memory", "-m", type=str, default="10GB", help="RAM string", envvar="MARKOLABCLI_SLURM_MEM", show_envvar=True, )
    @click.option( "--wall-time", "-w", type=str, default="3:00:00", help="Wall time", envvar="MARKOLABCLI_SLURM_WALLTIME", show_envvar=True, )
    @click.option( "--qos", type=str, default="inferno", help="QOS name", envvar="MARKOLABCLI_SLURM_QOS", show_envvar=True, )
    @click.option( "--prefix", type=str, default=None, help="Command prefix", envvar="MARKOLABCLI_SLURM_PREFIX", show_envvar=True, )
    @click.option( "--suffix", type=str, default=None, help="Command suffix", envvar="MARKOLABCLI_SLURM_SUFFIX", show_envvar=True, )
    @click.option( "--account", type=str, default=None, help="Account name", envvar="MARKOLABCLI_SLURM_ACCOUNT", show_envvar=True, )
    @click.option( "--ngpus", type=int, default=0, help="Number of GPUs to include in request", envvar="MARKOLABCLI_SLURM_NGPUS", show_envvar=True, )
    @click.option("--gpu-type", type=str, default=None, help="GPU type (e.g. A100, V100)", envvar="MARKOLABCLI_SLURM_GPUTYPE", show_envvar=True)
    @click.option("--constraint", type=str, default=None, multiple=True, help="constraint to add")
    @functools.wraps(func)
    # fmt: on
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


# fmt: off
@cli.command(name="convert-dat-to-avi", context_settings={"show_default": True})
@click.argument("input-filename", type=click.Path(exists=True, resolve_path=True))
@click.option("-o", "--output-filename", type=click.Path(), default=None, help="Path to output file" )
@click.option("--force", type=bool, is_flag=True)
# fmt: on
@convert_params
def convert_dat_to_avi_cli(input_filename, output_filename, force, chunk_size, delete, threads, codec):
    from markolab_cli.io import convert_dat_to_avi
    convert_dat_to_avi(
        input_filename=input_filename,
        output_filename=output_filename,
        chunk_size=chunk_size,
        delete=delete,
        threads=threads,
        codec=codec,
        force=force
    )


# fmt: off
@cli.command(name="convert-dat-to-avi-batch", context_settings={"show_default": True})
@click.option("-d", "--chk-dir", type=click.Path(), default=None, help="Directory to check")
@click.option("-f", "--file-filter", type=str, default="*.dat", help="File filter")
# fmt: on
@convert_params
def convert_dat_to_avi_cli_batch(chk_dir, file_filter, chunk_size, delete, threads, codec):
    import os
    import glob
    from markolab_cli.io import convert_dat_to_avi

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


# fmt: off
@cli.command( name="create-slurm-batch", context_settings={"show_default": True, "auto_envvar_prefix": "MARKOLABCLI_BATCH"}, )
@click.argument("command", type=str)
@click.option("-d", "--chk-dir", type=click.Path(), default=None, help="Directory to check")
@click.option("-f", "--file-filter", type=str, default="*.dat", help="File filter", show_envvar=True )
@click.option( "--is-dir", is_flag=True )
# fmt: on
@slurm_params
def create_slurm_batch_cli(
    command, chk_dir, file_filter, is_dir, ncpus, memory, wall_time, qos, prefix, suffix, account, ngpus, gpu_type, constraint
):
    import os
    import glob

    if chk_dir is None:
        chk_dir = os.getcwd()

    files_proc = sorted(glob.glob(os.path.join(chk_dir, "**", file_filter), recursive=True))

    if is_dir:
        files_proc = filter(os.path.isdir, files_proc)

    if prefix is not None:
        base_command = f"{prefix};"
    else:
        base_command = ""

    if (gpu_type is not None) and (ngpus > 0):
        gpu_cmd = f"{gpu_type}:{ngpus}"
    else:
        gpu_cmd = f"{ngpus}"
    

    cluster_prefix = f'sbatch --gpus-per-node={gpu_cmd} --nodes 1 --ntasks-per-node 1 --cpus-per-task {ncpus:d} --mem={memory} -q {qos} -t {wall_time} -A {account} '
    
    try:
        iter(constraint)
    except TypeError as te:
        if constraint is not None:
            constraint = [constraint]

    if constraint is not None:
        for _constraint in constraint:
            cluster_prefix += f'--constraint="{_constraint}" '

    cluster_prefix += '--wrap "'

    issue_command = f"{cluster_prefix}{base_command}"
    for f in files_proc:
        if (suffix is not None) and ("@BASE" in suffix):
            # BASE is in suffix strip out and replace with basename...
            use_suffix = deepcopy(suffix)
            use_suffix.replace("@BASE", os.path.splitext(f)[0]) # if we want to do something with the filename...
            run_command = f'{issue_command}{command} \\"{f}\\" {use_suffix}"'
        elif suffix is not None:
            run_command = f'{issue_command}{command} \\"{f}\\" {use_suffix}"'
        else:
            run_command = f'{issue_command}{command} \\"{f}\\""'
        print(run_command)


# fmt: off
@cli.command( name="create-slurm", context_settings={"show_default": True, "auto_envvar_prefix": "MARKOLABCLI_SLURM"}, )
@click.argument("command", type=str)
# fmt: on
@slurm_params
def create_slurm_cli(command, ncpus, memory, wall_time, qos, prefix, suffix, account, ngpus, gpu_type, constraint):
    if prefix is not None:
        base_command = f"{prefix};"
    else:
        base_command = ""

    if (gpu_type is not None) and (ngpus > 0):
        gpu_cmd = f"{gpu_type}:{ngpus}"
    else:
        gpu_cmd = f"{ngpus}"

    cluster_prefix = f'sbatch --gpus-per-node={gpu_cmd} --nodes 1 --ntasks-per-node 1 --cpus-per-task {ncpus:d} --mem={memory} -q {qos} -t {wall_time} -A {account} '

    try:
        iter(constraint)
    except TypeError as te:
        if constraint is not None:
            constraint = [constraint]

    if constraint is not None:
        for _constraint in constraint:
            cluster_prefix += f'--constraint="{_constraint}" '

    cluster_prefix += '--wrap "'

    issue_command = f"{cluster_prefix}{base_command}"
    if suffix is not None:
        run_command = f'{issue_command}{command}{suffix}"'
    else:
        run_command = f'{issue_command}{command}"'
    print(run_command)


if __name__ == "__main__":
    cli()
