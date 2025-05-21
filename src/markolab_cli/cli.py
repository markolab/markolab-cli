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
@click.option("-r", "--recursive", is_flag=True)
# fmt: on
@slurm_params
def create_slurm_batch_cli(
    command, chk_dir, file_filter, is_dir, ncpus, memory, wall_time, qos, prefix, suffix, account, ngpus, gpu_type, constraint, recursive
):
    import os
    import glob

    if chk_dir is None:
        chk_dir = os.getcwd()

    if recursive:
        files_proc = sorted(glob.glob(os.path.join(chk_dir, "**", file_filter), recursive=recursive))
    else:
        files_proc = sorted(glob.glob(os.path.join(chk_dir, file_filter), recursive=recursive))

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
        if (suffix is not None) and ("@base" in suffix.lower()):
            # BASE is in suffix strip out and replace with basename...
            use_suffix = deepcopy(suffix)
            use_suffix = use_suffix.replace("@BASE", os.path.splitext(f)[0]) # if we want to do something with the filename...
            use_suffix = use_suffix.replace("@base", os.path.splitext(f)[0]) # if we want to do something with the filename...
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

# TODO:
# 1. batch this out, call grep to find jobs that timed out...
# fmt: off
@cli.command( name="create-sleap-resume-cmd")
@click.argument("job_id", type=int)
@click.option("--no-checkpoint", "-n", type=bool, is_flag=True, help="Skip adding base_checkpoint option")
@click.option("--update-lr", "-u", type=bool, is_flag=True, help="Update json config with latest learning rate")
def create_sleap_resume_cmd(job_id, no_checkpoint, update_lr):
    return _create_sleap_resume_cmd(job_id, no_checkpoint, update_lr)


def _create_sleap_resume_cmd(job_id, no_checkpoint, update_lr):
    import json
    import re
    import os
    import subprocess

    cmd_output = subprocess.check_output(f"sacct -B -j {job_id}", shell=True)
    sbatch_script = cmd_output.decode().splitlines()[5]
    match = re.search("sleap\-train \"(.*?)\"", sbatch_script)
    json_file = match.group(1)

    with open(json_file, "r") as f:
        config = json.load(f)

    config_output = config["outputs"]
    run_name = config_output["run_name_prefix"] + config_output["run_name"] + config_output["run_name_suffix"]
    model_filename = os.path.join(config_output["runs_folder"], run_name)
    if (not no_checkpoint) and (not os.path.exists(model_filename)):
        return None

    # strip out only the sbatch stuff...
    if ("base_checkpoint" not in sbatch_script) and (not no_checkpoint):
        new_sbatch_script = sbatch_script + f' --base_checkpoint "{model_filename}"'
    else:
        new_sbatch_script = sbatch_script

    new_sbatch_script = new_sbatch_script.replace('"','\\\"')
    cmd_output = subprocess.check_output(f"sacct --format=submitline%10000 --noheader -j {job_id}", shell=True)
    submitline = cmd_output.decode().splitlines()[0].split("--wrap")[0].lstrip()
    submitline = re.sub(r'--constraint=(\S+)',r'--constraint="\1"', submitline)
    
    print(submitline + f'--wrap "{new_sbatch_script}"')
    # print(new_sbatch_script)
    learning_rate = None

    logfile = f"slurm-{job_id}.out"
    if os.path.exists(logfile):
        with open(logfile, "r") as f:
            lines = f.readlines()

        for _line in lines:
                if "learning rate" in _line.lower():
                    _tmp = re.search("to (.*?)\.$", _line)
                    if _tmp is not None:
                        learning_rate = float(_tmp.group(1))    
        # print(f"Found final learning rate: {learning_rate}")
    elif update_lr:
        raise RuntimeError(f"Did not find logfile {logfile}")

    if update_lr and (learning_rate is not None):
        config["optimization"]["initial_learning_rate"] = learning_rate
        with open(json_file, "w") as f:
            json.dump(config, f, indent=4, sort_keys=False)
    elif update_lr:
        raise RuntimeError(f"Learning rate not parsed from {logfile}")
    

# TODO:
# 1. check for multiple instances with the same model and only use the last one...
# fmt: off
@cli.command( name="create-sleap-resume-batch")
@click.option("--file-filter", "-f", default="slurm*.out", type=str)
@click.option("--pattern", "-p", default="slurmstepd", type=str)
@click.option("-d", "--chk-dir", type=click.Path(), default=None, help="Directory to check")
@click.option("--update-lr", "-u", type=bool, is_flag=True, help="Update json config with latest learning rate")
def create_sleap_resume_batch(file_filter, pattern, chk_dir, update_lr):
    import os
    import glob
    import re

    if chk_dir is None:
        chk_dir = os.getcwd()

    proc_files = sorted(glob.glob(os.path.join(chk_dir, "**", file_filter), recursive=True))
    batch_files = {}

    for _file in proc_files:
        to_add = False
        key = None
        with open(_file, "r") as f:
            lines = f.read().splitlines()
            for _line in lines:
                # maybe get the model or directory...or some identifier here?
                if pattern in _line:
                    to_add = True
                if '"filename": ' in _line:
                    key = _line.split(": ")[1][1:-1] # removes double quotes at start/stop
            if to_add and (key is not None):
                batch_files[key] = _file
            elif (not to_add) and (key in batch_files.keys()):
                del batch_files[key] # remove the entry if a later run was successful (or is in progress)
    for _file in batch_files.values():
        try:
            job_id = int(re.search(r".*\-([0-9]+)\.out$", _file).group(1))
            _create_sleap_resume_cmd(job_id, update_lr)
        except AttributeError as e:
            pass






if __name__ == "__main__":
    cli()
