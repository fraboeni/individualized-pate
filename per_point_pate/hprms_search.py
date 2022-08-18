from loguru import logger
from os import makedirs
from pathlib import Path
import click
import yaml

from per_point_pate.models.pytorch.pytorch_model import ARCHITECTURES
from per_point_pate.main import run_pate

# TODO: unify this with new run_experiment_set from main.py


@click.command()
@click.option('--architectures',
              '-a',
              type=click.Choice(ARCHITECTURES.keys()),
              multiple=True)
@click.option('--learning_rates', '-l', type=float, multiple=True)
@click.option('--teacher_epochs', '-e', type=int, multiple=True)
@click.option('--template_path', '-t', type=click.Path(exists=True))
@click.option('--out_dir', '-o', type=click.Path(exists=True))
def generate_prms(architectures, learning_rates, teacher_epochs, template_path,
                  out_dir):
    for arch in architectures:
        for lr in learning_rates:
            for epochs in teacher_epochs:
                with open(template_path, 'r') as f:
                    prms = yaml.safe_load(f)
                    prms['models']['architecture'] = arch
                    prms['models']['lr'] = lr
                    prms['models']['teacher_epochs'] = epochs
                    path = Path(out_dir) / f"{arch}_lr_{lr}_epochs_{epochs}"
                    makedirs(path, exist_ok=True)
                    with open(path / 'parameters.yaml', 'w') as f:
                        yaml.dump(prms, f, default_flow_style=None)


@click.command()
@click.option('--root_dir', '-r', type=click.Path(dir_okay=True))
@click.option('--data_dir', '-d', type=click.Path(dir_okay=True))
@click.option('--n_reduce_teachers', '-n', type=int)
def hprms_search(root_dir, data_dir, n_reduce_teachers):

    folders = [f for f in Path(root_dir).iterdir() if f.is_dir()]

    for f in folders:
        logger.debug(f)
        f = f.resolve()
        params_path = f / 'parameters.yaml'

        logger.debug(f'starting run for hprms search')
        run_pate(params_path=params_path,
                 data_dir=data_dir,
                 out_dir=f,
                 n_reduce_teachers=n_reduce_teachers)


if __name__ == "__main__":
    hprms_search()
