import logging
import ast
import os
import click
from omegaconf import OmegaConf
from tldrai.run import main

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)


def apply_overrides(config, overrides):
    for override in overrides:
        key, value = override.split('=', 1)
        keys = key.split('.')
        cfg_node = config
        for k in keys[:-1]:
            cfg_node = cfg_node[k]
        try:
            parsed_value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed_value = value
        cfg_node[keys[-1]] = parsed_value


def set_logger_level(verbose):
    if verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)


@click.command()
@click.argument('question', nargs=-1)
@click.option('--config', type=click.Path(exists=True), default='config/ollama_stable_code.yml', help='Path to config file')
@click.option('--set', multiple=True, help='Set configuration key-value pairs (key=value)')
@click.option('-v', '--verbose', is_flag=True, help='Output all logs verbose')
@click.option('-s', '--search', is_flag=True, default=False, help='Search for similar questions on Stack Overflow')
def main_cli(question, config, set, verbose, search):
    config = load_config(config)

    config.question = ' '.join(question)
    if set:
        apply_overrides(config, set)

    config.verbose = verbose
    config.no_search = not search

    set_logger_level(verbose)

    main(config)


if __name__ == "__main__":
    main_cli()
