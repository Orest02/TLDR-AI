import ast
import logging
import os

import click
from omegaconf import OmegaConf

from tldrai.run import main

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration

    Args:
        config_path (str): Path to the configuration file

    Returns:
        OmegaConf: Configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)


def apply_overrides(config, overrides):
    """Applies CLI overrides to the configuration

    Args:
        config (OmegaConf): omegaconf configuration to override
        overrides (tuple of str): overrides to apply in the format key=value

    Returns:
        None
    """
    for override in overrides:
        key, value = override.split("=", 1)
        keys = key.split(".")
        cfg_node = config
        for k in keys[:-1]:
            cfg_node = cfg_node[k]
        try:
            parsed_value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed_value = value
        cfg_node[keys[-1]] = parsed_value


def set_logger_level(verbose):
    """Sets logger level

    Args:
        verbose (bool): Debug if True, Info if False

    Returns:
        None
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)


@click.command()
@click.argument("question", nargs=-1)
@click.option(
    "--config",
    type=click.Path(exists=True),
    default="config/ollama_stable_code.yml",
    help="Path to config file",
)
@click.option(
    "--set",
    "overrides",
    multiple=True,
    help="Set configuration key-value pairs (key=value)",
)
@click.option("-v", "--verbose", is_flag=True, help="Output all logs verbose")
@click.option(
    "-s",
    "--search",
    is_flag=True,
    default=False,
    help="Search for similar questions on Stack Overflow",
)
def main_cli(question, config, overrides, verbose, search):
    """Invokes the main runner

    Args:
        question (tuple of str): Question to ask the LLM
        config (str): Path to the configuration file
        overrides (tuple of str): Configuration key-value pairs to override
        verbose (bool): Output all logs verbose if True
        search (bool): Search for similar questions on Stack Overflow if True

    Returns:
        None
    """
    config = load_config(config)

    config.question = " ".join(question)
    if overrides:
        apply_overrides(config, overrides)

    config.verbose = verbose
    config.no_search = not search

    set_logger_level(verbose)

    main(config)


if __name__ == "__main__":
    main_cli()
