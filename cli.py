import argparse
import logging
import ast
import os
from omegaconf import OmegaConf
from tldrai.run import main

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='TLDR AI CLI Tool')
    parser.add_argument('question', nargs='+', help='Question to process')
    parser.add_argument('--config', type=str, default='config/ollama_stable_code.yml', help='Path to config file')
    parser.add_argument('--set', nargs='+', help='Set configuration key-value pairs (key=value)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Output all logs verbose')
    parser.add_argument('-s', '--search', action='store_true', help='Search for similar quetions on Stack Overflow')
    return parser.parse_args()


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


def main_cli():
    args = parse_args()
    config = load_config(args.config)

    config.question = ' '.join(args.question)
    if args.set:
        apply_overrides(config, args.set)

    config.verbose = args.verbose

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    main(config)


if __name__ == "__main__":
    main_cli()
