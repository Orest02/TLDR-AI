import pytest
from omegaconf import OmegaConf
from cli import apply_overrides


def test_apply_overrides():
    config = OmegaConf.create({
        'summarization_pipeline': {
            'model': 'default-model',
            'precision': 'float32'
        }
    })
    overrides = ['summarization_pipeline.model=new-model', 'summarization_pipeline.precision=bfloat16']
    apply_overrides(config, overrides)

    assert config.summarization_pipeline.model == 'new-model'
    assert config.summarization_pipeline.precision == 'bfloat16'
