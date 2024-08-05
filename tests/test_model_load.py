from omegaconf import OmegaConf

from tldrai.modules.generation_pipeline.ollama_pipeline import OllamaPipeline


def test_ollama_pipeline_load_model(mocker):
    config = OmegaConf.create({"model": "stable-code"})

    mocker.patch("ollama.list", return_value={"models": [{"name": "stable-code"}]})
    mocker.patch("ollama.pull", return_value=None)

    pipeline = OllamaPipeline(model=config.model)

    assert pipeline.model == "stable-code"
