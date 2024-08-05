from omegaconf import OmegaConf

from tldrai.modules.generation_pipeline.ollama_pipeline import OllamaPipeline


def test_ollama_pipeline_load_model(mocker):
    config = OmegaConf.create(
        {
            "model": "stable-code",
        }
    )

    mocker.patch("ollama.list", return_value={"models": [{"name": "stable-code"}]})
    mocker.patch("ollama.pull", return_value=None)

    pipeline = OllamaPipeline(model=config.model)

    assert pipeline.model == "stable-code"


def test_ollama_pipeline_generate_response(mocker):
    config = OmegaConf.create(
        {"model": "stable-code", "generation_params": {"max_new_tokens": 100}}
    )

    mocker.patch("ollama.list", return_value={"models": [{"name": "stable-code"}]})
    mocker.patch("ollama.pull", return_value=None)
    mocker.patch(
        "ollama.chat",
        return_value=iter(
            [
                {"message": {"content": "This "}},
                {"message": {"content": "is "}},
                {"message": {"content": "a "}},
                {"message": {"content": "generated "}},
                {"message": {"content": "response."}},
            ]
        ),
    )

    pipeline = OllamaPipeline(model=config.model)
    prompt = [{"role": "user", "content": "Test prompt"}]
    response, input_len, token_shape = pipeline.run(prompt, **config.generation_params)

    assert response == "This is a generated response."
    assert input_len == 11
    assert token_shape == (None, 40)


def test_ollama_pipeline_generate_with_animation(mocker):
    config = OmegaConf.create(
        {
            "model": "stable-code",
            "stream_responses": False,
            "generation_params": {"max_new_tokens": 100},
        }
    )

    mocker.patch("ollama.list", return_value={"models": [{"name": "stable-code"}]})
    mocker.patch("ollama.pull", return_value=None)
    mocker.patch(
        "ollama.chat",
        return_value={"message": {"content": "This is a generated response."}},
    )

    pipeline = OllamaPipeline(model=config.model)
    prompt = [{"role": "user", "content": "Test prompt"}]
    response, input_len, token_shape = pipeline._generate_with_animation(
        prompt, **config.generation_params
    )

    assert response == "This is a generated response."
    assert input_len == 11
    assert token_shape == (None, 40)
