import itertools
import logging
import threading
import time

import ollama

from tldrai.modules.utils.logging import configure_logging


class OllamaPipeline:
    def __init__(self, model, verbose=False, stream_responses=True, keep_alive="5m"):
        self.model = model
        self.verbose = verbose
        self.stream_responses = stream_responses
        self.keep_alive = keep_alive
        configure_logging(logging.DEBUG if self.verbose else logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"Initializing OllamaPipeline with model {self.model}")

        if not self.is_model_pulled(self.model):
            self.pull_model(self.model)

    def is_model_pulled(self, model_name):
        models = ollama.list()["models"]
        if ":" not in model_name:
            model_name += ":latest"
        return any(model["name"] == model_name for model in models)

    def pull_model(self, model_name):
        self.logger.info(f"Pulling model {model_name}...")
        ollama.pull(model_name)
        self.logger.info(f"Model {model_name} pulled successfully.")

    def run(self, prompt, **gen_params):
        if self.stream_responses:
            return self._stream_response(prompt, **gen_params)
        else:
            return self._generate_with_animation(prompt, **gen_params)

    def _stream_response(self, prompt, **gen_params):
        stream = ollama.chat(
            model=self.model,
            messages=prompt,
            stream=True,
            keep_alive=self.keep_alive,
            options=gen_params,
        )
        response = ""
        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
            response += chunk["message"]["content"]
        input_len = sum([len(x["content"]) for x in prompt])
        token_shape = (None, input_len + len(response))
        return response, input_len, token_shape

    def _generate_with_animation(self, prompt, **gen_params):
        done = False

        def animate():
            for c in itertools.cycle(["|", "/", "-", "\\"]):
                if done:
                    break
                print(f"\rGenerating response... {c}", end="", flush=True)
                time.sleep(0.1)

        t = threading.Thread(target=animate)
        t.start()

        response = ollama.chat(
            model=self.model,
            messages=prompt,
            stream=False,
            keep_alive=self.keep_alive,
            options=gen_params,
        )["message"]["content"]

        done = True
        t.join()
        print("\r" + " " * 30, end="\r", flush=True)
        print(response)

        input_len = sum([len(x["content"]) for x in prompt])
        token_shape = (None, input_len + len(response))
        return response, input_len, token_shape
