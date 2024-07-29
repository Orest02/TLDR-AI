import logging

import ollama
import time
import threading
import itertools

class OllamaPipeline:
    def __init__(self, model, stream_responses=True):
        self.model = model + ":latest"
        self.stream_responses = stream_responses
        self.check_and_pull_model()

    def check_and_pull_model(self):
        models = ollama.list()['models']
        if self.model not in [m['name'] for m in models]:
            logging.info(f"Pulling model {self.model}...")
            ollama.pull(self.model)
            logging.info(f"Model {self.model} pulled successfully.")
        else:
            logging.debug(f"Model {self.model} is already available.")

        logging.info("Loading model into memory..")

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
            options=gen_params
        )
        response = ''
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
            response += chunk['message']['content']
        return response, len(response.split()), (None, len(response.split()))

    def _generate_with_animation(self, prompt, **gen_params):
        done = False

        def animate():
            for c in itertools.cycle(['|', '/', '-', '\\']):
                if done:
                    break
                print(f'\rGenerating response... {c}', end='', flush=True)
                time.sleep(0.1)

        t = threading.Thread(target=animate)
        t.start()

        response = ollama.chat(
            model=self.model,
            messages=prompt,
            stream=False,
            options=gen_params
        )['message']['content']

        done = True
        t.join()
        print('\r' + ' ' * 30, end='\r', flush=True)
        print(response)
        return response, len(response.split()), (None, len(response.split()))
