import ollama

class OllamaPipeline:
    def __init__(self, model):
        self.model = model
        self.check_and_pull_model()

    def check_and_pull_model(self):
        try:
            models = ollama.list()
            if self.model not in [model['name'] for model in models['models']]:
                self.pull_model()
        except ollama._types.ResponseError as e:
            print(f"Error checking model: {e}")
            self.pull_model()

    def pull_model(self):
        try:
            print(f"Pulling model {self.model}...")
            response = ollama.pull(self.model)
            if response['status'] == 'success':
                print(f"Model {self.model} pulled successfully.")
            else:
                print(f"Failed to pull model {self.model}: {response}")
        except ollama._types.ResponseError as e:
            print(f"Error pulling model: {e}")

    def run(self, prompt, max_new_tokens=200, do_sample=True, temperature=0.5, top_p=0.95, use_cache=True,
            repetition_penalty=1.1, **kwargs):
        response = ollama.chat(
            model=self.model,
            messages=prompt,
            stream=False,  # Set to True if you want streaming responses
            options={
                'max_new_tokens': max_new_tokens,
                'do_sample': do_sample,
                'temperature': temperature,
                'top_p': top_p,
                'use_cache': use_cache,
                'repetition_penalty': repetition_penalty,
                **kwargs
            }
        )
        content = response['message']['content']
        input_len = len(prompt)
        token_shape = (None, len(content))  # Adjust token_shape to be a tuple

        return [content], input_len, token_shape
