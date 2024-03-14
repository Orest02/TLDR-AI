from transformers import pipeline


class PresetPipeline:
    def __init__(self, model_path, task="text-generation"):
        self.summarizer = pipeline(task, model=model_path, trust_remote_code=True)

    def run(self, prompt, max_new_tokens=250, min_length=3, do_sample=False):
        return self.summarizer(prompt, max_new_tokens=max_new_tokens, min_length=min_length, do_sample=do_sample)