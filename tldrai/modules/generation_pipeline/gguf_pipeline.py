from llama_cpp import Llama  # Import the Llama class from the llama-cpp-python library


class GGUFModelLoader:
    def __init__(self, model_path, use_gpu=True, chat_format=None, n_ctx=16384, n_threads=8, n_gpu_layers=0,
                 repo_id=None):
        self.repo_id = repo_id
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.chat_format = chat_format
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers if self.use_gpu else 0
        self.model = self.load_model()

    def load_model(self):
        # Initialize the Llama model with appropriate settings
        if self.repo_id is None:
            return Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                chat_format=self.chat_format,  # Assuming chat_format is supported here based on usage example
            )
        else:
            return Llama.from_pretrained(
                repo_id=self.repo_id,
                filename=self.model_path,
                verbose=False
            )

    def predict(self, prompt, max_tokens=512, stop=None, echo=False):
        # Perform inference using the model
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            stop=stop or ["</s>"],
            echo=echo
        )
        return output

    def create_chat_completion(self, messages):
        # Create a chat completion if the Llama model supports this API
        return self.model.create_chat_completion(messages=messages)

    def run(self, prompt, **gen_params):
        return self.model.create_chat_completion(messages=prompt, **gen_params)
