import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ConstructedPipeline:
    def __init__(self, model_path, device="cuda", chat_template=None):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(
            device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)

        if chat_template is not None:
            self.change_chat_template(chat_template)

        self.chat_template = chat_template

    def change_chat_template(self, jinja_template_path='./chat_templates/chat_templates/chatml.jinja'):
        chat_template = open(jinja_template_path).read()
        chat_template = chat_template.replace('    ', '').replace('\n', '')
        self.tokenizer.chat_template = chat_template

    def run(self, prompt, max_new_tokens=200, do_sample=True, temperature=0.5, top_p=0.95, use_cache=True, repetition_penalty=1.1):
        if self.chat_template is not None:
            prompt = self.tokenizer.apply_chat_template(
                # summarization_input,
                prompt,
                tokenize=False,
                add_generation_prompt=True)

        new_prompt_len = len(prompt)

        inputs = self.tokenizer(
            # summarization_input,
            prompt,
            return_tensors="pt", return_attention_mask=False).to('cuda')

        outputs = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache,
            repetition_penalty=repetition_penalty,
            # can try to uncomment this if model doesn't end
            # eos_token_id=tokenizer.eos_token_id
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=False), new_prompt_len, outputs.shape
