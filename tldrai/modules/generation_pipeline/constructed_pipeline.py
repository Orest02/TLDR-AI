import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ConstructedPipeline:
    def __init__(self, model_path, device="cuda", chat_template=None, dtype='bfloat16'):
        dtype = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                          torch_dtype=dtype).to(
            device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype)

        if chat_template is not None:
            if chat_template != 'default':
                self.change_chat_template(chat_template)

        self.chat_template = chat_template

    def change_chat_template(self, jinja_template_path='./chat_templates/chat_templates/chatml.jinja'):
        chat_template = open(jinja_template_path).read()
        chat_template = chat_template.replace('    ', '').replace('\n', '')
        self.tokenizer.chat_template = chat_template

    def run(self, prompt, max_new_tokens=200, do_sample=True, temperature=0.5, top_p=0.95, use_cache=True,
            repetition_penalty=1.1, attention_mask=False, **kwargs):
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
            return_tensors="pt", return_attention_mask=attention_mask).to('cuda')

        outputs = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs
            # can try to uncomment this if model doesn't end
            # eos_token_id=tokenizer.eos_token_id
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=False), new_prompt_len, outputs.shape
