import datetime
import os

import git
import hydra
import wandb
from omegaconf import DictConfig, omegaconf
from stackapi import StackAPI

from modules.generation_pipeline.constructed_pipeline import ConstructedPipeline
from modules.pre_inference.pre_summarization import prepare_summarization_input
from modules.stack_overflow.fetch import fetch_answers_for_questions
from modules.stack_overflow.process import process_answers, process_questions
from modules.stack_overflow.search import search_stack_overflow_questions

from wandb.sdk.data_types.trace_tree import Trace


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Initialize StackAPI with Hydra config
    start_time_ms = round(datetime.datetime.now().timestamp() * 1000)

    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    wandb.init(project="tldr-ai")
    SITE = StackAPI('stackoverflow')
    repo = git.Repo(search_parent_directories=True)
    SITE.page_size = cfg.stack_overflow.page_size

    question = cfg.question

    # Initialize summarization pipeline
    summarizer = hydra.utils.instantiate(cfg.summarization_pipeline)

    system_prompt = '' if not cfg.prompt else cfg.prompt.format(question)

    # Fetch and preprocess data
    if cfg.no_search:
        summarization_input = ''
        fetch_time_ms = process_time_ms = round(datetime.datetime.now().timestamp() * 1000)
    else:
        questions = search_stack_overflow_questions(SITE, question, num_questions=cfg.stack_overflow.num_questions)
        questions = process_questions(questions)
        question_ids = [str(x['question_id']) for x in questions]

        answers = fetch_answers_for_questions(SITE, question_ids, num_answers=cfg.stack_overflow.num_answers)
        fetch_time_ms = round(datetime.datetime.now().timestamp() * 1000)
        processed_answers = process_answers(answers['items'], questions, question)
        history_len = 0 if cfg.history is None else len(cfg.history)
        summarization_input = prepare_summarization_input(processed_answers, n=5,
                                                          max_new_tokens=cfg.generation_params.max_new_tokens,
                                                          token_limit=cfg.model_token_limit,
                                                          history_len=history_len
                                                          )

        process_time_ms = round(datetime.datetime.now().timestamp() * 1000)
        summarization_input = prepare_prompt_for_tokenizer(cfg, question, summarization_input, system_prompt)

    generation_params = cfg.generation_params
    try:
        summary, input_len, token_shape = summarizer.run(summarization_input, **generation_params)
        print("Summary:", summary[0])
        end_time_ms = round(
            datetime.datetime.now().timestamp() * 1000
        )
        status = "success"
        status_message = (None,)
        token_usage = token_shape[1]
        output = summary[0][input_len:]
    except Exception as e:
        end_time_ms = round(
            datetime.datetime.now().timestamp() * 1000
        )  # logged in milliseconds
        status = "error"
        status_message = str(e)
        token_usage = 0
        summary = ''
        output = ''
        input_len = 0

        print("Error:", status, status_message)

    run_params = dict(
        **generation_params,
        **cfg.stack_overflow,
        **cfg.summarization_pipeline,
        prompt=cfg.prompt,
        fetch_time_ms=fetch_time_ms - start_time_ms,
        process_time_ms=process_time_ms - fetch_time_ms,
        token_usage=token_usage,
        git_hash=repo.head.object.hexsha,
        success=True if status == "success" else False,
        output='' if status == "error" else output,
        question=question,
        status_message=status_message,
        inference_time_ms=end_time_ms - process_time_ms,
        summarization_input=summarization_input,
        prompt_ending=cfg.prompt_ending,
        is_prompt_codified=cfg.is_prompt_codified
    )

    # for key, param in run_params.items():
    #     wandb.log(key, param)

    # create a span in wandb
    root_span = Trace(
        name="root_span",
        kind="llm",  # kind can be "llm", "chain", "agent" or "tool"
        status_code=status,
        status_message=status_message,
        metadata=run_params,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        inputs={"system_prompt": system_prompt, "query": summarization_input},
        outputs={"response": summary},
    )

    root_span.log(name="openai_trace")

    wandb.log(run_params)
    wandb.config.update(run_params)


def prepare_prompt_for_tokenizer(cfg, question, summarization_input, system_prompt):
    user_content = summarization_input + cfg.prompt_ending.format(question)

    if cfg.is_prompt_codified:
        if cfg.prompt:

            summarization_input = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        else:
            summarization_input = [
                {"role": "user", "content": user_content}
            ]

        if cfg.history:
            history = [{"role": key, "content": value} for key, value in cfg.history.items()]
            summarization_input = history + summarization_input
    else:
        summarization_input = system_prompt + user_content

        if cfg.history:
            summarization_input = "\n".join([v for k, v in cfg.history.items()]) + "\n" + summarization_input
    return summarization_input


if __name__ == "__main__":
    wandb_key = open("keys/WANDB_KEY").read()
    os.environ["WANDB_API_KEY"] = wandb_key

    main()
