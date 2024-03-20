import datetime
import os

import hydra
import wandb
from omegaconf import DictConfig
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
    SITE = StackAPI('stackoverflow')
    SITE.page_size = cfg.stack_overflow.page_size

    question = cfg.question

    # Initialize summarization pipeline
    summarizer = hydra.utils.instantiate(cfg.summarization_pipeline)

    system_prompt = cfg.prompt.format(question)

    prompt = system_prompt + "STACKOVERFLOW ANSWERS:\n"

    # Fetch and preprocess data
    questions = search_stack_overflow_questions(SITE, question, num_questions=cfg.stack_overflow.num_questions)
    questions = process_questions(questions)
    question_ids = [str(x['question_id']) for x in questions]

    answers = fetch_answers_for_questions(SITE, question_ids, num_answers=cfg.stack_overflow.num_answers)
    fetch_time_ms = round(datetime.datetime.now().timestamp() * 1000)
    processed_answers = process_answers(answers['items'], questions, question)
    summarization_input = prepare_summarization_input(processed_answers, n=5)

    process_time_ms = round(datetime.datetime.now().timestamp() * 1000)
    codified_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": summarization_input}
    ]

    summarization_input = prompt + summarization_input + "\nSummary:"
    generation_params = cfg.generation_params
    try:
        summary, token_shape = summarizer.run(codified_prompt, **generation_params)
        print("Summary:", summary[0])
        end_time_ms = round(
            datetime.datetime.now().timestamp() * 1000
        )
        status = "success"
        status_message = (None,)
        token_usage = token_shape[0]
    except Exception as e:
        end_time_ms = round(
            datetime.datetime.now().timestamp() * 1000
        )  # logged in milliseconds
        status = "error"
        status_message = str(e)
        token_usage = 0
        summary = ''


    # create a span in wandb
    root_span = Trace(
        name="root_span",
        kind="llm",  # kind can be "llm", "chain", "agent" or "tool"
        status_code=status,
        status_message=status_message,
        metadata=dict(**generation_params, **cfg.stack_overflow, **cfg.summarization_pipeline, prompt=cfg.prompt,
                      fetch_time_ms=fetch_time_ms, process_time_ms=process_time_ms, token_usage=token_usage),
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        inputs={"system_prompt": system_prompt, "query": summarization_input},
        outputs={"response": summary},
    )

    root_span.log(name="openai_trace")


if __name__ == "__main__":
    wandb_key = open("keys/WANDB_KEY").read()
    os.environ["WANDB_API_KEY"] = wandb_key

    wandb.init(project="tldr-ai")
    main()
