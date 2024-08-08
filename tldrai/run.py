"""
Main running script
"""

import datetime
import logging

import hydra
from omegaconf import DictConfig
from stackapi import StackAPI

from tldrai.modules.pre_inference.pre_summarization import (
    prepare_prompt_for_tokenizer,
    prepare_summarization_input,
)
from tldrai.modules.stack_overflow.fetch import fetch_answers_for_questions
from tldrai.modules.stack_overflow.process import process_answers, process_questions
from tldrai.modules.stack_overflow.search import search_stack_overflow_questions
from tldrai.modules.utils.logging import configure_logging

# Set up logging
logger = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    configure_logging(logging.DEBUG if cfg.get("verbose", False) else logging.INFO)

    start_time_ms = round(datetime.datetime.now().timestamp() * 1000)
    logger.debug("Starting process...")

    question = cfg.question
    logger.debug(f"Processing question: {question}")
    logger.debug(f"no_search set to: {cfg.no_search}")

    summarizer = hydra.utils.instantiate(cfg.summarization_pipeline)
    system_prompt = "" if not cfg.prompt else cfg.prompt.format(question)

    if cfg.no_search:
        summarization_input = ""
        fetch_time_ms = process_time_ms = round(
            datetime.datetime.now().timestamp() * 1000
        )
    else:
        SITE = StackAPI("stackoverflow")
        SITE.page_size = cfg.stack_overflow.page_size

        try:
            questions = search_stack_overflow_questions(
                SITE,
                question,
                num_questions=cfg.stack_overflow.num_questions,
                search_style=cfg.stack_overflow.search_style,
                tag=cfg.stack_overflow.tag,
            )
        except RuntimeError:
            logger.error(
                "\tNo questions found for the query. Try rephrasing and searching again. Alternatively, try asking the "
                "LLM without the search."
            )
            return

        questions = process_questions(questions)
        question_ids = [str(x["question_id"]) for x in questions]

        answers = fetch_answers_for_questions(
            SITE, question_ids, num_answers=cfg.stack_overflow.num_answers
        )
        fetch_time_ms = round(datetime.datetime.now().timestamp() * 1000)
        processed_answers = process_answers(answers["items"], questions, question)
        history_len = 0 if cfg.history is None else len(cfg.history)
        max_new_tokens = cfg.get("max_new_tokens", 0)
        summarization_input = prepare_summarization_input(
            processed_answers,
            n=5,
            max_new_tokens=max_new_tokens,
            token_limit=cfg.model_token_limit,
            history_len=history_len,
        )
        process_time_ms = round(datetime.datetime.now().timestamp() * 1000)

    summarization_input = prepare_prompt_for_tokenizer(
        cfg, question, summarization_input, system_prompt
    )

    generation_params = cfg.generation_params
    try:
        summary, input_len, token_shape = summarizer.run(
            summarization_input, **generation_params
        )
        print("\n")
        logger.debug("Summary generated successfully.")
        end_time_ms = round(datetime.datetime.now().timestamp() * 1000)
        status = "success"
        status_message = ""
        token_usage = token_shape[1]
        output = summary[input_len:]
    except Exception as e:
        end_time_ms = round(datetime.datetime.now().timestamp() * 1000)
        status = "error"
        status_message = str(e)
        token_usage = 0
        summary = ""
        output = ""
        input_len = 0

        logger.error(f"Error: {status_message}")
        raise

    run_params = dict(
        **generation_params,
        **cfg.stack_overflow,
        **cfg.summarization_pipeline,
        prompt=cfg.prompt,
        fetch_time_ms=fetch_time_ms - start_time_ms,
        process_time_ms=process_time_ms - fetch_time_ms,
        token_usage=token_usage,
        success=True if status == "success" else False,
        output="" if status == "error" else output,
        question=question,
        status_message=status_message,
        inference_time_ms=end_time_ms - process_time_ms,
        summarization_input=summarization_input,
        prompt_ending=cfg.prompt_ending,
        is_prompt_codified=cfg.is_prompt_codified,
    )

    logger.debug(f"Run parameters: {run_params}")
