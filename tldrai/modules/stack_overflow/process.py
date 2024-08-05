import logging
import re
from datetime import datetime

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from tldrai.modules.similarity_scoring.sentence_similarity import SentenceSimilarity
from tldrai.modules.utils.logging import configure_logging


def process_questions(questions):
    ret_questions = []

    if not questions:
        raise RuntimeError("No questions to process")

    for question in questions:
        if question.get("closed_reason") == "Duplicate":
            continue
        ret_questions.append(question)
    return ret_questions


def preprocess_answer(answer):
    return answer
    # Extract code blocks
    code_blocks = re.findall(r"<code>(.*?)</code>", answer, re.DOTALL)
    code_snippets = "\n".join(code_blocks)
    # Prepend code to the answer body to emphasize it
    return f"[Code]\n{code_snippets}\n[/Code]\n\n{answer}"


# Function to extract and merge code blocks
def extract_and_merge_code_blocks(
    html_content, delimiter="```\n```", delimiter_wrap="```"
):
    soup = BeautifulSoup(html_content, "lxml")

    # Find all code blocks
    code_blocks = soup.find_all("code")

    # Extract text from code blocks
    code_texts = ["\n" + code.get_text(strip=True) for code in code_blocks if code]

    # Merge code blocks with a delimiter
    merged_code = delimiter.join(code_texts)

    if delimiter_wrap:
        merged_code = delimiter + merged_code + delimiter

    return merged_code


def process_answers(answers, questions, user_question, verbose=False):
    configure_logging(logging.DEBUG if verbose else logging.INFO)
    logger = logging.getLogger(__name__)
    questions = pd.DataFrame(questions).set_index("question_id")
    answers = pd.DataFrame(answers)
    answers = pd.merge(
        answers, questions["title"], left_on="question_id", right_index=True
    )
    logger.debug(answers)
    similarity = (
        SentenceSimilarity(verbose=verbose).compare_base_to_others(
            base_sentence=user_question,
            other_sentences=answers["title"].tolist(),
        )
    ).numpy()
    answers["similarity"] = similarity[0]
    processed = []
    for _, answer in answers.sort_values(by="similarity", ascending=False).iterrows():
        # Convert creation date from epoch time to a readable format
        if not np.isnan(answer["creation_date"]):
            creation_date = datetime.fromtimestamp(
                int(answer["creation_date"])
            ).strftime("%Y-%m-%d")
        else:
            creation_date = "UNKNOWN"
        if not np.isnan(answer["last_activity_date"]):
            last_activity_date = datetime.fromtimestamp(
                int(answer["last_activity_date"])
            ).strftime("%Y-%m-%d")
        else:
            last_activity_date = "UNKNOWN"
        score = answer["score"]
        question = questions.loc[answer["question_id"], "title"]
        is_accepted = answer.get("is_accepted", False)
        # Example processing, customize based on needs
        processed.append(
            {
                "question_title": question,
                "creation_date": creation_date,
                "last_activity_date": last_activity_date,
                "score": score,
                "is_accepted": is_accepted,
                "code": extract_and_merge_code_blocks(answer["body"]),
                "body": preprocess_answer(answer["body"]),
            }
        )
    return processed
