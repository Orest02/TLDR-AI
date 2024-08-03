"""
Searches StackOverflow for questions matching the given query
"""

import logging

from tldrai.modules.utils.exceptions import ArgumentError
from tldrai.modules.utils.logging import configure_logging


def search_stack_overflow_questions(
    api, query, tag="", num_questions=3, search_style="excerpts", verbose=False
):
    """Searches StackOverflow for questions matching the given query

    Args:
        api: StackOverflow API object
        query: query to search
        tag: any special tag to search
        num_questions: number of questions to search
        search_style: "excerpts" or "simple"
        verbose: debug if True, otherwise info

    Returns:
        list(dict()): StackOverflow questions found
    """
    configure_logging(logging.DEBUG if verbose else logging.INFO)
    logger = logging.getLogger(__name__)
    if search_style == "simple":
        if tag:
            questions = api.fetch(
                "search",
                intitle=query,
                tagged=tag,
                sort="relevance",
                pagesize=num_questions,
            )
        else:
            questions = api.fetch(
                "search",
                intitle=query,
                sort="relevance",
                pagesize=num_questions,
            )
    elif search_style == "excerpts":
        questions = api.fetch(
            "search/excerpts",
            q=query,
            sort="relevance",
            pagesize=num_questions,
        )
    else:
        raise ArgumentError(
            f'Unknown search_style {search_style}. Currently "simple" and "excerpts" are supported'
        )
    logger.debug("Questions: ", questions)
    if not questions["items"]:
        raise RuntimeError(
            "No questions found for the query. try rephrasing and searching again"
        )
    return questions["items"]
