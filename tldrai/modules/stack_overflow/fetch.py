def fetch_answers_for_questions(api, question_ids, num_answers=5):
    all_answers = api.fetch(
        "questions/{}/answers".format(";".join(question_ids)),
        sort="votes",
        filter="withbody",
        pagesize=num_answers,
    )
    return all_answers
