from datetime import datetime
import re

import numpy as np
import pandas as pd

from tldrai.modules.similarity_scoring.sentence_similarity import SentenceSimilarity


def process_questions(questions):
    ret_questions = []

    for question in questions:
        if question.get('closed_reason') == 'Duplicate':
            continue
        ret_questions.append(question)
    return ret_questions


def preprocess_answer(answer):
    return answer
    # Extract code blocks
    code_blocks = re.findall(r'<code>(.*?)</code>', answer, re.DOTALL)
    code_snippets = "\n".join(code_blocks)
    # Prepend code to the answer body to emphasize it
    return f"[Code]\n{code_snippets}\n[/Code]\n\n{answer}"


def process_answers(answers, questions, user_question):
    questions = pd.DataFrame(questions).set_index("question_id")
    answers = pd.DataFrame(answers)
    answers = pd.merge(answers, questions["title"], left_on="question_id", right_index=True)
    print(answers)
    similarity = (SentenceSimilarity().compare_base_to_others(base_sentence=user_question, other_sentences=answers["title"].tolist())).numpy()
    print("similarity: ", similarity)
    answers["similarity"] = similarity[0]
    print("Question ids: ", questions.index)
    processed = []
    for _, answer in answers.sort_values(by="similarity", ascending=False).iterrows():
        # Convert creation date from epoch time to a readable format
        print(answer['creation_date'], type(answer['creation_date']))
        if not np.isnan(answer['creation_date']):
            creation_date = datetime.fromtimestamp(int(answer['creation_date'])).strftime('%Y-%m-%d')
        else:
            creation_date = "UNKNOWN"
        if not np.isnan(answer['last_activity_date']):
            last_activity_date = datetime.fromtimestamp(int(answer['last_activity_date'])).strftime('%Y-%m-%d')
        else:
            last_activity_date = "UNKNOWN"
        score = answer['score']
        question = questions.loc[answer["question_id"], "title"]
        is_accepted = answer.get('is_accepted', False)
        # Example processing, customize based on needs
        processed.append({
            'question_title': question,
            'creation_date': creation_date,
            'last_activity_date': last_activity_date,
            'score': score,
            'is_accepted': is_accepted,
            'body': preprocess_answer(answer['body'])
        })
    return processed
