# Define user query
from tldrai.modules.stack_overflow.fetch import fetch_answers_for_questions
from tldrai.modules.stack_overflow.process import process_answers
from tldrai.modules.stack_overflow.search import search_stack_overflow_questions

user_query = "how to sort a list in python"

# Search for top questions related to the query
top_questions = search_stack_overflow_questions(user_query, num_questions=5)
question_ids = [q['question_id'] for q in top_questions]

# Fetch answers for these questions
answers = fetch_answers_for_questions(question_ids)

# Process answers, considering metadata like date and votes
processed_answers = process_answers(answers)

# Example output, customize further as needed
for answer in processed_answers[:10]:  # Just showing top 10 for brevity
    print(f"Date: {answer['creation_date']}, Score: {answer['score']}, Accepted: {answer['is_accepted']}")
    # Add logic to display or summarize the body of the answer

    print(f"Answer: {answer['body']}")
