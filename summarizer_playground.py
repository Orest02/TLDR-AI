from stackapi import StackAPI
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from tldrai.modules.generation_pipeline.constructed_pipeline import ConstructedPipeline
from tldrai.modules.pre_inference.pre_summarization import prepare_summarization_input
from tldrai.modules.stack_overflow.fetch import fetch_answers_for_questions
from tldrai.modules.stack_overflow.process import process_answers, process_questions
from tldrai.modules.stack_overflow.search import search_stack_overflow_questions

SITE = StackAPI('stackoverflow')
SITE.page_size = 15

# question = "sort a list in python"
question = "apply a function to a pandas column"

# Initialize a summarization pipeline
model_name = "Open-Orca/oo-phi-1_5"
# summarizer = pipeline("text-generation", model=model_name, trust_remote_code=True)
summarizer = ConstructedPipeline(model_name, chat_template="chat_templates/chat_templates/chatml.jinja")

system_prompt = f"""INSTRUCTION: Given a collection of StackOverflow answers related to [{question}], please summarize the key points by doing the following:

    Identify and extract the most relevant code snippets related to the topic. Focus on those which question title is closest to the topic, have the biggest score and are freshest.

    Provide a concise summary for each code snippet. This summary should briefly explain:
        The purpose of the code: What problem does it solve or what functionality does it provide?
        How the code works: A very brief overview of the logic or method used.
        Any important notes or warnings: Include if the code requires specific versions, dependencies, or if there are common pitfalls.

    Highlight any variations or alternatives in the code snippets that might be useful for different scenarios or preferences.

    Summarize the non-code advice that accompanies the code snippets in a few sentences. Focus on tips, best practices, or explanations that are frequently mentioned or highly valued by the community.

Please ensure the summary is clear, to the point, and accessible to someone with a basic understanding of the topic.\n"""

prompt = system_prompt + "STACKOVERFLOW ANSWERS:\n"

# Fetch and preprocess data

# Fetch answers for these questions
questions = search_stack_overflow_questions(SITE, question, num_questions=5)
questions = process_questions(questions)
question_ids = [str(x['question_id']) for x in questions]

print("Questions:", question_ids)
# Fetch answers for these questions
answers = fetch_answers_for_questions(SITE, question_ids, num_answers=5)

print("Answers:", answers)
# Process answers, considering metadata like date and votes
processed_answers = process_answers(answers['items'], questions, question)

summarization_input = prepare_summarization_input(processed_answers)
print("Summarization input", summarization_input)
# raise Exception

codified_prompt = [
    {
        "role": "system",
        "content": system_prompt
     },
    {
        "role": "user",
        "content": summarization_input
    }
]

summarization_input = prompt + summarization_input + "\nSummary:"
summary = summarizer.run(codified_prompt)
print("Summary:", summary[0])
