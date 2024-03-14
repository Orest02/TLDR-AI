from stackapi import StackAPI

# Define your search query
SEARCH_QUERY = "exit vim"

# Choose the sorting criteria (options: "activity", "creation", "votes")
SORT_CRITERIA = "creation"  # Change to "creation" for sorting by date

# Choose the order (options: "asc", "desc")
SORT_ORDER = "desc"  # "asc" for oldest first, "desc" for newest first

# Connect to Stack Overflow using your API key
site = StackAPI("stackoverflow")

# Search for questions matching the query
questions = site.fetch('questions', search=SEARCH_QUERY)

# Check if any questions were found
if not questions:
    print("No questions found for the search query.")
else:
    print("Questions: ", questions)
    # Select the first question by default (adjust if needed)
    question_id = questions[0]['question_id']  # Get ID of the first question

    # Fetch answers for the selected question
    answers = site.fetch('answers', parent_id=question_id, sort=SORT_CRITERIA, order=SORT_ORDER)

    def sort_by_date(data):
        """Sorts data by creation date (descending order)"""
        return sorted(data, key=lambda x: x['creation_date'], reverse=True)

    # Sort answers and comments by creation date (newest first)
    sorted_answers = sort_by_date(answers)

    # Use the sorted answers similarly as in the previous example
    # ...

    print(sorted_answers)
