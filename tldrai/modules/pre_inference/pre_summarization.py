def prepare_summarization_input(processed_answers, n=3, max_new_tokens=200, token_limit=2048):
    combined_text = "<<<START OF LISTED ANSWERS>>>\n"

    # Combine answers, or select top n
    for count, answer in enumerate(processed_answers):
        if count == n:
            break
        answer_info = [f"{k}: {v}\n" for k, v in answer.items()]
        answer_info = f"Answer {count+1}:\n" + "".join(answer_info)

        if len(combined_text) + max_new_tokens * 1.2 + len("<<<END OF LISTED ANSWERS>>>") + len(answer_info) > token_limit:
            break

        combined_text += answer_info

    return combined_text + "<<<END OF LISTED ANSWERS>>>"
