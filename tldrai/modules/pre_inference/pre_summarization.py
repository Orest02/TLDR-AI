def prepare_summarization_input(processed_answers, n=3):
    combined_text = "<<<START OF LISTED ANSWERS>>>\n"

    # Combine answers, or select top n
    for count, answer in enumerate(processed_answers):
        if count == n:
            break
        answer_info = [f"{k}: {v}\n" for k, v in answer.items()]
        combined_text += f"Answer {count+1}:\n" + "".join(answer_info)
    return combined_text + "<<<END OF LISTED ANSWERS>>>"
