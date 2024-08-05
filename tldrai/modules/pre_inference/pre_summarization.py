def prepare_summarization_input(
    processed_answers, n=3, max_new_tokens=200, token_limit=2048, history_len=0
):
    combined_text = ""  # "<<<START OF LISTED ANSWERS>>>\n"

    # Combine answers, or select top n
    for count, answer in enumerate(processed_answers):
        if count == n:
            break
        answer_info = [f"{k}: {v}\n" for k, v in answer.items()]
        answer_info = f"Answer {count+1}:\n" + "".join(answer_info)

        if (
            len(combined_text)
            + max_new_tokens * 1.2
            + len("<<<END OF LISTED ANSWERS>>>")
            + len(answer_info)
            + history_len
            > token_limit
        ):
            break

        combined_text += answer_info

    return combined_text


def prepare_prompt_for_tokenizer(cfg, question, summarization_input, system_prompt):
    user_content = summarization_input + cfg.prompt_ending.format(question)

    if cfg.is_prompt_codified:
        if cfg.prompt:
            summarization_input = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        else:
            summarization_input = [{"role": "user", "content": user_content}]

        if cfg.history:
            history = [
                {"role": key, "content": value} for key, value in cfg.history.items()
            ]
            summarization_input = history + summarization_input
    else:
        summarization_input = system_prompt + user_content

        if cfg.history:
            summarization_input = (
                "\n".join([v for k, v in cfg.history.items()])
                + "\n"
                + summarization_input
            )

    return summarization_input
