from tldrai.modules.utils.exceptions import ArgumentError


def search_stack_overflow_questions(api, query, tag='', num_questions=3, search_style='excerpts'):
    if search_style == 'simple':
        if tag:
            questions = api.fetch('search', intitle=query, tagged=tag, sort='relevance', pagesize=num_questions)
        else:
            questions = api.fetch('search', intitle=query, sort='relevance', pagesize=num_questions)
    elif search_style == 'excerpts':
        questions = api.fetch('search/excerpts', q=query, body=query, sort='relevance', pagesize=num_questions)
    else:
        raise ArgumentError(f'Unknown search_style {search_style}. Currently "simple" and "excerpts" are supported')
    print("Questions: ", questions)
    return questions['items']
