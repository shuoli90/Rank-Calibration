def clean_generation(generation):
    strings_to_filter_on = [
                '.', '\n', 'Q:', 'A:', 'question:', 'answer:', 'Question:', 'Answer:', 'Questions:', 'questions:', 'QUESTION:',
                'ANSWER:'
            ]
    for string in strings_to_filter_on:
        if string in generation:
            generation = generation.split(string)[0]
    return generation.strip()
