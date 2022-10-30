def get_i_tag_from_b_tag(label: int):
    # labels_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    if label % 2 == 1:
        label += 1
    return label


def align_labels_with_tokens(labels, word_ids, special_token=-100):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = special_token if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(special_token)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            label = get_i_tag_from_b_tag(label)
            new_labels.append(label)
    return new_labels


def tokenize_and_align_labels(dataset, tokenizer, device):
    tokenized_inputs = tokenizer(
        dataset["tokens"], truncation=True, padding=True, is_split_into_words=True, return_tensors="pt").to(device)
    all_labels = dataset["label"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def format_result(test_string, tokens, predictions, tokenized_input, labels_names):
    results = dict()
    current_id = -1
    current_key_word = ""
    splitted_string = test_string.split()
    for token, prediction, word_id in zip(tokens, predictions[0], tokenized_input.word_ids(0)):
        if word_id is None:
            continue
        if current_id != word_id:
            current_id = word_id
            current_key_word = splitted_string[word_id]
            results[current_key_word] = {"Tokens": [], "Labels": []}
        results[current_key_word]["Tokens"].append(token)
        results[current_key_word]["Labels"].append(labels_names[prediction])
    return results
