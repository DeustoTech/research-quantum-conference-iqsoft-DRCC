def interpret_probabilities(state_probs: dict, num_classes:int = 2, method='module'):
    class_probs = {c: 0 for c in range(num_classes)}
    if method == 'module':
        for key, value in state_probs.items():
            class_probs[int(key, 2) % num_classes] += value
        return class_probs
    # TO DO GENERALIZE FOR MULTICLASS
    elif method == 'half':
        if num_classes != 2:
            raise ValueError('Half states method only works for binary classification')
        for key, value in state_probs.items():
            # 3 + 1 >= 2^5 / 2
            class_probs[1 if int(key, 2) + 1 >= (2^len(key))/2 else 0] += value
        return class_probs
    elif method == 'single_qubit':
        if num_classes != 2:
            raise ValueError('Single qubit method only works for binary classification')
        for key, value in state_probs.items():
            class_probs[int(key[-1])] += value
        return class_probs
    elif method == 'num_ones':
        if num_classes != 2:
            raise ValueError('Number of ones method only works for binary classification')
        for key, value in state_probs.items():
            class_probs[1 if key.count('1') > len(key) / 2 else 0] += value
        return class_probs
    else:
        raise ValueError('Method not recognized')