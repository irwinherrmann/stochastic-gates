import heapq

def gates_threshold(acts, threshold):
    keep_layers = []
    drop_layers = []
    for i, a in acts.items():
        if a > threshold:
            keep_layers.append(i)
        else:
            drop_layers.append(i)
    return keep_layers, drop_layers

def gates_topk(acts, topk=5):
    topk_keys_sorted_by_values = heapq.nlargest(topk, acts, key=acts.get)
    drop_layers = [i for i in acts.keys() if not i in topk_keys_sorted_by_values]
    return topk_keys_sorted_by_values, drop_layers

