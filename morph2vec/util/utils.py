def get_morph_parts_from_wlm_str(wlm_str: str):
    return [p for p in wlm_str.split('~') if p.startswith('m:')]


def morpho_dist(morph_parts_combinations):
    if not morph_parts_combinations:
        return 'N/A'
    return max([morpho_dist_per_pair(comb[0], comb[1]) for comb in morph_parts_combinations])


def morpho_dist_per_pair(w1_parts, w2_parts):
    max_len = max(len(w1_parts), len(w2_parts))
    for parts in [w1_parts, w2_parts]:
        for i in range(len(parts), max_len):
            parts.append('')
    return 1 - sum(el1 != el2 for el1, el2 in zip(w1_parts, w2_parts)) / float(max_len)
