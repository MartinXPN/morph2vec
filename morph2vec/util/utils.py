import urllib.request
from pathlib import Path

from tqdm import tqdm


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


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url: str, save_path: str, exists_ok: bool = True, verbose: int = 2):
    if url is None or save_path is None:
        raise ValueError('Both `url` and `save_path` need to be provided')

    if Path(save_path).exists():
        if not exists_ok:
            raise ValueError(f'File with the path `{save_path}` already exists')
        if verbose >= 1:
            print(f'File with the path `{save_path}` already exists')
        return

    description = url.split('/')[-1]
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=description, disable=verbose < 2) as t:
        urllib.request.urlretrieve(url=url, filename=save_path, reporthook=t.update_to)
