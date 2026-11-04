"""Conservative diagnostics for spatial QA."""
from collections import defaultdict
import math
import re
import unicodedata


_SCALAR = re.compile(
    r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?\.?'
    r'|[+-]?\.\d+(?:[eE][+-]?\d+)?\.?'
)


def normalized(text):
    text = unicodedata.normalize('NFC', str(text)).replace('−', '-').casefold()
    return ' '.join(text.split()).removesuffix('.')


def numeric(text):
    # A numeral embedded in prose can be a count, list index, or another quantity.
    # Score only bare scalars, avoiding guesses about units or which number to use.
    text = str(text).strip().replace('−', '-')
    if not _SCALAR.fullmatch(text):
        return None
    value = float(text.removesuffix('.').replace(',', ''))
    return value if math.isfinite(value) else None


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups['all'].append(row)
        groups['task:' + row.get('task', 'unspecified')].append(row)
    output = {}
    for name, group in groups.items():
        exact = [normalized(row['prediction']) == normalized(row['answer']) for row in group]
        result = {'examples': len(group), 'normalized_exact_match': sum(exact) / len(group)}
        numerical = []
        for row in group:
            target = numeric(row['answer'])
            if target is not None:
                numerical.append((target, numeric(row['prediction'])))
        if numerical:
            result['numeric_examples'] = len(numerical)
            result['numeric_parse_rate'] = sum(pred is not None for _, pred in numerical) / len(numerical)
            for tolerance in (.05, .1, .2, .25):
                correct = sum(
                    pred is not None and abs(pred - target) <= tolerance * abs(target)
                    for target, pred in numerical
                )
                result[f'numeric_relative_accuracy_{tolerance:g}'] = correct / len(numerical)
        output[name] = result
    return {'scope': 'Prototype diagnostics. Numeric accuracy uses bare scalar answers with no unit conversion or semantic judging.',
            'groups': output}
