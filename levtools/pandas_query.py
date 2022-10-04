import pandas as pd

pd.set_option('use_inf_as_na', True)


def isword(char):
    return (char >= 'a' and char <= 'z') or (char >= 'A' and char <= 'Z') or (char >= '0' and char <= '9') or (
                char in "_\"")


def issep(char):
    return char.isspace()


def split(text):
    words = []
    word = ''
    prev_is_word = isword(text[0])
    prev_is_sep = issep(text[0])

    for inx, char in enumerate(text):
        if (prev_is_word and not isword(char)) or \
                (not prev_is_word and isword(char)) or \
                (not prev_is_word and issep(char)):
            words = words + [word.strip()]
            word = char
        else:
            word = word + char

        prev_is_word = isword(char)

    return words + [word.strip()]


def get_query_mask(df, query_str, debug=False):
    assert (df.shape[0] > 0)

    binary_ops = ["==", "!=", ">=", "<=", ">", "<", "in", "and", "or", "&", "|"]

    for op in binary_ops:

        if op not in split(query_str):
            continue

        column, right = query_str.split(op)
        column = column.strip()
        right = right.strip()

        if right.startswith('dt(') and right.endswith(')'):
            year, mon, day = right[3:-1].split("-")
            year = int(year)
            mon = int(mon)
            day = int(day)
            right = f'datetime.date(year={year}, month={mon}, day={day})'
            mask_str = f'df[\'{column}\'] {op} {right}'

        elif right.startswith('\"') and right.endswith('\"'):
            mask_str = f'df[\'{column}\'] {op} {right}'

        elif right.startswith('@'):
            mask_str = f'df[\'{column}\'] {op} {right[1:]}'

        elif right in df.columns:
            mask_str = f'df[\'{column}\'] {op} df[\'{right}\']'

        elif right in {"None", "Null", "NA", "Nan", "inf", "-inf"}:
            if op == '==':
                mask_str = f'df[\'{column}\'].isnull()'
            elif op == '!=':
                mask_str = f'df[\'{column}\'].notnull()'
            else:
                raise Exception(f'Unkown relation {op} to operand {right}')
        else:
            mask_str = f'df[\'{column}\'] {op} {right}'
            # raise Exception(f'Uknown right {right} operand error')

        if debug:
            print(mask_str)
        mask = eval(mask_str)
        return mask

    raise Exception(f'Unprocessable {query_str}')


def query(df, query_str, debug=False):
    return df[get_query_mask(df, query_str, debug)]


def q_and(df, terms, debug=False):
    masks = [get_query_mask(df, term, debug) for term in terms]
    mask = masks[0]
    for mask1 in masks[1:]:
        mask = mask & mask1
    return df[mask]


def q_or(df, terms, debug=False):
    masks = [get_query_mask(df, term, debug) for term in terms]
    mask = masks[0]
    for mask1 in masks[1:]:
        mask = mask | mask1
    return df[mask]






