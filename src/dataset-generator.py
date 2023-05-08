import json
from jinja2 import Environment, FileSystemLoader
import random
import string
import inflect
import time
import math
import os


random.seed(12345)
number_to_word = inflect.engine()


def get_col(cols):
    if not cols:
        return None
    index = random.randrange(len(cols))
    col = cols[index]
    ref_index = random.randrange(len(col['refs']))
    alias = col['refs'][ref_index]
    return {
        "col_name": col['name'],
        "col_alias": alias
    }


def get_operator(operators):
    index = random.randrange(len(operators))
    return operators[index]


def get_text_col(cols):
    t_cols = list(filter(lambda c: c['type'] == 'text', cols))
    return get_col(t_cols)


def get_rand_text(cols):
    rd = random.randint(1, 10)
    if rd % 10 == 0:
        if random.randrange(1) == 0:
            return { 'text': random.randrange(2000) }
        else:
             return { 'text': number_to_word.number_to_words(random.randrange(2000)) }
    else:
        return { 'text': ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase, k=5)) }


def get_text_operator(cols):
    operators = [
        {
            'op': "$eq",
            'op_name': 'equal'
        },
        {
            'op': "$eq",
            'op_name': '='
        },
        {
            'op': "$neq",
            'op_name': 'not equal'
        },
        {
            'op': "$contains",
            'op_name': 'contains'
        },
        {
            'op': "$contains",
            'op_name': 'is'
        },
        {
            'op': "$contains",
            'op_name': 'like'
        },
        {
            'op': "$ncontains",
            'op_name': 'not contains'
        },
    ]
    return get_operator(operators)


def get_number_col(cols):
    t_cols = list(filter(lambda c: c['type'] == 'number', cols))
    return get_col(t_cols)


def get_rand_number(cols):
    max_num = 2000
    if random.randrange(1) == 0:
        num = random.randrange(max_num)
    else:
        num = round(random.uniform(0, 1) * max_num, 1)
    return {
        "num": num,
        "num_word": number_to_word.number_to_words(num)
    }


def get_number_operator(cols):
    operators = [
        {
            'op': "$eq",
            'op_name': 'equal'
        },
        {
            'op': "$eq",
            'op_name': '='
        },
        {
            'op': "$neq",
            'op_name': 'not equal'
        },
        {
            'op': "$neq",
            'op_name': '!='
        },
        {
            'op': "$gt",
            'op_name': 'greater than'
        },
        {
            'op': "$gt",
            'op_name': '>'
        },
        {
            'op': "$gt",
            'op_name': 'more than'
        },
        {
            'op': "$gte",
            'op_name': 'more than or equal to'
        },
        {
            'op': "$gte",
            'op_name': '>='
        },
        {
            'op': "$gte",
            'op_name': 'greater than or equal to'
        },
        {
            'op': "$lt",
            'op_name': 'lower than'
        },
        {
            'op': "$lt",
            'op_name': 'less than'
        },
        {
            'op': "$lt",
            'op_name': '<'
        },
        {
            'op': "$lte",
            'op_name': 'lower than or equal to'
        },
        {
            'op': "$lte",
            'op_name': 'less than or equal to'
        },
        {
            'op': "$lte",
            'op_name': '<='
        },
    ]
    return get_operator(operators)


def get_bool_col(cols):
    t_cols = list(filter(lambda c: c['type'] == 'bool', cols))
    return get_col(t_cols)


def get_rand_bool(cols):
    return { 'val': True if random.randrange(1) == 1 else False }


def get_date_col(cols):
    t_cols = list(filter(lambda c: c['type'] == 'date', cols))
    return get_col(t_cols)


def get_rand_date(cols):
    stime = time.mktime(time.strptime("1/1/2008 1:30 PM", '%m/%d/%Y %I:%M %p'))
    etime = time.mktime(time.strptime("1/1/2009 4:50 AM", '%m/%d/%Y %I:%M %p'))

    ptime = stime + random.random() * (etime - stime)

    return {
        'date': time.strftime('%m/%d/%Y %I:%M %p', time.localtime(ptime))
    }


def get_refs_col(cols):
    t_cols = list(filter(lambda c: c['type'] == 'collections', cols))
    return get_col(t_cols)


def get_any_col_with_value(col_num):
    def get_any(cols):
        ind = random.randrange(3)
        if ind == 0:
            col = get_text_col(cols)
            if col is None:
                return None
            txt = get_rand_text(cols)['text']
            op = get_text_operator(cols)
            return {
                f"col_name{col_num}": col['col_name'],
                f"col_alias{col_num}": col['col_alias'],
                f"col_value{col_num}": txt,
                f"col_value_alias{col_num}": txt,
                f"op{col_num}": op['op'],
                f"op_name{col_num}": op['op_name'],
            }
        elif ind == 1:
            col = get_number_col(cols)
            col_val = get_rand_number(cols)
            if col is None:
                return None
            op = get_number_operator(cols)
            return {
                f"col_name{col_num}": col['col_name'],
                f"col_alias{col_num}": col['col_alias'],
                f"col_value{col_num}": col_val['num'],
                f"col_value_alias{col_num}": col_val['num_word'],
                f"op{col_num}": op['op'],
                f"op_name{col_num}": op['op_name']
            }
        else:
            col = get_date_col(cols)
            if col is None:
                return None
            date = get_rand_date(cols)['date']
            op = get_number_operator(cols)
            return {
                f"col_name{col_num}": col['col_name'],
                f"col_alias{col_num}": col['col_alias'],
                f"col_value{col_num}": date,
                f"col_value_alias{col_num}": date,
                f"op{col_num}": op['op'],
                f"op_name{col_num}": op['op_name'],
            }
    return get_any


def get_join_operator(cols):
    operators = [
        {
            'op': '$and',
            'op_name': '&'
        },
        {
            'op': '$and',
            'op_name': 'and'
        },
        {
            'op': '$or',
            'op_name': 'or'
        },
        {
            'op': '$or',
            'op_name': '|'
        },
    ]
    return get_operator(operators)


def get_model_description(model):
    cols = [f"{c['name']} {c['type']} as {','.join(c['refs'])}" for c in model['cols']]
    return ';'.join(cols)


def get_model_names(models):
    model_names = [f"{m['name']} as {','.join(m['refs'])}" for m in models]
    return ';'.join(model_names)


TEMPLATES_MAP = {
    "text.txt": [
        get_text_col,
        get_rand_text,
        get_text_operator
    ],
    "number.txt": [
        get_number_col,
        get_rand_number,
        get_number_operator
    ],
    "bool.txt": [
        get_bool_col,
        get_rand_bool
    ],
    "date.txt": [
        get_date_col,
        get_rand_date,
        get_number_operator
    ],
    "count.txt": [
        get_refs_col,
        get_number_operator,
        get_rand_number
    ],
    "join.txt": [
        get_any_col_with_value(1),
        get_any_col_with_value(2),
        get_join_operator
    ]
}


TEMPLATES_DISTRIBUTIONS = {
    "text.txt": 0.25,
    "number.txt": 0.25,
    "join.txt": 0.15,
    "count.txt": 0.15,
    "date.txt": 0.15,
    "bool.txt": 0.05,
}


QUERY_DATASET_SIZE = 12000
MODEL_DATASET_SIZE = 5000
SPLITTER = "[-SPLITER-]"
DEV_VAL_TEST = (0.88, 0.06, 0.06)


def get_model(models):
    index = random.randrange(len(models))
    return models[index]


def get_model_alias(model):
    index = random.randrange(len(model['refs']))
    return model['refs'][index]


def get_search_word():
    words = ['find', 'get', 'show', 'search for', 'look up', 'fetch', 'provide with', 'select']
    index = random.randrange(len(words))
    return words[index]


def generate_query_ds(tm):
    environment = Environment(loader=FileSystemLoader("src/templates/query/"))
    with open('src/structure/structure-description.json', 'r') as f:
        models = json.load(f)
    ds = []
    for template_name in TEMPLATES_MAP:
        template = environment.get_template(template_name)
        actions = TEMPLATES_MAP[template_name]
        cur_size = 0
        size_for_template = math.ceil(QUERY_DATASET_SIZE * TEMPLATES_DISTRIBUTIONS[template_name])
        while cur_size < size_for_template:
            model = get_model(models)
            action_args = [a(model['cols']) for a in actions]
            if list(filter(lambda x: x is None, action_args)):
                continue
            args = {
                'search_word': get_search_word(),
                'model_name': model['name'],
                'model_alias': get_model_alias(model),
                'model_description': get_model_description(model)
            }
            for action_arg in action_args:
                args = { **args, **action_arg }
            utterance = template.render(args)
            utterance_parts = utterance.split(SPLITTER)
            ds.append({
                "utterance": utterance_parts[0],
                "answer": utterance_parts[1]
            })
            cur_size += 1
    with open(f'src/datasets/query/{tm}.json', 'w+') as f:
        json.dump(ds, f)
    return ds


def generate_model_ds(tm):
    environment = Environment(loader=FileSystemLoader("src/templates/model/"))
    with open('src/structure/structure-description.json', 'r') as f:
        models = json.load(f)
    model_names = get_model_names(models)
    ds = []
    template = environment.get_template("model.txt")
    while len(ds) < MODEL_DATASET_SIZE:
        model = get_model(models)
        col_args = get_any_col_with_value(1)(model['cols'])
        if col_args is None:
            continue
        args = {
            'search_word': get_search_word(),
            'model_name': model['name'],
            'model_alias': get_model_alias(model),
            'model_names': model_names,
            **col_args,
        }
        utterance = template.render(args)
        utterance_parts = utterance.split(SPLITTER)
        ds.append({
            "utterance": utterance_parts[0],
            "answer": utterance_parts[1]
        })
    with open(f'src/datasets/model/{tm}.json', 'w+') as f:
        json.dump(ds, f)
    return ds


if __name__ == "__main__":
    tm = round(time.time())
    query_ds = generate_query_ds(tm)
    model_ds = generate_model_ds(tm)
    random.shuffle(query_ds)
    random.shuffle(model_ds)

    train_query_ds = query_ds[:round(DEV_VAL_TEST[0] * len(query_ds))]
    val_query_ds = query_ds[len(train_query_ds):len(train_query_ds) + round(DEV_VAL_TEST[1] * len(query_ds))]
    test_query_ds = query_ds[len(train_query_ds) + len(val_query_ds):]

    train_model_ds = model_ds[:round(DEV_VAL_TEST[0] * len(model_ds))]
    val_model_ds = model_ds[len(train_model_ds):len(train_model_ds) + round(DEV_VAL_TEST[1] * len(model_ds))]
    test_model_ds = model_ds[len(train_model_ds) + len(val_model_ds):]

    train_ds = [*train_model_ds, *train_query_ds]
    val_ds = [*val_model_ds, *val_query_ds]
    test_ds = [*test_model_ds, *test_query_ds]

    base_path = f'src/datasets/target/{tm}'
    os.mkdir(base_path)

    with open(f'{base_path}/train.json', 'w+') as f:
        json.dump(train_ds, f)

    with open(f'{base_path}/validation.json', 'w+') as f:
        json.dump(val_ds, f)

    with open(f'{base_path}/test.json', 'w+') as f:
        json.dump(test_ds, f)