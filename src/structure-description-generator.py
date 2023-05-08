import json
import re


def split_camel_case(str):
    return ' '.join(re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', str))


def prepare_candidates(l):
    return list(set(filter(lambda x: x is not None, l)))


def prepare_string(str: str):
    if str:
        return str.lower().strip()
    return None


def describe_column(col, prev_col_path, prefixes, type):
    if "Localization" in col.get('tableName', '') or "ChangeTracking" in col.get('tableName', '') or col['name'] == "IsDeleted":
        return None
    candidates = prepare_candidates([prepare_string(col['caption']), prepare_string(split_camel_case(col['name']))])
    if prefixes:
        candidates = [f"{c} of {p}" for c, p in zip(candidates, prefixes)]
    return {
        "name": f"{prev_col_path}{col['name']}",
        "refs": candidates,
        "type": type
    }


def process_columns(columns, prev_col, prefixes, type):
    return filter(lambda x: x is not None, (map(lambda x: describe_column(x, prev_col, prefixes, type), columns)))


def process_all_columns(model, structure, depth = 0, prev_col = "", prefixes = []):
    if depth > 1:
        return []
    columns_ref = [
        *process_columns(model.get('booleanFieldsStructure', []), prev_col, prefixes, 'bool'),
        *process_columns(model.get('textFieldsStructure', []), prev_col, prefixes, 'text'),
        *process_columns(model.get('numberFieldsStructure', []), prev_col, prefixes, 'number'),
        *process_columns(model.get('dateTimeFieldsStructure', []), prev_col, prefixes, 'date'),
        *process_columns(model.get('entityCollectionsStructure', []), prev_col, prefixes, 'collections')
    ]

    for col in model.get('lookupFieldsStructure', []):
        next_model = list(filter(lambda x: x['tableName'] == col['tableName'], structure))
        if next_model:
            prefs = prepare_candidates([prepare_string(next_model[0]['caption']), prepare_string(split_camel_case(next_model[0]['tableName']))])
            columns_ref.extend(process_all_columns(next_model[0], structure, depth=depth + 1, prev_col=f"{col['name']}.", prefixes=prefs))
    
    return columns_ref


def describe_model(model, structure):
    if "Localization" in model['tableName'] or "ChangeTracking" in model['tableName']:
        return None

    description = {}

    model_names_candidates = prepare_candidates([prepare_string(model['caption']), prepare_string(split_camel_case(model['tableName']))])

    description["name"] = model['tableName']
    description["refs"] = model_names_candidates
    description["cols"] = process_all_columns(model, structure)

    return description


def generate():
    with open('src/structure/structure.json', 'r') as structure_file:
        structure = json.load(structure_file)

    descriptions = list(filter(lambda x: x is not None, map(lambda m: describe_model(m, structure), structure)))

    with open('src/structure/structure-description.json', 'w+') as f:
        json.dump(descriptions, f)


if __name__ == "__main__":
    generate()