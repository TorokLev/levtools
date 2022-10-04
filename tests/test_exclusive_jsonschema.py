from levtools import exclusive_jsonschema as ej


def test_required_attribues_check_is_passes():
    obj = {'a': '1'}
    schema = {'type': 'object',
              'properties': {
                  'a': {'type': 'string'}
              },
              'required': ['a']
              }

    ej.ExclusiveSchemaValidator().validate(obj, schema)


def test_missing_attribues_check_is_fails():

    obj = {'b': '1'}
    schema = {'type': 'object',
              'properties': {
                  'a': {'type': 'string'}
              },
              'required': ['a']
              }

    try:
        ej.ExclusiveSchemaValidator().validate(obj, schema)
        fails = False
    except:
        fails = True

    assert fails


def test_additional_attribues_check_is_fails():

    obj = {'a': '1', 'b': '2'}

    schema = {'type': 'object',
              'properties': {
                  'a': {'type': 'string'}
              },
              'required': ['a']
              }

    try:
        ej.ExclusiveSchemaValidator().validate(obj, schema)
        fails = False
    except:
        fails = True

    assert fails
