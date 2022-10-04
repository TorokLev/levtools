from . import utils as u


class ExclusiveSchemaValidator:

    def __init__(self, debug=False):
        self.debug = debug

    def validate(self, obj, schema):
        if self.debug:
            u.pprint(schema)

        self.definitions = schema['definitions'] if 'definitions' in schema else {}
        self.validate_recursive(obj, schema)

    def validate_recursive(self, obj, schema):
        if self.debug:
            print("obj: ", obj)
        check_obj_level(obj, schema)

        for schema_property_key, schema_property_value in schema['properties'].items():
            obj_prop = obj[schema_property_key]

            if list(schema_property_value) == ["$ref"]:
                def_key = schema['properties'][schema_property_key]['$ref'][len('#/definitions/'):]
                if self.debug:
                    print("recurse: ", schema_property_key)
                self.validate_recursive(obj[schema_property_key], schema=self.definitions[def_key])

            elif schema_property_value['type'] == 'object':
                if self.debug:
                    print("recurse: ", schema_property_key)
                self.validate_recursive(obj_prop, schema_property_value)
            elif schema_property_value['type'] == 'array':
                if self.debug:
                    print("array recurse: ", schema_property_key)
                assert type(obj_prop) == list
                if '$ref' in schema_property_value['items']:
                    def_key = schema_property_value['items']['$ref'][len('#/definitions/'):]
                    for item in obj_prop:
                        self.validate_recursive(item, schema=self.definitions[def_key])
                else:
                    for item in obj_prop:
                        self.validate_recursive(item, schema=schema_property_value['items'])


def check_obj_level(obj, schema):
    assert schema['type'] == 'object', f"Not object level check but {schema['type']}"

    if 'required' not in schema:  # TODO: if no required section in an object, we don't check for fields
        return

    for req_prop in schema['required']:
        assert req_prop in obj, f'Missing property {req_prop} from obj {obj}'

    for obj_key in obj.keys():
        assert obj_key in schema['required'], f"Additional property {obj_key} found in object {obj}"
