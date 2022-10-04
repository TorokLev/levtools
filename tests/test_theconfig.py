import os

from levtools import theconfig
from levtools import utils as u


def test_hierarchical_get():
    config = theconfig.Config({'key': {'subkey': 1234}})
    assert config['key.subkey'] == 1234
    assert config.get('key.subkey') == 1234
    assert config['key.subkey'] == 1234
    assert config['key']['subkey'] == 1234


def test_hierarchical_recall_w_type_correct_default_value():
    config = theconfig.Config({'key': {'subkey': 1234}})
    assert config.get('key.missing_key', default=343) == 343
    assert config.get('key.missing_key', default="stringvalue") == "stringvalue"


def test_get_config_part():
    config = theconfig.Config({'key': {'subkey': 1234}})
    assert config['key'].storage == {'subkey': 1234}


def test_hierarchical_set():
    config = theconfig.Config({'key1': 'a', 'key2': {'subkey': 'b'}})
    config['key1'] = 'c'
    assert config['key1'] == 'c'

    config['key2.subkey'] = 'd'
    assert config['key2.subkey'] == 'd'
    assert config['key2']['subkey'] == 'd'

    config['key2']['subkey'] = 'e'
    assert config['key2.subkey'] == 'e'
    assert config['key2']['subkey'] == 'e'


def test_environment_variable_overloader():
    if u.is_windows():
        config = theconfig.Config({'KEY1': 'a', 'KEY2': {'SUBKEY': 'b'}})
        os.environ['KEY1'] = 'c'
        os.environ['KEY2__SUBKEY'] = 'd'

        config.update_with_environ_variables()

        assert config['KEY1'] == 'c'
        assert config['KEY2.SUBKEY'] == 'd'
        assert config['KEY2']['SUBKEY'] == 'd'

    if u.is_linux():
        config = theconfig.Config({'key1': 'a', 'key2': {'subkey': 'b'}})
        os.environ['key1'] = 'c'
        os.environ['key2__subkey'] = 'd'

        config.update_with_environ_variables()

        assert config['key1'] == 'c'
        assert config['key2.subkey'] == 'd'
        assert config['key2']['subkey'] == 'd'


def test_hierarchical_set_with_missing_section():
    config = theconfig.Config({'key1': 'a', 'key2': {'subkey': 'b'}}, missing_section_exception=True)

    try:
        config['missingsection.newkey'] = 'something'
        exception = False
    except Exception as _:
        exception = True

    assert exception


def test_hierarchical_set_creating_missing_sections():
    config = theconfig.Config({'key1': 'a', 'key2': {'subkey': 'b'}}, missing_section_exception=False)
    config['missingsection.newkey'] = 'something'

    assert config.storage['missingsection']['newkey'] == 'something'


def test_if_all_can_be_found():
    config = theconfig.Config({'key1': 'a', 'key2': {'subkey': 'b'}})

    assert 'key1' in config
    assert 'key2' in config
    assert 'key2.subkey' in config
    assert config['key2'].storage == {'subkey': 'b'}


def test_update_subdict_value():
    config = theconfig.Config({'key1': 'a', 'key2': {'subkey': 'b'}})
    new_config = theconfig.Config({'key2': {'subkey': 'c'}})

    config.update(new_config)
    print("--")
    print(config.storage)
    print("==")

    assert config['key1'] == 'a'
    assert config['key2.subkey'] == 'c'
    assert config['key1'] == 'a'


def test_update_dict_by_value():
    config = theconfig.Config({'key1': 'a', 'key2': {'subkey': 'b'}})
    new_config = theconfig.Config({'key2': 'c'})

    config.update(new_config)

    assert config['key2'] == 'c'
    assert config['key1'] == 'a'


def test_update_key_by_dict():
    config = theconfig.Config({'key1': 'a', 'key2': {'subkey': 'b'}})
    new_config = theconfig.Config({'key1': {'key3': 'c'}})

    config.update(new_config)

    assert config['key1']['key3'] == 'c'
    assert config['key2']['subkey'] == 'b'


def test_update_empty():
    config = theconfig.Config({})
    new_config = theconfig.Config({'key1': 'a', 'key2': {'key3': 'c'}})

    config.update(new_config)

    assert config.storage == {'key1': 'a', 'key2': {'key3': 'c'}}


def test_package_name_substitution():
    config = theconfig.Config({'dir': '<numpy>'})
    assert os.path.isabs(config.get_path('dir'))


def test_loading_defaults_will_overwrite_properly(tmp_path):
    # given

    DEFAULT_JSON = str(tmp_path / theconfig.DEFAULT_JSON)
    TEST_JSON = str(tmp_path / "test.json")

    u.json_write_to_file({
        "key": {
            "sub_key_to_overwrite": "sub_key_overwritten"
        },
        "new_key": "new_key"
    },
        TEST_JSON)

    u.json_write_to_file({
        "key": {
            "sub_key_to_overwrite": "default_key",
            "sub_key_old_key": "old_key"
        },
        "old_key": "old_key"
    },
        DEFAULT_JSON)

    expected_config = {
        "key": {
            "sub_key_to_overwrite": "sub_key_overwritten",
            "sub_key_old_key": "old_key"
        },
        "old_key": "old_key",
        "new_key": "new_key"
    }

    # when
    actual_config = theconfig.load(TEST_JSON).storage

    # then
    assert actual_config == expected_config
