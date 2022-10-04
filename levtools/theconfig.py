#!/usr/bin/env python

import argparse
import os

from .  import utils as u

DEFAULT_JSON = "defaults.json"

class HierarchicalDefaultDict:
    NOT_FOUND = 0
    FOUND = 1

    def __init__(self, input_dict, hierarchy_sep='.', missing_section_exception=False):
        self.hierarchy_sep = hierarchy_sep
        self.storage = input_dict
        self.missing_section_exception = missing_section_exception

    def get(self, key, **kwargs):

        u.assert_for_unknown_params(kwargs, {'default'})
        
        has_found, value = self._recursive_get(self.storage, key)

        if has_found == self.FOUND:
            return value  # returns pure sub directory and not a HierarchicalDefaultDict if dict found

        if 'default' in kwargs:
            return kwargs['default']

        raise Exception("Missing key: " + key + " from dictionary: " + str(self.storage))

    def __contains__(self, key):
        found, value = self._recursive_get(self.storage, key)
        return found == self.FOUND

    def __getitem__(self, key):
        found, value = self._recursive_get(self.storage, key)

        if found == self.FOUND:
            return value  # returns pure sub directory and not a HierarchicalDefaultDict if dict found

        raise Exception("Missing key: " + key + " from dictionary: " + str(self.storage))

    def _recursive_get(self, dict_part, key):
        if self.hierarchy_sep in key:
            key0 = key.split(self.hierarchy_sep)[0]
            if key0 in dict_part:
                if isinstance(dict_part[key0], dict):
                    return self._recursive_get(dict_part[key0], key[len(key0) + 1:])
            else:
                return self.NOT_FOUND, None
        else:
            if key in dict_part.keys():
                return self.FOUND, dict_part[key]
            else:
                return self.NOT_FOUND, None

    def _recursive_set(self, dict_part, key, value):
        if self.hierarchy_sep in key:
            key0 = key.split(self.hierarchy_sep)[0]

            if key0 in dict_part:
                if isinstance(dict_part[key0], dict):
                    return self._recursive_set(dict_part[key0], key[len(key0) + 1:], value)
            else:
                if self.missing_section_exception:
                    raise Exception("Section doesn't exist in dict: " + str(key))

                dict_part[key0] = dict()
                return self._recursive_set(dict_part[key0], key[len(key0) + 1:], value)
        else:
            dict_part[key] = value

    def __setitem__(self, key, value):
        self._recursive_set(self.storage, key, value)

    def update(self, input_dict):
        if isinstance(input_dict, dict):
            input_dict = Config(input_dict)

        for key in input_dict.storage.keys():
            if key in self.storage:
                if isinstance(self.storage[key], dict) and isinstance(input_dict.storage[key], dict):
                    self.get(key).update(input_dict.get(key))
                    continue
            self.storage[key] = input_dict.storage[key]


class Config(HierarchicalDefaultDict):
    # relative path handling is added to this class over the HierarchicalDefaultDict
    def __init__(self, input_dict, config_dir="", hierarchy_sep='.', missing_section_exception=False):
        super(Config, self).__init__(input_dict=input_dict, hierarchy_sep=hierarchy_sep, missing_section_exception=missing_section_exception)
        self.config_dir = u.path(config_dir)

    def get_path(self, key, create_dirs=False, create_as_dir=False, **kwargs):
        """
        Returns path translated to operating system.
        """
        path = self.get(key, **kwargs)
        return u.path_resolver(path, create_dirs, create_as_dir, rebase_dir=self.config_dir)

    def get(self, key, **kwargs):
        value = super(Config, self).get(key, **kwargs )
        return Config(input_dict=value, config_dir=self.config_dir) if isinstance(value, dict) else value

    def __getitem__(self, key):
        return self.get(key)

    def update_with_environ_variables(self):
        for key in os.environ:
            key_translated = key.replace("__", self.hierarchy_sep)
            if self.__contains__(key_translated):
                self[key_translated] = os.getenv(key)


def load(cfg_filename, use_default=True, **kwargs):
    cfg_dir = os.path.dirname(cfg_filename)
    config = Config(config_dir=cfg_dir, input_dict={})

    if use_default:
        defaults_json_filename = os.path.join(cfg_dir, DEFAULT_JSON)
        if os.path.exists(defaults_json_filename):
            config.update(Config(input_dict=u.json_load_from_file(defaults_json_filename)))

    config_actual = Config(input_dict=u.json_load_from_file(cfg_filename))
    config.update(config_actual)

    config.update_with_environ_variables()

    config.update(Config(input_dict=kwargs))

    return config

