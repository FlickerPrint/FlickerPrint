#!/usr/bin/env python
import configparser as cp
from pathlib import Path

import strictyaml as yaml

""" Reader of constants files and store of global configuration values.

This provides a Singleton ``Config``, ensuring that this class is only created once.

Provides
--------

config
    Configuration for all the steps of the granule explorer, this can either by provided
    by a ``.yaml`` file or updated by the ``django`` app.

https://stackoverflow.com/questions/48351139/loading-config-from-class-in-such-a-way-that-it-behaves-like-a-property
"""

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CURRENT_DIR / "defaults.yaml"

SCHEMA = yaml.Map(
    {
        "workflow": yaml.Map(
            {
                "image_dir": yaml.Str(),
                "image_regex": yaml.Str(),
                "experiment_name": yaml.Str(),
            }
        ),
        "image_processing": yaml.Map(
            {
                "pixel_size": yaml.Float(),
                "method": yaml.Str(),
                "smoothing": yaml.Float(),

                "granule_minimum_radius": yaml.Float(),
                "granule_maximum_radius": yaml.Float(),
                "granule_minimum_intensity": yaml.Float(),
                "fill_threshold": yaml.Float(),
                "tracking_threshold": yaml.Float(),
                "granule_images": yaml.Bool(),
                "use_fft": yaml.Bool(),
            }
        ),
        "spectrum_fitting": yaml.Map(
            {
                "experimental_spectrum": yaml.Str(), 
                "fitting_orders": yaml.Int(),
                "temperature": yaml.Float(),
                "plot_spectra_and_heatmaps": yaml.Bool(),
            }
        ),
        "plotting": yaml.Map(
            {
                "latex": yaml.Str(),
            }
        )
    }
)


class _Config:
    def __init__(self, config_location: Path = None):
        """ Storage for the configuration parts of the document. """

        self.store = self.parse_config(config_location)
        self.defaults = self.parse_config(DEFAULT_CONFIG)
        self._validate()

    def refresh(self, config_location):
        """ Reload the configuration file. """
        config_location = Path(config_location)
        if not config_location.exists():
            raise FileNotFoundError(
                f"Unable to find configuration file at {config_location}: \n"
                f"{config_location.resolve()}"
            )
        self.store = self.parse_config(config_location)

    def parse_config(self, config_location: Path):
        """ Load the configuration variables from the yaml config. """
        if config_location is None or not config_location.exists():
            return {}
        with open(config_location, mode="r") as f:
            yaml_string = "".join(f.readlines())
            if not yaml_string:
                raise ValueError(
                    f"Attempting to read empty configuration file - {config_location}"
                )
            store = yaml.load(yaml_string)

        return store.data

    @staticmethod
    def _retrieve_keys(store, *keys):
        """ Get stacked keys from a dict. """
        value = store
        try:
            for key in keys:
                value = value[key]
            sucsess = True
        except KeyError:
            sucsess = False
            value = None

        return sucsess, value

    def __call__(self, *name):
        """ Allow us to access values by calling the function with the keys. """
        # First try to get the value in the ``store``, the user provided values
        sucsess, value = self._retrieve_keys(self.store, *name)
        if sucsess:
            return value

        # If not, fall back to the default values
        sucsess, value = self._retrieve_keys(self.defaults, *name)
        if sucsess:
            return value

        raise ValueError(f"{name} not found in configuration.")

    def _validate(self):
        """All of the entries in the user configuration file must also have an entry in
        the defaults list."""

        for label_section, section in self.store.items():
            default_values = self.defaults[label_section]

            for key, user_value in section.items():
                if key not in default_values:
                    raise ValueError(
                        f"{label_section}:{key} not found in default file "
                        "but is used in config.yaml"
                    )

    def _aggregate_all(self):
        """Move all of the configuration values into a single dictionary -> string.

        This is useful to parse the configuration values into yaml/json for storing in a
        configuration file.
        """
        merged_dict = {}
        summary = ""
        for label_section, section in self.defaults.items():
            if label_section in self.store.keys():
                user_values = self.store[label_section]
            else:
                user_values = self.defaults[label_section]
            merged_dict[label_section] = {}

            summary += f"[{label_section}]\n"
            for key, default_value in section.items():
                merged_val = user_values[key] if key in user_values else default_value
                merged_dict[label_section][key] = merged_val

                if key in user_values:
                    user_val = user_values[key]
                    summary += f"{key}: {default_value} -> {user_val}\n"
                else:
                    summary += f"{key}: {default_value}\n"
            summary += "\n"
        return yaml.as_document(merged_dict).as_yaml(), summary

    def _as_dict(self):
        """Move all of the configuration values into a single dictionary.

        This is useful to parse the configuration values into yaml/json for storing in a
        configuration file.
        """
        merged_dict = {}
        for label_section, section in self.defaults.items():
            if label_section in self.store.keys():
                user_values = self.store[label_section]
            else:
                user_values = self.defaults[label_section]

            for key, default_value in section.items():
                merged_val = user_values[key] if key in user_values else default_value
                merged_dict[key] = merged_val

        return merged_dict

def _pretty_print(values):
    """ Print the configuration file for debugging. """
    for key, value in values.items():
        print(f"{key}:")
        try:
            for key_one, value_one in value.items():
                print(f"    {key_one} = {value_one}")
        except AttributeError:
            print(f"    {value}")


def write_config(entries: dict, save_location: Path):
    """Write a configuration file to disk.

    This copies from the defaults files and should retain the default values where they
    are not explicitly given and retain the comments.
    """
    config_yaml = _new_config_to_yaml(entries=entries)
    with open(save_location, "w") as f:
        f.write(config_yaml)


def _new_config_to_yaml(entries):
    """Update a yaml config using the new values

    This copies from the defaults files and should retain the default values where they
    are not explicitly given and retain the comments.
    """

    with open(DEFAULT_CONFIG, "r") as f:
        yaml_string = "".join(f.readlines())
        modified_config = yaml.load(yaml_string, SCHEMA)

    if entries is not None:
        for new_key, new_val in entries.items():
            modified_config = _update_config(modified_config, new_key, new_val)

    return modified_config.as_yaml()


def _update_config(config, key, value):
    """ Update a key in the configuration with a new value. """
    # Look through all the sections
    for section_name, section in config.items():

        # If the entry is not contained in this section, then skip
        if key not in section:
            continue

            # print(f"{label} = {val}")
        config[section_name][key] = value
        # print(config[section_name][label])
        return config

    # If we reach the end then we have not updated the values
    raise ValueError(f"Unable to place {key} in config")


config = _Config(Path("config.yaml"))
