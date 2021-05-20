import hashlib
from typing import List, Union, AbstractSet

import h5py
import numpy
import secrets
import getpass


class AttributeHandler:
    def __init__(self, dataset: h5py.Dataset):
        self.dataset: h5py.Dataset = dataset
        self.attrs: h5py.AttributeManager = dataset.attrs

    def does_attribute_exist(self, attribute_name: str) -> bool:
        attribute_names: AbstractSet[str] = self.attrs.keys()
        return attribute_name in attribute_names

    def delete_attribute(self, attribute_name: str) -> None:
        if self.does_attribute_exist(attribute_name):
            del self.attrs[attribute_name]
        else:
            print("Attribute %s was not deleted as it does not exist" % {attribute_name})

    def get_attribute(self, attribute_name: str) -> Union[str, float, int, bool, numpy.array]:
        if not self.does_attribute_exist(attribute_name):
            print('Attribute %s does not exist!' % {attribute_name})
            return None
        return self.attrs[attribute_name]

    def set_attribute(self, attribute_name: str, value: Union[str, float, int, bool, numpy.array]) -> None:
        if self.does_attribute_exist(attribute_name):
            self.delete_attribute(attribute_name)

        self.attrs.create(attribute_name, value)

    def set_reference_modality_to(self, reference: "AttributeHandler") -> None:
        self.set_reference_modality_to([reference])

    def set_reference_modality_to(self, references: List["AttributeHandler"]) -> None:
        ref_id: List[str] = []

        for reference in references:
            if reference.does_attribute_exist('id'):
                ref_id.append(reference.get_attribute('id'))
            else:
                print('Could not set reference_images because id is not present in at least one dataset')
                return

        self.set_attribute('reference_images', ref_id)

    def add_creator(self) -> None:
        creator: str = getpass.getuser()
        self.set_attribute('created_by', creator)

    def add_id(self) -> None:
        hashstr: str = secrets.token_hex(50)

        attributes: list = ['brain_id', 'brain_part_id', 'section_id', 'image_modality', 'creation_time',
                            'created_by', 'software', 'software_revision', 'software_parameters']
        for attribute in attributes:
            hashstr: str = hashstr + attribute

        hash_obj: hashlib.sha256 = hashlib.sha256()
        hash_obj.update(hashstr)
        self.set_attribute('id', hash_obj.hexdigest())

    def copy_all_attributes_to(self, dest: "AttributeHandler", exceptions: List[str] = None) -> None:
        if exceptions is None:
            exceptions = []

        fixed_exceptions: List[str] = [
            # Attributes that MUST be manually set
            "created_by", "creation_time", "dashboard_id", "id", "image_modality", "reference_images",
            "software", "software_revision", "software_parameters",
            # Attributes set by DB software
            "filename", "path", "scale"
        ]

        attribute_names: AbstractSet[str] = self.attrs.keys()
        for attribute_name in attribute_names:
            if attribute_name not in fixed_exceptions and attribute_name not in exceptions:
                self.copy_attribute_to(dest, attribute_name)

    def copy_attributes_to(self, dest: "AttributeHandler", attributes: List[str] = None) -> None:
        if attributes is None:
            return

        for attribute in attributes:
            self.copy_attribute_to(dest, attribute)

    def copy_attribute_to(self, dest: "AttributeHandler", attribute_name: str) -> None:
        dest.set_attribute(attribute_name, self.get_attribute(attribute_name))
