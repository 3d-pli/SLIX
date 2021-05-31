import hashlib
import typing
import h5py
import numpy
import secrets
import getpass

__all__ = ['AttributeHandler']


class AttributeHandler:
    def __init__(self, dataset: h5py.Dataset):
        """
        Initialize the AttributeHandler with a already opened HDF5 dataset.
        This dataset will be used for all operations of this class.

        Args:

            dataset: h5py dataset
        """
        self.dataset: h5py.Dataset = dataset
        self.attrs: h5py.AttributeManager = dataset.attrs

    def does_attribute_exist(self, attribute_name: str) -> bool:
        """
        Check if the attribute already exists in the HDF5 dataset.
        This has to be done before doing any operations because
        writing to an HDF5 attribute without properly deleting it first
        can result in errors.

        Args:

            attribute_name: Name of the attribute you want to check
                            in the dataset.

        Returns:

            None

        """
        attribute_names: typing.AbstractSet[str] = self.attrs.keys()
        return attribute_name in attribute_names

    def delete_attribute(self, attribute_name: str) -> None:
        """
        Delete an attribute from a HDF5 dataset.

        Args:

            attribute_name: Name of the attribute you want to delete in the
                            dataset.

        Returns:

            None

        """
        if self.does_attribute_exist(attribute_name):
            del self.attrs[attribute_name]
        else:
            print("Attribute %s was not deleted as it does not "
                  "exist" % {attribute_name})

    def get_attribute(self, attribute_name: str) -> \
            typing.Union[str, float, int, bool, numpy.array]:
        """
        Get an attribute from the HDF5 dataset.

        Args:

            attribute_name: Name of the attribute you want to get from the
                            dataset.

        Returns:

            Value from the dataset (string, float, int, bool
            or numpy.array) if the attribute es present. Otherwise None
            will be returned.
        """
        if not self.does_attribute_exist(attribute_name):
            print('Attribute %s does not exist!' % {attribute_name})
            return None
        return self.attrs[attribute_name]

    def set_attribute(self, attribute_name: str,
                      value: typing.Union[str, float, int,
                                           bool, numpy.array]) \
            -> None:
        """
        Set an attribute in the HDF5 dataset.

        Args:

            attribute_name: Name of the attribute you want to get from the
                            dataset.

            value: String, Float, Integer, Boolean or numpy array you
                   want to set in the HDF5 attribute.

        Returns:

            None

        """
        if self.does_attribute_exist(attribute_name):
            self.delete_attribute(attribute_name)

        self.attrs.create(attribute_name, value)

    def set_reference_modality_to(self, reference: "AttributeHandler") -> None:
        """
        When SLIX generates an image based on a SLI measurement, the original
        HDF5 file can be saves as a reference for the future.
        This method adds the reference file and dataset to the output.
        However, the input HDF5 and dataset must contain an id attribute.

        Args:

            reference: Reference AttributeHandler containing the dataset of
                       the input file.

        Returns:

            None

        """
        self.set_reference_modality_to([reference])

    def set_reference_modality_to(self,
                                  references: typing.List["AttributeHandler"])\
            -> None:
        """
        When SLIX generates an image based on a SLI measurement, the original
        HDF5 file can be saves as a reference for the future.
        This method adds the reference file and dataset to the output.
        However, the input HDF5 and dataset must contain an id attribute.

        Args:

            references: Reference list of AttributeHandlers
                        containing the dataset of the input file.

        Returns:

            None

        """
        ref_id: typing.List[str] = []

        for reference in references:
            if reference.does_attribute_exist('id'):
                ref_id.append(reference.get_attribute('id'))
            else:
                print('Could not set reference_images because id is not '
                      'present in at least one dataset')
                return

        self.set_attribute('reference_images', ref_id)

    def add_creator(self) -> None:
        """
        Adds the creator of the HDF5 file to the dataset.
        Returns: None

        """
        creator: str = getpass.getuser()
        self.set_attribute('created_by', creator)

    def add_id(self) -> None:
        """
        Computes a unique ID that will be added to the dataset of the HDF5
        file. The ID is a sha256 hash containing some of the attributes
        as well as a 50 digit randomized prefix.
        Returns: None

        """
        hashstr: str = secrets.token_hex(50)

        attributes: list = ['brain_id', 'brain_part_id', 'section_id',
                            'image_modality', 'creation_time',
                            'created_by', 'software', 'software_revision',
                            'software_parameters']
        for attribute in attributes:
            hashstr: str = hashstr + attribute

        hash_obj: hashlib.sha256 = hashlib.sha256()
        hash_obj.update(hashstr.encode('ascii'))
        self.set_attribute('id', hash_obj.hexdigest())

    def copy_all_attributes_to(self, dest: "AttributeHandler",
                               exceptions: typing.List[str] = None) -> None:
        """
        Copies all attributes from one AttributeHandler to another.
        Exceptions can be given in a list. In general, the following attributes
        will not be copied: "created_by", "creation_time", "id",
        "image_modality", "reference_images", "software", "software_revision",
        "software_parameters", "filename", "path", "scale"

        Args:

            dest: Destination where the attributes of this handler should be
                  copied to.

            exceptions: Exceptions in form of a list with strings.
                        Those attributes will not be copied when calling the
                        method.

        Returns:

            None

        """
        if exceptions is None:
            exceptions = []

        fixed_exceptions: typing.List[str] = [
            # Attributes that MUST be manually set
            "created_by", "creation_time", "dashboard_id", "id",
            "image_modality", "reference_images",
            "software", "software_revision", "software_parameters",
            # Attributes set by DB software
            "filename", "path", "scale"
        ]

        attribute_names: typing.AbstractSet[str] = self.attrs.keys()
        for attribute_name in attribute_names:
            if attribute_name not in fixed_exceptions \
                    and attribute_name not in exceptions:
                self.copy_attribute_to(dest, attribute_name)

    def copy_attributes_to(self, dest: "AttributeHandler",
                           attributes: typing.List[str] = None) -> None:
        """
        Copies given attributes from one AttributeHandler to another.

        Args:

            dest: Destination where the attributes of this handler should be
                  copied to.

            attributes: Attributes as a list of strings which will be copied.

        Returns:

            None

        """
        if attributes is None:
            return

        for attribute in attributes:
            self.copy_attribute_to(dest, attribute)

    def copy_attribute_to(self, dest: "AttributeHandler",
                          attribute_name: str) -> None:
        """
        Copy a single attribute from one AttributeHandler to another.

        Args:

            dest: Destination where the attributes of this handler should be
                  copied to.

            attribute_name: Attribute name which will be copied.

        Returns:

            None

        """
        dest.set_attribute(attribute_name, self.get_attribute(attribute_name))
