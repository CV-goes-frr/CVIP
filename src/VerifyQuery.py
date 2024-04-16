import platform
import re

from .exceptions.InvalidBrackets import InvalidBracketsException
from .exceptions.InvalidFlags import InvalidFlagsException
from .exceptions.WrongFileFormat import WrongFileFormatException
from .exceptions.WrongFilename import WrongFilenameException


class VerifyQuery:
    def get_flags(string):
        """
        Extracts the flags from the given string.

        Args:
            string: The string to extract the flags from.

        Returns:
            A list of flags.
        """
        matches = re.findall(r"-.*?=", string)

        # Extract the text between the characters "[" and "=".
        text = [match[1:-1] for match in matches]
        return text

    def get_filenames_and_labels(string):
        """
        Extracts the filenames and labels from the given string.

        Args:
            string: The string to extract the filenames and labels from.

        Returns:
            A list of filenames and labels.
        """
        filenames = []
        matches = re.findall(r"=.*?]", string)

        # Extract the filenames from the matches.
        for match in matches:
            filename = match[1:-1]
            filenames.append(filename)
        return filenames

    def remove_names(string):
        """
        Removes the filenames and labels from the given string.

        Args:
            string: The string to remove the filenames and labels from.

        Returns:
            The string with the filenames and labels removed.
        """
        filenames = VerifyQuery.get_filenames_and_labels(string)

        # Remove everything except for the found filenames from the string.
        new_string = string
        for filename in filenames:
            new_string = new_string.replace(filename, "", 1)

        return new_string

    def validate_pairing_brackets(string):
        """
        Validates a string to ensure that it matches the pattern [anything]filter[anything] repeated several times.

        Args:
            string: The string to validate.

        Returns:
            True if the string is valid, False otherwise.
        """

        array = [i.split("]") for i in string.split("[")][1:]
        for item in range(1, len(array), 2):
            if array[item][1] != '':
                return False
        return True

    def validate_brackets(string):
        """
        Validates if the given string contains valid brackets.

        Args:
            string: The string to validate.

        Returns:
            True if the string contains valid brackets, False otherwise.
        """

        # Check if the string contains a correct number of brackets.
        if (string.count("[") + string.count("]")) % 4 != 0:
            # return False
            raise InvalidBracketsException("numbers of opening and closing brackets are not equal")

        # Check if the string contains any brackets inside brackets.
        if not (re.search(r"\[.*\].*\[.*\]", string)):
            # return False
            raise InvalidBracketsException("brackets inside brackets")

        # Check if the string contains any brackets without anything between them.
        if re.search(r"\[\]", string):
            # return False
            raise InvalidBracketsException("empty brackets found")

        string_without_filenames = VerifyQuery.remove_names(string)
        if " " in string_without_filenames:
            # return False
            raise InvalidBracketsException("empty spaces (not in filenames and label) are not allowed")

        if not (VerifyQuery.validate_pairing_brackets(string)):
            # return False
            raise InvalidBracketsException("wrong brackets order, please, follow the [label]operation[label] structure")

        return True

    def validate_flags(string):
        """
        Validates if the given string contains valid flags.

        Args:
            string: The string to validate.

        Returns:
            True if the string contains valid flags, False otherwise.
        """
        flags = VerifyQuery.get_flags(string)
        for flag in flags:
            if flag not in "io":
                # return False
                raise InvalidFlagsException(flag)
        return True

    def validate_filenames(string):
        """
        Validates if the given string contains valid filenames based on platform.

        Args:
            string: The string to validate.

        Returns:
            True if the string contains valid filenames, False otherwise.
        """
        system = platform.system()
        filenames = VerifyQuery.get_filenames_and_labels(string)
        for filename in filenames:
            if system == "Linux":
                if "\0" in filename:
                    # return False
                    raise WrongFilenameException(filename)
            elif system == "Darwin":
                if filename[0] == "." or ":" in filename:
                    # return False
                    raise WrongFilenameException(filename)
        return True

    def validate_formats(string):
        """
        Validates if the given string contains valid file formats.

        Args:
            string: The string to validate.

        Returns:
            True if the string contains valid formats, False otherwise.
        """

        filenames = VerifyQuery.get_filenames_and_labels(string)
        formats = ["png", "jpeg", "jpg", "mp4"]
        for filename in filenames:
            if "." in filename:
                format = filename.split(".")[-1]
            else:
                return True
            if format not in formats:
                # return False
                raise WrongFileFormatException(filename)
        return True

    def check(string):
        """
        Validates if the given string is valid.

        Args:
            string: The string to validate.

        Returns:
            True if the string is valid, False otherwise.
        """
        return (
                VerifyQuery.validate_brackets(string) and VerifyQuery.validate_flags(string)
                and VerifyQuery.validate_filenames(string) and VerifyQuery.validate_formats(string)
                )