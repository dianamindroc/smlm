def is_nested_dict(d):
    """
    Check if a variable is a nested dictionary.

    :param d: The variable to check.
    :return: True if it's a nested dictionary, False otherwise.
    """
    # First check if the variable itself is a dictionary
    if not isinstance(d, dict):
        return False

    # Check if any value in the dictionary is also a dictionary
    for value in d.values():
        if isinstance(value, dict):
            return True

    return False