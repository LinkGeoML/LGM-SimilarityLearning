import re

# --- GENERIC Utils ---
FIRST_CAP_RE = re.compile('(.)([A-Z][a-z]+)')
ALL_CAP_RE = re.compile('([a-z0-9])([A-Z])')


def camel_to_underscore(name: str):
    """
    This function converts a camel Cased name to it's underscore version.

    E.g: thisIsACamelCaseName --> this_is_a_camel_case_name

    Parameters
    ----------
    name : str
    A name that we want to transform from camelCase to it's underscored version

    Returns
    -------
    str

    """
    s1 = FIRST_CAP_RE.sub(r'\1_\2', name)

    return ALL_CAP_RE.sub(r'\1_\2', s1).lower()


def underscore_to_camel(name):
    """
    This function converts a underscored name to it's camel Cased version
    Parameters
    ----------
    name: str
    The name that we want to change

    Returns
    -------
    str

    """
    return ''.join(x.capitalize() or '_' for x in name.split('_'))
