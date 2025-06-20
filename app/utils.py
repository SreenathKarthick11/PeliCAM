# IMPORTS
import os
import sys

#------------------------------------------------------------------------------

def resource_path(relative_path):
    """ 
    Parameters:
        relative_path (str): The relative path to the resource file.

    Function:
        Returns the absolute path to a resource file, handling both
        development and PyInstaller bundled environments.
    """
    try:
        base_path = sys._MEIPASS  # temp folder used by PyInstaller
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

#------------------------------------------------------------------------------

def load_stylesheet(path):
    """
    Reads and returns the content of a QSS stylesheet file.

    Args:
        path (str): Path to the stylesheet file.

    Returns:
        str: The stylesheet content as a string.
    """
    with open(path, "r") as file:
        return file.read()