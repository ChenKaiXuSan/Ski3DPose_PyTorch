"""
File: utils.py
Project: utils
Created Date: 2023-09-03 13:02:25
Author: chenkaixu
-----
Comment:

Have a good code time!
-----
Last Modified: 2023-09-03 13:03:05
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

"""

import os, shutil


def del_folder(path, *args):
    """
    delete the folder which path/version

    Args:
        path (str): path
        version (str): version
    """
    if os.path.exists(os.path.join(path, *args)):
        shutil.rmtree(os.path.join(path, *args))


def make_folder(path, *args):
    """
    make folder which path/version

    Args:
        path (str): path
        version (str): version
    """
    if not os.path.exists(os.path.join(path, *args)):
        os.makedirs(os.path.join(path, *args))
        print("success make dir! where: %s " % os.path.join(path, *args))
    else:
        print("The target path already exists! where: %s " % os.path.join(path, *args))
