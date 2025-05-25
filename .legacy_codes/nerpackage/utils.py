# a/utils.py
import os

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)