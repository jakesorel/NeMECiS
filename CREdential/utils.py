import os

def mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
