import os

def mkdir_p(path: str):
    try:
        os.makedirs(path, exist_ok=True)  # python >= 3.2
    except OSError as exc:  # Python > 2.5
        raise