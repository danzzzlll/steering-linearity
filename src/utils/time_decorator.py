import time
from functools import wraps

def timecount(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        cls_name = args[0].__class__.__name__ if args else ''
        print(f"[{cls_name}.{func.__name__}] Starting...")
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"[{cls_name}.{func.__name__}] Done in {duration:.3f} sec")
        return result
    return wrapper
