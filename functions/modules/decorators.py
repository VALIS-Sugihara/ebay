def print_func(func):
    func_name = func.__name__

    def wrapper(*args, **kwargs):
        print('=== %s start ===' % (func_name,))
        print(args)
        print(kwargs)
        func(*args, **kwargs)
        print('=== %s end ===' % (func_name,))
    return wrapper
