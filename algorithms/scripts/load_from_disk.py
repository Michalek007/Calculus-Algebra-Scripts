

def load_from_disk(file_format: str, *args):
    def decorated(func):
        def wrapper(*args):
            print(args)
            if len(args) == 1:
                return func(args[0])
            try:
                with open(args[1] + '.' + file_format, 'r') as f:
                    print('Reading value from file...')
                    return int(f.read())
            except FileNotFoundError:
                with open(args[1] + '.' + file_format, 'w') as f:
                    print('Calculating data...')
                    data = func(args[0])
                    f.write(str(data))
                    return data
        return wrapper
    return decorated


@load_from_disk('csv')
def square(n):
    return n*n


if __name__ == '__main__':
    print(square(20, 'file'))
    print(square(20, 'file'))
    print(square(15, 'file2'))
    print(square(15, 'file2'))
    print(square(20))
