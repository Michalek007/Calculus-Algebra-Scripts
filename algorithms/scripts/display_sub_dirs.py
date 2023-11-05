import os


def number_of_files(directory: str):
    main_dir = os.listdir(directory)
    count = 0
    for item in main_dir:
        if os.path.isfile(directory + f'\\{item}'):
            count += 1
    return count


def tab_generator(count: int):
    tab_str = ''
    for _ in range(count):
        tab_str += '\t'
    return tab_str


def display_sub_dirs(directory: str, iteration: int = 0):
    sub_dirs = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    for item in sub_dirs:
        subdir_files = os.listdir(item)
        print(tab_generator(iteration) + item)
        for file in subdir_files:
            if os.path.isfile(os.path.join(item, file)):
                print(tab_generator(iteration) + '\t' + file)
        display_sub_dirs(item, iteration+1)


print(number_of_files('C:\\dev'))

display_sub_dirs('C:\\dev')
