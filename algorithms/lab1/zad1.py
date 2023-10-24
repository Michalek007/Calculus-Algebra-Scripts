import os


def number_of_files(directory: str):
    main_dir = os.listdir(directory)
    count = 0
    for item in main_dir:
        if os.path.isfile(directory + f'\\{item}'):
            count += 1
    return count


def display_sub_dirs(directory: str, iteration: int = 0):
    sub_dirs = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    for item in sub_dirs:
        subdir_files = os.listdir(item)
        if iteration == 1:
            print('\t' + item)
        else:
            print(item)
        for file in subdir_files:
            if os.path.isfile(os.path.join(item, file)):
                if iteration == 1:
                    print('\t\t' + file)
                else:
                    print('\t' + file)
        display_sub_dirs(item, 1)

# def display_sub_dirs(directory: str):
#     list_dirs = [d[0] for d in os.walk(directory)]
#     tab = ''
#     for item in list_dirs:
#         subdir_files = os.listdir(item)
#         # print(list_dirs[list_dirs.index(item) - 1])
#         print(item)
#         if list_dirs.index(item) == 0:
#             pass
#         elif item in list_dirs[list_dirs.index(item)-1]:
#             # print(list_dirs[list_dirs.index(item)-1])
#             # print(item)
#             tab += '\t'
#         # print(tab + item)
#         for file in subdir_files:
#             if os.path.isfile(os.path.join(item, file)):
#                 # print(tab + '\t' + file)
#                 pass

print(number_of_files('C:\\dev'))

display_sub_dirs('C:\\dev')
