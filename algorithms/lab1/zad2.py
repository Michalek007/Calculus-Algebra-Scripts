import re


with open('../data/plik.txt', 'r') as f:
    data = f.read()
print(data)


def delete_words(input_data: str, words_list: list):
    for word in words_list:
        input_data = input_data.replace(word, '')
    return input_data


formatted_data = delete_words(data, ['Macbeth', 'King', 'will'])

print('-----------------------------------------')
print(formatted_data)


def convert_words(input_data: str, words_dict: dict):
    for old_word, new_word in words_dict.items():
        input_data = input_data.replace(old_word, new_word)
    return input_data


formatted_data = convert_words(data, {'Macbeth': 'Dziekan', 'King': 'Jackolo', 'will': 'might'})
print('-----------------------------------------')
print(formatted_data)


def delete_words_re(input_data: str, words_list: list):
    for word in words_list:
        input_data = re.sub(f"([^ ]*{word}[^ )", '', input_data)
    return input_data


formatted_data = delete_words(data, ['Macbeth', 'King', 'will'])
print('-----------------------------------------')
print(formatted_data)


def convert_words_re(input_data: str, words_dict: dict):
    for old_word, new_word in words_dict.items():
        input_data = re.sub(f"([^ ]*{old_word}[^ )", new_word, input_data)
    return input_data


formatted_data = convert_words(data, {'Macbeth': 'Dziekan', 'King': 'Jackolo', 'will': 'might'})
print('-----------------------------------------')
print(formatted_data)
