import csv


def write_info_to_csv(output_file_path, arr, mode='w'):
    """
    write array to csv
    :param output_file_path: path to which the file should be saved
    :param arr: containing all row values that should be written
    :param mode: csv writer mode: 'w' for writing to a new file, 'a' for appending an existing one
    """
    with open(output_file_path, mode=mode, newline='') as output_file:
        writer = csv.writer(output_file, delimiter=',')
        writer.writerow(arr)


def save_string_to_txt_file(filepath, string):
    """
    save string to given filepath
    :param filepath: global path where to save the file
    :param string: string to write to the file
    """
    with open(filepath, 'w', encoding="utf-8") as text_file:
        text_file.write(string)
