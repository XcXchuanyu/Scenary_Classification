import linecache
import os
import dataAugmentation


def remove_ambiguous_image(file_path=r'./data_cleaning.txt', data_dir = "./train"):
    """
    This function will filter those images whose feature is ambiguous. data_cleaning.txt contains filtered image
    :param file_path:
    :return:
    """

    def get_line_context(file_path, line_number):
        return linecache.getline(file_path, line_number).strip()

    for i in range(1, 87):
        line_number = i
        if os.path.exists(data_dir + get_line_context(file_path, line_number)):
            os.remove(data_dir + get_line_context(file_path, line_number))


def data_augmentation(train_dir="./train/", save_dir='./augmented/'):
    """
    In this function, we will applied data augmentation(contains BrightnessEnhancement
    ContrastEnhancement, RandomRotation, Flip ) on train data and saved it in to a different directory
    which contains augmented train data the size of dataset is five times to the size of original dataset.
    :param train_dir: the directory of train data
    :param save_dir: the directory you want to save augmented data
    """
    categories = os.listdir(train_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for cat in categories:
        if not os.path.exists(save_dir + cat + "/"):
            os.mkdir(save_dir + cat + "/")
        dataAugmentation.createImage(os.path.join(train_dir,cat), os.path.join(save_dir, cat))


if __name__ == "__main__":
    remove_ambiguous_image()
    data_augmentation()
