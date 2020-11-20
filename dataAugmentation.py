from PIL import ImageEnhance
import os
import numpy as np
from PIL import Image


def brightnessEnhancement(root_path, img_name):
    """
    Enhance the brightness of picture, Here we set the multiple as 1.5
    :param root_path:
    :param img_name:
    :return: Enhanced image
    """
    image = Image.open(os.path.join(root_path, img_name))
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.5
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened


def contrastEnhancement(root_path, img_name):
    """
    Enhance the contrast of image. Here we set the multiple as 1.5
    :param root_path:
    :param img_name:
    :return: Enhanced image
    """
    image = Image.open(os.path.join(root_path, img_name))
    enh_con = ImageEnhance.Contrast(image)
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted


def rotation(root_path, img_name):
    """
    Rotate the image by in 90 degrees, 180 degrees, 270 degrees randomly
    :param root_path:
    :param img_name:
    :return: Enhanced image
    """
    img = Image.open(os.path.join(root_path, img_name))
    random_angle = np.random.randint(1, 4) * 90
    rotation_img = img.rotate(random_angle)
    return rotation_img


def flip(root_path, img_name):
    """
    Flip the image in horizontal symmetry
    :param root_path:
    :param img_name:
    :return: Enhanced image
    """
    img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return filp_img


def saveOriginal(root_path, img_name):
    """
    Copy the original image
    :param root_path:
    :param img_name:
    :return: original image
    """
    img = Image.open(os.path.join(root_path, img_name))
    return img


def createImage(image_dir, save_dir):
    """
    Save all the enhanced data
    :param image_dir: 
    :param save_dir: 
    """
    i = 0
    for name in os.listdir(image_dir):
        i = i + 1
        save_name = ["cesun" + str(i) + ".jpg", "flip" + str(i) + ".jpg", "brightnessE" + str(i) + ".jpg",
                     "rotate" + str(i) + ".jpg", "original" + str(i) + ".jpg"]
        saveImage = contrastEnhancement(image_dir, name)
        saveImage1 = flip(image_dir, name)
        saveImage2 = brightnessEnhancement(image_dir, name)
        saveImage3 = rotation(image_dir, name)
        saveImage4 = saveOriginal(image_dir, name)
        save_image = [saveImage, saveImage1, saveImage2, saveImage3, saveImage4]
        for index,j in enumerate(save_image):
            j.save(os.path.join(save_dir, save_name[index]))