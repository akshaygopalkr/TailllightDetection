import pandas as pd
import os
import cv2
import numpy as np
import json


def convert_masks(image_list):
    """
    Takes an image list of masked images and gets bounding box information from it
    :param image_list: List of masked images
    :return: List of bounding box labels from each image
    """

    bboxes = []

    for masked_image in image_list:

        mask = cv2.imread(masked_image, 0)
        analysis = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes = analysis[2][1:]
        box_list = []

        for box in boxes:
            x1, y1, x2, y2 = box[0], box[1], box[0]+box[2], box[1]+box[3]
            box_list.append([x1,y1,x2,y2])

        bboxes.append(box_list)

    return bboxes


def find_image_match(masked_image, image_dataset):

    """
    Finds the match of a masked image to an image from one of the datasets
    :param masked_image: The masked image name
    :param image_dataset: The image dataset
    :return: Image name from the image dataset that matches the
    """

    counter = 0
    for image in image_dataset:

        if masked_image in image:
            return image

        counter += 1


def get_original_images(image_names, dataset):

    """
    Finds original image from masked image name
    :param image_names: The masked image names
    :param dataset: The dataset the masked images belong to
    :return: List with the test
    """

    dataset_images = []
    matching_images = []
    full_path_images = {}

    # Go through and unpack all the images in each dataset folder
    for image_folder in os.listdir(os.path.join(os.curdir, 'OriginalImages', dataset)):

        images = os.listdir(os.path.join(os.curdir, 'OriginalImages', dataset, image_folder))
        full_path_images.update(dict(zip(images, [os.path.join(os.curdir, 'OriginalImages', dataset, image_folder, image_name) for image_name in images])))
        dataset_images.extend(images)



    for image_name in image_names:

        # Remove the mask_ part from the name and .png
        mask_image_name = image_name[5:-4]
        matching_image = find_image_match(mask_image_name, dataset_images)

        # Make sure the image has a matching name
        assert matching_image is not None
        matching_images.append(full_path_images[matching_image])

    return matching_images

def save_gt_images(images, bbox_labels):

    """
    Saves images that have masks into used Images folder
    """

    for image, bboxes in zip(images, bbox_labels):

        # Save each image
        image_name = os.path.normpath(image).split(os.path.sep)[-1]
        image_arr = cv2.imread(image)

        for bbox in bboxes:

            top_left, bottom_right = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))

            image_arr = cv2.rectangle(image_arr, top_left, bottom_right, (0,255,0), 2)

        cv2.imwrite(os.path.join(os.curdir, 'GroundTruthLabels', image_name), image_arr)


def get_bboxes_from_datasets(datasets):

    # TODO: Change image path to location of image in OriginalImages path
    """
    Returns the bounding boxes from each of the semantic images of the dataset
    :param datasets: The name of all the datasets in the MaskedImages of
    :return: Dictionary with key = image path and value = bounding boxes
    """

    labels_dict = {}

    for dataset in datasets:

        # Get all the masked images for this dataset
        image_names = os.listdir(os.path.join(os.curdir, 'MaskedImages', dataset))
        masked_images = [os.path.join(os.curdir, 'MaskedImages', dataset, image_name) for image_name in image_names]
        matching_images = get_original_images(image_names, dataset)

        image_pairs = list(zip([os.path.normpath(image).split(os.path.sep)[-1] for image in matching_images], masked_images))

        out_file = open("image_pairs.json", "w")

        json.dump(image_pairs, out_file, indent=6)

        out_file.close()


        # bbox_labels = convert_masks(masked_images)

        # save_gt_images(matching_images, bbox_labels)



if __name__ == '__main__':


    # Get list of datasets
    datasets = os.listdir(os.path.join(os.curdir, 'MaskedImages'))

    get_bboxes_from_datasets(datasets)