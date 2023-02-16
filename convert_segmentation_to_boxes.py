import pandas as pd
import os
import cv2
import numpy as np
import json
import itertools


def calc_sim(box1, box2):

    """
    Computes the similarity between two bounding boxes
    :param box1: bounding box 1
    :param box2: bounding box 2
    :return: the sum of the x and y distance of two boxes
    """

    box1_xmin, box1_ymin, box1_xmax, box1_ymax = box1
    box2_xmin, box2_ymin, box2_xmax, box2_ymax = box2
    x_dist = min(abs(box1_xmin - box2_xmin), abs(box1_xmin - box2_xmax), abs(box1_xmax - box2_xmin),
                 abs(box1_xmax - box2_xmax))
    y_dist = min(abs(box1_ymin - box2_ymin), abs(box1_ymin - box2_ymax), abs(box1_ymax - box2_ymin),
                 abs(box1_ymax - box2_ymax))
    dist = x_dist + y_dist
    return dist


def merge_boxes(box_list, masked_image):
    """
    Merges boxes together while there are boxes that are close enough to each
    other to be merged
    :param box_list: List of bounding boxes
    :return: box list with closed bounding boxes merged
    """

    box_list_copy = box_list

    for i,j in list(itertools.combinations(list(range(len(box_list))), 2)):

        box1_xmin, box1_ymin, box1_xmax, box1_ymax = box_list[i]
        box2_xmin, box2_ymin, box2_xmax, box2_ymax = box_list[j]

        # If the two boxes are close together than merge them
        if calc_sim(box_list[i][:4], box_list[j][:4]) < 15:

            new_box = [min(box1_xmin, box2_xmin),
            min(box1_ymin, box2_ymin),
            max(box1_xmax, box2_xmax),
            max(box1_ymax, box2_ymax)]

            print(masked_image)
            b1_area = (box1_xmax-box1_xmin)*(box1_ymax-box1_ymin)
            b2_area = (box2_xmax-box2_xmin)*(box2_ymax-box2_ymin)
            avg_area = (1/2)*(b1_area+b2_area)
            print(calc_sim(box_list[i][:4], box_list[j][:4])/avg_area)

            # Replace the old boxes with the new one
            box_list_copy[i] = new_box
            del box_list_copy[j]

            return True, box_list_copy

    return False, box_list

def convert_masks(image_list):

    # TODO: Merge two boxes if they are close together
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

        # Merge all the boxes fully
        need_to_merge = True
        while need_to_merge:
            need_to_merge, box_list = merge_boxes(box_list, masked_image)

        bboxes.append(box_list)

    return bboxes

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


    """
    Returns the bounding boxes from each of the semantic images of the dataset
    :param datasets: The name of all the datasets in the MaskedImages of
    :return: Dictionary with key = image path and value = bounding boxes
    """

    f = open('image_pairs.json')
    image_pairs = json.load(f)
    masked_images = [x[1] for x in image_pairs]
    matching_images = [os.path.join('UsedImages',x[0]) for x in image_pairs]
    bbox_labels = convert_masks(masked_images)
    save_gt_images(matching_images, bbox_labels)

    with open('image_pairs.json', 'w') as f:
        json.dump(image_pairs, f)



if __name__ == '__main__':


    # Get list of datasets
    datasets = os.listdir(os.path.join(os.curdir, 'MaskedImages'))

    get_bboxes_from_datasets(datasets)