import math


def get_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def expand_box(box, scale=1.25):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    expand_width = (scale - 1) * width / 2
    expand_height = (scale - 1) * height / 2

    return [
        x1 - expand_width,
        y1 - expand_height,
        x2 + expand_width,
        y2 + expand_height
    ]


def is_inside(person_box, object_box, scale=1.25):
    """
    Checks whether the center of a PPE object lies inside a person's expanded box.
    Kept for compatibility with your existing detect_image.py.
    """
    expanded_person_box = expand_box(person_box, scale)

    px1, py1, px2, py2 = expanded_person_box
    center_x, center_y = get_center(object_box)

    return px1 <= center_x <= px2 and py1 <= center_y <= py2


def calculate_iou(box1, box2):
    """
    Calculates IoU between two boxes.
    IoU = overlap area / total combined area
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection_area = intersection_width * intersection_height

    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0

    return intersection_area / union_area


def distance_between_boxes(box1, box2):
    """
    Calculates distance between center points of two boxes.
    """

    x1, y1 = get_center(box1)
    x2, y2 = get_center(box2)

    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def assign_to_closest_person(persons, object_box):
    """
    Assigns one PPE object to only one closest person.
    This prevents the same helmet/vest/glove being counted for multiple workers.
    """

    closest_person_index = None
    minimum_distance = float("inf")

    for index, person in enumerate(persons):
        person_box = person["box"]

        # First check if PPE object is inside expanded person box
        if is_inside(person_box, object_box, scale=1.35):
            distance = distance_between_boxes(person_box, object_box)

            if distance < minimum_distance:
                minimum_distance = distance
                closest_person_index = index

    return closest_person_index


def assign_ppe_to_workers(persons, ppe_objects):
    """
    Creates worker-wise PPE assignment.

    Output example:
    {
        0: {
            "helmet": True,
            "vest": True,
            "gloves": False,
            "no_gloves": True
        }
    }
    """

    worker_ppe = {}

    for index in range(len(persons)):
        worker_ppe[index] = {
            "helmet": False,
            "vest": False,
            "gloves": False,
            "goggles": False,
            "boots": False,
            "no_helmet": False,
            "no_gloves": False,
            "no_goggle": False
        }

    for obj in ppe_objects:
        class_name = obj["class_name"].lower()
        object_box = obj["box"]

        matched_worker_index = assign_to_closest_person(persons, object_box)

        if matched_worker_index is not None:
            if class_name in worker_ppe[matched_worker_index]:
                worker_ppe[matched_worker_index][class_name] = True

    return worker_ppe