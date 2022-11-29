import tensorflow as tf
import numpy as np
import os
import cv2
from omegaconf import DictConfig



class DataGenerator(tf.keras.utils.Sequence):
    """
        generate data for model by reading images and their bounding box file.
        taken from # https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

        Args:
            img_folder (str): path to images folder
            labels_folder (str): path to labels folder, which contain one text file of bounding boxes for each image
            batch_size (int):
            height (int): image height
            width (int): image width
            max_boxes (int): max boxes that model can predict
            anchors (numpy.array[int, 2]): normalized anchors of shape (int,2) ,
            The first and second columns of the numpy arrays respectively contain the anchors width and height.
            anchors_mask (numpy.array): mask against anchors
            first_stage (int): stride of first/lowest stage
            shuffle (boolean): whether shuffle data after each epoch or not
        """

    def __init__(self, img_folder, labels_folder, batch_size, height, width, max_boxes, anchors, anchors_mask,
                 first_stage, shuffle=True):
        'Initialization'
        self.img_folder = img_folder
        self.labels_folder = labels_folder
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.max_boxes = max_boxes
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.first_stage = first_stage
        self.shuffle = shuffle

        self.img_paths = os.listdir(self.img_folder)  # has only images name not full path
        self.labels_paths = os.listdir(self.labels_folder)
        self.img_paths.sort()
        self.labels_paths.sort()

        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.img_paths) / self.batch_size))

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.img_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        batch_images = []
        batch_labels_32, batch_labels_16, batch_labels_8 = [], [], []

        # Generate data
        for i, indx in enumerate(indexes):
            # Read an image from folder and resize
            temp_img_path = os.path.join(self.img_folder, self.img_paths[indx])
            image = cv2.imread(temp_img_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = cv2.resize(image, (W, H))
            image = image / 255.0  # normalize image
            image = image.astype(np.float32)

            # read labels file
            labels = self.read_bb_file(indx)

            labels = tf.convert_to_tensor(labels, tf.float32)  # convert to tenor
            labels = tf.expand_dims(labels, axis=0)  # add batch dimension

            # convert list of bb into yolo ground truth
            labels = self.transform_targets(labels, self.anchors, self.anchors_mask, self.width)

            # save one data item into list
            batch_images.append(image)
            batch_labels_32.append(labels[0])  # at stride 32
            batch_labels_16.append(labels[1])  # at stride 16
            batch_labels_8.append(labels[2])  # at stride 8

            # resize #boxes against max_boxes
            # labels = self.add_dummy_labels(labels)

        # convert list into a batch
        return np.array(batch_images), [np.concatenate(batch_labels_32, axis=0),
                                        np.concatenate(batch_labels_16, axis=0),
                                        np.concatenate(batch_labels_8, axis=0)]

    def read_bb_file(self, indx):
        """
             Read text file of bonding boxes line by line
            read bb+label from txt file as a list, list element has sub list with five elements

            Args:
                indx (int): index of file to read

                Returns:
                    labels (list): list of bounding boxes in image
        """
        labels = []
        # xmin, ymin, xmax, ymax, class
        temp_labels_path = os.path.join(self.labels_folder, self.mask_paths[indx])
        with open(temp_labels_path) as file_in:
            for line in file_in:  # read file line by line
                bb = line.strip().split(",")  # split
                # bb = list(np.float_(bb))  # convert list to float
                bb = list(np.array(bb).astype(np.float32))  # convert list to float32
                labels.append(bb)
        return labels

    def add_dummy_labels(self, labels_list):
        """
        all images ground truth boxes should be of equal numbers, so if ground truth boxes are not
        equal to max_boxes then add or remove boxes based on size
        Args:
            labels_list (list(list())): list of bounding boxes

        Returns:
            labels_list  (list(list())): label list of fixed size
        """

        # if it has more than max_boxes then remove extra last boxes
        if len(labels_list) > self.max_boxes:
            return labels_list[:self.max_boxes]
        # in case of less #boxes than append dummy data
        elif len(labels_list) < self.max_boxes:
            labels_list = labels_list + [[0, 0, 0, 0, 0]] * (self.max_boxes - len(labels_list))
            return labels_list
        # if equal do nothing
        else:
            return labels_list

    def transform_targets(self, y_train, anchors, anchor_masks, size):
        """
            assign bounding boxes into their respective grid cells.
            first find anchor index who has greatest iou with box and store its index after box
            then call transform_targets_for_output to actually assign  bounding boxes into their respective grid cell

            Args:
                y_train ((N, boxes, (x1, y1, x2, y2, class, best_anchor))): bounding boxes from txt file
                anchors (numpy.array[int, 2]): normalized anchors of shape (int,2) ,
                The first and second columns of the numpy arrays respectively contain the anchors width and height.
                anchor_masks (numpy.array): mask against anchors
                size (int): size/width of image

            Returns:
                tuple: of size 3, 0: (1, 13, 13, 3, 6), 1: (1, 26, 26, 3, 6), 2: (1, 52, 52, 3, 6)
                against stride 32,16,8
            """

        """ 
        consider input shapes are
        y_train: TensorShape([1, 8, 5]) ==> 1, #boxes, 5(xmin, ymin, xmax, ymax, class)
        anchors: (9, 2)
        anchor_masks: (3, 3)
        size: 416 image size
        now shapes are defined based on this input
        """
        y_outs = []
        grid_size = size // self.first_stage  # for lowest stride

        # calculate anchor index for true boxes
        anchors = tf.cast(anchors, tf.float32)
        anchor_area = anchors[..., 0] * anchors[..., 1]  # 9
        box_wh = y_train[..., 2:4] - y_train[..., 0:2]  # N,8,2
        box_wh = tf.tile(tf.expand_dims(box_wh, -2), (1, 1, tf.shape(anchors)[0], 1))  # N,8,9,2
        box_area = box_wh[..., 0] * box_wh[..., 1]  # 1,8,9
        intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(box_wh[..., 1],
                                                                                anchors[..., 1])  # N,8,9
        iou = intersection / (box_area + anchor_area - intersection)  # N,8,9
        anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)  # N,8
        anchor_idx = tf.expand_dims(anchor_idx, axis=-1)  # N,8,1

        y_train = tf.concat([y_train, anchor_idx], axis=-1)  # N,8,6

        for anchor_idxs in anchor_masks:
            # assign best anchor to grid
            temp = self.transform_targets_for_output(y_train, grid_size, anchor_idxs)
            y_outs.append(temp)
            grid_size *= 2

        return tuple(y_outs)

    @tf.function
    def transform_targets_for_output(self, y_true, grid_size, anchor_idxs):
        """
            Based on anchor size assign it to grid size.
            Args:
                y_true (N, boxes, (x1, y1, x2, y2, class, best_anchor)):
                grid_size (int): eiter image_size/32,16,8
                anchor_idxs (list): mask against anchors

                Returns:
                    yolo_ground_truth: (N, grid, grid, anchors, [x1, y1, x2, y2, obj, class])
        """
        N = tf.shape(y_true)[0]

        # y_true_out: (N, grid, grid, anchors, [x1, y1, x2, y2, obj, class])
        y_true_out = tf.zeros(
            (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

        anchor_idxs = tf.cast(anchor_idxs, tf.int32)

        indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
        updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
        idx = 0
        for i in tf.range(N):  # batch loop
            for j in tf.range(tf.shape(y_true)[1]):  # #bounding box loop
                if tf.equal(y_true[i][j][2], 0):  # if BB width is zero then skip
                    continue

                # bool array if anchor id lies in given index array
                anchor_eq = tf.equal(anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

                if tf.reduce_any(anchor_eq):  # if any one contain
                    anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)  # getting location of anchor 0,1 or 2

                    box = y_true[i][j][0:4]
                    box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2  # getting center
                    grid_xy = tf.cast(box_xy // (1 / grid_size), tf.int32)  # normalizing center with grid

                    # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                    indexes = indexes.write(
                        idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])  # grid[y][x][anchor]
                    updates = updates.write(
                        idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])  # (tx, ty, bw, bh, obj, class)
                    idx += 1

        # update grid with generated values
        return tf.tensor_scatter_nd_update(
            y_true_out, indexes.stack(), updates.stack())
