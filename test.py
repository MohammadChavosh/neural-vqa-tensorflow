import tensorflow as tf
from os.path import join
import data_loader
import utils
import argparse
import numpy as np
import h5py
import time

model_path = 'Data/vgg16.tfmodel'


def load_image_features(img_path):
	vgg_file = open(model_path)
	vgg16raw = vgg_file.read()
	vgg_file.close()

	graph_def = tf.GraphDef()
	graph_def.ParseFromString(vgg16raw)

	images = tf.placeholder("float", [None, 224, 224, 3])
	tf.import_graph_def(graph_def, input_map={"images": images})

	graph = tf.get_default_graph()

	for opn in graph.get_operations():
		print "Name", opn.name, opn.values()

	sess = tf.Session()

	image_batch = np.ndarray((1, 224, 224, 3))
	image_batch[0, :, :, :] = utils.load_image_array(img_path)

	feed_dict = {images: image_batch[0, :, :, :]}
	fc7_tensor = graph.get_tensor_by_name("import/Relu_1:0")
	fc7_batch = sess.run(fc7_tensor, feed_dict=feed_dict)
	return fc7_batch[0, :]


def main():
	print load_image_features('Data/train2014/COCO_train2014_000000465294')


if __name__ == '__main__':
	main()
