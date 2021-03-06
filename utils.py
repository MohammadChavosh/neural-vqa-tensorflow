import numpy as np
from scipy import misc
import tensorflow as tf


# crop_coordinates is in type [x1, y1, x2, y2]
def load_image_array(image_file, resize=True):
	img = misc.imread(image_file)
	# GRAYSCALE
	if len(img.shape) == 2:
		img_new = np.ndarray((img.shape[0], img.shape[1], 3), dtype='float32')
		img_new[:, :, 0] = img
		img_new[:, :, 1] = img
		img_new[:, :, 2] = img
		img = img_new
	if resize:
		img_resized = misc.imresize(img, (224, 224))
		return (img_resized / 255.0).astype('float32')
	return img


class FeatureExtractor:

	def __init__(self, model_path):
		vgg_file = open(model_path)
		vgg16raw = vgg_file.read()
		vgg_file.close()

		graph_def = tf.GraphDef()
		graph_def.ParseFromString(vgg16raw)

		self.images = tf.placeholder("float", [None, 224, 224, 3])
		tf.import_graph_def(graph_def, input_map={"images": self.images})
		self.graph = tf.get_default_graph()
		self.sess = tf.Session(graph=self.graph)

	def extract_fc7_features(self, img_array):

		image_batch = np.ndarray((1, 224, 224, 3))
		image_batch[0, :, :, :] = img_array

		feed_dict = {self.images: image_batch[0:1, :, :, :]}
		fc7_tensor = self.graph.get_tensor_by_name("import/Relu_1:0")
		fc7_batch = self.sess.run(fc7_tensor, feed_dict=feed_dict)
		return fc7_batch[0:1, :]
