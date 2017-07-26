import tensorflow as tf
import utils
import numpy as np

model_path = 'Data/vgg16.tfmodel'


def get_vgg_graph():
	vgg_file = open(model_path)
	vgg16raw = vgg_file.read()
	vgg_file.close()

	graph_def = tf.GraphDef()
	graph_def.ParseFromString(vgg16raw)

	images = tf.placeholder("float", [None, 224, 224, 3])
	tf.import_graph_def(graph_def, input_map={"images": images})
	graph = tf.get_default_graph()

	return graph, tf.Session(graph=graph), images


def load_image_features(img_path, graph, sess, images):

	image_batch = np.ndarray((1, 224, 224, 3))
	image_batch[0, :, :, :] = utils.load_image_array(img_path)

	feed_dict = {images: image_batch[0:1, :, :, :]}
	fc7_tensor = graph.get_tensor_by_name("import/Relu_1:0")
	fc7_batch = sess.run(fc7_tensor, feed_dict=feed_dict)
	return fc7_batch[0, :]


def main():
	graph, graph_sess, images = get_vgg_graph()
	print load_image_features('Data/train2014/COCO_train2014_000000465294.jpg', graph, graph_sess, images)
	print load_image_features('Data/train2014/COCO_train2014_000000465294.jpg', graph, graph_sess, images)
	print load_image_features('Data/train2014/COCO_train2014_000000465294.jpg', graph, graph_sess, images)

	g1 = tf.Graph()
	with g1.as_default() as g:
		with g.name_scope("g1"):
			matrix1 = tf.constant([[3., 3.]])
			matrix2 = tf.constant([[2.],[2.]])
			product = tf.matmul(matrix1, matrix2)

	with tf.Session(graph=g1) as sess:
		print sess.run(product)

	print load_image_features('Data/train2014/COCO_train2014_000000465294.jpg', graph, graph_sess, images)


if __name__ == '__main__':
	main()
