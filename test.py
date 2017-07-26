import tensorflow as tf
from extract_fc7 import FeatureExtractor


def main():
	feature_extractor = FeatureExtractor()
	print feature_extractor.extract_image_features('Data/train2014/COCO_train2014_000000465294.jpg')
	print feature_extractor.extract_image_features('Data/train2014/COCO_train2014_000000465285.jpg')
	print feature_extractor.extract_image_features('Data/train2014/COCO_train2014_000000465269.jpg')

	g1 = tf.Graph()
	with g1.as_default() as g:
		with g.name_scope("g1"):
			matrix1 = tf.constant([[3., 3.]])
			matrix2 = tf.constant([[2.],[2.]])
			product = tf.matmul(matrix1, matrix2)

	with tf.Session(graph=g1) as sess:
		print sess.run(product)

	print feature_extractor.extract_image_features('Data/train2014/COCO_train2014_000000465266.jpg')


if __name__ == '__main__':
	main()
