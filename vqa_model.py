import tensorflow as tf
import vis_lstm_model
import data_loader
import numpy as np
import re


class VQAModel:
	data_dir = 'Data'
	version = 1
	vocab_data = data_loader.get_question_answer_vocab(version, data_dir)

	model_options = {
		'num_lstm_layers': 2,
		'rnn_size': 512,
		'embedding_size': 512,
		'word_emb_dropout': 0.5,
		'image_dropout': 0.5,
		'fc7_feature_length': 4096,
		'lstm_steps': vocab_data['max_question_length'] + 1,
		'q_vocab_size': len(vocab_data['question_vocab']),
		'ans_vocab_size': len(vocab_data['answer_vocab'])
	}

	graph = tf.Graph()
	with graph.as_default():
		model = vis_lstm_model.Vis_lstm_model(model_options)
		input_tensors, loss, accuracy, lstm_answer, predictions = model.build_for_rl()
	model_path = 'Data/Models/model199.ckpt'
	sess = tf.Session(graph=graph)
	with sess.as_default():
		with graph.as_default():
			saver = tf.train.Saver()
			saver.restore(sess, model_path)

	def __init__(self):
		pass

	@staticmethod
	def get_result(fc7_features, question, answer):
		word_regex = re.compile(r'\w+')
		question_ids = np.zeros((1, VQAModel.vocab_data['max_question_length']), dtype='int32')
		question_words = re.findall(word_regex, question)
		base = VQAModel.vocab_data['max_question_length'] - len(question_words)
		question_vocab = VQAModel.vocab_data['question_vocab']
		for i in range(0, len(question_words)):
			if question_words[i] in question_vocab:
				question_ids[0][base + i] = question_vocab[question_words[i]]
			else:
				question_ids[0][base + i] = question_vocab['UNK']

		answer_vocab = VQAModel.vocab_data['answer_vocab']
		answer_id = np.zeros((1, len(answer_vocab)))
		answer_id[0, answer_vocab[answer]] = 1.0

		loss, accuracy, lstm_answer, predictions = VQAModel.sess.run([VQAModel.loss, VQAModel.accuracy, VQAModel.lstm_answer, VQAModel.predictions], feed_dict={
			VQAModel.input_tensors['fc7']: fc7_features,
			VQAModel.input_tensors['sentence']: question_ids,
			VQAModel.input_tensors['answer']: answer_id
		})
		return loss, accuracy, lstm_answer, predictions