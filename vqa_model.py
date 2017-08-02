import tensorflow as tf
import vis_lstm_model
import data_loader
import numpy as np
import re


class VQAModel:
	def __init__(self):
		data_dir = 'Data'
		version = 1
		self.vocab_data = data_loader.get_question_answer_vocab(version, data_dir)
		model_options = {
			'num_lstm_layers': 2,
			'rnn_size': 512,
			'embedding_size': 512,
			'word_emb_dropout': 0.5,
			'image_dropout': 0.5,
			'fc7_feature_length': 4096,
			'lstm_steps': self.vocab_data['max_question_length'] + 1,
			'q_vocab_size': len(self.vocab_data['question_vocab']),
			'ans_vocab_size': len(self.vocab_data['answer_vocab'])
		}

		self.ans_map = {self.vocab_data['answer_vocab'][ans]: ans for ans in self.vocab_data['answer_vocab']}
		model = vis_lstm_model.Vis_lstm_model(model_options)
		self.input_tensors, self.loss, self.accuracy, self.predictions = model.build_model()

		model_path = 'Data/Models/model199.ckpt'
		self.sess = tf.InteractiveSession()
		saver = tf.train.Saver()
		saver.restore(self.sess, model_path)

	def get_result(self, fc7_features, question, answer):
		word_regex = re.compile(r'\w+')
		question_ids = np.zeros((1, self.vocab_data['max_question_length']), dtype='int32')
		question_words = re.findall(word_regex, question)
		base = self.vocab_data['max_question_length'] - len(question_words)
		question_vocab = self.vocab_data['question_vocab']
		for i in range(0, len(question_words)):
			if question_words[i] in question_vocab:
				question_ids[0][base + i] = question_vocab[question_words[i]]
			else:
				question_ids[0][base + i] = question_vocab['UNK']

		answer_vocab = self.vocab_data['answer_vocab']
		answer_id = np.zeros((1, len(answer_vocab)))
		answer_id[0, answer_vocab[answer]] = 1.0

		loss, accuracy, predictions = self.sess.run([self.loss, self.accuracy, self.predictions], feed_dict={
			self.input_tensors['fc7']: fc7_features,
			self.input_tensors['sentence']: question_ids,
			self.input_tensors['answer']: answer_id
		})
		return loss, accuracy, predictions
