import tensorflow as tf
import vis_lstm_model
import data_loader
import argparse
import numpy as np
from os.path import join
from utils import FeatureExtractor, load_image_array
import re


class VQAModel:
	def __init__(self, model_path):
		data_dir = 'Data'
		vocab_data = data_loader.get_question_answer_vocab(data_dir)
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
		question_vocab = vocab_data['question_vocab']
		word_regex = re.compile(r'\w+')
		question_ids = np.zeros((1, vocab_data['max_question_length']), dtype='int32')
		question_words = re.findall(word_regex, args.question)
		base = vocab_data['max_question_length'] - len(question_words)
		for i in range(0, len(question_words)):
			if question_words[i] in question_vocab:
				question_ids[0][base + i] = question_vocab[ question_words[i] ]
			else:
				question_ids[0][base + i] = question_vocab['UNK']


	ans_map = { vocab_data['answer_vocab'][ans] : ans for ans in vocab_data['answer_vocab']}
	model = vis_lstm_model.Vis_lstm_model(model_options)
	input_tensors, t_prediction, t_ans_probab = model.build_generator()
	sess = tf.InteractiveSession()
	saver = tf.train.Saver()
	saver.restore(sess, args.model_path)
	
	pred, answer_probab = sess.run([t_prediction, t_ans_probab], feed_dict={
        input_tensors['fc7']:fc7_features,
        input_tensors['sentence']:question_ids,
    })

	
	print "Ans:", ans_map[pred[0]]
	answer_probab_tuples = [(-answer_probab[0][idx], idx) for idx in range(len(answer_probab[0]))]
	answer_probab_tuples.sort()
	print "Top Answers"
	for i in range(5):
		print ans_map[ answer_probab_tuples[i][1] ]

if __name__ == '__main__':
	main()