import tensorflow as tf
import vis_lstm_model
import data_loader
import argparse
import numpy as np
import math


def init_weight(dim_in, dim_out, name=None, stddev=1.0):
	return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)


def init_bias(dim_out, name=None):
	return tf.Variable(tf.zeros([dim_out]), name=name)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--num_lstm_layers', type=int, default=2,
                       help='num_lstm_layers')
	parser.add_argument('--fc7_feature_length', type=int, default=4096,
                       help='fc7_feature_length')
	parser.add_argument('--rnn_size', type=int, default=512,
                       help='rnn_size')
	parser.add_argument('--embedding_size', type=int, default=512,
                       help='embedding_size'),
	parser.add_argument('--word_emb_dropout', type=float, default=0.5,
                       help='word_emb_dropout')
	parser.add_argument('--image_dropout', type=float, default=0.5,
                       help='image_dropout')
	parser.add_argument('--data_dir', type=str, default='Data',
                       help='Data directory')
	parser.add_argument('--batch_size', type=int, default=200,
                       help='Batch Size')
	parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Batch Size')
	parser.add_argument('--epochs', type=int, default=200,
                       help='Expochs')
	parser.add_argument('--debug', type=bool, default=False,
                       help='Debug')
	parser.add_argument('--resume_model', type=str, default=None,
                       help='Trained Model Path')
	parser.add_argument('--version', type=int, default=2,
                       help='VQA data version')

	args = parser.parse_args()
	print "Reading QA DATA"
	qa_data = data_loader.load_questions_answers(args.version, args.data_dir, True)

	for _type in ['training', 'validation']:
		new_qa = []
		for q in qa_data[_type]:
			if q['answer_type'] == 'number':
				new_qa.append(q)
		qa_data[_type] = new_qa
	
	print "Reading fc7 features"
	fc7_features, image_id_list = data_loader.load_fc7_features(args.data_dir, 'train')
	print "FC7 features", fc7_features.shape
	print "image_id_list", image_id_list.shape

	image_id_map = {}
	for i in xrange(len(image_id_list)):
		image_id_map[ image_id_list[i] ] = i

	ans_map = {qa_data['answer_vocab'][ans]: ans for ans in qa_data['answer_vocab']}

	model_options = {
		'num_lstm_layers' : args.num_lstm_layers,
		'rnn_size' : args.rnn_size,
		'embedding_size' : args.embedding_size,
		'word_emb_dropout' : args.word_emb_dropout,
		'image_dropout' : args.image_dropout,
		'fc7_feature_length' : args.fc7_feature_length,
		'lstm_steps' : qa_data['max_question_length'] + 1,
		'q_vocab_size' : len(qa_data['question_vocab']),
		'ans_vocab_size' : len(qa_data['answer_vocab'])
	}

	ans_size = 22

	model = vis_lstm_model.Vis_lstm_model(model_options)
	input_tensors, lstm_answer = model.build_numbers_model(ans_size)
	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

	saver = tf.train.Saver()
	if args.resume_model:
		saver.restore(sess, args.resume_model)

	ans_number_W = init_weight(model_options['rnn_size'], ans_size, name = 'ans_number_W')
	ans_number_b = init_bias(ans_size, name='ans_number_b')
	number_logits = tf.matmul(lstm_answer, ans_number_W) + ans_number_b

	number_ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=input_tensors['answer'], logits=number_logits, name='number_ce')
	number_loss = tf.reduce_sum(number_ce, name='number_loss')

	ans_probability = tf.nn.sigmoid(number_logits, name='number_answer_probab')
	answer_probability = ans_probability + tf.expand_dims(tf.maximum(0.6001 - tf.reduce_max(ans_probability, axis=1), 0), 1)
	tmp_indices = tf.where(tf.equal(tf.less(0.6, answer_probability), True))
	number_prediction = tf.segment_max(tmp_indices[:, 1], tmp_indices[:, 0])

	ans_tmp_indices = tf.where(tf.equal(input_tensors['answer'], 1.0))
	correct_ans = tf.segment_max(ans_tmp_indices[:, 1], ans_tmp_indices[:, 0])
	correct_predictions = tf.equal(correct_ans, number_prediction)
	number_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

	optimizer = tf.train.MomentumOptimizer(args.learning_rate, 0.95)
	var_list = [ans_number_W, ans_number_b]
	train_op = optimizer.minimize(number_loss, var_list=var_list)
	reset_opt_op = tf.variables_initializer([optimizer.get_slot(var, name) for name in optimizer.get_slot_names() for var in var_list])
	sess.run(reset_opt_op)

	init_new_vars_op = tf.initialize_variables([ans_number_W, ans_number_b])
	sess.run(init_new_vars_op)
	saver = tf.train.Saver()
	for i in xrange(args.epochs):
		batch_no = 0

		while (batch_no*args.batch_size) < len(qa_data['training']):
			sentence, answer, fc7 = get_training_batch(batch_no, args.batch_size, fc7_features, image_id_map, qa_data, 'train', ans_size)
			_, loss_value, accuracy, pred = sess.run([train_op, number_loss, number_accuracy, number_prediction],
				feed_dict={
					input_tensors['fc7']:fc7,
					input_tensors['sentence']:sentence,
					input_tensors['answer']:answer
				}
			)
			batch_no += 1
			if args.debug:
				for idx, p in enumerate(pred):
					print ans_map[p], ans_map[ np.argmax(answer[idx])]

				print "Loss", loss_value, batch_no, i
				print "Accuracy", accuracy
				print "---------------"
			else:
				print "Loss", loss_value, batch_no, i
				print "Training Accuracy", accuracy
			
		save_path = saver.save(sess, "Data/NumberModels/model{}.ckpt".format(i))
		

def get_training_batch(batch_no, batch_size, fc7_features, image_id_map, qa_data, split, ans_size):
	qa = None
	if split == 'train':
		qa = qa_data['training']
	else:
		qa = qa_data['validation']

	manualMap = {'none': '0',
	             'zero': '0',
	             'one': '1',
	             'two': '2',
	             'three': '3',
	             'four': '4',
	             'five': '5',
	             'six': '6',
	             'seven': '7',
	             'eight': '8',
	             'nine': '9',
	             'ten': '10'
	             }

	si = (batch_no * batch_size) % len(qa)
	ei = min(len(qa), si + batch_size)
	n = ei - si
	sentence = np.ndarray((n, qa_data['max_question_length']), dtype='int32')
	answer = np.zeros((n, ans_size))
	fc7 = np.ndarray((n, 4096))
	count = 0
	for i in range(si, ei):
		sentence[count, :] = qa[i]['question'][:]
		if qa[i]['ans_str'] in manualMap:
			qa[i]['ans_str'] = manualMap[qa[i]['ans_str']]
		ans_digit = 0
		if qa[i]['ans_str'].isdigit():
			ans_digit = int(qa[i]['ans_str'])
			if ans_digit > 20:
				ans_digit = 21
		for j in range(ans_digit + 1):
			answer[count, j] = 1.0
		fc7_index = image_id_map[qa[i]['image_id']]
		fc7[count, :] = fc7_features[fc7_index][:]
		count += 1
	
	return sentence, answer, fc7

if __name__ == '__main__':
	main()
