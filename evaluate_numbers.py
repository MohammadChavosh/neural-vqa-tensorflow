import tensorflow as tf
import vis_lstm_model
import data_loader
import argparse
import numpy as np
import json
from train_numbers import init_weight, init_bias


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
	parser.add_argument('--model_path', type=str, default = 'Data/Models/model21.ckpt',
                       help='Model Path')
	parser.add_argument('--version', type=int, default=2,
                       help='VQA data version')

	args = parser.parse_args()
	print "Reading QA DATA"
	# qa_data = data_loader.load_questions_answers(args)
	qa_data = data_loader.load_questions_answers(args.version, args.data_dir)

	for _type in ['training', 'validation']:
		new_qa = []
		for q in qa_data[_type]:
			if q['answer_type'] == 'number':
				new_qa.append(q)
		qa_data[_type] = new_qa

	print "Reading fc7 features"
	fc7_features, image_id_list = data_loader.load_fc7_features(args.data_dir, 'val')
	print "FC7 features", fc7_features.shape
	print "image_id_list", image_id_list.shape

	image_id_map = {}
	for i in xrange(len(image_id_list)):
		image_id_map[ image_id_list[i] ] = i

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
	input_tensors, lstm_answer = model.build_numbers_generator()

	ans_number_W = init_weight(model_options['rnn_size'], ans_size, name='ans_number_W')
	ans_number_b = init_bias(ans_size, name='ans_number_b')
	number_logits = tf.matmul(lstm_answer, ans_number_W) + ans_number_b

	ans_probability = tf.nn.sigmoid(number_logits, name='number_answer_probab')
	answer_probability = ans_probability + tf.expand_dims(tf.maximum(0.6001 - tf.reduce_max(ans_probability, axis=1), 0), 1)
	tmp_indices = tf.where(tf.equal(tf.less(0.6, answer_probability), True))
	number_prediction = tf.segment_max(tmp_indices[:, 1], tmp_indices[:, 0])

	sess = tf.InteractiveSession()
	saver = tf.train.Saver()

	avg_accuracy = 0.0
	total = 0
	saver.restore(sess, args.model_path)
	
	batch_no = 0
	result = []
	while (batch_no*args.batch_size) < len(qa_data['validation']):
		sentence, answer, fc7, question_ids = get_batch(batch_no, args.batch_size, 
			fc7_features, image_id_map, qa_data, 'val')
		
		pred = sess.run(number_prediction, feed_dict={
            input_tensors['fc7']:fc7,
            input_tensors['sentence']:sentence,
        })
		
		batch_no += 1
		cnt = 0
		if args.debug:
			for idx, p in enumerate(pred):
				# print ans_map[p], ans_map[ np.argmax(answer[idx])]
				result.append({'answer': str(p), 'question_id': question_ids[cnt]})
				cnt += 1

		correct_ans = np.shape(answer)[1] - np.argmax(answer[:, ::-1], axis=1) - 1
		correct_predictions = np.equal(correct_ans, pred)
		correct_predictions = correct_predictions.astype('float32')
		accuracy = correct_predictions.mean()
		print "Acc", accuracy
		avg_accuracy += accuracy
		total += 1

	print "Acc", avg_accuracy/total
	my_list = list(result)
	json.dump(my_list,open('result.json','w'))


def get_batch(batch_no, batch_size, fc7_features, image_id_map, qa_data, split):
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

	si = (batch_no * batch_size)%len(qa)
	ei = min(len(qa), si + batch_size)
	n = ei - si
	sentence = np.ndarray( (n, qa_data['max_question_length']), dtype = 'int32')
	answer = np.zeros( (n, len(qa_data['answer_vocab'])))
	fc7 = np.ndarray( (n,4096) )
	question_ids = []

	count = 0

	for i in range(si, ei):
		question_ids.append(qa[i]['question_id'])
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
	
	return sentence, answer, fc7, question_ids

if __name__ == '__main__':
	main()
