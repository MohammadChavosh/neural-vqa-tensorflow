import json

print('loading json files...')

with open('result1.json') as data_file:
	incomplete_data = json.load(data_file)

with open('result2.json') as data_file:
	complete_data = json.load(data_file)

tmp_answers = dict()
for i in xrange(len(incomplete_data)):
	tmp_answers[int(incomplete_data[i]['question_id'])] = incomplete_data[i]['answer']

data = []
for i in xrange(len(complete_data)):
	tmp_dict = dict()
	tmp_dict['question_id'] = int(complete_data[i]['question_id'])
	if complete_data[i]['question_id'] in tmp_answers:
		tmp_dict['answer'] = tmp_answers[tmp_dict['question_id']]
	else:
		tmp_dict['answer'] = 'FOR SURE THIS ANSWER IS NOT CORRECT, DO YOU AGREE TOO?'
	data.append(tmp_dict)

dd = json.dump(data, open('result.json', 'w'))
