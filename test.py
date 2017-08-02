from env import Environment
import random


def main():
	environment = Environment('Data/train2014/COCO_train2014_000000465294.jpg', 'What is the name of fruit in the picture?', 'banana')
	print environment.latest_loss
	print environment.crop_coordinates
	print '--------------------'
	for _ in range(10):
		valid_actions = environment.valid_actions()
		print 'valid_actions: ', valid_actions
		action = random.sample(valid_actions, 1)[0]
		print 'action: ', action
		print 'reward: ', environment.action(action)
		print 'coordinates: ', environment.crop_coordinates
		print 'latest_loss: ', environment.latest_loss
		print 'latest_accuracy: ', environment.latest_accuracy
		print '----------------------'


if __name__ == '__main__':
	main()
