from env import Environment
import random


def main():
	environment = Environment()
	print environment.latest_loss, environment.latest_accuracy
	print environment.crop_coordinates
	print '--------------------'
	for _ in range(10):
		valid_actions = environment.valid_actions()
		print 'valid_actions: ', valid_actions
		action = random.sample(valid_actions, 1)[0]
		print 'action: ', action
		print 'reward, done: ', environment.action(action)
		print 'coordinates: ', environment.crop_coordinates
		print 'latest_loss: ', environment.latest_loss
		print 'latest_accuracy: ', environment.latest_accuracy
		print '----------------------'
		with open("test.txt", "a") as f:
				f.write('salam\n')


if __name__ == '__main__':
	main()
