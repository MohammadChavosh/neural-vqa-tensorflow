from utils import load_image_array, FeatureExtractor
from scipy import misc
from vqa_model import VQAModel
from os.path import join

VALID_ACTIONS = ['End', 'Upper_Up', 'Upper_Down', 'Bottom_Up', 'Bottom_Down', 'Left_Left', 'Left_Right', 'Right_Left', 'Right_Right']


class Environment:
	feature_extractor = FeatureExtractor(join('Data', 'vgg16.tfmodel'))

	def __init__(self, img_path, question, answer):
		self.img_array = load_image_array(img_path, False)
		self.question = question
		self.answer = answer
		img_size = self.img_array.shape
		self.crop_coordinates = [0, 0, img_size[0], img_size[1]]
		self.x_alpha = img_size[0] / 10.0
		self.y_alpha = img_size[1] / 10.0
		self.vqa_model = VQAModel()
		self.img_features = self.get_resized_region_image_features()
		self.latest_loss, self.latest_accuracy, self.state, _ = self.vqa_model.get_result(self.img_features, self.question, self.answer)

		self.TRIGGER_NEGATIVE_REWARD = -3
		self.TRIGGER_POSITIVE_REWARD = 3
		self.MOVE_NEGATIVE_REWARD = -1
		self.MOVE_POSITIVE_REWARD = 1

	def action(self, action_type):
		img_size = self.img_array.shape
		if action_type == 'Upper_Up':
			self.crop_coordinates[0] = max(self.crop_coordinates[0] - self.x_alpha, 0)
		if action_type == 'Upper_Down':
			self.crop_coordinates[0] = min(self.crop_coordinates[0] + self.x_alpha, img_size[0])
		if action_type == 'Bottom_Up':
			self.crop_coordinates[2] = max(self.crop_coordinates[2] - self.x_alpha, 0)
		if action_type == 'Bottom_Down':
			self.crop_coordinates[2] = min(self.crop_coordinates[2] + self.x_alpha, img_size[0])
		if action_type == 'Left_Left':
			self.crop_coordinates[1] = max(self.crop_coordinates[1] - self.y_alpha, 0)
		if action_type == 'Left_Right':
			self.crop_coordinates[1] = min(self.crop_coordinates[1] + self.y_alpha, img_size[1])
		if action_type == 'Right_Left':
			self.crop_coordinates[3] = max(self.crop_coordinates[3] - self.y_alpha, 0)
		if action_type == 'Right_Right':
			self.crop_coordinates[3] = min(self.crop_coordinates[3] + self.y_alpha, img_size[1])
		self.img_features = self.get_resized_region_image_features()
		if action_type == 'End':
			self.latest_loss, self.latest_accuracy, self.state, _ = self.vqa_model.get_result(self.img_features, self.question, self.answer)
			if self.latest_accuracy < 0.1:
				return self.TRIGGER_NEGATIVE_REWARD, True
			if self.latest_accuracy > 0.9:
				return self.TRIGGER_POSITIVE_REWARD, True
		else:
			loss, self.latest_accuracy, self.state, _ = self.vqa_model.get_result(self.img_features, self.question, self.answer)
			if self.latest_loss > loss:
				self.latest_loss = loss
				return self.MOVE_POSITIVE_REWARD, False
			else:
				self.latest_loss = loss
				return self.MOVE_NEGATIVE_REWARD, False

	def get_resized_region_image_features(self):
		rounded_coordinates = map(lambda x:int(round(x)), self.crop_coordinates)
		img = self.img_array[rounded_coordinates[0]:rounded_coordinates[2], rounded_coordinates[1]:rounded_coordinates[3], :]
		img = misc.imresize(img, (224, 224))
		img = (img / 255.0).astype('float32')
		return Environment.feature_extractor.extract_fc7_features(img)

	def valid_actions(self):
		img_size = self.img_array.shape
		result = ['End']
		if self.crop_coordinates[0] - self.x_alpha >= 0:
			result.append('Upper_Up')
		if (self.crop_coordinates[0] + self.x_alpha < self.crop_coordinates[2]) and \
				(self.crop_coordinates[0] + self.x_alpha <= img_size[0]):
			result.append('Upper_Down')

		if (self.crop_coordinates[2] - self.x_alpha >= 0) and \
				(self.crop_coordinates[2] - self.x_alpha > self.crop_coordinates[0]):
			result.append('Bottom_Up')
		if self.crop_coordinates[2] + self.x_alpha <= img_size[0]:
			result.append('Bottom_Down')

		if self.crop_coordinates[1] - self.y_alpha >= 0:
			result.append('Left_Left')
		if (self.crop_coordinates[1] + self.y_alpha < self.crop_coordinates[3]) and \
				(self.crop_coordinates[1] + self.y_alpha <= img_size[1]):
			result.append('Left_Right')

		if self.crop_coordinates[3] - self.y_alpha >= 0 and \
				(self.crop_coordinates[3] - self.y_alpha > self.crop_coordinates[1]):
			result.append('Right_Left')
		if self.crop_coordinates[3] + self.y_alpha <= img_size[1]:
			result.append('Right_Right')

		return result

	def reset(self):
		img_size = self.img_array.shape
		self.crop_coordinates = [0, 0, img_size[0], img_size[1]]
		self.img_features = self.get_resized_region_image_features()
		self.latest_loss, self.latest_accuracy, self.state, _ = self.vqa_model.get_result(self.img_features, self.question, self.answer)
