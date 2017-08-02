from utils import load_image_array, FeatureExtractor
from scipy import misc
from vqa_model import VQAModel
from os.path import join


class Environment:

	def __init__(self, img_path, question, answer):
		self.img_array = load_image_array(img_path, False)
		self.question = question
		self.answer = answer
		img_size = self.img_array.shape
		self.crop_coordinates = [0, 0, img_size[0], img_size[1]]
		self.x_alpha = img_size[0] / 10.0
		self.y_alpha = img_size[1] / 10.0
		self.vqa_model = VQAModel()
		self.feature_extractor = FeatureExtractor(join('Data', 'vgg16.tfmodel'))
		img_features = self.get_resized_region_image_features()
		self.latest_loss, _, _ = self.vqa_model.get_result(img_features, self.question, self.answer)

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
		img_features = self.get_resized_region_image_features()
		if action_type == 'End':
			_, accuracy, _ = self.vqa_model.get_result(img_features, self.question, self.answer)
			if accuracy < 0.1:
				return self.TRIGGER_NEGATIVE_REWARD
			if accuracy > 0.9:
				return self.TRIGGER_POSITIVE_REWARD
		else:
			loss, _, _ = self.vqa_model.get_result(img_features, self.question, self.answer)
			if self.latest_loss > loss:
				self.latest_loss = loss
				return self.MOVE_POSITIVE_REWARD
			else:
				self.latest_loss = loss
				return self.MOVE_NEGATIVE_REWARD

	def get_resized_region_image_features(self):
		img = self.img_array[self.crop_coordinates[0]:self.crop_coordinates[2], self.crop_coordinates[1]:self.crop_coordinates[3], :]
		img = misc.imresize(img, (224, 224))
		img = (img / 255.0).astype('float32')
		return self.feature_extractor.extract_fc7_features(img)

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
