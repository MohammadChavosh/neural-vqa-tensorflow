from utils import load_image_array, FeatureExtractor
from scipy import misc


class Environment:

	def __init__(self, img_path, question):
		self.img_array = load_image_array(img_path, False)
		self.question = question
		img_size = self.img_array.shape
		self.crop_coordinates = [0, 0, img_size[0], img_size[1]]
		self.x_alpha = img_size[0] / 10.0
		self.y_alpha = img_size[1] / 10.0
		self.last_loss = None #TODO: get first step loss

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
		img = self.img_array[self.crop_coordinates[0]:self.crop_coordinates[2], self.crop_coordinates[1]:self.crop_coordinates[3], :]
		img = misc.imresize(img, (224, 224))
		img = (img / 255.0).astype('float32')



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