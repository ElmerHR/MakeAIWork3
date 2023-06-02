import os
import xml.etree.ElementTree as ET

path_train = '/home/elmer/dev_debian/working/python3/MakeAIWork3/project_3/yolo3/renamed_files/train'
path_test = '/home/elmer/dev_debian/working/python3/MakeAIWork3/project_3/yolo3/renamed_files/test'

paths = [path_train, path_test]

# iterate over xml files
for path in paths:
	for filename in os.listdir(path):
		if os.path.splitext(filename)[-1] == '.xml':
			tree = ET.parse(os.path.join(path, filename))
			root = tree.getroot()

			# retrieve values from root tree
			filename_img = filename[:-4] + '.jpg'
			root[1].text = filename_img
			root[2].text = os.path.join(path, filename_img)
			root[6][0].text = root[0].text
			# save edited xml file
			tree.write(os.path.join(path, filename))

