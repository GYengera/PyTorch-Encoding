import torch
import encoding
import os

# Get the model
model = encoding.models.get_model('Encnet_ResNet101_PContext', pretrained=True).cuda()
model.eval()

# Prepare the image
# url = 'https://github.com/zhanghang1989/image-data/blob/master/' + \
#       'encoding/segmentation/ade20k/ADE_val_00001142.jpg?raw=true'
# filename = 'example.jpg'
# img = encoding.utils.load_image(
#     encoding.utils.download(url, filename)).cuda().unsqueeze(0)
# img = encoding.utils.load_image('./image1.jpg').cuda().unsqueeze(0)

input_dir = '/home/yengera/Saarland/hlcv/project/data/mountain'
output_dir = '/home/yengera/Saarland/hlcv/project/data/mountain_segment'
os.mkdir(output_dir)

files = os.listdir('/home/yengera/Saarland/hlcv/project/data/mountain/')
files.remove('Thumbs.db')


for fimg in files:
	# Make prediction
	img = encoding.utils.load_image(os.path.join(input_dir, fimg)).cuda().unsqueeze(0)
	output = model.evaluate(img)
	predict = torch.max(output, 1)[1].cpu().numpy() + 1

	# Get color pallete for visualization
	mask = encoding.utils.get_mask_pallete(predict, 'detail')
	mask.save(os.path.join(output_dir, fimg[:-4]+'.png'))
