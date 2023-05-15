import random
import streamlit as st
# from PIL import Image
import numpy as np
# import stain_utils as utils
# import stainNorm_Reinhard
# import stainNorm_Macenko
import stainNorm_Vahadane
import cv2
import torch
import gdown
import torch.nn as nn
import torchvision
from models import NestedUNet
from skimage import measure
from skimage.segmentation import watershed

# url="https://drive.google.com/uc?id=1OuVDKB1ElJ3DZyV-vSHULmRrUKPwIxxp"
# output_f = "my_model"
# gdown.download(url, output_f, quiet=False)
# st.write("done")

st.title("Instance nuclei segmentation")
uploaded_file = st.file_uploader("Choose an image...", type=["jpeg", "jpg", "tif", "png"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
	model = NestedUNet(num_classes=[1, 1], input_channels=4, deep_supervision=True)
	model = nn.DataParallel(model)
	model.to(device)
	model.load_state_dict(torch.load("my_model", map_location=device))
	model.eval()
	return model

moddy = load_model()

@st.cache_data
def get_hema(im):
	n = stainNorm_Vahadane.Normalizer()
	hemax = n.hematoxylin(im)
	return hemax


@st.cache_data
def mywater(p_sema2, p_mark2, im_bord2):
	markers = measure.label(p_mark2 > 0.5)
	labels = watershed(-1.0 * p_sema2, markers, mask=p_sema2)
	for i in np.unique(labels)[1:]:
		temp = 1.0 * (labels == i)
		labels[labels == i] = random.randint(20, 255)
		temp = temp - cv2.erode(temp, np.ones((3, 3)), iterations=2)
		im_bord2[temp > 0] = (0, 0, 255)
	return labels, im_bord2


if uploaded_file is not None:
	image1 = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
	image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
	im_bord = np.copy(image)
	# st.image(image, caption="sghj Image")
	hem = get_hema(image)
	st.write("<h1 style='text-align: center; font-weight: bold;font-size: 20px;'>Input Image and Hematoxylin Extraction</h1>", unsafe_allow_html=True)
	col1, col2 = st.columns(2)
	# Add each image to a separate column
	with col1:
		st.image(image, caption="Input image")
	with col2:
		st.image(hem, caption="hematoxylin channel")
	
	image = torch.from_numpy(image1/255.0)
	image = image.permute(2, 0, 1)
	hem = torch.from_numpy(hem)
	hem = hem.unsqueeze(0)
	image = torch.cat((image, hem), 0)
	image = image.to(device, dtype=torch.float32)

	aa, bb = moddy(image.unsqueeze(0))
	aa = torch.sigmoid(aa)
	bb = torch.sigmoid(bb)
	# st.write(str(aa.shape))
	# st.write(str(bb.shape))
	aa = 255*aa.detach().numpy()[0][0]
	bb = 255*bb.detach().numpy()[0][0]
	p_sema = 1*(aa/255.0>0.5)
	p_mark = 1*(bb/255.0>0.5)
	# aa = 255*(aa>0.5)
	cc = 255.0*(1.0*(aa>100)+1.0*(bb>100))/2.0
	aa = aa.astype(np.uint8)
	bb = bb.astype(np.uint8)
	cc = cc.astype(np.uint8)

	aa = cv2.applyColorMap(aa, cv2.COLORMAP_VIRIDIS)
	bb = cv2.applyColorMap(bb, cv2.COLORMAP_VIRIDIS)
	cc = cv2.applyColorMap(cc, cv2.COLORMAP_VIRIDIS)
	# st.write("The two Outputs of the Model")
	st.write("<h1 style='text-align: center; font-weight: bold;font-size: 20px;'>The outputs of the model</h1>", unsafe_allow_html=True)

	# aa = np.repeat(aa[:, :, np.newaxis], 3, axis=2)
	col3, col4, colm = st.columns(3)
	with col3:
		st.image(aa, caption="semantic output", clamp=True)
	with col4:
		st.image(bb, caption="nuclei marker output", clamp=True)
	with colm:
		st.image(cc, caption="marker-border", clamp=True)

	labels2, im_bord = mywater(p_sema, p_mark, im_bord)
	# st.write("Instance Segmentation Results After Watershed")
	st.write("<h1 style='text-align: center; font-weight: bold;font-size: 20px;'>Instance Segmentation Results After Watershed</h1>", unsafe_allow_html=True)

	labels2 = labels2.astype(np.uint8)
	im_bord = im_bord.astype(np.uint8)
	labels2 = cv2.applyColorMap(labels2, cv2.COLORMAP_VIRIDIS)
	col5, col6 = st.columns(2)
	with col5:
		st.image(labels2, caption="Integer encoded", clamp=True)
	with col6:
		st.image(im_bord, caption="borders labelled", clamp=True)
	# st.write(str(np.unique(aa)))
	# st.write(str(type(aa)))
