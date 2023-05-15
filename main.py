import streamlit as st
# from PIL import Image
import numpy as np
# # import stain_utils as utils
# # import stainNorm_Reinhard
# # import stainNorm_Macenko
import stainNorm_Vahadane
import cv2
import gdown
import torch
import torch.nn as nn
import torchvision
from models import NestedUNet
from skimage import measure


# url="https://drive.google.com/uc?id=1OuVDKB1ElJ3DZyV-vSHULmRrUKPwIxxp"
# output_f = "my_model"
# gdown.download(url, output_f, quiet=False)
# st.write("done")

st.title("Image Uploader")
uploaded_file = st.file_uploader("Choose an image...", type=["jpeg", "jpg", "tif", "png"])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = NestedUNet(num_classes=[1, 1], input_channels=4, deep_supervision=True)
model = nn.DataParallel(model)
model.to(device)
model.load_state_dict(torch.load("my_model", map_location=device))
model.eval()
st.write(device)

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    n = stainNorm_Vahadane.Normalizer()
    hem = n.hematoxylin(image)

#     image = torch.from_numpy(image/255)
#     image = image.permute(2, 0, 1)

#     hem = torch.from_numpy(hem/255)
#     hem = hem.unsqueeze(0)
#     image = torch.cat((image, hem), 0)

#     image = image.to(device, dtype=torch.float32)
#     aa, bb = model(image.unsqueeze(0))
    st.image(hem, caption="Uploaded Image")
