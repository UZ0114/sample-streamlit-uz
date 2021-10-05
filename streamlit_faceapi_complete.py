#!/usr/bin/env python
# coding: utf-8

# In[87]:


import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import requests
import json
import io


# In[88]:


url= "https://japaneast.api.cognitive.microsoft.com/customvision/v3.0/Prediction/535c8c00-9724-4c33-b019-82a1ee534f49/classify/iterations/Iteration1/image"
headers = {
    "Prediction-Key":"4c1ae36ebc48444695ca95c9419de750",
    "Content-Type":"application/octet-stream"
}

st.title(r"UZのAI画像認識アプリ")


# In[92]:


upload_img = st.file_uploader("画像を選んでください",type="jpg")
if upload_img is not None:
    img = Image.open(upload_img)
    with io.BytesIO() as output:#画像をバイナリ形式にしてリクエストする
        img.save(output,format="JPEG")
        binary_img = output.getvalue()
    r = requests.post(url,data=binary_img,headers=headers)
    pred = r.json()["predictions"]
    st.image(img,caption="アップロード画像",use_column_width=True)
    st.write("■AIによる予測結果")
    st.dataframe(pd.DataFrame(pred).drop("tagId",axis=1),width=2000,height=1500)


# In[85]:


df = pd.DataFrame(
np.random.randn(20,3),
columns=["国語","数学","英語"]
)
#if (st.checkbox(r"表示しますか？")) & (upload_img is not None):
if st.checkbox(r"表示しますか？"):
    st.line_chart(df)
    st.bar_chart(df)
    st.area_chart(df)


# In[ ]:





# In[ ]:


#df.style.highlight_max(axis=0)


# In[ ]:


#画像をバイナリ形式にしてリクエストする
# with io.BytesIO() as output:
#     img.save(output,format="JPEG")
#     binary_img = output.getvalue()

