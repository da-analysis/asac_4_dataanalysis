import os
import streamlit as st
from databricks import sql
from PIL import Image, ImageOps
from io import BytesIO
import requests
import anthropic

# 이미지 크기 조정 함수 정의
def resize_image(image_url, target_width=150, target_height=150):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img_resized = img.resize((target_width, target_height), Image.LANCZOS)
    return img_resized

# 이미지 출력 함수 정의
def display_image(img, caption):
    st.image(img, caption=caption, width=150)

    
no_image_url = "?"

# 페이지 변경 함수
def change_page(page):
    st.session_state.page = page
    st.experimental_rerun()


# Databricks 연결 정보 설정
HOST = '?'
HTTP_PATH = '?'
PERSONAL_ACCESS_TOKEN = '?'

# Anthropic API 키 설정
YOUR_API_KEY = "?"
