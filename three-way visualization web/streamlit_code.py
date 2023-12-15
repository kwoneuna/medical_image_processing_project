import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from skimage import morphology
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import plotly.express as px
import seaborn as sns
import nibabel as nib
import scipy.ndimage as nd
import scipy.stats as stats
from skimage.filters import frangi, hessian,threshold_local
from skimage import measure
from skimage import filters, morphology,exposure
from skimage.color import rgb2gray
from scipy.stats import entropy, pearsonr,skew
from skimage.measure import shannon_entropy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load



radi_file_path = "/Users/mac/Desktop/의료영상처리/프로젝트/radiologist_memo.txt" #메모장 저장 path
clini_memo_path = '/Users/mac/Desktop/의료영상처리/프로젝트/clini_memo.txt'#메모장 저장 path 설정하기 
image_file_path = "/Users/mac/Desktop/의료영상처리/프로젝트/patient_img.jpg" #환자 이미지 저장 Path
 




st.set_page_config(page_title="Three-way", page_icon="🧑🏻‍⚕️🙍🏻‍♂️", layout="wide")

menu_selection = st.sidebar.selectbox("메뉴 선택", ["HOME", "For Radiologist", "For Clinician","For Patient"])


def draw_on_image(image_pil, output_path,coordinates):
    image_pil = image_pil.convert("RGB") ##여기추가
    # 이미지에 그림 그리기
    draw = ImageDraw.Draw(image_pil)
    draw.rectangle(coordinates, outline="red", width=2)
    image_pil.save(output_path)

def add_grid(image,grid_size): ##여기 함수 추가
    image = image.convert("RGB") ##여기추가
    # 격자 표시를 위한 Matplotlib 설정
    fig, ax = plt.subplots()
    ax.imshow(image)

    # 이미지 크기 가져오기
    image_width, image_height = image.size

    # 격자 크기 계산
    grid_width = image_width // grid_size
    grid_height = image_height // grid_size

    # 격자 그리기
    for i in range(1, grid_size):
        ax.axvline(i * grid_width, color='w', linestyle='--', linewidth=1,alpha=0.3)
        ax.axhline(i * grid_height, color='w', linestyle='--', linewidth=1,alpha=0.3)

    # X축 라벨 표시
    for i in range(0, image_width, grid_width):
        ax.text(i, -5, str(i), color='blue', fontsize=7, ha='center', va='center')

    # Y축 라벨 표시
    for i in range(0, image_height, grid_height):
        ax.text(-5, i, str(i), color='blue', fontsize=7, ha='center', va='center')

    # Matplotlib 플롯을 Streamlit에 표시
    st.pyplot(fig)



def radi_memo_page():
    st.title("소견서")

    # 사용자에게 텍스트 입력 받기
    user_input = st.text_area("소견을 입력하세요:", height=200)
    clicked = st.button("저장")
    # 저장 버튼
    if clicked:
        radi_save_memo(user_input)
        st.success("저장되었습니다.")
    



# 메모를 파일에 저장하는 함수
def radi_save_memo(memo_text):
    with open(radi_file_path, "w", encoding="utf-8") as file:
        file.write(memo_text)


# 파일에서 메모를 불러오는 함수
def radi_load_memo():
    try:
        with open(radi_file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return "저장된 메모가 없습니다."


def clini_memo_page():
    st.title("소견서")

    # 사용자에게 텍스트 입력 받기
    clini_input = st.text_area("소견을 입력하세요:"""," ", height=200)
    clicked = st.button("저장")
    # 저장 버튼
    if clicked:
        clini_save_memo(clini_input)
        st.success("저장되었습니다.")



# 메모를 파일에 저장하는 함수
def clini_save_memo(memo_text):
    with open(clini_memo_path, "w", encoding="utf-8") as file:
        file.write(memo_text)


# 파일에서 메모를 불러오는 함수
def clini_load_memo():
    try:
        with open(clini_memo_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return "저장된 메모가 없습니다."


def save_image(image):
    image.save(image_file_path)


def image_preprocessing(image_path,type=str):
    
    # gray scale로 open 
    # 논문에 보면 gray scale로 open하는 게 좋다고 함
    # 문제는 애초에 2D라는 거임, gray...
    # 혹시 몰라서 일단 넣어놓음..

    f = Image.open(image_path).convert('L')
    
    #image => numpy : 여러 연산 하려고
    f = np.array(f)


    ## equalized -> morphology.closing => median filter 적용
    equalized_image = exposure.equalize_adapthist(f)
    closing_image = morphology.closing(equalized_image,np.ones((3,3))) #check
    median_filtered_image = filters.median(equalized_image,np.ones((3,3))) #check

    vessel_enhanced = tophat(median_filtered_image)
    hessian_image = hessian(vessel_enhanced) # check 

    binary_image = vessel_enhanced > threshold_local(median_filtered_image, 81, offset=.05)
    if type == 'binary':
        return binary_image
    elif type == 'image':
        return vessel_enhanced


def left_right_volume(image,type):
    #사람마다 breast 위치가 다를 수 있기 때문에 z를 만들어 놓았음
    # 이후 데이터 확인 후 결정할 것 
    z = image.shape[1] * 5 //8
    z2 = image.shape[1]//2

    d = image.shape[0]//2

    right_chest_binary = image[:d,:z2]
    left_chest_binary = image[:d,-z2:]

    left_chest_volume = np.sum(left_chest_binary == 1)
    right_chest_volume = np.sum(right_chest_binary == 1)


    # type에 따라 binary mask를 반환할지, volume을 반환할지 결정
    if type == 'binary':
        return left_chest_binary
    elif type == 'volume':
        return left_chest_volume, right_chest_volume


def feature_extraction(image_path):
    path = image_path
    binary_image = image_preprocessing(image_path=path,type='binary')
    enhanced_image = image_preprocessing(image_path=path,type='image')
    #crop
    crop_mask = left_right_volume(image=binary_image,type='binary')
    right, left= left_right_volume(binary_image,type = 'volume')
    crop_volume = left

    #crop image
    crop_image = image_crop(enhanced_image)

    #feature
    sd_value = np.std(binary_image)
    entropy_value = entropy(binary_image.flatten())
    shannon_entropy_value = shannon_entropy(binary_image.flatten())
    mean_squared_error = np.mean((binary_image - np.mean(binary_image))**2)
    mean_value = np.mean(binary_image)
    skew_value = skew(crop_image.flatten())
    binary_skew_value = skew(crop_mask.flatten())

    return crop_volume, entropy_value, sd_value, mean_value, skew_value, binary_skew_value



def image_crop(image):
    z = image.shape[1]//2
    d = image.shape[0]//2

    right_chest_image = image[:d,:z]
    left_chest_image = image[:d,-z:]

    return left_chest_image


def tophat(image):
    selem = morphology.disk(20)
    vessel_enhanced = morphology.white_tophat(image, selem)
    return vessel_enhanced


def vessel(image):
    hxx, hxy, hyy = hessian_matrix(image, sigma=3)
    _, i2 = hessian_matrix_eigvals([hxx, hxy, hyy])
    return i2



# HOME 페이지
if menu_selection == "HOME":
    st.title("Three-way medical service")
    st.write("*Three-way medical service에 오신 것을 환영합니다.*")
    st.write("\n")
    st.info(
        '''
        Three-way medical service은 의사분들께 진단에 필요한 기능을 제공해 드리고, \n
        환자분들께는 질환에 대한 정보와 진단 결과를 제공하여 의사와 환자간의 communication을 증진시키기 위한 서비스입니다.\n
        더 나은 서비스를 제공하기 위해 최선을 다하겠습니다.
        '''
    )


if menu_selection == "For Radiologist":
    st.title("[ Radiologist page ]")
    uploaded_file = st.file_uploader("사진을 업로드하세요.", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        image_np = np.array(image_pil)
        save_image(image_pil)
        check=0

        # 이미지 처리
        image_tophat = tophat(image_np.copy())
        image_vessel = vessel(image_np.copy())
        image_vessel_normalized = (image_vessel - np.min(image_vessel)) / (np.max(image_vessel) - np.min(image_vessel))
        image_vessel_normalized = np.clip(image_vessel_normalized, 0.0, 1.0)


        st.write("## ▹ Patient Image")
        # 이미지 열을 만듭니다.
        col1,col2,col3 = st.columns(3)
        col1.image(image_np, use_column_width=True)
        col2.image(image_tophat, use_column_width=True)
        col3.image(image_vessel_normalized, use_column_width=True)
        st.write("\n")

        st.write("## ▹ Quantitative Evaluation")
        col4,col5 = st.columns(2)
        #predict 및 그래프 plot
        image_path = image_file_path
        f = Image.open(image_path).convert('L')
        f = np.array(f)
        equalized_image = exposure.equalize_adapthist(f)
        closing_image = morphology.closing(equalized_image,np.ones((3,3))) #check
        median_filtered_image = filters.median(equalized_image,np.ones((3,3))) #check

        vessel_enhanced = tophat(median_filtered_image)
        visualze_hessian = vessel(vessel_enhanced) # check 

        binary_image = vessel_enhanced > threshold_local(median_filtered_image, 81, offset=.05)

        result = feature_extraction(image_path)
        crop_volume, entropy_value, sd_value, mean_value, skew_value, binary_skew_value = result
        recur_0 = pd.read_csv('/Users/mac/Desktop/의료영상처리/프로젝트/recur_0.csv') #은아 언니가 준 csv 파일 path
        recur_1 = pd.read_csv('/Users/mac/Desktop/의료영상처리/프로젝트/recur_1.csv')
        recur_0['recur'] = 0
        recur_1['recur'] = 1
        df_combined = pd.concat([recur_0,recur_1],ignore_index=True)
        #print(df_combined.head())
        df_combined.to_csv('/Users/mac/Desktop/의료영상처리/프로젝트/concat_file.csv',index=False)   #concat 만들어서 저장할 path               

        df = pd.read_csv('/Users/mac/Desktop/의료영상처리/프로젝트/concat_file.csv') # 저장된 concat 파일 불러오는 path 위와 동일
        feature = df.drop('recur',axis=1)
        feature = feature.drop('number',axis=1)
        labels = df['recur']
        #feature 

        X_train, X_test,y_train,y_test = train_test_split(feature,labels,test_size=0.2,random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train,y_train)
        dump(model,'random_forest_model.joblib')
        load_model = load('random_forest_model.joblib')

        selected_features = ['volume', 'entropy', 'sd', 'mean', 'skew_value', 'binary_skew_value']

        # 임의의 ploint
        sample_data_point = {
            'volume': crop_volume,
            'entropy': entropy_value,
            'sd': sd_value,
            'mean': mean_value,
            'skew_value': skew_value,
            'binary_skew_value': binary_skew_value
        }
        fig, axs = plt.subplots(len(selected_features), 2, figsize=(10, 2*len(selected_features)))
        #predict 값
        new_data = pd.DataFrame(sample_data_point, index=[0])



        for i, feature in enumerate(selected_features):
            # recur_0 feature 분포
            axs[i, 0].hist(recur_0[feature], bins=20, alpha=0.5, color='blue', label='Recur 0')
            axs[i, 0].axvline(x=sample_data_point[feature], color='red', linestyle='dashed', linewidth=2, label='Sample Point')
            axs[i, 0].set_title(f'Distribution of {feature} for Recur 0')
            axs[i, 0].legend()

            # recur_1 feature 분포
            axs[i, 1].hist(recur_1[feature], bins=20, alpha=0.5, color='orange', label='Recur 1')
            axs[i, 1].axvline(x=sample_data_point[feature], color='red', linestyle='dashed', linewidth=2, label='Sample Point')
            axs[i, 1].set_title(f'Distribution of {feature} for Recur 1')
            axs[i, 1].legend()
        plt.tight_layout()

        image_file_path = "/Users/mac/Desktop/의료영상처리/프로젝트/plot_image.png" #플랏한 함수 저장할 path
        plt.savefig(image_file_path)
        col4.image(image_file_path,use_column_width=True)

        new_prediction = model.predict(new_data)
        new_probabilities = model.predict_proba(new_data)
        with col5:
            st.markdown("## ◽️ AI 재발 예측 결과")
            st.info(f'Predicted Class: {new_prediction[0]}')
            st.text(f'\t- Probability of Class 0: {new_probabilities[0][0]*100:.2f}%')
            st.text(f'- Probability of Class 1: {new_probabilities[0][1]*100:.2f}%')
        
        predicted_class = new_prediction[0]
        probabilities = [new_probabilities[0][0]*100, new_probabilities[0][1]*100]
        # SessionState를 사용하여 상태 저장
        if '1st_page_state' not in st.session_state:
            st.session_state['1st_page_state'] = {'predicted_class': predicted_class, 'probabilities': probabilities}


        st.write("\n")
        st.write("## ▹ ROI")
        col4,col5 = st.columns(2)
        height, width = image_vessel_normalized.shape

        grid_size = st.slider("격자 크기 선택", min_value=1, max_value=50, value=10) ##여기 추가
        image_to = Image.fromarray(image_tophat)
        
        col6,col7 = st.columns(2)
        with col6:
            add_grid(image_to, grid_size) ##여기 추가

        with col7:
            check=0
            # 사용자로부터 그릴 영역의 좌표를 입력 받기
            start_x = st.number_input('시작 X 좌표', 0, width, 0)
            end_x = st.number_input('끝 X 좌표', 0, width, width)
            start_y = st.number_input('시작 Y 좌표', 0, height, 0)
            end_y = st.number_input('끝 Y 좌표', 0, height, height)

            # 그릴 영역 좌표로 tuple 생성
            coordinates = (start_x, start_y, end_x, end_y)

            if st.button('ROI 추출'):
                output_path = "/Users/mac/Desktop/의료영상처리/프로젝트/drawn_image.jpg"  # ROI 이미지 컴퓨터 path 저장 path
                draw_on_image(image_to.copy(), output_path, coordinates)
                check=1

         
        st.write('\n')
        st.markdown("---")
        st.write('\n')
        col8,col9 = st.columns(2)
        if check == 1:
            image_draw = Image.open(output_path)
            col8.image(image_draw, use_column_width=True)

        with col9:
            radi_memo_page()


if menu_selection == "For Clinician":
    st.title("[ Clinician page ]")
    st.write("## ▹ Patient Image ")
    

    image_np = "/Users/mac/Desktop/의료영상처리/프로젝트/patient_img.jpg" #저장된 환자 이미지 가져오는 path 맨 위에 image_file_path에 저장해 논 path와 동일
    image_pil = Image.open(image_np)
    image_np_array = np.array(image_pil)

    image_pil = Image.open(image_np)
    image_np_array = np.array(image_pil)

    image_tophat = tophat(image_np_array.copy())
    image_vessel = vessel(image_np_array.copy())
    image_vessel_normalized = (image_vessel - np.min(image_vessel)) / (np.max(image_vessel) - np.min(image_vessel))
    image_vessel_normalized = np.clip(image_vessel_normalized, 0.0, 1.0)

    col1, col2, col3 = st.columns(3)
    col1.image(image_pil, use_column_width=True)
    col2.image(image_tophat, use_column_width=True)
    col3.image(image_vessel_normalized, use_column_width=True)
    st.write('\n')


    st.write("## ▹ Quantitative Evaluation")
    col4,col5 = st.columns(2)

    col4.image('/Users/mac/Desktop/의료영상처리/프로젝트/plot_image.png',use_column_width=True) #plot한 image path 가져오기
    with col5:
        # 1페이지에서 저장한 상태 불러오기
        if '1st_page_state' in st.session_state:
            state = st.session_state['1st_page_state']
            predicted_class = state['predicted_class']
            probabilities = state['probabilities']

        st.markdown("## ◽️ AI 재발 예측 결과")
        st.info(f'Predicted Class: {predicted_class}')
        st.text(f'\t- Probability of Class 0: {probabilities[0]:.2f}%')
        st.text(f'- Probability of Class 1: {probabilities[1]:.2f}%')



    st.write('\n')
    st.write("## ▹ Radiologist 소견")
    coll1,coll2 = st.columns(2)

    with coll1:
        image_pil = Image.open('/Users/mac/Desktop/의료영상처리/프로젝트/drawn_image.jpg')# 여기로 bounding box 이미지 들어와야함
        st.image(image_pil, caption="Patient Image", use_column_width=True)

    with coll2:
        if st.button("소견 불러오기"):
            loaded_memo = radi_load_memo()
            st.text("Radiologist 소견:")
            st.info(loaded_memo)


    st.write('\n')
    st.markdown("---")
    st.write('\n')
    clini_memo_page()



if menu_selection == "For Patient":
    st.title("[ 진단서 ]")
    st.write("# ▹ 유방암에 대해")

    st.write(

       ''' **1. 유방암이란 무엇인가요?**

        유방암은 유방 세포에서 시작된 암 종양으로, 조기 발견 시 치료 가능한 질병입니다. ''')


    st.write(
         ''' **2. 정기적인 건강 검진이 중요한 이유**

        조기 발견은 치료 성공의 핵심입니다. 정기적인 유방 검진을 통해 빠른 시일 내에 문제를 발견할 수 있습니다.''')

    st.write(
        '''  **3. 건강한 생활 습관의 중요성**

        규칙적인 운동, 건강한 식단, 스트레스 관리는 유방 건강을 지키는 데 도움이 될 수 있습니다.''')

    st.write(

     '''  **4. 의료 전문가와 상담**

        어떠한 의문이나 걱정사항이 있을 경우 의료 전문가와 상담하는 것이 중요합니다. 전문가의 도움을 받아보세요!''')

    st.write(

    '''   **5. 유용한 자원**

        - [한국 유방암 협회](https://www.kbca.info/)
        - [국립암센터 암 정보 서비스](https://www.cancer.go.kr/)

        환자 페이지는 여러분의 건강을 지키기 위한 정보와 지원을 제공합니다. 언제든지 의료 전문가와 상담하여 건강을 챙기세요!''')
    

    st.write('\n')
    st.write("## ▹ Quantitative Evaluation")
    col4,col5 = st.columns(2)

    col4.image('/Users/mac/Desktop/의료영상처리/프로젝트/plot_image.png',use_column_width=True) #플랏 이미지
    with col5:
        # 1페이지에서 저장한 상태 불러오기
        if '1st_page_state' in st.session_state:
            state = st.session_state['1st_page_state']
            predicted_class = state['predicted_class']
            probabilities = state['probabilities']


        st.markdown("## ◽️ AI 재발 예측 결과")
        st.info(f'Predicted Class: {predicted_class}')
        st.text(f'\t- Probability of Class 0: {probabilities[0]:.2f}%')
        st.text(f'- Probability of Class 1: {probabilities[1]:.2f}%')



    st.write('\n')
    st.markdown("---")
    st.write('\n')
    col1,col2 = st.columns(2)

    with col1:
        image_pil = Image.open('/Users/mac/Desktop/의료영상처리/프로젝트/drawn_image.jpg')# 여기로 bounding box 이미지 들어와야함
        st.image(image_pil, caption="Patient Image", use_column_width=True)
               
    with col2:
        loaded_memo = radi_load_memo()
        st.text("Radiologist 소견 소견:")
        st.info(loaded_memo)    

        st.write("\n")

        loaded_memo = clini_load_memo()
        st.text("Clinician 소견 소견:")
        st.info(loaded_memo)  
    