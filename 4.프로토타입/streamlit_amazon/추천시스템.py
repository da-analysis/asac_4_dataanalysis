from meta import meta_info
from func import *


# Databricks 연결
with sql.connect(server_hostname=HOST, http_path=HTTP_PATH, access_token=PERSONAL_ACCESS_TOKEN) as conn:
    with conn.cursor() as cursor:
        # Streamlit 앱
        st.title("추뽀취뽀 추천시스템")

        # 세션 상태 초기화
        if "page" not in st.session_state:
            st.session_state.page = "main"
        if "asin1" not in st.session_state:
            st.session_state.asin1 = ""
        if "recommendations" not in st.session_state:
            st.session_state.recommendations = []
        if "selected_asin2" not in st.session_state:
            st.session_state.selected_asin2 = ""
#######
        




#####
        # 메인 페이지
        if st.session_state.page == "main":
            # 사이드바 설정
            st.sidebar.title("기준 상품 입력")
            asin1 = st.sidebar.text_input("기준 상품 입력", st.session_state.asin1, key="_asin1")
            st.session_state.asin1 = asin1

            # 랜덤 asin1 선택 버튼 추가
            if st.sidebar.button("랜덤 선택"):
                query = "SELECT asin1 FROM `hive_metastore`.`asac`.`asin1_unique` ORDER BY RAND() LIMIT 1"
                cursor.execute(query)
                random_asin1 = cursor.fetchone()[0]
                st.session_state.asin1 = random_asin1
                asin1 = random_asin1

            if asin1:
                # 입력한 asin1에 대한 정보 조회
                query = f"""
                SELECT asin1, asin1_image_url
                FROM `hive_metastore`.`asac`.`image_model_final`
                WHERE asin1 = '{asin1}'
                """
                cursor.execute(query)
                item1 = cursor.fetchone()

                if item1 and item1[1]:  # item1과 item1[1]이 유효한지 확인
                    # 이미지 URL에서 이미지 불러오기
                    response = requests.get(item1[1])
                    img = Image.open(BytesIO(response.content))

                    # 이미지 크기 조정
                    width = 200
                    img = img.resize((width, int(img.height * (width / img.width))), Image.LANCZOS)

                    # 이미지 출력
                    st.sidebar.image(img, caption=f"입력한 상품 (asin: {asin1})", use_column_width=True)
                else:
                    st.sidebar.write(f"입력한 ASIN1: {asin1}에 대한 이미지 찾을 수 없습니다.")

                # 추천 결과 생성
                recommendations = []

                # 이미지 기반 모델 결과
                query = f"""
                SELECT asin2, asin2_image_url, cosine_similarity
                FROM `hive_metastore`.`asac`.`image_model_final`
                WHERE asin1 = '{asin1}'
                ORDER BY cosine_similarity DESC
                LIMIT 5
                """
                cursor.execute(query)
                similar_items = cursor.fetchall()
                recommendations.extend(similar_items)

                # 아이템 기반 모델 결과
                query = f"""
                SELECT asin2, asin2_image_url, rank, prob
                FROM `hive_metastore`.`asac`.`item_model_final`
                WHERE asin1 = '{asin1}'
                ORDER BY rank ASC
                LIMIT 5
                """
                cursor.execute(query)
                similar_items = cursor.fetchall()
                recommendations.extend(similar_items)

                # 텍스트 기반 모델 결과
                query = f"""
                SELECT asin2, asin2_image_url, cosine_top3
                FROM `hive_metastore`.`asac`.`text_model_final`
                WHERE asin1 = '{asin1}'
                ORDER BY cosine_top3 DESC
                LIMIT 5
                """
                cursor.execute(query)
                similar_items = cursor.fetchall()
                recommendations.extend(similar_items)

                st.session_state.recommendations = recommendations

                # 추천 결과 출력
                with st.expander("이미지 기반 모델 결과"):
                    cols = st.columns(5)
                    for i, (asin2, asin2_image_url, cosine_similarity) in enumerate(st.session_state.recommendations[:5]):
                        if asin2_image_url:
                            img_resized = resize_image(asin2_image_url)
                            with cols[i % 5]:
                                if st.button(f"추천상품: {asin2}", key=f"image_{i}"):
                                    st.session_state.selected_asin2 = asin2
                                    change_page("메타정보")
                                display_image(img_resized, f"유사도: {cosine_similarity:.3f}")
                        else:
                            img_resized = resize_image(no_image_url)
                            with cols[i % 5]:
                                if st.button(f"추천상품: {asin2}", key=f"image_{i}"):
                                    st.session_state.selected_asin2 = asin2
                                    change_page("메타정보")
                                display_image(img_resized, f"유사도: {cosine_similarity:.3f}")

                with st.expander("하이브리드 모델 결과"):
                    cols = st.columns(5)
                    for i, (asin2, asin2_image_url, rank, prob) in enumerate(st.session_state.recommendations[5:10]):
                        if asin2_image_url:
                            img_resized = resize_image(asin2_image_url)
                            with cols[i % 5]:
                                if st.button(f"추천상품: {asin2}", key=f"item_{i}"):
                                    st.session_state.selected_asin2 = asin2
                                    change_page("메타정보")
                                display_image(img_resized, f"prob: {prob:.3f}")
                        else:
                            img_resized = resize_image(no_image_url)
                            with cols[i % 5]:
                                if st.button(f"추천상품: {asin2}", key=f"item_{i}"):
                                    st.session_state.selected_asin2 = asin2
                                    change_page("메타정보")
                                display_image(img_resized, f"prob: {prob:.3f}")

                with st.expander("텍스트 기반 모델 결과"):
                    cols = st.columns(5)
                    for i, (asin2, asin2_image_url, cosine_top3) in enumerate(st.session_state.recommendations[10:15]):
                        if asin2_image_url:
                            img_resized = resize_image(asin2_image_url)
                            with cols[i % 5]:
                                if st.button(f"추천상품: {asin2}", key=f"text_{i}"):
                                    st.session_state.selected_asin2 = asin2
                                    change_page("메타정보")
                                display_image(img_resized, f"유사도: {cosine_top3:.3f}")
                        else:
                            img_resized = resize_image(no_image_url)
                            with cols[i % 5]:
                                if st.button(f"추천상품: {asin2}", key=f"text_{i}"):
                                    st.session_state.selected_asin2 = asin2
                                    change_page("메타정보")
                                display_image(img_resized, f"유사도: {cosine_top3:.3f}")

        # 메타정보 페이지
        elif st.session_state.page == "메타정보":
            if st.button("메인 페이지로 이동"):
                change_page("main")
            meta_info(conn, cursor)