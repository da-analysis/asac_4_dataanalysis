from func import *


# Anthropic API 클라이언트 설정
client = anthropic.Anthropic(api_key=YOUR_API_KEY)

def get_summary_and_keywords(review_text):
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0,
        system="You are the assistant who summarizes 10 reviews in Korean language  and provides 5 keywords in Korean language.\n1. 전반적인 리뷰 요약 10가지 \n2. 핵심 키워드 5가지 \n 한국어로 전박전인 리뷰요약,핵심 키워드 5가지 문단으로 구성해서 보여줘" ,
        messages=[
            {"role": "user", "content": [{"type": "text", "text": review_text}]}
        ]
    )
    content = message.content[0].text
    content = content.replace('\\n', '\n')
    content = content.replace('[TextBlock(text=\'', '')
    content = content.replace(', type=\'text\')]', '')
    return content


def get_review_data(asin1):
    with sql.connect(server_hostname=HOST, http_path=HTTP_PATH, access_token=PERSONAL_ACCESS_TOKEN) as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT asin1, asin2, cosine_top3, asin1_combined_reviews, asin2_combined_reviews, asin1_image_url, asin2_image_url FROM hive_metastore.asac.text_summary WHERE asin1 = '{asin1}';")
            result = cursor.fetchone()
            if result:
                return result
            else:
                return None

def main():
    st.title("리뷰 요약 및 키워드 생성기")
    asin1 = st.text_input("ASIN1을 입력하세요:")
    
    if st.button("요약 및 키워드 생성"):
        review_data = get_review_data(asin1)
        
        if review_data:
            asin1, asin2, cosine_top3, asin1_combined_reviews, asin2_combined_reviews, asin1_image_url, asin2_image_url = review_data
            
            st.write(f"유사도: {cosine_top3}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"기준 상품 ({asin1})")
                asin1_image = resize_image(asin1_image_url if asin1_image_url else no_image_url)
                display_image(asin1_image, "기준 상품 이미지")
                asin1_summary = get_summary_and_keywords(asin1_combined_reviews)
                if asin1_summary:
                    st.markdown(asin1_summary)
                else:
                    st.write("기준상품 리뷰 요약 및 키워드 생성에 실패했습니다.")
            
            with col2:
                st.subheader(f"추천 상품 ({asin2})")
                asin2_image = resize_image(asin2_image_url if asin2_image_url else no_image_url)
                display_image(asin2_image, "추천 상품 이미지")
                asin2_summary = get_summary_and_keywords(asin2_combined_reviews)
                if asin2_summary:
                    st.markdown(asin2_summary)
                else:
                    st.write("추천 상품 리뷰 요약 및 키워드 생성에 실패했습니다.")

if __name__ == "__main__":
    main()