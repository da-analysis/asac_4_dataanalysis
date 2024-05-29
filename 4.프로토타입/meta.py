from func import *

def meta_info(conn, cursor):
    asin2 = st.session_state.selected_asin2
    st.write(f"선택한 상품 ASIN2: {asin2}")

    # 상품 메타 정보 조회
    query = f"""
    SELECT brand, category, price, title, cat1, cat2, cat3 ,new_price, feature , description
    FROM `hive_metastore`.`asac`.`meta_cell_phones_and_accessories_fin`
    WHERE asin = '{asin2}'
    """
    cursor.execute(query)
    meta_info = cursor.fetchone()
    if meta_info:
        brand, category, price, title, cat1, cat2, cat3, new_price,feature,description = meta_info
        st.write(f"브랜드: {brand}")
        st.write(f"카테고리: {category}")
        st.write(f"상품명: {title}")
        st.write(f"대 카테고리: {cat1}")
        st.write(f"중 카테고리: {cat2}")
        st.write(f"소 카테고리: {cat3}")
        st.write(f"가격: {new_price}")
        
        st.write(f"상품 특징: {feature}")
        st.write(f"상품 설명: {description}")
    else:
        st.write("상품 정보를 찾을 수 없습니다.")

