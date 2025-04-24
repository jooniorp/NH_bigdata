import pandas as pd
import numpy as np
import csv
import os

PATH = os.path.join(os.getcwd(),'data')

# CSV 파일 불러오기
NHTI_etf = pd.read_csv(os.path.join(PATH,'NHTI_ETF.csv'))
NHTI_stk = pd.read_csv(os.path.join(PATH,'NHTI_STK.csv'))

def calculate_nhti_with_input(user_id):
    user_inputs = []

    with open(os.path.join(PATH,'USER_INFO.csv'), mode = 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0]==user_id:
                user_inputs.append({"매수 종목": str(row[1]), "매수 수량": str(row[2]), "매수 일자": str(row[3])})

    # Step 1: 총 매수 수량 계산
    total_quantity = sum(int(input_data["매수 수량"]) for input_data in user_inputs)

    # Step 2: 가중합 점수 초기화
    weighted_scores = {
        "stk_Risk_Score": 0,
        "stk_Bandwagon_Score": 0,
        "stk_Interest_Score": 0,
        "stk_Volatility_Score": 0
    }

    # Step 3: 각 매수 내역에 대해 데이터 필터링 및 가중치 적용
    for input_data in user_inputs:
        stock_code = input_data["매수 종목"]
        quantity = int(input_data["매수 수량"])
        purchase_date = input_data["매수 일자"].replace("-", "")  # YYYYMMDD 형식으로 변환

        # 필터링: 매수 종목과 매수 일자를 기준으로 데이터 필터링
        filtered_data = NHTI_stk[(NHTI_stk['tck_iem_cd'] == stock_code) & (NHTI_stk['bse_dt'] == int(purchase_date))]

        if not filtered_data.empty:
            # Step 4: 매수 수량을 기준으로 가중치 계산 (100 기준)
            weight = (quantity / total_quantity) * 100

            # Step 5: 각 점수에 가중치를 적용하여 가중합 점수 계산
            weighted_scores["stk_Risk_Score"] += filtered_data.iloc[0]["stk_Risk_Score"] * (weight / 100)
            weighted_scores["stk_Bandwagon_Score"] += filtered_data.iloc[0]["stk_Bandwagon_Score"] * (weight / 100)
            weighted_scores["stk_Interest_Score"] += filtered_data.iloc[0]["stk_Interest_Score"] * (weight / 100)
            weighted_scores["stk_Volatility_Score"] += filtered_data.iloc[0]["stk_Volatility_Score"] * (weight / 100)

    # Step 6: 최종 점수 결과 출력
    print("\n사용자의 최종 NHTI 성향 점수:")
    print(f"Risk: {weighted_scores['stk_Risk_Score']:.2f} / 100")
    print(f"Bandwagon: {weighted_scores['stk_Bandwagon_Score']:.2f} / 100")
    print(f"Interest: {weighted_scores['stk_Interest_Score']:.2f} / 100")
    print(f"Volatility: {weighted_scores['stk_Volatility_Score']:.2f} / 100")

    # Step 7: 최종 NHTI 타입 결정
    def determine_nhti_type(risk, crowd, interest, volatility):
        risk_type = 'A' if risk >= 50 else 'B'
        bandwagon_type = 'G' if crowd >= 50 else 'R'
        interest_type = 'P' if interest >= 50 else 'U'
        volatility_type = 'X' if volatility >= 50 else 'S'
        return risk_type, bandwagon_type, interest_type, volatility_type

    risk_type, bandwagon_type, interest_type, volatility_type = determine_nhti_type(
        weighted_scores["stk_Risk_Score"],
        weighted_scores["stk_Bandwagon_Score"],
        weighted_scores["stk_Interest_Score"],
        weighted_scores["stk_Volatility_Score"]
    )

    nhti_type = f"{risk_type}{bandwagon_type}{interest_type}{volatility_type}"
    print(f"사용자의 최종 NHTI 유형: {nhti_type}")

    # Step 8: 추천 종목 추출 (사용자의 성향과 유사한 종목 찾기)
    NHTI_etf['Distance'] = np.sqrt(
        (NHTI_etf['Risk_Score'] - weighted_scores['stk_Risk_Score'])**2 +
        (NHTI_etf['Bandwagon_Score'] - weighted_scores['stk_Bandwagon_Score'])**2 +
        (NHTI_etf['Interest_Score'] - weighted_scores['stk_Interest_Score'])**2 +
        (NHTI_etf['Volatility_Score'] - weighted_scores['stk_Volatility_Score'])**2
    )

    # 거리 기준 상위 5개 종목 추천(중복 제거)
    top_5_recommendations = NHTI_etf.drop_duplicates(subset='etf_tck_cd').nsmallest(5, 'Distance')

    # 출력 포맷
    a = f"""\nYour NHTI is {nhti_type}. {risk_type} strength of {round(weighted_scores['stk_Risk_Score'])}, 
          {bandwagon_type} strength of {round(weighted_scores['stk_Bandwagon_Score'])}, 
          {interest_type} strength of {round(weighted_scores['stk_Interest_Score'])}, 
          {volatility_type} strength of {round(weighted_scores['stk_Volatility_Score'])}.\n
          Here is recommended ETF for you:\n"""

    b = ""
    for idx, row in top_5_recommendations.iterrows():
        b += f"- {row['etf_tck_cd']}, ETF NHTI: {row['NHTI_Type']}, {risk_type} strength of {round(row['Risk_Score'])}, {bandwagon_type} strength of {round(row['Bandwagon_Score'])}, {interest_type} strength of {round(row['Interest_Score'])}, {volatility_type} strength of {round(row['Volatility_Score'])}"

    print(a + b)
    return a + b

# 함수 호출
# calculate_nhti_with_input('USR_A')
# calculate_nhti_with_input('USR_B')
# calculate_nhti_with_input('USR_C')
# calculate_nhti_with_input('USR_D')
# calculate_nhti_with_input('USR_E')

def get_etf(etf_id):
    result = """
            ETF Composition:
            - GGL : Google, URL : http://google.com
            - APL : Apple, URL : http://apple.com
            - MT : Meta, URL : http://meta.com
    """