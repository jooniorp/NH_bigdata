import pandas as pd
import numpy as np
import csv
import os

# CSV 경로 설정
PATH = os.path.join(os.getcwd(), 'data')

# CSV 파일 읽기
NHTI_etf = pd.read_csv(os.path.join(PATH, 'NHTI_ETF.csv'))
NHTI_stk = pd.read_csv(os.path.join(PATH, 'NHTI_STK.csv'))
ETF_info = pd.read_csv(os.path.join(PATH, 'ETF_INFO.csv'),encoding='cp949')

# 사용자 입력 기반 NHTI 계산 함수
def calculate_nhti_with_input(user_id):
    try:
        # USER_INFO 데이터에서 해당 사용자 ID의 데이터 필터링
        user_inputs = []
        with open(os.path.join(PATH, 'USER_INFO.csv'), mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == user_id:
                    user_inputs.append({"Stock_Code": str(row[1]), "Quantity": str(row[2]), "Purchase_Date": str(row[3])})

        # 사용자 입력 확인
        if not user_inputs:
            print(f"No transactions found for user ID: {user_id}.")
            return f"No transactions found for user ID: {user_id}."

        # Step 1: 총 매수 수량 계산
        total_quantity = sum(int(input_data["Quantity"]) for input_data in user_inputs)

        # [수정 1: 총 매수 수량 확인]
        if total_quantity == 0:
            print("Error: Total quantity of stocks purchased is zero.")
            return "Error: Total quantity of stocks purchased is zero."

        # Step 2: 가중합 점수 초기화
        weighted_scores = {
            "stk_Risk_Score": 0,
            "stk_Bandwagon_Score": 0,
            "stk_Interest_Score": 0,
            "stk_Volatility_Score": 0
        }

        # Step 3: 각 매수 내역에 대해 데이터 필터링 및 가중치 적용
        for input_data in user_inputs:
            stock_code = input_data["Stock_Code"]
            quantity = int(input_data["Quantity"])
            purchase_date = input_data["Purchase_Date"].replace("-", "")  # YYYYMMDD 형식으로 변환

            # NHTI_stk에서 매수 종목과 매수 일자를 기준으로 데이터 필터링
            filtered_data = NHTI_stk[(NHTI_stk['tck_iem_cd'] == stock_code) & (NHTI_stk['bse_dt'] == int(purchase_date))]

            if filtered_data.empty:
                print(f"No matching data found for stock code {stock_code} on date {purchase_date}.")
                continue

            # Step 4: 매수 수량을 기준으로 가중치 계산
            weight = (quantity / total_quantity) * 100

            # Step 5: 각 점수에 가중치를 적용하여 가중합 점수 계산
            weighted_scores["stk_Risk_Score"] += filtered_data.iloc[0]["stk_Risk_Score"] * (weight / 100)
            weighted_scores["stk_Bandwagon_Score"] += filtered_data.iloc[0]["stk_Bandwagon_Score"] * (weight / 100)
            weighted_scores["stk_Interest_Score"] += filtered_data.iloc[0]["stk_Interest_Score"] * (weight / 100)
            weighted_scores["stk_Volatility_Score"] += filtered_data.iloc[0]["stk_Volatility_Score"] * (weight / 100)

        # 결과 출력
        print("\nFinal Weighted Scores:")
        print(f"Risk: {weighted_scores['stk_Risk_Score']:.2f} / 100")
        print(f"Bandwagon: {weighted_scores['stk_Bandwagon_Score']:.2f} / 100")
        print(f"Interest: {weighted_scores['stk_Interest_Score']:.2f} / 100")
        print(f"Volatility: {weighted_scores['stk_Volatility_Score']:.2f} / 100")

        # Step 6: NHTI 유형 결정
        def determine_nhti_type(risk, bandwagon, interest, volatility):
            risk_type = 'A' if risk >= 50 else 'B'
            bandwagon_type = 'G' if bandwagon >= 50 else 'R'
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
        print(f"User's Final NHTI Type: {nhti_type}")

        # Step 7: 유사한 ETF 추천
        required_columns = ['Risk_Score', 'Bandwagon_Score', 'Interest_Score', 'Volatility_Score']
        missing_columns = [col for col in required_columns if col not in NHTI_etf.columns]
        if missing_columns:
            print(f"Error: Missing required columns in ETF dataset: {', '.join(missing_columns)}")
            return f"Error: Missing required columns in ETF dataset: {', '.join(missing_columns)}"

        NHTI_etf['Distance'] = np.sqrt(
            (NHTI_etf['Risk_Score'] - weighted_scores['stk_Risk_Score']) ** 2 +
            (NHTI_etf['Bandwagon_Score'] - weighted_scores['stk_Bandwagon_Score']) ** 2 +
            (NHTI_etf['Interest_Score'] - weighted_scores['stk_Interest_Score']) ** 2 +
            (NHTI_etf['Volatility_Score'] - weighted_scores['stk_Volatility_Score']) ** 2
        )

        # 거리 기준 상위 5개 추천 ETF
        top_5_recommendations = NHTI_etf.drop_duplicates(subset='etf_tck_cd').nsmallest(5, 'Distance')
        
        # [수정 3: 추천 결과가 비어있는 경우 처리]
        if top_5_recommendations.empty:
            print("No matching ETFs found for recommendation.")
            return "No matching ETFs found for recommendation."

        # 출력 결과
        result = f"\nUser's NHTI: {nhti_type}\nRecommended ETFs:\n"
        for idx, row in top_5_recommendations.iterrows():
            result += f"- {row['etf_tck_cd']} (NHTI Type: {row['NHTI_Type']}, Risk: {round(row['Risk_Score'])}, Bandwagon: {round(row['Bandwagon_Score'])}, Interest: {round(row['Interest_Score'])}, Volatility: {round(row['Volatility_Score'])})\n"

        print(result)
        return result

    except Exception as ex:
        print("Error occurred:", ex)
        return "Sorry, there was an issue processing the user's data."


# 특정 ETF 구성 요소 검색
def get_etf(etf_ticker: str) -> str:
    try:
        # 입력값 유효성 검사
        if not isinstance(etf_ticker, str) or etf_ticker.strip() == "":
            return "Error: ETF ticker cannot be empty. Please provide a valid ticker."

        # ETF 데이터프레임 유효성 검사
        if ETF_info.empty:
            return "Error: The ETF dataset is empty. Please check the data source."

        # ETF 데이터 필터링
        etf_data = ETF_info[ETF_info['ETF_Ticker_Code'] == etf_ticker]

        # [수정 4: 필수 컬럼 확인]
        required_columns = {'Stock_Item_Code', 'Sector_Name_English_Y', 'Website_Address', 'Ex_Dividend_Date', 'Dividend_Amount_Per_Stock', 'Dividend_Payment_Currency_Code'}
        missing_columns = [col for col in required_columns if col not in etf_data.columns]
        if missing_columns:
            return f"Error: Missing required columns in the dataset: {', '.join(missing_columns)}."

        # [수정 5: 결과 데이터프레임이 비어있는 경우 처리]
        if etf_data.empty:
            return f"Error: No components found for ETF '{etf_ticker}'."

        # 결과 반환: 필수 컬럼만 반환
        return etf_data[['Stock_Item_Code', 'Sector_Name_English_Y', 'Website_Address', 'Ex_Dividend_Date', 'Dividend_Amount_Per_Stock', 'Dividend_Payment_Currency_Code']].reset_index(drop=True)

    except KeyError as ke:
        print("KeyError in get_etf:", ke)
        return "Error: Missing required keys in the ETF dataset."
    except Exception as ex:
        print("Error in get_etf:", ex)
        return "Error: An unexpected issue occurred while processing the ETF data."
#print(get_etf("FDRV"))