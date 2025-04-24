import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import shapiro, probplot
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os

full_path  = os.getcwd()
data_path  = os.path.join(full_path, 'data')

#NH_CONTEST_NW_FC_STK_IEM_IFO, NH_CONTEST_STK_DT_QUT, NH_CONTEST_NHDATA_STK_DD_IFO 3개 파일 병합
# CSV 파일 읽기 (인코딩 지정) 다른 병합 데이터와 구분하기 위해 이름 변경
df1 = pd.read_csv(os.path.join(data_path, 'NH_CONTEST_NW_FC_STK_IEM_IFO.csv'), encoding='cp949')
df2 = pd.read_csv(os.path.join(data_path, 'NH_CONTEST_STK_DT_QUT.csv'), encoding='cp949')
df3 = pd.read_csv(os.path.join(data_path, 'NH_CONTEST_NHDATA_STK_DD_IFO.csv'), encoding='cp949')

# 공백 제거 및 대문자 변환
df1['tck_iem_cd'] = df1['tck_iem_cd'].astype(str).str.strip().str.upper()
df2['tck_iem_cd'] = df2['tck_iem_cd'].astype(str).str.strip().str.upper()
df3['tck_iem_cd'] = df3['tck_iem_cd'].astype(str).str.strip().str.upper()

# bse_dt 공백 제거
df2['bse_dt'] = df2['bse_dt'].astype(str).str.strip()
df3['bse_dt'] = df3['bse_dt'].astype(str).str.strip()

# df1에서 tck_iem_cd와 ltg_tot_stk_qty 가져오기
ltg_tot_stk_qty_df = df1[['tck_iem_cd', 'ltg_tot_stk_qty']]

# df2와 df3에서 공통 bse_dt와 tck_iem_cd 찾기
common_keys_df2_df3 = pd.merge(df2[['bse_dt', 'tck_iem_cd']], df3[['bse_dt', 'tck_iem_cd']], on=['bse_dt', 'tck_iem_cd'])
filtered_df2 = df2[df2.set_index(['bse_dt', 'tck_iem_cd']).index.isin(common_keys_df2_df3.set_index(['bse_dt', 'tck_iem_cd']).index)]
filtered_df3 = df3[df3.set_index(['bse_dt', 'tck_iem_cd']).index.isin(common_keys_df2_df3.set_index(['bse_dt', 'tck_iem_cd']).index)]

# df2에 ltg_tot_stk_qty를 tck_iem_cd 기준으로 결합
filtered_df2 = filtered_df2.merge(ltg_tot_stk_qty_df, on='tck_iem_cd', how='left')

# 시가총액을 시계열 데이터로 만들기 위해 종목종가와 곱해서 mkt_pr_tot_amt 생성 (ltg_tot_stk_qty와 IEM_END_PR 곱하기)
filtered_df2['mkt_pr_tot_amt'] = filtered_df2['ltg_tot_stk_qty'] * filtered_df2['iem_end_pr']

# ltg_tot_stk_qty 컬럼 삭제
filtered_df2.drop(columns=['ltg_tot_stk_qty'], inplace=True)

# 최종 병합: df2와 df3
merged_df = pd.merge(filtered_df2, filtered_df3, on=['bse_dt', 'tck_iem_cd'], suffixes=('_df2', '_df3'))

# 특정 조건에 맞는 값 업데이트
merged_df.loc[(merged_df['tck_iem_cd'] == 'ZVRA') & (merged_df['bse_dt'] == '20240802'),
               ['iem_ong_pr', 'iem_hi_pr', 'iem_low_pr', 'iem_end_pr']] = 6.3

# 결과를 새로운 CSV 파일로 저장
output_file = 'merge_output.csv'  # 결과 파일 이름
merged_df.to_csv(data_path+'/'+output_file, index=False)

# 병합한 CSV 파일 불러오기
df = pd.read_csv(data_path+'/'+output_file, encoding='cp949')

# 데이터 정렬
df.sort_values(by=['tck_iem_cd', 'bse_dt'], inplace=True)

# TR 계산
def calculate_tr(group):
    group['Previous Close'] = group['iem_end_pr'].shift(1)
    group['TR'] = group[['iem_hi_pr', 'iem_low_pr', 'Previous Close']].apply(
        lambda x: max(x['iem_hi_pr'] - x['iem_low_pr'],
                      abs(x['iem_hi_pr'] - x['Previous Close']),
                      abs(x['iem_low_pr'] - x['Previous Close'])), axis=1)
    return group

# NATR 계산 (여러 기간)
def calculate_natr_multiple(group, periods=[7, 14, 30]):
    group = calculate_tr(group)  # TR 계산
    for period in periods:
        group[f'NATR_{period}'] = (group['TR'].rolling(window=period).mean() / group['iem_end_pr'])
    return group

# 반올림 함수
def round_values(group, columns):
    for column in columns:
        group[column] = group[column].round(4)  # 셋째 자리에서 반올림
    return group

# 그룹화 및 변화량 계산
def calculate_trends(group):
    # 인덱스를 bse_dt로 설정
    group.set_index('bse_dt', inplace=True)

    # 변화량 계산 (비거래일 고려)
    group['slope_tco_avg_eal_pls_1d'] = (group['tco_avg_eal_pls'].shift(1) - group['tco_avg_eal_pls']) / 1
    group['slope_tco_avg_eal_pls_5d'] = (group['tco_avg_eal_pls'].shift(5) - group['tco_avg_eal_pls']) / 5
    group['slope_tco_avg_eal_pls_14d'] = (group['tco_avg_eal_pls'].shift(14) - group['tco_avg_eal_pls']) / 14
    group['slope_tco_avg_pft_rt_1d'] = (group['tco_avg_pft_rt'].shift(1) - group['tco_avg_pft_rt']) / 1
    group['slope_tco_avg_pft_rt_5d'] = (group['tco_avg_pft_rt'].shift(5) - group['tco_avg_pft_rt']) / 5
    group['slope_tco_avg_pft_rt_14d'] = (group['tco_avg_pft_rt'].shift(14) - group['tco_avg_pft_rt']) / 14
    group['slope_lss_ivo_rt_1d'] = (group['lss_ivo_rt'].shift(1) - group['lss_ivo_rt']) / 1
    group['slope_lss_ivo_rt_5d'] = (group['lss_ivo_rt'].shift(5) - group['lss_ivo_rt']) / 5
    group['slope_lss_ivo_rt_14d'] = (group['lss_ivo_rt'].shift(14) - group['lss_ivo_rt']) / 14
    group['slope_ifw_act_cnt_1d'] = (group['ifw_act_cnt'].shift(1) - group['ifw_act_cnt']) / 1
    group['slope_ifw_act_cnt_5d'] = (group['ifw_act_cnt'].shift(5) - group['ifw_act_cnt']) / 5
    group['slope_ifw_act_cnt_14d'] = (group['ifw_act_cnt'].shift(14) - group['ifw_act_cnt']) / 14
    group['slope_ofw_act_cnt_1d'] = (group['ofw_act_cnt'].shift(1) - group['ofw_act_cnt']) / 1
    group['slope_ofw_act_cnt_5d'] = (group['ofw_act_cnt'].shift(5) - group['ofw_act_cnt']) / 5
    group['slope_ofw_act_cnt_14d'] = (group['ofw_act_cnt'].shift(14) - group['ofw_act_cnt']) / 14
    group['slope_vw_tgt_cnt_1d'] = (group['vw_tgt_cnt'].shift(1) - group['vw_tgt_cnt']) / 1
    group['slope_vw_tgt_cnt_5d'] = (group['vw_tgt_cnt'].shift(5) - group['vw_tgt_cnt']) / 5
    group['slope_vw_tgt_cnt_14d'] = (group['vw_tgt_cnt'].shift(14) - group['vw_tgt_cnt']) / 14
    group['slope_rgs_tgt_cnt_1d'] = (group['rgs_tgt_cnt'].shift(1) - group['rgs_tgt_cnt']) / 1
    group['slope_rgs_tgt_cnt_5d'] = (group['rgs_tgt_cnt'].shift(5) - group['rgs_tgt_cnt']) / 5
    group['slope_rgs_tgt_cnt_14d'] = (group['rgs_tgt_cnt'].shift(14) - group['rgs_tgt_cnt']) / 14
    group['slope_trd_cst_1d'] = (group['trd_cst'].shift(1) - group['trd_cst']) / 1
    group['slope_trd_cst_5d'] = (group['trd_cst'].shift(5) - group['trd_cst']) / 5
    group['slope_trd_cst_14d'] = (group['trd_cst'].shift(14) - group['trd_cst']) / 14
    
    # 미래 변화량 계산(데이터 내에서 각 시점을 기준으로 시점 이후 데이터 사용)
    group['future_slope_tco_avg_eal_pls_1d'] = (group['tco_avg_eal_pls'].shift(-1) - group['tco_avg_eal_pls']) / 1
    group['future_slope_tco_avg_eal_pls_5d'] = (group['tco_avg_eal_pls'].shift(-5) - group['tco_avg_eal_pls']) / 5
    group['future_slope_tco_avg_eal_pls_14d'] = (group['tco_avg_eal_pls'].shift(-14) - group['tco_avg_eal_pls']) / 14
    group['future_slope_tco_avg_pft_rt_1d'] = (group['tco_avg_pft_rt'].shift(-1) - group['tco_avg_pft_rt']) / 1
    group['future_slope_tco_avg_pft_rt_5d'] = (group['tco_avg_pft_rt'].shift(-5) - group['tco_avg_pft_rt']) / 5
    group['future_slope_tco_avg_pft_rt_14d'] = (group['tco_avg_pft_rt'].shift(-14) - group['tco_avg_pft_rt']) / 14
    group['future_slope_lss_ivo_rt_1d'] = (group['lss_ivo_rt'].shift(-1) - group['lss_ivo_rt']) / 1
    group['future_slope_lss_ivo_rt_5d'] = (group['lss_ivo_rt'].shift(-5) - group['lss_ivo_rt']) / 5
    group['future_slope_lss_ivo_rt_14d'] = (group['lss_ivo_rt'].shift(-14) - group['lss_ivo_rt']) / 14
    group['future_slope_ifw_act_cnt_1d'] = (group['ifw_act_cnt'].shift(-1) - group['ifw_act_cnt']) / 1
    group['future_slope_ifw_act_cnt_5d'] = (group['ifw_act_cnt'].shift(-5) - group['ifw_act_cnt']) / 5
    group['future_slope_ifw_act_cnt_14d'] = (group['ifw_act_cnt'].shift(-14) - group['ifw_act_cnt']) / 14
    group['future_slope_ofw_act_cnt_1d'] = (group['ofw_act_cnt'].shift(-1) - group['ofw_act_cnt']) / 1
    group['future_slope_ofw_act_cnt_5d'] = (group['ofw_act_cnt'].shift(-5) - group['ofw_act_cnt']) / 5
    group['future_slope_ofw_act_cnt_14d'] = (group['ofw_act_cnt'].shift(-14) - group['ofw_act_cnt']) / 14
    group['future_slope_vw_tgt_cnt_1d'] = (group['vw_tgt_cnt'].shift(-1) - group['vw_tgt_cnt']) / 1
    group['future_slope_vw_tgt_cnt_5d'] = (group['vw_tgt_cnt'].shift(-5) - group['vw_tgt_cnt']) / 5
    group['future_slope_vw_tgt_cnt_14d'] = (group['vw_tgt_cnt'].shift(-14) - group['vw_tgt_cnt']) / 14
    group['future_slope_rgs_tgt_cnt_1d'] = (group['rgs_tgt_cnt'].shift(-1) - group['rgs_tgt_cnt']) / 1
    group['future_slope_rgs_tgt_cnt_5d'] = (group['rgs_tgt_cnt'].shift(-5) - group['rgs_tgt_cnt']) / 5
    group['future_slope_rgs_tgt_cnt_14d'] = (group['rgs_tgt_cnt'].shift(-14) - group['rgs_tgt_cnt']) / 14
    group['future_slope_trd_cst_1d'] = (group['trd_cst'].shift(-1) - group['trd_cst']) / 1
    group['future_slope_trd_cst_5d'] = (group['trd_cst'].shift(-5) - group['trd_cst']) / 5
    group['future_slope_trd_cst_14d'] = (group['trd_cst'].shift(-14) - group['trd_cst']) / 14
    
    # NATR 계산 추가 (7일, 14일, 30일)
    group = calculate_natr_multiple(group)  # NATR 계산

    return group.reset_index()  # 인덱스 초기화 및 bse_dt 컬럼 유지

# 그룹화 적용
result = df.groupby('tck_iem_cd').apply(calculate_trends).reset_index(drop=True)

# 필요한 컬럼만 선택
final_result = result[['tck_iem_cd', 'bse_dt', 'mkt_pr_tot_amt', 'trd_cst',
    'slope_tco_avg_eal_pls_1d', 'slope_tco_avg_eal_pls_5d', 'slope_tco_avg_eal_pls_14d',
    'future_slope_tco_avg_eal_pls_1d', 'future_slope_tco_avg_eal_pls_5d', 'future_slope_tco_avg_eal_pls_14d',
    'slope_tco_avg_pft_rt_1d', 'slope_tco_avg_pft_rt_5d', 'slope_tco_avg_pft_rt_14d',
    'future_slope_tco_avg_pft_rt_1d', 'future_slope_tco_avg_pft_rt_5d', 'future_slope_tco_avg_pft_rt_14d',
    'slope_lss_ivo_rt_1d', 'slope_lss_ivo_rt_5d', 'slope_lss_ivo_rt_14d',
    'future_slope_lss_ivo_rt_1d', 'future_slope_lss_ivo_rt_5d', 'future_slope_lss_ivo_rt_14d',
    'slope_ifw_act_cnt_1d', 'slope_ifw_act_cnt_5d', 'slope_ifw_act_cnt_14d',
    'future_slope_ifw_act_cnt_1d', 'future_slope_ifw_act_cnt_5d', 'future_slope_ifw_act_cnt_14d',
    'slope_ofw_act_cnt_1d', 'slope_ofw_act_cnt_5d', 'slope_ofw_act_cnt_14d',
    'future_slope_ofw_act_cnt_1d', 'future_slope_ofw_act_cnt_5d', 'future_slope_ofw_act_cnt_14d',
    'slope_vw_tgt_cnt_1d', 'slope_vw_tgt_cnt_5d', 'slope_vw_tgt_cnt_14d',
    'future_slope_vw_tgt_cnt_1d', 'future_slope_vw_tgt_cnt_5d', 'future_slope_vw_tgt_cnt_14d',
    'slope_rgs_tgt_cnt_1d', 'slope_rgs_tgt_cnt_5d', 'slope_rgs_tgt_cnt_14d',
    'future_slope_rgs_tgt_cnt_1d', 'future_slope_rgs_tgt_cnt_5d', 'future_slope_rgs_tgt_cnt_14d',
    'slope_trd_cst_1d', 'slope_trd_cst_5d', 'slope_trd_cst_14d',
    'future_slope_trd_cst_1d', 'future_slope_trd_cst_5d', 'future_slope_trd_cst_14d', 'NATR_7', 'NATR_14', 'NATR_30'
]]

from sklearn.preprocessing import StandardScaler

# 로그 변환과 Standard Scaling 적용 함수 / 기존 MinMaxScaler에서 변경
def log_standardize(df, columns):
    normalized_df = df.copy()
    standard_scaler = StandardScaler()

    for column in columns:
        # NaN 값 제외
        filtered_column = normalized_df[column].dropna()

        # 음수와 양수를 모두 처리할 수 있는 로그 변환 적용 / 이 부분에서 문제가 됐었다. 코드를 확인해보니 음수값의 로그 변환이 적용되지 않았던 듯
        transformed_data = np.sign(filtered_column) * np.log1p(np.abs(filtered_column))
        
        # 무한대 값 제거 (Infinity가 발생한 경우) / 무한대 값을 nan값으로 바꾼 후 다시 nan값을 제거한다. 
        transformed_data = transformed_data.replace([np.inf, -np.inf], np.nan).dropna() 
        
        # Standard Scaling 수행
        if not transformed_data.empty:
            # reshape(-1, 1)에서 -1은 행의 수를 자동 계산, 1은 하나의 열로 배열하라는 의미이다. 이후 결과를 1차원으로 만들기 위해 flatten()을 이용한다. 
            standardized_data = standard_scaler.fit_transform(transformed_data.values.reshape(-1, 1)).flatten() 
            
            # 변환된 값을 데이터프레임에 적용
            transformed_column = pd.Series(data=standardized_data, index=transformed_data.index)
            normalized_df.loc[transformed_column.index, column] = transformed_column

    return normalized_df


# 정규화할 열 리스트
columns_to_normalize = ['mkt_pr_tot_amt', 'trd_cst',
    'slope_tco_avg_eal_pls_1d', 'slope_tco_avg_eal_pls_5d', 'slope_tco_avg_eal_pls_14d',
    'future_slope_tco_avg_eal_pls_1d', 'future_slope_tco_avg_eal_pls_5d', 'future_slope_tco_avg_eal_pls_14d',
    'slope_tco_avg_pft_rt_1d', 'slope_tco_avg_pft_rt_5d', 'slope_tco_avg_pft_rt_14d',
    'future_slope_tco_avg_pft_rt_1d', 'future_slope_tco_avg_pft_rt_5d', 'future_slope_tco_avg_pft_rt_14d',
    'slope_lss_ivo_rt_1d', 'slope_lss_ivo_rt_5d', 'slope_lss_ivo_rt_14d',
    'future_slope_lss_ivo_rt_1d', 'future_slope_lss_ivo_rt_5d', 'future_slope_lss_ivo_rt_14d',
    'slope_ifw_act_cnt_1d', 'slope_ifw_act_cnt_5d', 'slope_ifw_act_cnt_14d',
    'future_slope_ifw_act_cnt_1d', 'future_slope_ifw_act_cnt_5d', 'future_slope_ifw_act_cnt_14d',
    'slope_ofw_act_cnt_1d', 'slope_ofw_act_cnt_5d', 'slope_ofw_act_cnt_14d',
    'future_slope_ofw_act_cnt_1d', 'future_slope_ofw_act_cnt_5d', 'future_slope_ofw_act_cnt_14d',
    'slope_vw_tgt_cnt_1d', 'slope_vw_tgt_cnt_5d', 'slope_vw_tgt_cnt_14d',
    'future_slope_vw_tgt_cnt_1d', 'future_slope_vw_tgt_cnt_5d', 'future_slope_vw_tgt_cnt_14d',
    'slope_rgs_tgt_cnt_1d', 'slope_rgs_tgt_cnt_5d', 'slope_rgs_tgt_cnt_14d',
    'future_slope_rgs_tgt_cnt_1d', 'future_slope_rgs_tgt_cnt_5d', 'future_slope_rgs_tgt_cnt_14d',
    'slope_trd_cst_1d', 'slope_trd_cst_5d', 'slope_trd_cst_14d',
    'future_slope_trd_cst_1d', 'future_slope_trd_cst_5d', 'future_slope_trd_cst_14d', 'NATR_7', 'NATR_14', 'NATR_30'
]


# 로그 변환과 Standard Scaling 수행
final_result_normalized = log_standardize(final_result, columns_to_normalize)

# CSV 파일로 저장
output_file = 'slopes_about_log_standardize.csv'  # 정규화된 결과 파일 이름
final_result_normalized.to_csv(os.path.join(data_path,output_file), index=False)

# 데이터 불러오기
PATH = os.path.join(os.getcwd(),'data')
df = pd.read_csv(os.path.join(PATH,'slopes_about_log_standardize.csv'))

# 사용할 각 특성 그룹 정의 / 특성이 제대로 분류에 적용되는지 확인하기  
risk_features = ['mkt_pr_tot_amt', 'trd_cst']
bandwagon_features = [
    'slope_tco_avg_eal_pls_1d', 'slope_tco_avg_eal_pls_5d', 'slope_tco_avg_eal_pls_14d',
    'future_slope_tco_avg_eal_pls_1d', 'future_slope_tco_avg_eal_pls_5d', 'future_slope_tco_avg_eal_pls_14d',
    'slope_tco_avg_pft_rt_1d', 'slope_tco_avg_pft_rt_5d', 'slope_tco_avg_pft_rt_14d',
    'future_slope_tco_avg_pft_rt_1d', 'future_slope_tco_avg_pft_rt_5d', 'future_slope_tco_avg_pft_rt_14d',
    'slope_lss_ivo_rt_1d', 'slope_lss_ivo_rt_5d', 'slope_lss_ivo_rt_14d',
    'future_slope_lss_ivo_rt_1d', 'future_slope_lss_ivo_rt_5d', 'future_slope_lss_ivo_rt_14d'
]
interest_features = [
    'slope_ifw_act_cnt_1d', 'slope_ifw_act_cnt_5d', 'slope_ifw_act_cnt_14d',
    'future_slope_ifw_act_cnt_1d', 'future_slope_ifw_act_cnt_5d', 'future_slope_ifw_act_cnt_14d',
    'slope_ofw_act_cnt_1d', 'slope_ofw_act_cnt_5d', 'slope_ofw_act_cnt_14d',
    'future_slope_ofw_act_cnt_1d', 'future_slope_ofw_act_cnt_5d', 'future_slope_ofw_act_cnt_14d',
    'slope_vw_tgt_cnt_1d', 'slope_vw_tgt_cnt_5d', 'slope_vw_tgt_cnt_14d',
    'future_slope_vw_tgt_cnt_1d', 'future_slope_vw_tgt_cnt_5d', 'future_slope_vw_tgt_cnt_14d',
    'slope_rgs_tgt_cnt_1d', 'slope_rgs_tgt_cnt_5d', 'slope_rgs_tgt_cnt_14d',
    'future_slope_rgs_tgt_cnt_1d', 'future_slope_rgs_tgt_cnt_5d', 'future_slope_rgs_tgt_cnt_14d',
    'trd_cst'
]
volatility_features = ['NATR_7', 'NATR_14', 'NATR_30']

# 결측치 및 무한대 값 처리
for feature in risk_features + bandwagon_features + interest_features + volatility_features:
    df[feature].replace([np.inf, -np.inf], np.nan, inplace=True)

# 숫자형 열만 선택하여 결측치 채우기 
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# 각 특성 그룹에 대해 PCA를 적용하여 1차원으로 축소 
def apply_pca(features, data, component_name):
    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(data[features])
    data[component_name] = pca_result.flatten()

apply_pca(risk_features, df, 'PCA_Risk')
apply_pca(bandwagon_features, df, 'PCA_Bandwagon')
apply_pca(interest_features, df, 'PCA_Interest')
apply_pca(volatility_features, df, 'PCA_Volatility')

Risk_median = df['PCA_Risk'].median()
Bandwagon_median = df['PCA_Bandwagon'].median()
Interest_median = df['PCA_Interest'].median()
Volatility_median = df['PCA_Volatility'].median()

# A와 B 타입 구분 및 성향 계산
df['Risk_Type'] = df['PCA_Risk'].apply(lambda x: 'A' if x >= Risk_median else 'B')  # 0을 기준으로 타입 분류
df['Risk_Strength'] = df['PCA_Risk'].abs()  # 절대값으로 강도 계산

# I와 G 타입 구분 및 성향 계산
df['Bandwagon_Type'] = df['PCA_Bandwagon'].apply(lambda x: 'G' if x >= Bandwagon_median else 'R')  # 0을 기준으로 타입 분류
df['Bandwagon_Strength'] = df['PCA_Bandwagon'].abs()  # 절대값으로 강도 계산

# U와 P 타입 구분 및 성향 계산
df['Interest_Type'] = df['PCA_Interest'].apply(lambda x: 'P' if x >= Interest_median else 'U')  # 0을 기준으로 타입 분류
df['Interest_Strength'] = df['PCA_Interest'].abs()  # 절대값으로 강도 계산

# Volatility Type 구분 및 강도 계산
df['Volatility_Type'] = df['PCA_Volatility'].apply(lambda x: 'X' if x >= Volatility_median else 'S')  # 0을 기준으로 타입 분류
df['Volatility_Strength'] = df['PCA_Volatility'].abs()  # 절대값으로 강도 계산

score_columns = []

PCA_li = ['PCA_Risk', 'PCA_Bandwagon', 'PCA_Interest', 'PCA_Volatility']
median_li = [Risk_median,Bandwagon_median,Interest_median,Volatility_median]
# 음수, 양수 각 각 스케일링
for component, median_val in zip(PCA_li, median_li):
    # 음수 값들 (0 미만)
    negative_values = df[df[component] < median_val][component].values.reshape(-1, 1)
    neg_scaler = MinMaxScaler(feature_range=(0, 49.99999))
    negative_scaled = neg_scaler.fit_transform(negative_values)

    # 양수 값들 (0 이상)
    positive_values = df[df[component] >= median_val][component].values.reshape(-1, 1)
    pos_scaler = MinMaxScaler(feature_range=(50, 100))
    positive_scaled = pos_scaler.fit_transform(positive_values)

    # 음수와 양수 데이터를 합침 (인덱스를 그대로 유지)
    negative_scaled_df = pd.DataFrame(negative_scaled, index=df[df[component] < median_val][component].index)
    positive_scaled_df = pd.DataFrame(positive_scaled, index=df[df[component] >= median_val][component].index)

    # 합치기 (기존 df와 index 동일하게)
    full_scaled = pd.concat([negative_scaled_df, positive_scaled_df]).sort_index()

    # 결과를 새로운 리스트에 추가
    score_columns.append(full_scaled)

# Step 3: 최종 결과로 각 열을 합침
df_scored = pd.concat(score_columns, axis=1)

# 열 이름에 '_Score' 추가
df_scored.columns = [col + '_Score' for col in ['PCA_Risk', 'PCA_Bandwagon', 'PCA_Interest', 'PCA_Volatility']]

# 인덱스를 원본 데이터프레임의 인덱스로 설정
df_scored.index = df.index

df_final = pd.concat([df,df_scored], axis=1)

# NHTI 유형 결정 함수
def determine_nhti_type(row):
    risk_type = 'A' if row['PCA_Risk_Score'] >= 50 else 'B'
    bandwagon_type = 'G' if row['PCA_Bandwagon_Score'] >= 50 else 'R'
    interest_type = 'P' if row['PCA_Interest_Score'] >= 50 else 'U'
    volatility_type = 'X' if row['PCA_Volatility_Score'] >= 50 else 'S'
    return f"{risk_type}{bandwagon_type}{interest_type}{volatility_type}"

# NHTI 유형 결정 및 타입 강도 스케일링 결과 추가
df_final['NHTI_Type'] = df_final.apply(determine_nhti_type, axis=1)

# 필요한 열들만 선택하여 새로운 데이터프레임 생성
df_final = df_final[['tck_iem_cd','bse_dt','NHTI_Type', 'PCA_Risk_Score', 'PCA_Bandwagon_Score', 'PCA_Interest_Score', 'PCA_Volatility_Score']]
df_final.columns = ['tck_iem_cd','bse_dt','stk_NHTI', 'stk_Risk_Score', 'stk_Bandwagon_Score', 'stk_Interest_Score', 'stk_Volatility_Score']

etf_holdings = pd.read_csv(os.path.join(PATH, 'NH_CONTEST_DATA_ETF_HOLDINGS.csv'), encoding='cp949')
stk_iem_ifo = pd.read_csv(os.path.join(PATH, 'NH_CONTEST_NW_FC_STK_IEM_IFO.csv'), encoding='cp949')

new_etf = pd.merge(etf_holdings,stk_iem_ifo[['tck_iem_cd','stk_etf_dit_cd']],on='tck_iem_cd',how='left')

new_etf.loc[(new_etf['stk_etf_dit_cd']=='주식') & (new_etf['sec_tp']!="ST"),"sec_tp"]='ST'
new_etf.loc[(new_etf['stk_etf_dit_cd']=='ETF') & (new_etf['sec_tp']!="EF"),"sec_tp"]='EF'

new_etf.drop('stk_etf_dit_cd',axis=1,inplace=True)

# 구성종목에 주식이 존재하는 etf
filtered_etf = new_etf[new_etf['sec_tp'] == 'ST']

df_recently = df_final.drop_duplicates(subset='tck_iem_cd', keep='last')

df_final.drop_duplicates(subset='tck_iem_cd', keep='last')['bse_dt'].value_counts()

df_nhti = pd.merge(filtered_etf,df_recently,on='tck_iem_cd', how='left')
df_nhti.dropna(axis=0,how='any',subset='stk_NHTI', inplace = True)
df_nhti['wht_pct_sum'] = df_nhti.groupby(['etf_tck_cd','sec_tp'])['wht_pct'].transform('sum')

df_nhti['pct_to_Risk_Score'] = (df_nhti['wht_pct']/df_nhti['wht_pct_sum'])*df_nhti['stk_Risk_Score']
df_nhti['pct_to_Bandwagon_Score'] = (df_nhti['wht_pct']/df_nhti['wht_pct_sum'])*df_nhti['stk_Bandwagon_Score']
df_nhti['pct_to_Interest_Score'] = (df_nhti['wht_pct']/df_nhti['wht_pct_sum'])*df_nhti['stk_Interest_Score']
df_nhti['pct_to_Volatility_Score'] = (df_nhti['wht_pct']/df_nhti['wht_pct_sum'])*df_nhti['stk_Volatility_Score']

df_nhti['Risk_Score'] = df_nhti.groupby(['etf_tck_cd'])['pct_to_Risk_Score'].transform('sum')
df_nhti['Bandwagon_Score'] = df_nhti.groupby(['etf_tck_cd'])['pct_to_Bandwagon_Score'].transform('sum')
df_nhti['Interest_Score'] = df_nhti.groupby(['etf_tck_cd'])['pct_to_Interest_Score'].transform('sum')
df_nhti['Volatility_Score'] = df_nhti.groupby(['etf_tck_cd'])['pct_to_Volatility_Score'].transform('sum')

# NHTI 유형 결정 함수
def determine_nhti_type(row):
    risk_type = 'A' if row['Risk_Score'] >= 50 else 'B'
    bandwagon = 'G' if row['Bandwagon_Score'] >= 50 else 'R'
    interest_type = 'P' if row['Interest_Score'] >= 50 else 'U'
    volatility_type = 'X' if row['Volatility_Score'] >= 50 else 'S'
    return f"{risk_type}{bandwagon}{interest_type}{volatility_type}"

# NHTI 유형 결정 및 타입 강도 스케일링 결과 추가
df_nhti['NHTI_Type'] = df_nhti.apply(determine_nhti_type, axis=1)
df_nhti = df_nhti[df_nhti['mkt_vlu']!=0]

NHTI_etf = df_nhti
NHTI_stk = df_final

need_cols = ['etf_tck_cd','tck_iem_cd','fc_sec_eng_nm','fc_sec_krl_nm',
             'wht_pct','wht_pct_sum','Risk_Score','Bandwagon_Score','Interest_Score',
             'Volatility_Score','NHTI_Type']
NHTI_etf = NHTI_etf[need_cols]
NHTI_etf.to_csv(os.path.join(PATH,'NHTI_ETF.csv'),index=False)
NHTI_stk.to_csv(os.path.join(PATH,'NHTI_STK.csv'), index=False)

NHTI_etf = pd.read_csv(os.path.join(PATH,'NHTI_ETF.csv'))

etf_holdings = pd.read_csv(os.path.join(PATH, 'NH_CONTEST_DATA_ETF_HOLDINGS.csv'), encoding='cp949')
stk_iem_ifo = pd.read_csv(os.path.join(PATH, 'NH_CONTEST_NW_FC_STK_IEM_IFO.csv'), encoding='cp949')

stk_iem_ifo['fc_sec_eng_nm'] = [i.strip() for i in stk_iem_ifo['fc_sec_eng_nm'].to_list()]
stk_iem_ifo = stk_iem_ifo.replace('-', np.nan)

etf_info = pd.merge(etf_holdings,stk_iem_ifo,on='tck_iem_cd', how='left')
dividend = pd.read_csv(os.path.join(PATH,'NH_CONTEST_DATA_HISTORICAL_DIVIDEND.csv'))
recently_dividend = dividend.sort_values(['etf_tck_cd','ediv_dt']).drop_duplicates(subset='etf_tck_cd', keep='last').reset_index(drop=True)
etf_info = pd.merge(etf_info, recently_dividend,on='etf_tck_cd', how='left')

etf_info.rename(columns={
    'etf_tck_cd': 'ETF_Ticker_Code',
    'tck_iem_cd': 'Stock_Item_Code',
    'mkt_vlu': 'Market_Value',
    'fc_sec_eng_nm_x': 'Sector_Name_English_X',
    'fc_sec_krl_nm_x': 'Sector_Name_Korean_X',
    'stk_qty': 'Stock_Quantity',
    'wht_pct': 'Weight_Percentage',
    'sec_tp': 'Section_Type',
    'fc_sec_krl_nm_y': 'Sector_Name_Korean_Y',
    'fc_sec_eng_nm_y': 'Sector_Name_English_Y',
    'stk_etf_dit_cd': 'Stock_ETF_Distinction_Code',
    'ltg_tot_stk_qty': 'Total_Listed_Stock_Quantity',
    'fc_mkt_dit_cd': 'Market_Distinction_Code',
    'co_adr': 'Company_Address',
    'web_adr': 'Website_Address',
    'btp_cfc_nm': 'Business_Name',
    'ceo_nm': 'CEO_Name',
    'eng_utk_otl_cts': 'Key_Overview_English',
    'ser_cfc_nm': 'Service_Name',
    'ids_nm': 'Industry_Name',
    'mkt_pr_tot_amt': 'Total_Market_Price_Amount',
    'ediv_dt': 'Ex_Dividend_Date',
    'ddn_amt': 'Dividend_Amount',
    'aed_stkp_ddn_amt': 'Dividend_Amount_Per_Stock',
    'ddn_bse_dt': 'Dividend_Base_Date',
    'ddn_pym_dt': 'Dividend_Payment_Date',
    'pba_dt': 'Public_Announcement_Date',
    'ddn_pym_fcy_cd': 'Dividend_Payment_Currency_Code'
}, inplace=True)

etf_info['Section_Type'] = etf_info['Section_Type'].replace('-', 'no info')
etf_info['Stock_Item_Code'] = etf_info['Stock_Item_Code'].replace('-', 'no info')
etf_info['Sector_Name_Korean_X'] = etf_info['Sector_Name_Korean_X'].replace('-', 'no info')
etf_info.fillna('no info', inplace=True)
etf_info.to_csv(os.path.join(PATH,'ETF_INFO.csv'),index=False, encoding='cp949')