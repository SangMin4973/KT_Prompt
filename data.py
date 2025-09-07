import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# result 폴더 경로 지정
base_path = "result/template1"

# CSV 파일들을 담을 리스트
csv_list = []

# result 폴더 내부 순회
for root, dirs, files in os.walk(f"{base_path}/분포"):
    for file in files:
        if file.endswith(".csv"):  # CSV 파일만 선택
            file_path = os.path.join(root, file)
            print(f"불러오는 중: {file_path}")
            df = pd.read_csv(file_path)
            csv_list.append(df)

# 모든 CSV 파일 합치기
merged_df = pd.concat(csv_list, ignore_index=True)

# 합쳐진 CSV 저장
merged_df.to_csv(f"{base_path}/merged_result.csv", index=False, encoding="utf-8-sig")

print("모든 CSV 파일이 merged_result.csv 로 저장되었습니다.")

df_유형 = df[['user_prompt','유형','유형_예측']]
df_극성 = df[['user_prompt','극성', '극성_예측']]
df_시제 = df[['user_prompt','시제', '시제_예측']]
df_확실성 = df[['user_prompt','확실성', '확실성_예측']]

df_유형 = df_유형[df_유형["유형"] != df_유형["유형_예측"]]
df_극성 = df_극성[df_극성["극성"] != df_극성["극성_예측"]]
df_시제 = df_시제[df_시제["시제"] != df_시제["시제_예측"]]
df_확실성 = df_확실성[df_확실성["확실성"] != df_확실성["확실성_예측"]]

df_유형.to_csv(f"{base_path}/df_유형.csv", index=False, encoding="utf-8-sig")
df_극성.to_csv(f"{base_path}/df_극성.csv", index=False, encoding="utf-8-sig")
df_시제.to_csv(f"{base_path}/df_시제.csv", index=False, encoding="utf-8-sig")
df_확실성.to_csv(f"{base_path}/df_확실성.csv", index=False, encoding="utf-8-sig")

# 분석할 컬럼 쌍
pairs = [
    ("유형", "유형_예측"),
    ("극성", "극성_예측"),
    ("시제", "시제_예측"),
    ("확실성", "확실성_예측")
]

with open(f"{base_path}/analysis_result.txt", "w", encoding="utf-8") as f:
    for true_col, pred_col in pairs:
        f.write(f"\n=== {true_col} 오답/정답 분포 ===\n")
        
        # (정답, 예측) 조합별 개수
        pattern = (
            df.groupby([true_col, pred_col])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        
        # 전체 데이터 기준 비율
        pattern["비율(%)"] = pattern["count"] / len(df) * 100
        
        # DataFrame을 문자열로 변환해서 저장
        f.write(pattern.to_string(index=False))
        f.write("\n" + "-"*50 + "\n")