from dotenv import load_dotenv
import os
from openai import OpenAI
import pandas as pd
import matplotlib.pyplot as plt
from template import *
from openai import RateLimitError

load_dotenv()
api_key = os.getenv("OPEN_API_KEY")
client = OpenAI(api_key=api_key)

data_list = os.listdir("data/")

# data_num = data_list[0]
# data_path = f"data/{data_num}"
# data = pd.read_csv(data_path).reset_index(drop=False)
# data_name = data_num.strip('.csv')
# file_path = f"result/{data_name}"
# os.makedirs(file_path, exist_ok=True)

template = TEMPLATE5


def chatbot(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.4,
            messages=[
                {"role": "user", "content" : f"{template}\n\n 입력문:{prompt}"}
            ])

        text = response.choices[0].message.content
        return text
    except RateLimitError as e:
        print(f"⚠️ RateLimitError 발생 → 문장 스킵: {prompt[:30]}...")
        return None  # 또는 "에러 발생" 같은 기본 값 반환

def run(data, sample_size=100):

    if len(data) > sample_size:
        data_sample = data.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        data_sample = data.copy().reset_index(drop=True)

    sample_input = data_sample['user_prompt']
    result = []

    for i in range(len(sample_input)):
        output = chatbot(sample_input[i])
        result.append(output)

    df_result = pd.DataFrame(result, columns=["result"])
    df_result_split = df_result["result"].str.split(",", expand=True)
    df_result_split.columns = ["유형_예측", "극성_예측", "시제_예측", "확실성_예측"]
    df_result = pd.concat([data_sample, df_result_split], axis=1)
    df_result = df_result[["index", "user_prompt", "유형", "유형_예측", "극성", "극성_예측", "시제", "시제_예측", "확실성", "확실성_예측"]]

    df_result.to_csv(f"{file_path}/result_{data_num}", index=False, encoding="utf-8-sig")

    print(f"{data} done!")
    sum = 0

    for i in range(len(df_result)):
        if df_result['유형_예측'][i] == df_result['유형'][i]:
            sum += 1
        if df_result['극성_예측'][i] == df_result['극성'][i]:
            sum += 1
        if df_result['시제_예측'][i] == df_result['시제'][i]:
            sum += 1
        if df_result['확실성_예측'][i] == df_result['확실성'][i]:
            sum += 1
    
    length = len(result) * 4
    print(f"정확도: {sum/length*100}%")
    return df_result

plt.rc("font", family="Malgun Gothic")

def draw_graph(data, file_name):

    data_name = file_name.strip('.csv')
    data = data[["유형_예측", "극성_예측", "시제_예측", "확실성_예측"]]
    for col in data.columns:
        counts = data[col].value_counts()
        plt.figure(figsize=(5,3))
        counts.plot(kind='bar')
        plt.title(f'{data_name}_{col}의 분포')
        plt.xlabel(col)
        plt.ylabel('개수')
        plt.xticks(rotation=0)
        plt.savefig(f"{file_path}/{data_name}_{col}_graph.png")
        plt.close()

if __name__ == "__main__":

    # for i in range(len(data_list)):
    #     data_num = data_list[i]
    #     data_path = f"data/{data_num}"
    #     data = pd.read_csv(data_path).reset_index(drop=False)
    #     data_name = data_num.strip('.csv')
    #     file_path = f"result/{data_name}"
    #     os.makedirs(file_path, exist_ok=True)

    #     result = run(data)
    #     draw_graph(result, data_num)
    data_num = "대화형.csv"
    data = pd.read_csv(f"{data_num}").reset_index(drop=False)
    data_name = data_num.strip('.csv')
    file_path = f"result/{data_name}_수정5"
    os.makedirs(file_path, exist_ok=True)
    result = run(data)
    draw_graph(result, data_num)
    # question = "DPF는 성능보다는 환경 부품이고 장착해서 정기적인 클리닝과 관리를 하면 약 90% 이상의 미세먼지를 줄인다."
    # result = chatbot(question)
    # print(f"질문: {question}")
    # print(f"답변: {result}")
    