import pickle
import matplotlib.pyplot as plt

# 피클 파일에서 데이터를 불러오는 함수
def ga_load_data_from_pickle(file_path):  # 세대별 fitness
    l=[]
    a=file_path.split('.')
    for i in range(1,11):
        path=str(f'{a[0]}{i}.{a[1]}')
        with open(path, 'rb') as file:
            data = pickle.load(file)
        l=l+data
    return l


def load_data_from_pickle(file_path): # 휴리스틱 fitness
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def plot_data(data, hu):
    plt.figure(figsize=(10, 5))
    plt.plot(data)
    plt.title('20 Customer')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)


    plt.axhline(y=hu[0], color='r', linestyle='--', label=f'Huristic1')

    plt.axhline(y=hu[1], color='g', linestyle='--', label=f'Huristic2')
    plt.axhline(y=hu[2], color='b', linestyle='--', label=f'Huristic3')
    plt.legend()
    plt.savefig('20개,10000세대/result.png')
    plt.show()


# 피클 파일 경로 지정
file_path = '20개,10000세대\ga_result.pickle'  # 파일 경로를 설정하세요.
#
# # 데이터 불러오기 및 그래프 그리기
data = ga_load_data_from_pickle(file_path)
hu=load_data_from_pickle("20개,10000세대\huristic_result.pickle")
print(hu)
plot_data(data,hu)
