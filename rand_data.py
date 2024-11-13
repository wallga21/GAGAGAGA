import random


def generate_random_dataset(size, min_profit=50, max_profit=1000, min_distance=50, max_distance=400):
    """
    원하는 크기의 이동 시간 매트릭스를 랜덤으로 생성합니다.

    매개변수:
    - size: 상점(고객의 수)
    - max_distance: 랜덤 이동 시간의 최대 값 (1000과 0은 특별한 값으로 설정됨)

    반환값:
    - profits: 수익 리스트
    - travel_times: 대칭성을 가진 이동 시간 매트릭스
    """
    # 수익 리스트 초기화 (0번 인덱스는 0, 1~size 인덱스는 랜덤값 부여)
    profits = [0] * (1+size)
    for i in range(1, size+1):  # 상점에만 랜덤 수익 할당
        profits[i] = random.randint(min_profit, max_profit)

    # 이동 시간 매트릭스 초기화
    travel_times = [[float('inf') if i == j else None for j in range(2*size+2)] for i in range(2*size+2)]

    # 대칭성을 가지도록 랜덤 이동 시간 설정
    for i in range(2*size+2):
        for j in range(i + 1, 2*size+2):
            if travel_times[i][j] is not None: continue
            travel_times[i][j] = travel_times[j][i] = random.randint(min_distance, max_distance)

    # 마지막 행과 열의 값 중 출발점과 도착점 관련 float('inf') 처리
    for i in range(2*size+2):
      travel_times[2*size+1][i] = float('inf')
    for i in range(1, 2*size+2):
      travel_times[i][0] = float('inf')
    for i in range(size+1, 2*size+2):
      travel_times[0][i] = float('inf')
    for i in range(2*size+2):
      del travel_times[i][0]
    for i in range(size+1):
      travel_times[i][2*size] = float('inf')
    for i in range(size+1,2*size+2):
      travel_times[i][2*size] = 0
    for i in range(size):
      travel_times[i+size+1][i] = float('inf')
    del travel_times[1+size*2]

    return profits, travel_times
'''profits, travel_times = generate_random_dataset(5)

# 생성된 데이터셋 확인
print("profits =", profits)
print(*travel_times, sep='\n')'''