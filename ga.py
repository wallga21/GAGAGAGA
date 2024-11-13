"""유전 알고리즘 구현"""

# @title
import random
import itertools
import pickle


class DeliveryOptimizer:
    def __init__(self, num_orders, profits, travel_times, max_time):
        """
        초기화 메서드.

        매개변수:
        - num_orders: 주문 수 (K)
        - profits: 리스트, 0번은 출발점, 1번~K번은 상점, K+1번~2K번은 고객, 2K+1번은 도착점
        - travel_times: 2차원 리스트 [[travel_time]]
        - max_time: 최대 운행 시간 (T)
        """
        self.num_orders = num_orders
        self.profits = profits
        self.travel_times = travel_times  # 2차원 리스트
        self.max_time = max_time
        self.order=0

        # 모든 노드의 리스트를 명시적으로 인덱스화하지 않고 자연스럽게 리스트 인덱스 사용
        self.nodes = list(range(2 * num_orders + 2))

    def initialize_population(self, pop_size):
        population = []
        for _ in range(pop_size):
            chromosome = self.create_chromosome()
            population.append(chromosome)
        return population

    def create_chromosome(self):
        """
        주문 ID를 사용하여 랜덤한 염색체를 생성합니다.
        각 주문 ID i는 정확히 두 번 나타납니다.
        """
        chromosome = []
        selected_orders = random.sample(list(range(1, self.num_orders + 1)), random.randint(1, self.num_orders))
        for order_id in selected_orders:
            chromosome.extend([order_id, order_id])  # 주문 ID를 두 번 추가
        random.shuffle(chromosome)
        return chromosome

    def calculate_fitness(self, chromosome):
        """
        염색체의 적합도와 총 이동 시간을 계산합니다.
        반환값: (fitness, total_travel_time)
        """
        included_orders = set(chromosome)
        # 포함된 주문에 대한 수익 계산 (상점의 수익을 계산)
        total_profit = sum(self.profits[order_id] for order_id in included_orders)

        # 노드 시퀀스 생성
        node_sequence = []

        #유전자에서 등장했는지 확인하는 리스트
        visited = [False]*(self.num_orders+1)

        for gene in chromosome:
            if not visited[gene]:
                # 첫 번째 발생: 상점 노드
                visited[gene] = True
                node_sequence.append(gene)
            else:
                # 두 번째 발생: 고객 노드
                node_sequence.append(gene + self.num_orders)

        # 시작 및 종료 노드 추가
        route_nodes = [0] + node_sequence + [2 * self.num_orders + 1]

        # 총 이동 시간 계산
        total_travel_time = 0
        for i in range(len(route_nodes) - 1):
            from_idx = route_nodes[i]
            to_idx = route_nodes[i + 1]
            total_travel_time += self.travel_times[from_idx][to_idx]

        # 총 이동 시간이 최대 운행 시간을 초과하면 적합도는 -무한대
        if total_travel_time > self.max_time:
            return -float('inf')

        return 10 *total_profit - total_travel_time # P1은 10, P2는 1로 가정

    def tournament_selection(self, population, fitnesses, tournament_size=3):
        selected = random.sample(list(zip(population, fitnesses)), tournament_size)
        selected.sort(reverse=True)
        return selected[0][0]

    def crossover(self, parent1, parent2):
        # 일점 교차
        if len(parent1) < 2 or len(parent2) < 2:
            return parent1[:], parent2[:]
        point1 = random.randint(1, len(parent1) - 1)
        point2 = random.randint(1, len(parent2) - 1)

        offspring1 = parent1[:point1] + parent2[point2:]
        offspring2 = parent2[:point2] + parent1[point1:]

        # 자손 염색체 복구
        offspring1 = self.repair_chromosome(offspring1)
        offspring2 = self.repair_chromosome(offspring2)

        return offspring1, offspring2

    def repair_chromosome(self, chromosome): #todo: 누락된 유전자 추가 부분이 비효율적
        # 각 유전자가 정확히 두 번 나타나도록 조정
        gene_idxs = [[] for _ in range(self.num_orders+1)]
        for idx, gene in enumerate(chromosome):
            gene_idxs[gene].append(idx)

        # 유전자 조정
        repaired_chromosome = []
        for gene in range(1, self.num_orders+1):
            if len(gene_idxs[gene]) == 4:
              sample = random.sample(gene_idxs[gene], 2)
              for i in sample:
                chromosome[i] = None

            elif len(gene_idxs[gene]) == 3:
              sample = random.sample(gene_idxs[gene], 1)
              for i in sample:
                chromosome[i] = None

            elif len(gene_idxs[gene]) == 1:
              chromosome[gene_idxs[gene][0]] = None

        for k in chromosome:
          if k:
            repaired_chromosome.append(k)

        return repaired_chromosome


    def add_mutation(self, chromosome, mutation_rate=0.3):
        if random.random() < mutation_rate:
          # 추가 돌연변이
          current_orders = set(chromosome)
          available_orders = set(range(1, self.num_orders + 1)) - current_orders
          if available_orders:
              order_to_add = random.choice(list(available_orders))
              # 주문 ID를 두 번 추가
              idx1 = random.randint(0, len(chromosome))
              chromosome.insert(idx1, order_to_add)
              idx2 = random.randint(0, len(chromosome))
              chromosome.insert(idx2, order_to_add)
        return chromosome

    def del_mutation(self, chromosome, mutation_rate=0.1):
        if random.random() < mutation_rate:
          # 제거 돌연변이
          current_orders = set(chromosome)
          if current_orders:
              order_to_remove = random.choice(list(current_orders))
              chromosome = [gene for gene in chromosome if gene != order_to_remove]
        return chromosome

    def three_opt(self, chromosome):
        # 3-opt는 염색체의 순서를 변경하여 최적화를 시도합니다.
        best = chromosome[:]
        best_fitness = self.calculate_fitness(best)
        improved = True
        while improved:
            improved = False
            for (i, j, k) in itertools.combinations(range(len(chromosome)), 3):
                new_chromosome = best[:]
                # 세 부분을 교환
                new_chromosome[i], new_chromosome[j], new_chromosome[k] = new_chromosome[k], new_chromosome[i], \
                new_chromosome[j]
                # 첫 번째 발생이 앞에 오도록 순서 조정
                new_chromosome = self.repair_chromosome(new_chromosome)
                new_fitness = self.calculate_fitness(new_chromosome)
                if new_fitness > best_fitness:
                    best = new_chromosome[:]
                    best_fitness = new_fitness
                    improved = True
                    break  # 첫 번째 개선 후 종료
            if improved:
                continue
        return best

    def run(self, pop_size=50, generations=100, add_mutation_prob = 0.3, del_mutation_prob = 0.1):
        population = self.initialize_population(pop_size)
        fitnesses = [self.calculate_fitness(chromosome) for chromosome in population]
        best_chromosome = None
        best_fitness = -float('inf')

        result = [0]*(1000)

        with open("huristic_result.pickle", "rb") as f:
            hu_results = pickle.load(f)
        compare = [0]*3

        for generation in range(generations):
            # 안정 상태 GA: 두 개의 자손 생성 후 최악의 개체 대체
            parent1 = self.tournament_selection(population, fitnesses)
            parent2 = self.tournament_selection(population, fitnesses)

            offspring1, offspring2 = self.crossover(parent1, parent2)

            offspring1 = self.add_mutation(offspring1, add_mutation_prob)
            offspring1 = self.del_mutation(offspring1, del_mutation_prob)

            offspring2 = self.add_mutation(offspring2)
            offspring2 = self.del_mutation(offspring2)

            # 지역 최적화
            offspring1 = self.three_opt(offspring1)
            offspring2 = self.three_opt(offspring2)

            # 적합도 평가
            fitness1 = self.calculate_fitness(offspring1)
            fitness2 = self.calculate_fitness(offspring2)

            # 최상의 염색체 업데이트
            if fitness1 > best_fitness:
                best_fitness = fitness1
                best_chromosome = offspring1[:]
            if fitness2 > best_fitness:
                best_fitness = fitness2
                best_chromosome = offspring2[:]

            # 최악의 개체 대체
            worst_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:2]
            population[worst_indices[0]] = offspring1
            population[worst_indices[1]] = offspring2
            fitnesses[worst_indices[0]] = fitness1
            fitnesses[worst_indices[1]] = fitness2

            for i in range(3):
                if compare[i]==0 and best_fitness >= hu_results[i]:
                    print(f"huristic{i+1}: {hu_results[i]}, ga {generation}th gen: {best_fitness}")
                    compare[i] = generation


            # 진행 상황 출력 (선택 사항)
            print(f"세대 {generation + 1}, 최고 수익: {best_fitness}")
            result[(generation)%1000] = best_fitness
            if (generation+1)%1000==0:
                with open(f"ga_result{(generation+1)//1000}.pickle", "wb") as f:
                    pickle.dump(result, f)
                result = [0] * (1000)

        with open("ga_win_gen.txt", "w") as f:
            for gen in compare:
                f.write(str(gen))


        return best_chromosome, best_fitness

