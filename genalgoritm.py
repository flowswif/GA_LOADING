import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Параметры предметов (добавлены floor_only и stackable)
ITEMS = [
    {"id": 1, "weight": 2, "value": 10, "dimensions": (1, 1, 1), "color": "red", "floor_only": True, "stackable": False, "priority": 2},
    {"id": 2, "weight": 3, "value": 15, "dimensions": (1, 1, 1), "color": "green", "floor_only": False, "stackable": True, "priority": 1},
    {"id": 3, "weight": 4, "value": 20, "dimensions": (1, 1, 1), "color": "blue", "floor_only": False, "stackable": True, "priority": 1},
    {"id": 4, "weight": 2, "value": 12, "dimensions": (1, 1, 1), "color": "yellow", "floor_only": True, "stackable": False, "priority": 2},
    {"id": 5, "weight": 3, "value": 18, "dimensions": (1, 1, 1), "color": "purple", "floor_only": False, "stackable": False, "priority": 3},
    {"id": 6, "weight": 25, "value": 8, "dimensions": (1, 1, 1), "color": "orange", "floor_only": False, "stackable": True, "priority": 1},
    {"id": 7, "weight": 5, "value": 25, "dimensions": (1, 1, 1), "color": "cyan", "floor_only": False, "stackable": True, "priority": 1},
    {"id": 8, "weight": 1, "value": 30, "dimensions": (1, 1, 1), "color": "magenta", "floor_only": True, "stackable": False, "priority": 2}
]

# Габариты и ограничения
KNAPSACK_DIMENSIONS = (10, 10, 10)
MAX_WEIGHT = 25  
POPULATION_SIZE = 100
GENERATIONS = 200
MUTATION_RATE = 0.1

# Функция для автоматического назначения приоритета
def calculate_priority(item):
    if item["stackable"] and item["floor_only"]:
        return 1
    elif item["stackable"] and not item["floor_only"]:
        return 2
    elif not item["stackable"] and item["floor_only"]:
        return 3
    else:
        return 4

def create_individual():
    return [1 for _ in range(len(ITEMS))]  # Все предметы включены

def check_placement(individual):
    total_weight = sum(ITEMS[i]["weight"] for i, gene in enumerate(individual) if gene == 1)
    if total_weight > MAX_WEIGHT:
        return None

    placed_items = []
    occupied = np.zeros(KNAPSACK_DIMENSIONS, dtype=bool)
    
    # Сначала размещаем предметы, которые можно ставить только на пол
    floor_items = [i for i in range(len(ITEMS)) 
                  if individual[i] == 1 and ITEMS[i]["floor_only"]]
    other_items = [i for i in range(len(ITEMS)) 
                  if individual[i] == 1 and not ITEMS[i]["floor_only"]]
    
    # Размещаем floor_only предметы
    for i in sorted(floor_items, key=lambda x: -np.prod(ITEMS[x]["dimensions"])):
        item = ITEMS[i]
        dims = item["dimensions"]
        placed = False
        
        # Пытаемся разместить на полу (z=0)
        z = 0
        for x in range(KNAPSACK_DIMENSIONS[0] - dims[0] + 1):
            for y in range(KNAPSACK_DIMENSIONS[1] - dims[1] + 1):
                if not occupied[x:x+dims[0], y:y+dims[1], z:z+dims[2]].any():
                    occupied[x:x+dims[0], y:y+dims[1], z:z+dims[2]] = True
                    placed_items.append({
                        "item": i,
                        "position": (x, y, z),
                        "dimensions": dims,
                        "color": item["color"]
                    })
                    placed = True
                    break
            if placed: break
        if not placed:
            return None
    
    # Размещаем остальные предметы
    for i in sorted(other_items, key=lambda x: -np.prod(ITEMS[x]["dimensions"])):
        item = ITEMS[i]
        dims = item["dimensions"]
        placed = False
        
        # Для stackable ищем место выше других предметов
        max_z = KNAPSACK_DIMENSIONS[2] - dims[2]
        for z in range(max_z, -1, -1) if item["stackable"] else [0]:
            for x in range(KNAPSACK_DIMENSIONS[0] - dims[0] + 1):
                for y in range(KNAPSACK_DIMENSIONS[1] - dims[1] + 1):
                    if not occupied[x:x+dims[0], y:y+dims[1], z:z+dims[2]].any():
                        # Проверяем поддержку снизу для stackable
                        if z > 0 and not occupied[x:x+dims[0], y:y+dims[1], z-1:z].any():
                            continue
                        
                        occupied[x:x+dims[0], y:y+dims[1], z:z+dims[2]] = True
                        placed_items.append({
                            "item": i,
                            "position": (x, y, z),
                            "dimensions": dims,
                            "color": item["color"]
                        })
                        placed = True
                        break
                if placed: break
            if placed: break
        if not placed:
            return None
    
    return placed_items

# В fitness-функции:
def fitness(individual):
    placement = check_placement(individual)
    if placement is None:
        return float('inf')
    
    # Суммируем приоритеты всех размещённых предметов
    total_priority = sum(ITEMS[item["item"]]["priority"] for item in placement)
    return total_priority

# Селекция (турнирная)
def selection(population, fitness_values):
    tournament = random.sample(list(zip(population, fitness_values)), 3)
    return min(tournament, key=lambda x: x[1])[0]

# Одноточечное скрещивание
def crossover(parent1, parent2):
    point = random.randint(1, len(ITEMS)-1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Мутация (инвертирование бита с учетом ограничений)
def mutate(individual):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = 1  # Всегда 1, так как нужно разместить все
    return individual

# Визуализация
def visualize_placement(placement):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    rx, ry, rz = KNAPSACK_DIMENSIONS
    ax.set_xlim([0, rx])
    ax.set_ylim([0, ry])
    ax.set_zlim([0, rz])
    ax.set_xlabel('Длина')
    ax.set_ylabel('Ширина')
    ax.set_zlabel('Высота')
    ax.set_title('Оптимальное размещение всех грузов')
    
    for item in placement:
        x, y, z = item["position"]
        dx, dy, dz = item["dimensions"]
        
        vertices = [
            [x, y, z],
            [x+dx, y, z],
            [x+dx, y+dy, z],
            [x, y+dy, z],
            [x, y, z+dz],
            [x+dx, y, z+dz],
            [x+dx, y+dy, z+dz],
            [x, y+dy, z+dz]
        ]
        
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[3], vertices[7], vertices[4]]
        ]
        
        ax.add_collection3d(Poly3DCollection(
            faces, 
            facecolors=item["color"], 
            linewidths=1, 
            edgecolors='k', 
            alpha=.7
        ))
        
        # Подписываем предметы
        ax.text(x + dx/2, y + dy/2, z + dz/2, 
                f"{ITEMS[item['item']]['id']}", 
                color='black',
                ha='center',
                va='center')
    
    plt.tight_layout()
    plt.show()

def genetic_algorithm():
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    
    for generation in range(GENERATIONS):
        fitness_values = [fitness(ind) for ind in population]
        best_idx = np.argmin(fitness_values)
        best_fitness = fitness_values[best_idx]
        
        print(f"Поколение {generation}: Лучший приоритет = {best_fitness}")
        
        new_population = [population[best_idx]]
        while len(new_population) < POPULATION_SIZE:
            parent1 = selection(population, fitness_values)
            parent2 = selection(population, fitness_values)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])
        
        population = new_population
    
    best_individual = min(population, key=fitness)
    placement = check_placement(best_individual)
    
    if placement:
        print("\nОптимальное размещение найдено!")
        print("Приоритеты предметов:")
        for item in placement:
            print(f"Предмет {ITEMS[item['item']]['id']}: Приоритет = {ITEMS[item['item']]['priority']}")
        visualize_placement(placement)
    else:
        print("Решение не найдено")

if __name__ == "__main__":
    genetic_algorithm()