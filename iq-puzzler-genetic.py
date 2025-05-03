import numpy as np
import matplotlib.pyplot as plt
import random
import copy

BOARD_WIDTH = 11
BOARD_HEIGHT = 5
POPULATION_SIZE = 100
NUM_GENERATIONS = 100

MUTATION_RATE = 0.2
TOURNAMENT_SIZE = 5

# قطعات تعریف‌شده
PIECES = {
    0: np.array([[1, 1], [1, 1]]),
    1: np.array([[1, 1, 1], [0, 1, 0]]),
    2: np.array([[1, 1, 1, 1]]),
    3: np.array([[1, 0], [1, 1], [1, 0]]),
    4: np.array([[1, 1, 0], [0, 1, 1]]),
    5: np.array([[1, 1, 1], [1, 0, 0]]),
    6: np.array([[1, 1], [1, 0], [1, 0]]),
    7: np.array([[1, 1, 1], [0, 0, 1]]),
    8: np.array([[0, 1], [1, 1], [1, 0]]),
    9: np.array([[1, 1], [1, 1], [1, 0]]),
    10: np.array([[1, 1, 0], [0, 1, 1]]),
    11: np.array([[1, 0], [1, 1], [0, 1]]),
}

def rotate_piece(piece):
    return [np.rot90(piece, k) for k in range(4)]

def get_all_variants(piece):
    variants = []
    for rot in rotate_piece(piece):
        if not any(np.array_equal(rot, v) for v in variants):
            variants.append(rot)
        flip = np.fliplr(rot)
        if not any(np.array_equal(flip, v) for v in variants):
            variants.append(flip)
    return variants

ALL_VARIANTS = {k: get_all_variants(p) for k, p in PIECES.items()}

def place_piece(board, piece, x, y):
    px, py = piece.shape
    if x + px > board.shape[0] or y + py > board.shape[1]:
        return False
    sub = board[x:x+px, y:y+py]
    if np.any((sub != -1) & (piece == 1)):
        return False
    board[x:x+px, y:y+py][piece == 1] = 1
    return True

def evaluate(individual):
    board = np.full((BOARD_HEIGHT, BOARD_WIDTH), -1)
    filled = 0
    for i, (piece_id, variant, x, y) in enumerate(individual):
        piece = ALL_VARIANTS[piece_id][variant]
        temp_board = board.copy()
        if place_piece(temp_board, piece, x, y):
            mask = (piece == 1)
            temp_board[x:x+piece.shape[0], y:y+piece.shape[1]][mask] = piece_id
            board = temp_board
            filled += np.sum(piece)
    return filled


def create_individual():
    pieces = list(PIECES.keys())
    random.shuffle(pieces)
    individual = []
    for piece_id in pieces:
        variants = ALL_VARIANTS[piece_id]
        variant = random.randint(0, len(variants) - 1)
        piece = variants[variant]
        x = random.randint(0, BOARD_HEIGHT - piece.shape[0])
        y = random.randint(0, BOARD_WIDTH - piece.shape[1])
        individual.append((piece_id, variant, x, y))
    return individual

def mutate(individual):
    new_ind = copy.deepcopy(individual)
    idx = random.randint(0, len(new_ind) - 1)
    piece_id, variant, x, y = new_ind[idx]
    variants = ALL_VARIANTS[piece_id]
    variant = random.randint(0, len(variants) - 1)
    piece = variants[variant]
    x = random.randint(0, BOARD_HEIGHT - piece.shape[0])
    y = random.randint(0, BOARD_WIDTH - piece.shape[1])
    new_ind[idx] = (piece_id, variant, x, y)
    return new_ind

def crossover(p1, p2):
    cut = random.randint(1, len(p1) - 2)
    child = p1[:cut] + p2[cut:]
    seen = set()
    result = []
    for gene in child:
        if gene[0] not in seen:
            result.append(gene)
            seen.add(gene[0])
    # تکمیل ژن‌های حذف‌شده
    missing = [pid for pid in PIECES if pid not in seen]
    for pid in missing:
        variants = ALL_VARIANTS[pid]
        variant = random.randint(0, len(variants) - 1)
        piece = variants[variant]
        x = random.randint(0, BOARD_HEIGHT - piece.shape[0])
        y = random.randint(0, BOARD_WIDTH - piece.shape[1])
        result.append((pid, variant, x, y))
    return result

def tournament(population):
    return max(random.sample(population, TOURNAMENT_SIZE), key=lambda ind: evaluate(ind))

def draw(individual):
    board = np.full((BOARD_HEIGHT, BOARD_WIDTH), -1)
    for i, (piece_id, variant, x, y) in enumerate(individual):
        piece = ALL_VARIANTS[piece_id][variant]
        # بررسی اینکه قطعه قابل قرار دادن هست
        temp_board = board.copy()
        if place_piece(temp_board, piece, x, y):
            mask = (piece == 1)
            board[x:x+piece.shape[0], y:y+piece.shape[1]][mask] = piece_id
    cmap = plt.colormaps.get_cmap('tab20')
    plt.imshow(board, cmap=cmap)
    plt.axis('off')
    plt.show()


# اجرای الگوریتم ژنتیک
population = [create_individual() for _ in range(POPULATION_SIZE)]

for gen in range(NUM_GENERATIONS):
    population.sort(key=lambda ind: evaluate(ind), reverse=True)
    best = evaluate(population[0])
    print(f"Generation {gen}, Best fitness: {best}")
    if best == BOARD_WIDTH * BOARD_HEIGHT:
        break

    new_pop = [population[0]]
    while len(new_pop) < POPULATION_SIZE:
        p1 = tournament(population)
        p2 = tournament(population)
        child = crossover(p1, p2)
        if random.random() < MUTATION_RATE:
            child = mutate(child)
        new_pop.append(child)
    population = new_pop

# نمایش بهترین جواب
draw(population[0])
