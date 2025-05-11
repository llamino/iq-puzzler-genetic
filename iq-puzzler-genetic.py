import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # or 'Qt5Agg', 'Agg', etc., depending on what you have installed

import matplotlib.pyplot as plt
import random
import copy

BOARD_WIDTH = 11
BOARD_HEIGHT = 5
POPULATION_SIZE = 400
NUM_GENERATIONS = 200

MUTATION_RATE = 0.4
TOURNAMENT_SIZE = 5

# تعریف اولیه قطعات با مختصات
SHAPES = [
    [(0, 0), (0, 1), (0, 2), (1, 2)],                         # Black
    [(0, 0), (0, 1), (1, 0), (1, 1)],                         # Gray
    [(0, 0), (0, 1), (1, 1)],                                 # Yellow
    [(0, 0), (1, 0), (2, 0), (2, 1), (3, 0)],                 # Green
    [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],                 # Cream
    [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1)],                 # Orange
    [(0, 1), (0, 2), (1, 0), (1, 1), (2, 0)],                 # Purple
    [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],                 # Blue
    [(0, 0), (0, 1), (0, 2), (0, 3)],                         # Pink
    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)],                 # White
    [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],                 # Red
    [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)],                 # Light blue
]

# تبدیل به آرایه باینری
def shape_to_array(shape):
    max_x = max(x for x, y in shape)
    max_y = max(y for x, y in shape)
    arr = np.zeros((max_x + 1, max_y + 1), dtype=int)
    for x, y in shape:
        arr[x, y] = 1
    return arr

PIECES = {i: shape_to_array(s) for i, s in enumerate(SHAPES)}

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
    used_piece_ids = set()
    for piece_id, variant, x, y in individual:
        if piece_id in used_piece_ids:
            continue
        piece = ALL_VARIANTS[piece_id][variant]
        temp_board = board.copy()
        if place_piece(temp_board, piece, x, y):
            mask = (piece == 1)
            temp_board[x:x+piece.shape[0], y:y+piece.shape[1]][mask] = piece_id
            board = temp_board
            filled += np.sum(piece)
            used_piece_ids.add(piece_id)
    return filled

def print_board(individual):
    board = np.full((BOARD_HEIGHT, BOARD_WIDTH), -1)
    for piece_id, variant, x, y in individual:
        piece = ALL_VARIANTS[piece_id][variant]
        if place_piece(board, piece, x, y):
            mask = (piece == 1)
            board[x:x+piece.shape[0], y:y+piece.shape[1]][mask] = piece_id

    print("Final Board:")
    for row in board:
        print(" ".join(f"{cell:2}" if cell != -1 else " ." for cell in row))

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
    piece_id, variant, _, _ = new_ind[idx]
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

draw(population[0])
print_board(population[0])
