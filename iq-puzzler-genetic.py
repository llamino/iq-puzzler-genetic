import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# تعریف تکه‌ها: مختصات نسبی
PIECES = [
    [(0, 0), (1, 0), (2, 0), (2, 1)],      # L-shaped (horizontal)
    [(0, 0), (1, 0), (2, 0), (2, 1)],      # L-shaped (vertical)
    [(0, 0), (1, 0), (2, 0), (3, 0)],      # Line
    [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)],  # Reverse +
    [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)],  # Cross
    [(0, 0), (1, 0), (2, 0), (3, 0)],      # Horizontal line
    [(1, 0), (0, 1), (1, 1), (2, 1)],      # T-shape
    [(0, 0), (1, 0), (1, 1)],              # L mini
    [(0, 0), (1, 0), (2, 0)],              # Small line
    [(0, 0), (1, 0), (0, 1), (1, 1)],      # Square
    [(0, 0), (1, 0), (2, 0), (2, -1)],     # Left semicircle
    [(1, 0), (0, 1), (1, 1)]               # T mini
]

# ایجاد پازل خالی
def generate_puzzle(width=5, length=11):
    return np.full((width, length), -1)

# نمایش متنی پازل
def display_puzzle(puzzle):
    for row in puzzle:
        print(" ".join(f"{cell:2}" if cell != -1 else " ." for cell in row))

# چرخش تصادفی 0°, 90°, 180°, 270°
def random_rotation_dice(piece):
    rotations = [
        lambda x, y: (x, y),
        lambda x, y: (-y, x),
        lambda x, y: (-x, -y),
        lambda x, y: (y, -x),
    ]
    rotation = random.choice(rotations)
    return [rotation(x, y) for (x, y) in piece]

# قرار دادن یک تکه روی پازل
def place_piece(puzzle, piece, x, y, piece_id):
    new_puzzle = puzzle.copy()
    for dx, dy in piece:
        nx, ny = x + dx, y + dy
        if 0 <= nx < puzzle.shape[0] and 0 <= ny < puzzle.shape[1]:
            if new_puzzle[nx][ny] != -1:
                return None
        else:
            return None
    for dx, dy in piece:
        nx, ny = x + dx, y + dy
        new_puzzle[nx][ny] = piece_id
    return new_puzzle

# تولید یک فرد تصادفی اولیه
def generate_random_individual(pieces, puzzle_shape):
    puzzle = generate_puzzle(*puzzle_shape)
    placement = []
    for i, piece in enumerate(pieces):
        placed = False
        for _ in range(100):
            rotated = random_rotation_dice(piece)
            x = random.randint(0, puzzle_shape[0] - 1)
            y = random.randint(0, puzzle_shape[1] - 1)
            new_puzzle = place_piece(puzzle, rotated, x, y, i)
            if new_puzzle is not None:
                puzzle = new_puzzle
                placement.append((rotated, x, y))
                placed = True
                break
        if not placed:
            placement.append(None)
    return placement

# تابع امتیازدهی (جمع طول همه تکه‌های قرارگرفته)
def fitness(individual, puzzle_shape):
    puzzle = generate_puzzle(*puzzle_shape)
    score = 0
    for i, placement in enumerate(individual):
        if placement is not None:
            piece, x, y = placement
            result = place_piece(puzzle, piece, x, y, i)
            if result is not None:
                puzzle = result
                score += len(piece)
    return score

# جهش ساده با احتمال کم
def mutate(individual, pieces, puzzle_shape):
    if random.random() > 0.05:
        return individual
    idx = random.randint(0, len(individual) - 1)
    new_piece = random_rotation_dice(pieces[idx])
    x = random.randint(0, puzzle_shape[0] - 1)
    y = random.randint(0, puzzle_shape[1] - 1)
    new_ind = copy.deepcopy(individual)
    new_ind[idx] = (new_piece, x, y)
    return new_ind

# جستجوی محلی ساده (hill-climbing)
def local_hill_climb(individual, pieces, puzzle_shape, iters=20):
    best = copy.deepcopy(individual)
    best_score = fitness(best, puzzle_shape)
    for _ in range(iters):
        i = random.randrange(len(best))
        new = copy.deepcopy(best)
        rotated = random_rotation_dice(pieces[i])
        x = random.randint(0, puzzle_shape[0] - 1)
        y = random.randint(0, puzzle_shape[1] - 1)
        new[i] = (rotated, x, y)
        sc = fitness(new, puzzle_shape)
        if sc > best_score:
            best, best_score = new, sc
    return best

# کراس‌اور بهبود یافته با اولویت قطعات بزرگ و جستجوی محلی
def crossover_improved(parent1, parent2, pieces, puzzle_shape, base_fill_attempts=30):
    # 1) کراس‌اور یکنواخت
    prelim = [(parent1[i] if random.random()<0.5 else parent2[i]) for i in range(len(parent1))]

    # 2) تعمیر
    child = [None] * len(prelim)
    temp_puzzle = generate_puzzle(*puzzle_shape)
    for i, placement in enumerate(prelim):
        if placement:
            p, x, y = placement
            res = place_piece(temp_puzzle, p, x, y, i)
            if res is not None:
                temp_puzzle = res
                child[i] = placement

    # 3) تکمیل هوشمند: بزرگ‌ترها اول
    missing = [i for i, g in enumerate(child) if g is None]
    missing.sort(key=lambda i: -len(pieces[i]))
    for i in missing:
        attempts = base_fill_attempts + len(pieces[i]) * 5
        for _ in range(attempts):
            rotated = random_rotation_dice(pieces[i])
            x = random.randint(0, puzzle_shape[0] - 1)
            y = random.randint(0, puzzle_shape[1] - 1)
            res = place_piece(temp_puzzle, rotated, x, y, i)
            if res is not None:
                temp_puzzle = res
                child[i] = (rotated, x, y)
                break

    # 4) جستجوی محلی برای بهبود نهایی
    child = local_hill_climb(child, pieces, puzzle_shape, iters=20)
    return child

# نمایش گرافیکی پازل نهایی
def display_puzzle_graphically(puzzle):
    cmap = plt.cm.get_cmap('tab20', np.max(puzzle) + 2)
    norm = mcolors.BoundaryNorm(boundaries=np.arange(-1, np.max(puzzle) + 2), ncolors=np.max(puzzle) + 2)
    plt.figure(figsize=(11, 5))
    plt.imshow(puzzle, cmap=cmap, norm=norm)
    plt.xticks([])
    plt.yticks([])
    plt.title("IQ Puzzler Pro - Final Solution")
    plt.grid(False)
    plt.show()

# الگوریتم ژنتیک اصلی
def iq_puzzler_genetic(puzzle, pieces, population_size=500, generations=2000):
    puzzle_shape = puzzle.shape
    population = [generate_random_individual(pieces, puzzle_shape) for _ in range(population_size)]

    for gen in range(generations):
        scored = [(ind, fitness(ind, puzzle_shape)) for ind in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        maintain_len = int(population_size * 0.3)
        elites = [ind for ind, _ in scored[:maintain_len]]
        best_fit = scored[0][1]
        print(f"Generation {gen}, Best fitness: {best_fit}")
        if best_fit >= sum(len(p) for p in pieces):
            break

        new_gen = []
        while len(new_gen) < population_size - maintain_len:
            p1, p2 = random.sample(elites, 2)
            child = crossover_improved(p1, p2, pieces, puzzle_shape)
            if random.random() < 0.01:
                child = mutate(child, pieces, puzzle_shape)
            new_gen.append(child)
        population = elites + new_gen

    best_ind = scored[0][0]
    final = generate_puzzle(*puzzle_shape)
    for i, placement in enumerate(best_ind):
        if placement:
            p, x, y = placement
            res = place_piece(final, p, x, y, i)
            if res is not None:
                final = res
    return final

if __name__ == '__main__':
    puzzle = generate_puzzle(5, 11)
    solved = iq_puzzler_genetic(puzzle, PIECES, population_size=3000, generations=500)
    display_puzzle(solved)
    display_puzzle_graphically(solved)