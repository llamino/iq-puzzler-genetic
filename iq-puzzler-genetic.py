import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define pieces: relative (x, y) positions
PIECES = [
    [(0, 0), (1, 0), (2, 0), (2, 1)],     # L-shaped (horizontal)
    [(0, 0), (1, 0), (2, 0), (2, 1)],     # L-shaped (vertical)
    [(0, 0), (1, 0), (2, 0), (3, 0)],     # Line
    [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)], # Reverse +
    [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)], # Cross
    [(0, 0), (1, 0), (2, 0), (3, 0)],     # Horizontal line
    [(1, 0), (0, 1), (1, 1), (2, 1)],     # T-shape
    [(0, 0), (1, 0), (1, 1)],             # L mini
    [(0, 0), (1, 0), (2, 0)],             # Small line
    [(0, 0), (1, 0), (0, 1), (1, 1)],     # Square
    [(0, 0), (1, 0), (2, 0), (2, -1)],    # Left semicircle
    [(1, 0), (0, 1), (1, 1)]              # T mini
]

# Puzzle board
def generate_puzzle(width=5, length=11):
    return np.full((width, length), -1)

# Text output
def display_puzzle(puzzle):
    for row in puzzle:
        print(" ".join(f"{cell:2}" if cell != -1 else " ." for cell in row))

# Random rotation (0°, 90°, 180°, 270°)
def random_rotation_dice(piece):
    rotations = [
        lambda x, y: (x, y),
        lambda x, y: (-y, x),
        lambda x, y: (-x, -y),
        lambda x, y: (y, -x),
    ]
    rotation = random.choice(rotations)
    return [rotation(x, y) for (x, y) in piece]

# Try placing a piece on the board
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

# Generate one chromosome (individual)
def generate_random_individual(pieces, puzzle_shape):
    puzzle = generate_puzzle(*puzzle_shape)
    placement = []
    for i, piece in enumerate(pieces):
        placed = False
        for _ in range(100):
            rotated = random_rotation_dice(piece)
            x = random.randint(0, puzzle.shape[0] - 1)
            y = random.randint(0, puzzle.shape[1] - 1)
            new_puzzle = place_piece(puzzle, rotated, x, y, i)
            if new_puzzle is not None:
                puzzle = new_puzzle
                placement.append((rotated, x, y))
                placed = True
                break
        if not placed:
            placement.append(None)
    return placement

# Score a chromosome
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

# Mutation with low chance
def mutate(individual, pieces, puzzle_shape):
    if random.random() > 0.05:
        return individual  # 5% mutation chance

    idx = random.randint(0, len(individual) - 1)
    new_piece = random_rotation_dice(pieces[idx])
    x = random.randint(0, puzzle_shape[0] - 1)
    y = random.randint(0, puzzle_shape[1] - 1)
    new_ind = copy.deepcopy(individual)
    new_ind[idx] = (new_piece, x, y)
    return new_ind

# Smarter crossover: keep good pieces from parent1
def crossover(parent1, parent2, puzzle_shape):
    child = [None] * len(parent1)
    temp_puzzle = generate_puzzle(*puzzle_shape)
    used_indices = set()

    for i, placement in enumerate(parent1):
        if placement is None:
            continue
        piece, x, y = placement
        result = place_piece(temp_puzzle, piece, x, y, i)
        if result is not None:
            temp_puzzle = result
            child[i] = placement
            used_indices.add(i)

    for i, placement in enumerate(parent2):
        if i not in used_indices:
            child[i] = placement

    return child

# Graphical display
def display_puzzle_graphically(puzzle):
    cmap = plt.cm.get_cmap('tab20', np.max(puzzle)+2)
    norm = mcolors.BoundaryNorm(boundaries=np.arange(-1, np.max(puzzle)+2), ncolors=np.max(puzzle)+2)

    plt.figure(figsize=(11, 5))
    plt.imshow(puzzle, cmap=cmap, norm=norm)
    plt.xticks([])
    plt.yticks([])
    plt.title("IQ Puzzler Pro - Final Solution")
    plt.grid(False)
    plt.show()

# Main genetic algorithm
def iq_puzzler_genetic(puzzle, pieces, population_size=500, generations=2000):
    puzzle_shape = puzzle.shape
    population = [generate_random_individual(pieces, puzzle_shape) for _ in range(population_size)]

    for gen in range(generations):
        # ارزیابی جمعیت فعلی
        scored = [(individual, fitness(individual, puzzle_shape)) for individual in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        maintain_generation_length = int(len(population) * 0.3)
        elites = [ind for ind, fit in scored[:maintain_generation_length]]
        best_fit = scored[0][1]
        print(f"Generation {gen}, Best fitness: {best_fit}")

        if best_fit >= sum(len(p) for p in pieces):
            break

        # تولید ژن‌های جدید
        new_generation = []
        while len(new_generation) < population_size * (1 - maintain_generation_length):
            p1, p2 = random.sample(elites, 2)
            child = crossover(p1, p2)
            if random.random() < 0.01:  # احتمال پایین برای جهش
                child = mutate(child, pieces, puzzle_shape)
            new_generation.append(child)

        population = elites + new_generation

    # بازسازی بهترین پاسخ نهایی
    best = scored[0][0]
    print(f"Generation {gen}, Best fitness: {best}")
    final_puzzle = generate_puzzle(*puzzle_shape)
    print(f'puzzle shape: {puzzle_shape}')
    for i, placement in enumerate(best):
        if placement is not None:
            piece, x, y = placement
            result = place_piece(final_puzzle, piece, x, y, i)
            if result is not None:
                final_puzzle = result
    return final_puzzle
# Run
if __name__ == '__main__':
    puzzle = generate_puzzle(5, 11)
    solved = iq_puzzler_genetic(puzzle, PIECES, population_size=3000, generations=200)
    display_puzzle(solved)
    display_puzzle_graphically(solved)
