"""Block matching algorithm for slide evaluation using Hungarian algorithm."""
import numpy as np
from copy import deepcopy
from collections import Counter
from difflib import SequenceMatcher
from scipy.optimize import linear_sum_assignment

def calculate_similarity(block1: dict, block2: dict) -> float:
    if block1.get("text", None) is None or block2.get("text", None) is None:
        return 0.0
    text_similarity = SequenceMatcher(None, block1["text"], block2["text"]).ratio()
    return text_similarity

def create_cost_matrix(A, B):
    n = len(A)
    m = len(B)
    cost_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            cost_matrix[i, j] = -calculate_similarity(A[i], B[j])
    return cost_matrix

def adjust_cost_for_context(cost_matrix: list[list[float]], consecutive_bonus=1.0, window_size=20):
    if window_size <= 0:
        return cost_matrix

    n, m = cost_matrix.shape
    adjusted_cost_matrix = np.copy(cost_matrix)

    for i in range(n):
        for j in range(m):
            bonus = 0
            if adjusted_cost_matrix[i][j] >= -0.5:
                continue
            nearby_matrix = cost_matrix[max(0, i - window_size):min(n, i + window_size + 1), max(0, j - window_size):min(m, j + window_size + 1)]
            flattened_array = nearby_matrix.flatten()
            sorted_array = np.sort(flattened_array)[::-1]
            sorted_array = np.delete(sorted_array, np.where(sorted_array == cost_matrix[i, j])[0][0])
            top_k_elements = sorted_array[- window_size * 2:]
            sum_top_k = np.sum(top_k_elements)
            bonus = consecutive_bonus * sum_top_k
            adjusted_cost_matrix[i][j] += bonus
    return adjusted_cost_matrix

def calculate_current_cost(cost_matrix, row_ind, col_ind):
    return cost_matrix[row_ind, col_ind].tolist()

def find_maximum_matching(A, B, consecutive_bonus, window_size):
    cost_matrix = create_cost_matrix(A, B)
    cost_matrix = adjust_cost_for_context(cost_matrix, consecutive_bonus, window_size)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    current_cost = calculate_current_cost(cost_matrix, row_ind, col_ind)
    return list(zip(row_ind.tolist(), col_ind.tolist())), current_cost, cost_matrix


def print_matching(matching, blocks1, blocks2, cost_matrix):
    for i, j in matching:
        print(f"{blocks1[i]} matched with {blocks2[j]}, cost {cost_matrix[i][j]}")

# %% to revise maybe 
def merge_blocks_wo_check(block1, block2):
    # Concatenate text
    merged_text = block1['text'] + " " + block2['text']

    # Calculate bounding box
    x_min = min(block1['bbox'][0], block2['bbox'][0])
    y_min = min(block1['bbox'][1], block2['bbox'][1])
    x_max = max(block1['bbox'][0] + block1['bbox'][2], block2['bbox'][0] + block2['bbox'][2])
    y_max = max(block1['bbox'][1] + block1['bbox'][3], block2['bbox'][1] + block2['bbox'][3])
    merged_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

    # Average color
    merged_color = tuple(
        (color1 + color2) // 2 for color1, color2 in zip(block1['color'], block2['color'])
    )

    return {'text': merged_text, 'bbox': merged_bbox, 'color': merged_color}

def difference_of_means(list1, list2):
    counter1 = Counter(list1)
    counter2 = Counter(list2)

    for element in set(list1) & set(list2):
        common_count = min(counter1[element], counter2[element])
        counter1[element] -= common_count
        counter2[element] -= common_count

    unique_list1 = [item for item in counter1.elements()]
    unique_list2 = [item for item in counter2.elements()]

    mean_list1 = sum(unique_list1) / len(unique_list1) if unique_list1 else 0
    mean_list2 = sum(unique_list2) / len(unique_list2) if unique_list2 else 0

    if mean_list1 - mean_list2 > 0:
        if min(unique_list1) > min(unique_list2):
            return mean_list1 - mean_list2
        else:
            return 0.0
    else:
        return mean_list1 - mean_list2

def remove_indices(lst, indices):
    for index in sorted(indices, reverse=True):
        if index < len(lst):
            lst.pop(index)
    return lst

def merge_blocks_by_list(blocks, merge_list):
    pop_list = []
    while True:
        if len(merge_list) == 0:
            remove_indices(blocks, pop_list)
            return blocks

        i = merge_list[0][0]
        j = merge_list[0][1]
    
        blocks[i] = merge_blocks_wo_check(blocks[i], blocks[j])
        pop_list.append(j)
    
        merge_list.pop(0)
        if len(merge_list) > 0:
            new_merge_list = []
            for k in range(len(merge_list)):
                if merge_list[k][0] != i and merge_list[k][1] != i and merge_list[k][0] != j and merge_list[k][1] != j:
                    new_merge_list.append(merge_list[k])
            merge_list = new_merge_list

def find_possible_merge(A, B, consecutive_bonus, window_size, debug=False):
    merge_bonus = 0.0
    merge_windows = 1

    def sortFn(value):
        return value[2]

    while True:
        A_changed = False
        B_changed = False

        matching, current_cost, cost_matrix = find_maximum_matching(A, B, merge_bonus, merge_windows)
        if debug:
            print("Current cost of the solution:", current_cost)
            print_matching(matching, A, B, cost_matrix)
    
        if len(A) >= 2:
            merge_list = []
            for i in range(len(A) - 1):
                new_A = deepcopy(A)
                new_A[i] = merge_blocks_wo_check(new_A[i], new_A[i + 1])
                new_A.pop(i + 1)
    
                updated_matching, updated_cost, cost_matrix = find_maximum_matching(new_A, B, merge_bonus, merge_windows)
                diff = difference_of_means(current_cost, updated_cost)
                if  diff > 0.05:
                    merge_list.append([i, i + 1, diff])
                    if debug:
                        print(new_A[i]['text'], diff)

            merge_list.sort(key=sortFn, reverse=True)
            if len(merge_list) > 0:
                A_changed = True
                A = merge_blocks_by_list(A, merge_list)
                matching, current_cost, cost_matrix = find_maximum_matching(A, B, merge_bonus, merge_windows)
                if debug:
                    print("Cost after optimization A:", current_cost)

        if len(B) >= 2:
            merge_list = []
            for i in range(len(B) - 1):
                new_B = deepcopy(B)
                new_B[i] = merge_blocks_wo_check(new_B[i], new_B[i + 1])
                new_B.pop(i + 1)
    
                updated_matching, updated_cost, cost_matrix = find_maximum_matching(A, new_B, merge_bonus, merge_windows)
                diff = difference_of_means(current_cost, updated_cost)
                if diff > 0.05:
                    merge_list.append([i, i + 1, diff])
                    if debug:
                        print(new_B[i]['text'], diff)

            merge_list.sort(key=sortFn, reverse=True)
            if len(merge_list) > 0:
                B_changed = True
                B = merge_blocks_by_list(B, merge_list)
                matching, current_cost, cost_matrix = find_maximum_matching(A, B, merge_bonus, merge_windows)
                if debug:
                    print("Cost after optimization B:", current_cost)

        if not A_changed and not B_changed:
            break
    matching, _, _ = find_maximum_matching(A, B, consecutive_bonus, window_size)
    return A, B, matching
