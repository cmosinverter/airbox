import numpy as np

def categorize_values(numbers, step=0.05):
    categories = []
    for number in numbers:
        if number < 0:
            categories.append(0)
        elif number > 1:
            categories.append(int((1/step)+1))
        else:
            section = int(number / 0.05)
            categories.append(section)
    return categories

# Example usage
numbers = [0.1, 0.3, -0.2, 0.9, 1.2, 0.6]
categories = categorize_values(numbers)
print(categories)
# Perform one-hot encoding
one_hot_encoded = np.eye(22)[categories]

print(one_hot_encoded)