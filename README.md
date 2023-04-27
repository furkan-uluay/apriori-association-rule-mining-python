# Association Rule Mining using Apriori Algorithm

This script is an implementation of association rule mining using the Apriori algorithm in Python. It uses the `mlxtend` library for preprocessing, frequent itemset generation, and association rule mining.

## Installation

To run the script, you will need Python 3 and the following libraries installed:

- `pandas`
- `mlxtend`

You can install these libraries using pip by running the following command:

```python
pip install pandas mlxtend
```

## Usage

To use the script, you need to provide the path to the dataset file, the minimum support, and the minimum confidence for the association rules. The dataset file should be a CSV file with one transaction per line, where each transaction contains a list of items separated by commas.

```python
file_path = 'dataset.csv'
min_support = 0.2
min_confidence = 0.4
data = read_data(file_path)

frequent_itemsets, rules, execution_time = apply_apriori(data, min_support, min_confidence)
print_frequent_itemsets(frequent_itemsets, 'Apriori', min_support, execution_time)
print_rules(rules, 'Apriori', min_confidence, execution_time)
```

## Dataset Format

The dataset file should be a CSV file with one transaction per line, where each transaction contains a list of items separated by commas. Here's an example dataset:

```csv
A,B,C
B,C,D
A,B
A,C,D
B,C
```

## Contributing

If you find any issues with the script or have suggestions for improvements, please feel free to open an issue or a pull request on the GitHub repository.
