import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import time


def read_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            linex = line.strip().split(',')
            linex = list(filter(None, linex))
            data.append(linex)
    return data


def apply_apriori(data, min_support, min_confidence):
    te = TransactionEncoder()
    te_ary = te.fit(data).transform(data)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    start_time = time.time()
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    end_time = time.time()
    execution_time = end_time - start_time
    return frequent_itemsets, rules, execution_time


def print_frequent_itemsets(frequent_itemsets, algorithm_name, min_support, execution_time):
    print(f"Frequent Itemsets ({algorithm_name}, Support={min_support}, Execution Time={execution_time:.5f}s):")
    print("---------------------------------------------------")
    frequent_itemsets.sort_values(by=['support'], ascending=False, inplace=True)
    frequent_itemsets.reset_index(drop=True, inplace=True)
    frequent_itemsets.index += 1
    frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
    frequent_itemsets['support'] = frequent_itemsets['support'].apply(lambda x: f"{x:.2%}")
    print(frequent_itemsets[['support', 'itemsets']].to_string(index=True))
    print("---------------------------------------------------")


def print_rules(rules, algorithm_name, min_confidence, execution_time):
    print(f"Association Rules ({algorithm_name}, Confidence={min_confidence}, Execution Time={execution_time:.5f}s):")
    print("---------------------------------------------------")
    rules.sort_values(by=['lift'], ascending=False, inplace=True)
    rules.reset_index(drop=True, inplace=True)
    rules.index += 1
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    rules['support'] = rules['support'].apply(lambda x: f"{x:.2%}")
    rules['confidence'] = rules['confidence'].apply(lambda x: f"{x:.2%}")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_string(index=True))
    print("---------------------------------------------------")


if __name__ == '__main__':
    file_path = 'data_set.csv'
    min_support = 0.2
    min_confidence = 0.4
    data = read_data(file_path)

    frequent_itemsets_apriori, rules_apriori, apriori_execution_time = apply_apriori(data, min_support, min_confidence)
    print_frequent_itemsets(frequent_itemsets_apriori, 'Apriori', min_support, apriori_execution_time)
    print_rules(rules_apriori, 'Apriori', min_confidence, apriori_execution_time)
