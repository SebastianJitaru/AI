import random 
from typing import Union, List
from treepredict import *

def train_test_split(dataset, test_size: Union[float, int], seed=None):
    if seed:
        random.seed(seed)

    # If test size is a float, use it as a percentage of the total rows
    # Otherwise, use it directly as the number of rows in the test dataset
    n_rows = len(dataset)
    if float(test_size) != int(test_size):
        test_size = int(n_rows * test_size)  # We need an integer number of rows

    # From all the rows index, we get a sample which will be the test dataset
    choices = list(range(n_rows))
    test_rows = random.choices(choices, k=test_size)

    test = [row for (i, row) in enumerate(dataset) if i in test_rows]
    train = [row for (i, row) in enumerate(dataset) if i not in test_rows]

    return train, test

def get_accuracy(classifier, dataset):
    correct = 0
    for row in dataset:
        classification = classify(row, classifier)
        #print(row)
        #print(classification)
        if row[-1] in classification:
            correct += 1
    return correct / len(dataset)



def mean(values: List[float]):
    return sum(values) / len(values)



def cross_validation(dataset, k, agg, seed, scoref, beta, threshold):
    """
    t12: Cross validation
    """
    if seed:
        random.seed(seed)

    n_rows = len(dataset)
    subset_size = n_rows // k
    accuracy_list = []
    for i in range(k):
        # Get the current subset
        test_start = i * subset_size
        test_end = test_start + subset_size
        test_set = dataset[test_start:test_end]
        train_set = dataset[:test_start] + dataset[test_end:]

        # Build the tree and test it
        tree = buildtree(train_set, scoref)
        prune(tree,threshold)
        #print_tree(tree)
        accuracy = get_accuracy(tree, test_set)
        accuracy_list.append(accuracy)
    # Return the aggregate performance
    return agg(accuracy_list)

def main():
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = "iris.csv"


    
    headers, data = read(filename)
    seed = 2
    random.seed(seed)
    random.shuffle(data)

    train,test = train_test_split(data,0.2,seed)
    
    thresholds = [0.1, 0.3, 0.5, 0.7]
    maxac=0
    best_threshold = 0
    for t in thresholds:
        ac= cross_validation(dataset=train,k=5,seed=seed,scoref=entropy,threshold=t,agg=mean,beta=0)
        print(f"Threshold: {t}, Accuracy: {ac}")
        if ac > maxac:
            maxac = ac
            best_threshold = t
    print(f"Best threshold: {best_threshold} with accuracy: {maxac}")
    
    tree = buildtree(data)
    prune(tree,1)
    print_tree(tree, headers)


if __name__ == "__main__":
    main()


