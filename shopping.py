import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    evidence = list()
    labels = list()

    with open(filename, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            admin = int(row[0])
            adminDur = float(row[1])
            info = int(row[2])
            infoDur = float(row[3])
            prod = int(row[4])
            prodDur = float(row[5])
            bounceR = float(row[6])
            exitR = float(row[7])
            pageV = float(row[8])
            specialD = float(row[9])
            month = convert_month(row[10])
            opSystem = int(row[11])
            browser = int(row[12])
            region = int(row[13])
            trafficType = int(row[14])

            visitorType = convert_visType(row[15])
            weekend = 1 if row[16] == 'TRUE' else 0

            evidenceElem = [admin, adminDur, info, infoDur, prod, prodDur, bounceR, exitR, pageV, specialD, month, opSystem, browser, region, trafficType, visitorType, weekend]
            evidence.append(evidenceElem)

            label = 1 if row[-1] == 'TRUE' else 0
            labels.append(label)

    return (evidence, labels)


def convert_month(month):
    if month == 'Jan':
        return 1
    elif month == 'Feb':
        return 2
    elif month == 'Mar':
        return 3
    elif month == 'Apr':
        return 4
    elif month == 'May':
        return 5
    elif month == 'June':
        return 6
    elif month == 'Jul':
        return 7
    elif month == 'Aug':
        return 8
    elif month == 'Sep':
        return 9
    elif month == 'Oct':
        return 10
    elif month == 'Nov':
        return 11
    elif month == 'Dec':
        return 11
    else:
        return None


def convert_visType(visitorType):
    if visitorType == 'Returning_Visitor':
        return 1
    else:
        return 0


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    correct_positive = 0
    correct_negative = 0

    total_positive = labels.count(True)
    total_negative = labels.count(False)

    for actual, predicted in zip(labels, predictions):
        if actual == predicted and actual:
            correct_positive += 1
        elif actual == predicted and not actual:
            correct_negative += 1

    sensitivity = correct_positive / total_positive
    specificity = correct_negative / total_negative

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
