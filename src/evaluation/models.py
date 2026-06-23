from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier

# MLPClassifier approximates Dense(128,relu)->Dense(64,relu) from the deep selectors.
# BatchNorm and Dropout are not available in sklearn; early_stopping acts as regularization.
default_models = {
    'SupportVectorMachine': SVC(),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'NaiveBayes': GaussianNB(),
    'ZeroR': DummyClassifier(),
    'MLP': MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    ),
}
