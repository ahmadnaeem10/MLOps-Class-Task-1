from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

def train_model():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(model, 'model.pkl')
    
    # Print a message to indicate success
    print("Model trained and saved as model.pkl")

# Call the function to execute the training
train_model()
