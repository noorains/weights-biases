# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import wandb
import yaml  # If you're using a YAML configuration file

# Load configuration from the file
with open('config.yaml', 'r') as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)

# Initialize WandB with project and entity from the configuration
wandb.init(project=config['project'], entity=config['entity'])

# Extract hyperparameters from the configuration
test_size = config['test_size']
max_depth = config['max_depth']
min_samples_split = config['min_samples_split']
criterion = config['criterion']

# Log hyperparameters to WandB
wandb.config.test_size = test_size
wandb.config.max_depth = max_depth
wandb.config.min_samples_split = min_samples_split
wandb.config.criterion = criterion

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Split the dataset into a training set and a testing set using the configured test_size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Create a decision tree classifier with hyperparameters
model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)

# Log the accuracy to WandB
wandb.log({"Accuracy": accuracy})

# Finish the WandB run
wandb.finish()
