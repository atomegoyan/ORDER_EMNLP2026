import os 
import pickle

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(DATA_DIR, "RAG_results_multihop")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings_1887_cohere")


CLASSIFIER_TYPE = "rf"  # 'rf' for Random Forest, 'lr' for Logistic Regression
MODEL_DIR = os.path.join(DATA_DIR, 'models')
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'rf_source_classifier.pkl')
LR_MODEL_PATH = os.path.join(MODEL_DIR, 'lr_source_classifier.pkl')


def load_classifier_model(classifier_type="rf"):
    """
    Load a pre-trained classification model for query rerouting.
    
    Args:
        classifier_type: 'rf' for Random Forest, 'lr' for Logistic Regression
    
    Returns:
        Loaded sklearn classifier model
    """
    if classifier_type == "rf":
        model_path = RF_MODEL_PATH
        model_name = "Random Forest"
    elif classifier_type == "lr":
        model_path = LR_MODEL_PATH
        model_name = "Logistic Regression"
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}. Use 'rf' or 'lr'.")
    
    print(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_name} model not found at: {model_path}\n"
            f"Please train the model first using the classification notebook."
        )
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"  Loaded {model_name} classifier from: {model_path}")
    return model