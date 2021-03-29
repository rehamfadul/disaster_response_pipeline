import sys
import pandas as pd 
from sqlalchemy import create_engine
import re
import nltk 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV 
from sklearn.metrics import classification_report
import pickle

nltk.download(['wordnet', 'punkt', 'stopwords'])

def load_data(database_filepath):
    """
    A function to load data
    
    Args:
        database_filepath: String. Filepath for the database
    
    Returns:
        X: Dataframe containing the features data
        Y: Dataframe containing the labels (categories) data
        category_names: List of labels (categories) names
    """
    # load data from the database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_response_table',engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X,Y,category_names


def tokenize(text):
    '''
    A function to normalize and tokenize messages, and return the root form of the words after removing stop words

    Args: 
        text: String. Sentence containing a message
    
    Returns: 
        clean_tokens: List of message words/tokens
    '''
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    words = word_tokenize(text)
    
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Lemmatization
    clean_tokens = [WordNetLemmatizer().lemmatize(w).strip() for w in words]
    return clean_tokens


def build_model():
    '''
    A function to build a model, create pipeline, and perfom gridsearchcv
    Args:
        None
    Returns:
        model: the model
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters =  {'clf__estimator__min_samples_split': [2, 4]}
            #   'clf__estimator__n_estimators': [50, 100], 
              
    model = GridSearchCV(pipeline, param_grid=parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    A function to evaluate a model and return the classificatio report and accurancy score
    
    Args:
        model: The model
        X_test: Pandas dataframe containing test features
        Y_test: Pandas dataframe containing test labels (categories)
        category_names: List of labels (categories) names
    
    Returns:
        None
    '''
    Y_pred = model.predict(X_test)
    print(classification_report(Y_pred, Y_test, target_names=category_names))


def save_model(model, model_filepath):
    '''
    A function to save the model as a pickle file in the given filepath
    
    Args: 
        model: The model
        model_filepath: File path of the model
    Returns: 
        None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()