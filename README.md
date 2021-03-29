# Disaster Response Pipeline Project

### Project Motivation:
Figure 8 is a human-in-the-loop machine learning and artificial intelligence company that creates customized large scale high-quality training data using machine learning. Disaster response can greatly benefit from data modeling and machine learning by analyzing disaster data to build a model for an API that classifies disaster messages.

### Required Libraries:
- pandas
- sqlalchemy
- re
- nltk 
- sklearn
- pickle

### Files:
#### ETL Pipeline
File data/process_data.py contains an ETL data cleaning pipeline that:
- Loads the messages and categories datasets.
- Merges the two datasets.
- Cleans the data.
- Stores it in a SQLite database.

#### ML Pipeline
File models/train_classifier.py contains a ML machine learning pipeline that:
- Loads data from the SQLite database.
- Splits the dataset into training and test sets.
- Builds a text processing and machine learning pipeline.
- Trains and tunes a model using GridSearchCV.
- Outputs results on the test set.
- Exports the final model as a pickle file.

#### Flask Web App
File app/run.py contains a web application that:
- allows an emergency worker to input a new message and get classification results in several categories.
- display visualizations of the data.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/messages.csv data/categories.csv data/disaster_response.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/disaster_response.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### References:
Disaster messages and disaster categories datasets were downloaded from Figure 8.

### Acknowledgement
This project is part of Udacity Data Scientist Nanodegree.
