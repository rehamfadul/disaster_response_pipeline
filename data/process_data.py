import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    A function to load and merge datasets

    Args:
        messages_filepath: String. Filepath for the messages dataset
        categories_filepath: String. Filepath for the categories dataset
    
    Returns:
        df: Pandas dataframe containing messages and respective categories
    '''
    #load data from csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #merge two datasets on 'id'
    df = messages.merge(categories, how="inner", on="id")
    return df


def clean_data(df):
    '''
    A function to clean dataframes and remove duplicates
    
    Args:
        df: Pandas dataframe containing messages and categories
    
    Returns:
        df: Pandas dataframe containing a clean version of messages and categories
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [c.split('-')[0] for c in row.tolist()]
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.split('-').str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # cleaning 'related' column
    categories.loc[(categories.related==2),'related']=0
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    # drop duplicates
    df.drop_duplicates(inplace = True)
    return df

def save_data(df, database_filename):
    """
    A function to save the cleaned data
    
    Args:
        df: Pandas dataframe containing clean data for messages and respective categories
        database_filename: String. Filename for the output database
    
    Returns: 
        None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_response_table', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()