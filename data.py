# Data for future nearby locations
# TODO: encode questions for past context.
import pandas as pd
import spacy

train_data = {
    'door': [
        'Find next door',
        'Go to next door',
        'Check in next room', 
        'Look for doors',
        'Can you locate any room?',
        'Can you locate any room nearby?',
        'Keep an eye for other doors',
        'Can you figure a way into the room?',
        'Can you locate any door nearby?',
        'Check for other victims in the next room', 
    ], 
    'goal': [
        'Save the victims', 
        'Go search for other victims and triage them',
        'Locate other victims now',
        'Check for other casualities',
        # 'Have you seen any injured ones?',
        'Look for injured and save them',
        'Look for any injured victims',
        'Can you locate any people?',
        # 'Can you locate anyone?',
        'Keep an eye for other casualities',
        'Can you figure a way to save victims?',
        # 'Can you locate any person nearby?',
    ],
    'key': [
        'Find a light switch',
        'Locate a lever nearby',
        'Turn on the switch',
        'Go to a lever', 
        'Figure out how to turn on the light',
    ]
}

test_data = {
    'door': [
        'Go to the next door',
        'Can you find the next door',
        'Let us check inside the next room', 
        'Keep an eye for other doors',
        'Can you figure a way into the room?',
        'Can you locate any door nearby?',
    ], 
    'goal': [
        'Search for next victim', 
        'Go search for other injured ones',
        'Locate other affected',
    ],
    'key': [
        'Look out for a light switch',
        'Go to the nearest lever',
        'Turn off the electricity',
    ]
}

def get_datalist_from_dict(data_dict):  #,lm):
    """
    data_dict: like train_data, test_data with 
        key as target object and 
        value as a list of language instructions
    lm: language model (like spaCy). 
        Ensure it is callable and 
        returns object that has vector attribute
    """

    data = []
    for key, values in data_dict.items():
        for val in values:
            data.append([val, key])
    return data

def to_csv(data, name):
    """
    data is a list or numpy array with each entry as a ['text', 'target']
    """
    df = pd.DataFrame(data)
    df.to_csv(name, header=['text', 'target'], index=False) 
    return df

if __name__ == "__main__":
    # nlp = spacy.load('en_core_web_sm')
    train_list = get_datalist_from_dict(train_dat) #@, nlp)
    train_df = to_csv(train_list, 'train_data.csv')
    test_list = get_datalist_from_dict(test_data, nlp)
    test_df = to_csv(test_list, 'test_data.csv')