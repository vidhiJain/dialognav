# Data for future nearby locations
# TODO: encode questions for past context.
import pandas as pd
# import spacy
from candidates import *
    # ['Find '+ token +' door' for token in list(REQUIRES_HISTORY.keys()) + list(SPATIAL_BIAS.keys())] + \
    # ['Go to '+ token +' door' for token in list(REQUIRES_HISTORY.keys()) + list(SPATIAL_BIAS.keys())] + \
    # [    

# sentence, object type, color, doesn't requires history?, spatial relation
data = {
    'door': [
        ['Find last door', 'door', None, False, None], 
        ['Find previous door', 'door', None, False, None], 
        ['Find the already seen door', 'door', None, False, None], 
        ['Go to the door that you had just found', 'door', None, False, None], 
        ['Find the door', 'door', None, False, None], 
        ['Reach the other door', 'door', None, False, None], 
        ['Now find the next door', 'door', None, True, None], 
        ['Find a new door', 'door', None, True, None], 
        ['Find a door', 'door', None, True, None], 
        ['Find another door', 'door', None, True, None], 
        ['Find door on your left', 'door', None, True, 'left'], 
        ['Find a door on the left side', 'door', None, True, 'left'], 
        ['Find the right door', 'door', None, True, 'right'], 
        ['Find a door on the right side', 'door', None, True, 'right'], 
        ['Find the front door', 'door', None, True, 'front'], 
        ['Find if there is any door in front of you', 'door', None, True, 'front'], 
        ['Find a back door', 'door', None, True, 'back'], 
        ['Go to last door', 'door', None, False, None], 
        ['Go to previous door', 'door', None, False, None], 
        ['Go to the already seen door', 'door', None, False, None], 
        ['Go to the other door you found', 'door', None, False, None], 
        ['Go to the door', 'door', None, True, None], 
        ['Go to the other door', 'door', None, True, None], 
        ['Go to the next door', 'door', None, True, None], 
        ['Go to new door', 'door', None, True, None], 
        ['Go to a door', 'door', None, True, None], 
        ['Go to another door', 'door', None, True, None], 
        ['Go to the left door', 'door', None, True, 'left'], 
        ['Go to door on the left side', 'door', None, True, 'left'], 
        ['Go to right door', 'door', None, True, 'right'], 
        ['Go to door on right side', 'door', None, True, 'right'], 
        ['Go to the door in front of you', 'door', None, True, 'front'], 
        ['Go to the back door', 'door', None, True, 'back'], 
        ['Check in the previous room', 'door', None, False, None], 
        ['Check in the next room', 'door', None, True, None], 
        ['Look for doors', 'door', None, True, None], 
        ['Reach adajecent room', 'door', None, True, None], 
        ['Move to last room', 'door', None, False, None], 
        ['Move to previous room', 'door', None, False, None], 
        ['Move to the already seen room', 'door', None, False, None], 
        ['Move to found room', 'door', None, False, None], 
        ['Move to the room', 'door', None, False, None], 
        ['Move to the other room', 'door', None, False, None], 
        ['Move to next room', 'door', None, True, None], 
        ['Move to new room', 'door', None, True, None], 
        ['Move to a room', 'door', None, True, None], 
        ['Move to another room', 'door', None, True, None], 
        ['Check in other room now', 'door', None, True, None], 
        ['Can you navigate to any room?', 'door', None, True, None], 
        ['Can you navigate to any room nearby?', 'door', None, True, None], 
        ['Keep an eye for other doors', 'door', None, True, None], 
        ['Can you find a way into the room?', 'door', None, False, None], 
        ['Can you navigate to any door nearby?', 'door', None, True, None], 
        ['Check for other victims in the room', 'door', None, False, None], 
        ['Can you find the next door?', 'door', None, True, None], 
        ['Can you find the way to the previous door?', 'door', None, False, None], 
        ['Can you find the way to the last door you entered?', 'door', None, False, None], 
        ['Can you find the way to a new room now?', 'door', None, True, None], 
        ['Can you find the way to another door?', 'door', None, True, None], 
        ['Can you find the way to the other door?', 'door', None, True, None], 
        ['Can you find the way to the last door you saw?', 'door', None, False, None)], 
        ['Can you figure out status of another room nearby?', 'door', None, True, None], 
        ['Check in last room', 'door', None, False, None], 
        ['Check in previous room', 'door', None, False, None], 
        ['Check in the already seen room', 'door', None, False, None], 
        ['Check in found room', 'door', None, False, None], 
        ['Check in the room', 'door', None, False, None], 
        ['Check in the other room', 'door', None, False, None], 
        ['Check in next room', 'door', None, True, None], 
        ['Check in new room', 'door', None, True, None], 
        ['Check in a room', 'door', None, True, None], 
        ['Check in another room', 'door', None, True, None], 
        ['Look for doors', 'door', None, True, None], 
        ['Reach adajecent room', 'door', None, True, None], 
        ['Check in other room now', 'door', None, True, None], 
        ['Can you navigate to any room?', 'door', None, True, None], 
        ['Can you navigate to any room nearby?', 'door', None, True, None], 
        ['Keep an eye for other doors', 'door', None, True, None], 
        ['Can you figure a way into the room?', 'door', None, True, None], 
        ['Can you navigate to any door nearby?', 'door', None, True, None], 
        ['Can you locate any door nearby?', 'door', None, True, None], 
    ], 
    'goal': [
        ['Search for yellow victim', 'goal', 'yellow', True, None],
        ['Search for green victim', 'goal', 'green', True, None],
        ['See if you can save yellow victim', 'goal', 'yellow', True, None],
        ['See if you can save green victim', 'goal', 'green', True, None],
        ['Go search for other victims and triage them', 'goal', None, True, None],
        ['Locate yellow victims now', 'goal', 'yellow', True, None],
        ['Locate green victims now', 'goal', 'green', True, None],
        ['Check for other casualities', 'goal', None, True, None],
        ['Can you save the injured ones quickly?', 'goal', None, True, None],
        ['Look for injured and save them', 'goal', None, True, None],
        ['Look for any injured victims', 'goal', None, True, None],
        ['Can you navigate to any people?', 'goal', None, True, None],
        ['Can you navigate to anyone?', 'goal', None, True, None],
        ['Keep an eye for other casualities', 'goal', None, True, None],
        ['Can you figure a way to save victims?', 'goal', None, True, None],
        ['Can you navigate to any person nearby?', 'goal', None, True, None],
        ["Search and rescue any victims found", 'goal', None, True, None],
        ['Go to the last victim', 'goal', None, False, None],
        ['Go to the previous victim', 'goal', None, False, None],
        ['Go to the already seen victim', 'goal', None, False, None],
        ['Go to the other victim', 'goal', None, False, None],
        ['Go to the next victim', 'goal', None, True, None],
        ['Go to a new victim now', 'goal', None, True, None],
        ['Go to another victim', 'goal', None, True, None],
        ['Go search for other injured ones', 'goal', None, True, None],
        ['Save the last victim you saw',  'goal', None, False, None],
        ['Save the previous victim you encountered',  'goal', None, False, None],
        ['Save the already spotted victims first',  'goal', None, False, None],
        ['Save the other victim', 'goal', None, False, None],
        ['Save the next victim', 'goal', None, True, None],
        ['Save a new victim', 'goal', None, True, None],
        ['Save another victim now', 'goal', None, True, None],
        ['Locate other affected persons', 'goal', None, True, None],
        ['Check if the person is okay', 'goal', None, True, None],
        ['Are there any person injured here? save them first', 'goal', None, True, None],
        ['Can you figure a way to save more victims?', 'goal', None, True, None],
    ],
    'key': [
        'Find a light switch', 
        'Locate a lever nearby', 
        'Turn on the switch', 
        'Go to a lever'
        'Figure out how to turn on the light', 
        'Check near the light switch', 
        'Look out for a light switch', 
        'Go to the nearest lever', 
        'Turn off the electricity', 
        'Examine this in light', 
        'Check if you can turn on the light', 
        'Careful, search for some light source.', 
        "How is the visibility? See if you can turn on the light.", 
    ],
    "lava": [
        "Check if there are any fires",
        "Find if any fires",
        "Search for fires if any",
        "Look for fires and any smokes",
        "Find if where is the smoke coming from",
        "Follow smoke to find if any fire in the place",
        "Check if there is anything burning",
        "Find an extinguisher",
        "Put off the flames",
        'See if there are any fires in the building', 
        'Extinguish the fire!',
        'Find a way through the fire.',
    ],
    "unseen":[
        "Search the area",
        "Explore this place",
        "Scan this space", 
        "Find more unexplored areas",
        "Search this place thoroughly",
        "Inspect the area",
        "Scan this place carefully",
        "Map the building",
        "Figure out the extent of damage",
        'Scout the area',
        "Quickly scan and see where help is needed first",
        'Minimize your passable frontiers',
        'Cover the area',
        'Maximize coverage',
    ]
}

breakpoint()

# demo_data = {
    # 'goal': ['Can you figure a way to save victims?'],
    # 'door': [
    #         'Check for other victims in the <spatial> room',
    #         'Look for doors',
    #         'Keep an eye for other doors',
    #         ],
    # 'key': [
    #         'Look out for a light switch',
    #         ],
    # 'lava': [
    #         'See if there are any fires in the building',
    #         ]
# }

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
    data is a list or numpy array with each entry as a ['instruction', 'target']
    """
    df = pd.DataFrame(data)
    df.to_csv(name, header=['instruction', 'target'], index=False) 
    return df

if __name__ == "__main__":
    # nlp = spacy.load('en_core_web_sm')
    train_list = get_datalist_from_dict(train_data) #@, nlp)
    train_df = to_csv(train_list, 'train_data.csv')
    test_list = get_datalist_from_dict(test_data) #, nlp)
    test_df = to_csv(test_list, 'test_data.csv')