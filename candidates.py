
candidate_labels = ["door", "room", 
                    "injured", "victim", # "person", 
                    "light switch", "lever", "switch", "electric switch", 
                    "fire", 
                    "explore"]

OBJECT_MAP = {
    'door': 'door',
    'room': 'door', 
    
    'victim':  'goal',
    'injured': 'goal',
    'person': 'goal',

    'light switch': 'key',
    'lever': 'key',
    'switch': 'key',
    'electric switch': 'key',

    'fire': 'lava', 

    'explore': 'unseen',
}

CODE_TO_COMMON_MAP = {
    'goal': ['victim', 'injured', 'casualities', 'who may need help', 'people', 'affected'], 
    'key': ['switch', 'electric switch', 'lever', 'light switch'],
    'door': ['door'],
    'lava': ['fire', 'hazard'],
    'unseen': ['explore'],
}

ACTION_MAP = {
    0: "Turning Left",
    1: "Turning Right",
    2: "Moving forward",
    3: "Done!", 
    4: "Turning back",
}
