
candidate_objects = ["door", "room", "corridor",
                    "injured", "victim", 
                    "light switch", "lever", "switch", "electric switch", 
                    "fire", 
                    "frontier"]

candidate_colors = ['green', 'yellow', 'red', 'white']

candidate_reference = ['this', 'that', 'same', 'different', 'other', 'another']

candidate_state = ['open', 'closed', 'triaged', 'wounded']

candidate_spatial_relation = ['left', 'left side',
                             'right', 'right side',
                             'front', 
                             'back']

candidate_time_relation = ['last', 'previous', 'before', 'found', 'the other', 'the', 
                    'next', 'new', 'another', 'a'
                ]

OBJECT_MAP = {
    'door': 'door',
    'room': 'door', 
    "corridor" : 'door', 

    'victim':  'goal',
    'injured': 'goal',
    'person': 'goal',

    'light switch': 'key',
    'lever': 'key',
    'switch': 'key',
    'electric switch': 'key',

    'fire': 'lava', 

    'frontier': 'unseen',
    # 'explore' : 'unseen',
}

REQUIRES_HISTORY = {
    'last': True, 
    'previous': True, 
    'already seen': True, 
    # 'found': True, 
    # 'the': True,
    'the other': True,
    'next' : False, 
    'new' : False,
    # 'a': False, 
    'another': False,
}

# 0: left, 1: right, 2: move, 5: toggle, -1:done
SPATIAL_BIAS = {
    'left': [0],
    'left side': [0],
    'right': [1],
    'right side': [1],
    'front': [2],
    'back': [0, 0],
    'stop': [-1],
}

CODE_TO_COMMON_MAP = {
    'goal': ['victim', 'injured', 'casualities', 'who may need help', 'people', 'affected'], 
    'key': ['switch', 'electric switch', 'lever', 'light switch'],
    'door': ['door', 'wooden door'],
    'lava': ['fire', 'hazard'],
    'unseen': ['frontier', 'explore'],
}

ACTION_MAP = {
    0: "Turning Left",
    1: "Turning Right",
    2: "Moving forward",
    3: "Done!", 
    4: "Turning back",
}
