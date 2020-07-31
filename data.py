# Data for future nearby locations
# TODO: encode questions for past context.

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
    ], 
    'goal': [
        'Find next victim', 
        'Go search for other victims',
        'Locate other victims',
        'Check for other casualities',
        'Have you seen any injured ones?',
        'Check for other victims in the next room', 
        'Look for any injured victims',
        'Can you locate any people?',
        'Can you locate anyone?',
        'Keep an eye for other casualities',
        'Can you figure a way to reach other victims?',
        'Can you locate any person nearby?',
    ],
    'key': [
        'Find a light switch',
        'Locate a lever nearby',
        'Turn on the switch',
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