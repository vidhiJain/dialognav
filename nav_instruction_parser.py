from typing import List
from transformers import pipeline
from candidates import *

spatial_attribute_detection_threshold = 0.50
color_attribute_detection_threshold = 0.50
# class ObjectAttributes:
#     def __init__(self, object):
#         self.objectname = object
#         self.color = None
#         self.size = None
#         self.state = None

#     def set_color(self, color):
#         self.color = color

#     def set_size(self, size):
#         self.size = size

#     def set_state(self, state):
#         self.state = state


# class Goal:
#     def __init__(self, object):
#         self.goal_object = object
#         self.goal_object_attr = ObjectAttributes(object)
#         self.spatial_relation = None
#         self.associated_object = None
#         self.associated_object_attr = None


class InstructionProcessor:
    def __init__(self):
        self.classifier = pipeline('zero-shot-classification')
        # self.candidate_goal = ["victim", "switch", "door"]
        # self.candidate_color = ["red", "green", "yellow", "white", "blue"]
        # self.candidate_time = ["last", "previous", "next"]
        # # self.candidate_preposition = ['inside', 'outside', 'before', 'after', 'over', 'below', 'under']


        self.OBJECT_TO_IDX = {
            'unseen'        : 0,
            'empty'         : 1,
            'wall'          : 2,
            'floor'         : 3,
            'door'          : 4,
            'switch'        : 5,
            'key'           : 5,
            'ball'          : 6,
            'box'           : 7,
            'goal'          : 8,
            'victim'        : 8,
            'lava'          : 9,
            'agent'         : 10,
            }

        self.COLOR_TO_IDX = {
            'red': 18,
            'white': 24, 
            'yellow': 35, 
            'green': 28,
            'brown': 34
        }

        self.candidate_goal = list(OBJECT_MAP.keys())
        self.candidate_color = list(self.COLOR_TO_IDX.keys())
        self.candidate_time = list(REQUIRES_HISTORY.keys())
        self.candidate_spatial = list(SPATIAL_BIAS.keys())

    def defineGoal(self, instruction): 
        goalObject = self.classifier(instruction, self.candidate_goal)
        colorObject = self.classifier(instruction, self.candidate_color)
        timeAttribute = self.classifier(instruction, self.candidate_time)
        spatialAttribute = self.classifier(instruction, self.candidate_spatial)

        goalIndex = goalObject['scores'].index(max(goalObject['scores']))
        colorIndex = colorObject['scores'].index(max(colorObject['scores']))
        timeIndex = timeAttribute['scores'].index(max(timeAttribute['scores']))
        spatialIndex = spatialAttribute['scores'].index(max(spatialAttribute['scores']))
        
        #defining goal
        goal = self.OBJECT_TO_IDX[OBJECT_MAP[goalObject['labels'][goalIndex]]]
        
        #defining color
        if colorObject['scores'][colorIndex] > color_attribute_detection_threshold: 
            color = self.COLOR_TO_IDX[colorObject['labels'][colorIndex]]
        else:
            color = None
        
        #defining if we want to go to a previously visited goal
        previous = REQUIRES_HISTORY[timeAttribute['labels'][timeIndex]]

        if spatialAttribute['scores'][spatialIndex] > spatial_attribute_detection_threshold:
            immediate_actions = SPATIAL_BIAS[spatialAttribute['labels'][spatialIndex]]
        else:
            immediate_actions = None
        #defining the goal tuple.
        goalTuple = [goal, color, None]

        return goalTuple, previous, immediate_actions


