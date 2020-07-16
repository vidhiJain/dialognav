import spacy
import random
import time
import sys
import os


import gym_minigrid
from gym_minigrid.index_mapping import COLOR_TO_IDX, STATE_TO_IDX, malmo_object_to_index as MALMO_OBJECT_TO_INDEX
from utils import *


nlp = spacy.load("en_core_web_sm")


class ParseTree():
    def __init__(self, root):
        self.root = root
    
    def __str__(self):
        def navigate(p, children, i=0):
            print(p, children, i)
            t = ""
            for c in children:
                s = "\n" + " " * i + "|"
                s += " -> " + c.token
                s += navigate(c, c.children, i + 1)
                t += s
            return t

        s = str(self.root)
        s += navigate(self.root, self.root.children)
        return s
    
    def __repr__(self):
        return self.__str__()

class ParseNode():
    def __init__(self, TOKEN, POS, DEP):
        self.token = TOKEN
        self.pos = POS
        self.dep = DEP
        self.children = []

    def __str__(self):
        return self.token
    
    def __repr__(self):
        return self.__str__()



class DialogProcessing():

    def __init__(self, agent=None, env=None, malmo_agent=None, window=None):
        self.agent = agent
        self.env = env
        self.malmo_agent = malmo_agent
        self.window = window

        self.ACCEPTED_POS = set(["NOUN", "ADP", "ADJ", "VERB", "ADV"])
        self.ACTION_WORDS = {"go", "locate", "search", "find", "move", "turn"}
        self.DESCRIPTION_WORDS = {"is", "are", "was", "list", "seen", "visible", "located", "found", "location", "position"}


        self.OBJECT_WORDS = {"victim": "victim", "door": "door", "switch": "switch",
            "victims": "victim", "switches": "switch", "doors": "door"
        }
        self.OBJECT_DESCRIPTION_WORDS = {"color": 0, "state": 1}
        self.OBJECT_STATE_WORDS = {"open": 0, "close": 1}
        self.OBJECT_COLOR_WORDS = {"red": 0, "brown": 1, "green": 2, "blue": 3, "yellow": 4} 
        self.OTHER_WORDS = {"many", "all", "front"}
        self.LOCATION_WORDS = {"where", "location", "position", "seen", "all", "located"}
        self.PLURAL_WORDS = {"doors", "switches", "victims"}

        self.RESPONSE_POS_WORDS = {"found", "located", "seen", "discovered"}
        self.RESPONSE_MATCH_WORDS = {"matching", "potential"}
        self.COLOR_MAP = {"yellow2": "blue", "inv_indianyellow": "yellow", "inv_blue": "blue"}
        self.COLOR_MAP_INV = {"blue": "yellow2", "yellow": "inv_indianyellow"}# "blue": "inv_blue"}
        
        self.prev_subject = None

    def parse_doc(self, doc):
        # Find the ROOT
        for token in doc:
            if token.dep_ == "ROOT":
                break
        root = ParseNode(token.text, token.pos_, token.dep_)
        parseTree = ParseTree(root)

        def parse(nodes, children):
            for c in nodes:
                if c.pos_ in self.ACCEPTED_POS:
                    node = ParseNode(c.text, c.pos_, c.dep_)
                    children.append(node)
                    parse(c.children, node.children)
                else:
                    parse(c.children, children)
        
        parse(token.children, parseTree.root.children)
        return parseTree

    def adjust_text(self, text):
        def locate_word(words, word):
            try:
                return words.index(word)
            except:
                return None

        # Lower case everything
        words = text.lower().split(" ")
        
        i = locate_word(words, "it")
        if i is None: i = locate_word(words, "its")

        # Check for it and substitute with previous subject if found
        if i is not None:
            if self.prev_subject is not None:
                words[i] = self.prev_subject
            else:
                return -1
        return " ".join(words)

    def process_dialog(self, text):
        if len(text) < 3:
            return ""
        text = self.adjust_text(text)
        if text == -1:
            return "Reference to `It` cannot be resolved!"

        doc = nlp(text)
        parseTree = self.parse_doc(doc)

        items = self.get_items(parseTree.root)
        items["text"] = text
        print(parseTree)
        print(items)
        
        if (items["type"] is None) or ("object" not in items) or (items["object"]["name"] not in self.OBJECT_WORDS):

            if items["type"] is None:
                return "Malformed query"
            elif "to" in items["items"]:
                pass
            elif ("object" not in items):
                if "items" in items and len(items["items"]) > 0:
                    return "Item {} is not valid".format(items["items"][0])
            elif items["object"]["name"] not in self.OBJECT_WORDS:
                return "{} is not a valid object".format(items["object"]["name"])
            else:
                return "Malformed query"
        else:
            self.prev_subject = items["object"]["name"]

        response = self.gen_response(items, text)

        return response


    def get_object_id(self, object_name):
        if object_name == "victim":
            goal_id = OBJECT_TO_IDX["goal"]
        elif object_name == "door":
            goal_id = OBJECT_TO_IDX["door"]
        elif object_name == "switch":
            goal_id = OBJECT_TO_IDX["key"]
            # response += "switch"
        return goal_id

    def navigation_command(self, parsed_dialog):
        
        def get_coords(words):
            hasCoords = False
            x, y = None, None
            try: x = int(words[-2])
            except: pass
            try: y = int(words[-1])
            except: pass
            if (x is not None) and (y is not None):
                hasCoords = True
            return hasCoords, (x, y)

        response = ""


        words = parsed_dialog["text"].split(" ")
        hasCoords, coords = get_coords(words)

        if not hasCoords:
            object_name = parsed_dialog["object"]["name"]
            obj_id = self.get_object_id(object_name)
            state = None
            color = None
            desc = None

            if "object" in parsed_dialog:
                desc = None if len(parsed_dialog["object"]["desc"]) == 0 else parsed_dialog["object"]["desc"][0]
            if desc is not None:
                if desc in self.OBJECT_COLOR_WORDS:
                    color = COLOR_TO_IDX[desc]
                elif desc in self.OBJECT_STATE_WORDS:
                    state = STATE_TO_IDX[desc]
            
            goal_id = (obj_id, color, state)
            response += object_name
        else:
            goal_id = coords
            response += str(coords)
        
        minigrid_obs = self.env.gen_obs()
        done = False
        
        print(goal_id)
        # If malmo env 
        if self.malmo_agent is not None:
            world_state = self.malmo_agent.getWorldState()

            while not world_state.has_mission_begun:
                world_state = self.malmo_agent.getWorldState()
                print(".", end="")
                time.sleep(0.1)

        # minigrid_action, action = self.agent.Act(goal_id, minigrid_obs, action_type="malmo")
        # minigrid_obs = take_minigrid_action(self.env, minigrid_action, self.window)
        action = 0
        while action!="done":
            # If malmo env
            if self.malmo_agent is not None:
                if not world_state.is_mission_running:
                    break
                world_state = self.malmo_agent.getWorldState()
                while world_state.number_of_video_frames_since_last_state < 1:
                    time.sleep(0.05)
                    world_state = self.malmo_agent.getWorldState()
                
                world_state = self.malmo_agent.getWorldState()
            minigrid_action, action = self.agent.Act(goal_id, minigrid_obs, action_type="malmo")

            # If malmo env 
            if self.malmo_agent is not None:
                self.malmo_agent.sendCommand(action)
            
            minigrid_obs = take_minigrid_action(self.env, minigrid_action, self.window)
            time.sleep(0.5)
        
        response += " was reached!"
        return response 

    def gen_response(self, parsed_dialog, dialog):

        def queries_location(dialog):
            for word in self.LOCATION_WORDS:
                if word in dialog: return True
            return False

        def queries_multiple(dialog):
            for word in self.PLURAL_WORDS:
                if word in dialog: return True
            return False
        # def queries_description(dialog):
        #     for word in self

        # Navigation Response
        if parsed_dialog['type'] == "action":
            response = self.navigation_command(parsed_dialog)
        else:
            # Dialog response
            response = ""
            observed_objects, visible_objects = self.find_all_objects()

            relevant_objects = self.get_relevant_objects(parsed_dialog, observed_objects, visible_objects)

            if len(relevant_objects) == 0:
                response += "No such {} seen!".format(parsed_dialog["object"]["name"])
                return response

            # Answer the questions
            if queries_location(dialog):
                # Location of object
                if queries_multiple(dialog):
                    response += "{} {} {} {} {} at coordinates [X, Y]\n{}".format(
                        len(relevant_objects),
                        random.sample(self.RESPONSE_MATCH_WORDS, 1)[0],
                        parsed_dialog["object"]["name"],
                        "was" if len(relevant_objects) == 1 else "were",
                        random.sample(self.RESPONSE_POS_WORDS, 1)[0],
                        ", ".join(str(i[0]) for i in relevant_objects)
                    )
                else:
                    response += "A {} was {} at coordinates [X, Y]\n{}".format(
                        parsed_dialog["object"]["name"],
                        random.sample(self.RESPONSE_POS_WORDS, 1)[0],
                        relevant_objects[0][0]
                    )
            elif len(parsed_dialog["object"]["desc"]) != 0:
                
                # Description of objects
                oneObj = relevant_objects[0][1]
                ans = None
                if parsed_dialog["object"]["desc"][0] == "color":
                    ans = self.COLOR_MAP[oneObj.color]
                elif parsed_dialog["object"]["desc"][0] == "state":
                    if parsed_dialog["object"]["name"] == "door":
                        ans = "open" if oneObj.is_open else "closed"

                    elif parsed_dialog["object"]["name"] == "swtich":
                        return "Need to modify minigrid!"
                        #ans = "on" if oneObj.is_open else "off"

                if ans is None:
                    return "{} does not have an attribute {}".format(parsed_dialog["object"]["name"], parsed_dialog["object"]["desc"][0])
                
                response += "The {} of the {} is {}!".format(
                    parsed_dialog["object"]["desc"][0],
                    parsed_dialog["object"]["name"],
                    ans
                )
            elif len(relevant_objects):
                response += "{} {} {} at coordinates [X, Y]\n{}".format(
                    len(relevant_objects),
                    random.sample(self.RESPONSE_POS_WORDS, 1)[0],
                    parsed_dialog["object"]["name"],
                    ", ".join(str(i[0]) for i in relevant_objects)
                )
            else:
                response += "Not enough info!" 
        return response
    
    def get_relevant_objects(self, parsed_dialog, obs_objs, vis_objs):

        def filter_color(q_objs, desc, negate=False):
            filtered_objs = []
            for coords, obj in q_objs:
                if not negate:
                    if obj.color == desc:
                        filtered_objs.append([coords, obj])
                else:
                    if obj.color != desc:
                        filtered_objs.append([coords, obj])
            return filtered_objs

        def filter_state(q_objs, desc, negate=False):
            filtered_objs = []
            if desc == "open":
                desc = True
            elif desc == "close":
                desc = False
            else:
                return q_objs

            for coords, obj in q_objs:
                if not negate:
                    if obj.is_open == desc:
                        filtered_objs.append([coords, obj])
                else:
                    if obj.is_open != desc:
                        filtered_objs.append([coords, obj])
            return filtered_objs

        def filter_obj(obj, queried_objs):
            for desc in obj["desc"]:
                # Filter color
                if desc in self.COLOR_MAP_INV:
                    queried_objs = filter_color(queried_objs,  self.COLOR_MAP_INV[desc])
                # Filter state
                elif desc in self.OBJECT_STATE_WORDS:
                    queried_objs = filter_state(queried_objs, desc)
            return queried_objs

        def filter_closeness(q_objs, c_objs, min_dist=3, negate=False):
            filtered_objs = []
            get_dist = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
            for q_coord, q_obj in q_objs:
                found = False
                for c_coord, c_obj in c_objs:
                    # Skip filtering if same object
                    if q_coord == c_coord:
                        continue
                    # Get distance
                    dist = get_dist(q_coord, c_coord)
                    filtered_objs.append([q_coord, q_obj, dist])
                filtered_objs.sort(key=lambda x: x[2], reverse=negate)
            filtered_objs = [[x[0], x[1]] for x in filtered_objs]
            return filtered_objs
        
        def unique_filter(q_objs):
            seen_set = set()
            filtered_objs = []
            for obj_dets in q_objs:
                if obj_dets[0] not in seen_set:
                    seen_set.add(obj_dets[0])
                    filtered_objs.append(obj_dets) 
            return filtered_objs


        mainObj = parsed_dialog["object"]
        queried_objs = None

        # Get objects based on visiblity condition
        if "front" in mainObj["desc"]:
            queried_objs = vis_objs[mainObj["name"]]
        else:
            queried_objs = obs_objs[mainObj["name"]]

        # Apply filters to queries objects based on object description
        queried_objs = filter_obj(mainObj, queried_objs)

        close_objs = []
        # Filter if close object specified
        if "near" in parsed_dialog:
            nearObj = parsed_dialog["near"]
            close_objs = obs_objs[nearObj["name"]]

            # Now filter near objects
            close_objs = filter_obj(nearObj, close_objs)

            # Reduce queries objects based on closeness with near objects
            queried_objs = filter_closeness(queried_objs, close_objs, min_dist=3, negate=False)
        # Else filter with agent location
        elif len(queried_objs) != 0:
            agent_coords = self.env.unwrapped.agent_pos
            queried_objs.sort(key=lambda x: abs(agent_coords[0] - x[0][0]) + abs(agent_coords[1] - x[0][1]))
        
        # Filter our repeated objects
        queried_objs = unique_filter(queried_objs)

        return queried_objs

    def find_all_objects(self):
        grid, visibility_mask, observed_mask, tile_size = self.env.grid.grid, self.env.visible_grid, self.env.observed_absolute_map, self.env.tile_size

        w, h = visibility_mask.shape
        visibility_mask = visibility_mask.reshape(visibility_mask.size)
        observed_mask = observed_mask.reshape(observed_mask.size)

        observed_objects = {"door": [], "victim": [], "switch": []}
        visible_objects = {"door": [], "victim": [], "switch": []}
        for i, (obj, isVisible, wasObserved) in enumerate(zip(grid, visibility_mask, observed_mask)):
            coords = (i%w * tile_size + tile_size // 2, i//w * tile_size + tile_size // 2)#(i%w, i//w)#
            if isinstance(obj, gym_minigrid.minigrid.Goal) and obj not in visible_objects["victim"]:
                if isVisible:
                    visible_objects["victim"].append([coords, obj])
                if wasObserved:
                    observed_objects["victim"].append([coords, obj])

            if isinstance(obj, gym_minigrid.minigrid.Door) and obj not in visible_objects["door"]:
                if isVisible:
                    visible_objects["door"].append([coords, obj])
                if wasObserved:
                    observed_objects["door"].append([coords, obj])

            if isinstance(obj, gym_minigrid.minigrid.Key) and obj not in visible_objects["switch"]:
                if isVisible:
                    visible_objects["switch"].append([coords, obj])
                if wasObserved:
                    observed_objects["switch"].append([coords, obj])

        return observed_objects, visible_objects

    def get_items(self, node):
        parsed_dialog = {}

        # Determine type of dialog
        if node.token in self.ACTION_WORDS:
            parsed_dialog['type'] = "action"
        elif node.token in self.DESCRIPTION_WORDS:
            parsed_dialog['type'] = "description"
        else:
            # Invalid query
            parsed_dialog['type'] = None
            return parsed_dialog
        parsed_dialog['root'] = node.token
        
        def get_words(children, item_list):
            for c in children:
                item_list.append(c.token)
                get_words(c.children, item_list)

        items = []
        get_words(node.children, items)

        # Determine the object and its propoerties
        for word in items:
            if word in self.OBJECT_WORDS:
                word = self.OBJECT_WORDS[word]
                if "object" not in parsed_dialog: 
                    parsed_dialog["object"] = {"name": word, "desc": []}
                elif "object" in parsed_dialog and parsed_dialog["object"]["name"] == "":
                    parsed_dialog["object"]["name"] = word
                elif "near" in parsed_dialog and parsed_dialog["near"]["name"] == "":
                    parsed_dialog["near"]["name"] = word
                elif "near" not in parsed_dialog:
                    parsed_dialog["near"] = {"name": word, "desc": []}
                else:
                    print("Parsed Items", items)
                    raise ValueError("Invalid case!")

            elif word in self.OBJECT_DESCRIPTION_WORDS or word in self.COLOR_MAP_INV \
                or word in self.OBJECT_STATE_WORDS or word in self.OTHER_WORDS or word in self.OBJECT_COLOR_WORDS:

                if "object" not in parsed_dialog:
                    # Main object not encountered yet
                    parsed_dialog["object"] = {"name": "", "desc": []}

                if "near" not in parsed_dialog:
                    # this description belong to main object
                    parsed_dialog["object"]["desc"].append(word)
                else:
                    # This word describes the next object
                    parsed_dialog["near"]["desc"].append(word)

        parsed_dialog["items"] = items

        return parsed_dialog
    
