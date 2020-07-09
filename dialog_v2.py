import spacy
import gym_minigrid
import random
import time
from run_planner_malmo import *

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
        self.ACTION_WORDS = {"go", "locate", "search", "find"}
        self.DESCRIPTION_WORDS = {"is", "are", "was", "list", "seen", "visible", "located", "found"}


        self.OBJECT_WORDS = {"victim": "victim", "door": "door", "key": "key",
            "victims": "victim", "keys": "key", "doors": "door"
        }
        self.OBJECT_DESCRIPTION_WORDS = {"color": 0, "state": 1}
        self.OBJECT_STATE_WORDS = {"open": 0, "close": 1}
        self.OBJECT_COLOR_WORDS = {"red": 0, "brown": 1, "green": 2, "blue": 3, "yellow": 4} 
        self.OTHER_WORDS = {"many", "all", "front"}
        self.LOCATION_WORDS = {"where", "location", "position", "seen"}

        self.RESPONSE_POS_WORDS = {"found", "located", "seen", "discovered"}
        self.RESPONSE_MATCH_WORDS = {"matching", "potential"}
        
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
        # Lower case everything
        text = text.lower()
        
        # Check for it and substitute with previous subject if found
        if "it" in text or "its" in text or "it's" in text:
            if self.prev_subject is not None:
                text = text.replace("it", self.prev_subject)
            else:
                return -1
        return text

    def process_dialog(self, text):
        if len(text) < 3:
            return ""
        text = self.adjust_text(text)
        if text == -1:
            return "Reference to `It` cannot be resolved!"

        doc = nlp(text)
        parseTree = self.parse_doc(doc)
        print("Parse Tree:\n", parseTree)
        print("------")

        items = self.get_items(parseTree.root)
        print("Items:\n", items)
        print("------")

        
        if items["type"] is None:
            return "Can not process the dialog!"

        response = self.gen_response(items, text)

        return response

    def navigation_command(self, parsed_dialog):
 
        response = ""

        object_name = parsed_dialog["object"]["name"]
        if object_name == "victim":
            goal_id = 8
            response += "victim"
        elif object_name == "door":
            goal_id = 4
            response += "door"
        elif object_name == "key":
            goal_id = 5
            response += "key" 
        
        minigrid_obs = self.env.gen_obs()
        done = False

        if self.malmo_agent is not None:
            world_state = self.malmo_agent.getWorldState()

            while not world_state.has_mission_begun:
                world_state = self.malmo_agent.getWorldState()
                print(".", end="")
                time.sleep(0.1)
            print()

        minigrid_action, action = self.agent.Act(goal_id, minigrid_obs, action_type="malmo")
        minigrid_obs = take_minigrid_action(self.env, minigrid_action, self.window)
        while action!="done":
            if self.malmo_agent is not None:
                if not world_state.is_mission_running:
                    break
                world_state = self.malmo_agent.getWorldState()
                while world_state.number_of_video_frames_since_last_state < 1:
                    time.sleep(0.05)
                    world_state = self.malmo_agent.getWorldState()
                world_state = self.malmo_agent.getWorldState()
                print("action: {}".format(action))
                self.malmo_agent.sendCommand(action)
            minigrid_action, action = self.agent.Act(goal_id, minigrid_obs, action_type="malmo")
            minigrid_obs = take_minigrid_action(self.env, minigrid_action, self.window)
            time.sleep(0.2)
        
        response += " was reached!"
        return response 

    def gen_response(self, parsed_dialog, dialog):

        def queries_location(dialog):
            for word in self.LOCATION_WORDS:
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

            # Answer the questions
            if queries_location(dialog):
                # Location of object
                if len(relevant_objects) > 1:
                    response += "{} {} {} were {} at coordinates [X, Y]\n{}".format(
                        len(relevant_objects),
                        random.sample(self.RESPONSE_MATCH_WORDS, 1)[0],
                        parsed_dialog["object"]["name"],
                        random.sample(self.RESPONSE_POS_WORDS, 1)[0],
                        ", ".join(str(i[0]) for i in relevant_objects)
                    )
                else:
                    response += "A {} was {} at coordinates [X, Y]\n{}".format(
                        parsed_dialog["object"]["name"],
                        random.sample(self.RESPONSE_POS_WORDS, 1)[0],
                        ", ".join(str(i[0]) for i in relevant_objects)
                    )
            else:
                print(relevant_objects)
                # Description of objects
                oneObj = relevant_objects[0][1]
                
                if parsed_dialog["object"]["desc"][0] == "color":
                    ans = oneObj.color
                elif parsed_dialog["object"]["desc"][0] == "state":
                    ans = oneObj.is_open
                response += "The {} of the  {} is {}".format(
                    parsed_dialog["object"]["desc"][0],
                    parsed_dialog["object"]["name"],
                    ans
                )
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

        def filter_obj(obj, queried_objs):
            for desc in obj["desc"]:
                # Filter color
                if desc in self.OBJECT_COLOR_WORDS:
                    queried_objs = filter_color(queried_objs, desc)
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
                #     print(dist, q_coord, c_coord)
                #     if not negate:
                #         if dist <= min_dist:
                #             filtered_objs.append([q_coord, q_obj])
                #         else:
                #             continue
                #     else:
                #         # Don't keep if any other specified object in close enough
                #         if dist <= min_dist:
                #             found = True
                #             break
                # if negate and (not found):
                #     filtered_objs.append([q_coord, q_obj])
                filtered_objs.sort(key=lambda x: x[2], reverse=negate)
            return filtered_objs

        mainObj = parsed_dialog["object"]
        queried_objs = None

        # Get objects based on visiblity condition
        if "front" in mainObj:
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
        
        return queried_objs

    def find_all_objects(self):
        grid, visibility_mask, observed_mask, tile_size = self.env.grid.grid, self.env.visible_grid, self.env.observed_absolute_map, self.env.tile_size

        w, h = visibility_mask.shape
        visibility_mask = visibility_mask.reshape(visibility_mask.size)
        observed_mask = observed_mask.reshape(observed_mask.size)

        observed_objects = {"door": [], "victim": [], "key": []}
        visible_objects = {"door": [], "victim": [], "key": []}
        for i, (obj, isVisible, wasObserved) in enumerate(zip(grid, visibility_mask, observed_mask)):
            coords = (i%w * tile_size + tile_size // 2, i//w * tile_size - tile_size // 2)
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

            if isinstance(obj, gym_minigrid.minigrid.Key) and obj not in visible_objects["key"]:
                if isVisible:
                    visible_objects["key"].append([coords, obj])
                if wasObserved:
                    observed_objects["key"].append([coords, obj])

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
        print(items)

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

            elif word in self.OBJECT_DESCRIPTION_WORDS or word in self.OBJECT_COLOR_WORDS \
                or word in self.OBJECT_STATE_WORDS or word in self.OTHER_WORDS:

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