"""
    Purpose:
        1. Dialog based navigation
        2. Dialog based question answering
"""

import numpy as np
import planner 
import gym_minigrid


discovered =  {"previous_subject": None, "victim": {}, "door": {}, "key": {}}

def start_dialog(env):
    observations = env.reset()
    action_planner = planner.astar_planner(observations)

    while True:
        ip = input(">> ").lower().strip()
        if ip == "quit":
            break
        response = process_dialog(ip, action_planner, observations)
        print(response)
    print("Done")

def process_dialog(text, planner, observations, env):
    # Identify dialog type
    if("go to" in text):
        response = follow_nav_command(text, planner, observations, env)
    else:
        dialog_type = get_dialog_type(text)
        response = gen_response(dialog_type, env)
        if "V" in dialog_type:
            discovered["previous_subject"] = "victim"
        elif "D" in dialog_type:
            discovered["previous_subject"] = "door"
        elif "K" in dialog_type:
            discovered["previous_subject"] = "key"
    return response

def follow_nav_command(text, planner, observations, env):
    done = False
    response = ""
    if ("victim" in text):
        goal = 8
        response = "victim"
    elif("door" in text):
        goal = 4
        response = "door"
    elif("key" in text):
        goal = 5
        response = "key"

    actionList = planner.Act(obs=observations, goal=goal, action_type="minigrid")
    
    if(len(actionList)==0):
        response += " does not exist in the current env"
        return response

    while len(actionList):
        action = actionList.pop()
        obs, reward, done, info = env.step(action)

def get_dialog_type(text):
    dialog_type = ""

    # Check for it
    if "it" in text or "its" in text or "it's" in text:
        if discovered['previous_subject'] is not None:
            text = text.replace("it", discovered['previous_subject'])
        else:
            return ""
    print("get_dialog - text:" ,text)

    # Victim
    if "victim" in text or "victims" in text:
        dialog_type += "V"
        if "found" in text or "located" in text or "seen" in text or "location" in text :
            dialog_type += "1"
        if "color" in text:
            dialog_type += "2"
        if "all" in text or "how many" in text:
            dialog_type += "3"
        if "front" in text or "visible" in text:
            if "1" not in dialog_type: dialog_type += "1"
            dialog_type += "4"
    
    # Door
    if "door" in text or "doors" in text:

        dialog_type += "D"
        if "found" in text or "located" in text or "seen" in text or "location" in text :
            dialog_type += "1"
        if "color" in text:
            dialog_type += "2"
        if "all" in text or "how many" in text:
            dialog_type += "3"
        if "front" in text or "visible" in text:
            if "1" not in dialog_type: dialog_type += "1"
            dialog_type += "4"
        if "state" in text:
            dialog_type += "5"

    # Key
    if "key" in text or "keys" in text:
        dialog_type += "K"
        if "found" in text or "located" in text or "seen" in text or "location" in text :
            dialog_type += "1"
        if "color" in text:
            dialog_type += "2"
        if "all" in text or "how many" in text:
            dialog_type += "3"
        if "front" in text or "visible" in text:
            if "1" not in dialog_type: dialog_type += "1"
            dialog_type += "4"
    return dialog_type

def gen_response(dialog_type, env):
    ans = ""
    observed_objects, visible_objects = find_objects(env.grid.grid, env.visible_grid, env.observed_absolute_map, env.tile_size)

    # Victim
    if "V" in dialog_type:

        # Regardin the victim only
        if ("D" not in dialog_type) and ("K" not in dialog_type):
            # Search problem for one victim
            # Ex. Where was the last victim found
            if "1" in dialog_type and "3" not in dialog_type:
                # Visibility mask
                if "4" in dialog_type:
                    if len(visible_objects["victim"]) == 0:
                        return "No victim is visible."
                    x, y = visible_objects["victim"][0][0]
                else:
                    if len(observed_objects["victim"]) == 0:
                        return "No victims discovered yet!"
                    x, y = observed_objects["victim"][-1][0]
                rno = np.random.randint(0, 3)
                if rno == 0:
                    ans = "Found a victim at ({}, {}).".format(x, y)
                elif rno == 1:
                    ans = "Victim was found at ({}, {}).".format(x, y)
                else :
                    ans = "Victim seen at coordinates ({}, {}).".format(x, y)

            # Victim color
            # ex. What color is the victim
            elif "2" in dialog_type:
                if len(observed_objects["victim"]) == 0:
                    return "No victims discovered yet!"
                color =  observed_objects["victim"][-1][1].color 
                rno = np.random.randint(0, 3)
                if rno == 0:
                    ans = "Victim is {} in color".format(color)
                elif rno == 1:
                    ans = "{} is the color of the victim".format(color)
                elif rno == 2:
                    ans = "Color {}".format(color)
                else:
                    ans = "{}".format(color)
        
            # Search problem for all vitims
            # ex. Where all were the victims founds
            elif "1" in dialog_type and "3" in dialog_type:
                # Visibility mask
                if "4" in dialog_type:
                    if len(visible_objects["victim"]) == 0:
                        return "No victim is visible."
                    coords = [ob[0] for ob in visible_objects["victim"]]
                else:
                    if len(observed_objects["victim"]) == 0:
                        return "No victims discovered yet!"
                    coords = [ob[0] for ob in observed_objects["victim"]]
                rno = np.random.randint(0, 3)
                coords = ["({}, {})".format(x, y) for x, y in coords]
                if rno == 0:
                    ans = "Victim were found at " + ", ".join(coords)
                else:
                    ans = "List of all victims that were found " + ", ".join(coords)
    
        elif "D" in dialog_type:
                # victim close to door
                if len(observed_objects["door"]) == 0:
                    return "No door discovered close to the victim"

                xd, yd = observed_objects["victim"][-1][0]
                closest_door = None
                min_dist = 1e32
                for obj in observed_objects['door']:
                    x, y = obj[0]
                    dist = ((x - xd)**2 + (y - yd)**2) ** (1/2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_door = obj[0]
                
                rno = np.random.randint(0, 3)
                if rno == 0:
                    ans = "The closest door is located at ({}, {})".format(*closest_door)
                elif rno == 1:
                    ans = "Door located at ({}, {}) is the closest".format(*closest_door)
                elif rno == 2:
                    ans = "({}, {}) is the location of the closest door".format(*closest_door)
                else:
                    ans = "Door at ({}, {})".format(*closest_door)

        elif "K" in dialog_type:
                # victim close to door
                if len(observed_objects["key"]) == 0:
                    return "No door discovered close to the victim"

                xd, yd = observed_objects["victim"][-1][0]
                closest_door = None
                min_dist = int('inf')
                for obj in observed_objects['key']:
                    x, y = obj[0]
                    dist = ((x - xd)**2 + (y - yd)**2) ** (1/2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_door = obj[0]
                
                rno = np.random.randint(0, 3)
                if rno == 0:
                    ans = "The closest key is located at ({}, {})".format(*closest_door)
                elif rno == 1:
                    ans = "key located at ({}, {}) is the closest".format(*closest_door)
                elif rno == 2:
                    ans = "({}, {}) is the location of the closest key".format(*closest_door)
                else:
                    ans = "key at ({}, {})".format(*closest_door)

    elif "D" in dialog_type:
        if "1" in dialog_type and "3" not in dialog_type:
            # Search problem for one door
            # Ex. Where was the last door found
            # Visibility mask
            if "4" in dialog_type:
                if len(visible_objects["door"]) == 0:
                    return "No door is visible."
                x, y = visible_objects["door"][0][0]
            else:
                if len(observed_objects["door"]) == 0:
                    return "No door discovered yet!"
                x, y = observed_objects["door"][-1][0]
            rno = np.random.randint(0, 3)
            if rno == 0:
                ans = "Found a door at ({}, {}).".format(x, y)
            elif rno == 1:
                ans = "Door was found at ({}, {}).".format(x, y)
            else :
                ans = "Door seen at coordinates ({}, {}).".format(x, y)

        elif "2" in dialog_type:
            # Victim color
            # ex. What color is the victim
            if len(observed_objects["door"]) == 0:
                return "No door discovered yet!"
            rno = np.random.randint(0, 3)
            if rno == 0:
                ans = "Color of the door brown"
            elif rno == 1:
                ans = "Brown"
            else :
                ans = "Looks to be brown in color"
            
        elif "1" in dialog_type and "3" in dialog_type:
            # Search problem for all Doors
            # ex. Where all were the Door founds
            # Visibility mask
            if "4" in dialog_type:
                if len(visible_objects["door"]) == 0:
                    return "No door is visible."
                coords = [ob[0] for ob in visible_objects["door"]]
            else:
                if len(observed_objects["door"]) == 0:
                    return "No doors discovered yet!"
                coords = [ob[0] for ob in observed_objects["door"]]
            rno = np.random.randint(0, 3)
            coords = ["({}, {}),".format(x, y) for x, y in coords]
            if rno == 0:
                ans = "Doors were found at " + ", ".join(coords)
            else:
                ans = "List of all Doors that were found " + ", ".join(coords)

        elif "4" in dialog_type:
            # apply visibility mask
            # ex. Is there a door in front
            pass
        elif "5" in dialog_type:
            if len(visible_objects["door"]) == 0:
                return "No door discovered yet!"
            rno = np.random.randint(0, 3)
            door_obj = visible_objects["door"][-1][1]
            door_state = "Open" if door_obj.is_open else "Closed"
            if rno == 0:
                ans = "The state of the door is {}".format(door_state)
            elif rno == 1:
                ans = "Door is in an {} state".format(door_state)
            else:
                ans = "The door is {}".format(door_state)

    elif "K" in dialog_type:
        if "1" in dialog_type and "3" not in dialog_type:
            # Search problem for one key
            # Ex. Where was the last key found
            # Visibility mask
            if "4" in dialog_type:
                if len(visible_objects["key"]) == 0:
                    return "No key is visible."
                x, y = visible_objects["key"][0][0]
            else:
                if len(observed_objects["key"]) == 0:
                    return "No keys discovered yet!"
                x, y = observed_objects["key"][-1][0]
            rno = np.random.randint(0, 3)
            if rno == 0:
                ans = "Found a key at ({}, {}).".format(x, y)
            elif rno == 1:
                ans = "key was found at ({}, {}).".format(x, y)
            else :
                ans = "key seen at coordinates ({}, {}).".format(x, y)

        elif "2" in dialog_type:
            # key color
            # ex. What color is the key
            if len(observed_objects["key"]) == 0:
                    return "No key discovered yet!"
            rno = np.random.randint(0, 3)
            if rno == 0:
                ans = "Color of the key Yellow"
            elif rno == 1:
                ans = "Yellow"
            else :
                ans = "Looks to be brown in Yellow"
        elif "1" in dialog_type and "3" in dialog_type:
            # Search problem for all keys
            # ex. Where all were the keys founds
            # Visibility mask
            if "4" in dialog_type:
                if len(visible_objects["key"]) == 0:
                    return "No key is visible."
                coords = [ob[0] for ob in visible_objects["key"]]
            else:
                if len(observed_objects["key"]) == 0:
                    return "No keys discovered yet!"
                coords = [ob[0] for ob in observed_objects["key"]]
            rno = np.random.randint(0, 3)
            coords = ["({}, {}),".format(x, y) for x, y in coords]
            if rno == 0:
                ans = "Keys were found at " + ", ".join(coords)
            else:
                ans = "List of all Keys that were found " + ", ".join(coords)

        elif "4" in dialog_type:
            # apply visibility mask
            # ex. Is there a victim in front
            pass
    
    else:
        # raise NotImplementedError("Invalid in dialog_type: ", dialog_type)
        a = np.random.randint(0, 4)
        if a == 0:
           ans = "Don't know!"
        elif a == 1:
            ans = "Can't say!"
        elif a == 2:
            ans = "I don't know!"
        else:
            ans = "Can not answer!"
    
    return ans

def find_objects(grid, visibility_mask, observed_mask, tile_size):
    w, h = visibility_mask.shape
    visibility_mask = visibility_mask.reshape(visibility_mask.size)
    observed_mask = observed_mask.reshape(observed_mask.size)

    observed_objects = {"door": [], "victim": [], "key": []}
    visible_objects = {"door": [], "victim": [], "key": []}
    for i, (obj, isVisible, wasObserved) in enumerate(zip(grid, visibility_mask, observed_mask)):
        coords = (i%w * tile_size + tile_size // 2, i//w * tile_size - tile_size // 2)
        if isinstance(obj, gym_minigrid.minigrid.Goal) or isinstance(obj, gym_minigrid.minigrid.Box) and obj not in visible_objects["victim"]:
            if isVisible:
                visible_objects["victim"].append([coords, obj])
            if wasObserved:
                observed_objects["victim"].append([coords, obj])
            discovered["victim"][coords] = obj

        if isinstance(obj, gym_minigrid.minigrid.Door) and obj not in visible_objects["door"]:
            if isVisible:
                visible_objects["door"].append([coords, obj])
            if wasObserved:
                observed_objects["door"].append([coords, obj])
            discovered["door"][coords] = obj

        if isinstance(obj, gym_minigrid.minigrid.Key) and obj not in visible_objects["key"]:
            if isVisible:
                visible_objects["key"].append([coords, obj])
            if wasObserved:
                observed_objects["key"].append([coords, obj])
            discovered["key"][coords] = obj

    return observed_objects, visible_objects