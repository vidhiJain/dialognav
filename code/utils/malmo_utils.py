def getMissionXML(mission_file):
    """
    Returns the mission xml in python string 
    """
    with open(mission_file, 'r') as f:
        print("Loading mission from %s" % mission_file)
        mission_xml = f.read()
    return mission_xml


def inview2D_with_opaque_objects(grid, yaw, distance, angle=60):
    """
    Using polar coordinates to return the blocks in the 
    field of view of minecraft player

    Parameters:
        the grid and player's yaw, 
        line of sight distance, and 
        angle for field of view
    Returns: 
        visible grid: 2D numpy array

    Note: 
    envsize should be same as the grid size.
    Assuming that the player has a headlight with them
    This function is independent of environment's visibility

    Depends on function `is_opaque`: 
        to check the item is opaue or not.
    """


    envsize = grid.shape[0]
    visible_grid = np.zeros((envsize, envsize)) 
    agent_pos = {'z': envsize//2, 'x':envsize//2}
    for theta_in_deg in range(int(yaw-angle), int(yaw+angle)):
        # print('theta_in_deg', theta_in_deg)
        for r in range(1, int(distance)):
            theta_in_rad = np.pi*theta_in_deg/180
            z = int(r*np.cos(theta_in_rad)) + agent_pos['z']
            x = - int(r*np.sin(theta_in_rad)) + agent_pos['x']
            # debug
            # print('grid[z][x]', z, x, grid[z][x])
            if is_opaque(grid[z][x]):
                visible_grid[z][x] = 1
                break
            else:
                visible_grid[z][x] = 1
    return visible_grid


def get_cardinal_action_commands(solution_path):
    """
    Parameter:
    solution_path: list of the nodes 
    from goal position to start position 
    represented as strings

    Returns a list of cardinal movement actions 
    (for Minecraft Malmo sendCommand: movenorth 1, etc.)
    """

    action_list = []
    initial_position = get_state_coord(solution_path.pop())
    # goal_position = get_state_coord(solution_path.pop(0))
    while len(solution_path) != 0:
        current_position = get_state_coord(solution_path.pop())
        difference_z = current_position[0] - initial_position[0]
        difference_x = current_position[1] - initial_position[1]
        if difference_z == 1:
            action_list.append("movesouth 1")
        elif difference_z == -1:
            action_list.append("movenorth 1")
        else:
            # print("no move in z direction")
            pass 

        if difference_x == 1:
            action_list.append("moveeast 1")
        elif  difference_x == -1:
            action_list.append("movewest 1")
        else:
            pass
        initial_position = current_position

    return action_list



# def get_relative_action_commands(solution_path):
#     """
#     TODO: 
#     Parameter:
#     solution_path: list of the nodes 
#     from goal position to start position 
#     represented as strings

#     Returns a list of relative movement actions 
#     (for Minecraft Malmo sendCommand: turn 1, turn -1, move 1)
#     """

#     action_list = []
#     initial_position = get_state_coord(solution_path.pop())
#     # goal_position = get_state_coord(solution_path.pop(0))
#     while len(solution_path) != 0:
#         current_position = get_state_coord(solution_path.pop())
#         difference_z = current_position[0] - initial_position[0]
#         difference_x = current_position[1] - initial_position[1]
#         if difference_z == 1:
#             # action_list.append("movesouth 1")
#             action_list.append("turn 1")
#         elif difference_z == -1:
#             # action_list.append("movenorth 1")
#         else:
#             # print("no move in z direction")
#             pass 

#         if difference_x == 1:
#             # action_list.append("moveeast 1")
#         elif  difference_x == -1:
#             # action_list.append("movewest 1")
#         else:
#             pass
#         initial_position = current_position

#     # Turn for the final action instead of move
#     # last_action = action_list.pop()
#     # difference_z = current_position[0] - goal_position[0]
#     # difference_x = current_position[1] - goal_position[1]
#     # if difference_z == 1:
#     #     action_list.append("")
    
#     # LOOK each time into the direction if you move to it?

#     return action_list


