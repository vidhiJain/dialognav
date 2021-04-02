from planner.queues import PriorityQueue
import numpy as np
import pdb

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
    # Pointing right (positive X)
    np.array((1, 0)),
]

DIR_TO_8_VEC = [
    # Down (positive Y)
    np.array((0, 1)),
    np.array((-1, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    np.array((-1, -1)),
    # Up (negative Y)
    np.array((0, -1)),
    np.array((1, -1)),
    # Pointing right (positive X)
    np.array((1, 0)),
    np.array((1, 1)),
]

class Node():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.g = None
        self.closed = False

    def __lt__(self, other):
        return self.g <= other.g

class AstarPlanner:
    def __init__(self, obs, gridConnection=4, incremental=False):
        """
        World Map should be a 2D array with:
        0 : free space
        1 : obstacles
        2 : frontiers to explore
        3 : current position.
        """
        self.incremental = incremental

        if self.incremental:
            self.worldMap = obs['image_fov']
        else:
            self.worldMap = obs['image_fov'][:,:,0]
            self.color = obs['image_fov'][:,:,1]
            self.state = obs['image_fov'][:,:,2]

        self.visitedGoals = np.zeros((self.worldMap.shape[0], self.worldMap.shape[1]))
        self.visitedGoalQueue = {}
        # self.state = obs['image'][:,:,2]
        self.rows = self.worldMap.shape[0] 
        self.cols = self.worldMap.shape[1]
        self.openList = PriorityQueue()
        self.openNodesList = {}
        self.epsilon = 1
        self.gridConnection = gridConnection
        
        if self.gridConnection == 4: 
            self.dx = [-1, 0, 1, 0]
            self.dy = [0, -1, 0, 1]
            self.dirToVec = [
                        # Pointing left (negative X)
                        np.array((-1, 0)),
                        # Up (negative Y)
                        np.array((0, -1)),
                        # Pointing right (positive X)
                        np.array((1, 0)),
                        # Down (positive Y)
                        np.array((0, 1)),
                    ]

        elif self.gridConnection == 8: 
            self.dx = [-1, -1, -1, 0, 0, 1, 1, 1]
            self.dy = [-1, 0, 1, -1, 1, -1, 0, 1]
            self.dirToVec = [
                        # Down (positive Y)
                        np.array((0, 1)),
                        np.array((-1, 1)),
                        # Pointing left (negative X)
                        np.array((-1, 0)),
                        np.array((-1, -1)),
                        # Up (negative Y)
                        np.array((0, -1)),
                        np.array((1, -1)),
                        # Pointing right (positive X)
                        np.array((1, 0)),
                        np.array((1, 1)),
                    ]
        
        self.maxSteps = 3
        self.steps = 0
        self.actionList = []

    def CalculateKey(self, x, y):
        return self.cols*x + y

    def CalculateCost(self, currentNode, neighborNode):
        differenceX = neighborNode.x - currentNode.x
        differenceY = neighborNode.y - currentNode.y

        if differenceX == 0 or differenceY == 0:
            return 1
        return 1.414

    def IsTerminal(self, node, goal, pointNav, previous):
        if pointNav and (node.x, node.y) in goal:
            return True
        
        elif(self.incremental):
            if not pointNav and (self.worldMap[node.x, node.y] == 0 or (self.worldMap[node.x, node.y] == 4)):
                return True
            
        else:
            if not pointNav and (self.worldMap[node.x, node.y] == 0 or (self.worldMap[node.x, node.y] == 4 and self.state[node.x][node.y] == 1)):
                return True

        return False

    def IsValidNeighbor(self, neighborX, neighborY, goal, pointNav):
        if (neighborX < 0 or neighborX >= self.rows): return False
        elif (neighborY < 0 or neighborY >= self.cols): return False
        
        if (self.worldMap[neighborX][neighborY] == 1 or self.worldMap[neighborX][neighborY] == 0 or self.worldMap[neighborX][neighborY] == 4): return True
        elif (not pointNav and self.worldMap[neighborX][neighborY] == goal[0]): return True 
        elif (pointNav and (neighborX,neighborY) in goal): return True
        return False

    def CalculatePath(self, goal, pointNav, previous):
        #set paraneters of start and possible goal positions.
        self.startPos = np.where(self.worldMap==10)
        self.startNode = Node(self.startPos[0][0], self.startPos[1][0])
        self.startNode.g = 0
        self.openList.put(self.startNode)
        self.openNodesList[self.CalculateKey(self.startNode.x, self.startNode.y)] = self.startNode
        while(not self.openList.empty()):
            currentNode = self.openList.pop()
            #if already popped, continue.
            if currentNode.closed:
                continue

            currentNode.closed = True
            if (self.IsTerminal(currentNode, goal, pointNav, previous)):
                #The path obtained here is in reverse order. The last coordinate in the path is the current position of the agent.
                path = self.BacktracePath(currentNode) 
                return path

            #shuffle through neigbors, and add to openlist
            for idx in range(len(self.dx)):
                neighborX = currentNode.x+self.dx[idx]
                neighborY = currentNode.y+self.dy[idx]
                neighborKey = self.CalculateKey(neighborX,neighborY)
                if self.IsValidNeighbor(neighborX, neighborY, goal, pointNav):
                    #if neighboring node has been visited before, retrieve it, update it's g value and push it in the open list.
                    if neighborKey in self.openNodesList:
                        neighborNode = self.openNodesList[neighborKey]
                        transitionCost = self.CalculateCost(currentNode, neighborNode)
                        if neighborNode.closed:
                            continue
                        if (neighborNode.g>currentNode.g+transitionCost):
                            neighborNode.g = currentNode.g+transitionCost
                            neighborNode.parent = currentNode
                            self.openList.put(neighborNode)

                    else:
                        neighborNode = Node(neighborX, neighborY)
                        transitionCost = self.CalculateCost(currentNode, neighborNode)
                        neighborNode.g = currentNode.g+transitionCost
                        neighborNode.parent = currentNode
                        self.openList.put(neighborNode)
                        self.openNodesList[neighborKey] = neighborNode

        #if we got here then path does not exist.
        path = []
        return path

    def BacktracePath(self, node):
        path = []
        while(node!=None):
            path.append((node.x, node.y))
            node = node.parent
        #the path obtained here is obtained by backtracking and is in reverse order. The last coordinate in the path is the current position of the agent.
        return path

    #malmo path to actions. Don't know if this works. 
    def get_cardinal_action_commands(self, yaw, path):
        curr_yaw = yaw%360
        action_list = []
        if(len(path)==0):
            return action_list
        curr_pos = path.pop()
        # goal_position = get_state_coord(solution_path.pop(0))
        while len(path) != 0:
            dest_pos = path.pop()
            # print("curr_pos, dest_pos: {} {}".format(curr_pos, dest_pos))
            diff_x = dest_pos[0] - curr_pos[0]
            diff_z = dest_pos[1] - curr_pos[1]
            # print("diff : ({}, {})".format(diff_x, diff_z))
            if diff_x == 1:
                dest_yaw = 270
                if(curr_yaw!=dest_yaw):
                    action_list.extend(self.get_turn_actions(curr_yaw, dest_yaw))
                action_list.append("moveeast 1")
            elif diff_x == -1:
                dest_yaw = 90
                if(curr_yaw!=dest_yaw):
                    action_list.extend(self.get_turn_actions(curr_yaw, dest_yaw))
                action_list.append("movewest 1")
            else:
                # print("no move in z direction")
                pass 

            if diff_z == 1:
                dest_yaw = 0
                if(curr_yaw!=dest_yaw):
                    action_list.extend(self.get_turn_actions(curr_yaw, dest_yaw))
                action_list.append("movesouth 1")
            elif diff_z == -1: 
                dest_yaw = 180
                if(curr_yaw!=dest_yaw):
                    action_list.extend(self.get_turn_actions(curr_yaw, dest_yaw))
                action_list.append("movenorth 1")
            else:
                pass
            curr_pos = dest_pos*1
        action_list.reverse()
        return action_list


    def get_turn_actions(self,curr_yaw, dest_yaw):
        '''
        curr_yaw, and dest_yaw are in %360
        '''
        curr_yaw = curr_yaw // 90
        dest_yaw = dest_yaw // 90
        # print("curr_yaw, dest_yaw: {} {}".format(curr_yaw, dest_yaw))
        yaw_diff = dest_yaw - curr_yaw

        actions = []
        
        if(yaw_diff==3 or yaw_diff==-3):
            yaw_diff = int((0 - yaw_diff) / 3)

        turn_dir = -1
        if(yaw_diff>0):
            turn_dir = 1

        for i in range(np.abs(yaw_diff)):
            actions.append("turn {}".format(turn_dir))
        # print("turn actions: {}".format(actions))
        actions = []
        return actions

    #minigrid path to actions. Does not work right now.
    def PathToAction(self, path, agentDirection, goal, pointNav):
        #the path obtained here is in reverse order. The last coordinate in the path is the current position of the agent.
        actionList = []
        currentPosition = path.pop()
        currentAgentDirection = agentDirection

        while len(path) != 0:
            nextPosition = path.pop()
            differenceX = nextPosition[0] - currentPosition[0]
            differenceY = nextPosition[1] - currentPosition[1]

            #deciding the next direction according to a 4 connected grid.
            if self.gridConnection == 4:
                if differenceX == 1:
                    nextAgentDirection = 2
                elif differenceX == -1:
                    nextAgentDirection = 0
                elif differenceY == 1:
                    nextAgentDirection = 3
                elif differenceY == -1:
                    nextAgentDirection = 1

                differenceAction = nextAgentDirection - currentAgentDirection
            
                #rotation according to a 4 connected grid.
                if (differenceAction == 1 or differenceAction == -3):
                    actionList.append(1)
                elif (differenceAction == -1 or differenceAction == 3):
                    actionList.append(0)
                elif (differenceAction == 2):
                    actionList.append(1)
                    actionList.append(1)
                elif (differenceAction == -2):
                    actionList.append(0)
                    actionList.append(0)

            elif self.gridConnection == 8:
                #deciding the next direction according to an 8 connected grid. Based on agent direction in visdialwrapper.
                if differenceX == -1 and differenceY == 1:
                    nextAgentDirection = 0
                elif differenceX == -1 and differenceY == 0:
                    nextAgentDirection = 1
                elif differenceX == -1 and differenceY == -1:
                    nextAgentDirection = 2
                elif differenceX == 0 and differenceY == -1:
                    nextAgentDirection = 3
                elif differenceX == 1 and differenceY == -1:
                    nextAgentDirection = 4
                elif differenceX == 1 and differenceY == 0:
                    nextAgentDirection = 5
                elif differenceX == 1 and differenceY == 1:
                    nextAgentDirection = 6
                elif differenceX == 0 and differenceY == 1:
                    nextAgentDirection = 7

                differenceAction = nextAgentDirection - currentAgentDirection

                # rotation according to a 8 connected grid.
                #right rotation by one turn
                if (differenceAction == 1 or differenceAction == -7):
                    actionList.extend([1])
                #right rotation by two turns
                elif (differenceAction == 2 or differenceAction == -6):
                    actionList.extend([1,1])
                #right rotation by three turns
                elif (differenceAction == 3 or differenceAction == -5):
                    actionList.extend([1,1,1])
                #left rotation by one turn
                elif (differenceAction == -1 or differenceAction == 7):
                    actionList.extend([0])
                #left rotation by two turn
                elif (differenceAction == -2 or differenceAction == 6):
                    actionList.extend([0,0])
                #left rotation by three turn
                elif (differenceAction == -3 or differenceAction == 5):
                    actionList.extend([0,0,0])            
                elif (differenceAction == 4):
                    actionList.extend([1,1,1,1])            
                elif (differenceAction == -4):
                    actionList.extend([0,0,0,0])

            #if in the next position, there is a door and it is closed, open the door.
            # if(self.incremental):
            #     if self.worldMap[nextPosition[0], nextPosition[1]] == 4:
            #         actionList.append(5)
            # else:
            if self.worldMap[nextPosition[0], nextPosition[1]] == 4 and self.state[nextPosition[0], nextPosition[1]] == 1:
                actionList.append(5)

            actionList.append(2)
            currentPosition = nextPosition
            currentAgentDirection = nextAgentDirection
        
        #if the plan found does not have a frontier at the end, we have reached the intended goal.
        if self.worldMap[currentPosition[0],currentPosition[1]] != 0 and self.worldMap[currentPosition[0],currentPosition[1]] != 4:
            actionList.pop()
            actionList.append(-1)

        #actionList right now is is the correct order. However, since we are popping an action everytime we call Act, I reversed the order below.
        actionList.reverse()

        #output of the function is action list in reverse order. We can pop the action one by one.
        return actionList

    def minigrid_actions_to_malmo(self, mini_actionList):
        actionList = []
        for mini_action in mini_actionList:
            if(mini_action==0):
                action = 'turn -1'
            elif(mini_action==1):
                action = 'turn 1'
            elif(mini_action==2):
                action = 'move 1'
            elif(mini_action==5):
                action = 'attack 1'
            elif(mini_action==-1):
                action = 'done'
            
            actionList.append(action)

        return actionList
    
    #should be changed to incorporate partial observability.
    def IntegrateMap(self, obs):
        if self.incremental:
            self.worldMap = obs['image_fov']
        else:
            self.worldMap = obs['image_fov'][:,:,0]
            self.color = obs['image_fov'][:,:,1]
            self.state = obs['image_fov'][:,:,2]
        return

    def SetPlannerGoal(self, goal, previous):
        if self.incremental:
            plannerGoal = set()
            candidateGoals = np.where(self.worldMap == goal[0])
            for idx in range(candidateGoals[0].size):
                candidateX = candidateGoals[0][idx]
                candidateY = candidateGoals[1][idx]

                if goal[1] != None and goal[2] != None:
                    plannerGoal.add((candidateX,candidateY))

            if len(plannerGoal) != 0:
                pointNav = True
            else:
                pointNav = False
                plannerGoal = tuple(goal)

        else:
            #set pointNav charecteristics.
            if len(goal) == 3: 
                #if we want to go to previously visited goals.
                if previous:
                    pointNav = True
                    plannerGoal = set()
                    if self.visitedGoalQueue:
                        visited = self.visitedGoalQueue.get(goal[0], None)
                        if visited is not None:
                            plannerGoal.add(visited[-1]) # -1 for immediatiately previous goal, TODO: more keys to detect the goal retrieval like first, second last, etc.

                #if we are in objectNav and the goal is visible in the latest map.
                #find all the candidate goals in the visible map and form a set.
                elif not previous:
                    plannerGoal = set()
                    candidateGoals = np.where(self.worldMap == goal[0])
                    for idx in range(candidateGoals[0].size):
                        candidateX = candidateGoals[0][idx]
                        candidateY = candidateGoals[1][idx]

                        # Does it keep track of list of important goals implicitly? like previous door even if it was not a commanded goal before.
                        if self.visitedGoals[candidateX][candidateY] == 1:
                            continue
                        if goal[1] != None and goal[2] != None and goal[1] == self.color[candidateX][candidateY] and goal[2] == self.state[candidateX][candidateY]:
                            plannerGoal.add((candidateX,candidateY))
                        elif goal[1] == None and goal[2] != None and goal[2] == self.state[candidateX][candidateY]:
                            plannerGoal.add((candidateX,candidateY))
                        elif goal[1] != None and goal[2] == None and goal[1] == self.color[candidateX][candidateY]:
                            plannerGoal.add((candidateX,candidateY))
                        elif goal[1] == None and goal[2] == None:
                            plannerGoal.add((candidateX,candidateY))

                if len(plannerGoal) != 0:
                    pointNav = True
                else:
                    pointNav = False
                    plannerGoal = tuple(goal)

            else:
                pointNav = True
                plannerGoal = set()
                plannerGoal.add(tuple(goal))

        return plannerGoal, pointNav

    def Act(self, goal, obs = None, yaw = 0, action_type = "minigrid", previous = False, immediate_actions=None):

        #set the goal
        if self.incremental: previous = False
        plannerGoal, pointNav = self.SetPlannerGoal(goal, previous)
        # breakpoint()
        #if partial observable, integrate the current map.
        if obs is not None:
            self.IntegrateMap(obs)

        #if we have already tken maxSteps with the last plan, get a new plan with most recent observations.
        if self.steps == self.maxSteps or not self.actionList:
            self.openList = PriorityQueue()
            self.openNodesList = {}
            #the path returned from self.CalculatePath is in reverse order. The last coordinate would be the current position of the agent.
            path = self.CalculatePath(plannerGoal, pointNav, previous)
            if len(path) == 0:
                if(action_type=="malmo"):
                    return -1, 'quit'
                else:
                    return -1
            self.steps = 0
            self.actionList = self.PathToAction(path, obs['direction'], plannerGoal, pointNav)
            
            if immediate_actions is not None:
                self.actionList += immediate_actions
            
            if(action_type=="malmo"):
                self.malmo_actionList = self.minigrid_actions_to_malmo(self.actionList)

        #the action list is in reverse order. We can pop the action list one by one. We do this till etiher the action list becomes empty or we take maximum number of steps.
        #if the returned action is -1, then it means we have reached the goal.
        self.steps += 1
        action = self.actionList.pop()

        #if we have reached the goal and it's not pointNav. 
        if action == -1:
            currentAgentPos = np.where(self.worldMap == 10) #current position of the agent.
            currentAgentDirection = obs['direction']
            goalX = currentAgentPos[0][0] + self.dirToVec[currentAgentDirection][0]
            goalY = currentAgentPos[1][0] + self.dirToVec[currentAgentDirection][1]
            self.visitedGoals[goalX][goalY] = 1
            if goal[0] not in self.visitedGoalQueue:
                self.visitedGoalQueue[goal[0]] = []
            self.visitedGoalQueue[goal[0]].append((goalX, goalY))
        if(action_type=="malmo"):
            malmo_action = self.malmo_actionList.pop()
            return action, malmo_action
        else:
            return action
