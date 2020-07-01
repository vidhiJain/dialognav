from queues import PriorityQueue
import numpy as np
import pdb

class Node():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.g = None
        self.cost = 1
        self.closed = False

    def __lt__(self, other):
        return self.g <= other.g

class astar_planner:
    def __init__(self, obs, gridConnection=8):
        """
        World Map should be a 2D array with:
        0 : free space
        1 : obstacles
        2 : frontiers to explore
        3 : current position.
        """
        self.worldMap = obs['image'][:,:,0]
        self.visitedGoals = np.zeros((self.worldMap.shape[0], self.worldMap.shape[1]))
        self.state = obs['image'][:,:,2]
        self.rows = self.worldMap.shape[0] 
        self.cols = self.worldMap.shape[1]
        self.openList = PriorityQueue()
        self.openNodesList = {}
        self.epsilon = 1
        self.gridConnection = gridConnection
        if self.gridConnection == 4: 
            self.dx = [-1, 0, 1, 0]
            self.dy = [0, -1, 0, 1]
        if self.gridConnection == 8: 
            self.dx = [-1, -1, -1, 0, 0, 1, 1, 1]
            self.dy = [-1, 0, 1, -1, 1, -1, 0, 1]
        self.maxSteps = 200
        self.steps = 0
        self.actionList = []

    def CalculateKey(self, x, y):
        return self.cols*x + y

    def IsTerminal(self, node, goal):
        if np.any(self.worldMap == goal) and self.worldMap[node.x,node.y] == goal and not self.visitedGoals[node.x][node.y]:
            self.visitedGoals[node.x][node.y] = 1 #mark a goal as visited if we have reached there.
            return True #planning completed if goal is in the observed map and we have reached that goal.
        # elif (self.worldMap[node.x, node.y]==0 or (self.worldMap[node.x, node.y]==4 and self.state[node.x, node.y] == 1)) : return True #planning completed if goal is not in the observed map and we have reached a frontier or unexplored part of the map.
        elif (self.worldMap[node.x, node.y]==0): return True
        return False

    def CalculatePath(self, goal):
        #set paraneters of start and possible goal positions.
        self.startPos = np.where(self.worldMap==10)
        self.startNode = Node(self.startPos[0][0], self.startPos[1][0])
        self.startNode.g = 0
        self.openList.put(self.startNode)
        self.openNodesList[self.CalculateKey(self.startNode.x, self.startNode.y)] = self.startNode
        self.goalPosLists = np.where(self.worldMap==2)

        while(not self.openList.empty()):
            currentNode = self.openList.pop()
            #if already popped, continue.
            if currentNode.closed:
                continue

            currentNode.closed = True
            if (self.IsTerminal(currentNode, goal)):
                #The path obtained here is in reverse order. The last coordinate in the path is the current position of the agent.
                path = self.BacktracePath(currentNode) 
                return path

            #shuffle through neigbors, and add to openlist
            for idx in range(len(self.dx)):
                neighborX = currentNode.x+self.dx[idx]
                neighborY = currentNode.y+self.dy[idx]
                neighborKey = self.CalculateKey(neighborX,neighborY)
                # pdb.set_trace()
                if (neighborX>0 and neighborX<self.rows and neighborY>0 and neighborY<self.cols and (self.worldMap[neighborX][neighborY] == 1 or self.worldMap[neighborX][neighborY] == 4 or self.worldMap[neighborX][neighborY] == goal)):
                    #if neighboring node has been visited before, retrieve it, update it's g value and push it in the open list.
                    if neighborKey in self.openNodesList:
                        neighborNode = self.openNodesList[neighborKey]
                        if neighborNode.closed:
                            continue
                        if (neighborNode.g>currentNode.g+neighborNode.cost):
                            neighborNode.g = currentNode.g+neighborNode.cost
                            neighborNode.parent = currentNode
                            self.openList.put(neighborNode)

                    else:
                        neighborNode = Node(neighborX, neighborY)
                        neighborNode.g = currentNode.g+neighborNode.cost
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
        # path.reverse()
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
            diff_x = dest_pos[0] - curr_pos[0]
            diff_z = dest_pos[1] - curr_pos[1]
            if diff_z == 1:
                dest_yaw = 270
                if(curr_yaw!=dest_yaw):
                    action_list.extend(self.get_turn_actions(curr_yaw, dest_yaw))
                action_list.append("moveeast 1")
            elif diff_z == -1:
                dest_yaw = 90
                if(curr_yaw!=dest_yaw):
                    action_list.extend(self.get_turn_actions(curr_yaw, dest_yaw))
                action_list.append("movewest 1")
            else:
                # print("no move in z direction")
                pass 

            if diff_x == -1:
                dest_yaw = 0
                if(curr_yaw!=dest_yaw):
                    action_list.extend(self.get_turn_actions(curr_yaw, dest_yaw))
                action_list.append("movesouth 1")
            elif diff_x == 1: 
                dest_yaw = 180
                if(curr_yaw!=dest_yaw):
                    action_list.extend(self.get_turn_actions(curr_yaw, dest_yaw))
                action_list.append("movenorth 1")
            else:
                pass
            curr_pos = dest_pos*1
        return action_list

    def get_turn_actions(self,curr_yaw, dest_yaw):
        '''
        curr_yaw, and dest_yaw are in %360
        '''
        curr_yaw = curr_yaw // 90
        dest_yaw = dest_yaw // 90
        yaw_diff = dest_yaw - curr_yaw

        actions = []
        
        if(yaw_diff==3 or yaw_diff==-3):
            yaw_diff = int((0 - yaw_diff) / 3)

        turn_dir = -1
        if(yaw_diff<0):
            turn_dir = 1

        for i in range(np.abs(yaw_diff)):
            actions.append("turn {}".format(turn_dir))

        return actions

    #minigrid path to actions. Does not work right now.
    def PathToAction(self, path, agentDirection, goal):
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
                    nextAgentDirection = 0
                elif differenceX == -1:
                    nextAgentDirection = 2
                elif differenceY == 1:
                    nextAgentDirection = 1
                elif differenceY == -1:
                    nextAgentDirection = 3

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
            if self.worldMap[nextPosition[0], nextPosition[1]] == 4 and self.state[nextPosition[0], nextPosition[1]] == 1:
                actionList.append(5)

            actionList.append(2)
            currentPosition = nextPosition
            currentAgentDirection = nextAgentDirection
        
        #if we had found the plan till the goal, append a -1 at the end. This is used as done.
        if self.worldMap[currentPosition[0],currentPosition[1]] == goal:
            actionList.append(-1)

        #actionList right now is is the correct order. However, since we are popping an action everytime we call Act, I reversed the order below.
        actionList.reverse()

        #output of the function is action list in reverse order. We can pop the action one by one.
        return actionList

    #should be changed to incorporate partial observability.
    def IntegrateMap(self, obs):
        self.worldMap = obs['image'][:,:,0]
        self.state = obs['image'][:,:,2]
        return

    def Act(self, goal, obs=None, yaw =0, action_type="minigrid"):
        #if partial observable, integrate the current map.
        if(action_type=="malmo"):
            if(obs['direction']==0):
                yaw = 90
            elif(obs['direction']==1):
                yaw = 0
            elif(obs['direction']==2):
                yaw = 270
            elif(obs['direction']==3):
                yaw = 180

        if obs is not None:
            self.IntegrateMap(obs)

        #if we have already tken maxSteps with the last plan, get a new plan with most recent observations.
        if self.steps == self.maxSteps or not self.actionList:
            print("replanning")
            self.openList = PriorityQueue()
            self.openNodesList = {}
            #the path returned from self.CalculatePath is in reverse order. The last coordinate would be the current position of the agent.
            path = self.CalculatePath(goal)
            print('path ', path)
            self.steps = 0
            if(action_type=="minigrid"):
                self.actionList = self.PathToAction(path, obs['direction'], goal)
            elif(action_type=="malmo"):
                self.actionList = self.get_cardinal_action_commands(yaw, path)
                return self.actionList
            # pdb.set_trace()

        #the action list is in reverse order. We can pop the action list one by one. We do this till etiher the action list becomes empty or we take maximum number of steps.
        #if the returned action is -1, then it means we have reached the goal.
        self.steps += 1
        return self.actionList.pop()
        # return self.actionList
        
if __name__ == '__main__':
    worldMap = np.array([[0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,2,0,0,2,0,0],
                         [0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,2,0,0,1,0,0,0],
                         [0,0,0,0,0,0,1,0,0,0],
                         [0,0,0,0,0,1,1,0,0,0],
                         [0,0,0,2,0,1,0,0,0,0],
                         [0,0,0,2,0,1,1,1,0,3]])

    planner = astar_planner(worldMap)
    path = planner.CalculatePath()
    print(path)
