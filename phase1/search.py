# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"    
    from util import Stack

    stack = Stack()

    visitedStates = []
    path = []

    # check if player is in goal state at first or not
    if problem.isGoalState(problem.getStartState()):
        return []
    else:
        stack.push((problem.getStartState(),[]))

        def givFinalDFS():

            # check that we have any state to go
            if stack.isEmpty():
                return []
            else:
                # returns us our current state
                currentState,path = stack.pop()
                visitedStates.append(currentState)

                # Check that if there is goal or not
                if problem.isGoalState(currentState):
                    return path
                else:
                    successor = problem.getSuccessors(currentState)

                    # new states will add here
                    if successor:
                        for item in successor:
                            i = 0
                            for visitedItem in visitedStates:
                                if (item[0] == visitedItem):
                                    i = 1
                            if (i == 1):
                                continue
                            newPath = path + [item[1]]
                            stack.push((item[0],newPath))
            return givFinalDFS()
        return givFinalDFS()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue

    queue = Queue()

    visitedStates = []
    path = []

    # check if player is in goal state at first or not
    if problem.isGoalState(problem.getStartState()):
        return []
    else:
        queue.push((problem.getStartState(),[]))

        while(True):

            # check that we have any state to go
            if queue.isEmpty():
                return []
            else:
                # returns us our current state
                currentState,path = queue.pop()
                visitedStates.append(currentState)

                # Check that if there is goal or not
                if problem.isGoalState(currentState):
                    return path
                else:
                    successor = problem.getSuccessors(currentState)

                    # new states will add here
                    if successor:
                        for item in successor:
                            i = 0
                            for visitedItem in visitedStates:
                                if (item[0] == visitedItem):
                                    i = 1
                            for state in queue.list:
                                if (item[0] == state[0]):
                                    i = 1
                            if (i == 1):
                                continue
                            newPath = path + [item[1]]
                            queue.push((item[0],newPath))

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    # queue: ((x,y),[path],priority) #
    queue = PriorityQueue()

    visitedStates = []

    # Check if initial state is goal state #
    if problem.isGoalState(problem.getStartState()):
        return []
    else:                                 #
        queue.push((problem.getStartState(), []), 0)

        while not queue.isEmpty():
            # Terminate condition: can't find solution #
            if queue.isEmpty():
                return []
            else:
                current = queue.pop()               
                if problem.isGoalState(current[0]):
                    return current[1]
                if current[0] not in visitedStates:
                    visitedStates.append(current[0])
                    for successor in problem.getSuccessors(current[0]):
                        if successor[0] not in visitedStates:
                            new_path = current[1]+[successor[1]]   
                            queue.push((successor[0], new_path), problem.getCostOfActions(new_path))
        if queue.isEmpty():
            return queue

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    # queue: ((x,y),[path],priority) #
    queue = PriorityQueue()

    visitedStates = []

    # Check if initial state is goal state #
    if problem.isGoalState(problem.getStartState()):
        return []

    queue.push((problem.getStartState(), [], 0), 0)

    while True:
        current = queue.pop()
        if problem.isGoalState(current[0]):
            return current[1]
        if current[0] not in visitedStates:
            visitedStates.append(current[0])
            for successor in problem.getSuccessors(current[0]):
                if successor[0] not in visitedStates:
                    cost = current[2] + successor[2]
                    totalCost = cost + heuristic(successor[0], problem)
                    queue.push((successor[0], current[1] + [successor[1]], cost), totalCost)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
