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

import heapq
import util
import node
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
    
    fringe = util.Stack()
    n = node.Node(problem.getStartState())

    if problem.isGoalState(n.state): return n.total_path()
    fringe.push(n)

    expanded = set()

    while not fringe.isEmpty():
        n = fringe.pop()
        expanded.add(n.state)
        for state, action, cost in problem.getSuccessors(n.state):
            new_node = node.Node(state, n, action, cost)
            if new_node.state not in expanded and new_node not in fringe.list:
                if problem.isGoalState(state): return new_node.total_path()
                fringe.push(new_node)

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    n = node.Node(problem.getStartState())
    expanded = set()
    expanded.add(n.state)
    
    fringe = util.Queue()
    fringe.push(n)

    while not fringe.isEmpty():
        current = fringe.pop()
        expanded.add(current.state)
        if problem.isGoalState(current.state): return current.total_path()
        for state, action, cost in problem.getSuccessors(current.state):
            new_node = node.Node(state, current, action, cost)
            if new_node.state not in expanded and new_node not in fringe.list:
                fringe.push(new_node)
                expanded.add(new_node.state)

                
        

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    fringe = util.PriorityQueue()
    n = node.Node(problem.getStartState())
    fringe.push(n, 0)
    expanded = dict()

    while not fringe.isEmpty():
        current = fringe.pop()
        if problem.isGoalState(current.state): return current.total_path()
        expanded[current.state] = current
        for state, action, cost in problem.getSuccessors(current.state):
            new_node = node.Node(state, current, action, cost+current.cost)
            if state not in expanded:
                fringe.update(new_node, current.cost) 

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    fringe = util.PriorityQueue()
    n = node.Node(problem.getStartState())
    fringe.push(n, heuristic(n.state,problem))
    expanded = dict()
    
    while not fringe.isEmpty():
        current = fringe.pop()
        if problem.isGoalState(current.state): return current.total_path()
        expanded[current.state] = current
        for state, action, cost in problem.getSuccessors(current.state):
            new_node = node.Node(state, current, action, cost+current.cost)
            if state not in expanded:
                fringe.update(new_node, current.cost+heuristic(new_node.state,problem)) 


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
