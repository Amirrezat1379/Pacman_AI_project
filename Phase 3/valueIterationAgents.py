# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        next_values = util.Counter()
        for iteration in range(0, self.iterations):
            for state in self.mdp.getStates():
                values = []
                if self.mdp.isTerminal(state):
                    values.append(0)

                for action in self.mdp.getPossibleActions(state):
                    values.append(self.computeQValueFromValues(state, action))

                next_values[state] = max(values)

            self.values = next_values.copy()


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        finalStates = self.mdp.getTransitionStatesAndProbs(state, action)
        weightedAverage = 0
        for finalState in finalStates:
            nextState = finalState[0]
            probability = finalState[1]
            weightedAverage += (probability * (self.mdp.getReward(state, action, nextState) + (self.discount * self.values[nextState])))

        return weightedAverage

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)

        if not actions:
            return None

        QValue = -1e10
        nextAction = None
        for action in actions:
            value = self.computeQValueFromValues(state, action)
            if value > QValue:
                QValue = value
                nextAction = action

        return nextAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        # initialize value function to all 0 values.
        for state in states:
            self.values[state] = 0

        numberOfStates = len(states)

        for i in range(self.iterations):
            stateIndex = i % numberOfStates
            state = states[stateIndex]

            terminal = self.mdp.isTerminal(state)
            if not terminal:
                action = self.getAction(state)
                QValue = self.getQValue(state, action)
                self.values[state] = QValue

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        preDict = {}
        for state in [s for s in states if not self.mdp.isTerminal(s)]:
            for action in self.mdp.getPossibleActions(state):
                for nextState in [pair[0] for pair in self.mdp.getTransitionStatesAndProbs(state, action)]:
                    if nextState not in preDict:
                        preDict[nextState] = set()
                    preDict[nextState].add(state)

        prioQueue = util.PriorityQueue()
        for state in states:
            if not self.mdp.isTerminal(state):
                maxQ = max([self.computeQValueFromValues(state, a) for a in self.mdp.getPossibleActions(state)])
                diff = abs(maxQ - self.getValue(state))
                prioQueue.update(state, -diff)

        for i in range(self.iterations):
            if prioQueue.isEmpty():
                break

            state = prioQueue.pop()
            if not self.mdp.isTerminal(state):
                maxQ = max([self.computeQValueFromValues(state, a) for a in self.mdp.getPossibleActions(state)])
                self.values[state] = maxQ
                for pred in preDict[state]:
                    maxQ = max([self.computeQValueFromValues(pred, a) for a in self.mdp.getPossibleActions(pred)])
                    diff = abs(maxQ - self.getValue(pred))
                    if diff > self.theta:
                        prioQueue.update(pred, -diff)
