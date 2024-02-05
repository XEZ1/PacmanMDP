# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import game
import util


class MDPAgent(Agent):

    def __init__(self):
        """
        Initializes the MDPAgent with default values and essential configurations.
        """
        # Basic configurations
        self.directions = {
            Directions.NORTH: (0, 1),
            Directions.SOUTH: (0, -1),
            Directions.EAST: (1, 0),
            Directions.WEST: (-1, 0)
        }

    def registerInitialState(self, state):
        """
        Updates the agent's understanding of the game environment based on the given state.
        This includes Pacman's position, map layout, food, capsules, and ghost information.
        """
        print "Starting up MDPAgent!"
        # Update the map layout
        is_small_board = (max(x for x, _ in api.walls(state)) + 1) < 9 and \
                         (max(y for _, y in api.walls(state)) + 1) < 9

        # Some of the values have been taken from various papers I came across in the internet, I lost the particular one
        # that explained why some of these values are the best to use (like the gamma facotr and some of other rewards,
        # but I feel like it's important to mention that I didn't come up with some of these values fully myself.
        # One of the papers I used is the following:
        # https://leyankoh.wordpress.com/2017/12/14/an-mdp-solver-for-pacman-to-navigate-a-nondeterministic-environment/
        self.gameElements = {
            'walls': set(api.walls(state)),
            'grid_size': (max(x for x, _ in api.walls(state)) + 1, max(y for _, y in api.walls(state)) + 1),
            'direction_probabilities': api.directionProb,
            'is_big_board': not is_small_board,
            'is_small_board': is_small_board,
            'discount': 0.95 if not is_small_board else 0.6,
            'amount_of_iterations': 20 if not is_small_board else 40,
            'threshold': 0.001,
            'reward': 0,
            'ghost_reward': -2.5,
            'food_reward': 1,
            'capsules_reward': 3,
            'distance_ghost_avoid': 3,
            'time_ghost_avoid': 5,
            'distance_ghost_panic': 2,
            'distance_capsules_hunt': 3,
        }
        self.updateGameState(state)

    def updateGameState(self, state):
        """
        Updates the dynamic components of the game state including Pacman's position,
        food locations, capsules, and ghost information.
        """
        self.gameElements.update({
            'pacman': api.whereAmI(state),
            'food': set(api.food(state)),
            'capsules': set(api.capsules(state)),
            'ghosts': api.ghosts(state),
            'ghostStates': api.ghostStates(state),
            'edibleGhosts': api.ghostStatesWithTimes(state),
            'legal_moves': [action for action in api.legalActions(state) if action != Directions.STOP],
            'utilities': {
                (i, j): self.gameElements['reward']
                for i in range(1, self.gameElements['grid_size'][0] - 1)
                for j in range(1, self.gameElements['grid_size'][1] - 1)
                if (i, j) not in self.gameElements['walls']
            },
            'prev_utilities': {},
        })

        # Run value iteration to update the utilities of each state
        self.convergeUtilities(state)

    def getAction(self, state):
        """
        Determines the best move for Pacman based on the current utilities of the states.
        This method considers the utilities of the immediate next positions and chooses
        the direction that leads to the state with the highest utility, avoiding walls and ghosts.

        Returns:
        The optimal direction for Pacman to move next.
        """
        # Update the game state with current information
        self.updateGameState(state)

        # Initialize variables to find the best action
        best_action = Directions.STOP
        highest_utility = -float('inf')

        # Evaluate the utility of each possible next position for legal moves
        for action in self.gameElements['legal_moves']:
            dx, dy = self.directions.get(action, (0, 0))
            next_position = (self.gameElements['pacman'][0] + dx, self.gameElements['pacman'][1] + dy)

            # Skip walls or ghosts & consider the current state as fallback if they are at those positions
            next_position = self.gameElements['pacman'] if next_position in self.gameElements['walls'] or next_position \
                                                           in self.gameElements['ghosts'] else next_position

            utility = self.gameElements['utilities'].get(next_position, -float('inf'))

            # Choose the action that leads to the state with the highest utility
            if utility > highest_utility:
                highest_utility, best_action = utility, action

        # self.printMap(state)

        # Return the best action
        return api.makeMove(best_action if best_action is not None else Directions.STOP,
                            list(self.gameElements['legal_moves']))

    def final(self, state):
        """
        Called at the end of each game.
        """
        print 'Final game'

    def calculateReward(self, state):
        """
        Calculates the reward for a given state based on the proximity and state of the nearest ghost,
        presence of capsules, and whether the state is a food location. Adjusts the reward calculation
        based on the size of the map (small or medium).

        Args:
        state: The current state for which to calculate the reward.

        Returns:
        reward: The calculated reward for the given state.
        """
        # Extract information about the nearest ghost
        distance_to_ghost, closest_ghost = self.calculateClosestGhost(state)
        # Calculate the distance from Pacman to the closest ghost
        distance_to_pacman = util.manhattanDistance(closest_ghost, self.gameElements['pacman'])
        # Get the index of the closest ghost
        ghost_idx = list(self.gameElements['ghosts']).index(closest_ghost)
        # Calculate the remaining time of the ghost being edible
        time_left = self.gameElements['edibleGhosts'][ghost_idx][1]
        # Reset the reward
        reward = 0

        if self.gameElements['is_small_board']:
            for ghost in self.gameElements['ghosts']:
                trap_area = self.findTrapArea(state)
                central_point = (int(self.gameElements['grid_size'][0] / 2), int(self.gameElements['grid_size'][1] / 2))
                if state == central_point and state in trap_area:
                    # Case1: This would be a terminal state where the only food left is at 3,3 hence we don't care
                    # about the ghost and its trap anymore
                    # Case2: Not a terminal state yet, so we have to account for the ghost and its trap area
                    # Hence we go there only if it is safe
                    # Case3: It is not a terminal state and the ghost is not on a safe distance from the agent hence we
                    # cannot risk it and we have to avoid it
                    reward += 10 if ((len(self.gameElements['food']) == 1) and (3, 3) in self.gameElements['food']) or \
                                    (ghost[0] == 1 or ghost[0] == 2) else -10

        # Add reward for food
        reward += self.gameElements['food_reward'] if state in self.gameElements['food'] else 0

        # Add penalty for ghosts in proximity
        reward += self.gameElements['ghost_reward'] if state in self.gameElements['ghosts'] and time_left < \
                                                       self.gameElements['time_ghost_avoid'] else 0

        # Adjust reward for capsules in mediumClassic
        if self.gameElements['is_big_board'] and state in self.gameElements['capsules']:
            reward += self.gameElements['capsules_reward'] + 1 if distance_to_pacman < \
            self.gameElements['distance_capsules_hunt'] and self.gameElements['ghostStates'][ghost_idx][1] == 0 else 1

        # Calculate reward based on proximity to the nearest ghost and its state
        if self.gameElements['is_small_board']:
            reward += (2 * self.gameElements['ghost_reward']) / distance_to_ghost if distance_to_ghost \
                                                                    < self.gameElements['distance_ghost_panic'] else 0
        else:  # big grid
            if distance_to_ghost < self.gameElements['distance_ghost_avoid']:
                reward += self.gameElements['ghost_reward'] / distance_to_ghost if time_left < \
                    self.gameElements['time_ghost_avoid'] else -self.gameElements['ghost_reward'] / distance_to_ghost

        return reward

    def calculateUtility(self, state):
        """
        Computes the utility of a given state by considering the potential utility of moving
        in each possible direction from the current state, including the effects of directional probability.

        Args:
        state: The current state for which to compute the utility.

        Returns:
        The highest utility value achievable from the given state.
        """
        max_utility = -float('inf')

        # Assess each possible direction from the current state
        for direction, (dx, dy) in self.directions.items():
            next_state = (state[0] + dx, state[1] + dy) if (state[0] + dx, state[1] + dy) not in \
                                                           self.gameElements['walls'] else state

            self.gameElements['prev_utilities'][next_state] = 1 if self.gameElements['prev_utilities'][next_state] == 0 \
                else self.gameElements['prev_utilities'][next_state]

            # This idea of calculating the utility using probabilities & the complimentary ones is taken from the
            # following paper: https://leyankoh.wordpress.com/2017/12/14/an-mdp-solver-for-pacman-to-navigate-a-nondeterministic-environment/
            # and from Russel and Norvig book
            utility = self.gameElements['direction_probabilities'] * self.gameElements['prev_utilities'][next_state] + \
                      ((1 - self.gameElements['direction_probabilities']) * (self.gameElements['prev_utilities'][next_state])) \
                      if direction in [Directions.RIGHT[direction], Directions.LEFT[direction]] else \
                      self.gameElements['direction_probabilities'] * self.gameElements['prev_utilities'][next_state]

            # Update max utility if this direction offers a better utility
            max_utility = max(max_utility, utility)

        return max_utility

    def convergeUtilities(self, state):
        """
        Performs value iteration to compute the optimal policy for Pacman. The method iteratively updates the utilities of each state in the map based on the expected rewards and the discounted utilities of future states.
        """
        for _ in range(self.gameElements['amount_of_iterations']):
            self.gameElements['prev_utilities'] = self.gameElements['utilities'].copy() \
                if self.gameElements['prev_utilities'] != self.gameElements['utilities'] else self.gameElements[
                'prev_utilities']
            has_converged = True

            for state in self.gameElements['prev_utilities']:
                # Calculate the updated value of the state based on bellman equation
                updated_value = self.calculateReward(state) + self.gameElements['discount'] * self.calculateUtility(
                    state)

                # Check if the update is within the convergence threshold
                if abs(self.gameElements['utilities'].get(state, 0) - updated_value) > self.gameElements['threshold']:
                    has_converged = False

                self.gameElements['utilities'][state] = updated_value

            # check for convergence
            if has_converged:
                break

    # This is the function used to find the optimal policy that pacman can use
    # to move around the map to eat the food.
    # This function updates the map with the appropriate values for each state (or position/square)
    def calculateClosestGhost(self, state):
        """
        Retrieves information about the closest ghost to the given state (position).

        Args:
        state: The current state or position for which we are finding the closest ghost.

        Returns:
        distance: The Manhattan distance to the closest ghost from the state.
        """

        # Initialize the closest ghost and its distance with default values
        closest_ghost = None
        closest_distance = float('inf')  # Using 'inf' to represent an arbitrarily large distance

        # Iterate through each ghost to find the closest one
        for ghost in self.gameElements['ghosts']:
            distance = util.manhattanDistance(ghost, state)
            closest_ghost, closest_distance = (ghost, distance) if distance < closest_distance else \
                (closest_ghost, closest_distance)

        # Early return if no ghost is found
        if closest_ghost is None:
            return -1, 0

        return closest_distance if closest_distance != 0 else 1, closest_ghost

    def findTrapArea(self, state):
        """
        Identifies the trap area based on food locations and the maximum A* distance from the ghost.
        The trap area is defined as the food positions offering the maximum A* distance from Pacman to the ghost.
        """
        _, closest_ghost = self.calculateClosestGhost(state)
        max_distance = -1
        trap_area = []
        central_point = (int(self.gameElements['grid_size'][0] / 2), int(self.gameElements['grid_size'][1] / 2))

        # Iterate over food positions to find the area with the maximum A* distance as
        # this is most likely to be the trap
        for food_pos in self.gameElements['food']:
            distance = self.findDistance(food_pos, closest_ghost, state)
            if distance > max_distance:
                max_distance = distance
                trap_area = [food_pos]
            elif distance == max_distance:
                trap_area.append(food_pos)
        if central_point not in self.gameElements['walls']:
            trap_area.append(central_point)

        return trap_area

    def findTrapAreaAndAdjacentLocation(self, state):
        """
        Identifies the trap area based on food locations and the maximum A* distance from the ghost.
        Also identifies the entry points to the trap area.
        """
        _, closest_ghost = self.calculateClosestGhost(state)
        max_distance = -1
        trap_area = []
        entry_points = []

        # Calculate A* distances for each food position and find the maximum distance
        for food_pos in self.gameElements['food']:
            distance = self.findDistance(food_pos, closest_ghost, state)
            if distance > max_distance:
                max_distance = distance
                trap_area = [food_pos]
            elif distance == max_distance:
                trap_area.append(food_pos)

        # For each position in the trap area, find adjacent positions as potential entry points
        for trap_pos in trap_area:
            for dx, dy in self.directions.values():
                adjacent_pos = (trap_pos[0] + dx, trap_pos[1] + dy)
                if adjacent_pos not in self.gameElements['walls'] and adjacent_pos not in trap_area:
                    entry_points.append(adjacent_pos)

        return trap_area + list(set(entry_points))  # Remove duplicates in entry_points

    def aStarSearchDistance(self, start, goal, state):
        """
        Performs A* search from start to goal without using heapq.

        Args:
        start: The starting state.
        goal: The goal state.

        Returns:
        The actual shortest distance between the start and goal states considering the walls in between.
        """

        def heuristic(position, goal):
            return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

        def getSuccessors(position):
            successors = []
            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                x, y = position
                dx, dy = game.Actions.directionToVector(action)
                nextx, nexty = int(x + dx), int(y + dy)
                if (nextx, nexty) not in self.gameElements['walls']:
                    successors.append((nextx, nexty))
            return successors

        openList = [(heuristic(start, goal), start)]  # Using tuple (F-score, Node)
        closedList = set()
        gScore = {start: 0}

        while openList:
            # Sort the list based on F-score and pop the node with lowest F-score
            openList.sort(key=lambda x: x[0])
            current_fscore, current = openList.pop(0)

            if current == goal:
                return gScore[current]

            closedList.add(current)

            for successor in getSuccessors(current):
                tentative_gScore = gScore[current] + 1
                if successor in closedList and tentative_gScore >= gScore.get(successor, float('inf')):
                    continue

                if tentative_gScore < gScore.get(successor, float('inf')) or successor not in [item[1] for item in
                                                                                               openList]:
                    gScore[successor] = tentative_gScore
                    fScore = tentative_gScore + heuristic(successor, goal)
                    openList.append((fScore, successor))

        return float('inf')

    def findDistance(self, start, end, state):
        """
        A helper function to execute the A* search
        Finds the actual shortest distance considering walls.
        """
        if isinstance(end, list) and end:
            end = end[0]  # Assuming end is a list of tuples, take the first one
        return self.aStarSearchDistance(start, end, state)

    def printMap(self, state):
        """
        Prints the current state of the game map, including walls, food, capsules, Pacman,
        and ghosts, along with the utility values of each cell, formatted to two decimal places.
        """
        walls = self.gameElements['walls']
        food = self.gameElements['food']
        capsules = self.gameElements['capsules']
        pacman = self.gameElements['pacman']
        ghosts = self.gameElements['ghosts']

        for y in range(self.gameElements['grid_size'][1] - 1, -1, -1):
            row = ""
            for x in range(self.gameElements['grid_size'][0]):
                cell = ""
                if (x, y) in food and (x, y) in ghosts:
                    cell = "<'"
                elif (x, y) in walls:
                    cell = "#"
                elif (x, y) == pacman:
                    cell = "@"
                    utility = self.gameElements['utilities'].get((x, y), 0)
                    cell += "{:.2f}".format(utility)
                elif (x, y) in food:
                    cell = "'"
                    utility = self.gameElements['utilities'].get((x, y), 0)
                    cell += "{:.2f}".format(utility)
                elif (x, y) in capsules:
                    cell = "''"
                    utility = self.gameElements['utilities'].get((x, y), 0)
                    cell += "{:.2f}".format(utility)
                elif (x, y) in ghosts:
                    cell = "<"
                    utility = self.gameElements['utilities'].get((x, y), 0)
                    cell += "{:.2f}".format(utility)
                else:
                    utility = self.gameElements['utilities'].get((x, y), 0)
                    cell += "{:.2f}".format(utility)
                row += "{:>6} ".format(cell)
            print(row)
        print '\n'