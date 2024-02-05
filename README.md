# Pac-Man MDP Solver

## Introduction
This repository contains the `MDPAgent` for the classic Pac-Man game, implementing a Markov Decision Process (MDP) solver designed to optimize Pac-Man's navigation and ghost evasion strategies. This solver calculates the best moves for Pac-Man based on reward maximization under the game's uncertain dynamics. It's built to integrate with the Pac-Man AI framework from UC Berkeley's AI course, operating specifically with Python 2.7.

## Prerequisites
- Python 2.7
- Pac-Man AI framework from [UC Berkeley's AI course](http://ai.berkeley.edu/project_overview.html)

## Setup
To use this MDP solver with the Pac-Man game, follow these steps:

1. Ensure Python 2.7 is installed on your system.
2. Download the necessary Pac-Man AI framework files from [UC Berkeley's AI course](http://ai.berkeley.edu/project_overview.html).
3. Clone this repository or download the `MDPAgent` file:
    ```
    git clone XEZ1/PacmanMDP
    ```
4. Place the `MDPAgent.py` file in the same directory as the Pac-Man AI framework files.

## Running the Solver
To start the game with the `MDPAgent`, navigate to the directory containing the game files and the `MDPAgent.py` file, then run:
```
python2.7 pacman.py -q -n 25 -p MDPAgent -l smallGrid
```

```
python2.7 pacman.py -q -n 25 -p MDPAgent -l mediumClassic
```

`-l` is shorthand for `-layout`, `-p` is shorthand for `-pacman`, and `-q` runs the game without the interface (making it faster).

You can customize the game settings by adding additional command-line options as described in the original Pac-Man AI project documentation.

## Contributing
Contributions to enhance or extend the `MDPAgent` are welcome. If you're interested in contributing, please fork the repository, make your changes, and submit a pull request.

## License
This `MDPAgent` is intended for educational purposes and follows the licensing agreement of the Pac-Man AI projects developed at UC Berkeley.