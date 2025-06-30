




###----------------------- Exercise 4 ----------------------###

# You are building a game. Create a base class GameEntity with method interact(). 
# Implement Player, Enemy, and NPC that override interact() differently. 
# Create a list of entities with all objects.
# Create a function run_interactions(entities) that returns the logic of the list 


class GameEntity:
    def interact(self):
        raise NotImplementedError
    
class Player(GameEntity):
    def interact(self):
        print(f"I'm a player")

class Enemy(GameEntity):
    def interact(self):
        print(f"I'm going to kill you")

class NPC(GameEntity):
    def interact(self):
        print(f"I'm useless")

def run_interactions(entities):
        for entity in entities:
            entity.interact()


entities = [Player(), Enemy(), NPC()]

run_interactions(entities)


"""
I'm a player
I'm going to kill you
I'm useless
"""


    