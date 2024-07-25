from itertools import product
import random
import lle


class Labyrinth:
    def __init__(self):
        self._world = lle.World(""" 
@  @ . . @ @ .
S0 . . . . . .
.  . @ @ . @ @
@  . @ . . . @
.  . @ . @ . .
.  @ . . . @ .
.  . . @ . . X""")
        self._done = False
        self._first_render = True
        all_positions = set(product(range(self._world.height), range(self._world.width)))
        self._initial_position = None 
        self._start_positions = list(all_positions - set(self._world.wall_pos) - set(self._world.exit_pos))

    def reset(self):
        self._done = False
        self._world.reset()
        initial_position = random.choice(self._start_positions)
        self._world.set_state(lle.WorldState([initial_position], []))

    
    def get_initial_position(self):
        return self._initial_position

    def step(self, action: int) -> float:
        """Effectue une action sur l'environnement et retourne la récompense associée"""
        events = self._world.step([lle.Action(action)])
        for event in events:
            if event.event_type == lle.EventType.AGENT_EXIT:
                self._done = True
                return 0.0
        return -1.0

    def get_observation(self) -> tuple[int, int]:
        return self._world.agents_positions[0]

    def is_done(self) -> bool:
        return self._done

    def render(self):
        import cv2

        img = self._world.get_image()
        if self._first_render:
            # Solves a bug such that the first rendering is not displayed correctly the first time
            cv2.imshow("Labyrinth", img)
            cv2.waitKey(1)
            self._first_render = False
            import time

            time.sleep(0.2)

        cv2.imshow("Labyrinth", img)
        cv2.waitKey(1)

# EnvMDP
class ENVMDP(MDP[WorldState, list[Action]]):
    def __init__(self, world: World):
        self.world = world

    def available_actions(self, state: WorldState):
        self.world.set_state(state)
        available = self.world.available_actions()
        return list(product(*available))

    def transitions(
        self, state: WorldState, action: list[Action]
    ) -> list[tuple[WorldState, float]]:
        self.world.set_state(state)
        self.world.step(action)
        return [(self.world.get_state(), 1.0)]

    def is_final(self, state: WorldState) -> bool:
        self.world.set_state(state)
        return self.world.done

    def reward(
        self,
        state: WorldState,
        action: list[Action],
        new_state: WorldState,
    ) -> float:
        # Step the world and check if the new state is the same as the given one
        # If if is not the same, then test all the other available actions.
        self.world.set_state(state)
        actions = [action] + [[a] for a in self.world.available_actions()[0]]
        for a in actions:
            r = self.world.step(a)
            if self.world.get_state() == new_state:
                return r 
            self.world.set_state(state)
        raise ValueError("The new state is not reachable from the given state")

    def states(self) -> Iterable[WorldState]:
        all_positions = set(product(range(self.world.height), range(self.world.width)))
        all_positions = all_positions.difference(set(self.world.wall_pos))
        agents_positions = list(product(all_positions, repeat=self.world.n_agents))
        collection_status = list(product([True, False], repeat=self.world.n_gems))

        for pos, collected in product(agents_positions, collection_status):
            s = WorldState(list(pos), list(collected))
            yield s




    