import warnings
import typing
from collections import deque
from functools import wraps
import numpy as np
from .configurablecmaes import ConfigurableCMAES


def check_break_conditions(f:typing.Callable) -> typing.Callable:
    @wraps(f)
    def inner(self, *args, **kwargs) -> typing.Any:
        if any(self.break_conditions):
            raise StopIteration("Break conditions reached, ignoring call to: " +  
                f.__qualname__)
        return f(self, *args, **kwargs)
    return inner

class AskTellCMAES(ConfigurableCMAES):
    'Ask tell interface for the ConfigurableCMAES'

    def __init__(self, *args, **kwargs) -> None:
        'Override the fitness_function argument with an empty callable' 
        super().__init__(lambda: None, *args, **kwargs)

    def fitness_func(self, x: np.ndarray) -> None:
        'Overwrite function call for fitness_func, calls register_individual'
        self.register_individual(x)

    def sequential_break_conditions(self, i: int, f: float) -> None:
        '''Overwrite function sequential_break_conditions, 
        raise NotImplemented if sequential selection is enabled'''
        if self.parameters.sequential:
            raise NotImplementedError("Sequential selection is not implemented "
                             "for ask-tell interface")

    def step(self):
        'Overwrite function step, raise NotImplementedError'
        raise NotImplementedError("Step is undefined in this interface")

    def run(self):
        'Overwrite function strunep, raise NotImplementedError'
        raise NotImplementedError("Run is undefined in this interface")

    def register_individual(self, x: np.ndarray) -> None:
        '''Add new individuals to ask_queue
        Parameters
        ----------
        x: np.ndarray
            The vector to be added to the ask_queue
        '''
        self.ask_queue.append(x)

    @check_break_conditions
    def ask(self) -> np.ndarray:
        '''Retrieves the next indivual from the ask_queue.
        If the ask_queue is not defined yet, it is defined and mutate is
        called in order to fill it.
        
        Returns
        -------
        np.ndarray
        '''
        if not hasattr(self, 'ask_queue'):
            self.ask_queue = deque()
            self.mutate()
        return self.ask_queue.popleft() 

    @check_break_conditions
    def tell(self, xi:np.ndarray, fi: float) -> None:
        '''Processes a provided fitness value fi for a given individual xi.
        Parameters
        ----------
        xi: np.ndarray
            An individual previously returned by ask()
        fi: float
            The fitness value for xi
        Raises
        ------
        RuntimeError
            When ask() is not called before tell()
        ValueError
            When an unknown xi is provided to the method
        
        Warns
        -----
        UserWarning
            When the same xi is provided more than once
        '''
        if not self.parameters.population:
            raise RuntimeError("Call to tell without calling ask first is prohibited")

        indices, *_ = np.where((self.parameters.population.x == xi).all(axis=0)) 
        if len(indices) == 0:
            raise ValueError("Unkown xi provided")
        
        for index in indices:
            if self.parameters.population.f[index] == None:
                self.parameters.population.f[index] = fi
                break
        else:
            warnings.warn("Repeated call to tell with same xi", UserWarning)
            self.parameters.population.f[index] = fi
        
        self.parameters.used_budget += 1
        if len(self.ask_queue) == 0 and (self.parameters.population.f != None).all():
            self.select()
            self.recombine()
            self.parameters.adapt()
            self.mutate() 

 


