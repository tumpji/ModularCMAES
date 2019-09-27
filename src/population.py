import itertools
from typing import Any
import numpy as np
from .utils import AnnotatedStruct


class Population(AnnotatedStruct):
    '''AnnotatedStruct object for holding a Population of individuals. '''
    x: np.ndarray
    y: np.ndarray
    f: np.ndarray

    def sort(self) -> None:
        '''Sorts the population according to their fitness values'''
        rank = np.argsort(self.f)
        self.x = self.x[:, rank]
        self.y = self.y[:, rank]
        self.f = self.f[rank]

    def copy(self) -> "Population":
        '''Returns a new population object, with it's variables copied

        Returns
        ------
        Population
        '''

        return Population(self.x, self.y, self.f)

    def __add__(self, other: "Population") -> "Population":
        '''Adds two population objects with each other

        Parameters
        ----------
        other: Population
            another population which is to be used to perform the addition

        Returns
        ------
        Population
        '''
        assert isinstance(other, self.__class__)
        return Population(
            np.hstack([self.x, other.x]),
            np.hstack([self.y, other.y]),
            np.append(self.f, other.f)
        )

    def __getitem__(self, key: Any) -> "Population":
        '''Custom implemenation of the getitem method, allowing
        for indexing the entire population object as if it were a np.ndarray

        Parameters
        ----------
        key: int, [int], itertools.slice
            value by with to index the population

        Returns
        ------
        Population
        '''
        if isinstance(key, int):
            return Population(
                self.x[:, key].reshape(-1, 1),
                self.y[:, key].reshape(-1, 1),
                np.array([self.f[key]])
            )
        elif isinstance(key, slice):
            return Population(
                self.x[:, key.start: key.stop: key.step],
                self.y[:, key.start: key.stop: key.step],
                self.f[key.start: key.stop: key.step]
            )
        elif isinstance(key, list) and all(
                map(lambda x: isinstance(x, int) and x >= 0, key)):
            return Population(
                self.x[:, key],
                self.y[:, key],
                self.f[key]
            )
        else:
            raise KeyError("Key must be (list of non-negative) integer(s) or slice, not {}"
                           .format(type(key)))

    def __repr__(self) -> str:
        return "<Population d: {}, n: {}>".format(*self.x.shape)
