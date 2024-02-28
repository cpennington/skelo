# MIT License
#
# Copyright (c) 2022 Michael B Hynes
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging

import numpy as np
import pandas as pd
import glicko2
from typing import Tuple, Generic

from sklearn.base import BaseEstimator, ClassifierMixin

from skelo.model.base import RatingModel, RatingEstimator, T


logger = logging.getLogger(__name__)


class Glicko2Model(Generic[T], RatingModel[Tuple[float, float, float], T]):
    """
    Dictionary-based implementation of the `Glicko2 rating system <https://en.wikipedia.org/wiki/Glicko_rating_system>`_.

    This class creates a dictionary of Glicko2 ratings for each player inserted into the rating system, such
    that each match update will append new ratings for the respective match players, calculated according
    to the Glicko2 update formula.

    This model may be used directly, but is primarily intended as a utility class for an Glicko2Estimator.
    """

    transform_headers = ["r", "rd", "v"]

    def __init__(self, initial_value=(1500.0, 350.0, 0.06), initial_time=0, **kwargs):
        """
        Construct a Glicko2 RatingsModel.

        Args:
          initial_value (float, float, float): initial default rating and deviation assigned to a new player
          initial_time (int or orderable): the earliest "time" value for matches between players.
        """
        super().__init__(
            initial_value=initial_value, initial_time=initial_time, **kwargs
        )

    def get(self, key, timestamp=None, strict_past_data=True):
        current = super().get(key, timestamp, strict_past_data)
        player = glicko2.Player(*current["rating"])
        for _ in range(current["trailing_empty_periods"]):
            player.did_not_compete()
        return {**current, "rating": (player.getRating(), player.getRd(), player.vol)}

    def evolve_rating(self, player, matches):
        """
        Update a Glicko rating based on the outcome of a match.

        This is based on the example in the glicko2 package's unit tests,
        available `here <https://github.com/deepy/glicko2/blob/master/tests/tests.py>`_
        """
        player = glicko2.Player(*self.get(player)["rating"])
        rs, rds, vs, results = zip(
            *[[*self.get(opp)["rating"], result] for opp, result in matches]
        )
        player.update_player(list(rs), list(rds), list(results))
        return (player.getRating(), player.getRd(), player.vol)

    def compute_prob(self, r1, r2):
        """
        Return the probability of a player with rating r1 beating a player with rating r2.

        For more background, please see the `Glicko Paper <http://glicko.net/glicko/glicko.pdf>`_
        """
        p1 = glicko2.Player(*r1)
        p2 = glicko2.Player(*r2)
        return p1.win_prob(p2)


class Glicko2Estimator(RatingEstimator):
    """
    A scikit-learn Classifier for creating ratings according to the
    `Glicko2 rating system <https://en.wikipedia.org/wiki/Glicko_rating_system>`_.
    """

    RATING_MODEL_CLS = Glicko2Model
    RATING_MODEL_ATTRIBUTES = [
        "initial_value",
        "initial_time",
        "initial_ratings",
    ]

    def __init__(
        self,
        key1_field=None,
        key2_field=None,
        timestamp_field=None,
        initial_value=(1500.0, 350.0, 0.06),
        initial_time=None,
        rating_period_field=None,
        incremental_fit=False,
        initial_ratings=None,
        **kwargs,
    ):
        """
        Construct a classifier object, without fitting it.

        Args:
          key1_field (string): column name of the player1 key, if fit on a pandas DataFrame
          key2_field (string): column name of the player2 key, if fit on a pandas DataFrame
          timestamp_field (string): column name of the timestamp field, if fit on a pandas DataFrame
        """
        super().__init__(
            key1_field=key1_field,
            key2_field=key2_field,
            timestamp_field=timestamp_field,
            initial_value=initial_value,
            initial_time=initial_time,
            rating_period_field=rating_period_field,
            incremental_fit=incremental_fit,
            initial_ratings=initial_ratings,
            **kwargs,
        )
