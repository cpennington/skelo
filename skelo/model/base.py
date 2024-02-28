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

import bisect
import logging
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import (
    TypeVar,
    Generic,
    List,
    Tuple,
    Iterable,
    Literal,
    DefaultDict,
    Mapping,
    cast,
    TYPE_CHECKING,
    Any,
)
from typing_extensions import (
    TypedDict,
)

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison

    T = TypeVar("T", bound=SupportsRichComparison)
else:
    T = TypeVar("T")

R = TypeVar("R")

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)


class Rating(TypedDict, Generic[R, T]):
    rating: R
    valid_from: T
    valid_to: T | None
    trailing_empty_periods: int


RatingDict = Mapping[str, List[Rating[R, T]]]


class RatingModel(ABC, Generic[R, T]):
    """
    Base class defining a ratings system based on the ratings update formula, and also
    to storing the timeseries of player ratings as match outcomes are added to the system.

    The `RatingModel` (and its subclasses) are generally not necessary to instantiate directly,
    but are intended as factory objects for performing the ratings data storage and retrieval
    within the `sklean`-compatible estimator classes available to users.
    """

    def __init__(
        self,
        initial_value,
        initial_time,
        initial_ratings: RatingDict | None = None,
    ):
        self.initial_value = initial_value
        self.initial_time = initial_time
        self.ratings: DefaultDict[str, List[Rating[R, T]]] = defaultdict(
            lambda: [
                {
                    "rating": self.initial_value,
                    "valid_from": self.initial_time,
                    "valid_to": None,
                    "trailing_empty_periods": 0,
                }
            ],
            initial_ratings or {},
        )

    @abstractmethod
    def evolve_rating(self, player: str, matches: List[Tuple[str, Literal[0, 1]]]) -> R:
        raise NotImplementedError

    @property
    def keys(self) -> Iterable[str]:
        """
        The current keys (player identifiers) for all players in the system, which may
        be any hashable object.
        """
        return self.ratings.keys()

    def build(self, keys):
        """
        Lazily initialize the RatingModel by populating the ratings dictionary.

        Args:
          keys: iterable of player keys to use for identifying players in the system

        Returns:
          RatingModel: self
        """
        for key in keys:
            self.add(key, value=self.initial_value)
        return self

    def add(self, key, value=None):
        """
        Add a single player to the rating system.
        If the player already exists, no action is performed.

        Args:
          key: identifier for the player
          value: initial value of the player's rating

        Returns:
          RatingModel: self
        """
        if key in self.ratings:
            return self
        self.ratings[key] = [
            {
                "rating": value or self.initial_value,
                "valid_from": self.initial_time,
                "valid_to": None,
                "trailing_empty_periods": 0,
            }
        ]
        return self

    def get(
        self, key: str, timestamp: T | None = None, strict_past_data: bool = True
    ) -> Rating[R, T]:
        """
        Retrieve a player rating payload for the specified system `timestamp`.
        The `timestamp` will be used to retrieve the dictionary of rating data
        for the player as of that time. If `timestamp=None` is passed (the default),
        the last appended rating record is returned.

        If `strict_past_data=True`, then no ratings data from matches at times
        equal to or greater than `timestamp` will be returned. This is useful to
        ensure that no future information leaks into the retrieved ratings.

        If `strict_past_data=False`, then in the event that there exist 1 or more
        matches at `timestamp`, the last appended rating for that timestamp will be
        returned.

        As an example, consider the following ratings for a player:

        .. code-block::

          [
            {'valid_from': 0, 'valid_to': 2,    'rating': 1500},
            {'valid_from': 2, 'valid_to': 2,    'rating': 1510},
            {'valid_from': 2, 'valid_to': 3,    'rating': 1520},
            {'valid_from': 3, 'valid_to': None, 'rating': 1530},
          ]

        These ratings correspond to 3 matches played at times `2`, `2`, and `3`. The first
        rating is simply a book-keeping record that gives the initial rating value
        at the system initial time, `0`. It is permissible to have 2 matches occur at the
        same time (but not recommended since you cannot in general retrieve a
        physically meaningful rating in the event of collisions in time).

        In this example:

        - the call `get(key, 1)` returns record 1 (rating 1500), regardless of the value
          of strict_past_data, since this record should be returned for any timestamp
          prior to the first observed match for the player
        - the call `get(key, 2, strict_past_data=True)` returns record 1 (rating 1500)
        - the call `get(key, 2, strict_past_data=False)` returns record 3 (rating 1520)
        - the call `get(key, 3, strict_past_data=True)` returns record 3 (rating 1530)
        - the call `get(key, 4)` returns record 4 (rating 1520)
        - the call `get(key)` returns record 4 (rating 1520), since no timestamp is given

        For use cases in which future data leakage is problematic (e.g. forecasting),
        `strict_past_data=True` should always be used. For use cases where simple ratings
        retrieval is desired and the match start time is a more convenient ledger timestamp,
        `strict_past_data=False` is appropriate.

        Args:
          key: identifier for the player
          timestamp: system time at which to retrieve the ratings data
          strict_past_data (bool): if `True`, ensure no future data are returned if the
            provided `timestamp` exactly matches a match timestamp

        Returns:
          dict: a ratings payload dictionary of the rating data for a player with keys:
            `'rating', 'valid_from', 'valid_to'`

        Raises:
          KeyError: if key is not yet added to the system with `add()`
        """
        ratings = self.ratings[key]
        if ratings is None:
            raise KeyError(f"key: {key} is not yet stored in the model.")

        if timestamp is None:
            idx = -1
        else:
            start_ts = [r["valid_from"] for r in ratings]
            bisect_fn = bisect.bisect_left if strict_past_data else bisect.bisect_right
            idx_unbounded = bisect_fn(start_ts, timestamp) - 1
            idx = min(len(ratings), max(0, idx_unbounded))
        return ratings[idx]

    def update(
        self,
        winners: Iterable[str],
        losers: Iterable[str],
        period_end: T,
        rating_period: bool = False,
    ):
        """
        Update the ratings for all the supplied matches in a given rating period
        starting at the `period_end` by performing the following operations:

        - set all players' latest ratings' `valid_to` field to `period_end`
        - calculate the new ratings and append these to players' timeseries with
          `valid_from` set to `period_end`

        Args:
          winners: identifier for the player who won each match
          losers: identifier for the player who lost each match
          period_end: system time at which the rating period ended.
            This does not need to be a `datetime.datetime` or similar. Any orderable type
            is suitable.

        Returns:
          RatingModel: self

        Raises:
          KeyError: if winner or loser is not yet added to the system
          ValueError: if winner or loser is updated retroactively
        """
        players = set(winners) | set(losers)
        new_ratings = {}
        for player in players:
            rating = self.get(player)
            if period_end < rating["valid_from"]:
                raise ValueError(
                    f"Attempted to retrospectively update a rating for {player} at timestamp '{period_end}' "
                    f"which is earlier than the latest available rating at timestamp '{rating['valid_from']}'"
                )
            rating["valid_to"] = period_end

            player_matches = [
                (loser if winner == player else winner, 1 if winner == player else 0)
                for (winner, loser) in zip(winners, losers)
                if player in [winner, loser]
            ]

            new_ratings[player] = {
                "rating": self.evolve_rating(player, player_matches),
                "valid_from": period_end,
                "valid_to": None,
                "trailing_empty_periods": 0,
            }
        if rating_period:
            for player in self.ratings.keys() - players:
                rating = self.get(player)
                if period_end < rating["valid_from"]:
                    raise ValueError(
                        f"Attempted to retrospectively update a rating for {player} at timestamp '{period_end}' "
                        f"which is earlier than the latest available rating at timestamp '{rating['valid_from']}'"
                    )
                if rating["valid_to"] is None:
                    rating["valid_to"] = period_end
                rating["trailing_empty_periods"] += 1

        for player, rating in new_ratings.items():
            self.ratings[player].append(rating)
        return self

    @abstractmethod
    def compute_prob(self, r1: R, r2: R) -> float:
        """
        Returns:
          float: the probability of a player with rating `r1` beating a player with rating `r2`.
        """
        raise NotImplementedError

    def predict_proba(self, key1, key2, timestamp=None, strict_past_data=True):
        """
        Compute the probability of victory of player `key1` over player `key2` at the
        given timestamp. If no timestamp is provided, then the latest rating for
        a player is used to calculate the probability.

        Either scalar values or iterables may be provided for `key1`, `key2`, and
        `timestamp` to provide a convenient API for bulk prediction (i.e. the caller need
        not write a for-loop to call `predict_proba` multiple times).

        Args:
          key1: identifier for player 1, or iterable of identifiers
          key2: identifier for player 2, or iterable of identifiers
          timestamp: match time, or iterable of match times (default: `None`)
          strict_past_data (bool): if `True`, ensure no future data are returned if the
            provided `timestamp` exactly matches a match timestamp in the system

        Returns:
          float or list[float]: probability of player `key1` beating player `key2` at `timestamp`

        Raises:
          ValueError: if `key1`, `key2`, `timestamp` are iterables of unequal lengths
        """
        is_scalar = type(key1) is str or not hasattr(key1, "__iter__")
        if is_scalar:
            key1 = [key1]
            key2 = [key2]
        else:
            if len(key1) != len(key2):
                raise ValueError(
                    f"Iterables of keys must be the same length; received {len(key1)} vs {len(key2)}."
                )

        if timestamp is None:
            tuples = zip(key1, key2, len(key1) * [None])
        else:
            if is_scalar:
                tuples = zip(key1, key2, [timestamp])
            else:
                if len(timestamp) != len(key1):
                    raise ValueError(
                        f"Provided timestamps should be iterable with same length as keys ({len(key1)})"
                    )
                tuples = zip(key1, key2, timestamp)

        probs = []
        report_interval = len(key1) / 10
        intervals_complete = 1
        taken = 0
        for p1, p2, ts in tuples:
            # Get the ratings for p1, p2 at the specified match time, `ts`.
            r1 = self.get(p1, ts, strict_past_data=strict_past_data)["rating"] or np.nan
            r2 = self.get(p2, ts, strict_past_data=strict_past_data)["rating"] or np.nan
            pr = self.compute_prob(r1, r2)
            probs.append(pr)
            taken += 1
            if taken >= report_interval:
                logger.debug(
                    f"Completed {intervals_complete}/10 predict_proba intervals"
                )
                taken = 0
                intervals_complete += 1

        if is_scalar:
            probs = probs[0]
        return probs

    def transform(
        self, key1, key2, timestamp: List[T] | T | None = None, strict_past_data=True
    ):
        """
        Transform the player key pairs into their respective ratings at the
        given timestamp. If no timestamp is provided, then the latest ratings
        are used.

        Either scalar values or iterable values may be provided for `key1`, `key2`, and
        `timestamp` to provide a convenient API for bulk prediction (i.e. the caller need
        not write a for-loop to call `transform` multiple times).

        Args:
          key1: identifier for player 1, or iterable of identifiers
          key2: identifier for player 2, or iterable of identifiers
          timestamp: match time, or iterable of match times
          strict_past_data (bool): if `True`, ensure no future data are returned if the
            provided `timestamp` exactly matches a match timestamp in the system

        Returns:
          tuple[float] or list[tuple[float]]: ratings for the players or list thereof
        """
        is_scalar = type(key1) is str or not hasattr(key1, "__iter__")
        if is_scalar:
            key1 = [key1]
            key2 = [key2]
            timestamp = [timestamp]
        ratings = []
        report_interval = len(key1) / 10
        intervals_complete = 1
        taken = 0
        for p1, p2, ts in zip(key1, key2, timestamp):
            # Get the ratings for p1, p2 at the given match_at time
            r1 = self.get(p1, ts, strict_past_data=strict_past_data)["rating"] or np.nan
            r2 = self.get(p2, ts, strict_past_data=strict_past_data)["rating"] or np.nan
            ratings.append((r1, r2))
            taken += 1
            if taken >= report_interval:
                logger.debug(f"Completed {intervals_complete}/10 transform intervals")
                taken = 0
                intervals_complete += 1
        if is_scalar:
            ratings = ratings[0]
        return ratings

    def to_frame(self, keys=None):
        """
        Return the stored player ratings as a pandas DataFrame.

        Args:
          keys (iterable): a subset of player keys for which to return ratings

        Returns:
          `DataFrame`: a `pandas DataFrame` of player ratings, suitable for a lookup table
        """
        columns = ["key", "rating", "valid_from", "valid_to"]
        keys = keys or self.ratings.keys()
        dfs = []
        for key in keys:
            df = pd.DataFrame(self.ratings[key])
            df["key"] = key
            dfs.append(df)
        if len(dfs) == 0:
            return pd.DataFrame(columns=columns)
        return pd.concat(dfs, axis=0, ignore_index=True)[columns]


class RatingEstimator(BaseEstimator, ClassifierMixin):
    """
    Base class for a `scikit-learn` classifier implementing a rating system.

    This class creates an RatingModel ratings object and provides the scikit-learn API for using it:

    - `fit(X, y)` to fit a model
    - `predict_proba(X)` to generate continuous-valued predictions of match outcomes
    - `predict(X)` to generate binary prediction labels for match outcomes
    - `transform(X)` to map player tuples into their respective ratings

    Child classes can be used analogously to any `sklearn` classifier; the call signature will look
    similar to the following:

    .. code-block::

      X_train, y_train = ... # Training Design matrix & match labels
      model = RatingEstimator().fit(X_train, y_train)
      y_pred = model.predict(X_test)

    Child classes that implement a particular ratings model need to override the following class variables:

    - `RATING_MODEL_CLS`
    - `RATING_MODEL_ATTRIBUTES`
    """

    # Define the type of RatingModel. This should be overridden by child classes.
    RATING_MODEL_CLS = RatingModel

    # Define a list of attributes of this object that should be passed as kwargs
    # when instantiating a RatingModel during calls to fit(). This should be overridden
    # by child classes.
    RATING_MODEL_ATTRIBUTES = [
        "initial_time",
        "initial_value",
        "initial_ratings",
    ]

    def __init__(
        self,
        key1_field=None,
        key2_field=None,
        timestamp_field=None,
        initial_value=None,
        initial_time=None,
        rating_period_field=None,
        incremental_fit=False,
        initial_ratings=None,
        **kwargs,
    ):
        """
        Construct a rating classifier, without fitting it.

        Args:
          key1_field (string): name of the player1 key field, if fit on a `DataFrame`
          key2_field (string): name of the player2 key field, if fit on a `DataFrame`
          timestamp_field (string): name of the timestamp field, if fit on a `DataFrame`
          initial_value: initial rating value to assign a new player. The type of `initial_value`
            depends on the underlying `RatingModel` of the `RatingEstimator`
          initial_time: earliest possible time in the rating system (treat this like `-np.inf`)
          rating_period_field: The name of the field indicating which rating period a game takes place in
          incremental_fit (bool): if `False`, subsequent calls to fit refit the model and discard old ratings.
            If `True`, the model's ratings may be incrementally updated with (net new) training data.
            When fitting incrementally, ensure that new match data for any player are monotonically
            increasing in time.
          **kwargs: keyword arguments to pass to the parent `BaseEstimator`
        """
        print(kwargs)
        super().__init__(**kwargs)
        self.initial_value = initial_value
        self.initial_time = initial_time
        self.rating_model = None
        self.incremental_fit = incremental_fit
        self.initial_ratings = initial_ratings

        self.key1_field = key1_field
        self.key2_field = key2_field
        self.timestamp_field = timestamp_field
        self.rating_period_field = rating_period_field
        if any([key1_field, key2_field, timestamp_field]):
            if not all([key1_field, key2_field, timestamp_field]):
                raise ValueError(
                    f"All fields (key1_field, key2_field, timestamp_field) must be provided to fit using a DataFrame."
                )
        self._can_transform_dataframe = key1_field is not None
        self._fit = False

    def fit(self, X, y):
        """
        Fit the classifier by computing the ratings for each player given the match data.

        Args:
          X (ndarray or DataFrame): design matrix of matches with key1, key2, timestamp data
          y (ndarray or DataFrame): vector of match outcomes, where 1 denotes player key1 won
        Returns:
          EloEstimator: self
        """
        logger.info(f"Starting {self.__class__.__name__}.fit")
        x: pd.DataFrame

        key1 = self.key1_field or "p1"
        key2 = self.key2_field or "p2"
        ts = self.timestamp_field or "ts"

        if type(X) is pd.DataFrame:
            if self._can_transform_dataframe:
                x = X[
                    [
                        self.key1_field,
                        self.key2_field,
                        self.timestamp_field,
                        self.rating_period_field,
                    ]
                ].copy()
            else:
                logger.warning(
                    f"Attempting to transform a dataframe without attributes [key1_field, key2_field, timestamp_field]; using columns [0, 1, 2]"
                )
                x = X.iloc[:, :4]
        else:
            x = pd.DataFrame(X, columns=[key1, key2, ts])

        if type(y) is pd.DataFrame:
            x["result"] = y.iloc[:, 0]
        else:
            x["result"] = y

        min_time = x[ts].min()
        time_dtype = type(min_time)
        initial_time = None
        attempt_init_times = [min_time, -np.inf, 0]
        for val in attempt_init_times:
            try:
                initial_time = initial_time or time_dtype(val)
            except Exception as e:
                initial_time = None
        if initial_time is None:
            raise ValueError(
                "Could not create an initial timestamp to use for ratings during fit. "
                "Please verify the dtype of the column."
            )
        self.initial_time = initial_time

        if not self._fit or not self.incremental_fit:
            # Create a new underlying ratings model if fit() has not been called,
            # or fit() is called with incremental_fit=False.
            parameters = {
                attr: getattr(self, attr) for attr in self.RATING_MODEL_ATTRIBUTES
            }
            self.rating_model = self.RATING_MODEL_CLS(**parameters)

        # Update the keyset. If fit(...) is called when incremantal_fit is True, this operation
        # may add nonexistent keys but will not affect any existing data in the system.
        keys = set(x[key1])
        keys.update(set(x[key2]))
        self.rating_model.build(keys)

        # Add new match observations into the rating system
        x.sort_values([ts, key1, key2], inplace=True)
        x["winner"] = x.apply(lambda r: r[key1] if r.result else r[key2], axis=1)
        x["loser"] = x.apply(lambda r: r[key2] if r.result else r[key1], axis=1)
        if self.rating_period_field:
            periods = x.groupby(self.rating_period_field)
            report_interval = len(periods) / 10
            intervals_complete = 1
            taken = 0
            for _, period in sorted(periods):
                self.rating_model.update(
                    period.winner, period.loser, period[ts].max(), True
                )
                taken += 1
                if taken >= report_interval:
                    logger.debug(f"Completed {intervals_complete}/10 fit intervals")
                    taken = 0
                    intervals_complete += 1
        else:
            report_interval = len(periods) / 10
            intervals_complete = 1
            taken = 0

            def update_row(row):
                self.rating_model.update([row.winner], [row.loser], row[ts])
                taken += 1
                if taken >= report_interval:
                    logger.debug(f"Completed {intervals_complete}/10 fit intervals")
                    taken = 0
                    intervals_complete += 1

            x.apply(update_row, axis=1)

        self._fit = True
        logger.info(f"Ending {self.__class__.__name__}.fit")
        return self

    def transform(self, X, strict_past_data=True):
        """
        Transform the player identifier matrix into either the player ratings or the estimated
        probability of player 1 defeating player 2.

        This method may be used after the model has been fit to convert a design matrix of
        player identifiers into numerical quantities at either historical times or future times.
        What distinguishes `transform` from `predict_proba` and `predict` is that `predict_proba`
        and `predict` return predictions that *only* use past data, and *cannot* cheat
        by leaking future data into a forecast. However, when `transform` is called with
        `strict_past_data=False`, it is possible to compute ratings that *peek* into the
        future, and could return ratings updated using match outcomes pushed (slightly)
        back in time to the match start timestamp. This is a specific convenience
        utility for non-forecasting use cases in which the match start time is a more
        convenient timestamp with which to index and manipulate data.

        Args:
          X (numpy.ndarray or pandas DataFrame): design matrix of matches with key1, key2, timestamp data
          output_type (string): either 'prob' or 'rating' to specify the type of transformation
          strict_past_data (bool): if True, ensure no future data are returned if the
            provided timestamp exactly matches a match timestamp (default: True)

        Returns:
          An ndarry or pandas Series of transformed ratings or probabilities.
        """
        logger.info(f"Starting {self.__class__.__name__}.transform")
        if not self._fit:
            raise ValueError(".fit() has not been called on this model.")

        dtype = type(X)
        if dtype is pd.DataFrame:
            if self._can_transform_dataframe:
                x = X[[self.key1_field, self.key2_field, self.timestamp_field]].values
            else:
                logger.warning(
                    f"Attempting to transform a dataframe without attributes [key1_field, key2_field, timestamp_field]; using columns [0, 1, 2]"
                )
                x = X.iloc[:, :3].values
        else:
            x = X

        logger.info(f"Computing probabilities in {self.__class__.__name__}.transform")
        prob = np.array(
            self.rating_model.predict_proba(
                x[:, 0], x[:, 1], x[:, 2], strict_past_data=strict_past_data
            )
        )

        logger.info(f"Computing ratings in {self.__class__.__name__}.transform")
        ratings = np.array(
            self.rating_model.transform(
                x[:, 0], x[:, 1], x[:, 2], strict_past_data=strict_past_data
            )
        )

        logger.info(f"Restacking columns in {self.__class__.__name__}.transform")
        results = np.column_stack([prob, ratings.reshape((len(prob), -1))])

        logger.info(f"Ending {self.__class__.__name__}.transform")
        if dtype is pd.DataFrame:
            return pd.DataFrame(
                index=X.index,
                data=results,
                columns=["prob"]
                + [f"{header}1" for header in self.rating_model.transform_headers]
                + [f"{header}2" for header in self.rating_model.transform_headers],
            )
        return results

    def predict_proba(self, X):
        """
        Predict the probability of of player 1 defeating player 2 for player identifiers in
        the provided design matrix using strict past data for each prediction.

        Args:
          X (numpy.ndarray or pandas DataFrame): design matrix of matches with key1, key2, timestamp data

        Returns:
          An `ndarray` or `DataFrame` of the probabilities of victory for player 1 and 2, respectively.
        """
        pr = self.transform(X, output_type="prob", strict_past_data=True)
        pr_inv = 1 - pr
        if type(pr) is pd.Series:
            return pd.concat([pr.rename("pr1"), pr_inv.rename("pr2")], axis=1)
        else:
            return np.c_[pr, pr_inv]

    def predict(self, X):
        """
        Create a binary prediction label for player 1 to defeat player 2 for the player
        identifiers in the provided design matrix in *future* matches, using *only* the
        latest available rating for each player after the model has been fit.

        Args:
          X (numpy.ndarray or pandas DataFrame): design matrix of matches with `key1`, `key2`, `timestamp` data

        Returns:
          A `ndarry` or `DataFrame` of predicted labels of victory for player 1.
        """
        labels = np.round(self.predict_proba(X))
        if type(labels) is pd.DataFrame:
            return labels.iloc[:, 0]
        return labels[:, 0]
