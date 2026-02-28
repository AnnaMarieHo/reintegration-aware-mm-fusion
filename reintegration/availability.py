from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class TwoStateMarkovParams:
    """
    Simple 2-state Markov chain for modality availability.

    States:
        0 = OFF (modality unavailable)
        1 = ON  (modality available)

    Parameters:
        p_off_on:  P(S_t = 1 | S_{t-1} = 0)
        p_on_off:  P(S_t = 0 | S_{t-1} = 1)
        initial_on_prob: Optional override for initial ON probability; if None,
            the stationary distribution implied by (p_off_on, p_on_off) is used.
    """

    p_off_on: float
    p_on_off: float
    initial_on_prob: Optional[float] = None

    def stationary_on_prob(self) -> float:
        denom = self.p_off_on + self.p_on_off
        if denom == 0.0:
            # Degenerate case: no transitions; treat ON as extremely unlikely.
            return 0.0
        return self.p_off_on / denom


def sample_two_state_markov(
    length: int,
    params: TwoStateMarkovParams,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample a binary availability sequence of given length from a 2-state
    Markov chain with parameters `params`.

    Returns:
        mask: np.ndarray of shape (length,), dtype=bool where True means ON.
    """
    if length <= 0:
        return np.zeros((0,), dtype=bool)

    if rng is None:
        rng = np.random.default_rng()

    p0 = 1.0 - (params.initial_on_prob if params.initial_on_prob is not None else params.stationary_on_prob())
    p1 = 1.0 - p0

    states = np.zeros(length, dtype=np.int8)
    states[0] = rng.choice([0, 1], p=[p0, p1])

    for t in range(1, length):
        if states[t - 1] == 0:
            states[t] = rng.choice([0, 1], p=[1.0 - params.p_off_on, params.p_off_on])
        else:
            states[t] = rng.choice([0, 1], p=[params.p_on_off, 1.0 - params.p_on_off])

    return states.astype(bool)


def availability_history_counter(mask: np.ndarray) -> np.ndarray:
    """
    Compute r_t as defined in the project spec for a single modality:

        r_t =
          0                 if a_t = 0
          1                 if a_t = 1 and a_{t-1} = 0
          r_{t-1} + 1       if a_t = 1 and a_{t-1} = 1

    Args:
        mask: boolean availability array of shape (T,) where True means ON.

    Returns:
        r: integer array of shape (T,) with availability history counters.
    """
    mask = np.asarray(mask, dtype=bool)
    T = mask.shape[0]
    r = np.zeros(T, dtype=np.int32)

    if T == 0:
        return r

    if mask[0]:
        r[0] = 1

    for t in range(1, T):
        if not mask[t]:
            r[t] = 0
        elif not mask[t - 1]:
            r[t] = 1
        else:
            r[t] = r[t - 1] + 1

    return r


def reintegration_events(mask: np.ndarray) -> np.ndarray:
    """
    Return a boolean array of shape (T,) indicating OFF→ON reintegration
    events for a single modality sequence.

    By definition, a reintegration event for modality m occurs at timestep t
    when a_t = 1 and a_{t-1} = 0. This function sets event[0] = False.
    """
    mask = np.asarray(mask, dtype=bool)
    T = mask.shape[0]
    if T == 0:
        return np.zeros((0,), dtype=bool)
    events = np.zeros(T, dtype=bool)
    prev = mask[:-1]
    curr = mask[1:]
    events[1:] = np.logical_and(curr, np.logical_not(prev))
    return events


def bucket_history_counters(
    r: np.ndarray,
    bucket_edges: Sequence[int] = (0, 1, 2, 3, 4, 8),
) -> np.ndarray:
    """
    Bucket r_t values into a small number of discrete windows suitable for
    plotting windowed accuracy/confidence curves.

    Example with default edges:
        edges = [0, 1, 2, 3, 4, 8]
        buckets:
          [0]      -> bucket 0
          [1]      -> bucket 1
          [2]      -> bucket 2
          [3]      -> bucket 3
          [4..7]   -> bucket 4
          [8..inf) -> bucket 5

    Args:
        r: integer array of shape (T,) with availability counters.
        bucket_edges: monotonically increasing integer edges.

    Returns:
        bucket_ids: integer array of shape (T,) giving bucket index per step.
    """
    r = np.asarray(r, dtype=np.int32)
    edges = np.asarray(bucket_edges, dtype=np.int32)
    bucket_ids = np.zeros_like(r, dtype=np.int32)

    for b, edge in enumerate(edges):
        bucket_ids[r == edge] = b

    if len(edges) > 0:
        last_edge = edges[-1]
        bucket_ids[r > last_edge] = len(edges)

    return bucket_ids


def generate_availability_schedule(
    dataset_name: str,
    len_a: int,
    len_b: int,
    markov_params: TwoStateMarkovParams,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate availability schedule for a single sample.

    Args:
        dataset_name: name of the dataset, e.g. "meld".
        len_a: per-sample sequence length for modality A (audio).
        len_b: per-sample sequence length for modality B (text).
        markov_params: parameters of the 2-state Markov chain.
        seed: optional RNG seed for reproducibility (e.g. hash of client_id + sample idx).

    Returns:
        a_mask, b_mask: bool arrays (T_a,) and (T_b,) where True means ON.
        events_a, events_b: bool arrays indicating OFF→ON reintegration events.
        r_a, r_b: int arrays, availability history counters.
    """
    if dataset_name == "meld":
        rng_a = np.random.default_rng(seed) if seed is not None else None
        rng_b = np.random.default_rng(seed + 1) if seed is not None else None
        a_mask = sample_two_state_markov(len_a, markov_params, rng=rng_a)
        b_mask = sample_two_state_markov(len_b, markov_params, rng=rng_b)
        events_a = reintegration_events(a_mask)
        events_b = reintegration_events(b_mask)
        r_a = availability_history_counter(a_mask) #do we want history precomuted? I have no idea - we will have to see
        r_b = availability_history_counter(b_mask)
        return a_mask, b_mask, events_a, events_b, r_a, r_b
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")