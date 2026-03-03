from __future__ import annotations

import warnings
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


import numpy as np
from typing import Optional

def sample_two_state_markov(
    length: int,
    params,
    rng: Optional[np.random.Generator] = None,
    *,
    # guards (tune these)
    min_on: int = 1,
    min_off: int = 0,
    require_reintegration: bool = False,   # OFF->ON at least once
    require_no_reintegration: bool = False,  # no OFF->ON (stable / control)
    min_flips: int = 0,                    # total state changes
    max_flips: Optional[int] = None,       # cap total flips (e.g. 0 => stable)
    max_tries: int = 50,
) -> np.ndarray:
    """
    Sample a binary availability sequence (True=ON) from a 2-state Markov chain,
    optionally enforcing simple guards (nonempty, reintegration, no reintegration, etc.) by resampling.

    Returns:
        (mask, conformed): mask is the sequence; conformed is True if guards were satisfied,
        False if max_tries was exhausted and the last non-conforming sample was returned.
    """
    if length <= 0:
        return np.zeros((0,), dtype=bool), True

    if rng is None:
        rng = np.random.default_rng()

    p0 = 1.0 - (params.initial_on_prob if params.initial_on_prob is not None else params.stationary_on_prob())
    p1 = 1.0 - p0

    def ok(mask: np.ndarray) -> bool:
        on = int(mask.sum())
        off = int((~mask).sum())
        flips = int(np.sum(mask[1:] != mask[:-1]))
        reintegration = bool(np.any((~mask[:-1]) & (mask[1:])))
        if on < min_on: return False
        if off < min_off: return False
        if flips < min_flips: return False
        if max_flips is not None and flips > max_flips: return False
        if require_reintegration and not reintegration: return False
        if require_no_reintegration and reintegration: return False
        return True

    last = None
    for _ in range(max_tries):
        states = np.zeros(length, dtype=np.int8)
        states[0] = rng.choice([0, 1], p=[p0, p1])

        for t in range(1, length):
            if states[t - 1] == 0:
                states[t] = rng.choice([0, 1], p=[1.0 - params.p_off_on, params.p_off_on])
            else:
                states[t] = rng.choice([0, 1], p=[params.p_on_off, 1.0 - params.p_on_off])

        mask = states.astype(bool)
        last = mask
        if ok(mask):
            return mask, True

    # If we fail to satisfy guards, return last sample (baseline-friendly fallback)
    warnings.warn(
        f"sample_two_state_markov: failed to satisfy guards after {max_tries} tries "
        f"(length={length}, require_no_reint={require_no_reintegration}, max_flips={max_flips}). "
        "Returning last non-conforming sample.",
        RuntimeWarning,
        stacklevel=2,
    )
    return last, False

# Sentinel for start-of-sequence ON: no known prior OFF → treat as stably available (falls in last/overflow bucket)
STABLE_START_SENTINEL = 9999


def availability_history_counter(mask: np.ndarray) -> np.ndarray:
    """
    Compute r_t as defined in the project spec for a single modality:

        r_t =
          0                 if a_t = 0
          1                 if a_t = 1 and a_{t-1} = 0
          r_{t-1} + 1       if a_t = 1 and a_{t-1} = 1
          STABLE_START_SENTINEL  if t = 0 and a_0 = 1 (no prior state; treat as stable)

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

    # Start-of-sequence ON: no prior OFF; use sentinel so it lands in "long stable" bucket, not OFF bucket
    if mask[0]:
        r[0] = STABLE_START_SENTINEL

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
    Bucket r_t values into intervals [edges[b], edges[b+1]) for b < len(edges)-1,
    and [edges[-1], inf) for the last bucket.

    Example with default edges (0, 1, 2, 3, 4, 8):
        [0, 1)   -> bucket 0  (r=0, OFF)
        [1, 2)   -> bucket 1  (r=1, just reintegrated)
        [2, 3)   -> bucket 2
        [3, 4)   -> bucket 3
        [4, 8)   -> bucket 4  (r=4..7)
        [8, inf) -> bucket 5

    Args:
        r: integer array of shape (T,) with availability counters.
        bucket_edges: monotonically increasing integer edges.

    Returns:
        bucket_ids: integer array of shape (T,) giving bucket index per step.
    """
    r = np.asarray(r, dtype=np.int32)
    edges = list(bucket_edges)
    bucket_ids = np.full(r.shape, len(edges) - 1, dtype=np.int32)  # default: last bucket
    for b in range(len(edges) - 1):
        mask = (r >= edges[b]) & (r < edges[b + 1])
        bucket_ids[mask] = b
    bucket_ids[r >= edges[-1]] = len(edges) - 1
    return bucket_ids


# For length-scaled Markov: target expected OFF->ON events per sequence (≈ same reintegration rate for audio and text)
DEFAULT_TARGET_REINT_EVENTS = 2.0
STATIONARY_OFF_PROB_FOR_SCALING = 0.5


def generate_availability_schedule(
    dataset_name: str,
    len_a: int,
    len_b: int,
    markov_params: TwoStateMarkovParams,
    seed: Optional[int] = None,
    *,
    require_reintegration: bool = False,
    require_no_reintegration: bool = False,
    max_flips: Optional[int] = None,
    fallback_count: Optional[dict] = None,
    target_reint_events: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate availability schedule for a single sample.

    Args:
        dataset_name: name of the dataset, e.g. "meld".
        len_a: per-sample sequence length for modality A (audio).
        len_b: per-sample sequence length for modality B (text).
        markov_params: parameters of the 2-state Markov chain (used when target_reint_events is None).
        seed: optional RNG seed for reproducibility (e.g. hash of client_id + sample idx).
        require_reintegration: if True, sample must have at least one OFF->ON.
        require_no_reintegration: if True, sample must have no OFF->ON (stable control).
        max_flips: if set, sample must have total state flips <= this (e.g. 0 for stable).
        fallback_count: optional dict with "audio" and "text" keys (int) to increment when
            sample_two_state_markov exhausts max_tries and returns a non-conforming sample.
        target_reint_events: if set, use length-scaled p_off_on per modality so expected
            OFF->ON events per sequence ≈ this value (avoids text having ~0 events when T is small).
            Uses symmetric chain (p_off_on = p_on_off) with stationary_off=0.5.

    Returns:
        a_mask, b_mask: bool arrays (T_a,) and (T_b,) where True means ON.
        events_a, events_b: bool arrays indicating OFF→ON reintegration events.
        r_a, r_b: int arrays, availability history counters.
    """
    if dataset_name == "meld":
        rng_a = np.random.default_rng(seed) if seed is not None else None
        rng_b = np.random.default_rng(seed + 1) if seed is not None else None

        # Stable sidecar: ignore target_reint_events and use an all-ON chain so
        # require_no_reintegration + max_flips=0 can be satisfied without fallbacks.
        if require_no_reintegration:
            params_a = params_b = TwoStateMarkovParams(
                p_off_on=0.0,
                p_on_off=0.0,
                initial_on_prob=1.0,
            )
        elif target_reint_events is not None:
            # Length-scaled: p so that E[OFF->ON] ≈ T * stationary_off * p = target_reint_events
            # With symmetric chain (p_off_on = p_on_off = p), stationary_off = 0.5 => p = target / (T*0.5)
            p_a = min(1.0, target_reint_events / (max(1, len_a) * STATIONARY_OFF_PROB_FOR_SCALING))
            p_b = min(1.0, target_reint_events / (max(1, len_b) * STATIONARY_OFF_PROB_FOR_SCALING))
            params_a = TwoStateMarkovParams(p_off_on=p_a, p_on_off=p_a)
            params_b = TwoStateMarkovParams(p_off_on=p_b, p_on_off=p_b)
        else:
            params_a = params_b = markov_params

        a_mask, a_conformed = sample_two_state_markov(
            len_a, params_a, rng=rng_a, min_on=1,
            require_reintegration=require_reintegration,
            require_no_reintegration=require_no_reintegration,
            max_flips=max_flips,
        )
        if fallback_count is not None and not a_conformed:
            fallback_count["audio"] = fallback_count.get("audio", 0) + 1
        b_mask, b_conformed = sample_two_state_markov(
            len_b, params_b, rng=rng_b, min_on=1,
            require_reintegration=require_reintegration,
            require_no_reintegration=require_no_reintegration,
            max_flips=max_flips,
        )
        if fallback_count is not None and not b_conformed:
            fallback_count["text"] = fallback_count.get("text", 0) + 1
        events_a = reintegration_events(a_mask)
        events_b = reintegration_events(b_mask)
        r_a = availability_history_counter(a_mask) #do we want history precomuted? I have no idea - we will have to see
        r_b = availability_history_counter(b_mask)
        return a_mask, b_mask, events_a, events_b, r_a, r_b
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")