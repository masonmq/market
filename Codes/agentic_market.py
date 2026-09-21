#!/usr/bin/env python3
"""
Agentic LMSR market over 5 popularity levels {1,2,3,4,5}.

Uses:
  - post_metadata.py to get the post category domain d(x)
  - llm_agent.py to get each expert's probability distribution over levels
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Any

import llm_agent as llm
import numpy as np
from post_metadata import post_metadata


K_LEVELS: tuple[str, ...] = ("1", "2", "3", "4", "5")

DOMAIN_MAP: dict[int, str] = {
    1: "Animal",
    2: "Electronics",
    3: "Entertainment",
    4: "Family",
    5: "Fashion",
    6: "Food",
    7: "Holiday&Celebrations",
    8: "Social&People",
    9: "Travel&Active&Sports",
    10: "Urban",
    11: "Whether&Season",
}

DOMAIN_TO_ID: dict[str, int] = {v: k for k, v in DOMAIN_MAP.items()}


def _softmax_prices(q: list[float], beta: float) -> dict[str, float]:
    z = sum(math.exp(beta * qi) for qi in q)
    return {K_LEVELS[i]: math.exp(beta * q[i]) / z for i in range(5)}


def _lmsr_cost(q: list[float], beta: float) -> float:
    # C(q) = (1/beta) * log(sum_j exp(beta q_j))
    z = sum(math.exp(beta * qi) for qi in q)
    return (1.0 / beta) * math.log(z)


def _lmsr_buy_one_share_cost(q: list[float], beta: float, k_index: int) -> float:
    q2 = list(q)
    q2[k_index] += 1.0
    return _lmsr_cost(q2, beta) - _lmsr_cost(q, beta)


def _extract_category(image_path: str) -> str:
    post_data = json.loads(post_metadata(image_path))
    cat = post_data.get("category")
    if not isinstance(cat, str) or not cat:
        raise TypeError(f"post_metadata category must be a non-empty string, got {cat!r}")
    return cat


def _call_expert(
    image_path: str,
    expert_domain: str,
    *,
    market_prices: dict[str, float],
) -> dict[str, float]:
    out = llm.llm_agent(image_path=image_path, expert_domain=expert_domain, market_prices=market_prices)
    probs = out.get("popularity_level_probabilities")
    if not isinstance(probs, dict):
        raise TypeError("llm_agent did not return popularity_level_probabilities object")
    return {k: float(probs[k]) for k in K_LEVELS}


@dataclass(frozen=True)
class MarketResult:
    q: list[float]
    prices: dict[str, float]
    prices_over_time: "np.ndarray"
    k_hat: str
    popularity_level_probability: "np.ndarray"
    popularity_level_probability_path: str | None
    prices_over_time_path: str | None


def run_agentic_market(
    image_path: str,
    *,
    beta: float = 1.0,
    T: int = 3,
    num_other_agents: int = 3,
    tau: float = 0.0,
    seed: int | None = 0,
    save_tensor_path: str | None = "popularity_level_probability.npy",
    save_prices_path: str | None = "prices_over_time.npy",
) -> MarketResult:
    """
    Implements the pseudocode in the prompt using 1-share LMSR trades.

    - Initializes q=(0,0,0,0,0)
    - For t=1..T:
        A_t = {a_{d(x)}} ∪ Sample(D\\{d(x)}) of size num_other_agents
        Each agent gets probabilities via llm_agent(image_path, expert_domain)
        Each agent may buy 1 share of best k* if (pi_k* - kappa_k*(1)) >= tau
    - Returns final prices p_T and argmax k_hat.
    """
    if beta <= 0:
        raise ValueError("beta must be > 0")
    if T <= 0:
        raise ValueError("T must be >= 1")
    if num_other_agents < 0:
        raise ValueError("num_other_agents must be >= 0")

    rng = random.Random(seed)

    q: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0]
    d_x = _extract_category(image_path)

    # Tensor: (T, 12, 5). Indices d=1..11 map to DOMAIN_MAP.
    # d=0 is reserved (unused) and kept as initialized values.
    popularity_level_probability = np.full((T, 12, 5), 0.2, dtype=np.float64)

    # Prices tensor: (T, 5), prices after each iteration t.
    prices_over_time = np.zeros((T, 5), dtype=np.float64)

    if d_x not in DOMAIN_TO_ID:
        raise KeyError(
            f"Category {d_x!r} not found in DOMAIN_MAP. "
            "Update DOMAIN_MAP / DOMAIN_TO_ID to match your dataset categories."
        )
    d_x_id = DOMAIN_TO_ID[d_x]

    all_domain_ids = list(DOMAIN_MAP.keys())  # [1..11]
    other_domain_ids = [d for d in all_domain_ids if d != d_x_id]

    for t in range(T):
        if t > 0:
            popularity_level_probability[t, :, :] = popularity_level_probability[t - 1, :, :]

        sampled_ids = rng.sample(
            other_domain_ids, k=min(num_other_agents, len(other_domain_ids))
        )
        participants_ids = [d_x_id, *sampled_ids]

        for d_id in participants_ids:
            d_name = DOMAIN_MAP[d_id]
            current_prices = _softmax_prices(q, beta)
            pi = _call_expert(image_path, d_name, market_prices=current_prices)
            popularity_level_probability[t, d_id, :] = np.array(
                [pi[k] for k in K_LEVELS], dtype=np.float64
            )

            # Compute kappa^k(1) for each k in {1..5}
            kappa = [
                _lmsr_buy_one_share_cost(q=q, beta=beta, k_index=i) for i in range(5)
            ]

            # Choose k* = argmax_k (pi_k - kappa_k)
            gains = [pi[K_LEVELS[i]] - kappa[i] for i in range(5)]
            k_star = max(range(5), key=lambda i: gains[i])

            if gains[k_star] >= tau:
                q[k_star] += 1.0

        prices_over_time[t, :] = np.array(
            [_softmax_prices(q, beta)[k] for k in K_LEVELS], dtype=np.float64
        )

    prices = _softmax_prices(q, beta)
    k_hat = max(prices, key=prices.get)
    saved_path: str | None = None
    if save_tensor_path:
        np.save(save_tensor_path, popularity_level_probability)
        saved_path = save_tensor_path

    saved_prices_path: str | None = None
    if save_prices_path:
        np.save(save_prices_path, prices_over_time)
        saved_prices_path = save_prices_path

    return MarketResult(
        q=q,
        prices=prices,
        prices_over_time=prices_over_time,
        k_hat=k_hat,
        popularity_level_probability=popularity_level_probability,
        popularity_level_probability_path=saved_path,
        prices_over_time_path=saved_prices_path,
    )


def agentic_market(image_path: str, expert_domain: str | None = None) -> dict[str, Any]:
    """
    Convenience wrapper that matches the project theme.
    `expert_domain` is accepted but not needed because the market uses all domains and d(x).
    """
    _ = expert_domain  # intentionally unused
    res = run_agentic_market(image_path)
    return {"popularity_level_probabilities": res.prices}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run the agentic LMSR market for one image.")
    p.add_argument("image_path", nargs="?", default="train/3175@N73/30916.jpg")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--T", type=int, default=2)
    p.add_argument("--num-other-agents", type=int, default=1)
    p.add_argument("--tau", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--save-tensor",
        default="popularity_level_probability.npy",
        help="Where to save popularity_level_probability tensor (.npy). Use empty string to disable.",
    )
    p.add_argument(
        "--save-prices",
        default="prices_over_time.npy",
        help="Where to save prices over time (T,5) array (.npy). Use empty string to disable.",
    )
    p.add_argument("--api-key", default=None, help="OpenAI API key (passed directly).")
    args = p.parse_args()

    if args.api_key:
        llm.OPENAI_API_KEY = args.api_key

    result = run_agentic_market(
        args.image_path,
        beta=args.beta,
        T=args.T,
        num_other_agents=args.num_other_agents,
        tau=args.tau,
        seed=args.seed,
        save_tensor_path=(args.save_tensor or None),
        save_prices_path=(args.save_prices or None),
    )
    print(
        json.dumps(
            {
                "q": result.q,
                "p_T": result.prices,
                "k_hat": result.k_hat,
                "popularity_level_probability_path": result.popularity_level_probability_path,
                "prices_over_time_path": result.prices_over_time_path,
            },
            ensure_ascii=False,
        )
    )

