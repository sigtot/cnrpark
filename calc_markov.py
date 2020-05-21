#!/usr/bin/env python3
import re
import numpy as np
from statistics import NormalDist

from typing import List, Dict, Optional, Tuple, Iterable

start_hour = 6
end_hour = 20


def main():
    T = Dict[str, List[int]]
    cam_dates_spots: T = dict()
    with open("./LABELS/all.txt", "r") as f:
        for line in f.readlines():
            line_splt: List[str] = line.split()
            img, label = line_splt[0], int(line_splt[1])
            matches = re.match("[A-Z]+\/[\d|-]+\/camera\d+\/[A-Z]_([\d|-]+)_([\d|.]+)_([A-Z]\d+)_(\d+)", img)
            date, time, cam, spot = matches.groups()
            d_str = f"{cam}_{date}_{spot}"
            if cam_dates_spots.get(d_str) is None:
                cam_dates_spots[d_str] = init_cam_array()
            cam_dates_spots[d_str][time_to_index(time)] = label

    pi_fts, pi_tfs = calc_pi_list(cam_dates_spots.values())
    pi_ft = float(np.mean(pi_fts))
    pi_tf = float(np.mean(pi_tfs))
    sigma_ft = float(np.var(pi_fts))
    sigma_tf = float(np.var(pi_tfs))
    pi_ff = 1 - pi_ft
    pi_tt = 1 - pi_tf
    Pi = np.array([[pi_ff, pi_ft], [pi_tf, pi_tt]])

    print(f"Pi = {Pi}")
    print(f"sigmas = [{sigma_ft}, {sigma_tf}]")
    print(f"cost from dist: {cost(list(cam_dates_spots.values())[:5], pi_ft, pi_tf, sigma_ft, sigma_tf)}")

    fake_series = [
        [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
        [1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1],
    ]
    print(f"fake cost: {cost(fake_series, pi_ft, pi_tf, sigma_ft, sigma_tf)}")

    zero_series = [[0] * 12, [0] * 12, [0] * 12, [0] * 12, [0] * 12]
    print(f"zero cost: {cost(zero_series, pi_ft, pi_tf, sigma_ft, sigma_tf)}")

    likely_series = [
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    ]
    print(f"likely cost: {cost(likely_series, pi_ft, pi_tf, sigma_ft, sigma_tf)}")


def cost(time_series_list: Iterable[List[Optional[int]]], pi_ft, pi_tf, sigma_ft, sigma_tf):
    pi_ft_hats, pi_tf_hats = calc_pi_list(time_series_list)
    ovl_ft = calc_ovl(pi_ft_hats, pi_ft, sigma_ft)
    ovl_tf = calc_ovl(pi_tf_hats, pi_tf, sigma_tf)
    return ovl_ft + ovl_tf


def calc_ovl(pi_list: List[float], pi: float, sigma: float, ) -> float:
    tol = 10e-7
    if len(pi_list) == 0:
        return 0
    pi_hat = np.mean(pi_list)
    sigma_hat = np.var(pi_list)
    if sigma_hat > tol:
        return NormalDist(mu=pi, sigma=sigma).overlap(NormalDist(mu=pi_hat, sigma=sigma_hat))
    else:
        return 0


def calc_pi_list(time_series_list: Iterable[List[Optional[int]]]) -> Tuple[List[float], List[float]]:
    pi_fts: List[float] = []
    pi_tfs: List[float] = []

    for time_series in time_series_list:
        pi_ft, pi_tf = calc_pi(time_series)
        if pi_ft is not None:
            pi_fts.append(pi_ft)
        if pi_tf is not None:
            pi_tfs.append(pi_tf)
    return pi_fts, pi_tfs


def calc_pi(time_series: List[Optional[int]]) -> Tuple[Optional[float], Optional[float]]:
    n_ff, n_ft, n_tf, n_tt = 0, 0, 0, 0
    for v_0, v_1 in zip(time_series[:-1], time_series[1:]):
        if v_0 is None or v_1 is None:
            continue
        if (v_0, v_1) == (0, 0):
            n_ff += 1
        elif (v_0, v_1) == (0, 1):
            n_ft += 1
        elif (v_0, v_1) == (1, 0):
            n_tf += 1
        elif (v_0, v_1) == (1, 1):
            n_tt += 1
    pi_ft, pi_tf = None, None
    try:
        pi_ft = n_ft / (n_ft + n_ff)
    except ZeroDivisionError:
        pass
    try:
        pi_tf = n_tf / (n_tf + n_tt)
    except ZeroDivisionError:
        pass
    return pi_ft, pi_tf


def get_transitions(time_series: List[int]) -> List[Tuple[int, int]]:
    return list(zip(time_series[:-1], time_series[1:]))


def likelihood_transitions(transitions: List[Tuple[int, int]], Pi) -> float:
    likelihoods = [likelihood(transition, Pi) for transition in transitions]
    return np.product(likelihoods)


def likelihood_series(time_series: List[int], Pi: np.array) -> float:
    likelihoods = [likelihood(transition, Pi) for transition in get_transitions(time_series)]
    return np.product(likelihoods)


def likelihood(transition: Tuple[int, int], Pi: np.array) -> float:
    return Pi[transition]


def init_cam_array() -> List[Optional[int]]:
    return [None] * 2 * (end_hour - start_hour)


def time_to_index(time_str: str) -> int:
    h, m = int(time_str.split(".")[0]), int(time_str.split(".")[1])
    dh = h - start_hour
    return 2 * dh + (m // 30)


if __name__ == "__main__":
    main()
