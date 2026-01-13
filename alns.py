"""
CVRP XL Solver with ALNS Integration

Complete integration of:
- Stage 1: Sweep clustering + PyVRP + SPP (unchanged)
- Stage 2: ALNS-based LNS with adaptive operator selection + SPP cycles
- Stage 3: Hexaly refinement (unchanged)

New components integrated:
- SISR (Slack Induction by String Removals)
- ALNS (Adaptive Large Neighborhood Search)
- New repair operators (greedy, regret-k, perturbed, blinks, weighted)
- New local search operators (cross-exchange, ejection chains, route elimination, icross)
- Worst removal destroy operator

Author: Eduardo (CVRPLib BKS Challenge 2026)
"""

import math
import time
import random
import heapq
import os
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Set, Callable

import gurobipy as gp
from gurobipy import GRB

from pyvrp import Model
from pyvrp.stop import MaxRuntime

# Import CVRP parser
from cvrp_parser import load_cvrp_instance
import vrplib


# =============================================================================
# SECTION 1: DISTANCE MATRIX & KNN (unchanged)
# =============================================================================

def build_distance_matrix(coords):
    C = np.asarray(coords, dtype=np.int64)
    dx = C[:, 0][:, None] - C[:, 0][None, :]
    dy = C[:, 1][:, None] - C[:, 1][None, :]
    D = np.rint(np.sqrt(dx * dx + dy * dy)).astype(np.int64)
    np.fill_diagonal(D, 0)
    return D


def build_knn_from_D(D, k=40, depot=0):
    n = D.shape[0]
    knn = [[] for _ in range(n)]
    for i in range(n):
        order = np.argsort(D[i])
        order = order[order != i]
        knn[i] = list(order[:k])
    return knn


# =============================================================================
# SECTION 2: COST FUNCTIONS (unchanged)
# =============================================================================

def route_cost_D(route, D, depot=0):
    if not route:
        return 0
    c = D[depot, route[0]]
    for u, v in zip(route, route[1:]):
        c += D[u, v]
    c += D[route[-1], depot]
    return int(c)


def total_cost_D(routes, D, depot=0):
    return sum(route_cost_D(r, D, depot) for r in routes)


# =============================================================================
# SECTION 3: POOL MANAGEMENT (unchanged)
# =============================================================================

def canonical_route(route):
    t = tuple(route)
    tr = tuple(reversed(route))
    return t if t < tr else tr


def add_routes_to_pool_safe(routes, route_pool, D, n_customers, depot=0,
                            max_pool=200000, min_cover=2):
    for r in routes:
        if not r:
            continue
        rt = canonical_route(r)
        if rt not in route_pool:
            route_pool[rt] = int(route_cost_D(list(rt), D, depot))

    if len(route_pool) <= max_pool:
        return

    cover = defaultdict(list)
    for rt, c in route_pool.items():
        for i in rt:
            if 1 <= i <= n_customers:
                cover[i].append((c, rt))

    protected = set()
    for i in range(1, n_customers + 1):
        if cover[i]:
            cover[i].sort(key=lambda x: x[0])
            for _, rt in cover[i][:min_cover]:
                protected.add(rt)

    remaining_slots = max_pool - len(protected)
    if remaining_slots < 0:
        keep = protected
    else:
        candidates = [(c, rt) for rt, c in route_pool.items() if rt not in protected]
        extra = heapq.nsmallest(remaining_slots, candidates, key=lambda x: x[0])
        keep = set(protected) | {rt for _, rt in extra}

    new_pool = {rt: route_pool[rt] for rt in keep}
    route_pool.clear()
    route_pool.update(new_pool)


# =============================================================================
# SECTION 4: SISR IMPLEMENTATION
# =============================================================================

class SISR:
    """
    Slack Induction by String Removals for CVRP.
    
    Based on Christiaens & Vanden Berghe (2020).
    """
    
    def __init__(
        self,
        D: np.ndarray,
        demand: np.ndarray,
        capacity: int,
        depot: int = 0,
        c_bar: float = 10.0,
        L_max: int = 10,
        alpha: float = 0.5,
        beta: float = 0.01,
        gamma: float = 0.01,
    ):
        self.D = D
        self.demand = np.asarray(demand, dtype=np.int64)
        self.capacity = capacity
        self.depot = depot
        self.c_bar = c_bar
        self.L_max = L_max
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n = len(D)
        self.adj = self._build_adjacency_lists()

    def _build_adjacency_lists(self) -> List[List[int]]:
        adj = []
        customers = [i for i in range(self.n) if i != self.depot]
        for i in range(self.n):
            if i == self.depot:
                adj.append([])
            else:
                sorted_customers = sorted(customers, key=lambda j: self.D[i, j])
                adj.append(sorted_customers)
        return adj

    def ruin(self, routes: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
        """Remove adjacent strings from multiple routes."""
        if not routes or all(len(r) == 0 for r in routes):
            return routes, []

        customer_to_route: Dict[int, Tuple[int, int]] = {}
        for ri, route in enumerate(routes):
            for pi, customer in enumerate(route):
                customer_to_route[customer] = (ri, pi)

        avg_tour_cardinality = np.mean([len(r) for r in routes if r])
        l_max_s = min(self.L_max, avg_tour_cardinality)
        k_max_s = max(1, 4 * self.c_bar / (1 + l_max_s) - 1)
        k_s = int(random.uniform(1, k_max_s + 1))

        all_customers = [c for r in routes for c in r]
        if not all_customers:
            return routes, []
        c_seed = random.choice(all_customers)

        removed: List[int] = []
        ruined_routes: Set[int] = set()

        for c in self.adj[c_seed]:
            if len(ruined_routes) >= k_s:
                break

            if c not in customer_to_route:
                continue

            ri, _ = customer_to_route[c]
            if ri in ruined_routes:
                continue

            route = routes[ri]
            if not route:
                continue

            l_max_t = min(len(route), l_max_s)
            l_t = int(random.uniform(1, l_max_t + 1))

            try:
                c_star_pos = route.index(c)
            except ValueError:
                continue

            if random.random() < self.alpha:
                string_removed = self._remove_split_string(route, c_star_pos, l_t)
            else:
                string_removed = self._remove_string(route, c_star_pos, l_t)

            removed.extend(string_removed)
            ruined_routes.add(ri)

            for cust in string_removed:
                if cust in customer_to_route:
                    del customer_to_route[cust]

        routes = [r for r in routes if r]
        return routes, removed

    def _remove_string(self, route: List[int], c_star_pos: int, length: int) -> List[int]:
        if length > len(route):
            length = len(route)
        if length == 0:
            return []

        min_start = max(0, c_star_pos - length + 1)
        max_start = min(len(route) - length, c_star_pos)

        if max_start < min_start:
            start = max(0, len(route) - length)
        else:
            start = random.randint(min_start, max_start)

        removed = route[start:start + length]
        del route[start:start + length]
        return removed

    def _remove_split_string(self, route: List[int], c_star_pos: int, length: int) -> List[int]:
        if length >= len(route):
            removed = route[:]
            route.clear()
            return removed
        if length == 0:
            return []

        m = 1
        m_max = len(route) - length
        while m < m_max and random.random() >= self.beta:
            m += 1

        total_span = length + m
        if total_span > len(route):
            total_span = len(route)
            m = max(0, total_span - length)

        if m == 0:
            return self._remove_string(route, c_star_pos, length)

        min_start = max(0, c_star_pos - total_span + 1)
        max_start = min(len(route) - total_span, c_star_pos)

        if max_start < min_start:
            start = max(0, len(route) - total_span)
        else:
            start = random.randint(min_start, max_start)

        preserve_offset = random.randint(0, length)
        preserve_start = start + preserve_offset
        preserve_end = min(preserve_start + m, start + total_span)

        removed = route[start:preserve_start] + route[preserve_end:start + total_span]

        if preserve_end < start + total_span:
            del route[preserve_end:start + total_span]
        if start < preserve_start:
            del route[start:preserve_start]

        return removed

    def recreate(self, routes: List[List[int]], removed: List[int]) -> List[List[int]]:
        """Reinsert using greedy with blinks."""
        if not removed:
            return routes

        removed = self._sort_customers(removed)
        loads = [sum(self.demand[c] for c in r) for r in routes]

        for customer in removed:
            best_pos = None
            best_cost = float('inf')
            best_route = None

            route_order = list(range(len(routes)))
            random.shuffle(route_order)

            for ri in route_order:
                route = routes[ri]
                if loads[ri] + self.demand[customer] > self.capacity:
                    continue

                for pos in range(len(route) + 1):
                    if random.random() < self.gamma:
                        continue

                    cost = self._insertion_cost(route, pos, customer)
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = pos
                        best_route = ri

            if best_route is None:
                routes.append([customer])
                loads.append(int(self.demand[customer]))
            else:
                routes[best_route].insert(best_pos, customer)
                loads[best_route] += int(self.demand[customer])

        return routes

    def _sort_customers(self, customers: List[int]) -> List[int]:
        customers = customers[:]
        choice = random.choices(['random', 'demand', 'far', 'close'], weights=[4, 4, 2, 1])[0]

        if choice == 'random':
            random.shuffle(customers)
        elif choice == 'demand':
            customers.sort(key=lambda c: self.demand[c], reverse=True)
        elif choice == 'far':
            customers.sort(key=lambda c: self.D[self.depot, c], reverse=True)
        elif choice == 'close':
            customers.sort(key=lambda c: self.D[self.depot, c])

        return customers

    def _insertion_cost(self, route: List[int], pos: int, customer: int) -> float:
        if not route:
            return 2 * self.D[self.depot, customer]

        prev = self.depot if pos == 0 else route[pos - 1]
        next_ = self.depot if pos == len(route) else route[pos]

        return float(self.D[prev, customer] + self.D[customer, next_] - self.D[prev, next_])


# =============================================================================
# SECTION 5: ALNS MANAGER
# =============================================================================

class ALNSManager:
    """Adaptive Large Neighborhood Search operator selection."""
    
    def __init__(
        self,
        destroy_operators: List[str],
        repair_operators: List[str],
        sigma1: float = 33.0,
        sigma2: float = 9.0,
        sigma3: float = 13.0,
        reaction_factor: float = 0.1,
        segment_length: int = 100,
        min_weight: float = 0.1,
        initial_weight: float = 1.0,
    ):
        self.destroy_ops = destroy_operators
        self.repair_ops = repair_operators
        
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        
        self.r = reaction_factor
        self.segment_length = segment_length
        self.min_weight = min_weight
        
        self.destroy_weights = {op: initial_weight for op in destroy_operators}
        self.repair_weights = {op: initial_weight for op in repair_operators}
        
        self.destroy_scores = {op: 0.0 for op in destroy_operators}
        self.repair_scores = {op: 0.0 for op in repair_operators}
        self.destroy_uses = {op: 0 for op in destroy_operators}
        self.repair_uses = {op: 0 for op in repair_operators}
        
        self.iteration = 0

    def select_destroy(self) -> str:
        return self._roulette_select(self.destroy_weights)

    def select_repair(self) -> str:
        return self._roulette_select(self.repair_weights)

    def _roulette_select(self, weights: Dict[str, float]) -> str:
        ops = list(weights.keys())
        w = [weights[op] for op in ops]
        total = sum(w)
        
        r = random.random() * total
        cumsum = 0.0
        for op, wi in zip(ops, w):
            cumsum += wi
            if r <= cumsum:
                return op
        return ops[-1]

    def record_outcome(
        self,
        destroy_op: str,
        repair_op: str,
        new_cost: float,
        current_cost: float,
        best_cost: float,
        accepted: bool,
        prev_best_cost: float,
    ):
        self.destroy_uses[destroy_op] += 1
        self.repair_uses[repair_op] += 1
        
        if new_cost < prev_best_cost:
            score = self.sigma1
        elif new_cost < current_cost:
            score = self.sigma2
        elif accepted and new_cost >= current_cost:
            score = self.sigma3
        else:
            score = 0.0
        
        self.destroy_scores[destroy_op] += score
        self.repair_scores[repair_op] += score
        
        self.iteration += 1
        
        if self.iteration % self.segment_length == 0:
            self._update_weights()

    def _update_weights(self):
        for op in self.destroy_ops:
            if self.destroy_uses[op] > 0:
                avg_score = self.destroy_scores[op] / self.destroy_uses[op]
                self.destroy_weights[op] = (
                    self.destroy_weights[op] * (1 - self.r) + self.r * avg_score
                )
            self.destroy_weights[op] = max(self.min_weight, self.destroy_weights[op])
        
        for op in self.repair_ops:
            if self.repair_uses[op] > 0:
                avg_score = self.repair_scores[op] / self.repair_uses[op]
                self.repair_weights[op] = (
                    self.repair_weights[op] * (1 - self.r) + self.r * avg_score
                )
            self.repair_weights[op] = max(self.min_weight, self.repair_weights[op])
        
        self.destroy_scores = {op: 0.0 for op in self.destroy_ops}
        self.repair_scores = {op: 0.0 for op in self.repair_ops}
        self.destroy_uses = {op: 0 for op in self.destroy_ops}
        self.repair_uses = {op: 0 for op in self.repair_ops}

    def get_weight_summary(self) -> str:
        d_str = ' '.join(f"{op[:4]}:{w:.1f}" for op, w in self.destroy_weights.items())
        r_str = ' '.join(f"{op[:4]}:{w:.1f}" for op, w in self.repair_weights.items())
        return f"D[{d_str}] R[{r_str}]"


# =============================================================================
# SECTION 6: DESTROY OPERATORS
# =============================================================================

def ruin_random(routes, remove_count):
    all_nodes = [(ri, pi, routes[ri][pi]) for ri in range(len(routes)) for pi in range(len(routes[ri]))]
    remove_count = min(remove_count, len(all_nodes))
    pick = random.sample(all_nodes, remove_count)
    pick.sort(reverse=True)
    removed = []
    for ri, pi, node in pick:
        removed.append(node)
        routes[ri].pop(pi)
    routes = [r for r in routes if r]
    return removed, routes


def ruin_related(routes, knn, remove_count, seed_node=None):
    nodes = [v for r in routes for v in r]
    if not nodes:
        return [], routes
    if seed_node is None:
        seed_node = random.choice(nodes)

    to_remove = {seed_node}
    frontier = [seed_node]
    while len(to_remove) < min(remove_count, len(nodes)) and frontier:
        x = frontier.pop()
        for nb in knn[x]:
            if nb in nodes and nb not in to_remove:
                to_remove.add(nb)
                frontier.append(nb)
            if len(to_remove) >= remove_count:
                break

    removed = []
    for ri in range(len(routes)):
        newr = []
        for v in routes[ri]:
            if v in to_remove:
                removed.append(v)
            else:
                newr.append(v)
        routes[ri] = newr
    routes = [r for r in routes if r]
    return removed, routes


def ruin_string(routes, remove_count):
    removed = []
    if remove_count <= 0:
        return removed, routes

    while remove_count > 0 and len(routes) > 0:
        ri = random.randrange(len(routes))
        r = routes[ri]
        L = len(r)

        if L < 2:
            removed.extend(r)
            remove_count -= L
            routes.pop(ri)
            continue

        max_seg = min(15, L)
        seg_len = random.randint(2, max_seg)
        start = random.randint(0, L - seg_len)

        seg = r[start:start + seg_len]
        removed.extend(seg)
        del r[start:start + seg_len]
        remove_count -= seg_len

        if not r:
            routes.pop(ri)

    return removed, routes


def ruin_worst(routes, D, remove_count, depot=0, noise=0.1):
    """Remove customers with highest removal cost."""
    removal_costs = []
    
    for ri, route in enumerate(routes):
        for pi, customer in enumerate(route):
            prev = depot if pi == 0 else route[pi - 1]
            next_ = depot if pi == len(route) - 1 else route[pi + 1]
            
            cost_saved = D[prev, customer] + D[customer, next_] - D[prev, next_]
            if noise > 0:
                cost_saved *= (1 + noise * (random.random() - 0.5))
            
            removal_costs.append((cost_saved, ri, pi, customer))
    
    removal_costs.sort(reverse=True)
    remove_count = min(remove_count, len(removal_costs))
    to_remove = set(removal_costs[i][3] for i in range(remove_count))
    
    removed = []
    new_routes = []
    
    for route in routes:
        new_route = []
        for customer in route:
            if customer in to_remove:
                removed.append(customer)
            else:
                new_route.append(customer)
        if new_route:
            new_routes.append(new_route)
    
    return removed, new_routes


def ruin_route(routes, demand, max_routes=2):
    """Remove entire routes (for fleet minimization)."""
    if len(routes) <= 1:
        return [], routes
    
    # Score routes by size (prefer smaller routes)
    scored = [(sum(demand[c] for c in r), len(r), ri) for ri, r in enumerate(routes)]
    scored.sort()
    
    removed = []
    to_remove_indices = set()
    
    for _, _, ri in scored[:max_routes]:
        removed.extend(routes[ri])
        to_remove_indices.add(ri)
    
    new_routes = [r for ri, r in enumerate(routes) if ri not in to_remove_indices]
    return removed, new_routes


# =============================================================================
# SECTION 7: REPAIR OPERATORS
# =============================================================================

def _insertion_cost(route, pos, customer, D, depot):
    if not route:
        return 2 * D[depot, customer]
    prev = depot if pos == 0 else route[pos - 1]
    next_ = depot if pos == len(route) else route[pos]
    return D[prev, customer] + D[customer, next_] - D[prev, next_]


def _best_insertion(route, customer, D, depot, noise=0.0, blink_rate=0.0):
    if not route:
        cost = 2 * D[depot, customer]
        if noise > 0:
            cost *= (1 + noise * (random.random() - 0.5))
        return 0, cost
    
    best_pos = 0
    best_cost = float('inf')
    
    for pos in range(len(route) + 1):
        if blink_rate > 0 and random.random() < blink_rate:
            continue
        
        cost = _insertion_cost(route, pos, customer, D, depot)
        if noise > 0:
            cost *= (1 + noise * (random.random() - 0.5))
        
        if cost < best_cost:
            best_cost = cost
            best_pos = pos
    
    if best_cost == float('inf'):
        best_pos = random.randint(0, len(route))
        best_cost = _insertion_cost(route, best_pos, customer, D, depot)
    
    return best_pos, best_cost


def recreate_greedy(routes, removed, D, demand, Q, depot=0):
    """Greedy insertion: cheapest position first."""
    if not removed:
        return routes
    
    removed = list(removed)
    random.shuffle(removed)
    loads = [sum(demand[c] for c in r) for r in routes]
    
    for customer in removed:
        best_route = -1
        best_pos = 0
        best_cost = float('inf')
        
        for ri, route in enumerate(routes):
            if loads[ri] + demand[customer] > Q:
                continue
            pos, cost = _best_insertion(route, customer, D, depot)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
                best_route = ri
        
        if best_route == -1:
            routes.append([customer])
            loads.append(int(demand[customer]))
        else:
            routes[best_route].insert(best_pos, customer)
            loads[best_route] += int(demand[customer])
    
    return routes


def recreate_regret_k(routes, removed, D, demand, Q, depot=0, k=2):
    """Regret-k insertion."""
    if not removed:
        return routes
    
    removed = list(removed)
    loads = [sum(demand[c] for c in r) for r in routes]
    
    while removed:
        best_customer = None
        best_score = float('-inf')
        best_route = -1
        best_pos = 0
        best_insert_cost = float('inf')
        
        for customer in removed:
            options = []
            
            for ri, route in enumerate(routes):
                if loads[ri] + demand[customer] > Q:
                    continue
                pos, cost = _best_insertion(route, customer, D, depot)
                options.append((cost, ri, pos))
            
            new_route_cost = 2 * D[depot, customer]
            options.append((new_route_cost, -1, 0))
            options.sort(key=lambda x: x[0])
            
            if len(options) >= k:
                regret = options[k-1][0] - options[0][0]
            elif len(options) >= 2:
                regret = options[-1][0] - options[0][0]
            else:
                regret = 0
            
            if regret > best_score or (regret == best_score and options[0][0] < best_insert_cost):
                best_score = regret
                best_customer = customer
                best_insert_cost = options[0][0]
                best_route = options[0][1]
                best_pos = options[0][2]
        
        if best_route == -1:
            routes.append([best_customer])
            loads.append(int(demand[best_customer]))
        else:
            routes[best_route].insert(best_pos, best_customer)
            loads[best_route] += int(demand[best_customer])
        
        removed.remove(best_customer)
    
    return routes


def recreate_regret2_D(routes, removed, D, demand, Q, depot=0):
    return recreate_regret_k(routes, removed, D, demand, Q, depot, k=2)


def recreate_regret3(routes, removed, D, demand, Q, depot=0):
    return recreate_regret_k(routes, removed, D, demand, Q, depot, k=3)


def recreate_regret4(routes, removed, D, demand, Q, depot=0):
    return recreate_regret_k(routes, removed, D, demand, Q, depot, k=4)


def recreate_perturbed(routes, removed, D, demand, Q, depot=0, noise=0.25):
    """Greedy with noise injection."""
    if not removed:
        return routes
    
    removed = list(removed)
    random.shuffle(removed)
    loads = [sum(demand[c] for c in r) for r in routes]
    
    for customer in removed:
        best_route = -1
        best_pos = 0
        best_cost = float('inf')
        
        for ri, route in enumerate(routes):
            if loads[ri] + demand[customer] > Q:
                continue
            pos, cost = _best_insertion(route, customer, D, depot, noise=noise)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
                best_route = ri
        
        if best_route == -1:
            routes.append([customer])
            loads.append(int(demand[customer]))
        else:
            routes[best_route].insert(best_pos, customer)
            loads[best_route] += int(demand[customer])
    
    return routes


def recreate_blinks(routes, removed, D, demand, Q, depot=0, gamma=0.01):
    """Greedy with position skipping (SISR-style blinks)."""
    if not removed:
        return routes
    
    removed = list(removed)
    # Sort by demand (largest first)
    removed.sort(key=lambda c: demand[c], reverse=True)
    
    loads = [sum(demand[c] for c in r) for r in routes]
    
    for customer in removed:
        best_route = -1
        best_pos = 0
        best_cost = float('inf')
        
        route_order = list(range(len(routes)))
        random.shuffle(route_order)
        
        for ri in route_order:
            route = routes[ri]
            if loads[ri] + demand[customer] > Q:
                continue
            pos, cost = _best_insertion(route, customer, D, depot, blink_rate=gamma)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
                best_route = ri
        
        if best_route == -1:
            routes.append([customer])
            loads.append(int(demand[customer]))
        else:
            routes[best_route].insert(best_pos, customer)
            loads[best_route] += int(demand[customer])
    
    return routes


# =============================================================================
# SECTION 8: LOCAL SEARCH OPERATORS (existing + new)
# =============================================================================

def remove_delta_at(route, pos, D, depot=0):
    v = route[pos]
    prev = depot if pos == 0 else route[pos - 1]
    nxt = depot if pos == len(route) - 1 else route[pos + 1]
    return int(-D[prev, v] - D[v, nxt] + D[prev, nxt])


def insert_delta_at(route, pos, v, D, depot=0):
    prev = depot if pos == 0 else route[pos - 1]
    nxt = depot if pos == len(route) else route[pos]
    return int(D[prev, v] + D[v, nxt] - D[prev, nxt])


def two_opt_D(route, D, depot=0, max_tries=300):
    if len(route) < 4:
        return route
    best = route
    best_cost = route_cost_D(best, D, depot)
    n = len(route)

    tries = 0
    improved = True
    while improved and tries < max_tries:
        improved = False
        tries += 1
        for _ in range(min(400, n * (n - 1) // 2)):
            i = random.randint(0, n - 3)
            j = random.randint(i + 2, n - 1)
            if i == 0 and j == n - 1:
                continue
            cand = best[:i + 1] + list(reversed(best[i + 1:j + 1])) + best[j + 1:]
            cc = route_cost_D(cand, D, depot)
            if cc < best_cost:
                best, best_cost = cand, cc
                improved = True
                break
    return best


def relocate_best_improvement(routes, loads, D, demand, Q, knn, depot=0, max_moves=4000):
    best_gain = 0
    best_move = None

    n_routes = len(routes)
    if n_routes <= 1:
        return False

    where = {}
    for r_idx, r in enumerate(routes):
        for p, v in enumerate(r):
            where[v] = (r_idx, p)

    moves_checked = 0
    for v, (rA, pA) in list(where.items()):
        if moves_checked >= max_moves:
            break

        candidate_routes = set()
        for nb in knn[v][:15]:
            if nb in where:
                candidate_routes.add(where[nb][0])
        if len(candidate_routes) < 3:
            candidate_routes.add(random.randrange(n_routes))

        for rB in candidate_routes:
            if rB == rA:
                continue
            if loads[rB] + demand[v] > Q:
                continue

            A = routes[rA]
            B = routes[rB]

            prevA = depot if pA == 0 else A[pA - 1]
            nextA = depot if pA == len(A) - 1 else A[pA + 1]
            remove_delta = -D[prevA, v] - D[v, nextA] + D[prevA, nextA]

            cand_pos = set([0, len(B)])
            for nb in knn[v][:10]:
                if nb in where and where[nb][0] == rB:
                    pb = where[nb][1]
                    cand_pos.update([pb, pb + 1])
            for _ in range(3):
                cand_pos.add(random.randint(0, len(B)))

            for pB in cand_pos:
                prevB = depot if pB == 0 else B[pB - 1]
                nextB = depot if pB == len(B) else B[pB]
                insert_delta = D[prevB, v] + D[v, nextB] - D[prevB, nextB]

                gain = -(remove_delta + insert_delta)
                moves_checked += 1
                if gain > best_gain:
                    best_gain = gain
                    best_move = (rA, pA, rB, pB)

    if best_move and best_gain > 0:
        rA, pA, rB, pB = best_move
        v = routes[rA].pop(pA)
        loads[rA] -= demand[v]
        if not routes[rA]:
            routes.pop(rA)
            loads.pop(rA)
            if rB > rA:
                rB -= 1
        routes[rB].insert(pB, v)
        loads[rB] += demand[v]
        return True

    return False


def swap_best_improvement(routes, loads, D, demand, Q, knn, depot=0, max_moves=5000):
    best_gain = 0
    best_move = None

    n_routes = len(routes)
    if n_routes <= 1:
        return False

    where = {}
    for r_idx, r in enumerate(routes):
        for p, v in enumerate(r):
            where[v] = (r_idx, p)

    moves_checked = 0
    for v, (rA, pA) in list(where.items()):
        if moves_checked >= max_moves:
            break

        A = routes[rA]
        prevA = depot if pA == 0 else A[pA - 1]
        nextA = depot if pA == len(A) - 1 else A[pA + 1]

        candidate_routes = set()
        for nb in knn[v][:15]:
            if nb in where:
                candidate_routes.add(where[nb][0])
        if len(candidate_routes) < 3:
            candidate_routes.add(random.randrange(n_routes))

        for rB in candidate_routes:
            if rB == rA:
                continue
            B = routes[rB]

            cand_nodes = []
            for nb in knn[v][:12]:
                if nb in where and where[nb][0] == rB:
                    cand_nodes.append(nb)
            if not cand_nodes:
                cand_nodes = [random.choice(B)]

            for w in cand_nodes:
                pB = where[w][1]

                newA = loads[rA] - demand[v] + demand[w]
                newB = loads[rB] - demand[w] + demand[v]
                if newA > Q or newB > Q:
                    continue

                prevB = depot if pB == 0 else B[pB - 1]
                nextB = depot if pB == len(B) - 1 else B[pB + 1]

                deltaA = (-D[prevA, v] - D[v, nextA] + D[prevA, w] + D[w, nextA])
                deltaB = (-D[prevB, w] - D[w, nextB] + D[prevB, v] + D[v, nextB])

                gain = -(deltaA + deltaB)
                moves_checked += 1
                if gain > best_gain:
                    best_gain = gain
                    best_move = (rA, pA, rB, pB)

    if best_move and best_gain > 0:
        rA, pA, rB, pB = best_move
        v = routes[rA][pA]
        w = routes[rB][pB]
        routes[rA][pA], routes[rB][pB] = w, v
        loads[rA] = loads[rA] - demand[v] + demand[w]
        loads[rB] = loads[rB] - demand[w] + demand[v]
        return True

    return False


def or_opt_best_improvement(routes, loads, D, demand, Q, knn, depot=0, seg_lens=(2, 3), max_moves=6000):
    best_gain = 0
    best_move = None

    n_routes = len(routes)
    if n_routes <= 1:
        return False

    where = {}
    for r_idx, r in enumerate(routes):
        for p, v in enumerate(r):
            where[v] = (r_idx, p)

    checked = 0
    for rA, A in enumerate(routes):
        LA = len(A)
        if LA < 2:
            continue

        for L in seg_lens:
            if LA < L:
                continue
            for start in range(0, LA - L + 1):
                if checked >= max_moves:
                    break

                seg = A[start:start + L]
                seg_demand = sum(demand[v] for v in seg)

                candidate_routes = set()
                pivot = seg[0]
                for nb in knn[pivot][:15]:
                    if nb in where:
                        candidate_routes.add(where[nb][0])
                if len(candidate_routes) < 3:
                    candidate_routes.add(random.randrange(n_routes))

                prevA = depot if start == 0 else A[start - 1]
                nextA = depot if start + L == LA else A[start + L]
                delta_remove = int(-D[prevA, seg[0]] - D[seg[-1], nextA] + D[prevA, nextA])

                for rB in candidate_routes:
                    if rB == rA:
                        continue
                    if loads[rB] + seg_demand > Q:
                        continue

                    B = routes[rB]

                    cand_pos = {0, len(B)}
                    for nb in knn[pivot][:10]:
                        if nb in where and where[nb][0] == rB:
                            pb = where[nb][1]
                            cand_pos.update([pb, pb + 1])
                    for _ in range(2):
                        cand_pos.add(random.randint(0, len(B)))

                    for posB in cand_pos:
                        prevB = depot if posB == 0 else B[posB - 1]
                        nextB = depot if posB == len(B) else B[posB]

                        delta_insert = int(D[prevB, seg[0]] + D[seg[-1], nextB] - D[prevB, nextB])

                        gain = -(delta_remove + delta_insert)
                        checked += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (rA, start, L, rB, posB)

            if checked >= max_moves:
                break
        if checked >= max_moves:
            break

    if best_move and best_gain > 0:
        rA, start, L, rB, posB = best_move

        seg = routes[rA][start:start + L]
        del routes[rA][start:start + L]
        loads[rA] -= sum(demand[v] for v in seg)

        if not routes[rA]:
            routes.pop(rA)
            loads.pop(rA)
            if rB > rA:
                rB -= 1

        routes[rB][posB:posB] = seg
        loads[rB] += sum(demand[v] for v in seg)
        return True

    return False


def two_opt_star_best_improvement(routes, loads, D, demand, Q, knn, depot=0, max_moves=7000):
    best_gain = 0
    best_move = None

    n_routes = len(routes)
    if n_routes <= 1:
        return False

    where = {}
    for r_idx, r in enumerate(routes):
        for p, v in enumerate(r):
            where[v] = (r_idx, p)

    checked = 0
    for rA, A in enumerate(routes):
        if len(A) < 2:
            continue

        for i in range(0, len(A) - 1):
            if checked >= max_moves:
                break
            a_i = A[i]
            a_next = A[i + 1]

            candidate_routes = set()
            for nb in knn[a_next][:15]:
                if nb in where:
                    candidate_routes.add(where[nb][0])
            if len(candidate_routes) < 3:
                candidate_routes.add(random.randrange(n_routes))

            tailA = A[i + 1:]
            tailA_dem = sum(demand[v] for v in tailA)
            headA_dem = loads[rA] - tailA_dem

            for rB in candidate_routes:
                if rB == rA:
                    continue
                B = routes[rB]
                if len(B) < 2:
                    continue

                for j in range(0, len(B) - 1):
                    b_j = B[j]
                    b_next = B[j + 1]

                    tailB = B[j + 1:]
                    tailB_dem = sum(demand[v] for v in tailB)
                    headB_dem = loads[rB] - tailB_dem

                    newA_load = headA_dem + tailB_dem
                    newB_load = headB_dem + tailA_dem
                    if newA_load > Q or newB_load > Q:
                        continue

                    delta = int(
                        -D[a_i, a_next] - D[b_j, b_next]
                        + D[a_i, b_next] + D[b_j, a_next]
                    )
                    gain = -delta
                    checked += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (rA, i, rB, j)

                    if checked >= max_moves:
                        break
                if checked >= max_moves:
                    break
        if checked >= max_moves:
            break

    if best_move and best_gain > 0:
        rA, i, rB, j = best_move
        A = routes[rA]
        B = routes[rB]

        tailA = A[i + 1:]
        tailB = B[j + 1:]

        newA = A[:i + 1] + tailB
        newB = B[:j + 1] + tailA

        routes[rA] = newA
        routes[rB] = newB

        loads[rA] = sum(demand[v] for v in newA)
        loads[rB] = sum(demand[v] for v in newB)

        routes[:] = [r for r in routes if r]
        loads[:] = [sum(demand[v] for v in r) for r in routes]
        return True

    return False


def swap_star_best_improvement(routes, loads, D, demand, Q, knn, depot=0, max_moves=8000):
    """SWAP*: swap with best reinsertion positions."""
    best_gain = 0
    best_move = None

    n_routes = len(routes)
    if n_routes <= 1:
        return False

    where = {}
    for r_idx, r in enumerate(routes):
        for p, v in enumerate(r):
            where[v] = (r_idx, p)

    def best_insertion_excluding(route, node, forbid_pos=None):
        if not route:
            return 0, int(2 * D[depot, node])
        best_pos, best_delta = 0, 10 ** 18
        for pos in range(0, len(route) + 1):
            if forbid_pos is not None and pos == forbid_pos:
                continue
            delta = insert_delta_at(route, pos, node, D, depot)
            if delta < best_delta:
                best_delta, best_pos = delta, pos
        return best_pos, int(best_delta)

    checked = 0
    for v, (rA, posA) in list(where.items()):
        if checked >= max_moves:
            break
        A = routes[rA]

        candidate_routes = set()
        for nb in knn[v][:18]:
            if nb in where:
                candidate_routes.add(where[nb][0])
        if len(candidate_routes) < 3:
            candidate_routes.add(random.randrange(n_routes))

        delta_remove_v = remove_delta_at(A, posA, D, depot)

        for rB in candidate_routes:
            if rB == rA:
                continue
            B = routes[rB]

            cand_ws = []
            for nb in knn[v][:14]:
                if nb in where and where[nb][0] == rB:
                    cand_ws.append(nb)
            if not cand_ws:
                cand_ws = [random.choice(B)]

            for w in cand_ws:
                posB = where[w][1]

                newA_load = loads[rA] - demand[v] + demand[w]
                newB_load = loads[rB] - demand[w] + demand[v]
                if newA_load > Q or newB_load > Q:
                    continue

                delta_remove_w = remove_delta_at(B, posB, D, depot)

                A_wo = A[:posA] + A[posA + 1:]
                B_wo = B[:posB] + B[posB + 1:]

                insPosA, delta_ins_w = best_insertion_excluding(A_wo, w, None)
                insPosB, delta_ins_v = best_insertion_excluding(B_wo, v, None)

                delta_total = int(delta_remove_v + delta_remove_w + delta_ins_w + delta_ins_v)
                gain = -delta_total
                checked += 1
                if gain > best_gain:
                    best_gain = gain
                    best_move = (rA, posA, rB, posB, insPosA, insPosB)

                if checked >= max_moves:
                    break
            if checked >= max_moves:
                break

    if best_move and best_gain > 0:
        rA, posA, rB, posB, insPosA, insPosB = best_move

        A = routes[rA]
        B = routes[rB]
        v = A[posA]
        w = B[posB]

        del A[posA]
        del B[posB]
        loads[rA] -= demand[v]
        loads[rB] -= demand[w]

        A.insert(insPosA, w)
        B.insert(insPosB, v)
        loads[rA] += demand[w]
        loads[rB] += demand[v]

        if not A:
            routes.pop(rA)
            loads.pop(rA)
            if rB > rA:
                rB -= 1
        if rB < len(routes) and not routes[rB]:
            routes.pop(rB)
            loads.pop(rB)

        return True

    return False


# NEW: Cross-exchange
def cross_exchange_best_improvement(routes, loads, D, demand, Q, knn, depot=0, max_seg_len=3, max_moves=5000):
    """Exchange segments of different lengths between two routes."""
    best_gain = 0
    best_move = None

    n_routes = len(routes)
    if n_routes < 2:
        return False

    where = {}
    for ri, route in enumerate(routes):
        for pi, v in enumerate(route):
            where[v] = (ri, pi)

    moves_checked = 0

    for rA in range(n_routes):
        A = routes[rA]
        if len(A) < 1:
            continue

        for startA in range(len(A)):
            if moves_checked >= max_moves:
                break

            for lenA in range(0, min(max_seg_len + 1, len(A) - startA + 1)):
                if moves_checked >= max_moves:
                    break

                segA = A[startA:startA + lenA] if lenA > 0 else []
                demandA = sum(demand[c] for c in segA)

                pivot = A[startA] if startA < len(A) else A[-1] if A else None
                if pivot is None:
                    continue

                candidate_routes = set()
                for nb in knn[pivot][:20]:
                    if nb in where:
                        candidate_routes.add(where[nb][0])

                for rB in candidate_routes:
                    if rB <= rA:
                        continue

                    B = routes[rB]
                    if len(B) < 1:
                        continue

                    for startB in range(len(B)):
                        for lenB in range(0, min(max_seg_len + 1, len(B) - startB + 1)):
                            if lenA == 0 and lenB == 0:
                                continue

                            segB = B[startB:startB + lenB] if lenB > 0 else []
                            demandB = sum(demand[c] for c in segB)

                            new_loadA = loads[rA] - demandA + demandB
                            new_loadB = loads[rB] - demandB + demandA

                            if new_loadA > Q or new_loadB > Q:
                                continue

                            prevA = depot if startA == 0 else A[startA - 1]
                            nextA = depot if startA + lenA >= len(A) else A[startA + lenA]
                            prevB = depot if startB == 0 else B[startB - 1]
                            nextB = depot if startB + lenB >= len(B) else B[startB + lenB]

                            if lenA > 0:
                                old_costA = D[prevA, segA[0]] + D[segA[-1], nextA]
                            else:
                                old_costA = D[prevA, nextA]

                            if lenB > 0:
                                old_costB = D[prevB, segB[0]] + D[segB[-1], nextB]
                            else:
                                old_costB = D[prevB, nextB]

                            if lenB > 0:
                                new_costA = D[prevA, segB[0]] + D[segB[-1], nextA]
                            else:
                                new_costA = D[prevA, nextA]

                            if lenA > 0:
                                new_costB = D[prevB, segA[0]] + D[segA[-1], nextB]
                            else:
                                new_costB = D[prevB, nextB]

                            gain = (old_costA + old_costB) - (new_costA + new_costB)
                            moves_checked += 1

                            if gain > best_gain:
                                best_gain = gain
                                best_move = (rA, startA, lenA, rB, startB, lenB)

    if best_move and best_gain > 0:
        rA, startA, lenA, rB, startB, lenB = best_move

        A = routes[rA]
        B = routes[rB]

        segA = A[startA:startA + lenA] if lenA > 0 else []
        segB = B[startB:startB + lenB] if lenB > 0 else []

        del A[startA:startA + lenA]
        del B[startB:startB + lenB]

        A[startA:startA] = segB
        B[startB:startB] = segA

        loads[rA] = sum(demand[c] for c in A)
        loads[rB] = sum(demand[c] for c in B)

        routes[:] = [r for r in routes if r]
        loads[:] = [sum(demand[c] for c in r) for r in routes]

        return True

    return False


# NEW: Route elimination
def route_elimination(routes, loads, D, demand, Q, depot=0, max_attempts=5):
    """Try to empty routes by redistributing customers."""
    if len(routes) <= 1:
        return False

    route_scores = []
    for ri, route in enumerate(routes):
        if not route:
            continue
        score = len(route) + 0.001 * loads[ri]
        route_scores.append((score, ri))

    route_scores.sort(key=lambda x: x[0])

    for _, target_ri in route_scores[:max_attempts]:
        target_route = routes[target_ri]
        if not target_route:
            continue

        customers = list(target_route)
        success = True
        insertions = []

        temp_loads = loads[:]

        for customer in customers:
            best_route = -1
            best_pos = 0
            best_cost = float('inf')

            for ri, route in enumerate(routes):
                if ri == target_ri:
                    continue

                if temp_loads[ri] + demand[customer] > Q:
                    continue

                temp_route = list(route)
                for ins_ri, ins_pos, ins_cust in insertions:
                    if ins_ri == ri:
                        temp_route.insert(ins_pos, ins_cust)

                for pos in range(len(temp_route) + 1):
                    cost = _insertion_cost(temp_route, pos, customer, D, depot)
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = pos
                        best_route = ri

            if best_route == -1:
                success = False
                break

            insertions.append((best_route, best_pos, customer))
            temp_loads[best_route] += demand[customer]

        if success:
            routes[target_ri] = []
            loads[target_ri] = 0

            by_route = defaultdict(list)
            for ri, pos, cust in insertions:
                by_route[ri].append((pos, cust))

            for ri, ins_list in by_route.items():
                # FIXED: No sorting, no offset - apply in planning order
                for pos, cust in ins_list:
                    routes[ri].insert(pos, cust)
                    loads[ri] += demand[cust]

            routes.pop(target_ri)
            loads.pop(target_ri)

            return True

    return False
# =============================================================================
# SECTION 9: COMBINED LOCAL SEARCH
# =============================================================================

def local_search_alns(routes, D, demand, Q, knn, depot=0, intra_2opt=True, max_rounds=3, use_new_ops=True):
    """Enhanced local search with all operators."""
    loads = [sum(demand[v] for v in r) for r in routes]

    if intra_2opt:
        for i in range(len(routes)):
            routes[i] = two_opt_D(routes[i], D, depot=depot, max_tries=220)

    for _ in range(max_rounds):
        moved = False

        moved |= relocate_best_improvement(routes, loads, D, demand, Q, knn, depot=depot, max_moves=2000)
        moved |= swap_best_improvement(routes, loads, D, demand, Q, knn, depot=depot, max_moves=2000)
        moved |= or_opt_best_improvement(routes, loads, D, demand, Q, knn, depot=depot, seg_lens=(2, 3), max_moves=2000)
        moved |= two_opt_star_best_improvement(routes, loads, D, demand, Q, knn, depot=depot, max_moves=2000)
        moved |= swap_star_best_improvement(routes, loads, D, demand, Q, knn, depot=depot, max_moves=2000)

        if use_new_ops:
            moved |= cross_exchange_best_improvement(routes, loads, D, demand, Q, knn, depot=depot, max_moves=1500)
            if _ % 2 == 0:
                moved |= route_elimination(routes, loads, D, demand, Q, depot=depot, max_attempts=3)

        if not moved:
            break

        if intra_2opt:
            for i in range(len(routes)):
                routes[i] = two_opt_D(routes[i], D, depot=depot, max_tries=200)

    return routes


# =============================================================================
# SECTION 10: INITIAL SOLUTION
# =============================================================================

def initial_sweep(coords, demand, Q, depot=0):
    n = len(coords)
    customers = [i for i in range(n) if i != depot]
    dx, dy = coords[depot]
    customers.sort(key=lambda i: math.atan2(coords[i][1] - dy, coords[i][0] - dx))

    routes = []
    cur, load = [], 0
    for i in customers:
        di = int(demand[i])
        if di > Q:
            raise ValueError(f"Demand of customer {i} ({di}) > capacity Q ({Q})")
        if load + di <= Q:
            cur.append(i)
            load += di
        else:
            routes.append(cur)
            cur, load = [i], di
    if cur:
        routes.append(cur)
    return routes


# =============================================================================
# SECTION 11: ALNS-BASED LNS
# =============================================================================

def lns_alns(
    coords, demand, Q, depot=0,
    iters=50000,
    time_limit=300.0,
    k_knn=60,
    seed=42,
    verbose=True,
    log_every=50,
    # ALNS config
    segment_length=100,
    reaction_factor=0.1,
    # Start/pool
    start_routes=None,
    D_in=None,
    route_pool_in=None,
    max_pool=100000,
    min_cover_pool=5,
    collect_every=1,
    # LS config
    ls_rounds=2,
    use_new_ls_ops=True,
):
    """
    ALNS-based LNS for CVRP with adaptive operator selection.
    
    Destroy operators: random, related, string, sisr, worst, route
    Repair operators: greedy, regret2, regret3, regret4, perturbed, blinks
    """
    random.seed(seed)
    np.random.seed(seed)

    n_customers = len(coords) - 1
    demand_arr = np.array(demand, dtype=np.int64)

    # Build distance matrix and kNN
    D = D_in if D_in is not None else build_distance_matrix(coords)
    knn = build_knn_from_D(D, k=k_knn, depot=depot)

    # Initialize SISR
    sisr = SISR(D, demand_arr, Q, depot=depot, c_bar=0.10 * n_customers, L_max=15)

    # Define operators
    destroy_names = ['random', 'related', 'string', 'sisr', 'worst']
    repair_names = ['greedy', 'regret2', 'regret3', 'perturbed', 'blinks']

    # Initialize ALNS manager
    alns = ALNSManager(
        destroy_operators=destroy_names,
        repair_operators=repair_names,
        segment_length=segment_length,
        reaction_factor=reaction_factor,
    )

    # Initialize solution
    if start_routes is None:
        routes = initial_sweep(coords, demand, Q, depot)
    else:
        routes = [list(r) for r in start_routes]

    routes = local_search_alns(routes, D, demand_arr, Q, knn, depot, 
                               intra_2opt=True, max_rounds=ls_rounds, use_new_ops=use_new_ls_ops)

    best_routes = [r[:] for r in routes]
    best_cost = total_cost_D(best_routes, D, depot)

    cur_routes = [r[:] for r in routes]
    cur_cost = total_cost_D(cur_routes, D, depot)

    # Route pool
    route_pool = route_pool_in if route_pool_in is not None else {}
    add_routes_to_pool_safe(cur_routes, route_pool, D, n_customers, depot, max_pool=max_pool, min_cover=min_cover_pool)

    # SA parameters
    T0 = 0.002 * best_cost

    # Adaptive remove fraction
    rf_base = (0.10, 0.35)
    no_improve = 0

    start_time = time.time()

    for it in range(1, iters + 1):
        if time.time() - start_time > time_limit:
            break

        prev_best = best_cost
        trial = [r[:] for r in cur_routes]

        # =====================
        # ALNS Operator Selection
        # =====================
        destroy_op = alns.select_destroy()
        repair_op = alns.select_repair()

        # =====================
        # DESTROY Phase
        # =====================
        # Adaptive remove count
        rf = rf_base
        if no_improve > 500:
            rf = (min(0.40, rf[0] + 0.10), min(0.50, rf[1] + 0.10))
        elif no_improve > 200:
            rf = (min(0.30, rf[0] + 0.05), min(0.40, rf[1] + 0.05))

        rc = max(1, int(random.uniform(rf[0], rf[1]) * n_customers))

        if destroy_op == 'random':
            removed, trial = ruin_random(trial, rc)
        elif destroy_op == 'related':
            nodes = [v for r in trial for v in r]
            seed_node = random.choice(nodes) if nodes else None
            removed, trial = ruin_related(trial, knn, rc, seed_node=seed_node)
        elif destroy_op == 'string':
            removed, trial = ruin_string(trial, rc)
        elif destroy_op == 'sisr':
            trial, removed = sisr.ruin(trial)
        elif destroy_op == 'worst':
            removed, trial = ruin_worst(trial, D, rc, depot)
        else:
            removed, trial = ruin_random(trial, rc)

        # =====================
        # REPAIR Phase
        # =====================
        if repair_op == 'greedy':
            trial = recreate_greedy(trial, removed, D, demand_arr, Q, depot)
        elif repair_op == 'regret2':
            trial = recreate_regret2_D(trial, removed, D, demand_arr, Q, depot)
        elif repair_op == 'regret3':
            trial = recreate_regret3(trial, removed, D, demand_arr, Q, depot)
        elif repair_op == 'regret4':
            trial = recreate_regret4(trial, removed, D, demand_arr, Q, depot)
        elif repair_op == 'perturbed':
            trial = recreate_perturbed(trial, removed, D, demand_arr, Q, depot, noise=0.25)
        elif repair_op == 'blinks':
            trial = recreate_blinks(trial, removed, D, demand_arr, Q, depot, gamma=0.01)
        else:
            trial = recreate_regret2_D(trial, removed, D, demand_arr, Q, depot)

        # =====================
        # LOCAL SEARCH Phase
        # =====================
        trial = local_search_alns(trial, D, demand_arr, Q, knn, depot, 
                                  intra_2opt=True, max_rounds=ls_rounds, use_new_ops=use_new_ls_ops)
        trial_cost = total_cost_D(trial, D, depot)

        # =====================
        # ACCEPTANCE (Simulated Annealing)
        # =====================
        accepted = False
        if trial_cost < cur_cost:
            accepted = True
        else:
            T = max(1e-9, T0 * (1 - it / iters))
            if random.random() < math.exp(-(trial_cost - cur_cost) / T):
                accepted = True

        # =====================
        # Update ALNS
        # =====================
        alns.record_outcome(
            destroy_op=destroy_op,
            repair_op=repair_op,
            new_cost=trial_cost,
            current_cost=cur_cost,
            best_cost=best_cost,
            accepted=accepted,
            prev_best_cost=prev_best,
        )

        # =====================
        # Update Solutions
        # =====================
        if accepted:
            cur_routes, cur_cost = trial, trial_cost

        if cur_cost < best_cost:
            best_cost = cur_cost
            best_routes = [r[:] for r in cur_routes]
            no_improve = 0
        else:
            no_improve += 1

        # Update pool
        if it % collect_every == 0:
            add_routes_to_pool_safe(trial, route_pool, D, n_customers, depot, max_pool=max_pool, min_cover=min_cover_pool)
            add_routes_to_pool_safe(best_routes, route_pool, D, n_customers, depot, max_pool=max_pool, min_cover=min_cover_pool)

        # =====================
        # Logging
        # =====================
        if verbose and it % log_every == 0:
            elapsed = time.time() - start_time
            weights_str = alns.get_weight_summary()
            print(f"[ALNS] it {it:6d} | cur {cur_cost:,.0f} | best {best_cost:,.0f} "
                  f"| routes {len(best_routes):3d} | pool {len(route_pool):5d} "
                  f"| {weights_str} | t={elapsed:.1f}s")

    return best_routes, best_cost, D, route_pool


# =============================================================================
# SECTION 12: INTENSIFY FROM SPP (with ALNS)
# =============================================================================

def intensify_from_spp_alns(
    spp_routes, D, coords, demand, Q,
    route_pool,
    time_limit=120.0,
    iters=30000,
    k_knn=60,
    seed=42,
    verbose=True,
    max_pool=100000,
    collect_every=1,
    min_cover=5,
    depot=0,
    segment_length=100,
    reaction_factor=0.1,
):
    """Intensify from SPP solution using ALNS."""
    random.seed(seed)
    np.random.seed(seed)

    n_customers = len(coords) - 1
    demand_arr = np.array(demand, dtype=np.int64)
    knn = build_knn_from_D(D, k=k_knn, depot=depot)

    # Initialize SISR
    sisr = SISR(D, demand_arr, Q, depot=depot, c_bar=0.05 * n_customers, L_max=12)

    # Intensification uses smaller perturbations
    destroy_names = ['random', 'related', 'string', 'sisr', 'worst']
    repair_names = ['greedy', 'regret2', 'regret3', 'blinks']

    alns = ALNSManager(
        destroy_operators=destroy_names,
        repair_operators=repair_names,
        segment_length=segment_length,
        reaction_factor=reaction_factor,
    )

    cur_routes = [list(r) for r in spp_routes]
    cur_routes = local_search_alns(cur_routes, D, demand_arr, Q, knn, depot, intra_2opt=True, max_rounds=2)

    cur_cost = total_cost_D(cur_routes, D, depot)
    best_routes = [r[:] for r in cur_routes]
    best_cost = cur_cost

    add_routes_to_pool_safe(best_routes, route_pool, D, n_customers, depot=depot, max_pool=max_pool, min_cover=min_cover)

    # Smaller remove fraction for intensification
    rf_base = (0.03, 0.15)

    T0 = 0.002 * best_cost

    start = time.time()
    log_every = 50

    for it in range(1, iters + 1):
        if time.time() - start > time_limit:
            break

        prev_best = best_cost
        trial = [r[:] for r in cur_routes]

        destroy_op = alns.select_destroy()
        repair_op = alns.select_repair()

        rc = max(1, int(random.uniform(rf_base[0], rf_base[1]) * n_customers))

        if destroy_op == 'random':
            removed, trial = ruin_random(trial, rc)
        elif destroy_op == 'related':
            nodes = [v for r in trial for v in r]
            seed_node = random.choice(nodes) if nodes else None
            removed, trial = ruin_related(trial, knn, rc, seed_node=seed_node)
        elif destroy_op == 'string':
            removed, trial = ruin_string(trial, rc)
        elif destroy_op == 'sisr':
            trial, removed = sisr.ruin(trial)
        elif destroy_op == 'worst':
            removed, trial = ruin_worst(trial, D, rc, depot)
        else:
            removed, trial = ruin_random(trial, rc)

        if repair_op == 'greedy':
            trial = recreate_greedy(trial, removed, D, demand_arr, Q, depot)
        elif repair_op == 'regret2':
            trial = recreate_regret2_D(trial, removed, D, demand_arr, Q, depot)
        elif repair_op == 'regret3':
            trial = recreate_regret3(trial, removed, D, demand_arr, Q, depot)
        elif repair_op == 'blinks':
            trial = recreate_blinks(trial, removed, D, demand_arr, Q, depot, gamma=0.01)
        else:
            trial = recreate_regret2_D(trial, removed, D, demand_arr, Q, depot)

        trial = local_search_alns(trial, D, demand_arr, Q, knn, depot, intra_2opt=True, max_rounds=1)
        trial_cost = total_cost_D(trial, D, depot)

        accepted = False
        if trial_cost < cur_cost:
            accepted = True
        else:
            T = max(1e-9, T0 * (1 - it / iters))
            if random.random() < math.exp(-(trial_cost - cur_cost) / T):
                accepted = True

        alns.record_outcome(
            destroy_op=destroy_op,
            repair_op=repair_op,
            new_cost=trial_cost,
            current_cost=cur_cost,
            best_cost=best_cost,
            accepted=accepted,
            prev_best_cost=prev_best,
        )

        if accepted:
            cur_routes, cur_cost = trial, trial_cost

        if (it % collect_every) == 0:
            add_routes_to_pool_safe(trial, route_pool, D, n_customers, depot=depot, max_pool=max_pool, min_cover=min_cover)
            add_routes_to_pool_safe(best_routes, route_pool, D, n_customers, depot=depot, max_pool=max_pool, min_cover=min_cover)

        if cur_cost < best_cost:
            best_cost = cur_cost
            best_routes = [r[:] for r in cur_routes]

        if verbose and it % log_every == 0:
            weights_str = alns.get_weight_summary()
            print(f"[INT] it {it:6d} | cur {cur_cost:,.0f} | best {best_cost:,.0f} "
                  f"| pool {len(route_pool):5d} | {weights_str} | t={(time.time() - start):.1f}s")

    return best_routes, best_cost, route_pool


# =============================================================================
# SECTION 13: SPP SOLVER (unchanged from your original)
# =============================================================================

def solve_spp_gurobi(route_pool, n_customers, time_limit=60.0, mip_gap=0.0, verbose=True,
                     prev_selected_routes=None, fallback_routes=None):
    """Set Partitioning Problem solver."""
    if isinstance(route_pool, dict):
        pool_dict = route_pool
    else:
        raise TypeError("route_pool must be a dict {route_tuple: cost}")

    filtered = {}
    for rt, c in pool_dict.items():
        if rt is None:
            continue
        rt = tuple(int(v) for v in rt)
        if len(rt) == 0:
            continue
        ok = True
        for v in rt:
            if v < 1 or v > n_customers:
                ok = False
                break
        if not ok:
            continue
        filtered[rt] = float(c)

    if len(filtered) == 0:
        raise ValueError("route_pool empty or no valid routes.")

    routes = list(filtered.keys())
    costs = [filtered[r] for r in routes]

    cover = [[] for _ in range(n_customers + 1)]
    for ridx, rt in enumerate(routes):
        for i in rt:
            cover[i].append(ridx)

    missing = [i for i in range(1, n_customers + 1) if len(cover[i]) == 0]
    if missing:
        raise ValueError(f"SPP infeasible: customers without coverage (e.g. {missing[:10]}...)")

    m = gp.Model("SPP_CVRP")
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.MIPFocus = 1
    m.Params.Presolve = 2
    m.Params.CliqueCuts = 2

    x = m.addVars(len(routes), vtype=GRB.BINARY, name="x")
    m.setObjective(gp.quicksum(costs[r] * x[r] for r in range(len(routes))), GRB.MINIMIZE)

    for i in range(1, n_customers + 1):
        m.addConstr(gp.quicksum(x[r] for r in cover[i]) == 1, name=f"cover_{i}")

    start_set = set()
    if prev_selected_routes is not None:
        start_set |= {tuple(map(int, r)) for r in prev_selected_routes}
    if (not start_set) and (fallback_routes is not None):
        start_set |= {tuple(map(int, r)) for r in fallback_routes}

    if start_set:
        pool_set = set(routes)
        for r in range(len(routes)):
            x[r].Start = 1.0 if routes[r] in pool_set and routes[r] in start_set else 0.0

    m.optimize()

    if m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        raise ValueError("SPP ended INFEASIBLE/INF_OR_UNBD.")
    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        raise ValueError(f"SPP ended with status={m.Status}")

    selected_routes = []
    selected_set = set()
    for r in range(len(routes)):
        if x[r].X > 0.5:
            selected_set.add(routes[r])
            selected_routes.append(list(routes[r]))

    best_cost = float(m.ObjVal) if m.SolCount > 0 else float("inf")

    return selected_routes, best_cost, m, selected_set


# =============================================================================
# SECTION 14: STAGE 1 - PyVRP + SPP (unchanged from your original)
# =============================================================================

def sweep_clustering(coords, demand, k, depot=0):
    coords = np.asarray(coords, dtype=np.float64)
    demand = np.asarray(demand, dtype=np.int64)

    depot_xy = coords[depot]
    customers = np.arange(1, len(coords))

    angles = np.arctan2(coords[customers, 1] - depot_xy[1],
                        coords[customers, 0] - depot_xy[0])
    order = customers[np.argsort(angles)]

    total_demand = demand[order].sum()
    target = total_demand / k

    clusters = [[] for _ in range(k)]
    acc = 0
    cid = 0
    for i in order:
        clusters[cid].append(int(i))
        acc += int(demand[i])
        if acc >= target and cid < k - 1:
            cid += 1
            acc = 0
    return clusters


def euclid_int_distance_matrix(coords):
    C = np.asarray(coords, dtype=np.int64)
    dx = C[:, 0][:, None] - C[:, 0][None, :]
    dy = C[:, 1][:, None] - C[:, 1][None, :]
    D = np.rint(np.sqrt(dx * dx + dy * dy)).astype(np.int64)
    np.fill_diagonal(D, 0)
    return D


def compute_tl_per_cluster(time_limit_cluster, k_clusters):
    tl = max(0.01, float(time_limit_cluster) / float(k_clusters))

    if k_clusters == 8:
        tl_per_cluster = 0.20 * tl
    elif k_clusters == 5:
        tl_per_cluster = 0.33 * tl
    elif k_clusters == 3:
        tl_per_cluster = 0.50 * tl
    elif k_clusters == 1:
        tl_per_cluster = tl / k_clusters
    else:
        tl_per_cluster = tl / k_clusters

    return tl_per_cluster


def solve_cluster_pyvrp_keep_order(cluster_nodes, coords, demand, Q,
                                   time_limit=10.0, n_restarts=10, seed=0, extra_vehicles=3):
    rng = random.Random(seed)
    routes_out = []

    local_coords = [coords[0]] + [coords[i] for i in cluster_nodes]
    local_demands = [0] + [int(demand[i]) for i in cluster_nodes]
    Dloc = euclid_int_distance_matrix(local_coords)

    total_dem = sum(local_demands[1:])
    min_veh = math.ceil(total_dem / Q)

    for _ in range(n_restarts):
        m = Model()
        x0, y0 = local_coords[0]
        m.add_depot(x=int(x0), y=int(y0))

        for idx in range(1, len(local_coords)):
            x, y = local_coords[idx]
            m.add_client(x=int(x), y=int(y),
                         delivery=int(local_demands[idx]), pickup=0)

        m.add_vehicle_type(num_available=min_veh + extra_vehicles, capacity=int(Q))

        data0 = m.data()
        data = data0.replace(distance_matrices=[Dloc], duration_matrices=[Dloc])
        m2 = Model.from_data(data)

        run_seed = rng.randint(0, 10 ** 9)
        res = m2.solve(stop=MaxRuntime(time_limit), seed=run_seed)

        for route in res.best.routes():
            visits = list(route.visits())
            ordered = [cluster_nodes[i - 1] for i in visits if i != 0]
            if not ordered:
                continue
            key = tuple(sorted(ordered))
            routes_out.append({"key": key, "order": ordered, "cost": int(route.distance())})

    return routes_out


def add_routes_to_pool_keep_order(pool, new_routes):
    for r in new_routes:
        key = r["key"]
        c = int(r["cost"])
        if (key not in pool) or (c < pool[key]["cost"]):
            pool[key] = {"cost": c, "order": list(r["order"])}


def solve_spp_keys_from_pool(pool, n_customers, vehicle_penalty=0.0, timelimit=None, verbose=True):
    keys = list(pool.keys())
    costs = [pool[k]["cost"] for k in keys]

    m = gp.Model("SPP_STAGE1")
    x = m.addVars(len(keys), vtype=GRB.BINARY, name="x")
    m.setObjective(gp.quicksum((costs[r] + vehicle_penalty) * x[r] for r in range(len(keys))), GRB.MINIMIZE)

    for i in range(1, n_customers + 1):
        m.addConstr(gp.quicksum(x[r] for r in range(len(keys)) if i in keys[r]) == 1, name=f"cover_{i}")

    m.Params.OutputFlag = 1 if verbose else 0
    if timelimit is not None:
        m.Params.TimeLimit = float(timelimit)

    m.optimize()
    if m.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(f"SPP_STAGE1 ended with status {m.status}")

    selected_keys = [keys[r] for r in range(len(keys)) if x[r].X > 0.5]
    return selected_keys, float(m.objVal)


def stage1_pyvrp_multik_build_pool_then_one_spp(coords, demand, Q,
                                                k_schedule=None,
                                                seed=0,
                                                time_limit_cluster=30.0,
                                                n_restarts=10,
                                                extra_vehicles=50,
                                                vehicle_penalty=0.0,
                                                spp_timelimit=120.0,
                                                verbose=True):
    n_customers = len(coords) - 1
    if k_schedule is None:
        k_schedule = [8, 5, 3, 1]
    k_schedule = [int(k) for k in k_schedule if 1 <= int(k) <= n_customers]
    if not k_schedule:
        raise ValueError("k_schedule is empty.")

    pool_global = {}

    if verbose:
        print(f"\n[STAGE1] Multi-k pooling. schedule={k_schedule}")

    for level, k_clusters in enumerate(k_schedule, start=1):
        if verbose:
            print("\n" + "=" * 60)
            print(f"[STAGE1] Level {level}/{len(k_schedule)} | k={k_clusters}")
            print("=" * 60)

        clusters = sweep_clustering(coords, demand, k_clusters)
        tl_per_cluster = compute_tl_per_cluster(time_limit_cluster, k_clusters)
        if verbose:
            print(f"[STAGE1] tl_per_cluster={tl_per_cluster:.4f}s")

        for cid, cluster in enumerate(clusters):
            if verbose:
                print(f"[STAGE1] k={k_clusters} | Cluster {cid + 1}/{k_clusters} | size={len(cluster)}")

            routes = solve_cluster_pyvrp_keep_order(
                cluster_nodes=cluster, coords=coords, demand=demand, Q=Q,
                time_limit=tl_per_cluster,
                n_restarts=n_restarts,
                seed=seed + 1000 * level + cid,
                extra_vehicles=extra_vehicles
            )
            add_routes_to_pool_keep_order(pool_global, routes)

            if verbose:
                print(f"[STAGE1] pool_global size: {len(pool_global)}")

    if verbose:
        print("\n" + "=" * 60)
        print("[STAGE1] Solving ONE final SPP over global pool...")
        print("=" * 60)

    selected_keys, obj = solve_spp_keys_from_pool(
        pool_global, n_customers,
        vehicle_penalty=vehicle_penalty,
        timelimit=spp_timelimit,
        verbose=verbose
    )
    start_routes = [pool_global[k]["order"] for k in selected_keys]

    if verbose:
        print(f"[STAGE1] FINAL SPP routes: {len(start_routes)} | obj: {obj:,.0f} | pool: {len(pool_global)}")

    return start_routes, obj, pool_global


# =============================================================================
# SECTION 15: MAIN PIPELINE - INTEGRATED WITH ALNS
# =============================================================================

def run_lns_spp_cycles_with_alns(
    coords, demand, Q,
    total_time=300.0,
    n_cycles=3,
    seed=42,
    depot=0,
    verbose=True,
    # Stage 1 config
    k_schedule=None,
    time_limit_cluster=10.0,
    n_restarts=10,
    extra_vehicles=2,
    spp1_timelimit=120.0,
    # ALNS config
    segment_length=100,
    reaction_factor=0.1,
):
    """
    Full pipeline with ALNS integration:
    
    Stage 1: PyVRP clustering + SPP (unchanged)
    Stage 2: ALNS-based LNS + SPP cycles
    """
    n_customers = len(coords) - 1
    start_global = time.time()

    # ==========================
    # STAGE 1: PyVRP+SPP start
    # ==========================
    if verbose:
        print("\n" + "=" * 60)
        print("STAGE 1: CLUSTER + PyVRP + SPP")
        print("=" * 60)

    start_routes, start_obj, pool_stage1 = stage1_pyvrp_multik_build_pool_then_one_spp(
        coords, demand, Q,
        k_schedule=k_schedule if k_schedule else [8, 5, 3, 1],
        seed=seed,
        time_limit_cluster=time_limit_cluster,
        n_restarts=n_restarts,
        extra_vehicles=extra_vehicles,
        vehicle_penalty=0.0,
        spp_timelimit=spp1_timelimit,
        verbose=verbose
    )

    # ==========================
    # STAGE 2: ALNS LNS + SPP cycles
    # ==========================
    D = build_distance_matrix(coords)

    # PHASE 0: ALNS EXPLORE
    t_explore = 0.15 * total_time
    if verbose:
        print("\n" + "=" * 60)
        print("STAGE 2 / PHASE 0: ALNS EXPLORE")
        print("=" * 60)

    best_routes, best_cost, D, route_pool = lns_alns(
        coords, demand, Q,
        depot=depot,
        iters=10 ** 7,
        time_limit=t_explore,
        k_knn=80 if n_customers > 300 else 40,
        seed=seed,
        verbose=verbose,
        log_every=50,
        segment_length=segment_length,
        reaction_factor=reaction_factor,
        start_routes=start_routes,
        D_in=D,
        route_pool_in=None,
        max_pool=150000 if n_customers > 500 else 30000,
        min_cover_pool=5,
        collect_every=1,
        ls_rounds=1 if n_customers > 500 else 2,
        use_new_ls_ops=True,
    )

    # ==========================
    # SPP 0
    # ==========================
    if verbose:
        print("\n=== SPP 0 ===")

    prev_sel = None
    spp_routes, spp_cost, spp_model, prev_sel = solve_spp_gurobi(
        route_pool,
        n_customers=n_customers,
        time_limit=600 if n_customers > 300 else 30,
        mip_gap=0.0,
        verbose=verbose,
        prev_selected_routes=prev_sel,
        fallback_routes=best_routes
    )

    global_best_routes = [list(r) for r in spp_routes]
    global_best_cost = spp_cost

    if verbose:
        print(f"SPP0 cost: {global_best_cost:,.0f} | routes: {len(global_best_routes)} | pool: {len(route_pool)}")

    # ==========================
    # CYCLES
    # ==========================
    remaining_after_explore = total_time - (time.time() - start_global)
    t_cycle = max(30, remaining_after_explore / (n_cycles + 1))

    for k in range(1, n_cycles + 1):
        if verbose:
            print(f"\n=== CYCLE {k}: ALNS INTENSIFY ===")

        best_k_routes, best_k_cost, route_pool = intensify_from_spp_alns(
            spp_routes,
            D, coords, demand, Q,
            route_pool=route_pool,
            time_limit=0.85 * t_cycle,
            iters=10 ** 7,
            k_knn=80 if n_customers > 300 else 50,
            seed=seed + k,
            verbose=verbose,
            max_pool=150000 if n_customers > 500 else 40000,
            collect_every=1,
            min_cover=5,
            depot=depot,
            segment_length=segment_length,
            reaction_factor=reaction_factor,
        )

        if verbose:
            print(f"\n=== SPP {k} ===")

        spp_routes, spp_cost, spp_model, prev_sel = solve_spp_gurobi(
            route_pool,
            n_customers=n_customers,
            time_limit=240,
            mip_gap=0.0,
            verbose=verbose,
            prev_selected_routes=prev_sel,
            fallback_routes=best_routes
        )

        if spp_cost < global_best_cost:
            global_best_cost = spp_cost
            global_best_routes = [list(r) for r in spp_routes]
            if verbose:
                print(f"*** New GLOBAL BEST after cycle {k}: {global_best_cost:,.0f} | routes: {len(global_best_routes)}")

    return global_best_routes, global_best_cost, D


# =============================================================================
# SECTION 16: HEXALY INTEGRATION (Stage 3)
# =============================================================================

def setup_hexaly():
    """
    Setup Hexaly optimizer paths and imports.
    
    Modify HEXALY_BIN and HEXALY_PY to match your installation.
    Returns True if Hexaly is available, False otherwise.
    """
    import sys
    import os
    
    # ============================================
    # MODIFY THESE PATHS FOR YOUR INSTALLATION
    # ============================================
    HEXALY_BIN = r"C:\Hexaly_14_0\bin"
    HEXALY_PY = r"C:\Hexaly_14_0\bin\python"
    
    # Linux alternative paths (uncomment if on Linux):
    # HEXALY_BIN = "/opt/hexaly_14_0/bin"
    # HEXALY_PY = "/opt/hexaly_14_0/bin/python"
    # ============================================
    
    try:
        # Add Python bindings to path
        if HEXALY_PY not in sys.path:
            sys.path.insert(0, HEXALY_PY)
        
        # Add DLL directory (Windows)
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(HEXALY_BIN)
        
        os.environ["PATH"] = HEXALY_BIN + os.pathsep + os.environ.get("PATH", "")
        
        # Test import
        import hexaly.optimizer
        print("Hexaly setup successful")
        return True
        
    except Exception as e:
        print(f"Hexaly setup failed: {e}")
        print("Stage 3 (Hexaly refinement) will be skipped.")
        return False


def _eucl_2d_int(a, b):
    """Euclidean distance rounded to integer."""
    return int(round(math.hypot(a[0] - b[0], a[1] - b[1])))


def build_hexaly_distance_data(coords):
    """
    Build distance data structures for Hexaly.
    
    Returns:
        dist_matrix: Customer-to-customer distances (n x n, 0-indexed customers)
        dist_depot: Depot-to-customer distances (length n)
    """
    depot = coords[0]
    cust = coords[1:]
    n = len(cust)
    
    dist_depot = [_eucl_2d_int(depot, cust[i]) for i in range(n)]
    dist_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = _eucl_2d_int(cust[i], cust[j])
    
    return dist_matrix, dist_depot


def apply_warmstart_to_hexaly(customers_sequences, warm_routes, nb_customers):
    """
    Apply warm-start solution to Hexaly model.
    
    Args:
        customers_sequences: List of Hexaly list variables
        warm_routes: Routes in original indices (1..n), e.g. [[5, 7, 2], [3, 8, 1]]
        nb_customers: Number of customers (n)
    """
    if warm_routes is None:
        return
    
    # Normalize routes: remove depot if present, convert to 0-indexed
    norm = []
    used = set()
    
    for r in warm_routes:
        rr = list(r)
        # Remove depot (0) if present at start/end
        if len(rr) >= 2 and rr[0] == 0 and rr[-1] == 0:
            rr = rr[1:-1]
        elif len(rr) >= 1 and rr[0] == 0:
            rr = rr[1:]
        elif len(rr) >= 1 and rr[-1] == 0:
            rr = rr[:-1]
        
        # Convert to 0-indexed (Hexaly uses 0..n-1)
        rr0 = []
        for v in rr:
            v = int(v)
            if v == 0:
                continue
            if not (1 <= v <= nb_customers):
                raise ValueError(f"Warm-start invalid: customer {v} out of range 1..{nb_customers}")
            vv = v - 1  # Convert to 0-indexed
            if vv in used:
                raise ValueError(f"Warm-start invalid: duplicate customer {v}")
            used.add(vv)
            rr0.append(vv)
        norm.append(rr0)
    
    if len(used) != nb_customers:
        missing = sorted(set(range(nb_customers)) - used)
        raise ValueError(f"Warm-start incomplete: missing {len(missing)} customers (e.g. {[m + 1 for m in missing[:10]]})")
    
    # Clear all lists
    for seq in customers_sequences:
        seq.value.clear()
    
    # Load routes into lists
    for k, rr0 in enumerate(norm):
        if k >= len(customers_sequences):
            break
        for v0 in rr0:
            customers_sequences[k].value.add(int(v0))


def solve_cvrp_hexaly(coords, demand, Q, time_limit=60, warmstart_routes=None, verbose=True):
    """
    Solve CVRP using Hexaly Optimizer.
    
    Args:
        coords: List of [x, y] coordinates (depot at index 0)
        demand: List of demands (depot=0 at index 0)
        Q: Vehicle capacity
        time_limit: Time limit in seconds
        warmstart_routes: Optional warm-start routes (1-indexed customers)
        verbose: Print progress
    
    Returns:
        dict with keys:
            - nb_trucks_model: Number of trucks in model
            - total_distance: Best solution cost
            - routes: List of routes as [0, c1, c2, ..., 0] (with depot)
    """
    try:
        import hexaly.optimizer
    except ImportError:
        print("Hexaly not available. Skipping Stage 3.")
        return None
    
    nb_customers = len(coords) - 1
    if nb_customers <= 0:
        return {"nb_trucks_model": 0, "total_distance": 0, "routes": []}
    
    demands_data = list(demand[1:])  # Customers only (0-indexed)
    dist_matrix_data, dist_depot_data = build_hexaly_distance_data(coords)
    
    # Estimate number of trucks needed
    nb_trucks = math.ceil(sum(demands_data) / Q) + 20
    
    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        model = optimizer.model
        
        # Create list variables for each truck
        customers_sequences = [model.list(nb_customers) for _ in range(nb_trucks)]
        
        # Partition constraint: each customer in exactly one route
        model.constraint(model.partition(customers_sequences))
        
        # Data arrays
        demands = model.array(demands_data)
        dist_matrix = model.array(dist_matrix_data)
        dist_depot = model.array(dist_depot_data)
        
        # Track which trucks are used
        trucks_used = [(model.count(customers_sequences[k]) > 0) for k in range(nb_trucks)]
        dist_routes = [None] * nb_trucks
        
        for k in range(nb_trucks):
            sequence = customers_sequences[k]
            c = model.count(sequence)
            
            # Capacity constraint
            demand_lambda = model.lambda_function(lambda j: demands[j])
            route_qty = model.sum(sequence, demand_lambda)
            model.constraint(route_qty <= Q)
            
            # Distance calculation
            dist_lambda = model.lambda_function(
                lambda i: model.at(dist_matrix, sequence[i - 1], sequence[i])
            )
            dist_routes[k] = model.sum(model.range(1, c), dist_lambda) + model.iif(
                c > 0,
                dist_depot[sequence[0]] + dist_depot[sequence[c - 1]],
                0
            )
        
        # Objective: minimize total distance
        total_distance = model.sum(dist_routes)
        model.minimize(total_distance)
        
        model.close()
        
        # Apply warm-start after model is closed
        if warmstart_routes is not None:
            try:
                apply_warmstart_to_hexaly(customers_sequences, warmstart_routes, nb_customers)
                if verbose:
                    print(f"[Hexaly] Warm-start applied with {len(warmstart_routes)} routes")
            except Exception as e:
                print(f"[Hexaly] Warm-start failed: {e}")
        
        # Solve
        optimizer.param.time_limit = int(time_limit)
        if verbose:
            print(f"[Hexaly] Starting optimization (time_limit={time_limit}s, trucks={nb_trucks})")
        
        optimizer.solve()
        
        # Extract solution
        routes = []
        for k in range(nb_trucks):
            if trucks_used[k].value != 1:
                continue
            seq = list(customers_sequences[k].value)  # 0-indexed
            # Convert to 1-indexed with depot
            routes.append([0] + [i + 1 for i in seq] + [0])
        
        result = {
            "nb_trucks_model": nb_trucks,
            "total_distance": int(total_distance.value),
            "routes": routes,
        }
        
        if verbose:
            print(f"[Hexaly] Done: cost={result['total_distance']:,} routes={len(routes)}")
        
        return result


def run_hexaly_refinement(coords, demand, Q, warmstart_routes, time_limit=3600, verbose=True):
    """
    Run Hexaly as Stage 3 refinement.
    
    Args:
        coords, demand, Q: Instance data
        warmstart_routes: Best routes from Stage 2 (1-indexed, no depot)
        time_limit: Hexaly time limit
        verbose: Print progress
    
    Returns:
        best_routes: Refined routes (1-indexed, no depot)
        best_cost: Solution cost
        success: Whether Hexaly ran successfully
    """
    if verbose:
        print("\n" + "=" * 60)
        print("STAGE 3: HEXALY REFINEMENT")
        print("=" * 60)
    
    # Setup Hexaly
    hexaly_available = setup_hexaly()
    if not hexaly_available:
        return warmstart_routes, total_cost_D(warmstart_routes, build_distance_matrix(coords), 0), False
    
    # Run Hexaly
    result = solve_cvrp_hexaly(
        coords, demand, Q,
        time_limit=time_limit,
        warmstart_routes=warmstart_routes,
        verbose=verbose
    )
    
    if result is None:
        return warmstart_routes, total_cost_D(warmstart_routes, build_distance_matrix(coords), 0), False
    
    # Extract routes without depot
    hex_routes = result["routes"]
    best_routes = [r[1:-1] for r in hex_routes if len(r) > 2]
    best_cost = result["total_distance"]
    
    if verbose:
        print(f"[Hexaly] Final: cost={best_cost:,} routes={len(best_routes)}")
    
    return best_routes, best_cost, True


# =============================================================================
# SECTION 17: FULL PIPELINE WITH HEXALY
# =============================================================================

def run_full_pipeline(
    coords, demand, Q,
    # Time allocation
    total_time=7200.0,
    hexaly_time=3600.0,
    n_cycles=100,
    seed=42,
    depot=0,
    verbose=True,
    # Stage 1 config
    k_schedule=None,
    time_limit_cluster=1800.0,
    n_restarts=1,
    extra_vehicles=10,
    spp1_timelimit=1800.0,
    # ALNS config
    segment_length=100,
    reaction_factor=0.1,
    # Hexaly config
    use_hexaly=True,
):
    """
    Full 3-stage CVRP solver pipeline:
    
    Stage 1: PyVRP clustering + SPP
    Stage 2: ALNS-based LNS + SPP cycles
    Stage 3: Hexaly refinement (optional)
    
    Args:
        coords: List of [x, y] coordinates (depot at index 0)
        demand: List of demands
        Q: Vehicle capacity
        total_time: Total time budget for Stages 1-2
        hexaly_time: Time budget for Stage 3 (Hexaly)
        n_cycles: Number of LNS+SPP cycles in Stage 2
        seed: Random seed
        depot: Depot index (default 0)
        verbose: Print progress
        k_schedule: Clustering schedule for Stage 1
        time_limit_cluster: Time for PyVRP per cluster level
        n_restarts: PyVRP restarts per cluster
        extra_vehicles: Extra vehicles in PyVRP
        spp1_timelimit: SPP time limit in Stage 1
        segment_length: ALNS segment length
        reaction_factor: ALNS learning rate
        use_hexaly: Whether to run Stage 3
    
    Returns:
        best_routes: Final routes (1-indexed, no depot)
        best_cost: Final solution cost
        D: Distance matrix
    """
    start_global = time.time()
    
    # ==========================
    # STAGES 1-2: ALNS Pipeline
    # ==========================
    best_routes, best_cost, D = run_lns_spp_cycles_with_alns(
        coords, demand, Q,
        total_time=total_time,
        n_cycles=n_cycles,
        seed=seed,
        depot=depot,
        verbose=verbose,
        k_schedule=k_schedule if k_schedule else [8, 5, 3, 1],
        time_limit_cluster=time_limit_cluster,
        n_restarts=n_restarts,
        extra_vehicles=extra_vehicles,
        spp1_timelimit=spp1_timelimit,
        segment_length=segment_length,
        reaction_factor=reaction_factor,
    )
    
    stage2_cost = best_cost
    stage2_routes = [r[:] for r in best_routes]
    
    if verbose:
        print(f"\n[STAGES 1-2] Complete: cost={best_cost:,} routes={len(best_routes)} time={time.time()-start_global:.1f}s")
    
    # ==========================
    # STAGE 3: HEXALY REFINEMENT
    # ==========================
    if use_hexaly and hexaly_time > 0:
        hex_routes, hex_cost, success = run_hexaly_refinement(
            coords, demand, Q,
            warmstart_routes=best_routes,
            time_limit=hexaly_time,
            verbose=verbose
        )
        
        if success and hex_cost < best_cost:
            best_routes = hex_routes
            best_cost = hex_cost
            if verbose:
                print(f"[Hexaly] Improved: {stage2_cost:,} -> {best_cost:,} (delta={stage2_cost-best_cost:,})")
        elif success:
            if verbose:
                print(f"[Hexaly] No improvement: Stage 2 cost {stage2_cost:,} <= Hexaly {hex_cost:,}")
    else:
        if verbose:
            print("\n[STAGE 3] Hexaly skipped (use_hexaly=False or hexaly_time=0)")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"FINAL RESULT: cost={best_cost:,} routes={len(best_routes)} total_time={time.time()-start_global:.1f}s")
        print(f"{'='*60}")
    
    return best_routes, best_cost, D


# =============================================================================
# SECTION 18: UTILITIES
# =============================================================================

def load_instance_from_file(vrp_file_path):
    """Load a CVRP instance from a .vrp file."""
    instance = load_cvrp_instance(vrp_file_path)

    depot_id = instance.depots[0]
    n = instance.dimension
    coords = [[0, 0]] * n

    depot_x, depot_y = instance.node_coords[depot_id]
    coords[0] = [int(depot_x), int(depot_y)]

    customer_idx = 1
    for node_id in sorted(instance.node_coords.keys()):
        if node_id == depot_id:
            continue
        x, y = instance.node_coords[node_id]
        coords[customer_idx] = [int(x), int(y)]
        customer_idx += 1

    demand = [0] * n
    demand[0] = 0

    customer_idx = 1
    for node_id in sorted(instance.node_coords.keys()):
        if node_id == depot_id:
            continue
        demand[customer_idx] = instance.demands[node_id]
        customer_idx += 1

    Q = instance.capacity

    print(f"Loaded instance: {instance.name}")
    print(f"  Dimension: {instance.dimension}")
    print(f"  Capacity: {Q}")
    print(f"  Total demand: {sum(demand)}")

    return coords, demand, Q


def make_random_instance(n_customers=1000, Q=50, demand_low=1, demand_high=10, seed=0):
    """Generate a random CVRP instance."""
    rng = np.random.default_rng(seed)
    depot = np.array([[500, 500]], dtype=int)
    pts = rng.integers(0, 1001, size=(n_customers, 2), dtype=int)
    coords = np.vstack([depot, pts])

    demand = np.zeros(n_customers + 1, dtype=int)
    demand[1:] = rng.integers(demand_low, demand_high + 1, size=n_customers)
    return coords.tolist(), demand.tolist(), Q


def write_solution(routes, cost, filepath):
    """Write solution to .sol file."""
    valid_routes = [r for r in routes if r]
    vrplib.write_solution(filepath, valid_routes, data={"Cost": cost})


def check_feasibility_cvrp(routes, demand, Q, n_customers, depot=0, verbose=True):
    """Check solution feasibility."""
    ok = True

    seen = []
    for r in routes:
        for v in r:
            seen.append(v)

    seen_set = set(seen)
    expected = set(range(1, n_customers + 1))

    missing = expected - seen_set
    duplicated = [v for v in seen_set if seen.count(v) > 1]

    if missing:
        ok = False
        if verbose:
            print("FAIL: Customers not covered:", sorted(list(missing))[:10], "...")

    if duplicated:
        ok = False
        if verbose:
            print("FAIL: Duplicate customers:", duplicated[:10], "...")

    if depot in seen_set:
        ok = False
        if verbose:
            print("FAIL: Depot appears in route.")

    for k, r in enumerate(routes):
        load = sum(demand[v] for v in r)
        if load > Q + 1e-9:
            ok = False
            if verbose:
                print(f"FAIL: Route {k} violates capacity: load={load} > Q={Q}")

    for v in seen_set:
        if v < 1 or v > n_customers:
            ok = False
            if verbose:
                print("FAIL: Customer out of range:", v)

    if ok and verbose:
        print("OK: Solution is FEASIBLE")

    return ok


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    # Parse arguments
    args = sys.argv[1:]
    config = {
        "mode": "file",
        "file": None,
        "n_customers": 1000,
        "Q": 100,
        "seed": 42,
        "hexaly_time": 3600,
        "total_time": 10800,
        "use_hexaly": True,
    }

    i = 0
    while i < len(args):
        if args[i] == "--file" and i + 1 < len(args):
            config["mode"] = "file"
            config["file"] = args[i + 1]
            i += 2
        elif args[i] == "--random":
            config["mode"] = "random"
            i += 1
        elif args[i] == "--n" and i + 1 < len(args):
            config["n_customers"] = int(args[i + 1])
            i += 2
        elif args[i] == "--Q" and i + 1 < len(args):
            config["Q"] = int(args[i + 1])
            i += 2
        elif args[i] == "--seed" and i + 1 < len(args):
            config["seed"] = int(args[i + 1])
            i += 2
        elif args[i] == "--hexaly-time" and i + 1 < len(args):
            config["hexaly_time"] = int(args[i + 1])
            i += 2
        elif args[i] == "--total-time" and i + 1 < len(args):
            config["total_time"] = int(args[i + 1])
            i += 2
        elif args[i] == "--no-hexaly":
            config["use_hexaly"] = False
            i += 1
        else:
            i += 1

    # Load instance
    if config["file"]:
        coords, demand, Q = load_instance_from_file(config["file"])
        instance_file = config["file"]
    elif config["mode"] == "random":
        coords, demand, Q = make_random_instance(
            n_customers=config["n_customers"],
            Q=config["Q"],
            seed=config["seed"]
        )
        instance_file = "random_instance"
    else:
        # Default: modify this path
        instance_file = "XL/XL/XL-n1048-k237.vrp"
        coords, demand, Q = load_instance_from_file(instance_file)

    # Run full pipeline (Stages 1-3)
    best_routes, best_cost, D = run_full_pipeline(
        coords, demand, Q,
        total_time=config["total_time"],
        hexaly_time=config["hexaly_time"],
        n_cycles=10,
        seed=config["seed"],
        depot=0,
        verbose=True,
        # Stage 1 config
        k_schedule=[8, 5, 3, 1],
        time_limit_cluster=1800.0,
        n_restarts=1,
        extra_vehicles=10,
        spp1_timelimit=1800.0,
        # ALNS config
        segment_length=100,
        reaction_factor=0.1,
        # Hexaly config
        use_hexaly=config["use_hexaly"],
    )

    print(f"\nFINAL: {best_cost:,.0f} | routes: {len(best_routes)}")

    # Verify feasibility
    check_feasibility_cvrp(
        best_routes, demand, Q,
        n_customers=len(coords) - 1,
        depot=0,
        verbose=True
    )

    # Write solution
    instance_name = os.path.splitext(os.path.basename(instance_file))[0]
    solution_file = f"{instance_name}.sol"
    write_solution(best_routes, best_cost, solution_file)
    print(f"\nSolution written to {solution_file}")