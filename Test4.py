# Converted from Test4.ipynb

# ===== Code Cell 1 =====
# ============================================================
#  CVRP XL: Stage-1 (Clusters + PyVRP + SPP)  ->  Stage-2 (Tu LNS+SPP Cycles)
#  - Integra TODO en un solo pipeline.
#  - Stage-1 produce start_routes (ordenadas) + pool inicial (opcional).
#  - Stage-2 corre tu LNS (explore) arrancando desde start_routes,
#    luego SPP0 y ciclos [intensify -> SPP warm-start].
# ============================================================

import math, time, random, heapq, os
import numpy as np
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from pyvrp import Model
from pyvrp.stop import MaxRuntime

# Import CVRP parser
from cvrp_parser import load_cvrp_instance

# Import vrplib for solution file I/O
import vrplib


# =========================
# 0) Generador de instancia
# =========================
def make_random_instance(n_customers=1000, Q=50, demand_low=1, demand_high=10, seed=0):
    rng = np.random.default_rng(seed)
    depot = np.array([[500, 500]], dtype=int)
    pts = rng.integers(0, 1001, size=(n_customers, 2), dtype=int)
    coords = np.vstack([depot, pts])

    demand = np.zeros(n_customers + 1, dtype=int)
    demand[1:] = rng.integers(demand_low, demand_high + 1, size=n_customers)
    return coords.tolist(), demand.tolist(), Q


def load_instance_from_file(vrp_file_path):
    """
    Load a CVRP instance from a .vrp file and convert to solver format.

    Args:
        vrp_file_path: Path to the .vrp file

    Returns:
        coords: list of [x, y] coordinates (depot at index 0, customers at 1..n)
        demand: list of demands (depot=0 at index 0, customers at 1..n)
        Q: vehicle capacity
    """
    instance = load_cvrp_instance(vrp_file_path)

    # Get depot (assuming first depot in list)
    depot_id = instance.depots[0]

    # Build coords list: depot first, then customers in order
    n = instance.dimension
    coords = [[0, 0]] * n

    # Place depot at index 0
    depot_x, depot_y = instance.node_coords[depot_id]
    coords[0] = [int(depot_x), int(depot_y)]

    # Place customers at indices 1..n-1
    customer_idx = 1
    for node_id in sorted(instance.node_coords.keys()):
        if node_id == depot_id:
            continue
        x, y = instance.node_coords[node_id]
        coords[customer_idx] = [int(x), int(y)]
        customer_idx += 1

    # Build demand list: depot=0 at index 0, then customers
    demand = [0] * n
    demand[0] = 0  # depot has no demand

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
    print(f"  Depot: {coords[0]}")

    return coords, demand, Q


# =========================
# 1) Distancias + kNN
# =========================
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


# =========================
# 2) Costos con D
# =========================
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


def write_solution(routes, cost, filepath):
    """
    Write CVRP solution to .sol file in VRPLIB format.

    Format:
        Route #1: 1 5 3 2
        Route #2: 4 6 7
        Cost: 1240.5

    Args:
        routes: list of routes, each route is a list of customer indices (no depot)
        cost: total solution cost
        filepath: output .sol file path
    """
    # Filter out empty routes
    valid_routes = [r for r in routes if r]
    vrplib.write_solution(filepath, valid_routes, data={"Cost": cost})


# =========================
# 3) Pool seguro (tu lógica)
# =========================
def canonical_route(route):
    t = tuple(route)
    tr = tuple(reversed(route))
    return t if t < tr else tr


def add_routes_to_pool_safe(routes, route_pool, D, n_customers, depot=0,
                            max_pool=150000, min_cover=2):
    # 1) agregar rutas nuevas
    for r in routes:
        if not r:
            continue
        rt = canonical_route(r)
        if rt not in route_pool:
            route_pool[rt] = int(route_cost_D(list(rt), D, depot))

    if len(route_pool) <= max_pool:
        return

    # 2) cobertura: cliente -> rutas
    cover = defaultdict(list)
    for rt, c in route_pool.items():
        for i in rt:
            if 1 <= i <= n_customers:
                cover[i].append((c, rt))

    # 3) protegidas: min_cover por cliente
    protected = set()
    for i in range(1, n_customers + 1):
        if cover[i]:
            cover[i].sort(key=lambda x: x[0])
            for _, rt in cover[i][:min_cover]:
                protected.add(rt)

    # 4) completar con más baratas hasta max_pool
    remaining_slots = max_pool - len(protected)
    if remaining_slots < 0:
        keep = protected
    else:
        candidates = [(c, rt) for rt, c in route_pool.items() if rt not in protected]
        extra = heapq.nsmallest(remaining_slots, candidates, key=lambda x: x[0])
        keep = set(protected) | {rt for _, rt in extra}

    # 5) poda
    new_pool = {rt: route_pool[rt] for rt in keep}
    route_pool.clear()
    route_pool.update(new_pool)


# ============================================================
# 4) (Stage-1) Sweep clustering + PyVRP por cluster + SPP
#    IMPORTANTE: preserva orden de ruta para usarlo como start del LNS.
# ============================================================
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


#def make_cluster_schedule(n_customers):
    #"""
    #Default schedule tipo pirámide. Puedes cambiarlo.
    #Para 1000 -> [25,20,15,12,8,5,3]
    #"""
    #sched = [8, 5, 3, 1]
    # filtra ks inválidos
    #sched = [k for k in sched if 2 <= k <= n_customers]
    # asegura terminar en 3 si es posible
    #if n_customers >= 3 and (3 not in sched):
        #sched.append(3)
    # orden descendente (de más clusters a menos clusters)
    #sched = sorted(set(sched), reverse=True)
    #return sched

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
        # fallback general (por si usas otros k)
        tl_per_cluster = tl / k_clusters

    return tl_per_cluster


def stage1_pyvrp_multik_build_pool_then_one_spp(coords, demand, Q,
                                                k_schedule=None,
                                                seed=0,
                                                time_limit_cluster=30.0,
                                                n_restarts=10,
                                                extra_vehicles=50,
                                                vehicle_penalty=0.0,
                                                spp_timelimit=120.0,
                                                verbose=True):
    """
    Stage-1:
      - Recorre varios k (clusterizaciones)
      - Genera rutas con PyVRP por cluster y acumula TODO en un pool global
      - Resuelve UN SOLO SPP al final usando el pool global
    Retorna: start_routes, obj, pool_global
    """
    n_customers = len(coords) - 1
    if k_schedule is None:
        k_schedule = [8, 5, 3, 1]
    k_schedule = [int(k) for k in k_schedule if 1 <= int(k) <= n_customers]
    if not k_schedule:
        raise ValueError("k_schedule quedó vacío. Revisa valores.")

    pool_global = {}  # key -> {"cost": int, "order": list[int]}

    if verbose:
        print(f"\n[STAGE1] Multi-k pooling only. schedule={k_schedule}")

    for level, k_clusters in enumerate(k_schedule, start=1):
        if verbose:
            print("\n" + "=" * 60)
            print(f"[STAGE1] Level {level}/{len(k_schedule)} | k={k_clusters}")
            print("=" * 60)

        clusters = sweep_clustering(coords, demand, k_clusters)
        #tl_per_cluster = max(0.01, float(time_limit_cluster) / float(k_clusters))
        tl_per_cluster = compute_tl_per_cluster(time_limit_cluster, k_clusters)
        if verbose:
            print(f"[STAGE1] time_limit_cluster={time_limit_cluster} -> tl_per_cluster={tl_per_cluster:.4f}s")

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

    # ---- UN SOLO SPP AL FINAL ----
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
            key = tuple(sorted(ordered))  # set-key para SPP/dedup (sin orden)
            routes_out.append({"key": key, "order": ordered, "cost": int(route.distance())})

    return routes_out


def add_routes_to_pool_keep_order(pool, new_routes):
    # pool: key -> {"cost": int, "order": list[int]}
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

    # cobertura exacta
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


def stage1_pyvrp_cluster_spp(coords, demand, Q, k_clusters=5, seed=0,
                             time_limit_cluster=10.0, n_restarts=10, extra_vehicles=50,
                             vehicle_penalty=0.0, spp_timelimit=120.0, verbose=True):
    n_customers = len(coords) - 1
    clusters = sweep_clustering(coords, demand, k_clusters)

    pool = {}
    for cid, cluster in enumerate(clusters):
        if verbose:
            print(f"\n[STAGE1] Cluster {cid + 1}/{k_clusters} | size={len(cluster)}")
        routes = solve_cluster_pyvrp_keep_order(
            cluster_nodes=cluster, coords=coords, demand=demand, Q=Q,
            time_limit=time_limit_cluster, n_restarts=n_restarts,
            seed=seed + cid, extra_vehicles=extra_vehicles
        )
        add_routes_to_pool_keep_order(pool, routes)
        if verbose:
            print(f"[STAGE1] pool size: {len(pool)}")

    selected_keys, obj = solve_spp_keys_from_pool(pool, n_customers, vehicle_penalty, spp_timelimit, verbose=verbose)
    start_routes = [pool[k]["order"] for k in selected_keys]  # rutas ORDENADAS para Stage-2
    if verbose:
        print(f"\n[STAGE1] SPP routes: {len(start_routes)} | obj: {obj:,.0f}")
    return start_routes, obj, pool


# ============================================================
# 5) ===== TU LNS y operadores (mínimo necesario) =====
#     Nota: aquí NO re-copio todo tu bloque gigante: asumo que
#     lo pegaste tal cual en tu notebook (ruin_*, recreate_*, LS, etc).
#     Pero sí necesitamos dos funciones:
#       - initial_sweep (ya la tienes)
#       - local_search_sotaish (ya la tienes)
#       - ruin_random, ruin_related, ruin_string (ya las tienes)
#       - recreate_regret2_D (ya la tienes)
#       - etc.
# ============================================================
####  ==== Guarda rutas === ####
## ===== Helpers ====== ##

def canonical_route(route):
    """
    Normaliza orientación para evitar duplicados: (1,2,3) == (3,2,1)
    """
    t = tuple(route)
    tr = tuple(reversed(route))
    return t if t < tr else tr


def add_routes_to_pool_safe(routes, route_pool, D, n_customers, depot=0,
                            max_pool=12000, min_cover=5):
    """
    route_pool: dict{route_tuple: cost}
    Garantiza que cada cliente tenga >= min_cover rutas en el pool antes de podar.
    """
    # 1) agregar rutas nuevas
    for r in routes:
        if not r:
            continue
        rt = canonical_route(r)
        if rt not in route_pool:
            route_pool[rt] = int(route_cost_D(list(rt), D, depot))

    if len(route_pool) <= max_pool:
        return

    # 2) construir cobertura: para cada cliente, lista de rutas que lo contienen (por costo)
    cover = defaultdict(list)  # i -> [(cost, route_tuple), ...]
    for rt, c in route_pool.items():
        for i in rt:
            if 1 <= i <= n_customers:
                cover[i].append((c, rt))

    # 3) rutas protegidas: asegurar min_cover por cliente
    protected = set()
    for i in range(1, n_customers + 1):
        if cover[i]:
            cover[i].sort(key=lambda x: x[0])
            for _, rt in cover[i][:min_cover]:
                protected.add(rt)

    # 4) completar con las más baratas hasta max_pool
    remaining_slots = max_pool - len(protected)
    if remaining_slots < 0:
        # si protegido excede max_pool, al menos mantenemos protegido (raro)
        keep = protected
    else:
        candidates = [(c, rt) for rt, c in route_pool.items() if rt not in protected]
        extra = heapq.nsmallest(remaining_slots, candidates, key=lambda x: x[0])
        keep = set(protected) | {rt for _, rt in extra}

    # 5) aplicar poda
    new_pool = {rt: route_pool[rt] for rt in keep}
    route_pool.clear()
    route_pool.update(new_pool)


# ============================================================
# 1) DISTANCIAS: matriz D (entera, TSPLIB-like) + kNN exacto con D
# ============================================================
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
        if i == depot:
            # para el depot no importa mucho
            knn[i] = list(np.argsort(D[i])[1:k + 1])
        else:
            # vecinos cercanos excluyendo i
            order = np.argsort(D[i])
            order = order[order != i]
            knn[i] = list(order[:k])
    return knn


# ============================================================
# 2) COSTOS con D
# ============================================================
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


### ==============
# Nuevos Deltas
### ==============
def remove_delta_at(route, pos, D, depot=0):
    """Delta de costo al remover route[pos] de la ruta."""
    v = route[pos]
    prev = depot if pos == 0 else route[pos - 1]
    nxt = depot if pos == len(route) - 1 else route[pos + 1]
    return int(-D[prev, v] - D[v, nxt] + D[prev, nxt])


def insert_delta_at(route, pos, v, D, depot=0):
    """Delta de costo al insertar v en 'route' en posición pos."""
    prev = depot if pos == 0 else route[pos - 1]
    nxt = depot if pos == len(route) else route[pos]
    return int(D[prev, v] + D[v, nxt] - D[prev, nxt])


def best_insertion_position_excluding(route, node, D, depot=0, forbid_pos=None):
    """
    Mejor inserción de node en route, opcionalmente prohibiendo una posición (para evitar degeneraciones).
    forbid_pos: una posición entera que no se puede usar.
    """
    if not route:
        return 0, int(2 * D[depot, node])

    best_pos, best_delta = 0, 10 ** 18

    # posiciones candidatas: todas (para 1k está ok)
    for pos in range(0, len(route) + 1):
        if forbid_pos is not None and pos == forbid_pos:
            continue
        delta = insert_delta_at(route, pos, node, D, depot)
        if delta < best_delta:
            best_delta, best_pos = delta, pos

    return best_pos, int(best_delta)


##### Nuevos Movimientos ######
def or_opt_best_improvement(routes, loads, D, demand, Q, knn, depot=0,
                            seg_lens=(2, 3), max_moves=6000):
    """
    Or-opt inter-rutas: mueve un segmento consecutivo (len 2 o 3) de ruta A a ruta B.
    Restringe rutas candidatas con kNN del primer nodo del segmento.
    """
    best_gain = 0
    best_move = None  # (rA, start, L, rB, posB)

    n_routes = len(routes)
    if n_routes <= 1:
        return False

    # map cliente->(ruta,pos)
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

                # rutas candidatas por vecinos del primer nodo del segmento
                candidate_routes = set()
                pivot = seg[0]
                for nb in knn[pivot][:15]:
                    if nb in where:
                        candidate_routes.add(where[nb][0])
                if len(candidate_routes) < 3:
                    candidate_routes.add(random.randrange(n_routes))

                # delta por remover el segmento de A
                prevA = depot if start == 0 else A[start - 1]
                nextA = depot if start + L == LA else A[start + L]
                delta_remove = int(-D[prevA, seg[0]] - D[seg[-1], nextA] + D[prevA, nextA])
                # y se "pierden" arcos internos del segmento? no, permanecen dentro del segmento al insertarlo en B.

                for rB in candidate_routes:
                    if rB == rA:
                        continue
                    if loads[rB] + seg_demand > Q:
                        continue

                    B = routes[rB]

                    # posiciones candidatas en B: endpoints + cerca de vecinos del pivot
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

                        # delta insertar segmento (prevB -> seg[0] ... seg[-1] -> nextB)
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

        # si A queda vacía, elimina ruta y ajusta índice de rB
        if not routes[rA]:
            routes.pop(rA)
            loads.pop(rA)
            if rB > rA:
                rB -= 1

        # inserta segmento en B
        routes[rB][posB:posB] = seg
        loads[rB] += sum(demand[v] for v in seg)
        return True

    return False


def two_opt_star_best_improvement(routes, loads, D, demand, Q, knn, depot=0, max_moves=7000):
    """
    2-opt* (CROSS): corta dos rutas en posiciones i y j e intercambia las colas.
    A = a0..ai | ai+1..end
    B = b0..bj | bj+1..end
    -> A' = a0..ai + (bj+1..end)
    -> B' = b0..bj + (ai+1..end)
    """
    best_gain = 0
    best_move = None  # (rA, i, rB, j)

    n_routes = len(routes)
    if n_routes <= 1:
        return False

    # map cliente->(ruta,pos)
    where = {}
    for r_idx, r in enumerate(routes):
        for p, v in enumerate(r):
            where[v] = (r_idx, p)

    checked = 0
    for rA, A in enumerate(routes):
        if len(A) < 2:
            continue

        for i in range(0, len(A) - 1):  # corte después de i
            if checked >= max_moves:
                break
            a_i = A[i]
            a_next = A[i + 1]

            # rutas candidatas: rutas que contengan vecinos de a_next
            candidate_routes = set()
            for nb in knn[a_next][:15]:
                if nb in where:
                    candidate_routes.add(where[nb][0])
            if len(candidate_routes) < 3:
                candidate_routes.add(random.randrange(n_routes))

            # demanda cola de A (desde i+1)
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

                    # factibilidad por capacidad tras swap de colas
                    newA_load = headA_dem + tailB_dem
                    newB_load = headB_dem + tailA_dem
                    if newA_load > Q or newB_load > Q:
                        continue

                    # delta costo: se reemplazan arcos (a_i->a_next) y (b_j->b_next)
                    # por (a_i->b_next) y (b_j->a_next)
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

        # construir nuevas rutas
        newA = A[:i + 1] + tailB
        newB = B[:j + 1] + tailA

        routes[rA] = newA
        routes[rB] = newB

        loads[rA] = sum(demand[v] for v in newA)
        loads[rB] = sum(demand[v] for v in newB)

        # opcional: si alguna queda vacía (raro), limpias
        routes[:] = [r for r in routes if r]
        loads[:] = [sum(demand[v] for v in r) for r in routes]
        return True

    return False


def swap_star_best_improvement(routes, loads, D, demand, Q, knn, depot=0, max_moves=8000):
    """
    SWAP*: intercambia v de A con w de B, escogiendo las mejores posiciones de inserción para ambos.
    (Implementación práctica: w se busca en candidatos por vecinos; posiciones se optimizan por escaneo).
    """
    best_gain = 0
    best_move = None  # (rA, posA, rB, posB, insPosA, insPosB)

    n_routes = len(routes)
    if n_routes <= 1:
        return False

    # map cliente->(ruta,pos)
    where = {}
    for r_idx, r in enumerate(routes):
        for p, v in enumerate(r):
            where[v] = (r_idx, p)

    checked = 0
    for v, (rA, posA) in list(where.items()):
        if checked >= max_moves:
            break
        A = routes[rA]

        # rutas candidatas: rutas que contienen vecinos de v
        candidate_routes = set()
        for nb in knn[v][:18]:
            if nb in where:
                candidate_routes.add(where[nb][0])
        if len(candidate_routes) < 3:
            candidate_routes.add(random.randrange(n_routes))

        # delta por remover v de A
        delta_remove_v = remove_delta_at(A, posA, D, depot)

        for rB in candidate_routes:
            if rB == rA:
                continue
            B = routes[rB]

            # candidatos w: vecinos de v que estén en B, o 1 aleatorio
            cand_ws = []
            for nb in knn[v][:14]:
                if nb in where and where[nb][0] == rB:
                    cand_ws.append(nb)
            if not cand_ws:
                cand_ws = [random.choice(B)]

            for w in cand_ws:
                posB = where[w][1]

                # factibilidad por carga después del intercambio
                newA_load = loads[rA] - demand[v] + demand[w]
                newB_load = loads[rB] - demand[w] + demand[v]
                if newA_load > Q or newB_load > Q:
                    continue

                # remover w de B
                delta_remove_w = remove_delta_at(B, posB, D, depot)

                # construir rutas "sin" v y "sin" w para buscar mejor inserción
                A_wo = A[:posA] + A[posA + 1:]
                B_wo = B[:posB] + B[posB + 1:]

                # mejor inserción de w en A_wo y de v en B_wo
                insPosA, delta_ins_w = best_insertion_position_excluding(A_wo, w, D, depot=depot)
                insPosB, delta_ins_v = best_insertion_position_excluding(B_wo, v, D, depot=depot)

                # delta total
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

        # remover en orden: primero el de mayor índice en cada ruta
        del A[posA]
        del B[posB]
        loads[rA] -= demand[v]
        loads[rB] -= demand[w]

        # insertar w en A, v en B en sus mejores posiciones
        A.insert(insPosA, w)
        B.insert(insPosB, v)
        loads[rA] += demand[w]
        loads[rB] += demand[v]

        # limpiar rutas vacías (raro)
        if not A:
            routes.pop(rA);
            loads.pop(rA)
            if rB > rA:
                rB -= 1
        if rB < len(routes) and not routes[rB]:
            routes.pop(rB);
            loads.pop(rB)

        return True

    return False


# ============================================================
# 3) INICIAL: Sweep (igual que tú)
# ============================================================
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
            raise ValueError(f"Demanda de cliente {i} ({di}) > capacidad Q ({Q})")
        if load + di <= Q:
            cur.append(i)
            load += di
        else:
            routes.append(cur)
            cur, load = [i], di
    if cur:
        routes.append(cur)
    return routes


# ============================================================
# 4) 2-opt intra-ruta usando D (mucho más rápido)
# ============================================================
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


# ============================================================
# 5) Ruin operators: random, related, string-removal (SISR-lite)
# ============================================================
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
    # quita segmentos consecutivos de rutas (SISR-lite)
    removed = []
    if remove_count <= 0:
        return removed, routes

    # quita strings solo de rutas con longitud >= 2
    while remove_count > 0 and len(routes) > 0:
        ri = random.randrange(len(routes))
        r = routes[ri]
        L = len(r)

        if L < 2:
            # no podemos sacar un "string" de longitud >=2
            # opción: quitar ese único nodo (tipo random) o saltar
            # aquí lo quitamos para avanzar en el conteo:
            removed.extend(r)
            remove_count -= L
            routes.pop(ri)
            continue

        # longitud del string: 2..min(25, L)
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


# ============================================================
# 6) Recreate: regret-2 usando D (sin euclid)
# ============================================================
def best_insertion_position_D(route, node, D, depot=0):
    if not route:
        return 0, int(2 * D[depot, node])
    best_pos = 0
    best_delta = 10 ** 18

    # inicio
    delta = D[depot, node] + D[node, route[0]] - D[depot, route[0]]
    best_delta, best_pos = delta, 0

    # medio
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        delta = D[a, node] + D[node, b] - D[a, b]
        if delta < best_delta:
            best_delta, best_pos = delta, i + 1

    # final
    delta = D[route[-1], node] + D[node, depot] - D[route[-1], depot]
    if delta < best_delta:
        best_delta, best_pos = delta, len(route)

    return best_pos, int(best_delta)


def recreate_regret2_D(routes, removed, D, demand, Q, depot=0):
    removed = removed[:]
    random.shuffle(removed)

    loads = [sum(demand[v] for v in r) for r in routes]

    while removed:
        best_choice = None  # (key, node, ri, pos)
        for node in removed:
            di = demand[node]
            cand = []
            for ri in range(len(routes)):
                if loads[ri] + di <= Q:
                    pos, delta = best_insertion_position_D(routes[ri], node, D, depot)
                    cand.append((delta, ri, pos))
            if not cand:
                delta_new = int(2 * D[depot, node])
                cand = [(delta_new, -1, 0)]

            cand.sort(key=lambda x: x[0])
            best_delta, best_ri, best_pos = cand[0]
            second = cand[1][0] if len(cand) > 1 else cand[0][0] + 10 ** 6
            regret = second - best_delta
            key = (regret, -best_delta)
            if best_choice is None or key > best_choice[0]:
                best_choice = (key, node, best_ri, best_pos)

        _, node, ri, pos = best_choice
        di = demand[node]
        if ri == -1:
            routes.append([node])
            loads.append(di)
        else:
            routes[ri].insert(pos, node)
            loads[ri] += di
        removed.remove(node)

    return routes


# ============================================================
# 7) LOCAL SEARCH INTER-RUTAS (relocate + swap) restringido por kNN
# ============================================================
def relocate_best_improvement(routes, loads, D, demand, Q, knn, depot=0, max_moves=4000):
    """
    Mueve 1 cliente de una ruta a otra (mejor mejora), restringiendo candidatos por knn.
    """
    best_gain = 0
    best_move = None  # (from_r, pos_i, to_r, pos_j)

    n_routes = len(routes)
    if n_routes <= 1:
        return False

    # mapa cliente -> (ruta, pos) para referencias rápidas
    where = {}
    for r_idx, r in enumerate(routes):
        for p, v in enumerate(r):
            where[v] = (r_idx, p)

    moves_checked = 0
    for v, (rA, pA) in list(where.items()):
        if moves_checked >= max_moves:
            break

        # candidatos: rutas que contengan vecinos de v (reduce búsqueda)
        candidate_routes = set()
        for nb in knn[v][:15]:
            if nb in where:
                candidate_routes.add(where[nb][0])
        # también permite explorar alguna ruta al azar
        if len(candidate_routes) < 3:
            candidate_routes.add(random.randrange(n_routes))

        for rB in candidate_routes:
            if rB == rA:
                continue
            if loads[rB] + demand[v] > Q:
                continue

            A = routes[rA]
            B = routes[rB]

            # costo de quitar v de A
            prevA = depot if pA == 0 else A[pA - 1]
            nextA = depot if pA == len(A) - 1 else A[pA + 1]
            remove_delta = -D[prevA, v] - D[v, nextA] + D[prevA, nextA]

            # probar inserción de v en B en posiciones “cercanas” a vecinos (y algunas random)
            # generamos set de posiciones candidatas
            cand_pos = set([0, len(B)])
            # ubica posiciones alrededor de vecinos
            for nb in knn[v][:10]:
                if nb in where and where[nb][0] == rB:
                    pb = where[nb][1]
                    cand_pos.update([pb, pb + 1])
            # algunas random para diversificar
            for _ in range(3):
                cand_pos.add(random.randint(0, len(B)))

            for pB in cand_pos:
                prevB = depot if pB == 0 else B[pB - 1]
                nextB = depot if pB == len(B) else B[pB]
                insert_delta = D[prevB, v] + D[v, nextB] - D[prevB, nextB]

                gain = -(remove_delta + insert_delta)  # si positivo, mejora
                moves_checked += 1
                if gain > best_gain:
                    best_gain = gain
                    best_move = (rA, pA, rB, pB)

    if best_move and best_gain > 0:
        rA, pA, rB, pB = best_move
        v = routes[rA].pop(pA)
        loads[rA] -= demand[v]
        # ajustar pB si rutas iguales no aplica; si rA<rB puede haberse corrido el índice
        if rA == rB:
            return False
        # si rA < rB y se eliminó una ruta vacía, cuidado
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
    """
    Intercambia 1 cliente de ruta A con 1 cliente de ruta B (mejor mejora), restringido por knn.
    """
    best_gain = 0
    best_move = None  # (rA,pA,rB,pB)

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

        # rutas candidatas por vecinos
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

            # candidatos de swap en B: vecinos de v dentro de B
            cand_nodes = []
            for nb in knn[v][:12]:
                if nb in where and where[nb][0] == rB:
                    cand_nodes.append(nb)
            if not cand_nodes:
                # fallback aleatorio
                cand_nodes = [random.choice(B)]

            for w in cand_nodes:
                pB = where[w][1]

                # factibilidad por capacidad
                newA = loads[rA] - demand[v] + demand[w]
                newB = loads[rB] - demand[w] + demand[v]
                if newA > Q or newB > Q:
                    continue

                prevB = depot if pB == 0 else B[pB - 1]
                nextB = depot if pB == len(B) - 1 else B[pB + 1]

                # delta en A reemplazando v por w
                deltaA = (-D[prevA, v] - D[v, nextA] + D[prevA, w] + D[w, nextA])
                # delta en B reemplazando w por v
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


### Nuevo cambio ####
def local_search_sotaish(routes, D, demand, Q, knn, depot=0,
                         intra_2opt=True,
                         max_rounds=3):
    # cargas
    loads = [sum(demand[v] for v in r) for r in routes]

    # pulido intra-ruta
    if intra_2opt:
        for i in range(len(routes)):
            routes[i] = two_opt_D(routes[i], D, depot=depot, max_tries=220)

    # rondas inter-rutas (orden típico SOTA: relocate/swap -> or-opt -> 2opt* -> swap*)
    for _ in range(max_rounds):
        moved = False

        moved |= relocate_best_improvement(routes, loads, D, demand, Q, knn, depot=depot, max_moves=2000)
        moved |= swap_best_improvement(routes, loads, D, demand, Q, knn, depot=depot, max_moves=2000)

        moved |= or_opt_best_improvement(routes, loads, D, demand, Q, knn, depot=depot,
                                         seg_lens=(2, 3), max_moves=2000)

        moved |= two_opt_star_best_improvement(routes, loads, D, demand, Q, knn, depot=depot,
                                               max_moves=2000)

        moved |= swap_star_best_improvement(routes, loads, D, demand, Q, knn, depot=depot,
                                            max_moves=2000)

        if not moved:
            break

        # pulido rápido tras cambios
        if intra_2opt:
            for i in range(len(routes)):
                routes[i] = two_opt_D(routes[i], D, depot=depot, max_tries=200)

    return routes


# ============================================================
# 6) PARCHE: tu LNS para aceptar start_routes y route_pool_in
# ============================================================
def lns_cvrp_sotaish_startable(coords, demand, Q, depot=0,
                               iters=20000, time_limit=60.0,
                               k_knn=60,
                               remove_frac=(0.10, 0.35),
                               p_related=0.70,
                               p_string=0.20,
                               ls_rounds=2,
                               seed=42,
                               verbose=True,
                               max_pool=8000,
                               collect_every=1,
                               # ---- NUEVO ----
                               start_routes=None,
                               D_in=None,
                               route_pool_in=None,
                               min_cover_pool=2):
    random.seed(seed)
    np.random.seed(seed)

    # D + kNN
    D = D_in if D_in is not None else build_distance_matrix(coords)
    knn = build_knn_from_D(D, k=k_knn, depot=depot)

    # start
    if start_routes is None:
        routes = initial_sweep(coords, demand, Q, depot)
    else:
        routes = [list(r) for r in start_routes]

    routes = local_search_sotaish(routes, D, demand, Q, knn, depot=depot, intra_2opt=True, max_rounds=ls_rounds)

    best_routes = [r[:] for r in routes]
    best_cost = total_cost_D(best_routes, D, depot)

    cur_routes = [r[:] for r in routes]
    cur_cost = total_cost_D(cur_routes, D, depot)

    # pool (reusar o crear)
    route_pool = route_pool_in if route_pool_in is not None else {}
    n_customers = len(coords) - 1
    add_routes_to_pool_safe(cur_routes, route_pool, D, n_customers, depot=depot, max_pool=max_pool,
                            min_cover=min_cover_pool)
    add_routes_to_pool_safe(best_routes, route_pool, D, n_customers, depot=depot, max_pool=max_pool,
                            min_cover=min_cover_pool)

    start = time.time()
    log_every = 10
    no_improve = 0
    rf_cur = remove_frac
    p_rel_cur = p_related

    rf_min, rf_max = 0.05, 0.45
    p_rel_min, p_rel_max = 0.55, 0.90
    T0 = 0.001 * best_cost

    def pick_remove_count(rf):
        a, b = rf
        return max(1, int(random.uniform(a, b) * n_customers))

    for it in range(1, iters + 1):
        if time.time() - start > time_limit:
            break

        trial = [r[:] for r in cur_routes]

        rc = pick_remove_count(rf_cur)
        u = random.random()
        if u < p_string:
            removed, trial = ruin_string(trial, rc)
        elif u < p_string + p_rel_cur:
            nodes_trial = [v for r in trial for v in r]
            seed_node = random.choice(nodes_trial) if nodes_trial else None
            removed, trial = ruin_related(trial, knn, rc, seed_node=seed_node)
        else:
            removed, trial = ruin_random(trial, rc)

        trial = recreate_regret2_D(trial, removed, D, demand, Q, depot)
        trial = local_search_sotaish(trial, D, demand, Q, knn, depot=depot, intra_2opt=True, max_rounds=ls_rounds)
        trial_cost = total_cost_D(trial, D, depot)

        accepted = False
        if trial_cost < cur_cost:
            accepted = True
        else:
            T = max(1e-9, T0 * (1 - it / iters))
            if random.random() < math.exp(-(trial_cost - cur_cost) / T):
                accepted = True

        if accepted:
            cur_routes, cur_cost = trial, trial_cost

        if (it % collect_every) == 0:
            add_routes_to_pool_safe(trial, route_pool, D, n_customers, depot=depot, max_pool=max_pool, min_cover=5)
            add_routes_to_pool_safe(best_routes, route_pool, D, n_customers, depot=depot, max_pool=max_pool,
                                    min_cover=5)

        if cur_cost < best_cost:
            best_cost = cur_cost
            best_routes = [r[:] for r in cur_routes]
            no_improve = 0
            rf_cur = (max(rf_min, rf_cur[0] - 0.01), max(rf_min, rf_cur[1] - 0.01))
            p_rel_cur = min(p_rel_max, p_rel_cur + 0.02)
        else:
            no_improve += 1

        if no_improve > 400:
            rf_cur = (min(rf_max, rf_cur[0] + 0.03), min(rf_max, rf_cur[1] + 0.05))
            p_rel_cur = max(p_rel_min, p_rel_cur - 0.10)
            no_improve = 0

        if verbose and it % log_every == 0:
            elapsed = time.time() - start
            print(f"[LNS] it {it:6d} | cur {cur_cost:,.0f} | best {best_cost:,.0f} "
                  f"| #routes {len(best_routes):4d} | pool {len(route_pool):5d} "
                  f"| rf={rf_cur[0]:.2f}-{rf_cur[1]:.2f} p_rel={p_rel_cur:.2f} "
                  f"| t={elapsed:5.1f}s")

    return best_routes, best_cost, D, route_pool


# ============================================================
# 7) SPP (TU versión: solve_spp_gurobi) + intensify_from_spp
#    Se asume que ya los pegaste tal cual.
# ============================================================
import gurobipy as gp
from gurobipy import GRB


def solve_spp_gurobi(route_pool,
                     n_customers,
                     time_limit=60.0,
                     mip_gap=0.0,
                     verbose=True,
                     prev_selected_routes=None,
                     fallback_routes=None):
    """
    Set Partitioning Problem (SPP) sobre un pool de rutas.

    Decide un subconjunto de rutas (variables binarias) tal que:
      - Cada cliente i=1..n_customers esté cubierto EXACTAMENTE una vez.
      - Se minimice la suma de costos de rutas seleccionadas.

    Inputs
    ------
    route_pool:
        - dict {route_tuple: cost_int}  (recomendado), o
        - iterable/list de rutas (cada ruta iterable de ints).
          En este caso, el costo se asume desconocido y se lanza error (mejor pasar dict).
    n_customers: int
        # de clientes excluyendo depot (clientes = 1..n_customers)
    time_limit: float
    mip_gap: float
    verbose: bool
    prev_selected_routes:
        iterable de rutas (tuplas) a usar como warm-start (del SPP anterior).
        Deben estar en el pool o se ignoran.
    fallback_routes:
        iterable de rutas (tuplas) (por ejemplo best_routes del LNS) para construir un start factible
        si prev_selected_routes no está disponible.

    Returns
    -------
    selected_routes_list: list[list[int]]
    best_cost: float
    model: gurobipy.Model
    selected_routes_set: set[tuple[int]]
    """

    # -------------------------
    # Normaliza entrada
    # -------------------------
    if isinstance(route_pool, dict):
        pool_dict = route_pool
    else:
        raise TypeError(
            "route_pool debe ser un dict {route_tuple: cost}. "
            "Si tienes lista de rutas, conviértela a dict con costos antes."
        )

    # Filtra rutas vacías / inválidas y rutas con clientes fuera de rango
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
        raise ValueError("route_pool vacío o sin rutas válidas.")

    routes = list(filtered.keys())
    costs = [filtered[r] for r in routes]

    # -------------------------
    # Construir cobertura
    # cover[i] = lista de índices r tales que ruta r contiene i
    # -------------------------
    cover = [[] for _ in range(n_customers + 1)]
    for ridx, rt in enumerate(routes):
        for i in rt:
            cover[i].append(ridx)

    missing = [i for i in range(1, n_customers + 1) if len(cover[i]) == 0]
    if missing:
        raise ValueError(
            f"SPP infeasible: clientes sin cobertura en el pool (ej. {missing[:10]}...). "
            "Aumenta max_pool / collect_every o asegúrate de agregar best_routes/trial al pool."
        )

    # -------------------------
    # Modelo
    # -------------------------
    m = gp.Model("SPP_CVRP")
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.MIPFocus = 1  ### Soluciones rapidas
    m.Params.Presolve = 2  ### Movemos COLUMNAS (RUTAS) QUE NO GUROBI
    m.Params.CliqueCuts = 2  ### ESTA ME GUROBI PARA SET PARTITIONING FORMULATIONS

    # Variables: x[r] = 1 si seleccionas ruta r
    x = m.addVars(len(routes), vtype=GRB.BINARY, name="x")

    # Objetivo
    m.setObjective(gp.quicksum(costs[r] * x[r] for r in range(len(routes))), GRB.MINIMIZE)

    # Exact cover: cada cliente exactamente una vez
    for i in range(1, n_customers + 1):
        m.addConstr(gp.quicksum(x[r] for r in cover[i]) == 1, name=f"cover_{i}")

    # -------------------------
    # Warm-start (muy recomendable)
    # -------------------------
    start_set = set()

    if prev_selected_routes is not None:
        start_set |= {tuple(map(int, r)) for r in prev_selected_routes}

    if (not start_set) and (fallback_routes is not None):
        start_set |= {tuple(map(int, r)) for r in fallback_routes}

    if start_set:
        # Pon Start=1 solo si la ruta existe en este pool filtrado
        pool_set = set(routes)
        for r in range(len(routes)):
            x[r].Start = 1.0 if routes[r] in pool_set and routes[r] in start_set else 0.0

    # -------------------------
    # Optimiza
    # -------------------------
    m.optimize()

    if m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        # (en SPP debería ser raro si hay cobertura suficiente + dummies, etc.)
        raise ValueError("SPP terminó INFEASIBLE/INF_OR_UNBD. Revisa el pool o agrega rutas dummy.")
    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        raise ValueError(f"SPP terminó con status={m.Status}")

    # -------------------------
    # Extraer solución
    # -------------------------
    selected_routes = []
    selected_set = set()
    for r in range(len(routes)):
        if x[r].X > 0.5:
            selected_set.add(routes[r])
            selected_routes.append(list(routes[r]))

    # costo (usa objVal del modelo)
    best_cost = float(m.ObjVal) if m.SolCount > 0 else float("inf")

    # Sanity check: cobertura exacta
    # (opcional, pero útil si estás debuggeando)
    if selected_routes:
        seen = []
        for rt in selected_routes:
            seen.extend(rt)
        if len(seen) != n_customers or len(set(seen)) != n_customers:
            # Esto NO debería pasar si las restricciones están bien
            raise RuntimeError("SPP devolvió una solución que no cubre exactamente una vez. Revisa el modelo.")

    return selected_routes, best_cost, m, selected_set


def intensify_from_spp(spp_routes, D, coords, demand, Q,
                       route_pool,  # <-- NUEVO: pool compartido
                       time_limit=120.0, iters=30000,
                       k_knn=40,
                       remove_frac=(0.03, 0.15),
                       p_related=0.85,
                       p_string=0.10,
                       ls_rounds=2,
                       seed=42,
                       verbose=True,
                       max_pool=20000,
                       collect_every=1,
                       min_cover=5,
                       depot=0):
    random.seed(seed)
    np.random.seed(seed)

    n_customers = len(coords) - 1
    knn = build_knn_from_D(D, k=k_knn, depot=depot)

    # Start desde SPP
    cur_routes = [list(r) for r in spp_routes]
    cur_routes = local_search_sotaish(cur_routes, D, demand, Q, knn,
                                      depot=depot, intra_2opt=True, max_rounds=ls_rounds)

    cur_cost = total_cost_D(cur_routes, D, depot)
    best_routes = [r[:] for r in cur_routes]
    best_cost = cur_cost

    # IMPORTANTE: NO reiniciar route_pool
    # solo asegurar que best esté adentro
    add_routes_to_pool_safe(best_routes, route_pool, D, n_customers,
                            depot=depot, max_pool=max_pool, min_cover=min_cover)

    def pick_remove_count():
        a, b = remove_frac
        return max(1, int(random.uniform(a, b) * n_customers))

    T0 = 0.002 * best_cost

    start = time.time()
    log_every = 10

    for it in range(1, iters + 1):
        if time.time() - start > time_limit:
            break

        trial = [r[:] for r in cur_routes]

        rc = pick_remove_count()
        u = random.random()
        if u < p_string:
            removed, trial = ruin_string(trial, rc)
        elif u < p_string + p_related:
            nodes_trial = [v for r in trial for v in r]
            seed_node = random.choice(nodes_trial) if nodes_trial else None
            removed, trial = ruin_related(trial, knn, rc, seed_node=seed_node)
        else:
            removed, trial = ruin_random(trial, rc)

        trial = recreate_regret2_D(trial, removed, D, demand, Q, depot)

        trial = local_search_sotaish(trial, D, demand, Q, knn,
                                     depot=depot, intra_2opt=True, max_rounds=ls_rounds)

        trial_cost = total_cost_D(trial, D, depot)

        if trial_cost < cur_cost:
            cur_routes, cur_cost = trial, trial_cost
        else:
            T = max(1e-9, T0 * (1 - it / iters))
            if random.random() < math.exp(-(trial_cost - cur_cost) / T):
                cur_routes, cur_cost = trial, trial_cost

        # guardar en el MISMO pool global
        if (it % collect_every) == 0:
            add_routes_to_pool_safe(trial, route_pool, D, n_customers,
                                    depot=depot, max_pool=max_pool, min_cover=min_cover)
            add_routes_to_pool_safe(best_routes, route_pool, D, n_customers,
                                    depot=depot, max_pool=max_pool, min_cover=min_cover)

        if cur_cost < best_cost:
            best_cost = cur_cost
            best_routes = [r[:] for r in cur_routes]

        if verbose and it % log_every == 0:
            print(f"[INT] it {it:6d} | cur {cur_cost:,.0f} | best {best_cost:,.0f} "
                  f"| pool {len(route_pool):5d} | t={(time.time() - start):5.1f}s")

    return best_routes, best_cost, route_pool


# ============================================================
# 8) FUNCIÓN FINAL INTEGRADA:
#    Stage-1 PyVRP+SPP -> Stage-2 tu run_lns_spp_cycles (con start_routes)
# ============================================================
def run_lns_spp_cycles_with_pyvrp_start(coords, demand, Q,
                                       total_time=300.0,
                                       n_cycles=3,
                                       seed=42,
                                       depot=0,
                                       verbose=True,
                                       k_schedule=None,
                                       time_limit_cluster=10.0,
                                       n_restarts=10,
                                       extra_vehicles=2,
                                       spp1_timelimit=120.0):

    n_customers = len(coords) - 1
    start_global = time.time()
    # ==========================
    # STAGE 1: PyVRP+SPP start
    # ==========================
    if verbose:
        print("\n==============================")
        print("STAGE 1: CLUSTER + PyVRP + SPP")
        print("==============================")

    start_routes, start_obj, pool_stage1 = stage1_pyvrp_multik_build_pool_then_one_spp(
        coords, demand, Q,
        k_schedule=[8, 5, 3, 1],  # o None para default
        seed=seed,
        time_limit_cluster=time_limit_cluster,
        n_restarts=n_restarts,
        extra_vehicles=extra_vehicles,
        vehicle_penalty=0.0,
        spp_timelimit=spp1_timelimit,
        verbose=verbose
    )

    # ==========================
    # STAGE 2: TU pipeline LNS+SPP cycles
    # ==========================
    # Construimos D UNA sola vez aquí y lo reusamos en todo Stage 2
    D = build_distance_matrix(coords)

    # FASE 0: LNS EXPLORE (arrancando desde start_routes)
    t_explore = 0.15 * total_time
    if verbose:
        print("\n==============================")
        print("STAGE 2 / PHASE 0: LNS EXPLORE (start=PyVRP+SPP)")
        print("==============================")

    best_routes, best_cost, D, route_pool = lns_cvrp_sotaish_startable(
        coords, demand, Q,
        depot=depot,
        iters=10 ** 7,
        time_limit=t_explore,
        k_knn=80 if n_customers > 300 else 30,
        remove_frac=(0.20, 0.45),
        p_related=0.70,
        p_string=0.20,
        ls_rounds=1,
        seed=seed,
        verbose=verbose,
        max_pool=200000 if n_customers > 500 else 25000,
        collect_every=1,
        start_routes=start_routes,
        D_in=D,  # ✅ ahora sí existe
        route_pool_in=None,
        min_cover_pool=5
    )

    # ==========================
    # SPP 0 (warm-start con best_routes)
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
    # CICLOS
    # ==========================
    remaining_time = total_time - (time.time() - start_global)
    #if remaining_time <= 0:
        #return global_best_routes, global_best_cost, D

    #t_cycle = remaining_time / max(1, n_cycles)
    t_cycle = 90

    for k in range(1, n_cycles + 1):
        #if time.time() - start_global > total_time:
            #break

        if verbose:
            print(f"\n=== CYCLE {k}: INTENSIFY ===")

        best_k_routes, best_k_cost, route_pool = intensify_from_spp(
            spp_routes,
            D, coords, demand, Q,
            route_pool=route_pool,
            time_limit=0.85 * t_cycle,
            iters=10 ** 7,
            k_knn=80 if n_customers > 300 else 45,
            remove_frac=(0.05, 0.25),
            p_related=0.85,
            p_string=0.10,
            ls_rounds=1 if n_customers > 300 else 2,
            seed=seed + k,
            verbose=verbose,
            max_pool=150000 if n_customers > 500 else 30000,
            collect_every=1,
            depot=depot
        )

        if verbose:
            print(f"\n=== SPP {k} (warm-start) ===")

        spp_routes, spp_cost, spp_model, prev_sel = solve_spp_gurobi(
            route_pool,
            n_customers=n_customers,
            time_limit=300,
            mip_gap=0.0,
            verbose=verbose,
            prev_selected_routes=prev_sel,
            fallback_routes=best_routes
        )

        if spp_cost < global_best_cost:
            global_best_cost = spp_cost
            global_best_routes = [list(r) for r in spp_routes]
            if verbose:
                print(f"★ New GLOBAL BEST after cycle {k}: {global_best_cost:,.0f} | routes: {len(global_best_routes)}")
                solution_file = f"XL-n1048-k237_{global_best_cost}.sol"
                final_cost = total_cost_D(global_best_routes, D, depot=0)
                write_solution(global_best_routes, final_cost, solution_file)


    return global_best_routes, global_best_cost, D


def parse_args():
    """
    Simple argument parser for command-line usage.
    Usage: python Test4.py --file path/to/instance.vrp
           python Test4.py --random --n 1000 --Q 100
    """
    import sys
    args = sys.argv[1:]

    if not args:
        return None  # Use default in script

    config = {
        "mode": "file",
        "file": None,
        "n_customers": 1000,
        "Q": 100,
        "seed": 42
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
        else:
            i += 1

    return config


# ===== Code Cell 2 =====
# ============================================================
# 9) EJEMPLO DE USO
# ============================================================

# Check if command-line arguments were provided
config = parse_args()

if config is not None:
    # Use command-line arguments
    if config["mode"] == "file":
        if config["file"] is None:
            print("Error: --file requires a path argument")
            print("Usage: python Test4.py --file path/to/instance.vrp")
            exit(1)
        coords, demand, Q = load_instance_from_file(config["file"])
    else:  # random
        coords, demand, Q = make_random_instance(
            n_customers=config["n_customers"],
            Q=config["Q"],
            seed=config["seed"]
        )
else:
    # Default behavior (modify as needed):
    # OPTION 1: Use a random generated instance
    # coords, demand, Q = make_random_instance(n_customers=1000, Q=100, seed=42)

    # OPTION 2: Load from a .vrp file (CVRPLib format)
    instance_file = "XL/XL/XL-n1048-k237.vrp"
    coords, demand, Q = load_instance_from_file(instance_file)

best_routes, best_cost, D = run_lns_spp_cycles_with_pyvrp_start(
    coords, demand, Q,
    total_time=7200.0,
    n_cycles=30,
    seed=42,
    k_schedule=[8,5,3,1],     # <- NUEVO
    time_limit_cluster=1800.0,
    n_restarts=1,
    extra_vehicles=10,                  # (si lo tienes en la firma; si no, quítalo)
    spp1_timelimit=1800.0,
    verbose=True
)
print("FINAL:", best_cost, "routes:", len(best_routes))

# ===== Code Cell 3 =====
### Vibe coding para vereficar factibilidad ###

def check_feasibility_cvrp(routes, demand, Q, n_customers, depot=0, verbose=True):
    """
    routes: lista de rutas (listas o tuplas de clientes)
    demand: demandas (indexadas por cliente)
    Q: capacidad
    n_customers: número de clientes (sin depot)
    """
    ok = True

    # 1) cobertura exacta
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
            print("❌ Clientes NO cubiertos:", sorted(list(missing))[:10], "...")

    if duplicated:
        ok = False
        if verbose:
            print("❌ Clientes duplicados:", duplicated[:10], "...")

    # 2) depot no debe aparecer
    if depot in seen_set:
        ok = False
        if verbose:
            print("❌ El depot aparece en alguna ruta.")

    # 3) capacidad por ruta
    for k, r in enumerate(routes):
        load = sum(demand[v] for v in r)
        if load > Q + 1e-9:
            ok = False
            if verbose:
                print(f"❌ Ruta {k} viola capacidad: load={load} > Q={Q}")

    # 4) rangos válidos
    for v in seen_set:
        if v < 1 or v > n_customers:
            ok = False
            if verbose:
                print("❌ Cliente fuera de rango:", v)

    if ok and verbose:
        print("✅ Solución FACTIBLE (CVRP)")

    return ok


# ===== Code Cell 4 =====
### Imprime RUTA SPP ###

def plot_solution_with_cost(routes, coords, D, depot=0, max_routes=1000):
    # Costo total (consistente con SPP / LNS)
    total_cost = total_cost_D([list(r) for r in routes], D, depot)

    plt.figure(figsize=(8, 8))

    # clientes
    xs = [coords[i][0] for i in range(len(coords))]
    ys = [coords[i][1] for i in range(len(coords))]
    plt.scatter(xs[1:], ys[1:], s=6, c="black", alpha=0.6, label="Clients")

    # depot
    plt.scatter(xs[depot], ys[depot], c="red", s=120, marker="s", label="Depot")

    # rutas (limitamos cuántas dibujar)
    for r in routes[:max_routes]:
        path = [depot] + list(r) + [depot]
        px = [coords[i][0] for i in path]
        py = [coords[i][1] for i in path]
        plt.plot(px, py, linewidth=0.5)

    plt.title(
        f"Best SPP solution | Total cost = {total_cost:,.0f} "
        f"| Routes = {len(routes)}"
    )
    plt.axis("equal")
    #plt.legend(loc="upper right")
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.show()

# ===== Code Cell 5 =====
# verificación
check_feasibility_cvrp(
    best_routes,
    demand,
    Q,
    n_customers=len(coords)-1,
    depot=0,
    verbose=True
)

import matplotlib.pyplot as plt
# plot
plot_solution_with_cost(best_routes, coords, D, depot=0, max_routes=1000)

# ===== Code Cell 6 =====
print(best_routes)

# ===== Code Cell 7 =====
import os, sys

HEXALY_BIN = r"C:\Hexaly_14_0\bin"
HEXALY_PY = r"C:\Hexaly_14_0\bin\python"

# 1) Que Python encuentre el paquete hexaly
if HEXALY_PY not in sys.path:
    sys.path.insert(0, HEXALY_PY)

# 2) Que Windows encuentre las DLLs nativas (CRÍTICO)
os.add_dll_directory(HEXALY_BIN)
os.environ["PATH"] = HEXALY_BIN + ";" + os.environ.get("PATH", "")

# 3) Ya puedes importar
import hexaly.optimizer

# 4) Licencia#hexaly.HxVersion.license_path = r"C:\Hexaly_14_0\license.dat"

print("Hexaly imports OK")


# ===== Code Cell 8 =====
def apply_warmstart_to_hexaly(customers_sequences, warm_routes, nb_customers):
    """
    customers_sequences: lista de variables Hexaly list(nb_customers)
    warm_routes: rutas en índices originales (con depot 0 opcional), e.g. [0, 5, 7, 2, 0]
                 o también puedes pasar [5,7,2] (sin depot)
    nb_customers: n
    """
    if warm_routes is None:
        return

    # Normaliza rutas: quita depot 0 si viene y convierte a 0..n-1
    norm = []
    used = set()
    for r in warm_routes:
        rr = list(r)
        if len(rr) >= 2 and rr[0] == 0 and rr[-1] == 0:
            rr = rr[1:-1]
        # ahora rr tiene clientes en 1..n (tu convención)
        rr0 = []
        for v in rr:
            v = int(v)
            if v == 0:
                continue
            if not (1 <= v <= nb_customers):
                raise ValueError(f"Warm-start inválido: cliente {v} fuera de 1..{nb_customers}")
            vv = v - 1  # Hexaly usa 0..n-1
            if vv in used:
                raise ValueError(f"Warm-start inválido: cliente repetido {v}")
            used.add(vv)
            rr0.append(vv)
        norm.append(rr0)

    if len(used) != nb_customers:
        missing = sorted(set(range(nb_customers)) - used)
        # OJO: si quieres permitir warm-start parcial, puedes quitar este raise.
        raise ValueError(f"Warm-start incompleto: faltan {len(missing)} clientes (ej: {[m + 1 for m in missing[:10]]})")

    # 1) Limpia TODAS las listas
    for seq in customers_sequences:
        seq.value.clear()

    # 2) Carga rutas en las primeras listas
    for k, rr0 in enumerate(norm):
        if k >= len(customers_sequences):
            break
        for v0 in rr0:
            customers_sequences[k].value.add(int(v0))


def solve_cvrp_hexaly(coords, demand, Q, time_limit=20, warmstart_routes=None):
    nb_customers = len(coords) - 1
    if nb_customers <= 0:
        return {"nb_trucks_used": 0, "total_distance": 0, "routes": []}

    demands_data = list(demand[1:])  # clientes 0..n-1
    dist_matrix_data, dist_depot_data = build_distance_data(coords)

    nb_trucks = math.ceil(sum(demands_data) / Q) + 20

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        model = optimizer.model
        #model.set_parameter("LogPeriod", 10.0)

        customers_sequences = [model.list(nb_customers) for _ in range(nb_trucks)]
        model.constraint(model.partition(customers_sequences))

        demands = model.array(demands_data)
        dist_matrix = model.array(dist_matrix_data)
        dist_depot = model.array(dist_depot_data)

        trucks_used = [(model.count(customers_sequences[k]) > 0) for k in range(nb_trucks)]
        dist_routes = [None] * nb_trucks

        for k in range(nb_trucks):
            sequence = customers_sequences[k]
            c = model.count(sequence)

            demand_lambda = model.lambda_function(lambda j: demands[j])
            route_qty = model.sum(sequence, demand_lambda)
            model.constraint(route_qty <= Q)

            dist_lambda = model.lambda_function(
                lambda i: model.at(dist_matrix, sequence[i - 1], sequence[i])
            )
            dist_routes[k] = model.sum(model.range(1, c), dist_lambda) + model.iif(
                c > 0,
                dist_depot[sequence[0]] + dist_depot[sequence[c - 1]],
                0
            )

        total_distance = model.sum(dist_routes)
        model.minimize(total_distance)

        model.close()

        # ✅ WARM-START AQUÍ (model ya está cerrado)
        if warmstart_routes is not None:
            apply_warmstart_to_hexaly(customers_sequences, warmstart_routes, nb_customers)

        optimizer.param.time_limit = int(time_limit)
        #optimizer.param.Time_between_displays = 5
        optimizer.solve()

        routes = []
        for k in range(nb_trucks):
            if trucks_used[k].value != 1:
                continue
            seq = list(customers_sequences[k].value)  # 0..n-1
            routes.append([0] + [i + 1 for i in seq] + [0])  # regresa a 1..n

        return {
            "nb_trucks_model": nb_trucks,
            #"nb_trucks_used": int(nb_trucks_used.value),
            "total_distance": int(total_distance.value),
            "routes": routes,
        }


def _eucl_2d_int(a, b):
    return int(round(math.hypot(a[0] - b[0], a[1] - b[1])))


def build_distance_data(coords):
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


# ===== Code Cell 9 =====
sol = solve_cvrp_hexaly(coords, demand, Q, time_limit=3600, warmstart_routes=best_routes)


# ===== Code Cell 10 =====
def route_distance_from_D(route, coords):
    # route: [0, i, j, ..., 0] con indices de coords (0..n)
    dist = 0
    for a, b in zip(route, route[1:]):
        dist += _eucl_2d_int(coords[a], coords[b])
    return dist

def route_load(route, demand):
    # route incluye depot 0
    return sum(demand[i] for i in route if i != 0)

def print_routes(routes, coords, demand):
    for t, r in enumerate(routes, 1):
        d = route_distance_from_D(r, coords)
        q = route_load(r, demand)
        print(f"Truck {t:02d} | load={q:4d} | dist={d:6d} | {r}")


# ===== Code Cell 11 =====
hex_routes = sol["routes"]              # rutas tipo [0, i, j, ..., 0]
start_routes = [r[1:-1] for r in hex_routes if len(r) > 2]
plot_solution_with_cost(
    start_routes,
    coords,
    D,
    depot=0,
    max_routes=1000
)
print_routes(hex_routes, coords, demand)

# Write final solution to .sol file
instance_name = os.path.splitext(os.path.basename(instance_file))[0]
solution_file = f"{instance_name}.sol"
final_cost = total_cost_D(start_routes, D, depot=0)
write_solution(start_routes, final_cost, solution_file)
print(f"\nSolution written to {solution_file} with cost: {final_cost}")
