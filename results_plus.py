# ============================================================
# RESULTS ENHANCEMENT (robustness checks), built on the full consistent run:
# FULL CONSISTENT RUN: one network, one set of points, for centrality,
# baseline simulation and scenario tests.  (built on scenario_v1 / sim_v1)
# Scenario tests v1 (built on the untuned baseline simulation):
# does adding capacity near a location reduce simulated network-wide
# travel time, and does the diagnosis predict where it helps most?
#
# Run in Colab (needs internet for the OSM download).
#   !pip install osmnx pandas==2.2.3 -q   -> then Runtime > Restart
# Upload Banglore_traffic_Dataset.csv, then run this file.
# Runtime: a few minutes.
#
# Nothing here is fitted to the Kaggle numbers. They are only
# used at the end, as a check.
# ============================================================
import time
import numpy as np
import pandas as pd
import networkx as nx

CSV_PATH = "Banglore_traffic_Dataset.csv"

# --- Standard default assumptions (textbook-style, NOT tuned) ---
FREE_SPEED = {  # km/h
    "motorway": 60, "motorway_link": 40, "trunk": 50, "trunk_link": 35,
    "primary": 40, "primary_link": 30, "secondary": 30, "secondary_link": 25,
}
CAPACITY = {  # vehicles/hour per road segment (rough)
    "motorway": 4000, "motorway_link": 2000, "trunk": 3000, "trunk_link": 1500,
    "primary": 2200, "primary_link": 1200, "secondary": 1600, "secondary_link": 900,
}
BPR_ALPHA, BPR_BETA = 0.15, 4          # standard BPR delay curve
STEPS = (0.4, 0.3, 0.2, 0.1)            # incremental assignment
N_ORIGINS, N_DESTS = 150, 10            # trips per run = 1500
DEMAND_LEVELS = [60, 120, 240]          # veh/hr per trip (sensitivity)
SEEDS = [0, 1, 2]
DEMAND_MODELS = ["uniform", "center_weighted"]
CENTER = (12.9750, 77.6070)             # MG Road area, for the center-weighted model

# Paste your VERIFIED coordinates here (lat, lon), as checked on Google Maps.
LOCATIONS = {
    "100 Feet Road": (12.9716, 77.6412), "CMH Road": (12.9819, 77.6389),
    "Marathahalli Bridge": (12.9569, 77.7011), "ITPL Main Road": (12.9860, 77.7371),
    "Sony World Junction": (12.9351, 77.6146), "Sarjapur Road": (12.9258, 77.6309),
    "Trinity Circle": (12.9737, 77.6178), "Anil Kumble Circle": (12.9789, 77.6088),
    "Jayanagar 4th Block": (12.9308, 77.5838), "South End Circle": (12.9412, 77.5831),
    "Hebbal Flyover": (13.0355, 77.5971), "Ballari Road": (13.0410, 77.5960),
    "Yeshwanthpur Circle": (13.0243, 77.5540), "Tumkur Road": (13.0280, 77.5450),
    "Silk Board Junction": (12.9172, 77.6228), "Hosur Road": (12.8990, 77.6360),
}


# ---------------- core model (no internet needed) ----------------
def build_digraph(G):
    """OSM MultiDiGraph -> simple DiGraph with free-flow time and capacity."""
    D = nx.DiGraph()
    for n, d in G.nodes(data=True):
        D.add_node(n, x=d["x"], y=d["y"])
    for u, v, k, d in G.edges(keys=True, data=True):
        h = d.get("highway", "secondary")
        if isinstance(h, list):
            h = h[0]
        speed_ms = FREE_SPEED.get(h, 30) * 1000 / 3600
        t0 = d.get("length", 100.0) / speed_ms
        if D.has_edge(u, v) and D[u][v]["t0"] <= t0:
            continue
        D.add_edge(u, v, t0=t0, cap=CAPACITY.get(h, 1200), flow=0.0, t=t0)
    comp = max(nx.strongly_connected_components(D), key=len)
    return D.subgraph(comp).copy()


def snap_locations(D, locations):
    nodes = list(D.nodes)
    xs = np.array([D.nodes[n]["x"] for n in nodes])
    ys = np.array([D.nodes[n]["y"] for n in nodes])
    out = {}
    for name, (lat, lon) in locations.items():
        dx = (xs - lon) * 111.32 * np.cos(np.radians(lat))
        dy = (ys - lat) * 110.57
        out[name] = nodes[int(np.argmin(np.hypot(dx, dy)))]
    return out


def make_od(D, model, q, rng):
    nodes = list(D.nodes)
    if model == "uniform":
        w = np.ones(len(nodes))
    else:  # more trips start/end near the centre
        xs = np.array([D.nodes[n]["x"] for n in nodes])
        ys = np.array([D.nodes[n]["y"] for n in nodes])
        dx = (xs - CENTER[1]) * 111.32 * np.cos(np.radians(CENTER[0]))
        dy = (ys - CENTER[0]) * 110.57
        w = 1.0 / (1.0 + np.hypot(dx, dy) / 3.0)
    w = w / w.sum()
    od = []
    for oi in rng.choice(len(nodes), N_ORIGINS, p=w):
        for di in rng.choice(len(nodes), N_DESTS, p=w):
            if di != oi:
                od.append((nodes[oi], nodes[di], q))
    return od


def assign(D, od):
    """Incremental assignment: route, update delays, repeat."""
    for _, _, d in D.edges(data=True):
        d["flow"], d["t"] = 0.0, d["t0"]
    for frac in STEPS:
        by_origin = {}
        for o, dst, q in od:
            by_origin.setdefault(o, []).append((dst, q * frac))
        for o, lst in by_origin.items():
            paths = nx.single_source_dijkstra_path(D, o, weight="t")
            for dst, q in lst:
                p = paths.get(dst)
                if p is None or len(p) < 2:
                    continue
                for a, b in zip(p[:-1], p[1:]):
                    D[a][b]["flow"] += q
        for _, _, d in D.edges(data=True):
            d["t"] = d["t0"] * (1 + BPR_ALPHA * (d["flow"] / d["cap"]) ** BPR_BETA)


def location_scores(D, snapped):
    out = {}
    for name, n in snapped.items():
        vc = [d["flow"] / d["cap"] for _, _, d in D.out_edges(n, data=True)]
        vc += [d["flow"] / d["cap"] for _, _, d in D.in_edges(n, data=True)]
        out[name] = float(np.mean(vc)) if vc else np.nan
    return pd.Series(out)


def run_all(D, snapped, kaggle):
    rows, per_loc = [], {}
    for model in DEMAND_MODELS:
        for q in DEMAND_LEVELS:
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                assign(D, make_od(D, model, q, rng))
                sc = location_scores(D, snapped)
                allvc = np.array([d["flow"] / d["cap"] for _, _, d in D.edges(data=True)])
                rho = sc.corr(kaggle.reindex(sc.index), method="spearman")
                rows.append({"model": model, "q": q, "seed": seed, "spearman": rho,
                             "net_mean_vc": allvc.mean(), "net_p95_vc": np.percentile(allvc, 95)})
                per_loc[(model, q, seed)] = sc
    return pd.DataFrame(rows), per_loc



# ================= Scenario tests =================
import os

CONFIGS = [("center_weighted", 60)]   # (demand pattern, veh/hr per trip); add more to test sensitivity
SEEDS_SC = [0, 1, 2, 3]               # same trips are reused for baseline and every scenario within a seed
CAP_FACTOR = 1.5                      # +50% capacity on treated segments
RADIUS_KM = 1.5                       # segments with midpoint within this distance of the site are treated
CEN_CSV = "real_osm_centrality_results.csv"   # optional: gives the centrality-based priority score


def treated_edges(D, snapped):
    out = {}
    for name, n in snapped.items():
        x0, y0 = D.nodes[n]["x"], D.nodes[n]["y"]
        lst = []
        for u, v in D.edges:
            xm = (D.nodes[u]["x"] + D.nodes[v]["x"]) / 2
            ym = (D.nodes[u]["y"] + D.nodes[v]["y"]) / 2
            dx = (xm - x0) * 111.32 * np.cos(np.radians(y0))
            dy = (ym - y0) * 110.57
            if np.hypot(dx, dy) <= RADIUS_KM:
                lst.append((u, v))
        out[name] = lst
    return out


def snapshot(D):
    flow = {(u, v): d["flow"] for u, v, d in D.edges(data=True)}
    tstt = sum(d["flow"] * d["t"] for _, _, d in D.edges(data=True))      # total system travel time
    over = sum(1 for _, _, d in D.edges(data=True) if d["flow"] / d["cap"] > 1)
    return {"flow": flow, "tstt": tstt, "over": over}


def run_scenarios(D, snapped, treated, configs, seeds, progress=True):
    rows = []
    for _, _, d in D.edges(data=True):
        d["cap0"] = d["cap"]
    for model, q in configs:
        for seed in seeds:
            od = make_od(D, model, q, np.random.default_rng(seed))
            assign(D, od)
            base = snapshot(D)
            base_loc = location_scores(D, snapped)
            for site, edges in treated.items():
                for u, v in edges:
                    D[u][v]["cap"] = D[u][v]["cap0"] * CAP_FACTOR
                assign(D, od)
                sc = snapshot(D)
                for u, v in edges:
                    D[u][v]["cap"] = D[u][v]["cap0"]
                tset = set(edges)
                spill = sum(1 for e, f in sc["flow"].items()
                            if e not in tset and f > base["flow"][e] * 1.10 and f - base["flow"][e] > 20)
                rows.append({"model": model, "q": q, "seed": seed, "site": site, "n_treated": len(edges),
                             "d_tstt_pct": 100 * (sc["tstt"] - base["tstt"]) / base["tstt"],
                             "d_over": sc["over"] - base["over"], "spill": spill,
                             "site_base_vc": base_loc[site]})
            if progress:
                print(f"  done: {model}, q={q}, seed={seed}", flush=True)
    return pd.DataFrame(rows)


def summarize(res, kaggle):
    tab = res.groupby("site").agg(
        n_treated=("n_treated", "first"), d_tstt_pct=("d_tstt_pct", "mean"), d_tstt_sd=("d_tstt_pct", "std"),
        d_over=("d_over", "mean"), spill=("spill", "mean"), site_base_vc=("site_base_vc", "mean"))
    tab["improvement_pct"] = -tab["d_tstt_pct"]            # positive = travel time falls
    tab["obs_congestion"] = kaggle.reindex(tab.index)
    if os.path.exists(CEN_CSV):
        cen = pd.read_csv(CEN_CSV).set_index("Road/Intersection Name")
        tab["priority_score"] = cen["bottleneck_score"].reindex(tab.index)
    tab = tab.sort_values("improvement_pct", ascending=False)
    print("=== Effect of +%d%% capacity within %.1f km of each site (mean over seeds) ===" % (100 * (CAP_FACTOR - 1), RADIUS_KM))
    print("improvement_pct: drop in total network travel time (bigger = more help)")
    print("d_over: change in number of segments over capacity (negative = fewer)")
    print("spill: untreated segments that gain >10% traffic (diversion)\n")
    print(tab.round(2).to_string())
    print("\n=== Does the diagnosis predict where capacity helps? (Spearman, n = %d sites) ===" % len(tab))
    for col in ["obs_congestion", "site_base_vc", "priority_score"]:
        if col in tab.columns:
            print(f"  improvement vs {col:15s}: {tab['improvement_pct'].corr(tab[col], method='spearman'):+.2f}")
    top = tab["obs_congestion"].nlargest(5).index
    print("\nMean improvement, 5 most congested (observed) sites : %.3f%%" % tab.loc[top, "improvement_pct"].mean())
    print("Mean improvement, other 11 sites                    : %.3f%%" % tab.drop(top)["improvement_pct"].mean())
    return tab



# ================= Full consistent run =================
TOP = 6                       # "matched" = top TOP in both observed congestion and simulated load
MERGE_SHARED = True           # locations that land on the same network node become one site
BASE_LEVELS = [60, 120]       # baseline demand levels (240 dropped: network oversaturated)
BASE_SEEDS = [0, 1, 2]
SEEDS_A = [0, 1, 2]           # scenario seeds, radius treatment, both demand patterns
RUN_EQUAL_SIZE = True         # also treat the K nearest segments at every site (removes zone-size confound)
K_NEAREST = 60
SEEDS_B = [0, 1, 2]           # equal-size treatment, uniform demand only
OUT = "full_run"


def merge_sites(snapped, kaggle_raw):
    if not MERGE_SHARED:
        return dict(snapped), kaggle_raw.copy()
    groups = {}
    for name, n in snapped.items():
        groups.setdefault(n, []).append(name)
    merged, kag = {}, {}
    for n, names in groups.items():
        label = " / ".join(sorted(names))
        merged[label] = n
        kag[label] = float(np.mean([kaggle_raw[x] for x in names]))
        if len(names) > 1:
            print(f"  NOTE: {names} land on the same node -> merged into one site '{label}'")
    return merged, pd.Series(kag)


def nearest_k_edges(D, snapped, K):
    edges = list(D.edges)
    mx = np.array([(D.nodes[u]["x"] + D.nodes[v]["x"]) / 2 for u, v in edges])
    my = np.array([(D.nodes[u]["y"] + D.nodes[v]["y"]) / 2 for u, v in edges])
    out = {}
    for name, n in snapped.items():
        x0, y0 = D.nodes[n]["x"], D.nodes[n]["y"]
        dx = (mx - x0) * 111.32 * np.cos(np.radians(y0))
        dy = (my - y0) * 110.57
        idx = np.argsort(np.hypot(dx, dy))[:K]
        out[name] = [edges[i] for i in idx]
    return out


def centrality_table(D, snapped, kaggle):
    Gu = D.to_undirected()
    n = Gu.number_of_nodes()
    if n <= 3500:
        bc = nx.betweenness_centrality(Gu, weight="t0", normalized=True)
    else:
        print(f"  large graph ({n} nodes): using sampled betweenness (k=500)")
        bc = nx.betweenness_centrality(Gu, k=500, weight="t0", normalized=True, seed=42)
    t = pd.DataFrame({"betweenness": {s: bc[nd] for s, nd in snapped.items()}, "avg_congestion": kaggle})
    t["centrality_rank"] = t["betweenness"].rank(ascending=False)
    t["congestion_rank"] = t["avg_congestion"].rank(ascending=False)
    N = len(t)
    t["bottleneck_score"] = (N + 1 - t["centrality_rank"]) + (N + 1 - t["congestion_rank"])
    t.index.name = "Road/Intersection Name"
    return t



# ================= Robustness checks =================
from scipy import stats
RUN_P2 = True      # capacity sensitivity (+25%, +100%)
RUN_P3 = True      # zone-size sensitivity (30 and 120 nearest segments)
RUN_P4 = True      # random-site control
N_RANDOM = 15
SEEDS3 = [0, 1, 2]
SEEDS2 = [0, 1]


def robust_row(label, res, diag, obs, prio):
    t = res[res.site.isin(diag)].groupby("site")["d_tstt_pct"].mean().mul(-1).reindex(diag)
    top = obs.nlargest(5).index
    r1 = stats.spearmanr(t, obs.reindex(t.index))
    r2 = stats.spearmanr(t, prio.reindex(t.index))
    mw = stats.mannwhitneyu(t[top], t.drop(top), alternative="greater")[1]
    return {"design": label, "rho_obs": r1[0], "p_obs": r1[1], "rho_prio": r2[0], "p_prio": r2[1],
            "top5_mean": t[top].mean(), "others_mean": t.drop(top).mean(), "mwu_p": mw,
            "overlap_top5": len(set(t.nlargest(5).index) & set(top)), "max_effect": t.max()}


def run_extra(G, kaggle_raw):
    global CAP_FACTOR
    t_start = time.time()
    D = build_digraph(G)
    print(f"Network (reachable core): {D.number_of_nodes()} nodes, {D.number_of_edges()} directed edges")
    snapped, kaggle = merge_sites(snap_locations(D, LOCATIONS), kaggle_raw)
    print(f"Distinct sites: {len(snapped)}")
    cen = centrality_table(D, snapped, kaggle)
    prio = cen["bottleneck_score"]
    diag = list(snapped)
    for _, _, d in D.edges(data=True):
        d["cap0"] = d["cap"]

    rows, allres = [], []

    def go(label, model, q, K, factor, seeds, extra=None):
        global CAP_FACTOR
        CAP_FACTOR = factor
        sn = dict(snapped)
        if extra:
            sn.update(extra)
        treated = nearest_k_edges(D, sn, K)
        t0 = time.time()
        res = run_scenarios(D, sn, treated, [(model, q)], seeds, progress=False)
        res["design"] = label
        allres.append(res)
        rows.append(robust_row(label, res, diag, kaggle, prio))
        r = rows[-1]
        print(f"[{(time.time() - t_start) / 60:5.1f} min] {label}: rho_obs={r['rho_obs']:+.2f} rho_prio={r['rho_prio']:+.2f} "
              f"top5={r['top5_mean']:.2f}% others={r['others_mean']:.2f}% max={r['max_effect']:.2f}%  ({time.time() - t0:.0f}s)", flush=True)
        return res

    # E1: completes the demand x design matrix
    go("E1 equal60 x1.5 center", "center_weighted", 60, 60, 1.5, SEEDS3)

    if RUN_P2:
        go("cap x1.25 equal60 uniform", "uniform", 60, 60, 1.25, SEEDS2)
        go("cap x2.00 equal60 uniform", "uniform", 60, 60, 2.0, SEEDS2)
    if RUN_P3:
        go("K=30 x1.5 uniform", "uniform", 60, 30, 1.5, SEEDS2)
        go("K=120 x1.5 uniform", "uniform", 60, 120, 1.5, SEEDS2)
    if RUN_P4:
        rng = np.random.default_rng(123)
        nodes = [n for n in D.nodes if n not in set(snapped.values())]
        pick = rng.choice(len(nodes), N_RANDOM, replace=False)
        extra = {f"rand_{i}": nodes[j] for i, j in enumerate(pick)}
        res = go("ref equal60 x1.5 uniform + random control", "uniform", 60, 60, 1.5, SEEDS2, extra)
        imp = res.groupby("site")["d_tstt_pct"].mean().mul(-1)
        top5 = kaggle.nlargest(5).index
        rnd = imp[[k for k in imp.index if k.startswith("rand_")]]
        print("\n=== Random-site control (equal60, x1.5, uniform) ===")
        print(f"diagnosed top-5 sites: mean {imp[top5].mean():.3f}%   all 15 diagnosed: mean {imp[diag].mean():.3f}%")
        print(f"{N_RANDOM} random nodes : mean {rnd.mean():.3f}%  median {rnd.median():.3f}%  90th pct {rnd.quantile(0.9):.3f}%  max {rnd.max():.3f}%")
        print(f"random sites at or above the diagnosed top-5 mean: {(rnd >= imp[top5].mean()).sum()} of {N_RANDOM}")
        print(f"random sites at or above the diagnosed all-15 mean: {(rnd >= imp[diag].mean()).sum()} of {N_RANDOM}")

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("\n=== Robustness table (15 diagnosed sites; improvement vs diagnosis) ===")
    print(out.round(3).to_string(index=False))
    pd.concat(allres).to_csv("results_plus_runs.csv", index=False)
    out.to_csv("results_plus_robustness.csv", index=False)
    print(f"\nTotal time: {(time.time() - t_start) / 60:.1f} min")
    print("Saved: results_plus_runs.csv, results_plus_robustness.csv")


def main():
    import osmnx as ox
    df = pd.read_csv(CSV_PATH)
    kaggle_raw = df.groupby("Road/Intersection Name")["Congestion Level"].mean()
    lats = [c[0] for c in LOCATIONS.values()]
    lons = [c[1] for c in LOCATIONS.values()]
    pad = 0.02
    flt = ('["highway"~"motorway|motorway_link|trunk|trunk_link|'
           'primary|primary_link|secondary|secondary_link"]')
    print("Downloading arterial network...")
    G = ox.graph_from_bbox((min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad),
                           custom_filter=flt, simplify=True)
    print(f"Downloaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    run_extra(G, kaggle_raw)


if __name__ == "__main__":
    main()
