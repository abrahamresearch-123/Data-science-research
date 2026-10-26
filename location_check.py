# ============================================================
# Location check: which network node does each of the 16
# locations actually land on, and is that plausible?
#
# Run in Colab (needs internet for the OSM download; ~2 minutes).
#   !pip install osmnx pandas==2.2.3 -q   -> then Runtime > Restart
# No CSV upload needed.
#
# It prints, for every location: the node it lands on, how far that
# node is from your point (metres), the road names at that node, a
# Google Maps link to check it, and flags for anything suspicious.
# ============================================================
import time
import numpy as np
import pandas as pd
import networkx as nx

GEOCODE = True       # also compare against Nominatim geocoding (set False to skip)
SUSPECT_M = 200      # flag if the nearest node is further than this from the point

# Same coordinates the simulations used (lat, lon).
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
CSV_PATH = "Banglore_traffic_Dataset.csv"   # not needed for this check


def dist_m(lat1, lon1, lat2, lon2):
    dx = (lon2 - lon1) * 111320 * np.cos(np.radians(lat1))
    dy = (lat2 - lat1) * 110570
    return float(np.hypot(dx, dy))


def nearest(G, nodes, lat, lon):
    best, bd = None, 1e18
    for n in nodes:
        d = dist_m(lat, lon, G.nodes[n]["y"], G.nodes[n]["x"])
        if d < bd:
            best, bd = n, d
    return best, bd


def road_names(G, n, k=4):
    names = []
    edges = list(G.out_edges(n, data=True)) + list(G.in_edges(n, data=True))
    for e in edges:
        nm = e[2].get("name")
        for x in (nm if isinstance(nm, list) else [nm]):
            if x and x not in names:
                names.append(x)
    return ", ".join(names[:k]) if names else "(unnamed)"


def check_locations(G, locations, geocoded=None):
    """G: osmnx-style MultiDiGraph with node attrs x (lon), y (lat)."""
    D = nx.DiGraph()
    D.add_nodes_from(G.nodes)
    D.add_edges_from((u, v) for u, v, _ in G.edges(keys=False, data=True))
    scc = max(nx.strongly_connected_components(D), key=len)
    all_nodes = list(G.nodes)
    scc_nodes = list(scc)
    rows = []
    for name, (lat, lon) in locations.items():
        n_all, d_all = nearest(G, all_nodes, lat, lon)
        n_scc, d_scc = nearest(G, scc_nodes, lat, lon)
        row = {"location": name, "node": n_scc, "dist_m": round(d_scc),
               "node_lat": round(G.nodes[n_scc]["y"], 5), "node_lon": round(G.nodes[n_scc]["x"], 5),
               "roads_at_node": road_names(G, n_scc),
               "moved_by_SCC": n_all != n_scc,
               "maps": f"https://www.google.com/maps?q={G.nodes[n_scc]['y']:.5f},{G.nodes[n_scc]['x']:.5f}"}
        if geocoded is not None:
            g = geocoded.get(name)
            row["geocode_vs_point_m"] = round(dist_m(lat, lon, g[0], g[1])) if g else None
        rows.append(row)
    out = pd.DataFrame(rows)
    dup = out.groupby("node")["location"].transform("count")
    out["shares_node_with"] = [
        ", ".join(out.loc[(out.node == nd) & (out.location != loc), "location"]) if c > 1 else ""
        for nd, loc, c in zip(out.node, out.location, dup)]
    flags = []
    for _, r in out.iterrows():
        f = []
        if r.dist_m > SUSPECT_M:
            f.append(f"far({r.dist_m}m)")
        if r.shares_node_with:
            f.append("SHARED NODE")
        if r.moved_by_SCC:
            f.append("moved by SCC")
        if geocoded is not None and r.get("geocode_vs_point_m") and r["geocode_vs_point_m"] > 500:
            f.append(f"geocode differs({r['geocode_vs_point_m']}m)")
        flags.append("; ".join(f))
    out["FLAGS"] = flags
    return out


def main():
    import osmnx as ox
    lats = [c[0] for c in LOCATIONS.values()]
    lons = [c[1] for c in LOCATIONS.values()]
    pad = 0.02
    flt = ('["highway"~"motorway|motorway_link|trunk|trunk_link|'
           'primary|primary_link|secondary|secondary_link"]')
    print("Downloading arterial network (same settings as the simulations)...")
    G = ox.graph_from_bbox((min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad),
                           custom_filter=flt, simplify=True)
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    geocoded = None
    if GEOCODE:
        geocoded = {}
        print("Geocoding each location with Nominatim for comparison...")
        for name in LOCATIONS:
            try:
                geocoded[name] = ox.geocode(f"{name}, Bengaluru, India")
            except Exception:
                geocoded[name] = None
            time.sleep(1.2)

    out = check_locations(G, LOCATIONS, geocoded)
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    show = ["location", "node", "dist_m", "roads_at_node", "shares_node_with", "FLAGS"]
    if GEOCODE:
        show.insert(3, "geocode_vs_point_m")
    print("\n=== Where each location lands ===")
    print(out[show].to_string(index=False))
    print("\n=== Flagged locations: open the link and check it is the right junction ===")
    bad = out[out.FLAGS != ""]
    if len(bad) == 0:
        print("none flagged")
    for _, r in bad.iterrows():
        print(f"- {r.location}: {r.FLAGS}\n    node ({r.node_lat}, {r.node_lon})  {r.maps}")
    out.to_csv("location_check.csv", index=False)
    print("\nSaved: location_check.csv")


if __name__ == "__main__":
    main()
