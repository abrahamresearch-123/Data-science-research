# Bengaluru bottleneck framework

Code and results for **"A Data-Driven Framework for Identifying and Evaluating Congestion Bottlenecks in Bengaluru under Sparse Traffic Data"**.

The framework combines the OpenStreetMap road network with traffic observations from 16 monitored locations (a public Kaggle dataset) to rank bottleneck candidates (betweenness centrality plus observed congestion), and tests how adding road capacity near each site changes simulated network travel time. It is a building block for work with richer data, not a validated planning tool.

## Files

| File | What it does | Run time | Output |
|---|---|---|---|
| `location_check.py` | Shows the network node each location is matched to, its distance, and flags | 2 min | `location_check.csv` |
| `full_run.py` | Centrality, priority score, baseline simulation, scenario tests | 25 min | `full_run_*.csv` (5 files) |
| `results_plus.py` | Robustness checks and random-site control | 35-40 min | `results_plus_*.csv` (2 files) |

All output CSVs are in `Results`.

## Run (Google Colab)

1. `!pip install osmnx pandas==2.2.3 -q`, then restart the runtime.
2. Download the dataset from Kaggle ("Bangalore Traffic Pulse") and upload `Banglore_traffic_Dataset.csv` to Colab. It is not included here.
3. Paste a script into a cell and run it.

Requirements: Python 3, `osmnx`, `networkx`, `pandas`, `numpy`, `scipy`, and internet access. Versions used: 2.1.1, 3.7, 2.2.3,


## Main settings

- Network: arterial roads only (motorway, trunk, primary, secondary and links); reachable core of about 2,294 nodes.
- Priority score: (N + 1 - centrality rank) + (N + 1 - congestion rank), with N = 15 sites.
- Simulation: delay t = t0 [1 + 0.15 (v/c)^4]; incremental assignment; 1,500 trips per run; generic road-class speeds and capacities; nothing is fitted to the Kaggle data.
- Scenarios: capacity x1.5 on segments within 1.5 km of a site, or on the 60 nearest segments; outcome is the change in total system travel time.
- Seeds are fixed (0, 1, 2; control nodes use 123).

## Notes

- Location coordinates are approximate (set manually in `LOCATIONS`) and matched to nodes 6-782 m away. Hebbal Flyover and Ballari Road share a node and are merged (15 sites).
- OpenStreetMap changes over time, so a fresh download may give slightly different numbers.
