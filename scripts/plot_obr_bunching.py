"""Reproduce OBR Chart 3.C: bunching in the VAT turnover distribution at the
registration threshold (EFO March 2023, "The impact of the frozen VAT
registration threshold"). Source data: OBR March 2023 EFO charts-and-tables
workbook, sheet C3.C (OBR processing of HMRC VAT trader microdata; Open
Government Licence v3.0). Counts are number of businesses (thousands) in
GBP 1,000 turnover bins.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_FIG = ROOT / "paper" / "figures" / "obr_vat_bunching_85k.png"
OUT_CSV = ROOT / "data" / "processed" / "obr_vat_bunching.csv"

# turnover (GBP), then businesses (thousands) by year. OBR sheet C3.C.
ROWS = [
    (65000, 17.538, 18.260, 18.706, 20.403, 20.028, 22.852),
    (66000, 16.519, 17.408, 17.943, 19.195, 18.536, 21.150),
    (67000, 15.580, 17.265, 17.475, 18.819, 18.687, 21.322),
    (68000, 15.507, 17.203, 17.608, 18.912, 18.767, 21.413),
    (69000, 17.328, 16.745, 17.283, 18.628, 18.200, 20.766),
    (70000, 14.947, 16.660, 17.353, 19.246, 18.323, 20.907),
    (71000, 14.981, 16.141, 16.626, 18.005, 17.304, 19.744),
    (72000, 14.554, 16.177, 16.981, 17.963, 17.741, 20.242),
    (73000, 14.556, 15.875, 16.322, 17.354, 16.713, 19.070),
    (74000, 15.881, 15.681, 16.258, 17.137, 16.938, 19.326),
    (75000, 14.215, 16.239, 16.413, 17.778, 17.274, 19.710),
    (76000, 15.655, 15.718, 15.875, 16.871, 16.710, 19.066),
    (77000, 14.740, 15.617, 15.802, 16.786, 16.619, 18.962),
    (78000, 15.587, 16.456, 16.350, 17.386, 17.475, 19.939),
    (79000, 20.435, 17.197, 16.520, 17.773, 17.602, 20.084),
    (80000, 15.204, 17.657, 16.848, 17.706, 17.725, 20.224),
    (81000, 10.228, 18.276, 16.987, 17.596, 17.718, 20.216),
    (82000, 9.496, 17.855, 17.130, 18.560, 18.692, 21.838),
    (83000, 9.046, 10.909, 16.579, 19.602, 20.189, 25.773),
    (84000, 9.210, 10.643, 16.569, 20.381, 22.073, 29.949),
    (85000, 9.176, 9.772, 10.001, 10.435, 10.375, 11.304),
    (86000, 8.497, 9.339, 9.441, 9.867, 9.779, 10.624),
    (87000, 8.350, 9.155, 9.316, 9.632, 9.533, 10.343),
    (88000, 8.535, 9.028, 9.247, 9.607, 9.416, 10.210),
    (89000, 8.474, 8.904, 9.128, 9.343, 9.229, 9.996),
    (90000, 9.039, 9.074, 9.035, 9.396, 9.512, 10.319),
]
COLS = ["turnover", "2014-15", "2016-17", "2017-18", "2018-19", "2019-20", "2025-26"]
df = pd.DataFrame(ROWS, columns=COLS)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)

TEAL = "#326b77"
ORANGE = "#c44e1a"

fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.plot(df["turnover"] / 1000, df["2019-20"], color=TEAL, lw=2,
        marker="o", ms=3, label="2019–20 (outturn)")
ax.plot(df["turnover"] / 1000, df["2025-26"], color=ORANGE, lw=2,
        ls="--", marker="s", ms=3, label="2025–26 (projection)")
ax.axvline(85, color="0.35", lw=1, ls=":")
ax.text(85, ax.get_ylim()[1] * 0.97, " Registration\n threshold (£85k)",
        ha="left", va="top", fontsize=9, color="0.3")

ax.set_xlabel("Annual turnover (£000s)")
ax.set_ylabel("Number of businesses (thousands)")
ax.set_xlim(65, 90)
ax.set_ylim(0, None)
ax.grid(True, ls="--", lw=0.5, alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
print("wrote", OUT_FIG)
print("wrote", OUT_CSV)
