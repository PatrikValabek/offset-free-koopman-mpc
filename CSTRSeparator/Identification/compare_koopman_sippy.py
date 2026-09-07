#!/usr/bin/env python3
"""Compare optimized Koopman C/noC models against SIPPY order sweep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

IDENT_DIR = Path(__file__).resolve().parent
DATA_DIR = IDENT_DIR / "../data"
RESULTS_DIR = IDENT_DIR / "../data/hpo_results"


def load_sippy_order_results() -> list[dict]:
    """Run SIPPY order sweep or load cached results if present."""
    cache_path = RESULTS_DIR / "sippy_order_results.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    import warnings

    import sippy_unipi.functionsetSIM as fsetSIM
    import sippy_unipi.OLSims_methods as olm
    from sippy_unipi import system_identification
    from sklearn.preprocessing import StandardScaler

    sys.path.insert(0, str(IDENT_DIR.parent.parent / "src"))
    from helper.koopman import load_cstr_separator_data

    def _Vn_mat_dot(y, yest):
        eps = np.asarray(y).reshape(-1) - np.asarray(yest).reshape(-1)
        return float(np.dot(eps, eps) / max(eps.size, 1))

    fsetSIM.Vn_mat = _Vn_mat_dot
    olm.Vn_mat = _Vn_mat_dot

    _, train, dev, test, y_names, _ = load_cstr_separator_data(
        str(DATA_DIR / "cstr_separator_ident.npz")
    )
    scaler = StandardScaler().fit(train["Y"])
    scalerU = StandardScaler().fit(train["U"])
    train_sim = {
        "Y": scaler.transform(train["Y"]),
        "U": scalerU.transform(train["U"]),
    }
    test_sim = {
        "Y": scaler.transform(test["Y"]),
        "U": scalerU.transform(test["U"]),
    }
    y_tot = train_sim["Y"].T.copy()
    U = train_sim["U"].T.copy()
    U_test = test_sim["U"].T
    Y_test = test_sim["Y"].T

    order_results = []
    SS_f = 40
    for n in range(4, 21):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=".*encountered in matmul",
            )
            ident = system_identification(
                y_tot.copy(),
                U.copy(),
                "N4SID",
                SS_fixed_order=int(n),
                SS_f=SS_f,
                SS_A_stability=True,
            )
        _, y_hat = fsetSIM.SS_lsim_process_form(
            ident.A,
            ident.B,
            ident.C,
            ident.D,
            U_test,
            np.linalg.pinv(ident.C) @ Y_test[:, :1],
        )
        err = Y_test - y_hat
        mae_std = np.mean(np.abs(err), axis=1)
        mae_phys = mae_std * scaler.scale_
        order_results.append(
            {
                "order": n,
                "mae_sum": float(np.sum(mae_phys)),
                "mae_mean": float(np.mean(mae_phys)),
                "mae_per_output": [float(v) for v in mae_phys],
                "rho": float(np.max(np.abs(np.linalg.eigvals(ident.A)))),
            }
        )
        print(f"SIPPY order {n:2d}: test MAE sum={order_results[-1]['mae_sum']:.6g}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(order_results, indent=2), encoding="utf-8")
    return order_results


def load_koopman_meta(variant: str) -> dict | None:
    meta_path = DATA_DIR / f"model_cstr_separator_{variant}_optimized_meta.json"
    if not meta_path.exists():
        alt = RESULTS_DIR / variant / "stage_C_final_meta.json"
        if alt.exists():
            return json.loads(alt.read_text(encoding="utf-8"))
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-sippy",
        action="store_true",
        help="Skip SIPPY sweep and use cached results only",
    )
    args = parser.parse_args()

    if args.skip_sippy:
        sippy = json.loads(
            (RESULTS_DIR / "sippy_order_results.json").read_text(encoding="utf-8")
        )
    else:
        sippy = load_sippy_order_results()

    meta_c = load_koopman_meta("C")
    meta_noc = load_koopman_meta("noC")

    comparison = {
        "sippy": sippy,
        "koopman_C": meta_c,
        "koopman_noC": meta_noc,
    }
    out_json = RESULTS_DIR / "comparison_summary.json"
    out_json.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out_json}")

    fig, ax = plt.subplots(figsize=(10, 6))
    orders = [r["order"] for r in sippy]
    mae_sippy = [r["mae_sum"] for r in sippy]
    ax.plot(orders, mae_sippy, "o-", label="SIPPY N4SID", linewidth=2)

    if meta_c:
        nz_c = meta_c["params"]["nz"]
        ax.scatter(
            [nz_c],
            [meta_c["test_mae_sum"]],
            s=120,
            marker="*",
            label=f"Koopman C (nz={nz_c})",
            zorder=5,
        )
    if meta_noc:
        nz_noc = meta_noc["params"]["nz"]
        ax.scatter(
            [nz_noc],
            [meta_noc["test_mae_sum"]],
            s=120,
            marker="D",
            label=f"Koopman noC (nz={nz_noc})",
            zorder=5,
        )

    ax.set_xlabel("State dimension (nz / SIPPY order)")
    ax.set_ylabel("Test MAE sum (physical units)")
    ax.set_title("Koopman C vs noC vs SIPPY (CSTR-Separator)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plot_path = RESULTS_DIR / "comparison_plot.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Wrote {plot_path}")

    print("\n=== Summary ===")
    best_sippy = min(sippy, key=lambda r: r["mae_sum"])
    print(
        f"Best SIPPY: order={best_sippy['order']}, test MAE sum={best_sippy['mae_sum']:.6g}"
    )
    if meta_c:
        print(
            f"Koopman C: nz={meta_c['params']['nz']}, test MAE sum={meta_c['test_mae_sum']:.6g}"
        )
    if meta_noc:
        print(
            f"Koopman noC: nz={meta_noc['params']['nz']}, test MAE sum={meta_noc['test_mae_sum']:.6g}"
        )


if __name__ == "__main__":
    main()
