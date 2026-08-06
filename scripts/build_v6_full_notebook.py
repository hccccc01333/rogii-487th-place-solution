# -*- coding: utf-8 -*-
"""Build v6 FULL kernel: live A2 + live 68 well features + main x soft merge.

Mounts: koolbox-offline, rogii-mainline-assets, full773-inference-assets.
Embeds live_soft_features.py and the A2 model bundle so no dataset update is
needed (the pending v5 submission keeps its pinned inputs).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "project/notebooks/rogii-main-soft-full-v6"
OUT.mkdir(parents=True, exist_ok=True)

LIVE_FEATURES_SRC = (ROOT / "project/src/alignment/live_soft_features.py").read_text(
    encoding="utf-8").replace("from alignment.ms_pf_align import", "from ms_pf_align import")
A2_BUNDLE_B64 = base64.b64encode(
    (ROOT / "project/artifacts/pf128_softgate/assets/a2_models.joblib").read_bytes()
).decode("ascii")

MAIN_CODE = r'''# -*- coding: utf-8 -*-
"""ROGII Main x Soft FULL v6: live A2 + live 68 features, hidden-safe."""
import base64
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

t_start = time.time()
W_MAIN, W_SOFT = 0.6, 0.4
MEDIAN_PREFIX_RMSE = 0.5945
WEIGHT_SCALE = 10.0
CLIP_LOW, CLIP_HIGH = 0.05, 0.9
TAU_C, TAU_A, K_BOOST = 0.5, 0.5, 0.0
LGB_PARAMS = dict(n_estimators=200, learning_rate=0.04, num_leaves=15, max_depth=4,
                  min_child_samples=10, subsample=0.85, colsample_bytree=0.8,
                  reg_alpha=1.0, reg_lambda=5.0, random_state=42, n_jobs=2,
                  deterministic=True, force_col_wise=True, verbosity=-1)
EXCLUDE = {
    "well_id", "well", "fold", "n", "ad", "cb", "l5", "a2", "sse_share", "a2_win_rate", "a2_win",
    "error_type", "tail_cohort", "spatial_block", "suffix_len", "late_error",
    "onset_fraction", "max_abs_error", "sse_early", "sse_mid", "sse_tail",
    "tail_n", "tail_err", "tail_bias", "tail_rmse", "tail_sse_share",
    "early_rmse", "mid_rmse", "adaptive_calib_rmse", "A2_P3_rmse",
    "contact_calib_rmse", "lik5_rmse", "w", "y_true", "feat_delta_md_ps", "order",
    "cb_calib", "lik_pf_scale_5", "A2_P3_tvt", "prefix_calib",
}
FEATURES = ["nbr1", "gap", "suffix_len", "gr_nan_suffix_frac",
            "tcstd_best_lso", "gr_mean_dist", "gr_std_dist"]
THR_GAP = 25.0

for _p in glob.glob("/kaggle/input/**/koolbox*.whl", recursive=True):
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", str(_p)], check=False)
    break


def _kaggle_input_roots():
    roots = []
    base = Path("/kaggle/input")
    if not base.exists():
        return roots
    for p in base.iterdir():
        if p.is_dir() and p.name not in {"datasets", "competitions"}:
            roots.append(p)
    comp = base / "competitions"
    if comp.exists():
        roots.extend(p for p in comp.iterdir() if p.is_dir())
    ds = base / "datasets"
    if ds.exists():
        for owner in ds.iterdir():
            if owner.is_dir():
                roots.extend(p for p in owner.iterdir() if p.is_dir())
    return roots


def _find_root(key_files):
    for c in [os.environ.get(k) for k in ("MAINLINE_ASSETS", "FULL773_ASSETS")]:
        pass
    for p in _kaggle_input_roots():
        if all((p / k).exists() for k in key_files):
            return p
    return None


def find_comp_root():
    for c in [os.environ.get("COMPETITION_DATA_ROOT"),
              "/kaggle/input/competitions/rogii-wellbore-geology-prediction",
              "/kaggle/input/rogii-wellbore-geology-prediction"]:
        if c and Path(c).exists():
            return Path(c)
    for p in _kaggle_input_roots():
        if (p / "train").exists() and (p / "test").exists():
            return p
    raise RuntimeError("competition data root not found")


def find_main_assets():
    for c in [os.environ.get("MAINLINE_ASSETS"),
              "/kaggle/input/datasets/afterrain42/rogii-mainline-assets",
              "/kaggle/input/rogii-mainline-assets"]:
        if c and Path(c).exists() and (Path(c) / "main_inference.py").exists():
            return Path(c)
    p = _find_root(["main_inference.py", "components_train.parquet",
                    "well_features_train.csv", "gate_params_v2.json"])
    if p is not None:
        return p
    raise RuntimeError("mainline-assets not found")


def find_full773_assets():
    for c in [os.environ.get("FULL773_ASSETS"),
              "/kaggle/input/datasets/afterrain42/full773-inference-assets",
              "/kaggle/input/full773-inference-assets"]:
        if c and Path(c).exists() and (Path(c) / "models").exists():
            return Path(c)
    for p in _kaggle_input_roots():
        if (p / "assets/fold_assignments.csv").exists() and (p / "models").exists():
            return p
    raise RuntimeError("full773-inference-assets not found")


comp_root = find_comp_root()
main_assets = find_main_assets()
full773 = find_full773_assets()
sys.path.insert(0, str(main_assets))
import main_inference as mi  # noqa: E402

print("competition root:", comp_root, flush=True)
print("main assets:", main_assets, flush=True)
print("full773 assets:", full773, flush=True)


def _read_test_pair(data_root, wid):
    hw = tw = None
    try:
        hw = pd.read_csv(data_root / "test" / f"{wid}__horizontal_well.csv").sort_values("MD").reset_index(drop=True)
    except Exception as e:
        print("HW_READ_FALLBACK", wid, repr(e)[:100], flush=True)
    try:
        tw = pd.read_csv(data_root / "test" / f"{wid}__typewell.csv").sort_values("TVT").reset_index(drop=True)
    except Exception as e:
        print("TW_READ_FALLBACK", wid, repr(e)[:100], flush=True)
    return hw, tw


def _suffix(hw):
    if hw is None or len(hw) == 0:
        return np.array([], dtype=int), 0.0
    if "TVT_input" not in hw.columns:
        idx = np.arange(len(hw), dtype=int)
        return idx, 0.0
    tvt = pd.to_numeric(hw["TVT_input"], errors="coerce").to_numpy(float)
    known = np.flatnonzero(np.isfinite(tvt))
    ev = np.flatnonzero(~np.isfinite(tvt))
    last_known = float(tvt[known[-1]]) if len(known) else 0.0
    if len(ev) == 0 and len(hw):
        ev = np.arange(len(hw), dtype=int)
    return ev, last_known


def _build_r1_feats(data_root, branch_cells_path, models):
    try:
        g = {"__name__": "__main__", "COMPETITION_DATA_ROOT": str(data_root),
             "RIDGE_ARTIFACT_ROOT": str(main_assets)}
        code = Path(branch_cells_path).read_text(encoding="utf-8")
        code = code.replace("NCPU=min(4,multiprocessing.cpu_count())", "NCPU=1")
        code = code.replace(
            "    pos=np.empty(N); rate=np.empty(N); w=np.ones(N)/N",
            "    np.random.seed(42)\n    pos=np.empty(N); rate=np.empty(N); w=np.ones(N)/N",
        )
        exec(compile(code, str(branch_cells_path), "exec"), g)
        _bw = g.get("build_well")
        if _bw is not None:
            def _safe_bw(hp, tp, it):
                try:
                    return _bw(hp, tp, it)
                except Exception as _e:
                    print("BW_FALLBACK", repr(_e)[:100], flush=True)
                    return None
            g["build_well"] = _safe_bw
        test_df = g["build_dataset"](sorted((data_root / "test").glob("*__horizontal_well.csv")),
                                     is_train=False, label="test")
        feats = [c for c in test_df.columns if c not in {"well", "id", "target"}]
        X = test_df[feats]
        r1_pred = test_df["last_known_tvt"].to_numpy(float) + np.mean(
            [m.predict(X) for m in models], axis=0)
        return {wid: r1_pred[test_df["well"] == wid] for wid in test_df["well"].unique()}, test_df
    except Exception as e:
        print("R1_BUILD_FALLBACK", repr(e)[:150], flush=True)
        return {}, pd.DataFrame()


def _pf_scales(hw, tw, np_=500, ns=128):
    tw2 = tw.sort_values("TVT").drop_duplicates("TVT")
    tw_tvt = tw2["TVT"].to_numpy(float)
    tw_gr = pd.to_numeric(tw2["GR"], errors="coerce").interpolate(limit_direction="both").fillna(
        float(np.nanmedian(tw2["GR"].to_numpy(float)))).to_numpy(float)
    out, _, q = mi.lik_pf(hw, tw_tvt, tw_gr, n_particles=np_, n_seeds=ns,
                          scales=(3.0, 5.0, 8.0, 12.0), with_quality=True)
    return out, dict(q or {})


test_wids = sorted(p.stem.replace("__horizontal_well", "")
                   for p in (comp_root / "test").glob("*__horizontal_well.csv"))
print("test wells:", len(test_wids), flush=True)

try:
    models = mi.load_models(main_assets)
except Exception as e:
    models = []
    print("MODEL_LOAD_FALLBACK", repr(e)[:100], flush=True)
try:
    contact = mi.build_contact(comp_root / "train")
except Exception as e:
    contact = None
    print("CONTACT_BUILD_FALLBACK", repr(e)[:100], flush=True)
try:
    train_stats = mi.load_train_typewell_stats(comp_root)
except Exception as e:
    train_stats = None
    print("TRAIN_STATS_FALLBACK", repr(e)[:100], flush=True)
try:
    params = json.loads((main_assets / "gate_params_v2.json").read_text(encoding="utf-8"))
    coef = np.array(params["ridge_coef"], float)
    intercept = float(params["ridge_intercept"])
    default_thr = float(np.median(list(params["block_thresholds"].values())))
except Exception as e:
    coef = np.zeros(len(FEATURES))
    intercept = 0.0
    default_thr = 0.0
    print("GATE_PARAMS_FALLBACK", repr(e)[:100], flush=True)

r1_map, test_df_all = _build_r1_feats(comp_root, main_assets / "branch1_cells.py", models)
print("r1 wells:", len(r1_map), flush=True)


# ---------------- A2 live (full773 panel pipeline) ----------------
_A2_B64 = "__A2_B64_MARKER__"
_a2_path = Path("/kaggle/working/a2_models.joblib") if Path("/kaggle/working").exists() else Path("a2_models.joblib")
_a2_path.write_bytes(base64.b64decode(_A2_B64))
import joblib as _jb
_a2_bundle = _jb.load(_a2_path)


def _a2_setup():
    src = full773 / "src"
    scripts = full773 / "scripts"
    sys.path.insert(0, str(src))
    sys.path.insert(0, str(scripts))
    sys.path.insert(0, str(full773))
    import run_geo_parametric_state_v1 as frozen
    return frozen


def build_a2_rows():
    frozen = _a2_setup()
    os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"
    import lb_proxy.data as _lpd
    _orig_wellcache = _lpd.WellCache.__init__
    def _patched_wellcache(self, data_dir, cache_dir=None):
        _orig_wellcache(self, data_dir, cache_dir=Path("/tmp/well_cache"))
    _lpd.WellCache.__init__ = _patched_wellcache
    from d3.honest import HonestD3Baseline
    from data.wells import load_well_pair
    from features.builder import FeatureBuilder
    from self_gr.inference import DEFAULT_REGIME_Q
    fold_table = pd.read_csv(full773 / "assets/fold_assignments.csv")
    valid = fold_table[fold_table["is_val"] == 1]
    fold_map = dict(zip(valid["original_well_id"].astype(str), valid["fold_id"].astype(int)))
    fold_assets = _jb.load(full773 / "assets/baseline_assets_fitted.joblib")
    baseline = HonestD3Baseline(comp_root)
    rows = []
    a2_cols = _a2_bundle["cols"]
    medians = _a2_bundle["medians"]
    for wid in test_wids:
        try:
            hw, tw = load_well_pair(wid, split="test", data_dir=comp_root)
            fold = int(fold_map.get(wid, 0))
            assets = fold_assets[fold]
            known = np.flatnonzero(pd.to_numeric(hw["TVT_input"], errors="coerce").notna().to_numpy())
            ev = np.flatnonzero(pd.to_numeric(hw["TVT_input"], errors="coerce").isna().to_numpy())
            d3 = baseline.predict(hw.drop(columns=["TVT"], errors="ignore"), tw, wid, assets)
            part = FeatureBuilder(assets.train_summary, DEFAULT_REGIME_Q).build(
                hw.drop(columns=["TVT"], errors="ignore"), tw, wid, known, ev, d3)
            part["row_idx"] = ev
            part["panel_id"] = wid
            part["original_well_id"] = wid
            part["fold_id"] = -1
            part["baseline_tvt"] = np.asarray(d3.final_tvt, float)[ev]
            part["selector_noproj"] = np.asarray(d3.selector_noproj_path, float)[ev]
            for k in ("pf3", "pf5", "pf8", "pf12"):
                part[k] = np.asarray(d3.pf_paths[k.replace("pf", "pf_scale_")], float)[ev]
            rows.append(part)
        except Exception as e:
            print("A2_WELL_FALLBACK", wid, repr(e)[:150], flush=True)
    if not rows:
        return {}
    raw = pd.concat(rows, ignore_index=True)
    try:
        panel, _ = frozen.build_panel_table(raw, include_targets=False)
        for c in a2_cols:
            if c not in panel.columns:
                panel[c] = float(medians.get(c, 0.0))
        x = panel[a2_cols].replace([np.inf, -np.inf], np.nan).fillna(pd.Series(medians)).to_numpy(np.float32)
        merged = raw[["row_idx", "panel_id", "original_well_id", "fold_id",
                      "baseline_tvt", "feat_delta_md_ps"]].merge(
            panel[["panel_id", "original_well_id", "fold_id", *a2_cols]],
            on=["panel_id", "original_well_id", "fold_id"], validate="many_to_one")
        m_hat = _a2_bundle["models"]["m"].predict(x)
        s_hat = _a2_bundle["models"]["s"].predict(x)
        pm = pd.DataFrame({"panel_id": panel["panel_id"], "m": m_hat, "s": s_hat})
        merged = merged.merge(pm, on="panel_id", how="left")
        ulen = merged.groupby("panel_id")["feat_delta_md_ps"].transform("max").to_numpy(float)
        uu = np.clip(merged["feat_delta_md_ps"].to_numpy(float) / np.maximum(ulen, 1e-9), 0, 1)
        out = pd.DataFrame({"well_id": merged["original_well_id"].astype(str),
                            "row_idx": merged["row_idx"].astype(int),
                            "a2": merged["baseline_tvt"].to_numpy(float)
                                  + merged["m"].to_numpy(float)
                                  + merged["s"].to_numpy(float) * (uu - 0.5)})
        return {wid: g[["row_idx", "a2"]].set_index("row_idx")["a2"].to_dict()
                for wid, g in out.groupby("well_id")}
    except Exception as e:
        print("A2_PANEL_FALLBACK", repr(e)[:200], flush=True)
        return {}


a2_map = build_a2_rows()
print("a2 wells:", len(a2_map), flush=True)


# ---------------- live soft features (embedded module) ----------------
exec(compile(__LIVE_FEATURES_MARKER__, "<live_soft_features>", "exec"))


# ---------------- main + soft per well ----------------
main_rows = []
soft_parts = []
feat_rows = []
likpf_rows = []
for i, wid in enumerate(test_wids):
    if i and i % 5 == 0:
        print("progress", i, "/", len(test_wids), "runtime_s", round(time.time() - t_start, 1), flush=True)
    hw, tw = _read_test_pair(comp_root, wid)
    ev, last_known = _suffix(hw)
    n = len(ev)
    if n == 0:
        continue

    # main
    r1_arr = r1_map.get(wid)
    if r1_arr is None or len(r1_arr) != n:
        r1_arr = np.full(n, last_known, np.float32)
    base = np.asarray(r1_arr, np.float32).copy()
    lik5 = np.full(n, last_known, np.float32)
    _q_new = None
    try:
        if hw is not None and tw is not None and len(tw) >= 3 and contact is not None and train_stats is not None:
            contact_path, gap_sel = mi.predict_legal(contact, hw, wid, train_stats)
            _pf_out, _q_new = _pf_scales(hw, tw)
            lik5 = np.asarray(_pf_out["pf_scale_5"], np.float32)
            if len(lik5) != n:
                lik5 = np.interp(np.arange(n), np.linspace(0, n - 1, len(lik5)), lik5).astype(np.float32)
            likpf_rows.append((wid, np.asarray(ev, int), _pf_out))
            if len(lik5) != n:
                lik5 = np.interp(np.arange(n), np.linspace(0, n - 1, len(lik5)), lik5).astype(np.float32)
            if len(contact_path) != n:
                contact_path = np.interp(np.arange(n), np.linspace(0, n - 1, len(contact_path)), contact_path)
            use_legal = bool(np.isfinite(gap_sel)) and gap_sel <= THR_GAP
            legal = np.where(use_legal, 0.5 * contact_path + 0.5 * lik5, lik5).astype(np.float32)
            ens = (0.5 * np.asarray(r1_arr, np.float32) + 0.5 * legal).astype(np.float32)
            f = mi.sse_features(wid, hw, tw, gap_sel, train_stats)
            Xf = np.array([float(f[col]) for col in FEATURES], dtype=float)
            benefit = float(coef @ Xf + intercept)
            nbr1 = float(f["nbr1"])
            if nbr1 < 1500:
                mode = "ens"
            elif nbr1 < 6000:
                mode = "legal" if benefit > default_thr else "ens"
            else:
                mode = "R1" if (float(f["suffix_len"]) > 4000 or float(f["gr_nan_suffix_frac"]) > 0.20) else "ens"
            base = {"R1": np.asarray(r1_arr, np.float32), "legal": legal, "ens": ens}[mode]
    except Exception as e:
        print("MAIN_WELL_FALLBACK", wid, repr(e)[:120], flush=True)
    for i2, ridx in enumerate(ev):
        main_rows.append((wid, int(ridx), float(base[i2] if i2 < len(base) else last_known)))

    # soft components
    cb = np.full(n, last_known, np.float32)
    prefix_calib = MEDIAN_PREFIX_RMSE
    try:
        if hw is not None and tw is not None and len(tw) >= 3 and contact is not None:
            tvt = pd.to_numeric(hw["TVT_input"], errors="coerce").to_numpy(float) if "TVT_input" in hw.columns else np.full(len(hw), np.nan)
            known_idx = np.flatnonzero(np.isfinite(tvt))
            ev_idx = np.asarray(ev, int)
            if len(known_idx) >= 20:
                paths_c, rmse_c = contact.predict_well(hw, wid, known_idx, ev_idx,
                                                       row_stride=30, tc_calibrate=True, hidden_mode=True)
                if paths_c and rmse_c:
                    b = min(rmse_c, key=rmse_c.get)
                    cb = paths_c[b].astype(np.float32)
                    if len(cb) != n:
                        cb = np.interp(np.arange(n), np.linspace(0, n - 1, len(cb)), cb).astype(np.float32)
                    prefix_calib = float(rmse_c[b])
    except Exception as e:
        print("SOFT_COMP_FALLBACK", wid, repr(e)[:120], flush=True)

    a2_arr = None
    if wid in a2_map:
        d = a2_map[wid]
        a2_arr = np.array([d.get(int(r), np.nan) for r in ev], float)
        if np.isfinite(a2_arr).mean() < 0.5:
            a2_arr = None
    if a2_arr is None:
        a2_arr = 0.5 * cb + 0.5 * lik5
    a2_arr = np.asarray(a2_arr, np.float32)
    for i2, ridx in enumerate(ev):
        soft_parts.append((wid, int(ridx), float(cb[i2]), float(lik5[i2]),
                           float(a2_arr[i2]), float(prefix_calib)))

    # live well features (gate inputs)
    try:
        _, pf_old = _pf_scales(hw, tw, 300, 64)
        frow = assemble_well_features(wid, hw, tw, contact, cb, lik5, a2_arr,
                                      prefix_calib, n, pf_new=_q_new, pf_old=pf_old)
        feat_rows.append(frow)
    except Exception as e:
        print("FEAT_FALLBACK", wid, repr(e)[:150], flush=True)

main_df = pd.DataFrame(main_rows, columns=["well_id", "row_idx", "tvt_main"])
print("main rows:", len(main_df), "wells:", main_df.well_id.nunique(), flush=True)
soft_df = pd.DataFrame(soft_parts, columns=["well_id", "row_idx", "cb", "lik5", "a2", "prefix_calib"])
print("soft rows:", len(soft_df), "wells:", soft_df.well_id.nunique(), flush=True)
feat_live = pd.DataFrame(feat_rows)
print("feat rows:", len(feat_live), "cols:", feat_live.shape[1], flush=True)

# ---------------- pred_late / pred_easy (retrained on bundled train) ----------------
well_tr = pd.read_csv(main_assets / "well_features_train.csv")
Xtr_all, feat_cols_all = _feature_matrix(well_tr)
_clf_l = lgb.LGBMClassifier(**LGB_PARAMS)
_clf_l.fit(Xtr_all, (well_tr["error_type"] == "D_late_onset").astype(int))
_clf_e = lgb.LGBMClassifier(**LGB_PARAMS)
_clf_e.fit(Xtr_all, (well_tr["error_type"] == "E_easy_stable").astype(int))
if len(feat_live):
    Xte = _feature_matrix(feat_live)[0]
    Xte = Xte.reindex(columns=feat_cols_all).fillna(well_tr[feat_cols_all].median())
    feat_live["pred_late"] = _clf_l.predict_proba(Xte)[:, 1]
    feat_live["pred_easy"] = _clf_e.predict_proba(Xte)[:, 1]
else:
    feat_live["pred_late"] = 0.5
    feat_live["pred_easy"] = 0.5
print("pheno preds done", flush=True)


# ---------------- gate + merge ----------------
comp_train = pd.read_parquet(main_assets / "components_train.parquet")
gate = train_gate(comp_train, well_tr)
comp_test = soft_df.copy()
well_te = feat_live.copy()
soft_pred = predict_gate(comp_test, well_te, gate)
soft_out = soft_pred[["well_id", "row_idx", "pred"]].rename(columns={"pred": "tvt_soft"})
print("gate done; soft pred rows:", len(soft_out), flush=True)

merged = main_df.merge(soft_out, on=["well_id", "row_idx"], how="left")
missing = int(merged["tvt_soft"].isna().sum())
if missing:
    print("soft missing rows (fallback to main):", missing, flush=True)
merged["tvt_soft"] = merged["tvt_soft"].fillna(merged["tvt_main"])
merged["tvt"] = W_MAIN * merged["tvt_main"] + W_SOFT * merged["tvt_soft"]
merged["id"] = merged["well_id"].astype(str) + "_" + merged["row_idx"].astype(int).astype(str)
merged = merged.sort_values(["well_id", "row_idx"]).reset_index(drop=True)

for sp in comp_root.rglob("sample_submission.csv"):
    try:
        sample = pd.read_csv(sp)
        sm, ours = set(sample["id"].astype(str)), set(merged["id"].astype(str))
        print("sample rows:", len(sm), "| missing:", len(sm - ours), "| extra:", len(ours - sm), flush=True)
        break
    except Exception:
        pass

fallback = float(merged["tvt"].median()) if merged["tvt"].notna().any() else 0.0
merged["tvt"] = merged["tvt"].fillna(fallback)

# ---------------- S3 learned branch (booster-only, legal) ----------------
def _find_learned_models():
    _cands = [os.environ.get("LEARNED_MODEL_DIR"),
              "/kaggle/input/datasets/fleongg/rogii-claude-models-pub",
              "/kaggle/input/fleongg/rogii-claude-models-pub"]
    for c in [x for x in _cands if x]:
        p = Path(c)
        if p.exists() and (p / "features.json").exists() and list(p.glob("lgb*.pkl")):
            return p
    for p in _kaggle_input_roots():
        if (p / "features.json").exists() and list(p.glob("lgb*.pkl")):
            return p
    return None

_learned_dir = _find_learned_models()
_learned_feats = []
_learned_models = []
if _learned_dir is not None:
    _learned_feats = json.load(open(_learned_dir / "features.json", encoding="utf-8"))
    _learned_models = [_jb.load(p) for p in sorted(_learned_dir.glob("lgb*.pkl"))]
print("learned models:", len(_learned_models), "feats:", len(_learned_feats), flush=True)

if len(_learned_models) and len(test_df_all) and len(likpf_rows):
    _lk = []
    for _wid, _ev, _out in likpf_rows:
        _lk.append(pd.DataFrame({
            "id": [f"{_wid}_{int(r)}" for r in _ev],
            "likpf_scale_3": np.asarray(_out["pf_scale_3"], np.float32),
            "likpf_scale_5": np.asarray(_out["pf_scale_5"], np.float32),
            "likpf_scale_8": np.asarray(_out["pf_scale_8"], np.float32),
            "likpf_scale_12": np.asarray(_out["pf_scale_12"], np.float32),
            "likpf_mean": np.asarray(_out["pf_mean"], np.float32),
        }))
    _lkdf = pd.concat(_lk, ignore_index=True)
    _td = test_df_all.reset_index(drop=True).merge(_lkdf, on="id", how="left")
    _lk_cols = ["likpf_scale_3", "likpf_scale_5", "likpf_scale_8", "likpf_scale_12", "likpf_mean"]
    for _c in _lk_cols:
        _td[_c] = _td[_c].fillna(_td["last_known_tvt"])
        _td[_c + "_d"] = (_td[_c] - _td["last_known_tvt"]).astype(np.float32)
    for _c in _learned_feats:
        if _c not in _td.columns:
            _td[_c] = 0.0
    _Xl = _td[_learned_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    _model_delta = np.mean([_m.predict(_Xl) for _m in _learned_models], axis=0)
    _last = _td["last_known_tvt"].to_numpy(float)
    _md_since = _td["md_since"].to_numpy(float)
    _lik5f = _td["likpf_scale_5"].to_numpy(float)
    _sub1 = (1.0 - np.exp(-np.maximum(_md_since, 0.0) / 85.0)) * _model_delta
    _learned_raw = _last + 0.60 * _sub1 + 0.40 * (_lik5f - _last)
    from scipy.signal import savgol_filter as _sg
    _learned_out = _learned_raw.copy()
    _td_well = _td["well"].to_numpy() if "well" in _td.columns else _td["id"].astype(str).str[:8].to_numpy()
    for _wid in pd.unique(_td_well):
        _idx = np.flatnonzero(_td_well == _wid)
        _v = _learned_raw[_idx]
        _wl = min(61, len(_v))
        if _wl % 2 == 0:
            _wl -= 1
        if _wl >= 5:
            _learned_out[_idx] = _sg(_v, _wl, 3)
    _learned_map = dict(zip(_td["id"].astype(str), _learned_out))
    merged["learned"] = merged["id"].map(_learned_map)
    _learned_missing = int(merged["learned"].isna().sum())
    if _learned_missing:
        print("learned missing rows (fallback to merge):", _learned_missing, flush=True)
    merged["learned"] = merged["learned"].fillna(merged["tvt"])
    _W_LEARNED = 0.80
    merged["tvt"] = _W_LEARNED * merged["learned"] + (1.0 - _W_LEARNED) * merged["tvt"]
    print("learned branch blended; rows:", len(merged), flush=True)
else:
    print("LEARNED_BRANCH_UNAVAILABLE: keeping v6 merge", flush=True)

# ---------------- full-coverage guard: expected ids from test files ----------------
_exp_rows = []
for _wid in test_wids:
    try:
        _ehw = pd.read_csv(comp_root / "test" / f"{_wid}__horizontal_well.csv")
        if "TVT_input" in _ehw.columns:
            _ev = np.flatnonzero(pd.to_numeric(_ehw["TVT_input"], errors="coerce").isna().to_numpy())
        else:
            _ev = np.arange(len(_ehw))
    except Exception:
        _ev = np.array([], dtype=int)
    for _r in _ev:
        _exp_rows.append((_wid, int(_r)))
_exp = pd.DataFrame(_exp_rows, columns=["well_id", "row_idx"])
_exp["id"] = _exp["well_id"].astype(str) + "_" + _exp["row_idx"].astype(int).astype(str)
_have_ids = set(merged["id"].astype(str))
_missing_ids = [i for i in _exp["id"] if i not in _have_ids]
if _missing_ids:
    _fill = float(merged["tvt"].median()) if merged["tvt"].notna().any() else 0.0
    _miss = _exp[_exp["id"].isin(_missing_ids)].copy()
    _miss["tvt"] = _fill
    merged = pd.concat([merged, _miss], ignore_index=True)
    print("COVERAGE_FILL missing rows:", len(_missing_ids), flush=True)
merged = merged.sort_values(["well_id", "row_idx"]).reset_index(drop=True)

assert merged["id"].duplicated().sum() == 0
assert np.isfinite(merged["tvt"].to_numpy(float)).all()

sub = merged[["id", "tvt"]]
out_csv = Path("/kaggle/working/submission.csv") if Path("/kaggle/working").exists() else Path("submission.csv")
sub.to_csv(out_csv, index=False)
sha = hashlib.sha256(out_csv.read_bytes()).hexdigest()
print("rows:", len(sub), "| wells:", merged.well_id.nunique(),
      "| finite:", bool(np.isfinite(sub["tvt"].to_numpy(float)).all()),
      "| sha256:", sha, flush=True)
print("runtime_s:", round(time.time() - t_start, 1), flush=True)
'''

INTRO = """# ROGII Main x Soft FULL v6（完整实时版）

- main：R1 + legal contact + PF128 + SSE gate（实时）
- soft：**真实 A2**（full773 panel 实时构建）+ **真实 68 井特征**（raw/pf128/tail/late/div 实时计算）
- gate：bundled components_train/well_features_train 现场训练；pred_late/pred_easy 现场重训
- merge：0.6 main + 0.4 soft；ID 从 test 文件生成；逐井 try/except；NaN 兜底
- **S3 learned branch**（6.440 合法组件，booster-only）：fleongg 3×LGB + PP 公式
  （alpha=1, tau=85, w_sub1=0.60, likpf 多尺度 3/5/8/12 + SG 61/3），
  与 v6 merge 按 0.80/0.20 混合；无预计算回退、无同井副本。
"""

NOTES = """## 说明

- 数据源：`phongnguyn23021656/koolbox-offline`、`afterrain42/rogii-mainline-assets`、
  `afterrain42/full773-inference-assets`、`fleongg/rogii-claude-models-pub`。
- A2 模型与 live 特征代码内嵌，不修改任何既有数据集版本。
- S3 仅取 6.440 的 booster 推理 + PP 公式，已去掉预计算回退；OOF 6.83。
- 目标：恢复完整 merge OOF ≈ 7.95–7.96 的部署等价实现。
"""


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def main() -> None:
    main_code = (MAIN_CODE
                 .replace("__A2_B64_MARKER__", A2_BUNDLE_B64)
                 .replace("__LIVE_FEATURES_MARKER__", repr(LIVE_FEATURES_SRC)))
    # live features module needs the gate helpers (train_gate/predict_gate/_feature_matrix)
    pipe = (ROOT / "project/src/alignment/pf128_softgate_pipeline.py").read_text(encoding="utf-8")
    # strip the docstring to avoid duplicate '"""' inside the exec source
    pipe_body = pipe.split('"""', 2)[2] if pipe.startswith('"""') else pipe
    pipe_body = pipe_body.replace("from __future__ import annotations\n", "")
    # strip dead code: build_submission is never called and reads components_test
    _dead = pipe_body.find("def build_submission")
    if _dead >= 0:
        pipe_body = pipe_body[:_dead]
    setup_code = (
        "import numpy as np\nimport pandas as pd\nimport lightgbm as lgb\n"
        + pipe_body
        + "\n\n" + main_code
    )
    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "cells": [
            md(INTRO),
            code("import sys, os\nimport numpy as np\nimport pandas as pd\nfrom pathlib import Path\n"),
            code(setup_code),
            md(NOTES),
        ],
    }
    (OUT / "rogii-main-soft-full-v6.ipynb").write_text(
        json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    meta = {
        "id": "afterrain42/rogii-main-x-soft-full-v6",
        "title": "ROGII Main x Soft FULL v6",
        "code_file": "rogii-main-soft-full-v6.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [
            "phongnguyn23021656/koolbox-offline",
            "afterrain42/rogii-mainline-assets",
            "afterrain42/full773-inference-assets",
            "fleongg/rogii-claude-models-pub",
        ],
        "kernel_sources": [],
        "competition_sources": ["rogii-wellbore-geology-prediction"],
        "model_sources": [],
    }
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", OUT, "main_code len", len(main_code))


if __name__ == "__main__":
    main()
