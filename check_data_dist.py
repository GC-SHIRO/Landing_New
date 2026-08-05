#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 dynamic expert 数据分布:
- 各 motion_class 的 episode / transition 数量(均匀性)
- episode 长度分布
- success / terminal_reason 分布
- quality_accepted / quality_fatal 分布
- observation / action / reward 数值范围
- expert.phase 分布
"""
import json
import sys
import numpy as np

def load_episodes(path):
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            if isinstance(ep, dict) and "steps" in ep:
                ep = ep["steps"]
            episodes.append(ep)
    return episodes

def main(paths):
    for path in paths:
        print("=" * 78)
        print(f"FILE: {path}")
        eps = load_episodes(path)
        print(f"episodes: {len(eps)}")

        # transitions
        steps = [s for ep in eps for s in ep]
        print(f"transitions: {len(steps)}")

        # per-episode stats
        lens = np.array([len(ep) for ep in eps])
        print(f"episode len: min={lens.min()} max={lens.max()} mean={lens.mean():.1f} "
              f"median={np.median(lens):.0f}")

        # motion class distribution
        from collections import Counter
        mc = Counter()
        mc_steps = Counter()
        success_by_mc = Counter()
        succ_steps = Counter()
        for ep in eps:
            c = ep[0]["scenario"]["motion_class"] if ep else "?"
            mc[c] += 1
            mc_steps[c] += len(ep)
            ok = any(s.get("success", False) for s in ep)
            if ok:
                success_by_mc[c] += 1
                succ_steps[c] += len(ep)
        print("\n-- motion_class (episodes / steps / successful-episodes) --")
        for c in sorted(mc):
            print(f"  {c:20s} ep={mc[c]:4d}  steps={mc_steps[c]:7d}  succ_ep={success_by_mc.get(c,0):4d}")

        # terminal reason distribution
        tr = Counter()
        for ep in eps:
            for s in ep:
                if s.get("done", False):
                    tr[s.get("terminal_reason", "?")] += 1
        print("\n-- terminal_reason (done transitions) --")
        for k, v in tr.most_common():
            print(f"  {k:30s} {v}")

        # per-episode terminal outcome
        outcomes = Counter()
        for ep in eps:
            done = [s for s in ep if s.get("done", False)]
            if not done:
                outcomes["NO_DONE"] += 1
                continue
            r = done[-1].get("terminal_reason", "?")
            outcomes[r] += 1
        print("\n-- per-episode outcome --")
        for k, v in outcomes.most_common():
            print(f"  {k:30s} {v}")

        # quality
        qa = Counter(s.get("quality_accepted") for ep in eps for s in ep)
        qf = Counter(s.get("quality_fatal") for ep in eps for s in ep)
        print(f"\nquality_accepted: {dict(qa)}")
        print(f"quality_fatal:    {dict(qf)}")

        # expert phase
        ph = Counter(s["expert"]["phase"] for ep in eps for s in ep)
        print("\n-- expert.phase --")
        for k, v in ph.most_common():
            print(f"  {k:20s} {v}")

        # numeric ranges
        obs = np.array([s["observation"] for ep in eps for s in ep], dtype=np.float64)
        act = np.array([s["action"] for ep in eps for s in ep], dtype=np.float64)
        rws = np.array([s["reward"] for ep in eps for s in ep], dtype=np.float64)
        print(f"\nobs  shape={obs.shape} min={obs.min(axis=0)} max={obs.max(axis=0)}")
        print(f"      mean={obs.mean(axis=0)} std={obs.std(axis=0)}")
        print(f"act  shape={act.shape} min={act.min(axis=0)} max={act.max(axis=0)}")
        print(f"      mean={act.mean(axis=0)} std={act.std(axis=0)}")
        print(f"reward: min={rws.min():.4f} max={rws.max():.4f} mean={rws.mean():.4f} "
              f"std={rws.std():.4f}")

        # success reward breakdown
        succ_r = rws[steps_success_mask(steps)]
        fail_r = rws[~steps_success_mask(steps)]
        if len(succ_r):
            print(f"  success-step reward: n={len(succ_r)} mean={succ_r.mean():.2f} min={succ_r.min():.2f} max={succ_r.max():.2f}")
        print(f"  non-success reward: n={len(fail_r)} mean={fail_r.mean():.2f} min={fail_r.min():.2f} max={fail_r.max():.2f}")

        # near-touchdown (last 2 steps of each episode) terminal reward
        tail_r = [s["reward"] for ep in eps for s in ep[-2:]]
        tail_r = np.array(tail_r)
        print(f"  last-2-step reward: mean={tail_r.mean():.2f} min={tail_r.min():.2f} max={tail_r.max():.2f}")

        # per-step target speed profile (vx, vy range by motion class)
        print("\n-- scenario params sample --")
        for c in sorted(mc):
            ep0 = next(ep for ep in eps if ep and ep[0]["scenario"]["motion_class"] == c)
            sc = ep0[0]["scenario"]
            print(f"  {c:20s} vx={sc.get('vx'):+.4f} vy={sc.get('vy'):+.4f} "
                  f"radius={sc.get('radius')} period={sc.get('period')} "
                  f"amp={sc.get('amplitude')} wl={sc.get('wavelength')}")

def steps_success_mask(steps):
    return np.array([s.get("success", False) for s in steps])

if __name__ == "__main__":
    paths = sys.argv[1:] or ["expert_data_dynamic/random_dynamic_privileged_pd.jsonl",
                             "expert_data_dynamic/random_dynamic_privileged_pd_all.jsonl"]
    main(paths)
