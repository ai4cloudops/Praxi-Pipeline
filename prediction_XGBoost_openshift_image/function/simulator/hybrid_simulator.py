import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------------------------------------------------------
# Bin-packing helper
# -----------------------------------------------------------------------------
def pack_loads(loads, min_capacity, max_capacity):
    """
    Bin-pack a list of workloads into bins whose sums lie in
      [min_capacity, max_capacity]. Workloads > max_capacity
    are first split into max_capacity‐sized chunks + a remainder.
    Uses First-Fit Decreasing, then merges small bins, then
    repairs underfilled bins by moving small items.
    """
    # ── 0) Pre-split oversized workloads ─────────────────────
    split_items = []
    for w in loads:
        if w <= max_capacity:
            split_items.append(w)
        else:
            # break into q full bins + remainder
            q, r = divmod(w, max_capacity)
            split_items.extend([max_capacity] * q)
            if r > 0:
                split_items.append(r)

    # ── 1) First-Fit Decreasing to respect max_capacity ──────
    items = sorted(split_items, reverse=True)
    bins = []  # each entry: (current_sum, [item, item, ...])
    for w in items:
        # find the bin that would be fullest after adding w
        best_idx, best_after = None, None
        for i, (s, lst) in enumerate(bins):
            if s + w <= max_capacity:
                cand = s + w
                if best_after is None or cand > best_after:
                    best_after, best_idx = cand, i
        if best_idx is None:
            bins.append((w, [w]))
        else:
            s, lst = bins[best_idx]
            lst.append(w)
            bins[best_idx] = (s + w, lst)

    # ── 2) Merge two smallest bins if they still fit under max_capacity ─
    merged = True
    while merged and len(bins) > 1:
        bins.sort(key=lambda x: x[0])
        merged = False
        # try merging the two smallest
        if bins[0][0] + bins[1][0] <= max_capacity:
            s0, l0 = bins.pop(0)
            s1, l1 = bins.pop(0)
            bins.append((s0 + s1, l0 + l1))
            merged = True

    # ── 3) Repair any bin below min_capacity by "stealing" a small item ─
    for i in range(len(bins)):
        s_i, lst_i = bins[i]
        if s_i >= min_capacity:
            continue
        needed = min_capacity - s_i
        candidates = []
        for j in range(len(bins)):
            if j == i:
                continue
            s_j, lst_j = bins[j]
            # pick the smallest w ≤ needed such that donor stays ≥ min_capacity
            for w in sorted(lst_j):
                if w <= needed and s_j - w >= min_capacity:
                    candidates.append((w, j))
                    break
        if not candidates:
            continue
        w_move, donor = min(candidates, key=lambda x: x[0])
        # move w_move from donor→i
        bins[donor][1].remove(w_move)
        bins[donor] = (bins[donor][0] - w_move, bins[donor][1])
        lst_i.append(w_move)
        bins[i] = (s_i + w_move, lst_i)

    # return just the lists of items
    return [lst for (_s, lst) in bins]


# -----------------------------------------------------------------------------
# Simulation parameters
# -----------------------------------------------------------------------------
NUM_MODELS        = 1000
EPOCHS            = 50
PROVISION_DELAY   = 2        # epochs until a new VM becomes healthy
T_CIP             = 43       # minimal efficient VM load
IAAS_CAPACITY     = 200_000  # max samples/VM per epoch

VM_COST_PER_HOUR  = 0.085    # $/hour per VM
EPOCH_SLA_SEC     = 900      # seconds per epoch
SAMPLE_LATENCY    = 0.005    # seconds per sample

# FaaS pricing
FAAS_MEMORY_GB     = 8.845
FAAS_GBSEC_COST    = 0.000016667
FAAS_REQ_COST      = 0.20 / 1e6

# EWMA prediction parameters
W                 = 0.3
BETA              = -50

SEED              = 42
np.random.seed(SEED)

# -----------------------------------------------------------------------------
# Generate synthetic loads
# -----------------------------------------------------------------------------
load_trace = [
    np.maximum(0,
        np.random.normal(loc=500, scale=100000, size=NUM_MODELS).astype(int)
    )
    for _ in range(EPOCHS)
]

# -----------------------------------------------------------------------------
# State initialization
# -----------------------------------------------------------------------------
ewma_avg     = np.zeros(NUM_MODELS)
ewma_std     = np.zeros(NUM_MODELS)
healthy_vms  = 0
provision_q  = []  # list of (ready_epoch, count)
epoch_hours  = EPOCH_SLA_SEC / 3600.0

records = []

# -----------------------------------------------------------------------------
# Main simulation loop
# -----------------------------------------------------------------------------
for epoch, loads in enumerate(load_trace, start=1):
    # 1) per-model EWMA & STD update
    dev = np.abs(loads - ewma_avg)
    ewma_avg = (1 - W) * ewma_avg + W * loads
    ewma_std = (1 - W) * ewma_std + W * dev

    # 2) build predicted workloads
    predicted = np.maximum(0, ewma_avg + BETA * ewma_std).astype(int).tolist()

    # 3) decide desired VMs via packing on predicted
    pred_bins   = pack_loads(predicted, T_CIP, IAAS_CAPACITY)
    vms_desired = len(pred_bins)

    # 4) update healthy_vms from provisioning queue
    newly = sum(cnt for ready_ep, cnt in provision_q if ready_ep == epoch)
    healthy_vms += newly
    provision_q = [(r, c) for r, c in provision_q if r > epoch]

    # 5) scale up or down
    if vms_desired > healthy_vms:
        to_prov = vms_desired - healthy_vms
        provision_q.append((epoch + PROVISION_DELAY, to_prov))
    elif vms_desired < healthy_vms:
        healthy_vms = vms_desired

    # 6) pack actual loads into batches
    actual_bins = pack_loads(loads.tolist(), T_CIP, IAAS_CAPACITY)
    vm_bins     = [b for b in actual_bins if sum(b) >= T_CIP]
    small_bins  = [b for b in actual_bins if sum(b) < T_CIP]

    # 7) allocate to VMs, letting idle VMs pick up small_bins
    served = []
    # first fill with all vm_bins
    for b in vm_bins:
        if len(served) < healthy_vms:
            served.append(b)
        else:
            break
    # then if we still have VM slots, fill with small bins
    idle_slots = healthy_vms - len(served)
    if idle_slots > 0:
        # take up to idle_slots from small_bins
        extras = small_bins[:idle_slots]
        served.extend(extras)
        small_bins = small_bins[idle_slots:]

    # anything not served goes to faas
    faas = small_bins + vm_bins[len(served):]

    # 8) cost calculation
    vm_cost = healthy_vms * VM_COST_PER_HOUR * epoch_hours

    # (optional) FaaS cost if you need
    faas_samples    = sum(sum(b) for b in faas)
    faas_duration   = faas_samples * SAMPLE_LATENCY
    faas_gbsec_cost = faas_duration * FAAS_MEMORY_GB * FAAS_GBSEC_COST
    faas_req_cost   = len(faas) * FAAS_REQ_COST
    faas_cost       = faas_gbsec_cost + faas_req_cost

    # 9) record
    records.append({
        'epoch':            epoch,
        'vms_desired':      vms_desired,
        'healthy_vms':      healthy_vms,
        'provisioning':     sum(cnt for _, cnt in provision_q),
        'vm_batch_sizes':   [sum(b) for b in served],
        'faas_batch_sizes': [sum(b) for b in faas],
        'vm_cost':          vm_cost,
        'faas_cost':        faas_cost,
        'total_cost':       vm_cost + faas_cost
    })

# -----------------------------------------------------------------------------
# build DataFrame & plot
# -----------------------------------------------------------------------------
df = pd.DataFrame(records)

# VM Scaling Plot
fig = plt.figure(figsize=(8,3))
plt.plot(df.epoch, df.vms_desired,  label='Desired VMs')
plt.plot(df.epoch, df.healthy_vms,  label='Healthy VMs')
plt.xlabel('Epoch'); plt.ylabel('Count')
plt.title('Desired vs. Healthy VM Scaling')
plt.legend(); plt.grid(True)
plt.tight_layout()
fig.savefig('ss_vm_scaling_no_vm.pdf')   # save figure
plt.show()

# Cost Breakdown Plot
fig = plt.figure(figsize=(8,3))
plt.plot(df.epoch, df.vm_cost,        label='VM Cost',      linestyle='-')
plt.plot(df.epoch, df.faas_cost,   label='FaaS Cost', linestyle='--')
plt.plot(df.epoch, df.total_cost,      label='Total Cost',    linewidth=2)
plt.xlabel('Epoch'); plt.ylabel('Cost ($)')
plt.title('Cost Breakdown per Epoch')
plt.legend(); plt.grid(True)
plt.tight_layout()
fig.savefig('ss_cost_breakdown_no_vm.pdf')  # save figure
plt.show()

# print final DataFrame
pd.set_option('display.max_columns', None)
print(df[['epoch','vms_desired','healthy_vms','vm_batch_sizes','faas_batch_sizes','total_cost']])
