def pack_loads(loads, min_capacity, max_capacity):
    """
    Heuristic bin-packing with both min and max capacity constraints.
    Args:
        loads (List[int]): list of request counts.
        min_capacity (int): each bin must sum ≥ this.
        max_capacity (int): each bin must sum ≤ this.
    Returns:
        List[List[int]]: bins, each a list of loads.
    """
    # 1) Sort descending
    items = sorted(loads, reverse=True)
    # 2) First-Fit Decreasing for max_capacity
    bins = []             # list of tuples (current_sum, [items])
    for w in items:
        if w > max_capacity:
            raise ValueError(f"Item {w} > max_capacity {max_capacity}")
        # try to put in the bin that will be fullest but ≤ max_capacity
        best_idx = None
        best_after = None
        for i, (s, lst) in enumerate(bins):
            if s + w <= max_capacity:
                after = s + w
                if best_after is None or after > best_after:
                    best_after, best_idx = after, i
        if best_idx is None:
            bins.append((w, [w]))
        else:
            s, lst = bins[best_idx]
            lst.append(w)
            bins[best_idx] = (s + w, lst)

    # 3) Merge underfilled bins whenever two smallest fit under max_capacity
    #    This reduces bin count.
    merged = True
    while merged:
        merged = False
        # sort bins ascending by sum
        bins.sort(key=lambda x: x[0])
        if len(bins) >= 2:
            s0, lst0 = bins[0]
            s1, lst1 = bins[1]
            if s0 + s1 <= max_capacity:
                # merge them
                bins[1] = (s0 + s1, lst0 + lst1)
                del bins[0]
                merged = True

    # 4) Fix any remaining underfilled bins by "stealing" items from other bins
    #    until they reach min_capacity, if possible.
    #    For each underfilled bin B:
    #      - needed = min_capacity – sum(B)
    #      - look in other bins for the smallest item ≤ needed 
    #        whose donor bin remains ≥ min_capacity after removal.
    for i in range(len(bins)):
        s_i, lst_i = bins[i]
        if s_i >= min_capacity:
            continue
        needed = min_capacity - s_i
        # search donors
        candidates = []
        for j in range(len(bins)):
            if j == i:
                continue
            s_j, lst_j = bins[j]
            # donor must remain ≥ min_capacity after item removal
            for w in sorted(lst_j):
                if w <= needed and s_j - w >= min_capacity:
                    candidates.append((w, j))
                    break
        # pick the smallest donor item
        if not candidates:
            # cannot fix this bin
            continue
        w_move, donor_idx = min(candidates, key=lambda x: x[0])
        # perform move
        bins[donor_idx][1].remove(w_move)
        bins[donor_idx] = (bins[donor_idx][0] - w_move, bins[donor_idx][1])
        bins[i][1].append(w_move)
        bins[i] = (s_i + w_move, bins[i][1])

    # 5) Return only the item lists
    return [lst for (_s, lst) in bins]



if __name__ == "__main__":
    loads = [12, 87, 43, 56, 23, 99, 150, 35, 27, 111, 180] # permodel loads
    T_CIP = 190
    IAAS_CAPACITY = 200
    bins = pack_loads(loads, T_CIP, IAAS_CAPACITY)
    for i, b in enumerate(bins):
        print(f"Bin {i}: items={b}, sum={sum(b)}")