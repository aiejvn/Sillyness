# Pairwise Key Interactions — Future Enhancement

## Summary

The current `DecomposedQNetwork` treats each key independently:

    Q(s, a) = sum_k Q_k(s, a_k)

This misses key combinations that matter in-game (e.g. W+Shift = sprint,
mouse_left + key_1 = ability combo). A pairwise interaction term fixes this.

## Design

Extend the Q-function to:

    Q(s, a) = sum_k Q_k(s, a_k) + sum_{i<j} V_ij(s, a_i, a_j)

### Implementation (removed from v0, add back when baseline is stable)

```python
# In __init__:
num_pairs = num_keys * (num_keys - 1) // 2
self.pair_head = nn.Linear(state_dim, num_pairs)

# Precompute pair indices
pairs_i, pairs_j = [], []
for i in range(num_keys):
    for j in range(i + 1, num_keys):
        pairs_i.append(i)
        pairs_j.append(j)
self.register_buffer("pairs_i", torch.tensor(pairs_i))
self.register_buffer("pairs_j", torch.tensor(pairs_j))

# In forward:
v_pairs = self.pair_head(state)                              # (B, num_pairs)
pair_mask = keys[:, self.pairs_i] * keys[:, self.pairs_j]    # (B, num_pairs)
q_interaction = (v_pairs * pair_mask).sum(dim=1)              # (B,)
q_total = q_individual + q_interaction
```

### Parameter cost

With 22 keys: 22*(22-1)/2 = 231 pair terms.  The `pair_head` linear layer adds
256 * 231 + 231 = ~59k parameters — negligible relative to the CNN.

## When to add

- After the independent-key baseline trains and converges
- If per-key Q-values look reasonable but total Q prediction is poor, that
  signals missing interaction effects
- Can also try a smaller subset of "known synergy" pairs first (e.g. WASD +
  mouse_left, ability keys + number keys) before the full N^2 approach
