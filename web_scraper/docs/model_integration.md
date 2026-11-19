# Model integration boundary

## Current model entry point

The existing model loads four aggregate loss files:

- `economic_loss.csv`
- `response_costs.csv`
- `litigated_cases.csv`
- `fines_penalties.csv`

It uses their totals to create broad loss proportions for Incident Response Costs, Litigation Costs, Privacy Liability Coverage, and Business Interruption Loss. The current code also multiplies the stored loss amounts by 1,000 during loading.

The repository currently stores these files under `Data/`, while the inspected loader refers to `./data/`. This case and path mismatch should be resolved separately before treating the model as reproducible across macOS and Linux.

## Why the scraper does not overwrite those files

Professor Arora's target taxonomy is different from the legacy case-type and cost-type aggregation. A direct overwrite would create at least three risks:

1. Incorrect category mapping.
2. Double counting when a disclosed total overlaps component costs.
3. Mixing incident-level observations with already aggregated legacy totals.

## Proposed data flow

```text
Authorized sources
      |
      v
Source records and versions
      |
      v
Loss observations with evidence
      |
      +----> Review queue for ambiguous mappings
      |
      v
Research exports
      |
      +----> model_input_wide.csv
      |
      v
Approved aggregation adapter
      |
      v
Versioned model calibration inputs
```

The aggregation adapter should be added only after the professor or research team approves:

- the mapping from target taxonomy to model outputs;
- treatment of settlements, fines, demands, payments, and insurance recoveries;
- currency and inflation normalization;
- deduplication rules for the same incident across sources;
- whether the old and new datasets should be pooled or modeled separately.

Until then, the existing `Data/` files remain unchanged.
