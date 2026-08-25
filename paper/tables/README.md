# Paper tables

The manuscript contains five tables:

| File | Contents |
|:--|:--|
| `baseline_table.tex` | detectability, quality ratio, and model-call cost |
| `quality_table_full.tex` | greedy and multinomial task accuracy |
| `validity_table.tex` | observed false-positive rates across keys |
| `lsweep_table.tex` | probe-count and verification-cost ablation |
| `robustness_table.tex` | detection after editing |

`quality_table.tex` is a stable wrapper around `quality_table_full.tex`. The wrapper keeps the
section source unchanged when the full grid is regenerated.

Formatting conventions:

- method and setting columns are left-aligned;
- numerical columns are centered;
- booktabs rules separate headers and model groups;
- `oursemph` identifies ReplayMark;
- `finalres` highlights the primary ReplayMark configuration;
- caption spacing is controlled by `\abovecaptionskip`, without manual vertical offsets.

Every displayed value maps to a committed file under `results/`; comments in each table record the
corresponding experiment and cohort details.
