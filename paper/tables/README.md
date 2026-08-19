# Tables

`baseline_table.tex` — the quality vs detectability comparison. Style copied from the C4
paper (`/ssd1/ming/C4 copy/C4_paper`): booktabs rules, `\resizebox{\textwidth}`,
`\renewcommand{\arraystretch}`, `\textcolor{oursemph}` for our method, `\textcolor{failred}`
for numbers that fail the operating requirement, `\rowcolor{finalres}` for the strongest
baseline row, `\textcolor{notmeasured}` for anything not yet measured.

Two deliberate choices worth keeping:

1. **Each block has its own no-watermark control.** The three methods cannot share one,
   because they need different decoding regimes (block + top-k 3, strict left-to-right,
   full-vocabulary temperature sampling). Reporting one shared baseline would silently
   attribute the decoding regime's cost to whichever watermark used it.
2. **A detection-cost column.** dgMARK and KGW detect with zero model forwards; ReTrace
   needs `L+1 = 9`. Collapsing the comparison to TPR alone hides the method's main
   structural disadvantage.

Pending: KGW rows (re-running with bigram deduplication), and a shared-sampler variant so
the ReTrace block can be compared to the generation-time block directly rather than only
through per-block ratios.
