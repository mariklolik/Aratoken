# Revised manuscript and latexdiff

These artifacts compare repository baseline `5d7492eb13efb81273ea13d53673bba9f1eae80f` with the manuscript at `446acb6907caddbae04b5e01115866763827a8bb`. The baseline has not been confirmed as the exact journal submission; the comparison PDF states this on every page.

| File | Contents |
|---|---|
| [manuscript.pdf](manuscript.pdf) | Clean revised manuscript, 20 pages |
| [latexdiff.pdf](latexdiff.pdf) | Readable highlighted comparison, 20 pages |
| [latexdiff-source.zip](latexdiff-source.zip) | Clean and comparison sources, complete native latexdiff, source patch, removed lines, figures and publisher build assets |

The readable PDF highlights changed prose, tables and figure frames in yellow. Changed display mathematics and the long proof addition use blue marking to preserve legibility. Entire changed tables are marked, including unchanged cells. Deletions remain in the native diff and source patch rather than appearing inline in the readable PDF. Both delivered PDFs were built on FSK and all pages visually checked.

To rebuild the readable comparison, extract `latexdiff-source.zip`, change into the extracted directory and run the following with the recorded TeX Live environment:

```sh
pdflatex -interaction=nonstopmode -halt-on-error highlighted-draft.tex
pdflatex -interaction=nonstopmode -halt-on-error highlighted-draft.tex
pdflatex -interaction=nonstopmode -halt-on-error highlighted-draft.tex
```

The archive includes `complete-source.patch` and `removed-lines.tex` as well as both flattened source versions. These generated records retain their exact bytes, including whitespace. The comparison includes its bibliography inline. `complete-latexdiff.tex` preserves the unmodified native annotation; `highlighted-draft.tex` is the layout-checked readable version. The source archive retains the original changed figures under `old-figures/`. The recorded TeX installation resolves the class's mixed-case logo references to the supplied lowercase filenames through its casefold search.

The clean manuscript source SHA256 is `c8c69d6916af06e84f74e8f46166f88f01470a50f7d79398a59617604a95b069`. File hashes are in [SHA256SUMS](SHA256SUMS). Author photos, biography approval, the submitted-baseline identity and final disclosure checks remain pending. These are review drafts.
