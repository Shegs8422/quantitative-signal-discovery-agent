# GIFs & images (docs assets)

Place drops here, then reference from `../walkthrough.md`.

## Asset contract

| Filename | Length / size | Content | Caption |
|---|---|---|---|
| `nvidia-build-blueprint.png` | screenshot | build.nvidia.com blueprint page (hero) | "The NVIDIA blueprint I forked" |
| `phoenix-projects-dashboard.png` | screenshot | Projects grid at localhost:6006 | "All runs captured in one view" |
| `first-run-output.gif` | 10–30 s, ≤ 6–8 MB | Terminal scroll of `nat run` first execution | "The first full loop, start to finish" |
| `phoenix-trace-detail.gif` | 10–20 s, ≤ 6–8 MB | Inside a single trace (clicks into spans, token counts) | "Per-LLM span costs visible after the profiler fix" |
| `cost-aware-autopsy.gif` | 15–30 s, ≤ 8 MB | Scroll through one accepted signal's report + portfolio autopsy | "The tool that 'passes' the test that matters" |
| `benchmark-run-table.png` | screenshot | Final benchmark markdown table | "Entitlement map + per-model cost table" |

Drop GIFs at `docs/assets/*.gif`; the `docs/.gitattributes` (if added) selects LFS for large binaries.
Image elements in `walkthrough.md` reference `assets/foo.gif` (relative).

## Recording tips (Windows)

- **Peeking** — free, lightweight (`griffinrivers/peeking`). Capture a tight terminal or browser region; export GIF.
- **ScreenToGif** — screen portion capture; export to GIF or APNG.
- Crop to the narrowest region that makes sense (640–900 px wide is plenty for terminal content). Target ≤ 8 MB; if a GIF exceeds 6 MB, reduce frame rate or duration.

## After recording

1. Drop files into `docs/assets/`.
2. Commit `docs/CONTRIBUTING.md` updates if needed.
3. Push — `git add docs/assets && git commit -m "docs: add walkthrough assets" && git push origin feat/tier1-universes-assurance`.
