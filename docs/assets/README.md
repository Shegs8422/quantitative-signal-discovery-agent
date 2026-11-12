# GIFs & images (docs assets)

Reference assets used by `docs/walkthrough.md`. The files are already committed — no action needed unless you want to replace one.

## Asset inventory

| File | Size | Caption |
|---|---|---|
| `nvidia-build-blueprint.png` | 409 KB | NVIDIA build blueprint detail page — the catalog entry this started from |
| `nvidia-github-repo.png` | 150 KB | NVIDIA-AI-Blueprints GitHub repository page |
| `demo-pytest-collection.gif` | 62 KB | `uv run pytest tests/` — 115 passed in ~8 s |
| `demo-accuracy-report.gif` | 155 KB | Demo signal's full accuracy report rendering live — HAC vs. Bonferroni vs. Deflated Sharpe |
| `phoenix-hac-project.png` | 119 KB | Phoenix project — signal-discovery-workflow-hac (3 traces) |
| `phoenix-hac-trace-detail.png` | 294 KB | Phoenix span detail — HAC trace, full span tree, LLM spans with token counts and raw JSON |
| `phoenix-bolt-project.png` | 175 KB | Phoenix project — benchmark-bolt-e4b (8 traces) |
| `phoenix-bolt-trace-detail.png` | 236 KB | Phoenix span detail — benchmark trace, cumulative tokens + raw LLM output JSON |

## Notes

- All GIFs are downscaled (≤ 1000 px width) and palette-quantized (128 colors) — visually lossless for text; lossless for image structure.
- `demo-accuracy-report.gif` was 13.7 MB raw; 58× smaller after quantization with same frame count.
- No LFS needed — all files are below the GitHub warn-size threshold.
- Recording date: 2026-11-12.
