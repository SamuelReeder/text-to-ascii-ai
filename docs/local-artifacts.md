# Local artifacts

Training datasets (`data/`), outputs (`runs/`), and weights (`checkpoints/`)
are intentionally ignored. They do not inflate Git clones. Keep active training
states, the phase-1 checkpoint, the released model, manifests, and evaluation
records until a verified backup exists. Historical scratch images and duplicate
caches can be archived after inspecting their use; there is no automatic deletion.

The original local `cat.png` is a user-owned, untracked input. It is not a demo
asset and is excluded from the deployment allowlist. The demo ships only curated,
generated portfolio examples with provenance in `demo/examples/provenance.json`.

The original local `runs/chain*.sh` scripts are historical machine-specific notes.
Use `scripts/reproduce.sh` for new work; it resolves the repository location at
runtime and writes new checkpoints under a separate directory.
