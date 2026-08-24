# Isaac Sim tray-transfer demo

Visual-context demo only: prepared supply tray moves **Cabinet → Side Table**
(beside the monitor cart). Rendered Isaac frames are **not** model inputs or labels.

## What to install (tested versions)

| Component | Version | Where to get it |
| --- | --- | --- |
| **Isaac Sim** (standalone) | **6.0.1** | [NVIDIA Isaac Sim download](https://developer.nvidia.com/isaac-sim) — choose **Isaac Sim 6.0.1 standalone** (Windows or Linux). Not Isaac Lab. |
| **Isaac for Healthcare assets** | **v0.7.0** (`724f82e`) | [Isaac for Healthcare](https://developer.nvidia.com/isaac/healthcare) / [i4h-workflows](https://github.com/isaac-for-healthcare/i4h-workflows) — download the asset pack and unpack so the folder contains the I4H USD tree. |
| GPU | NVIDIA RTX recommended | RTX Real-Time viewport for capture |

This repo was authored against:

- Isaac Sim path example: `C:\isaac-sim-standalone-6.0.1-windows-x86_64`
- I4H assets path example: `...\i4h-assets\724f82e`

## Environment variables

**Linux / macOS:**

```bash
export ISAAC_SIM_ROOT=/path/to/isaac-sim-standalone-6.0.1
export I4H_ASSETS_ROOT=/path/to/i4h-assets/724f82e
```

**Windows PowerShell:**

```powershell
$env:ISAAC_SIM_ROOT = "C:\isaac-sim-standalone-6.0.1-windows-x86_64"
$env:I4H_ASSETS_ROOT = "D:\path\to\i4h-assets\724f82e"
```

## Build the scene (USDA)

From the MedPhyGraph repo root (with your normal Python env for this repo):

```bash
python scripts/isaac/export_tray_transfer_usda.py
```

Writes (git-ignored under `runs/`):

- `runs/transition_demo/tray_transfer_demo_i4h.usda`
- `runs/transition_demo/tray_transfer_layout.json`

Core authoring code: [`src/medphygraph/tray_transfer_demo.py`](../../src/medphygraph/tray_transfer_demo.py)

## Open and play in Isaac Sim

1. Launch **Isaac Sim 6.0.1** (`isaac-sim.bat` / `isaac-sim.sh`)
2. **File → Open** → `runs/transition_demo/tray_transfer_demo_i4h.usda`
3. Viewport camera: **DemoCamera** (one viewpoint is enough)
4. Disable **Camera Light**; use scene lights + **RTX Real-Time**
5. Play the **Timeline** (tray: Cabinet → Side Table)
6. Optional capture: **Window → Movie Capture** (or Sequencer) → export MP4

CLI helper:

```bash
python scripts/isaac/open_scene.py --help
```

## Publish a playable demo video on GitHub (not a live Isaac stream)

GitHub **cannot** live-stream your local Isaac Sim viewport. To let visitors **play** the demo in the browser:

1. Capture an MP4 in Isaac (Movie Capture, 1080p, DemoCamera).
2. Host the file somewhere public:
   - **GitHub Release** asset (recommended — keeps the repo small), or
   - YouTube / Vimeo (best for LinkedIn / sharing).
3. Link or embed it from the main [`README.md`](../../README.md).

Example Release embed in README (after you upload `tray_transfer_demo.mp4` to a Release):

```html
<video src="https://github.com/kamranghz/medphygraph/releases/download/vX.Y.Z/tray_transfer_demo.mp4" controls width="880"></video>
```

Or a simple link:

```markdown
[Watch Isaac tray-transfer demo](https://github.com/kamranghz/medphygraph/releases/download/vX.Y.Z/tray_transfer_demo.mp4)
```

## Related scripts (kept in git)

| Script | Purpose |
| --- | --- |
| `export_tray_transfer_usda.py` | Build the tray-transfer USDA |
| `open_scene.py` | Open Isaac scenes from CLI |
| `transition_demo_log.py` | Build a transition log for optional viewers |
| `open_transition_viewer.py` | Optional in-viewport support-graph beams |
| `graph_update_viewer.py` / `gui_viewer.py` | Optional graph viewers |
| `render_transition_graph.py` | Optional 2D matplotlib graph (no Isaac) |

Local calibration / measure scripts are not tracked in git.
