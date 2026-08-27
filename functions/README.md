# CV Pick — default saved functions

Auto-scanned into **Saved Functions** (extension group). These are **real Blockly
procedure trees** (same style as a user “Save to Library” export):
`function_param_get` / `function_param_set`, math/lists/loops, `import cv2`,
`function_call` to `cv2.*`, and `local_instance_call` for robot methods.

| Function | Params | Implementation (blocks) |
|----------|--------|-------------------------|
| *(capture — not included)* | — | Define later |
| `detect_coins` | `image` | `cvtColor` (+ **const** `COLOR_BGR2GRAY`) → blur → Canny → dilate → `findContours` (+ **const** `RETR_EXTERNAL` / `CHAIN_APPROX_SIMPLE`) → circularity filter → list of `[cx, cy]` |
| `map_to_robot` | `image_positions, a, b, tx, c, d, ty, z` | Pure math/lists: affine map to `[x, y, z]` poses |
| `pick_at` | `robot, pose` | `writeCoordinate` approach / descend, `pump(1)`, lift (`waitIdle` between) |

**Not used:** there is no call to a dedicated `cv_pick_helpers` module from these
JSON files. OpenCV is called via normal `function_call` blocks (`cv2....`).

## Runtime notes

- **detect_coins** needs `import cv2` and OpenCV installed in the **Blockly** env when you Run.
- Pose format is **lists**: image points `[px, py]`, robot poses `[x, y, z]` (easy with list blocks).
- Calibration numbers on the workflow match the tab: `a, b, tx, c, d, ty, z`.

## Workflow slots

1. Capture — your function  
2. Detect → `detect_coins`  
3. Map → `map_to_robot`  
4. Pick (loop) → `pick_at`  
