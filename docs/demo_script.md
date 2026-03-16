# Demo Script

## Before the Demo
1. Set up the work surface under consistent lighting.
2. Place all parts in their starting positions.
3. Mount the camera (table tripod or chest mount).
4. Connect RayNeo Air 4 via USB-C.
5. Run `python scripts/run_demo.py` and verify the HUD appears on the glasses.

## During the Demo
1. Put on the RayNeo glasses.
2. Stand/sit at the work surface so the camera captures the full area.
3. Follow the on-screen instructions step by step.
4. Pause briefly between steps so the system can detect transitions.
5. Intentionally make one mistake (e.g., wrong orientation) to show the
   warning system.

## After the Demo
1. Press `q` to quit the app.
2. Review the log file in `logs/` for frame-by-frame data.
3. Optionally replay with `python scripts/replay_log.py --log <logfile>`.

## Talking Points
- "The system watches what I'm doing and tells me the next step."
- "It detected I placed the bracket wrong and warned me."
- "All inference runs locally — no cloud dependency."
- "This is one specific task; the architecture supports adding new tasks."
