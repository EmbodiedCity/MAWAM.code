# General Multi-Robot Physical Debugging Skills

Task-agnostic engineering checks for co-evolving any collaborative manipulation task:

1. **Grasp Stability & Opposed Contacts**
   - Avoid fragile top-down pinch grasps on wide/heavy/two-handled objects; prefer symmetric opposed grasps with verified bilateral finger contacts.
   - Ensure airborne objects never rely on synthetic attachment or `pose_overwritten=True`.

2. **Geometric Clearance & Aperture Margins**
   - Audit wrist/link bounding spans against narrow openings, rims, cages, or partner arms during approach, insertion, and extraction.
   - Add staging waypoints or wrist compaction trajectories whenever clearance drops below 5 mm.

3. **Continuous IK & Joint Limits**
   - Keep consecutive 25 Hz control steps smooth (`max |Δq| <= 0.35 rad`) and strictly within `[joint_lower, joint_upper]`.

4. **Dynamic & Spring-Loaded Timing Windows**
   - Integrate spring-damper fixtures (`I * ddtheta = tau_robot - k*theta - c*dtheta`) at 250 Hz substeps so releasing early genuinely triggers snap-back.
   - For partially synchronous tasks, the holder must wait robustly across arbitrary partner delays; for synchronous tasks, partner lag must genuinely induce physical violation (tilt, spill, missed stream) beyond the narrow tolerance.

5. **Terminal Settling & Phase-Independent Evaluation**
   - Lower release heights and append a bounded settling window so all dynamic objects come to rest (`speed <= threshold`) over the full terminal window.
   - Evaluators must check physical state (contacts, geometry, angles, speeds), never expert phase strings, and must accept native SAPIEN `(3, 4)` as well as `(4, 4)` camera extrinsics.
