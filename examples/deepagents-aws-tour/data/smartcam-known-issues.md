# SmartCam V3 (SC-CAM-V3) - Known issues

## Night-vision mode degraded above 32°C ambient

**Severity:** Medium
**Affected SKU:** SC-CAM-V3
**First reported:** 2026-03-10
**Fixed in:** firmware v3.4.2 (released 2026-04-05)

### Symptom

Night-vision footage shows excessive noise and blur when ambient temperature exceeds 32°C. Most commonly reported by customers in warmer climates or in unconditioned spaces like attics and garages during summer months. Daytime color footage is unaffected.

### Root cause

The IR LED driver in firmware v3.4.0 and v3.4.1 throttles aggressively at high temperatures to prevent thermal damage to the LED array. The throttle thresholds were set too conservatively, kicking in at 32°C when the LEDs can safely operate up to 38°C.

### Fix

Firmware v3.4.2 raises the IR LED throttle threshold to 38°C and adds a separate watchdog for the imaging sensor's own thermal limits. Customers can update via the SmartHome app under Settings > Device > Cameras > SmartCam V3 > Check for updates.
