# SmartHome Hub V2 (SH-HUB-V2) - Known issues

## Intermittent wifi connection failure on firmware v2.1.4

**Severity:** High
**Affected SKU:** SH-HUB-V2
**First reported:** 2026-04-22
**Fixed in:** firmware v2.1.5 (released 2026-05-15)

### Symptom

Hub fails to maintain its wifi connection. Customers typically report "won't connect to wifi" after a power cycle or router reboot. The device pairs initially but loses connection within 24 hours and won't reconnect without manual intervention.

### Root cause

Race condition in the wifi reconnect logic shipped in firmware v2.1.4. The hub attempts to renew its DHCP lease before the wifi stack has fully initialized, causing the connection to drop and the device to enter an error state. The bug was introduced when the wifi driver was rebased onto a newer SDK in v2.1.4.

### Fix

Firmware update v2.1.5 was released on 2026-05-15. To apply it:

1. Reboot the device by unplugging power for 30 seconds, then plug back in
2. Open the SmartHome app on a phone connected to the same wifi network
3. Navigate to Settings > Device > SmartHome Hub
4. Accept the firmware update prompt when it appears
5. Wait approximately 5 minutes for the update to complete; the LED will flash blue during the update and turn solid green when done

### Verification

After the update, the hub should report firmware v2.1.5 under Settings > About in the SmartHome app. If the customer reports the issue persists after v2.1.5, escalate to the engineering team - there's a follow-up bug under investigation for routers using WPA3-only mode.
