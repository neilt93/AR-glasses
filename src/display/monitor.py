"""Monitor detection and display routing for AR glasses.

The RayNeo Air 4 Pro appears as a standard 1920x1080 external monitor
over USB-C DisplayPort Alt Mode. This module detects which monitor is
the glasses and provides helpers to position a fullscreen window on it.
"""

import subprocess
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class MonitorInfo:
    """Information about a connected display."""
    name: str           # e.g. "RayNeo Air 4 Pro" or "Generic PnP Monitor"
    index: int          # monitor index (0-based)
    x: int              # left edge position in virtual desktop
    y: int              # top edge position in virtual desktop
    width: int          # resolution width
    height: int         # resolution height
    is_primary: bool

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)


# Keywords to identify RayNeo glasses in monitor names
RAYNEO_KEYWORDS = ["rayneo", "air 4", "air4", "tcl", "nreal"]


def detect_monitors() -> list[MonitorInfo]:
    """Detect all connected monitors using Windows APIs.

    Uses PowerShell to query WMI for monitor info since this avoids
    extra dependencies beyond what Windows provides.
    """
    monitors = []

    # Get monitor positions and resolution from PowerShell
    ps_script = """
Add-Type -AssemblyName System.Windows.Forms
$i = 0
foreach ($screen in [System.Windows.Forms.Screen]::AllScreens) {
    $b = $screen.Bounds
    Write-Output "$i|$($screen.DeviceName)|$($b.X)|$($b.Y)|$($b.Width)|$($b.Height)|$($screen.Primary)"
    $i++
}
"""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.strip().splitlines():
            parts = line.strip().split("|")
            if len(parts) != 7:
                continue
            idx, device_name, x, y, w, h, primary = parts
            monitors.append(MonitorInfo(
                name=device_name.strip(),
                index=int(idx),
                x=int(x),
                y=int(y),
                width=int(w),
                height=int(h),
                is_primary=primary.strip() == "True",
            ))
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"[display] Failed to detect monitors via PowerShell: {e}")

    # Try to get friendly monitor names from WMI
    _enrich_monitor_names(monitors)

    return monitors


def _enrich_monitor_names(monitors: list[MonitorInfo]):
    """Try to get human-readable monitor names from WMI."""
    ps_script = """
Get-CimInstance -Namespace root\\wmi -ClassName WmiMonitorID -ErrorAction SilentlyContinue | ForEach-Object {
    $name = ($_.UserFriendlyName | Where-Object {$_ -ne 0} | ForEach-Object {[char]$_}) -join ''
    $instance = $_.InstanceName
    Write-Output "$instance|$name"
}
"""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True, text=True, timeout=10,
        )
        friendly_names = []
        for line in result.stdout.strip().splitlines():
            parts = line.strip().split("|", 1)
            if len(parts) == 2 and parts[1].strip():
                friendly_names.append(parts[1].strip())

        # Map friendly names to monitors by index (best effort)
        for i, monitor in enumerate(monitors):
            if i < len(friendly_names):
                monitor.name = friendly_names[i]
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass


def find_glasses(monitors: Optional[list[MonitorInfo]] = None) -> Optional[MonitorInfo]:
    """Find the AR glasses among connected monitors.

    Detection strategy:
    1. Look for a monitor with "RayNeo" or similar keywords in the name.
    2. If not found by name, look for a non-primary 1920x1080 display
       (the most common glasses resolution).
    3. Returns None if no glasses detected.
    """
    if monitors is None:
        monitors = detect_monitors()

    if not monitors:
        return None

    # Strategy 1: match by name
    for mon in monitors:
        name_lower = mon.name.lower()
        if any(kw in name_lower for kw in RAYNEO_KEYWORDS):
            return mon

    # Strategy 2: non-primary 1080p display
    for mon in monitors:
        if not mon.is_primary and mon.width == 1920 and mon.height == 1080:
            return mon

    return None


def find_glasses_or_fallback(monitors: Optional[list[MonitorInfo]] = None) -> MonitorInfo:
    """Find AR glasses, or fall back to primary monitor for development.

    Always returns a MonitorInfo — uses primary monitor if glasses not found.
    """
    if monitors is None:
        monitors = detect_monitors()

    glasses = find_glasses(monitors)
    if glasses is not None:
        return glasses

    # Fallback: primary monitor
    for mon in monitors:
        if mon.is_primary:
            print("[display] AR glasses not detected — using primary monitor")
            return mon

    # Last resort: synthetic monitor info
    print("[display] No monitors detected — using default 1920x1080")
    return MonitorInfo(
        name="default", index=0, x=0, y=0,
        width=1920, height=1080, is_primary=True,
    )


def print_monitor_info(monitors: Optional[list[MonitorInfo]] = None):
    """Print detected monitors for debugging."""
    if monitors is None:
        monitors = detect_monitors()

    print(f"[display] Detected {len(monitors)} monitor(s):")
    for mon in monitors:
        primary_tag = " [PRIMARY]" if mon.is_primary else ""
        print(f"  [{mon.index}] {mon.name} — {mon.width}x{mon.height} "
              f"at ({mon.x}, {mon.y}){primary_tag}")

    glasses = find_glasses(monitors)
    if glasses:
        print(f"[display] AR glasses detected: [{glasses.index}] {glasses.name}")
    else:
        print("[display] AR glasses not detected")
