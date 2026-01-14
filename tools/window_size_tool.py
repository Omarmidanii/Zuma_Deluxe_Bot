import time
import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32
dwmapi = ctypes.windll.dwmapi

# Helps make coordinates match real pixels on many setups
try:
    user32.SetProcessDPIAware()
except Exception:
    pass

# Win32 constants
DWMWA_EXTENDED_FRAME_BOUNDS = 9  # returns RECT in screen coords

class RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG),
                ("top", wintypes.LONG),
                ("right", wintypes.LONG),
                ("bottom", wintypes.LONG)]

class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG),
                ("y", wintypes.LONG)]

def _rect_to_tuple(r: RECT):
    w = int(r.right - r.left)
    h = int(r.bottom - r.top)
    return int(r.left), int(r.top), w, h

def get_foreground_hwnd():
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        raise RuntimeError("No foreground window (GetForegroundWindow returned NULL).")
    return hwnd

def get_window_rect(hwnd):
    r = RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(r)):
        raise ctypes.WinError()
    return r

def get_extended_frame_bounds(hwnd):
    r = RECT()
    hr = dwmapi.DwmGetWindowAttribute(
        hwnd,
        DWMWA_EXTENDED_FRAME_BOUNDS,
        ctypes.byref(r),
        ctypes.sizeof(r),
    )
    # S_OK = 0
    if hr != 0:
        return None
    return r

def get_client_rect_screen(hwnd):
    # client size (0..w, 0..h)
    cr = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(cr)):
        raise ctypes.WinError()

    # top-left client point in screen coords
    pt = POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        raise ctypes.WinError()

    w = int(cr.right - cr.left)
    h = int(cr.bottom - cr.top)
    return (int(pt.x), int(pt.y), w, h)

if __name__ == "__main__":
    print("1) Click the game window to focus it.")
    print("2) Measuring in 2 seconds...")
    time.sleep(2.0)

    hwnd = get_foreground_hwnd()
    print(f"Foreground HWND: 0x{hwnd:08X}")

    # Best: extended frame bounds (accurate drawn window bounds in screen space)
    efb = get_extended_frame_bounds(hwnd)
    if efb is not None:
        l, t, w, h = _rect_to_tuple(efb)
        print(f"[ExtendedFrameBounds] left={l} top={t} width={w} height={h}")
    else:
        print("[ExtendedFrameBounds] not available (DWM call failed).")

    # Fallback: GetWindowRect
    wr = get_window_rect(hwnd)
    l, t, w, h = _rect_to_tuple(wr)
    print(f"[GetWindowRect]         left={l} top={t} width={w} height={h}")

    # Client area (content only)
    cl, ct, cw, ch = get_client_rect_screen(hwnd)
    print(f"[ClientArea(Screen)]    left={cl} top={ct} width={cw} height={ch}")
