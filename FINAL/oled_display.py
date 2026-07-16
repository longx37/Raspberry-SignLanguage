"""OLED 显示模块 — SSD1306/SH1106 I2C OLED (128×64)
进度条由内部独立线程计时，不依赖主循环帧率。
"""

import os
import time
import threading

OLED_WIDTH = 128
OLED_HEIGHT = 64
PROGRESS_DURATION = 0.75
ANIMATION_HZ = 30

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


class OledDisplay:
    """OLED 显示屏 — 独立线程动画"""

    def __init__(self):
        self._device = None
        self._font_large = None
        self._font_small = None
        self._font_medium = None   # 24px 缓存，长词用
        self._lock = threading.Lock()
        self._word = ""
        self._show_progress = False
        self._progress_start = 0.0
        self._running = True
        self._dirty = True
        self._init_device()
        if self._device:
            self._thread = threading.Thread(target=self._animation_loop, daemon=True)
            self._thread.start()

    def _init_device(self):
        try:
            from luma.core.interface.serial import i2c
            from luma.oled.device import ssd1306, sh1106
        except ImportError:
            print("[OLED] luma.oled 未安装，OLED 功能禁用")
            return

        found = None
        for bus in [1, 0]:
            if not os.path.exists(f"/dev/i2c-{bus}"):
                continue
            for addr in [0x3C, 0x3D]:
                try:
                    serial = i2c(port=bus, address=addr)
                except Exception:
                    continue
                for cls in [ssd1306, sh1106]:
                    try:
                        dev = cls(serial, width=OLED_WIDTH, height=OLED_HEIGHT)
                        found = (dev,)
                        break
                    except Exception:
                        continue
                if found:
                    break
            if found:
                break

        if found is None:
            print("[OLED] 未检测到设备")
            return

        self._device = found[0]
        print(f"[OLED] 检测到 @ I2C-{bus}, 0x{addr:02X}")
        self._device.contrast(128)

        from PIL import ImageFont
        for fp in _FONT_CANDIDATES:
            if os.path.exists(fp):
                try:
                    self._font_large = ImageFont.truetype(fp, 36)
                    self._font_small = ImageFont.truetype(fp, 11)
                    self._font_medium = ImageFont.truetype(fp, 24)
                    break
                except Exception:
                    continue
        if self._font_large is None:
            self._font_large = ImageFont.load_default()
            self._font_small = ImageFont.load_default()
            self._font_medium = self._font_small

    @property
    def available(self):
        return self._device is not None

    def set_word(self, word, show_progress=False):
        """主循环调用 — 仅更新变量，不做 I/O"""
        if not self._device:
            return
        with self._lock:
            changed = (word != self._word) or (show_progress != self._show_progress)
            self._word = word
            if show_progress and not self._show_progress:
                self._progress_start = time.time()
            self._show_progress = show_progress
            if changed:
                self._dirty = True

    def clear(self):
        self._running = False
        if self._device:
            with self._lock:
                self._device.clear()

    def _animation_loop(self):
        interval = 1.0 / ANIMATION_HZ
        while self._running:
            with self._lock:
                word = self._word
                show_progress = self._show_progress
                dirty = self._dirty
                self._dirty = False

            if not word:
                time.sleep(interval)
                continue

            finished = False
            if show_progress:
                elapsed = time.time() - self._progress_start
                progress = min(1.0, elapsed / PROGRESS_DURATION)
                finished = (progress >= 1.0)
            else:
                progress = None

            if dirty or show_progress:
                self._draw(word, progress)

            if finished:
                with self._lock:
                    self._show_progress = False

            time.sleep(interval)

    def _draw(self, word, progress=None):
        from luma.core.render import canvas

        font = self._font_medium if len(word) > 3 else self._font_large

        with canvas(self._device) as draw:
            w, h = OLED_WIDTH, OLED_HEIGHT
            bbox = draw.textbbox((0, 0), word, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            text_bottom = h - 16 if progress is not None else h - 6
            x = (w - tw) // 2
            y = 4 + (text_bottom - 4 - th) // 2
            draw.text((x, y), word, font=font, fill="white")

            if progress is not None:
                bar_x, bar_y = 14, h - 12
                bar_w, bar_h = w - 28, 5
                draw.rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h),
                               outline="white", width=1)
                fill_w = int(bar_w * progress)
                if fill_w > 1:
                    draw.rectangle((bar_x, bar_y, bar_x + fill_w, bar_y + bar_h),
                                   fill="white")
