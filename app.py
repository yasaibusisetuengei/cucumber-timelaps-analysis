import inspect
import io
import math
import os
import re
import tempfile
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw
from skimage.segmentation import find_boundaries, slic
from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates
from transformers import pipeline

st.set_page_config(
    page_title="タイムラプス葉面積率解析アプリ",
    page_icon="🥒",
    layout="wide",
)

# ------------------------------------------------------------
# 定数・プリセット設定
# ------------------------------------------------------------
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
BRUSH_DISPLAY_WIDTH = 520
WORK_MAX_SIDE = 1024
MAX_VIDEO_BYTES = 500 * 1024 * 1024  # 500 MB
MAX_IMAGE_BYTES = 100 * 1024 * 1024  # 100 MB

CAMERA_PRESETS = {
    "タイムラプスカメラ": {
        "DIM": (1280, 720),
        "K": np.array(
            [
                [768.4324165833522, 0.0, 629.193977277274],
                [0.0, 768.0867082519276, 363.13481714845415],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        "D": np.array(
            [
                [-0.07176022472676662],
                [-0.01831839322401357],
                [-0.051792564889714565],
                [0.0897998816199078],
            ],
            dtype=np.float64,
        ),
    },
    "魚眼レンズ": {
        "DIM": (2448, 3264),
        "K": np.array(
            [
                [1503.5487709350377, 0.0, 1218.6077713125294],
                [0.0, 1488.2554066129592, 1605.577171477069],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        "D": np.array(
            [
                [0.22189357229646275],
                [-1.3020264389755558],
                [2.338021345056564],
                [-1.2946507886454646],
            ],
            dtype=np.float64,
        ),
    },
}

ROTATE_OPTIONS = {
    "反時計回りに90度": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "時計回りに90度": cv2.ROTATE_90_CLOCKWISE,
    "180度回転": cv2.ROTATE_180,
    "回転なし": None,
}

COLOR_PRESETS = {
    "キュウリ葉（標準・広め）": ((15, 95), (20, 255), (20, 255)),
    "緑（標準）": ((35, 85), (40, 255), (40, 255)),
    "濃緑（明所向け）": ((35, 90), (50, 255), (30, 220)),
}

MASK_DISPLAY_COLORS = {
    "水色": (0, 200, 255),
    "赤": (255, 60, 0),
    "緑": (0, 220, 80),
    "黄": (255, 220, 0),
    "紫": (180, 60, 255),
}

# ------------------------------------------------------------
# 色相環カスタムコンポーネント (app.py 準拠)
# ------------------------------------------------------------
HUE_WHEEL_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  body {
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    font-family: sans-serif;
    background: transparent;
    user-select: none;
  }
  canvas {
    cursor: grab;
    touch-action: none;
  }
  canvas.dragging {
    cursor: grabbing;
  }
  .info {
    margin-top: 4px;
    font-size: 13px;
    color: #555;
  }
</style>
</head>
<body>
  <canvas id="wheel" width="240" height="240"></canvas>
  <div class="info" id="info">H: 15 ～ 95 (0～179)</div>
<script>
  const canvas = document.getElementById("wheel");
  const ctx = canvas.getContext("2d");
  const infoEl = document.getElementById("info");

  const cx = 120, cy = 120;
  const rOuter = 104, rInner = 64;
  let hMin = 15;
  let hMax = 95;
  let lastSyncId = -1;
  let dragging = null;
  let dragStartAngle = 0;
  let startHMin = 15, startHMax = 95;

  function sendMessage(type, data) {
    window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type: type }, data), "*");
  }

  function hToAngle(h) {
    return (h / 180) * Math.PI * 2 - Math.PI / 2;
  }

  function angleToH(angle) {
    let deg = ((angle + Math.PI / 2) * 180 / Math.PI) % 360;
    if (deg < 0) deg += 360;
    return Math.round(deg / 2) % 180;
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let deg = 0; deg < 360; deg++) {
      const a0 = (deg - 0.8) * Math.PI / 180 - Math.PI / 2;
      const a1 = (deg + 0.8) * Math.PI / 180 - Math.PI / 2;
      ctx.beginPath();
      ctx.arc(cx, cy, rOuter, a0, a1);
      ctx.arc(cx, cy, rInner, a1, a0, true);
      ctx.closePath();
      ctx.fillStyle = `hsl(${deg}, 100%, 50%)`;
      ctx.fill();
    }

    const aMin = hToAngle(hMin);
    const aMax = hToAngle(hMax);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, rInner - 4, aMin, aMax, false);
    ctx.closePath();
    ctx.fillStyle = "rgba(0, 160, 255, 0.22)";
    ctx.fill();

    function drawHandle(h, color) {
      const a = hToAngle(h);
      const x2 = cx + Math.cos(a) * (rOuter + 4);
      const y2 = cy + Math.sin(a) * (rOuter + 4);

      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = "#111";
      ctx.lineWidth = 3;
      ctx.stroke();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      const hx = cx + Math.cos(a) * ((rOuter + rInner) / 2);
      const hy = cy + Math.sin(a) * ((rOuter + rInner) / 2);
      ctx.beginPath();
      ctx.arc(hx, hy, 8, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#111";
      ctx.stroke();
    }

    drawHandle(hMin, "#ffffff");
    drawHandle(hMax, "#ffeb3b");

    infoEl.textContent = `H (色相): ${hMin} ～ ${hMax}  (ドラッグで範囲調整)`;
  }

  function getPointerPos(e) {
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  function handleDist(h, px, py) {
    const a = hToAngle(h);
    const hx = cx + Math.cos(a) * ((rOuter + rInner) / 2);
    const hy = cy + Math.sin(a) * ((rOuter + rInner) / 2);
    return Math.hypot(px - hx, py - hy);
  }

  canvas.addEventListener("pointerdown", (e) => {
    const { x, y } = getPointerPos(e);
    const dMin = handleDist(hMin, x, y);
    const dMax = handleDist(hMax, x, y);

    if (dMin < 18 && dMin <= dMax) {
      dragging = "min";
    } else if (dMax < 18) {
      dragging = "max";
    } else {
      const dist = Math.hypot(x - cx, y - cy);
      if (dist <= rOuter + 10) {
        dragging = "range";
        dragStartAngle = Math.atan2(y - cy, x - cx);
        startHMin = hMin;
        startHMax = hMax;
      }
    }
    if (dragging) {
      canvas.setPointerCapture(e.pointerId);
      canvas.classList.add("dragging");
    }
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const { x, y } = getPointerPos(e);
    const angle = Math.atan2(y - cy, x - cx);

    if (dragging === "min") {
      hMin = angleToH(angle);
    } else if (dragging === "max") {
      hMax = angleToH(angle);
    } else if (dragging === "range") {
      const diffRad = angle - dragStartAngle;
      const diffH = Math.round((diffRad / (Math.PI * 2)) * 180);
      hMin = ((startHMin + diffH) % 180 + 180) % 180;
      hMax = ((startHMax + diffH) % 180 + 180) % 180;
    }
    draw();
  });

  function endDrag() {
    if (!dragging) return;
    dragging = null;
    canvas.classList.remove("dragging");
    sendMessage("streamlit:setComponentValue", {
      value: { h_min: hMin, h_max: hMax },
      dataType: "json",
    });
  }

  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);

  window.addEventListener("message", (event) => {
    if (event.data && event.data.type === "streamlit:render") {
      const args = event.data.args || {};
      if (typeof args.sync_id === "number" && args.sync_id !== lastSyncId) {
        lastSyncId = args.sync_id;
        if (typeof args.h_min === "number") hMin = args.h_min;
        if (typeof args.h_max === "number") hMax = args.h_max;
        draw();
      }
      sendMessage("streamlit:setFrameHeight", { height: 270 });
    }
  });

  draw();
  sendMessage("streamlit:componentReady", { apiVersion: 1 });
  sendMessage("streamlit:setFrameHeight", { height: 270 });
</script>
</body>
</html>
"""


@st.cache_resource
def get_hue_wheel_component():
    comp_dir = tempfile.mkdtemp(prefix="st_hue_wheel_")
    os.chmod(comp_dir, 0o700)
    index_path = os.path.join(comp_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(HUE_WHEEL_HTML)
    return components.declare_component("hue_wheel_v2", path=comp_dir)


hue_wheel_component = get_hue_wheel_component()


# ------------------------------------------------------------
# 深度推定モデル（リソースキャッシュ）
# ------------------------------------------------------------
@st.cache_resource(show_spinner="深度推定モデル (Depth-Anything-V2-Small) を読み込み中...")
def load_depth_pipeline():
    return pipeline("depth-estimation", model=DEPTH_MODEL_ID, device=-1)


def compute_depth_map(img_rgb: np.ndarray) -> np.ndarray:
    depth_pipe = load_depth_pipeline()
    h, w = img_rgb.shape[:2]
    depth_out = depth_pipe(Image.fromarray(img_rgb))
    raw = depth_out["predicted_depth"].squeeze().detach().cpu().numpy()
    norm = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.resize(norm, (w, h), interpolation=cv2.INTER_LANCZOS4)


# ------------------------------------------------------------
# ユーティリティ関数
# ------------------------------------------------------------
def sanitize_filename(filename: str, default_stem: str = "frame") -> str:
    base = os.path.basename(filename or "")
    stem = os.path.splitext(base)[0]
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", stem).strip("._-")
    return cleaned[:80] if cleaned else default_stem


def sanitize_csv_cell(val):
    if isinstance(val, str) and val.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + val
    return val


def encode_png(img_array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img_array).save(buf, format="PNG")
    return buf.getvalue()


def parse_datetime_str(dt_str: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(dt_str.strip(), fmt)
        except ValueError:
            continue
    raise ValueError("日時は 'YYYY-MM-DD HH:MM:SS' または 'YYYY/MM/DD HH:MM:SS' 形式で入力してください。")


def get_video_first_frame_rgb(video_bytes: bytes, suffix: str) -> Tuple[Optional[np.ndarray], int]:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return None, 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None, total
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), total
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def extract_and_undistort_frames(
    video_bytes: bytes,
    suffix: str,
    start_time: datetime,
    target_hour: int,
    target_minute: int,
    max_diff_hours: float,
    interval_hours: float,
    dim: Tuple[int, int],
    k_mat: np.ndarray,
    d_mat: np.ndarray,
    scale_factor: float,
    rotate_mode: Optional[int],
    apply_fisheye: bool,
) -> Tuple[List[dict], List[str]]:
    logs: List[str] = []
    extracted: List[dict] = []

    if apply_fisheye:
        nk = k_mat.copy()
        nk[0, 0] = k_mat[0, 0] / scale_factor
        nk[1, 1] = k_mat[1, 1] / scale_factor
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            k_mat, d_mat, np.eye(3), nk, dim, cv2.CV_16SC2
        )
    else:
        map1, map2 = None, None

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError("動画ファイルを開けませんでした。")

        date_groups = defaultdict(list)
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx == 0:
                frame_idx += 1
                continue
            current_time = start_time + timedelta(hours=frame_idx * interval_hours)
            date_str = current_time.strftime("%Y-%m-%d")
            # メモリ節約のため全フレーム保持せず日付ごとに最寄りのみ逐次更新
            target_time = datetime.strptime(
                f"{date_str} {target_hour:02d}:{target_minute:02d}:00", "%Y-%m-%d %H:%M:%S"
            )
            diff_sec = abs((current_time - target_time).total_seconds())
            if date_str not in date_groups or diff_sec < date_groups[date_str][0]:
                date_groups[date_str] = (diff_sec, frame_idx, current_time, frame)
            frame_idx += 1
        cap.release()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    for date_str in sorted(date_groups.keys()):
        diff_sec, f_idx, best_c_time, best_frame = date_groups[date_str]
        time_diff_hours = diff_sec / 3600.0
        if time_diff_hours > max_diff_hours:
            logs.append(
                f"【{date_str}】 抽出時刻 {best_c_time.strftime('%Y-%m-%d %H:%M:%S')} は目標時刻から {time_diff_hours:.2f} 時間離れているためスキップ"
            )
            continue

        proc_frame = best_frame
        if apply_fisheye:
            h, w = proc_frame.shape[:2]
            if (w, h) == (dim[1], dim[0]):
                proc_frame = cv2.rotate(proc_frame, cv2.ROTATE_90_CLOCKWISE)
            elif (w, h) != (dim[0], dim[1]):
                logs.append(f"【{date_str}】 サイズ不一致のためスキップ (w={w}, h={h}, 期待={dim})")
                continue
            proc_frame = cv2.remap(
                proc_frame,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

        if rotate_mode is not None:
            proc_frame = cv2.rotate(proc_frame, rotate_mode)

        img_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        time_str = best_c_time.strftime("%Y%m%d_%H%M%S")
        fname = f"7am_fish_{time_str}.png"
        extracted.append(
            {
                "name": fname,
                "stem": sanitize_filename(fname),
                "timestamp": best_c_time.strftime("%Y-%m-%d %H:%M:%S"),
                "frame_idx": f_idx,
                "orig_rgb": img_rgb,
            }
        )
        logs.append(f"【{date_str}】 {best_c_time.strftime('%Y-%m-%d %H:%M:%S')} -> 抽出・補正完了 ({fname})")

    return extracted, logs


# ------------------------------------------------------------
# マスク生成・オーバーレイ・編集関数
# ------------------------------------------------------------
def get_roi_bounds(h: int, w: int, roi_pct: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    t, b, l, r = roi_pct
    y1 = max(0, min(h - 1, int(h * t / 100)))
    y2 = max(y1 + 1, min(h, int(h * b / 100)))
    x1 = max(0, min(w - 1, int(w * l / 100)))
    x2 = max(x1 + 1, min(w, int(w * r / 100)))
    return y1, y2, x1, x2


def create_hsv_mask(
    img_rgb: np.ndarray,
    h_min: int,
    h_max: int,
    s_min: int,
    s_max: int,
    v_min: int,
    v_max: int,
) -> np.ndarray:
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    if h_min <= h_max:
        h_mask = (h_ch >= h_min) & (h_ch <= h_max)
    else:
        h_mask = (h_ch >= h_min) | (h_ch <= h_max)
    sv_mask = (s_ch >= s_min) & (s_ch <= s_max) & (v_ch >= v_min) & (v_ch <= v_max)
    return h_mask & sv_mask


def compute_masks_for_frame(
    img_rgb: np.ndarray,
    depth_map: np.ndarray,
    roi_pct: Tuple[int, int, int, int],
    depth_thresh: int,
    h_min: int,
    h_max: int,
    s_min: int,
    s_max: int,
    v_min: int,
    v_max: int,
    morph_kernel: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """全画像サイズ (H, W) の bool 型 (depth_mask, leaf_mask) を返す（ROI外は False）。"""
    h, w = img_rgb.shape[:2]
    y1, y2, x1, x2 = get_roi_bounds(h, w, roi_pct)

    c_depth = depth_map[y1:y2, x1:x2]
    c_rgb = img_rgb[y1:y2, x1:x2]

    # 1. 深度マスク（手前側を残す: ノートブック準拠 255 - DEPTH_THRESH）
    mask_d_roi = c_depth > (255 - depth_thresh)

    # 2. HSVカラーマスク
    mask_c_roi = create_hsv_mask(c_rgb, h_min, h_max, s_min, s_max, v_min, v_max)

    # 3. 統合 & モルフォロジー演算
    leaf_roi_u8 = (mask_c_roi & mask_d_roi).astype(np.uint8) * 255
    if morph_kernel >= 3:
        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        leaf_roi_u8 = cv2.morphologyEx(leaf_roi_u8, cv2.MORPH_OPEN, kernel)
        leaf_roi_u8 = cv2.morphologyEx(leaf_roi_u8, cv2.MORPH_CLOSE, kernel)

    depth_mask_full = np.zeros((h, w), dtype=bool)
    depth_mask_full[y1:y2, x1:x2] = mask_d_roi

    leaf_mask_full = np.zeros((h, w), dtype=bool)
    leaf_mask_full[y1:y2, x1:x2] = leaf_roi_u8 > 0

    return depth_mask_full, leaf_mask_full


def render_detected_overlay(
    img_rgb: np.ndarray,
    leaf_mask: np.ndarray,
    roi_pct: Tuple[int, int, int, int],
    bg_color_rgb: Tuple[int, int, int] = (0, 100, 255),
    alpha: float = 0.5,
) -> np.ndarray:
    """ノートブック準拠の検出結果オーバーレイ（ROI内の非葉領域を青系で半透明マスク）"""
    h, w = img_rgb.shape[:2]
    y1, y2, x1, x2 = get_roi_bounds(h, w, roi_pct)
    out = img_rgb.copy()

    c_rgb = out[y1:y2, x1:x2]
    c_mask = leaf_mask[y1:y2, x1:x2]
    bg_arr = np.full_like(c_rgb, bg_color_rgb, dtype=np.uint8)
    blended = cv2.addWeighted(c_rgb, 1.0 - alpha, bg_arr, alpha, 0)
    out[y1:y2, x1:x2] = np.where(c_mask[:, :, None], c_rgb, blended)
    cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 2)
    return out


def render_edit_overlay(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    boundaries: Optional[np.ndarray],
    roi_pct: Tuple[int, int, int, int],
    mask_color: Tuple[int, int, int] = (0, 200, 255),
    mask_alpha: float = 0.45,
    boundary_color: Tuple[int, int, int] = (255, 0, 0),
    show_boundaries: bool = True,
) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    y1, y2, x1, x2 = get_roi_bounds(h, w, roi_pct)
    canvas = img_rgb.astype(np.float32).copy()
    if np.any(mask):
        color_arr = np.array(mask_color, dtype=np.float32)
        canvas[mask] = canvas[mask] * (1.0 - mask_alpha) + color_arr * mask_alpha
    out = np.clip(canvas, 0, 255).astype(np.uint8)
    if show_boundaries and boundaries is not None:
        out[boundaries] = boundary_color
    cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 2)
    return out


@st.cache_data(show_spinner=False)
def downscale_for_work(img_rgb: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    if max(h, w) <= max_side:
        return img_rgb
    scale = max_side / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


@st.cache_data(show_spinner=False)
def compute_superpixels(
    img_rgb: np.ndarray,
    n_segments: int,
    compactness: float,
) -> Tuple[np.ndarray, np.ndarray]:
    segments = slic(
        img_rgb,
        n_segments=n_segments,
        compactness=compactness,
        sigma=1.0,
        start_label=0,
        channel_axis=-1,
    )
    boundaries = find_boundaries(segments, mode="inner")
    return segments.astype(np.int32), boundaries


def resize_for_display(img: np.ndarray, max_width: int) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    if w <= max_width:
        return img, 1.0
    scale = max_width / float(w)
    new_h = int(round(h * scale))
    resized = cv2.resize(
        img,
        (max_width, new_h),
        interpolation=cv2.INTER_NEAREST if img.ndim == 2 else cv2.INTER_AREA,
    )
    return resized, scale


def freedraw_objs_to_mask(
    objs: list,
    img_w: int,
    img_h: int,
    scale_x: float,
    scale_y: float,
    default_brush: int,
) -> np.ndarray:
    mask_img = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask_img)

    for obj in objs[:500]:
        if not isinstance(obj, dict):
            continue
        try:
            raw_sw = float(obj.get("strokeWidth", default_brush))
            if not math.isfinite(raw_sw):
                raw_sw = float(default_brush)
        except (TypeError, ValueError):
            raw_sw = float(default_brush)

        stroke_w = max(1, min(300, int(round(raw_sw * scale_x))))
        r = max(1, stroke_w // 2)
        pts: List[Tuple[int, int]] = []

        path_data = obj.get("path")
        if isinstance(path_data, list):
            for cmd in path_data[:10000]:
                if isinstance(cmd, (list, tuple)):
                    nums = []
                    for v in cmd[1:]:
                        try:
                            fv = float(v)
                            if math.isfinite(fv):
                                nums.append(fv)
                        except (TypeError, ValueError):
                            pass
                    for j in range(0, len(nums) - 1, 2):
                        px = int(np.clip(round(nums[j] * scale_x), 0, img_w - 1))
                        py = int(np.clip(round(nums[j + 1] * scale_y), 0, img_h - 1))
                        pts.append((px, py))
        elif isinstance(path_data, str):
            nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", path_data)[:20000]]
            for j in range(0, len(nums) - 1, 2):
                px = int(np.clip(round(nums[j] * scale_x), 0, img_w - 1))
                py = int(np.clip(round(nums[j + 1] * scale_y), 0, img_h - 1))
                pts.append((px, py))

        if len(pts) >= 2:
            draw.line(pts, fill=255, width=stroke_w)
            for px, py in pts:
                draw.ellipse([px - r, py - r, px + r, py + r], fill=255)
        elif len(pts) == 1:
            px, py = pts[0]
            draw.ellipse([px - r, py - r, px + r, py + r], fill=255)
        elif "left" in obj and "top" in obj:
            try:
                left = float(obj.get("left", 0))
                top = float(obj.get("top", 0))
                w = max(1.0, float(obj.get("width", 10)) * float(obj.get("scaleX", 1.0)))
                h = max(1.0, float(obj.get("height", 10)) * float(obj.get("scaleY", 1.0)))
                if all(math.isfinite(v) for v in (left, top, w, h)):
                    x1 = int(np.clip(round(left * scale_x), 0, img_w - 1))
                    y1 = int(np.clip(round(top * scale_y), 0, img_h - 1))
                    x2 = int(np.clip(round((left + w) * scale_x), 0, img_w - 1))
                    y2 = int(np.clip(round((top + h) * scale_y), 0, img_h - 1))
                    draw.rectangle([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], fill=255)
            except (TypeError, ValueError):
                pass

    return np.array(mask_img, dtype=np.uint8) > 0


def compute_leaf_stats(leaf_mask: np.ndarray, roi_pct: Tuple[int, int, int, int]) -> Tuple[int, int, float]:
    h, w = leaf_mask.shape[:2]
    y1, y2, x1, x2 = get_roi_bounds(h, w, roi_pct)
    roi_mask = leaf_mask[y1:y2, x1:x2]
    extracted_pixels = int(np.count_nonzero(roi_mask))
    total_pixels = int(roi_mask.size)
    leaf_pct = (extracted_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
    return extracted_pixels, total_pixels, round(leaf_pct, 2)


# ------------------------------------------------------------
# セッション初期化
# ------------------------------------------------------------
def init_state():
    defaults = {
        "frames": [],  # list of dicts: name, stem, timestamp, orig_rgb, depth_map, depth_mask, leaf_mask, auto_leaf_mask, edited
        "video_sig": None,
        "first_frame_rgb": None,
        "video_total_frames": 0,
        "extract_logs": [],
        "roi_top": 15,
        "roi_bottom": 85,
        "roi_left": 10,
        "roi_right": 90,
        "depth_thresh": 200,
        "morph_kernel": 5,
        "h_min": 15,
        "h_max": 95,
        "s_range": (20, 255),
        "v_range": (20, 255),
        "last_preset": "キュウリ葉（標準・広め）",
        "wheel_sync_id": 0,
        "last_wheel_val": None,
        "edit_idx": 0,
        "last_click": None,
        "sp_canvas_ver": 0,
        "sp_draw_objs": [],
        "sp_n_segments": 250,
        "sp_compactness": 10.0,
        "sp_show_boundaries": True,
        "mask_color_name": "水色",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ------------------------------------------------------------
# タブ1: 動画読み込み・フレーム抽出・魚眼補正
# ------------------------------------------------------------
def render_tab1_video_extract():
    st.subheader("① タイムラプス動画の読み込み ＆ フレーム抽出・魚眼補正")

    input_mode = st.radio(
        "入力ソース",
        ["タイムラプス動画 (.avi / .mp4 / .mov)", "抽出済み画像群を直接アップロード (.png / .jpg)"],
        horizontal=True,
    )

    if input_mode == "抽出済み画像群を直接アップロード (.png / .jpg)":
        uploaded_imgs = st.file_uploader(
            "画像を複数選択してアップロード",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="direct_img_uploader",
        )
        if uploaded_imgs and st.button("アップロード画像を解析セットに読み込む", type="primary"):
            loaded_frames = []
            for uf in sorted(uploaded_imgs, key=lambda x: x.name):
                raw = uf.getvalue()
                if len(raw) > MAX_IMAGE_BYTES:
                    st.warning(f"{uf.name} は上限 (100MB) を超えているためスキップしました。")
                    continue
                pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
                loaded_frames.append(
                    {
                        "name": uf.name,
                        "stem": sanitize_filename(uf.name),
                        "timestamp": uf.name,
                        "orig_rgb": np.array(pil_img, dtype=np.uint8),
                        "depth_map": None,
                        "depth_mask": None,
                        "leaf_mask": None,
                        "auto_leaf_mask": None,
                        "edited": False,
                    }
                )
            st.session_state.frames = loaded_frames
            st.session_state.edit_idx = 0
            st.session_state.sp_canvas_ver += 1
            st.session_state.sp_draw_objs = []
            st.success(f"{len(loaded_frames)} 枚の画像を読み込みました。タブ②で解析を実行してください。")
        return

    uploaded_video = st.file_uploader(
        "タイムラプス動画をアップロード (.avi / .mp4 / .mov / .mkv)",
        type=["avi", "mp4", "mov", "mkv"],
        key="video_uploader",
    )
    if uploaded_video is None:
        st.info("動画ファイルをアップロードすると、1フレーム目（開始日時の確認用）が表示されます。")
        return

    video_bytes = uploaded_video.getvalue()
    if len(video_bytes) > MAX_VIDEO_BYTES:
        st.error("動画ファイルサイズが上限 (500MB) を超えています。")
        return

    sig = f"{uploaded_video.name}_{len(video_bytes)}"
    ext = os.path.splitext(uploaded_video.name)[1].lower() or ".avi"
    if st.session_state.video_sig != sig:
        with st.spinner("動画の1フレーム目を読み込んでいます..."):
            first_rgb, total_frames = get_video_first_frame_rgb(video_bytes, ext)
        st.session_state.video_sig = sig
        st.session_state.first_frame_rgb = first_rgb
        st.session_state.video_total_frames = total_frames

    col_preview, col_params = st.columns([1.2, 1])
    with col_preview:
        st.markdown("**1フレーム目の確認（撮影開始日時の確認用）**")
        if st.session_state.first_frame_rgb is not None:
            st.image(
                st.session_state.first_frame_rgb,
                caption=f"最初のフレーム（総フレーム数: {st.session_state.video_total_frames}）",
                use_container_width=True,
            )
            st.caption("※画像内に表示されている日時を右側の「開始日時」に入力してください。")
        else:
            st.error("動画の1フレーム目を読み込めませんでした。")

    with col_params:
        st.markdown("**抽出・魚眼補正パラメータ**")
        start_time_str = st.text_input(
            "最初のフレームの撮影日時 (START_TIME_STR)",
            value="2026-09-16 16:08:48",
            help="YYYY-MM-DD HH:MM:SS または YYYY/MM/DD HH:MM:SS",
        )
        c_t1, c_t2, c_t3 = st.columns(3)
        with c_t1:
            target_hour = st.number_input("抽出時刻 (時)", min_value=0, max_value=23, value=7, step=1)
        with c_t2:
            target_minute = st.number_input("抽出時刻 (分)", min_value=0, max_value=59, value=0, step=1)
        with c_t3:
            max_diff_hours = st.number_input("許容時間差 (時間)", min_value=0.1, max_value=12.0, value=1.0, step=0.5)

        interval_hours = st.number_input("撮影間隔 (時間/フレーム)", min_value=0.05, max_value=24.0, value=1.0, step=0.25)

        apply_fisheye = st.checkbox("魚眼補正を適用する", value=True)
        camera_preset = st.selectbox(
            "カメラプリセット",
            ["タイムラプスカメラ", "魚眼レンズ", "手動入力(manual)"],
            disabled=not apply_fisheye,
        )
        scale_factor = st.number_input(
            "魚眼補正 scale_factor",
            min_value=0.5,
            max_value=3.0,
            value=1.2,
            step=0.1,
            disabled=not apply_fisheye,
        )

        if apply_fisheye and camera_preset == "手動入力(manual)":
            dim_str = st.text_input("DIM (幅, 高さ)", value="2448, 3264")
            k_str = st.text_area(
                "カメラ行列 K (9要素をカンマ区切り)",
                value="1991.1358579338798, 0.0, 1251.8570347183052, 0.0, 1998.2075244370055, 1623.313188533819, 0.0, 0.0, 1.0",
            )
            d_str = st.text_area(
                "歪み係数 D (4要素をカンマ区切り)",
                value="0.07090706193968417, 0.21295055793218973, -1.3569197441727983, 1.6377312004781808",
            )
        else:
            dim_str, k_str, d_str = "", "", ""

        rotate_label = st.selectbox("最終回転 (ROTATE_MODE)", list(ROTATE_OPTIONS.keys()), index=0)

        if st.button("フレーム抽出 ＆ 魚眼補正を実行", type="primary", use_container_width=True):
            try:
                start_dt = parse_datetime_str(start_time_str)
                if apply_fisheye:
                    if camera_preset == "手動入力(manual)":
                        dims = [int(x.strip()) for x in dim_str.split(",")]
                        k_vals = [float(x.strip()) for x in k_str.split(",")]
                        d_vals = [float(x.strip()) for x in d_str.split(",")]
                        if len(dims) != 2 or len(k_vals) != 9 or len(d_vals) != 4:
                            raise ValueError("DIMは2個、Kは9個、Dは4個の数値をカンマ区切りで入力してください。")
                        dim = (dims[0], dims[1])
                        k_mat = np.array(k_vals, dtype=np.float64).reshape(3, 3)
                        d_mat = np.array(d_vals, dtype=np.float64).reshape(4, 1)
                    else:
                        preset = CAMERA_PRESETS[camera_preset]
                        dim = preset["DIM"]
                        k_mat = preset["K"]
                        d_mat = preset["D"]
                else:
                    dim = (1280, 720)
                    k_mat = np.eye(3, dtype=np.float64)
                    d_mat = np.zeros((4, 1), dtype=np.float64)

                with st.spinner("動画を走査し、抽出・魚眼補正を実行しています..."):
                    extracted, logs = extract_and_undistort_frames(
                        video_bytes=video_bytes,
                        suffix=ext,
                        start_time=start_dt,
                        target_hour=int(target_hour),
                        target_minute=int(target_minute),
                        max_diff_hours=float(max_diff_hours),
                        interval_hours=float(interval_hours),
                        dim=dim,
                        k_mat=k_mat,
                        d_mat=d_mat,
                        scale_factor=float(scale_factor),
                        rotate_mode=ROTATE_OPTIONS[rotate_label],
                        apply_fisheye=apply_fisheye,
                    )

                st.session_state.frames = [
                    {
                        **item,
                        "depth_map": None,
                        "depth_mask": None,
                        "leaf_mask": None,
                        "auto_leaf_mask": None,
                        "edited": False,
                    }
                    for item in extracted
                ]
                st.session_state.extract_logs = logs
                st.session_state.edit_idx = 0
                st.session_state.sp_canvas_ver += 1
                st.session_state.sp_draw_objs = []

                if extracted:
                    st.success(f"合計 {len(extracted)} 枚の原本画像を抽出・補正しました。タブ②で深度推定・葉マスク生成を実行してください。")
                else:
                    st.warning("条件に一致するフレームが見つかりませんでした。抽出時刻・許容時間差・カメラサイズをご確認ください。")
            except Exception as e:
                st.error(f"処理エラー: {e}")

    if st.session_state.extract_logs:
        with st.expander("抽出・補正ログ", expanded=False):
            st.text("\n".join(st.session_state.extract_logs))

    if st.session_state.frames:
        st.markdown(f"**抽出済み原本画像プレビュー（{len(st.session_state.frames)} 枚）**")
        cols = st.columns(min(4, len(st.session_state.frames)))
        for idx, item in enumerate(st.session_state.frames[:8]):
            with cols[idx % len(cols)]:
                st.image(item["orig_rgb"], caption=item["name"], use_container_width=True)
        if len(st.session_state.frames) > 8:
            st.caption(f"※ほか {len(st.session_state.frames) - 8} 枚はタブ④で確認・ダウンロードできます。")


# ------------------------------------------------------------
# タブ2: 深度推定 & 自動マスク生成
# ------------------------------------------------------------
def render_tab2_auto_mask():
    st.subheader("② 深度推定 ＆ 自動マスク生成（ROI・深度・HSV・ノイズ除去）")
    frames = st.session_state.frames
    if not frames:
        st.info("先にタブ①で動画からフレームを抽出するか、画像をアップロードしてください。")
        return

    col_ctrl, col_view = st.columns([1, 1.8])

    with col_ctrl:
        st.markdown("**ROI（切り取り範囲 %）**")
        c1, c2 = st.columns(2)
        with c1:
            roi_top = st.slider("上 (%)", 0, 95, key="roi_top")
            roi_left = st.slider("左 (%)", 0, 95, key="roi_left")
        with c2:
            roi_bottom = st.slider("下 (%)", 5, 100, key="roi_bottom")
            roi_right = st.slider("右 (%)", 5, 100, key="roi_right")

        if roi_top >= roi_bottom:
            st.warning("ROIの「上」は「下」より小さくしてください。")
        if roi_left >= roi_right:
            st.warning("ROIの「左」は「右」より小さくしてください。")

        roi_pct = (roi_top, max(roi_top + 1, roi_bottom), roi_left, max(roi_left + 1, roi_right))

        st.divider()
        st.markdown("**深度・ノイズ除去パラメータ**")
        depth_thresh = st.slider(
            "深度しきい値 (DEPTH_THRESH)",
            min_value=0,
            max_value=255,
            key="depth_thresh",
            help="値が大きいほど奥の領域も含めます（判定: depth > 255 - DEPTH_THRESH）。",
        )
        morph_kernel = st.slider(
            "モルフォロジー カーネルサイズ",
            min_value=1,
            max_value=15,
            step=2,
            key="morph_kernel",
            help="3以上で細かいノイズ除去（OPEN）と穴埋め（CLOSE）を行います。",
        )

        st.divider()
        st.markdown("**HSVカラーフィルタ**")
        preset = st.selectbox("色プリセット", list(COLOR_PRESETS.keys()), key="hsv_preset_select")
        if preset != st.session_state.last_preset:
            st.session_state.last_preset = preset
            (h0, h1), s_def, v_def = COLOR_PRESETS[preset]
            st.session_state.h_min = h0
            st.session_state.h_max = h1
            st.session_state.s_range = s_def
            st.session_state.v_range = v_def
            st.session_state.wheel_sync_id += 1

        wheel_val = hue_wheel_component(
            h_min=st.session_state.h_min,
            h_max=st.session_state.h_max,
            sync_id=st.session_state.wheel_sync_id,
            key="hue_wheel_widget",
        )
        if (
            isinstance(wheel_val, dict)
            and "h_min" in wheel_val
            and "h_max" in wheel_val
            and wheel_val != st.session_state.last_wheel_val
        ):
            st.session_state.last_wheel_val = wheel_val
            st.session_state.h_min = int(np.clip(int(wheel_val["h_min"]), 0, 179))
            st.session_state.h_max = int(np.clip(int(wheel_val["h_max"]), 0, 179))

        s_min, s_max = st.slider("S (彩度) 範囲", 0, 255, key="s_range")
        v_min, v_max = st.slider("V (明度) 範囲", 0, 255, key="v_range")

        st.divider()
        overwrite_edited = st.checkbox("手動補正済みのフレームも上書きする", value=False)
        run_all = st.button("全フレームで深度推定 ＆ マスク生成を実行", type="primary", use_container_width=True)

    with col_view:
        preview_idx = st.selectbox(
            "プレビュー対象フレーム",
            range(len(frames)),
            format_func=lambda i: f"{i+1}/{len(frames)}: {frames[i]['name']}",
            key="tab2_preview_idx",
        )
        target_frame = frames[preview_idx]

        if run_all:
            prog = st.progress(0.0, text="深度推定とマスク生成を実行中...")
            for i, item in enumerate(frames):
                if item["depth_map"] is None:
                    item["depth_map"] = compute_depth_map(item["orig_rgb"])
                d_mask, l_mask = compute_masks_for_frame(
                    item["orig_rgb"],
                    item["depth_map"],
                    roi_pct,
                    depth_thresh,
                    st.session_state.h_min,
                    st.session_state.h_max,
                    s_min,
                    s_max,
                    v_min,
                    v_max,
                    morph_kernel,
                )
                item["depth_mask"] = d_mask
                item["auto_leaf_mask"] = l_mask.copy()
                if (not item["edited"]) or overwrite_edited or item["leaf_mask"] is None:
                    item["leaf_mask"] = l_mask.copy()
                    if overwrite_edited:
                        item["edited"] = False
                prog.progress((i + 1) / len(frames), text=f"処理中: {i+1}/{len(frames)} ({item['name']})")
            prog.empty()
            st.session_state.sp_canvas_ver += 1
            st.session_state.sp_draw_objs = []
            st.success("全フレームの深度マスク・葉マスク生成が完了しました！タブ③で手動補正、タブ④で出力できます。")

        # 選択フレームのリアルタイムプレビュー（深度マップ計算済みならスライダー変更を即時反映）
        if target_frame["depth_map"] is not None:
            d_prev, l_prev = compute_masks_for_frame(
                target_frame["orig_rgb"],
                target_frame["depth_map"],
                roi_pct,
                depth_thresh,
                st.session_state.h_min,
                st.session_state.h_max,
                s_min,
                s_max,
                v_min,
                v_max,
                morph_kernel,
            )
            overlay_prev = render_detected_overlay(target_frame["orig_rgb"], l_prev, roi_pct)
            ext_px, tot_px, pct = compute_leaf_stats(l_prev, roi_pct)

            st.metric("プレビュー葉面積率 (ROI内)", f"{pct:.2f} %", f"{ext_px:,} / {tot_px:,} px")
            p1, p2 = st.columns(2)
            with p1:
                st.image(target_frame["orig_rgb"], caption="原本画像", use_container_width=True)
                st.image((d_prev.astype(np.uint8)) * 255, caption="深度マスク（白黒）", use_container_width=True)
            with p2:
                st.image(overlay_prev, caption="検出オーバーレイ", use_container_width=True)
                st.image((l_prev.astype(np.uint8)) * 255, caption="葉のマスク（白黒）", use_container_width=True)
        else:
            # 深度未計算時は原本画像とROI枠のみプレビュー
            h, w = target_frame["orig_rgb"].shape[:2]
            y1, y2, x1, x2 = get_roi_bounds(h, w, roi_pct)
            roi_vis = target_frame["orig_rgb"].copy()
            cv2.rectangle(roi_vis, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 3)
            st.image(roi_vis, caption="原本画像 + ROI枠（左の実行ボタンで深度推定・マスク生成を行います）", use_container_width=True)


# ------------------------------------------------------------
# タブ3: マスクの手動補正 (app.py 準拠のスーパーピクセル & ブラシ編集)
# ------------------------------------------------------------
def render_tab3_manual_edit():
    st.subheader("③ マスクの手動補正（スーパーピクセル ＆ ブラシ）")
    frames = st.session_state.frames
    if not frames:
        st.info("先にタブ①・②で画像を読み込み、自動マスクを生成してください。")
        return

    ready_indices = [i for i, f in enumerate(frames) if f["leaf_mask"] is not None]
    if not ready_indices:
        st.warning("まだ葉のマスクが生成されていません。タブ②で「全フレームで深度推定 ＆ マスク生成を実行」を押してください。")
        return

    if st.session_state.edit_idx not in ready_indices:
        st.session_state.edit_idx = ready_indices[0]

    # フレーム切り替えナビゲーション
    nav_c1, nav_c2, nav_c3 = st.columns([1, 3, 1])
    with nav_c1:
        if st.button("◀ 前のフレーム", use_container_width=True):
            pos = ready_indices.index(st.session_state.edit_idx)
            st.session_state.edit_idx = ready_indices[(pos - 1) % len(ready_indices)]
            st.session_state.last_click = None
            st.session_state.sp_canvas_ver += 1
            st.session_state.sp_draw_objs = []
            st.rerun()
    with nav_c2:
        selected_idx = st.selectbox(
            "編集対象フレーム",
            ready_indices,
            index=ready_indices.index(st.session_state.edit_idx),
            format_func=lambda i: (
                f"{i+1}/{len(frames)}: {frames[i]['name']}"
                + (" [手動補正済み]" if frames[i]["edited"] else "")
            ),
            label_visibility="collapsed",
        )
        if selected_idx != st.session_state.edit_idx:
            st.session_state.edit_idx = selected_idx
            st.session_state.last_click = None
            st.session_state.sp_canvas_ver += 1
            st.session_state.sp_draw_objs = []
            st.rerun()
    with nav_c3:
        if st.button("次のフレーム ▶", use_container_width=True):
            pos = ready_indices.index(st.session_state.edit_idx)
            st.session_state.edit_idx = ready_indices[(pos + 1) % len(ready_indices)]
            st.session_state.last_click = None
            st.session_state.sp_canvas_ver += 1
            st.session_state.sp_draw_objs = []
            st.rerun()

    item = frames[st.session_state.edit_idx]
    orig_rgb = item["orig_rgb"]
    orig_h, orig_w = orig_rgb.shape[:2]
    roi_pct = (
        st.session_state.roi_top,
        max(st.session_state.roi_top + 1, st.session_state.roi_bottom),
        st.session_state.roi_left,
        max(st.session_state.roi_left + 1, st.session_state.roi_right),
    )

    # 作業解像度にリサイズ
    work_rgb = downscale_for_work(orig_rgb, WORK_MAX_SIDE) if max(orig_h, orig_w) > WORK_MAX_SIDE else orig_rgb
    work_h, work_w = work_rgb.shape[:2]
    if (work_h, work_w) != (orig_h, orig_w):
        work_mask = cv2.resize(
            item["leaf_mask"].astype(np.uint8),
            (work_w, work_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    else:
        work_mask = item["leaf_mask"].copy()

    col_ctrl, col_main, col_masks = st.columns([1, 1.8, 1.4])
    apply_brush_clicked = False

    with col_ctrl:
        st.markdown("**編集設定**")
        mask_color_name = st.selectbox("マスク表示色", list(MASK_DISPLAY_COLORS.keys()), key="mask_color_name")
        selected_mask_color = MASK_DISPLAY_COLORS[mask_color_name]

        edit_method = st.selectbox(
            "修正方法",
            ["スーパーピクセルをクリック（反転）", "直接描いて修正（ブラシ）"],
        )
        show_boundaries = st.checkbox("スーパーピクセル境界線を表示", key="sp_show_boundaries")
        st.divider()

        if edit_method == "スーパーピクセルをクリック（反転）":
            n_segments = st.slider(
                "スーパーピクセルの粒度（分割数）",
                min_value=50,
                max_value=1500,
                step=10,
                key="sp_n_segments",
            )
            compactness = st.slider(
                "コンパクトネス（形状の均一さ）",
                min_value=1.0,
                max_value=40.0,
                step=1.0,
                key="sp_compactness",
            )
            brush_radius = st.slider(
                "まとめてピクセルを選択",
                min_value=0,
                max_value=100,
                value=0,
                step=5,
                help="0にするとクリックした1つのスーパーピクセルのみを反転します。",
            )
        else:
            n_segments = st.session_state.sp_n_segments
            compactness = st.session_state.sp_compactness
            brush_action = st.radio("ブラシ操作", ["追加", "消去"], horizontal=True)
            brush_size = st.slider("ブラシの大きさ (px)", min_value=3, max_value=60, value=15, step=1)
            snap_to_sp = st.checkbox("描いた部分をスーパーピクセル単位に合わせる", value=False)
            if st.button("描画レイヤーをクリア", use_container_width=True):
                st.session_state.sp_canvas_ver += 1
                st.session_state.sp_draw_objs = []
                st.rerun()
            apply_brush_clicked = st.button("編集画面に適用", type="primary", use_container_width=True)

        st.divider()
        if item["auto_leaf_mask"] is not None and st.button("自動生成マスクにリセット", use_container_width=True):
            item["leaf_mask"] = item["auto_leaf_mask"].copy()
            item["edited"] = False
            st.session_state.last_click = None
            st.session_state.sp_canvas_ver += 1
            st.session_state.sp_draw_objs = []
            st.rerun()

        ext_px, tot_px, pct = compute_leaf_stats(item["leaf_mask"], roi_pct)
        st.metric("現在の葉面積率 (ROI内)", f"{pct:.2f} %", f"{ext_px:,} / {tot_px:,} px")

    with st.spinner("スーパーピクセルを計算中..."):
        segments, boundaries = compute_superpixels(work_rgb, n_segments, compactness)

    def save_work_mask_to_item(new_work_mask: np.ndarray):
        if (work_h, work_w) != (orig_h, orig_w):
            item["leaf_mask"] = cv2.resize(
                new_work_mask.astype(np.uint8),
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            item["leaf_mask"] = new_work_mask.copy()
        item["edited"] = True

    with col_main:
        st.markdown("**編集画面**")
        overlay_img = render_edit_overlay(
            work_rgb,
            work_mask,
            boundaries,
            roi_pct,
            mask_color=selected_mask_color,
            show_boundaries=show_boundaries,
        )
        disp_img, _ = resize_for_display(overlay_img, BRUSH_DISPLAY_WIDTH)
        disp_h, disp_w = disp_img.shape[:2]

        if edit_method == "直接描いて修正（ブラシ）":
            stroke_col = "#00FF88" if brush_action == "追加" else "#FF3333"
            sp_draw_res = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=brush_size,
                stroke_color=stroke_col,
                background_image=Image.fromarray(disp_img),
                update_streamlit=True,
                drawing_mode="freedraw",
                width=disp_w,
                height=disp_h,
                key=f"sp_draw_canvas_{st.session_state.edit_idx}_{st.session_state.sp_canvas_ver}",
            )

            if sp_draw_res.json_data is not None and "objects" in sp_draw_res.json_data:
                st.session_state.sp_draw_objs = sp_draw_res.json_data["objects"]

            if apply_brush_clicked:
                objs = st.session_state.get("sp_draw_objs", [])
                if len(objs) > 0:
                    changed = False
                    for obj in objs:
                        stroke_work = freedraw_objs_to_mask(
                            [obj],
                            work_w,
                            work_h,
                            work_w / float(disp_w),
                            work_h / float(disp_h),
                            brush_size,
                        )
                        if not np.any(stroke_work):
                            continue
                        if snap_to_sp:
                            touched_labels = np.unique(segments[stroke_work])
                            target_region = np.isin(segments, touched_labels)
                        else:
                            target_region = stroke_work

                        obj_col = str(obj.get("stroke", "")).upper()
                        if obj_col == "#00FF88":
                            work_mask[target_region] = True
                        elif obj_col == "#FF3333":
                            work_mask[target_region] = False
                        else:
                            work_mask[target_region] = (brush_action == "追加")
                        changed = True

                    if changed:
                        save_work_mask_to_item(work_mask)
                        st.session_state.sp_canvas_ver += 1
                        st.session_state.sp_draw_objs = []
                        st.rerun()
                    else:
                        st.warning("描画領域が取得できませんでした。もう一度描画してお試しください。")
                else:
                    st.warning("編集画面にブラシで描画してから「編集画面に適用」を押してください。")
        else:
            coord_kwargs = {"key": f"sp_canvas_{st.session_state.edit_idx}"}
            if "use_column_width" in inspect.signature(streamlit_image_coordinates).parameters:
                coord_kwargs["use_column_width"] = "always"

            coords = streamlit_image_coordinates(Image.fromarray(disp_img), **coord_kwargs)
            if coords is not None:
                click_sig = (st.session_state.edit_idx, coords["x"], coords["y"], brush_radius, n_segments)
                if st.session_state.last_click != click_sig:
                    st.session_state.last_click = click_sig
                    rendered_w = max(1, int(coords.get("width", disp_w)))
                    rendered_h = max(1, int(coords.get("height", disp_h)))
                    orig_x = int(np.clip(round(coords["x"] * work_w / rendered_w), 0, work_w - 1))
                    orig_y = int(np.clip(round(coords["y"] * work_h / rendered_h), 0, work_h - 1))

                    if brush_radius == 0:
                        target_region = segments == segments[orig_y, orig_x]
                    else:
                        yy, xx = np.ogrid[:work_h, :work_w]
                        circle = (xx - orig_x) ** 2 + (yy - orig_y) ** 2 <= brush_radius ** 2
                        target_labels = np.unique(segments[circle])
                        target_region = np.isin(segments, target_labels)

                    center_val = work_mask[orig_y, orig_x]
                    work_mask[target_region] = not center_val
                    save_work_mask_to_item(work_mask)
                    st.rerun()

    with col_masks:
        st.markdown("**現在のマスク（白黒）**")
        st.image((item["leaf_mask"].astype(np.uint8)) * 255, caption="葉のマスク（補正後）", use_container_width=True)
        if item["depth_mask"] is not None:
            st.image((item["depth_mask"].astype(np.uint8)) * 255, caption="深度マスク", use_container_width=True)


# ------------------------------------------------------------
# タブ4: 結果確認 ＆ 一括エクスポート (原本画像・深度マスク・葉マスク・CSV)
# ------------------------------------------------------------
def render_tab4_export():
    st.subheader("④ 解析結果の確認 ＆ 出力（原本画像・深度マスク・葉のマスク・CSV）")
    frames = st.session_state.frames
    ready_frames = [f for f in frames if f["leaf_mask"] is not None]
    if not ready_frames:
        st.info("まだ解析結果がありません。タブ①・②を実行してください。")
        return

    roi_pct = (
        st.session_state.roi_top,
        max(st.session_state.roi_top + 1, st.session_state.roi_bottom),
        st.session_state.roi_left,
        max(st.session_state.roi_left + 1, st.session_state.roi_right),
    )

    rows = []
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in ready_frames:
            stem = item["stem"]
            orig_rgb = item["orig_rgb"]
            depth_mask_u8 = (item["depth_mask"].astype(np.uint8)) * 255 if item["depth_mask"] is not None else np.zeros(orig_rgb.shape[:2], dtype=np.uint8)
            leaf_mask_u8 = (item["leaf_mask"].astype(np.uint8)) * 255
            detected_rgb = render_detected_overlay(orig_rgb, item["leaf_mask"], roi_pct)

            ext_px, tot_px, leaf_pct = compute_leaf_stats(item["leaf_mask"], roi_pct)
            rows.append(
                {
                    "filename": sanitize_csv_cell(item["name"]),
                    "timestamp": sanitize_csv_cell(item["timestamp"]),
                    "leaf_area_percentage (%)": leaf_pct,
                    "extracted_pixels": ext_px,
                    "total_pixels": tot_px,
                    "manual_edited": item["edited"],
                }
            )

            zf.writestr(f"original_images/{stem}.png", encode_png(orig_rgb))
            zf.writestr(f"depth_masks/{stem}_depth_mask.png", encode_png(depth_mask_u8))
            zf.writestr(f"leaf_masks/{stem}_leaf_mask.png", encode_png(leaf_mask_u8))
            zf.writestr(f"detected_images/detected_{stem}.png", encode_png(detected_rgb))

        df = pd.DataFrame(rows)
        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        zf.writestr("leaf_area_results.csv", csv_bytes)

    zip_buffer.seek(0)

    c_dl1, c_dl2 = st.columns(2)
    with c_dl1:
        st.download_button(
            label=f"📦 原本画像・深度マスク・葉のマスク・結果画像・CSVを一括ZIPダウンロード ({len(ready_frames)}枚)",
            data=zip_buffer,
            file_name="cucumber_leaf_area_results.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True,
        )
    with c_dl2:
        st.download_button(
            label="📄 解析結果CSVのみダウンロード",
            data=csv_bytes,
            file_name="leaf_area_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.dataframe(df, use_container_width=True)
    if len(df) >= 2:
        st.line_chart(df.set_index("filename")["leaf_area_percentage (%)"])

    st.divider()
    st.markdown("**個別フレームの確認 ＆ ダウンロード**")
    view_idx = st.selectbox(
        "表示するフレームを選択",
        range(len(ready_frames)),
        format_func=lambda i: f"{i+1}/{len(ready_frames)}: {ready_frames[i]['name']}",
        key="export_view_idx",
    )
    sel = ready_frames[view_idx]
    sel_stem = sel["stem"]
    sel_orig = sel["orig_rgb"]
    sel_depth_u8 = (sel["depth_mask"].astype(np.uint8)) * 255 if sel["depth_mask"] is not None else np.zeros(sel_orig.shape[:2], dtype=np.uint8)
    sel_leaf_u8 = (sel["leaf_mask"].astype(np.uint8)) * 255
    sel_det = render_detected_overlay(sel_orig, sel["leaf_mask"], roi_pct)

    v1, v2, v3, v4 = st.columns(4)
    with v1:
        st.image(sel_orig, caption="原本画像", use_container_width=True)
        st.download_button(
            "原本画像を保存",
            data=encode_png(sel_orig),
            file_name=f"{sel_stem}.png",
            mime="image/png",
            use_container_width=True,
        )
    with v2:
        st.image(sel_depth_u8, caption="深度マスク", use_container_width=True)
        st.download_button(
            "深度マスクを保存",
            data=encode_png(sel_depth_u8),
            file_name=f"{sel_stem}_depth_mask.png",
            mime="image/png",
            use_container_width=True,
        )
    with v3:
        st.image(sel_leaf_u8, caption="葉のマスク", use_container_width=True)
        st.download_button(
            "葉のマスクを保存",
            data=encode_png(sel_leaf_u8),
            file_name=f"{sel_stem}_leaf_mask.png",
            mime="image/png",
            use_container_width=True,
        )
    with v4:
        st.image(sel_det, caption="検出オーバーレイ", use_container_width=True)
        st.download_button(
            "オーバーレイを保存",
            data=encode_png(sel_det),
            file_name=f"detected_{sel_stem}.png",
            mime="image/png",
            use_container_width=True,
        )


# ------------------------------------------------------------
# メインエントリポイント
# ------------------------------------------------------------
def main():
    init_state()
    st.title("🥒 タイムラプス葉面積率解析 ＆ マスク補正アプリ")
    st.caption(
        "①動画から指定時刻のフレームを抽出・魚眼補正し、②Depth-Anything-V2とHSVで自動マスクを作成、③スーパーピクセル・ブラシで手動補正したうえで、④原本画像・深度マスク・葉のマスク・CSVを一括出力できます。"
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "① 動画抽出・魚眼補正",
            "② 深度推定・自動マスク",
            "③ マスク手動補正",
            "④ 結果確認・一括出力",
        ]
    )

    with tab1:
        render_tab1_video_extract()
    with tab2:
        render_tab2_auto_mask()
    with tab3:
        render_tab3_manual_edit()
    with tab4:
        render_tab4_export()


if __name__ == "__main__":
    main()