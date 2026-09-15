"""雷达特征计算功能：只负责数值特征，不负责 Qt 绘制。"""

import numpy as np

from api.algorithm.sensor_function import SensorFunction


class RadarComputeWorker:
    """计算雷达特征、归一化，并生成闭合雷达图坐标。"""

    FEATURE_NAMES = [
        "是否接触", "接触面积", "接触区域X方向质心", "接触区域Y方向质心",
        "接触状态转换强度", "是否滑移", "滑移事件强度", "峰值力/峰值形变",
        "合力大小", "形变强度", "时间形变变化率", "法向场方差",
        "法向变化量", "切向场方差", "X力矩", "Y力矩", "Z力矩",
    ]
    RADAR_LABELS = [
        "是否接触", "接触面积", "X质心(蓝+/橙-)", "Y质心(蓝+/橙-)",
        "接触状态\n转换强度", "是否滑移", "滑移事件强度", "峰值力/峰值形变",
        "合力大小", "形变强度", "时间形变变化率", "法向场方差",
        "法向变化量", "切向场方差", "X力矩(蓝+/橙-)",
        "Y力矩(蓝+/橙-)", "Z力矩(蓝+/橙-)",
    ]
    SIGNED_FEATURE_INDICES = (2, 3, 14, 15, 16)
    SIGNED_FEATURE_DEADBANDS = {
        14: 1e-6,
        15: 1e-6,
        16: 1e-6,
    }
    UNIT_FEATURES = frozenset({
        "是否接触", "接触状态转换强度", "是否滑移", "滑移事件强度",
    })
    SIGNED_FEATURES = frozenset({
        "接触区域X方向质心", "接触区域Y方向质心",
    })
    AREA_THRESHOLD = 0.5
    
    SLIDE_THRESHOLD = 0.3
    SLIDE_COHERENCE_THRESHOLD = 0.70
    SLIDE_FULL_SCALE = 1.0
    PIXEL_SIZE_M = (0.05e-3, 0.05e-3)
    SURFACE_Z_M = 0.5e-3

    def __init__(self):
        self.sf = SensorFunction()
        self.touch_hist = {}
        self.prev_hue_matrix = None
        self.prev_contact = 0.0
        self.prev_contact_area_ratio = 0.0
        self.prev_deform = None
        self.prev_timestamp = None
        self.max_tracker = {
            "接触面积": None,
            "峰值力/峰值形变": None,
            "合力大小": None,
            "形变强度": None,
            "时间形变变化率": None,
            "法向场方差": None,
            "法向变化量": None,
            "切向场方差": None,
            "X力矩": None,
            "Y力矩": None,
            "Z力矩": None,
        }
        self.angles_raw = np.linspace(
            0,
            2 * np.pi,
            len(self.FEATURE_NAMES),
            endpoint=False,
            dtype=np.float64,
        )

    def reset(self):
        """校准后清除跨帧状态和自适应尺度。"""
        self.touch_hist.clear()
        self._reset_contact_sequence()
        for name in self.max_tracker:
            self.max_tracker[name] = None

    def _reset_contact_sequence(self):
        """清除只应在一次连续接触中延续的跨帧状态。"""
        self.prev_hue_matrix = None
        self.prev_contact = 0.0
        self.prev_contact_area_ratio = 0.0
        self.prev_deform = None
        self.prev_timestamp = None

    def _build_idle_output(self):
        """空载帧直接返回全零特征，不参与特征计算和量程更新。"""
        self._reset_contact_sequence()
        features = {name: 0.0 for name in self.FEATURE_NAMES}
        coordinates = np.zeros(
            len(self.FEATURE_NAMES) + 1,
            dtype=np.float64,
        )
        return coordinates.copy(), coordinates, features

    def compute(
        self,
        fz,
        hue_matrix,
        flow_matrix,
        fx_matrix,
        fy_matrix,
        fz_matrix,
        timestamp,
        contact_threshold,
    ):
        features = {}
        fz_value = float(fz) if fz is not None and np.isfinite(fz) else 0.0
        contact_threshold_value = float(contact_threshold)
        if (
            not np.isfinite(contact_threshold_value)
            or contact_threshold_value < 0.0
        ):
            raise ValueError("contact_threshold must be finite and >= 0")
        touch_thresholds = {
            "contact": contact_threshold_value,
            "slide": self.SLIDE_THRESHOLD,
            "coherence": self.SLIDE_COHERENCE_THRESHOLD,
        }

        status, self.touch_hist = self.sf._detect_touch_status(
            abs(fz_value),
            hue_matrix,
            thresholds=touch_thresholds,
            history=self.touch_hist,
            flow_matrix_filtered=flow_matrix,
        )
        is_contact = status != "idle"
        is_slide = status == "slide"
        if not is_contact:
            return self._build_idle_output()

        features["是否接触"] = 1.0 if is_contact else 0.0
        features["是否滑移"] = 1.0 if is_slide else 0.0

        fz_abs = (
            np.abs(hue_matrix)
            if hue_matrix is not None and hue_matrix.size > 0
            else None
        )
        contact_area_ratio = 0.0
        if fz_abs is not None and is_contact:
            area = self.sf._compute_contact_area(
                fz_abs,
                self.AREA_THRESHOLD,
                cell_size=1.0,
            )
            total_pixels = max(fz_abs.size, 1)
            contact_area_ratio = area / total_pixels
            features["接触面积"] = area

            cx, cy = self.sf._compute_center_of_gravity(
                fz_abs,
                self.AREA_THRESHOLD,
            )
            height, width = fz_abs.shape[:2]
            center_x = (width - 1) / 2.0
            center_y = (height - 1) / 2.0
            features["接触区域X方向质心"] = (
                float(np.clip(
                    (cx - center_x) / max(center_x, 1.0),
                    -1.0,
                    1.0,
                ))
                if cx >= 0 else 0.0
            )
            features["接触区域Y方向质心"] = (
                float(np.clip(
                    (center_y - cy) / max(center_y, 1.0),
                    -1.0,
                    1.0,
                ))
                if cy >= 0 else 0.0
            )
        else:
            features["接触面积"] = 0.0
            features["接触区域X方向质心"] = 0.0
            features["接触区域Y方向质心"] = 0.0

        cur_contact = 1.0 if is_contact else 0.0
        cur_deform = self.sf._compute_deformation_magnitude(hue_matrix)
        timestamp_value = float(timestamp)
        if (
            self.prev_deform is None
            or self.prev_timestamp is None
            or not np.isfinite(timestamp_value)
        ):
            temporal_rate = 0.0
        else:
            dt = timestamp_value - self.prev_timestamp
            temporal_rate = (
                self.sf._compute_temporal_deformation_rate(
                    cur_deform,
                    self.prev_deform,
                    dt,
                )
                if np.isfinite(dt) and dt > 0.0
                else 0.0
            )
        transition = self.sf._compute_contact_transition_strength(
            cur_contact,
            self.prev_contact,
            contact_area_ratio,
            self.prev_contact_area_ratio,
            area_weight=0.5,
        )
        features["接触状态转换强度"] = transition
        self.prev_contact = cur_contact
        self.prev_contact_area_ratio = contact_area_ratio
        self.prev_deform = cur_deform
        self.prev_timestamp = (
            timestamp_value if np.isfinite(timestamp_value) else None)

        slide_metric = float(
            self.touch_hist.get("flow_delta_mean", 0.0))
        slide_strength = (
            np.clip(slide_metric / self.SLIDE_FULL_SCALE, 0.0, 1.0)
            if is_slide else 0.0
        )
        features["滑移事件强度"] = float(slide_strength)

        if fz_abs is not None and is_contact:
            features["峰值力/峰值形变"] = (
                self.sf._compute_max_force_cell(fz_abs)[0]
            )
        else:
            features["峰值力/峰值形变"] = 0.0

        fx_total = float(np.sum(fx_matrix)) if fx_matrix is not None else 0.0
        fy_total = float(np.sum(fy_matrix)) if fy_matrix is not None else 0.0
        fz_total = float(np.sum(fz_matrix)) if fz_matrix is not None else 0.0
        magnitude, _, _ = self.sf._get_3DForce(
            fx_total,
            fy_total,
            fz_total,
        )
        features["合力大小"] = magnitude
        features["形变强度"] = cur_deform
        features["时间形变变化率"] = temporal_rate
        features["法向场方差"] = self.sf._compute_normal_field_variance(
            fz_matrix)
        features["法向变化量"] = self.sf._compute_delta_force_normal(
            hue_matrix,
            self.prev_hue_matrix,
        )
        features["切向场方差"] = self.sf._compute_shear_field_variance(
            fx_matrix,
            fy_matrix,
        )

        if (
            fx_matrix is not None
            and fy_matrix is not None
            and fz_matrix is not None
            and is_contact
        ):
            try:
                wrench = self.sf._compute_wrench_from_force_matrices(
                    fx_matrix,
                    fy_matrix,
                    fz_matrix,
                    threshold= 0.5,
                    pixel_size_m=self.PIXEL_SIZE_M,
                    origin_pixel=None,
                    surface_z_m=0,
                    force_scale_to_n=1e-3,
                )
                torque = wrench["torque"]
                features["X力矩"] = float(torque[0])
                features["Y力矩"] = float(torque[1])
                features["Z力矩"] = float(torque[2])
            except Exception as exc:
                print(f"[RadarWorker] 力矩计算错误: {exc}")
                features["X力矩"] = 0.0
                features["Y力矩"] = 0.0
                features["Z力矩"] = 0.0
        else:
            features["X力矩"] = 0.0
            features["Y力矩"] = 0.0
            features["Z力矩"] = 0.0

        self.prev_hue_matrix = hue_matrix.copy()

        values = []
        for name in self.FEATURE_NAMES:
            value = features.get(name, 0.0)
            value = (
                float(value)
                if value is not None and np.isfinite(value)
                else 0.0
            )
            if name in self.SIGNED_FEATURES:
                normalized = float(np.clip(abs(value), 0.0, 1.0))
            elif name in self.UNIT_FEATURES:
                normalized = float(np.clip(value, 0.0, 1.0))
            else:
                abs_value = abs(value)
                previous_max = self.max_tracker.get(name)
                if previous_max is None:
                    new_max = max(abs_value * 1.2, 1e-12)
                else:
                    decay = 0.99 if name == "时间形变变化率" else 0.9995
                    new_max = max(
                        abs_value * 1.2,
                        previous_max * decay,
                        1e-12,
                    )
                if name == "时间形变变化率":
                    new_max = min(new_max, 200.0)
                self.max_tracker[name] = new_max
                normalized = float(np.clip(abs_value / new_max, 0.0, 1.0))
            values.append(normalized)

        values_closed = np.asarray(values + values[:1], dtype=np.float64)
        angles_closed = np.append(
            np.pi / 2 - self.angles_raw,
            np.pi / 2 - self.angles_raw[0],
        )
        xs = values_closed * np.cos(angles_closed)
        ys = values_closed * np.sin(angles_closed)
        return xs, ys, features
