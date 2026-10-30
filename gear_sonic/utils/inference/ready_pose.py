"""Scene-independent G1 body29 preparation targets; no hardware connections."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET

# Hardware body order -> planner upper-body order (waist, interleaved arms).
UPPER_ORDER = (12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28)


def load_ready_pose(path):
    return parse_ready_pose(Path(path).expanduser().read_text())


def parse_ready_pose(text):
    values = [float(x) for x in text.split()]
    if len(values) != 29 or not all(math.isfinite(x) for x in values):
        raise ValueError('准备姿态必须包含 29 个有限角度，单位 rad')
    model_path = Path(__file__).resolve().parents[2] / 'devices/inspire/assets/g1_29dof_rh56dfx.xml'
    model = ET.parse(model_path).getroot()
    joints = {j.get('name'): j for j in model.iter('joint') if j.get('name')}
    names = [a.get('joint') for a in model.find('actuator') if '_inspire_' not in a.get('joint', '')]
    if len(names) != 29:
        raise ValueError('准备姿态需要 G1 body29 关节顺序')
    for name, value in zip(names, values):
        low, high = map(float, joints[name].get('range').split())
        if not low <= value <= high:
            raise ValueError('准备姿态超出关节限位：' + name)
    return values


def upper_body_error(state, pose):
    measured = state.get('body_q', [])
    if len(measured) != 29 or not all(math.isfinite(float(x)) for x in measured):
        raise ValueError('准备姿态缺少有效的 29 维身体反馈')
    return max(abs(float(measured[i]) - pose[i]) for i in range(12, 29))
