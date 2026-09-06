"""Attach VRM per-bone convex hulls to the G1 MJCF, for collision cost measurement.

SPDX-License-Identifier: Apache-2.0

APPROXIMATION, STATED UP FRONT
------------------------------
VRM bones are not G1 bones, and this does NOT perform a real retarget.  The
goal is a *cost* measurement -- how much does per-bone hull collision cost at
scale -- not an anatomically correct avatar.  So:

* VRM bones map to the nearest-equivalent G1 body by an explicit table below.
  VRM secondary bones (``J_Sec_*``: cloth, hair, skirt) attach to the nearest
  plausible parent, since G1 has no counterpart at all.
* VRM is Y-up, G1 MJCF is Z-up, so hull vertices are rotated
  ``(x, y, z) -> (x, -z, y)``.
* Each hull is re-centred on its target body's origin.  Without this the hulls
  land in VRM world space, every pair interpenetrates, and the contact count
  explodes into a meaningless number.  Re-centring keeps hull *sizes* real
  (limb-sized) and positions plausible, which is what the cost depends on.

The resulting model is right for counting geoms and contacts and wrong for
anything anatomical.  Do not use it to judge a policy.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# VRM humanoid/secondary bone -> nearest-equivalent G1 body. Approximate.
# G1 has no neck, head, toe or finger bodies, so those fold into their parent.
BONE_TO_G1 = {
    "Hips": "pelvis", "Spine": "waist_yaw_link", "Chest": "waist_roll_link",
    "UpperChest": "torso_link", "Neck": "torso_link", "Head": "torso_link",
    "L_UpperLeg": "left_hip_yaw_link", "R_UpperLeg": "right_hip_yaw_link",
    "L_LowerLeg": "left_knee_link", "R_LowerLeg": "right_knee_link",
    "L_Foot": "left_ankle_roll_link", "R_Foot": "right_ankle_roll_link",
    "L_ToeBase": "left_ankle_roll_link", "R_ToeBase": "right_ankle_roll_link",
    "L_Shoulder": "left_shoulder_pitch_link", "R_Shoulder": "right_shoulder_pitch_link",
    "L_UpperArm": "left_shoulder_yaw_link", "R_UpperArm": "right_shoulder_yaw_link",
    "L_LowerArm": "left_elbow_link", "R_LowerArm": "right_elbow_link",
    "L_Hand": "left_wrist_yaw_link", "R_Hand": "right_wrist_yaw_link",
}
TORSO_FALLBACK = "torso_link"


def g1_body_for(vrm_bone: str) -> str:
    """Best-effort VRM bone name -> G1 body name."""
    b = vrm_bone
    for pfx in ("J_Bip_C_", "J_Bip_L_", "J_Bip_R_", "J_Sec_L_", "J_Sec_R_",
                "J_Sec_C_", "J_Roll_L_", "J_Roll_R_", "J_Adj_L_", "J_Adj_R_"):
        if b.startswith(pfx):
            side = "L_" if "_L_" in pfx else ("R_" if "_R_" in pfx else "")
            stem = b[len(pfx):]
            for key, body in BONE_TO_G1.items():
                if key.lstrip("LR_") and key.lstrip("LR_").lower() in stem.lower():
                    if key.startswith(("L_", "R_")) and not side:
                        continue
                    if side and key.startswith(("L_", "R_")) and not key.startswith(side):
                        continue
                    return body
            # secondary cloth/hair with no match: hang it off the torso
            if side == "L_":
                return "left_hip_yaw_link" if "leg" in stem.lower() else TORSO_FALLBACK
            if side == "R_":
                return "right_hip_yaw_link" if "leg" in stem.lower() else TORSO_FALLBACK
            return TORSO_FALLBACK
    return TORSO_FALLBACK


def yup_to_zup(v: np.ndarray) -> np.ndarray:
    out = np.empty_like(v)
    out[:, 0] = v[:, 0]
    out[:, 1] = -v[:, 2]
    out[:, 2] = v[:, 1]
    return out


def build(g1_xml: Path, hulls_npz: Path, out_xml: Path,
          max_hulls: int | None = None) -> dict:
    tree = ET.parse(g1_xml)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    bodies = {b.get("name"): b for b in root.iter("body")}

    data = np.load(hulls_npz)
    keys = sorted({k.rsplit("|", 1)[0] for k in data.files if k.startswith("hull|")})

    added, skipped, per_body = 0, 0, {}
    for key in keys:
        vk, fk = f"{key}|v", f"{key}|f"
        if vk not in data or fk not in data:
            continue
        bone = key.split("|")[1]
        target = g1_body_for(bone)
        body = bodies.get(target)
        if body is None:
            skipped += 1
            continue

        V = yup_to_zup(np.asarray(data[vk], dtype=np.float64))
        V = V - V.mean(axis=0)  # re-centre on the body origin
        F = np.asarray(data[fk], dtype=np.int64)
        if len(V) < 4 or len(F) < 4:
            skipped += 1
            continue

        mesh_name = f"hull_{added}"
        ET.SubElement(asset, "mesh", {
            "name": mesh_name,
            "vertex": " ".join(f"{x:.5g}" for x in V.reshape(-1)),
            "face": " ".join(str(int(i)) for i in F.reshape(-1)),
        })
        ET.SubElement(body, "geom", {
            "type": "mesh", "mesh": mesh_name, "class": "collision",
            "contype": "1", "conaffinity": "1", "group": "3",
            "rgba": "0.8 0.3 0.3 0.4",
        })
        per_body[target] = per_body.get(target, 0) + 1
        added += 1
        if max_hulls and added >= max_hulls:
            break

    tree.write(out_xml, encoding="utf-8", xml_declaration=True)
    return {"hulls_added": added, "skipped": skipped, "per_body": per_body}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--g1", type=Path, default=Path("g1.xml"))
    ap.add_argument("--hulls", type=Path, default=Path("hulls_pixiv.npz"))
    ap.add_argument("--out", type=Path, default=Path("g1_with_hulls.xml"))
    ap.add_argument("--max-hulls", type=int, default=None)
    a = ap.parse_args()
    info = build(a.g1, a.hulls, a.out, a.max_hulls)
    print(f"hulls added : {info['hulls_added']}")
    print(f"skipped     : {info['skipped']}")
    print("per G1 body :")
    for b, n in sorted(info["per_body"].items(), key=lambda kv: -kv[1]):
        print(f"   {b:<32}{n:>4}")
    print(f"written     : {a.out}")
