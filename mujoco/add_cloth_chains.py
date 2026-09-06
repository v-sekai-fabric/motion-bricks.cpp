"""Add VRM-SpringBone-shaped passive chains to an MJCF, for cost measurement.

SPDX-License-Identifier: Apache-2.0

Per the motionbricks formula, a VRM SpringBone chain maps to a chain of bodies
joined by passive ``<joint type="ball">`` with stiffness/damping and a sphere
collider per node, welded at the root to a skeleton bone.  This builds that
shape so the *cost* of cloth-in-the-sim-loop can be measured separately from
hull collision.

Defaults match the pixiv VRM 1.0 sample actually measured: 22 springs, so 22
chains.  Node count per chain is a parameter because the sample's per-spring
joint counts vary; 5 is mid-range for the 3-6 node cloth chains the formula
describes.

This is geometry for a cost benchmark, not a fitted garment.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

# Bones a real avatar hangs hair/skirt/sleeve chains from.
ANCHORS = [
    "torso_link", "waist_roll_link", "waist_yaw_link", "pelvis",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_hip_yaw_link", "right_hip_yaw_link",
]


def build(src: Path, out: Path, chains: int, nodes: int,
          stiffness: float, damping: float, radius: float,
          node_mass: float) -> dict:
    tree = ET.parse(src)
    root = tree.getroot()
    bodies = {b.get("name"): b for b in root.iter("body")}

    made = 0
    for c in range(chains):
        anchor = ANCHORS[c % len(ANCHORS)]
        parent = bodies.get(anchor)
        if parent is None:
            continue
        cur = parent
        for n in range(nodes):
            b = ET.SubElement(cur, "body", {
                "name": f"cloth{c}_{n}",
                "pos": f"0 0 {-0.05 if n else -0.08}",
            })
            ET.SubElement(b, "joint", {
                "name": f"cloth{c}_{n}_j", "type": "ball",
                "stiffness": f"{stiffness}", "damping": f"{damping}",
                # passive: no actuator, no reference
                "armature": "0.0001",
            })
            ET.SubElement(b, "geom", {
                "type": "sphere", "size": f"{radius}",
                "mass": f"{node_mass}",
                "contype": "1", "conaffinity": "1", "group": "4",
                "rgba": "0.3 0.5 0.9 0.5",
            })
            cur = b
            made += 1

    tree.write(out, encoding="utf-8", xml_declaration=True)
    return {"chains": chains, "nodes_per_chain": nodes, "bodies_added": made}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chains", type=int, default=22)
    ap.add_argument("--nodes", type=int, default=5)
    ap.add_argument("--stiffness", type=float, default=1.0)
    ap.add_argument("--damping", type=float, default=0.1)
    ap.add_argument("--radius", type=float, default=0.02)
    ap.add_argument("--node-mass", type=float, default=0.01)
    a = ap.parse_args()
    info = build(a.src, a.out, a.chains, a.nodes, a.stiffness, a.damping,
                 a.radius, a.node_mass)
    print(info)
    print(f"written: {a.out}")
