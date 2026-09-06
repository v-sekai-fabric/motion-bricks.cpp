"""Emit the magi peer's territory as ReBAC tuples for the OpenBao coordination store.

SPDX-License-Identifier: Apache-2.0

A ReBAC tuple is ``object#relation@subject``.  Per transport-shepherd's
``docs/design.md``, tuple keys carry the CN verbatim, so this peer is always
``peer:magi-16739d.agents.weftspun`` and never the short name -- a rename
without rewriting tuples leaves a peer authenticated but unauthorised.

Every row here is something verified on this box during the 2026-09-06 session,
not something read off a handoff document.  Where a claim came from a peer and
was not independently checked, it is omitted.

Writable path note: this peer holds CRUD on exactly
``agents/data/magi-16739d.agents.weftspun`` -- subpaths under it are ``deny``,
and ``rebac/*`` / ``territory/*`` do not exist for us.  So the map is stored as
one field on the peer's own row until a shared mount is provisioned.
"""

from __future__ import annotations

import json

PEER = "peer:magi-16739d.agents.weftspun"
HOST = "host:DESKTOP-AI4KUOU"

TUPLES: list[tuple[str, str, str]] = [
    # --- hardware, all health-checked 2026-09-06 -------------------------
    (HOST, "runs", PEER),
    ("gpu:rtx4090", "owned_by", HOST),
    ("gpu:rtx3090", "owned_by", HOST),
    ("npu:hailo-10h", "owned_by", HOST),
    ("gpu:rtx4090", "torch_index", "cuda:0"),
    ("gpu:rtx3090", "torch_index", "cuda:1"),
    ("gpu:rtx3090", "nvidia_smi_index", "0"),
    ("gpu:rtx4090", "nvidia_smi_index", "1"),
    ("gpu:rtx4090", "pci_bus", "0000:3B:00.0"),
    ("gpu:rtx3090", "pci_bus", "0000:24:00.0"),
    ("npu:hailo-10h", "usb_device", "usb/002:011"),

    # --- workspace ------------------------------------------------------
    ("workspace:C:/fabric-starforged", "checked_out_on", HOST),
    ("workspace:C:/fabric-starforged", "manifest", "weftspun/weftspun-keypoint@main"),
    ("workspace:C:/fabric-starforged", "project_count", "128"),

    # --- tasks this peer owns ------------------------------------------
    *[(f"task:#{n}", "owned_by", PEER) for n in (174, 175, 176, 161, 149, 177)],

    # --- code this peer works in ---------------------------------------
    ("repo:3-interactor/motion-bricks-cpp", "worked_by", PEER),
    ("repo:3-interactor/mujoco-mjx", "worked_by", PEER),
    ("repo:3-interactor/coacd-upstream", "worked_by", PEER),
    ("repo:3-interactor/datasource-flow/openusd-fabric", "worked_by", PEER),
    ("artifact:mujoco/rom_map.py", "lives_in", "repo:3-interactor/motion-bricks-cpp"),
    ("artifact:mujoco/rom_calibrate.py", "lives_in", "repo:3-interactor/motion-bricks-cpp"),
    ("artifact:mujoco/vrm_coacd.py", "lives_in", "repo:3-interactor/motion-bricks-cpp"),

    # --- pins, each verified against the stated hash ---------------------
    ("pin:coacd", "sha", "1401ce2a7ae1ed89c65ab958b48d489350c233c7"),
    ("pin:coacd", "license", "MIT"),
    ("pin:unitree_g1_mjcf", "source", "mujoco_menagerie/unitree_g1"),
    ("pin:unitree_g1_mjcf", "license", "BSD-3-Clause"),
    ("pin:openbao_client", "version", "2.6.2/dd9c19c3"),

    # --- data, by direction ---------------------------------------------
    ("hf:chibifire/starforged-std-3001-appendix-e", "read_by", PEER),
    ("hf:chibifire/motionbricks-g1-mujoco-policies", "published_by", PEER),
    ("github:chibifire-characters", "variant_store_for", PEER),
    ("github:chibifire-characters", "constraint", "no-git-lfs"),

    # --- coordination ----------------------------------------------------
    ("bao:https://100.124.200.34:8200", "coordinates", PEER),
    ("bao:https://100.124.200.34:8200", "tls_server_name", "weftspun-bao.internal"),
    ("bao:root_ca", "spki_der_sha256",
     "843b16ec73d7ffd89bf5714bf9df12017815f7b8fb6199fb30fcf7f8fb78027f"),
    (PEER, "client_key_spki_der_sha256",
     "92d0062bbb452915c8dacf63e8aa225f0c279d9ca34d972bf530c2bac371b494"),
    (PEER, "writes", "agents/data/magi-16739d.agents.weftspun"),
    (PEER, "reads", "agents/data/+"),
    (PEER, "coordinated_by", "peer:herd-a35a77.agents.weftspun"),
]


def as_rows() -> list[str]:
    return [f"{o}#{r}@{s}" for o, r, s in TUPLES]


def as_json() -> str:
    return json.dumps(as_rows(), separators=(",", ":"))


if __name__ == "__main__":
    rows = as_rows()
    for row in rows:
        print(row)
    print(f"\n{len(rows)} tuples, {len(as_json())} bytes as JSON")
