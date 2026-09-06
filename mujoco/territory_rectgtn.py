"""Emit the magi territory shard in RECTGTN temporal shape.

SPDX-License-Identifier: Apache-2.0

Operator doctrine 2026-09-06: "canonical temporal definition has start, end and
duration. baked from what is given."  So every fact carries all three; the
missing member is computed at ingestion.

Two decisions this module makes deliberately, both flagged to HERD:

1. Open-ended facts carry ``end = ".."`` and ``duration = ".."`` -- the
   ISO 8601-2:2019 open-end sentinel, adopted as the fleet standard.  NOT a
   span computed against a far-future date.  "Baked from what is given" means
   the bake reflects what was given; what was given is *open-ended*, so a
   concrete number there would be a manufactured value that looks like a
   measurement.

   Trade-off worth knowing: the earlier ``9999-12-31T23:59:59Z`` sentinel
   sorted naturally in range queries and ``".."`` does not, so readers doing
   range scans must handle the sentinel explicitly.  Portability won --
   any ISO 8601-2 parser reads ``".."``, whereas a bespoke sentinel needs
   documentation to travel with the data.

2. Every fact carries a ``basis``, because ``start`` means different things:

   - ``observed``  -- ``start`` is when THIS PEER VERIFIED the fact, which is
     not when the fact became true. The 4090 was in this machine long before
     2026-09-06; what began then is our knowledge of it. Recording the
     verification time as though it were the fact's origin would be
     fabrication, and the workspace runs a no-fabricated-content rule.
   - ``intrinsic`` -- the artefact itself carries the bounds (a certificate's
     notBefore/notAfter). These are the only facts whose ``start`` is a real
     origin.
   - ``assigned``  -- a decision was taken at a knowable time (task ownership
     handed over in the HERD handoff).
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

ETERNAL_END = ".."   # ISO 8601-2:2019 open-end sentinel
UNBOUNDED = ".."     # same sentinel, for symmetry (fleet standard)

PEER = "peer:magi-16739d.agents.weftspun"
HOST = "host:DESKTOP-AI4KUOU"

# Times this session can actually evidence.
HANDOFF = "2026-09-06T14:11:53Z"       # HERD handoff; magi peer role begins
V1_NOTBEFORE = "2026-09-06T15:44:39Z"  # openssl x509 -noout -dates
V2_NOTBEFORE = "2026-09-06T16:14:50Z"  # also the earliest evidence v1 was superseded
V2_NOTAFTER = "2026-10-06T16:15:20Z"
AGENTS_ROW_WRITE = "2026-09-06T16:20:05Z"  # kv metadata created_time


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def bake(start: str, end: str) -> str:
    """Compute duration from what is given; open-ended stays open-ended."""
    if end == ETERNAL_END:
        return UNBOUNDED
    delta = _parse(end) - _parse(start)
    if delta.total_seconds() == 0:
        return "PT0S"
    days, rem = delta.days, delta.seconds
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    out = "P" + (f"{days}D" if days else "")
    t = "".join(x for x in (f"{h}H" if h else "", f"{m}M" if m else "",
                            f"{s}S" if s else "") if x)
    return out + ("T" + t if t else "") if (days or t) else "PT0S"


def fact(tuple_str: str, start: str, end: str = ETERNAL_END,
         basis: str = "observed") -> dict:
    return {"tuple": tuple_str, "start": start, "end": end,
            "duration": bake(start, end), "basis": basis}


def build(verified_at: str) -> list[dict]:
    V = verified_at
    F: list[dict] = []

    # hardware -- basis observed; these predate our knowledge of them
    F += [
        fact(f"{HOST}#runs@{PEER}", HANDOFF, basis="assigned"),
        fact(f"gpu:rtx4090#owned_by@{HOST}", V),
        fact(f"gpu:rtx3090#owned_by@{HOST}", V),
        fact(f"npu:hailo-10h#owned_by@{HOST}", V),
        fact("gpu:rtx4090#compute_capability@sm_89", V),
        fact("gpu:rtx3090#compute_capability@sm_86", V),
        fact("gpu:rtx4090#pci_bus@0000:3B:00.0", V),
        fact("gpu:rtx3090#pci_bus@0000:24:00.0", V),
        fact("gpu:rtx4090#torch_index@cuda:0", V),
        fact("gpu:rtx3090#torch_index@cuda:1", V),
        fact("gpu:rtx4090#nvidia_smi_index@1", V),
        fact("gpu:rtx3090#nvidia_smi_index@0", V),
        fact("gpu:rtx4090#fp32_tflops@54.7", V),
        fact("gpu:rtx3090#fp32_tflops@25.6", V),
        fact("npu:hailo-10h#firmware@5.3.2", V),
        fact("npu:hailo-10h#usb_device@usb/002:011", V),
        fact(f"{HOST}#p2p_between_gpus@false", V),
    ]

    # identity -- intrinsic bounds from the certificates themselves
    F += [
        fact(f"cert:magi-v1#valid@{PEER}", V1_NOTBEFORE, V2_NOTBEFORE,
             basis="intrinsic"),
        fact(f"cert:magi-v2#valid@{PEER}", V2_NOTBEFORE, V2_NOTAFTER,
             basis="intrinsic"),
        fact("cert:magi-v2#serial@2d836d05094938e86ebf2db69d91a85e192f3853",
             V2_NOTBEFORE, V2_NOTAFTER, basis="intrinsic"),
        fact(f"{PEER}#client_key_spki_der_sha256@"
             "92d0062bbb452915c8dacf63e8aa225f0c279d9ca34d972bf530c2bac371b494",
             V1_NOTBEFORE, basis="intrinsic"),
        fact("bao:root_ca#spki_der_sha256@"
             "843b16ec73d7ffd89bf5714bf9df12017815f7b8fb6199fb30fcf7f8fb78027f",
             V, basis="observed"),
    ]

    # workspace and coordination
    F += [
        fact(f"workspace:C:/fabric-starforged#checked_out_on@{HOST}", V),
        fact("workspace:C:/fabric-starforged#manifest@weftspun/weftspun-keypoint@main", V),
        fact("workspace:C:/fabric-starforged#project_count@128", V),
        fact(f"bao:https://100.124.200.34:8200#coordinates@{PEER}", V1_NOTBEFORE),
        fact("bao:https://100.124.200.34:8200#tls_server_name@weftspun-bao.internal", V),
        fact(f"{PEER}#writes@agents/data/magi-16739d.agents.weftspun", AGENTS_ROW_WRITE),
        fact(f"{PEER}#writes@territory/data/magi-16739d.agents.weftspun", V),
        fact(f"{PEER}#reads@agents/data/+", AGENTS_ROW_WRITE),
        fact(f"{PEER}#coordinated_by@peer:herd-a35a77.agents.weftspun", HANDOFF,
             basis="assigned"),
    ]

    # tasks -- assigned at the handoff
    F += [fact(f"task:#{n}#owned_by@{PEER}", HANDOFF, basis="assigned")
          for n in (174, 175, 176, 161, 149)]
    F.append(fact(f"task:#177#owned_by@{PEER}", "2026-09-06T16:00:00Z",
                  basis="assigned"))

    # pins, each verified against its stated hash
    F += [
        fact("pin:coacd#sha@1401ce2a7ae1ed89c65ab958b48d489350c233c7", V),
        fact("pin:coacd#license@MIT", V),
        fact("pin:unitree_g1_mjcf#source@mujoco_menagerie/unitree_g1", V),
        fact("pin:unitree_g1_mjcf#license@BSD-3-Clause", V),
        fact("pin:unitree_g1_mjcf#actuated_dof@29", V),
        fact("pin:openbao_client#version@2.6.2/dd9c19c3", V),
        fact("pin:mjlab#python_max@3.13", V),
    ]

    # findings this peer established by measurement
    F += [
        fact("finding:e3_wrist_maps_to@wrist_yaw_joint", V),
        fact("finding:g1_anatomical_neutral_elbow_deg@+75.8", V),
        fact("finding:g1_anatomical_neutral_knee_deg@+13.8", V),
        fact("finding:g1_anatomical_neutral_shoulder_pitch_deg@+4.7", V),
        fact("finding:rom_gate_joint_count@8", V),
        fact("finding:obs_dim_body_plus_cloth@400", V),
        fact("finding:pixiv_vrm_springbone_chains@22", V),
    ]

    # data flows
    F += [
        fact(f"hf:chibifire/starforged-std-3001-appendix-e#read_by@{PEER}", V),
        fact(f"hf:chibifire/motionbricks-g1-mujoco-policies#published_by@{PEER}",
             HANDOFF, basis="assigned"),
        fact(f"github:chibifire-characters#variant_store_for@{PEER}", V,
             basis="assigned"),
        fact("github:chibifire-characters#constraint@no-git-lfs", V,
             basis="assigned"),
    ]

    # corpus + render findings, established 2026-09-06 by measurement
    F += [
        # This one CORRECTS a claim this peer relayed to HERD earlier: K_max is
        # meant to come from the USD schema rather than binary VRM parsing, but
        # nothing populates VSekaiSpringBoneAPI on this box yet.
        fact("finding:vrm_addon_for_blender#installed@false", V),
        fact("finding:k_max_from_usd_schema#currently_derivable@false", V),
        fact("finding:blend_to_usd_path#proven@true", V),
        fact("finding:blend_to_usd_path#blendshapes_preserved@108", V),
        fact("pin:anny#license@Apache-2.0", V),
        fact("pin:anny#phenotype_axes@11", V),
        fact("pin:anny#ancestry_axes@3", V),
        fact("finding:blender_boolean_solver#manifold_available@true", V),
        fact("finding:corpus_human_stratum#filled@50-of-50", V),
        fact("finding:corpus_game_stratum#unfilled@19-of-50", V),
        fact("finding:corpus_semi_humanoid#unsourced@19", V),
        fact("pin:mitsuba#version@3.9.1", V),
        fact("finding:mtoon_render_paths#found@3", V),
        fact("finding:mtoon_render_paths#consolidated_onto@mtoon_forward.py", V),
        fact("repo:6-datasource/anny-render-corpus#local_commit@ee735b7", V),
        fact("finding:mtoon_slang_gate#runnable_on_this_host@false", V),
    ]

    # honest coverage statement
    F.append(fact(f"coverage:{PEER}#asserts_only_about@magi-and-its-host", V))
    return F


def shard(verified_at: str | None = None) -> dict:
    v = verified_at or now()
    return {
        "shard_owner": "magi-16739d.agents.weftspun",
        "written_at": v,
        "session": "session_01958cvxqEkatQEJ2ucMi2yW",
        "temporal_schema": "RECTGTN start/end/duration, baked from what is given",
        "temporal_standard": "ISO 8601-2:2019; '..' is the open-end sentinel",
        "basis_values": {
            "observed": "start is when this peer VERIFIED the fact, not when it became true",
            "intrinsic": "the artefact carries its own bounds (e.g. cert notBefore/notAfter)",
            "assigned": "a decision taken at a knowable time",
        },
        "open_ended_duration": UNBOUNDED,
        "facts": build(v),
    }


if __name__ == "__main__":
    s = shard()
    print(json.dumps(s, indent=2)[:1200])
    print(f"\n... {len(s['facts'])} facts, "
          f"{len(json.dumps(s, separators=(',', ':')))} bytes compact")
    from collections import Counter
    print("basis:", dict(Counter(f["basis"] for f in s["facts"])))
