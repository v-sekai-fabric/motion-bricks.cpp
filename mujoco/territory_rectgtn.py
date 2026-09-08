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
TRELLIS_OBLITERATED = "2026-09-07T02:00:00Z"  # operator ruling adopted by HERD
REOPENED_AT         = "2026-09-07T03:15:00Z"  # HERD op re-read BLOCKLIST.md 1710-1758; scope is
                                              # generator-use, VoxHammer (VAST-AI) not covered


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
         basis: str = "observed", duration_override: str | None = None) -> dict:
    """RECTGTN fact. ``duration_override`` carries a closure REASON in the
    duration slot when a row is retracted rather than merely expired -- HERD's
    ``closed-retracted-<reason>`` convention. Machine-parseable readers still
    detect closure by ``end != ".."``; the duration string tells a human why.
    """
    return {"tuple": tuple_str, "start": start, "end": end,
            "duration": duration_override if duration_override else bake(start, end),
            "basis": basis}


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

    # Reopenings landed as a standalone block after the identity list, so all
    # facts are added inside build() rather than at module level.
    F += [
        # Reopened per operator/HERD confirmation that blocklist:trellis2 is
        # generator-use scoped (BLOCKLIST.md 1710-1758). VoxHammer is VAST-AI,
        # an editor, not in the four dropped repos. Closures above stay visible.
        fact("design:dress-on#env_edit@voxhammer-upstream/pixi.toml-default",
             "2026-09-07T03:15:00Z", basis="assigned",
             duration_override="reopened@voxhammer-alive-blocklist-scope-is-generator-use"),
        fact("design:dress-on#env_edit_torch@2.4.0+cu118-python3.11",
             "2026-09-07T03:15:00Z", basis="observed",
             duration_override="reopened@voxhammer-alive-blocklist-scope-is-generator-use"),
        fact("repo:3-interactor/voxhammer-upstream#branch@magi/voxhammer-windows-env-drafted-uncommitted",
             "2026-09-07T03:15:00Z", basis="observed",
             duration_override="reopened@voxhammer-alive-blocklist-scope-is-generator-use"),
        fact("bench:magi-4090#voxhammer_step1_2_3_wall@200s-first-run-then-cached",
             "2026-09-07T03:15:00Z", basis="observed",
             duration_override="reopened@voxhammer-alive-blocklist-scope-is-generator-use"),
        fact("doctrine:trellis2-blocklist#scope@generator-use-not-checkpoint-use-per-BLOCKLIST.md-1710-1758",
             "2026-09-07T03:15:00Z", basis="assigned"),
        fact("doctrine:trellis2-blocklist#voxhammer_status@alive-not-in-four-dropped-repos",
             "2026-09-07T03:15:00Z", basis="assigned"),
        fact("finding:framing-error@trellis2-blocklist-read-as-broader-than-it-said-corrected-after-2-rounds",
             "2026-09-07T03:15:00Z", basis="observed"),
    ]

    # identity -- intrinsic bounds from the certificates themselves.
    # v3 rotation 2026-09-07: fresh Ed25519 key (v1/v2 reused their key; v3
    # gives forward secrecy). v2 stays valid until natural expiry Oct 6 for
    # graceful cutover per operator ruling; v3 notAfter Oct 7 is one day past
    # v2's, closing the gap.
    V3_NOTBEFORE = "2026-09-07T02:59:32Z"
    V3_NOTAFTER  = "2026-10-07T03:00:02Z"
    F += [
        fact(f"cert:magi-v1#valid@{PEER}", V1_NOTBEFORE, V2_NOTBEFORE,
             basis="intrinsic"),
        # v2 now closes at v3 notBefore for the "which one is active" query,
        # while its own notAfter (2026-10-06) stays valid on-wire for cutover.
        fact(f"cert:magi-v2#valid@{PEER}", V2_NOTBEFORE, V2_NOTAFTER,
             basis="intrinsic"),
        fact(f"cert:magi-v2#active@{PEER}", V2_NOTBEFORE, V3_NOTBEFORE,
             basis="assigned"),
        fact("cert:magi-v2#serial@2d836d05094938e86ebf2db69d91a85e192f3853",
             V2_NOTBEFORE, V2_NOTAFTER, basis="intrinsic"),
        fact(f"cert:magi-v3#valid@{PEER}", V3_NOTBEFORE, V3_NOTAFTER,
             basis="intrinsic"),
        fact(f"cert:magi-v3#active@{PEER}", V3_NOTBEFORE, basis="assigned"),
        fact("cert:magi-v3#serial@0e9363209be217812b9d93e80fb08147b58009b8",
             V3_NOTBEFORE, V3_NOTAFTER, basis="intrinsic"),
        # v1 key SPKI still true across v1 and v2 (they reused it), then closed.
        fact(f"{PEER}#client_key_spki_der_sha256@"
             "92d0062bbb452915c8dacf63e8aa225f0c279d9ca34d972bf530c2bac371b494",
             V1_NOTBEFORE, V3_NOTBEFORE, basis="intrinsic"),
        # v3 key is genuinely new -- different SPKI, proof-of-fresh-key.
        fact(f"{PEER}#client_key_spki_der_sha256@"
             "6e1b613da082a5b868d89122e914f17cbc8dddcbba0a2d3b16b003974c93d964",
             V3_NOTBEFORE, basis="intrinsic"),
        # doctrine addition surfaced by HERD during the sign: LibreSSL's
        # `openssl` (macOS system default) has no Ed25519 and silently
        # SHA-256s empty input, producing a wrong-for-the-right-reason
        # PIN CHECK failure. Real OpenSSL 3.x only for these fingerprints.
        fact("doctrine:spki-der-sha256#tooling@requires-openssl-3.x-not-libressl-empty-input-silent-fail",
             V3_NOTBEFORE, basis="assigned"),
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
    # Successor storage rule for the retracted github-variant-store row.
    # Operator 2026-09-07T03:26Z: dress-on outputs go to HuggingFace 'since
    # it's a bulk dataset'. Reason class named alongside the rule so a reader
    # sees WHY HF wins here (bulk) even though other 'our variants' still may
    # not -- this is a class rule, not a blanket reversal of storage-triage.
    F += [
        fact("doctrine:storage-triage#bulk_dataset@huggingface-per-operator-2026-09-07",
             "2026-09-07T03:26:00Z", basis="assigned"),
        fact("design:dress-on#output_destination@hf:chibifire/anny-dress-on-stage-train",
             "2026-09-07T03:26:00Z", basis="assigned"),
        fact("doctrine:storage-triage#bulk_dataset_reason@bulk-transport-and-viewer-ergonomics-outweigh-variant-store-preference",
             "2026-09-07T03:26:00Z", basis="assigned"),
    ]

    # Operator 2026-09-07T03:35Z ruled joined-view over ETNF satellites
    # rather than the single-wide-row-with-nulls shape HERD proposed.
    # Storage stays ETNF-honest per CLAUDE.md; presentation is a view
    # served by the HF datasets loading script.
    F += [
        fact("design:dress-on#storage-shape@satellite-parquet-tables-target-plus-score-plus-artefact-refs-no-nulls",
             "2026-09-07T03:35:00Z", basis="assigned"),
        fact("design:dress-on#viewer-shape@joined-view-served-by-datasets-loading-script-not-underlying-storage",
             "2026-09-07T03:35:00Z", basis="assigned"),
        fact("finding:hf-joined-view#falsifiability@small-repo-with-two-parquet-tables-plus-loading-script-must-render-joined-in-viewer-before-shipping-real-shards",
             "2026-09-07T03:35:00Z", basis="assigned"),
    ]

    # Coordinator succession: operator confirmed on MAGI's own channel 2026-09-07T04:40:00Z
    # (AskUserQuestion, "Succession"). HERD parked; MAGI is fleet coordinator.
    F += [
        fact(f"{PEER}#role@fleet-coordinator-succession-of-herd", "2026-09-07T04:40:00Z", basis="assigned"),
        fact(f"{PEER}#promoted-by@operator-magi-channel-confirmation", "2026-09-07T04:40:00Z", basis="assigned"),
        fact("peer:herd-a35a77.agents.weftspun#role@parked-former-coordinator-succeeded-by-magi", "2026-09-07T04:40:00Z", basis="assigned"),
        fact("doctrine:fleet-coordinator#reading@succession-confirmed-by-both-operators", "2026-09-07T04:40:00Z", basis="assigned"),
        fact("doctrine:peer_must_confirm_with_own_operator#binds@coordinator-too", "2026-09-07T04:40:00Z", basis="assigned"),
        # bao token capabilities, measured 2026-09-07T04:40:00Z: coordinator role does not
        # widen the token. HERD's paths are read-only to MAGI; policies denied.
        fact(f"{PEER}#capability@territory/data/herd-a35a77.agents.weftspun/*:list,read", "2026-09-07T04:40:00Z"),
        fact(f"{PEER}#capability@agents/data/herd-a35a77.agents.weftspun:read", "2026-09-07T04:40:00Z"),
        fact(f"{PEER}#capability@sys/policies/acl/territory-rw:deny", "2026-09-07T04:40:00Z"),
        fact("finding:coordinator-scope#writes-to-herd-shard@not-possible-with-magi-token-raise-do-not-widen", "2026-09-07T04:40:00Z"),
        # Dress-on plan approved 2026-09-07T04:40:00Z; findings from its first steps.
        fact("finding:appendix-e#body_dimensions@present-E.2-stature+8-dims-p1-p50-p99-by-sex-E.5-mass-p1-p99-by-sex", "2026-09-07T04:40:00Z"),
        fact("finding:appendix-e#identity_grid@2-sex-x-3-percentile=6-identities-rows-reach-10-via-garment-pairing", "2026-09-07T04:40:00Z"),
        fact("finding:appendix-e#E.13@workspace-body-model-identity-spec-soma-77-joints-hammersley-views-twist-bones", "2026-09-07T04:40:00Z"),
        fact("finding:maskscore#MASKSCORE.md@cited-by-maskscore_rung_1_stubs.py:38-absent-from-disk", "2026-09-07T04:40:00Z"),
        fact("finding:voxhammer-example#frame@model-unit-scaled-y-up-mask-0.24-cube-same-frame", "2026-09-07T04:40:00Z"),
        fact("design:dress-on#pipeline@2d-tryon-composite-then-voxhammer-lift-two-half-torso-passes", "2026-09-07T04:40:00Z", basis="assigned"),
        fact("design:dress-on#renderer@mitsuba3-seeded-hammersley-emits-voxhammer-step1-contract", "2026-09-07T04:40:00Z", basis="assigned"),
        fact("design:dress-on#dataset_schema@maskscore-root-candidates-scores-plus-judge-satellite", "2026-09-07T04:40:00Z", basis="assigned"),
        fact("design:dress-on#identity@appendix-e-E.2-E.5-via-lbfgs-measurement-residual", "2026-09-07T04:40:00Z", basis="assigned"),
        fact("blocklist:garment-measurements#status@blocklisted-licence-operator-2026-09-07", "2026-09-07T04:40:00Z", basis="assigned"),
    ]

    # Operator directives during dress-on implementation, 2026-09-07T05:40:00Z.
    F += [
        fact("doctrine:renderer#blender#status@off-the-path-mitsuba3-only-no-fallback-bpy-leaves-env", "2026-09-07T05:40:00Z", basis="assigned"),
        fact("doctrine:lights#point-lights@rejected-do-not-exist-area-emitters-hidden-from-camera-instead", "2026-09-07T05:40:00Z", basis="assigned"),
        fact("doctrine:language#trajectory@python-glue-now-cpp-ggml-like-pixal3d.cpp-later-file-contracts-keep-steps-swappable", "2026-09-07T05:40:00Z", basis="assigned"),
        fact("doctrine:mesh-operators#format@openusd-canonical-usda-mesh-cube-prims-displayColor-primvar-ply-only-at-voxhammer-voxelizer-boundary", "2026-09-07T05:40:00Z", basis="assigned"),
        fact("finding:hf-api-key#bao_path@not-under-agents-or-readable-territory-other-mounts-deny-ask-operator", "2026-09-07T05:40:00Z"),
        fact("finding:mitsuba-gate#cameras@exact-max-dpos-6e-8-max-dR-2.5e-7-vs-seeded-blender", "2026-09-07T05:40:00Z"),
        fact("finding:mitsuba-gate#voxels@jaccard-0.972-before-matching-normalize-fixed-after", "2026-09-07T05:40:00Z"),
        fact("finding:mitsuba-gate#features-shipped-example@cosine-0.895-shading-mismatch-material-and-agx-not-geometry", "2026-09-07T05:40:00Z"),
        fact("finding:identity-solver#result@4-of-4-within-tol-after-muscle-released-and-stature-weighted-10x-negative-controls-fail-correctly", "2026-09-07T05:40:00Z"),
        fact("finding:composite#garment-photos@laid-flat-sideways-neckline-right-rotate-90ccw-fit-torso-height-centroid-align", "2026-09-07T05:40:00Z"),
    ]

    # Round 2 of dress-on implementation, 2026-09-07T06:20:00Z.
    F += [
        fact("peer:herd-a35a77.agents.weftspun#role@archived-operator-2026-09-07-shards-historical", "2026-09-07T06:20:00Z", basis="assigned"),
        fact(f"{PEER}#role@sole-fleet-coordinator", "2026-09-07T06:20:00Z", basis="assigned"),
        fact(f"{PEER}#credential@hf_token-stored-in-agents-row-field-hf_token-readable-by-agents-read-peers", "2026-09-07T06:20:00Z", basis="assigned"),
        fact("design:dress-on#publish_target@real-huggingface-chibifire-anny-dress-on-stage-train-viewtest-first", "2026-09-07T06:20:00Z", basis="assigned"),
        fact("finding:mitsuba-gate#bare-body@cams-exact-2.5e-7-voxels-identical-jaccard-1.0-feature-cosine-mean-0.972-p05-0.954-min-0.90", "2026-09-07T06:20:00Z"),
        fact("finding:mitsuba-gate#verdict@blender-off-by-directive-delta-recorded-not-gated", "2026-09-07T06:20:00Z"),
        fact("finding:voxhammer-pass1#anny-body@completed-238s-peak-10.5GB-92780v-185524f-146176-gaussians", "2026-09-07T06:20:00Z"),
        fact("finding:voxhammer-floor#source-decode@outside-mask-p50-0.0048-p95-0.060-vs-original-mesh-controls-must-use-source-recon", "2026-09-07T06:20:00Z"),
        fact("finding:voxhammer-mask#front-box@608-of-3214-surface-voxels-18.9pct-regenerated", "2026-09-07T06:20:00Z"),
        fact("finding:blender#bpy-exit@segfault-139-at-teardown-after-complete-write-both-runs", "2026-09-07T06:20:00Z"),
        fact("design:dress-on#assets@usda-canonical-mesh-displayColor-masks-cube-prims-metersPerUnit-recorded", "2026-09-07T06:20:00Z", basis="assigned"),
    ]

    # Round 3, 2026-09-07T07:05:00Z: renderer reads USD directly; colour findings; batch 1 launched.
    F += [
        fact("design:dress-on#renderer_input@mitsuba-mesh-built-in-memory-from-usd-stage-pxr-no-ply-on-render-path", "2026-09-07T07:05:00Z", basis="assigned"),
        fact("finding:mitsuba#ply-vertex-colour@uchar-red-green-blue-alpha-renders-black-float-r-g-b-reads-correctly", "2026-09-07T07:05:00Z"),
        fact("finding:voxhammer-pass2#anny-body@completed-174s-peak-13.7GB-alloc-23.6GB-gpu-total-110264v-220524f", "2026-09-07T07:05:00Z"),
        fact("finding:voxhammer-appearance#gaussian-colour@sh-dc-C0f+0.5-standard-decoded-body-0.18-dress-region-0.14-darker-than-body-white-garment-palette-not-transferred-pass1", "2026-09-07T07:05:00Z"),
        fact("finding:trellis#diff_gaussian_rasterization@missing-in-edit-env-splat-render-unavailable-vertex-colour-proxy-only", "2026-09-07T07:05:00Z"),
        fact("finding:garments#photo-station@laid-flat-neckline-right-90ccw-holds-for-ids-0-1-2", "2026-09-07T07:05:00Z"),
        fact("design:dress-on#batch1@3-rows-diagonal-female-p1xg0-female-p50xg1-male-p50xg2-rank3-next-garment-launched-2026-09-07T07:05:00Z", "2026-09-07T07:05:00Z", basis="assigned"),
        fact("doctrine:mitsuba#status@stays-it-is-cpp-python-binding-is-the-glue", "2026-09-07T07:05:00Z", basis="assigned"),
        fact("doctrine:language#target@entire-pipeline-under-ggml-voxhammer.cpp-python-steps-are-reference-impls-behind-file-contracts", "2026-09-07T07:05:00Z", basis="assigned"),
    ]

    # Round 4, 2026-09-07T08:10:00Z: PLY swapped for USD at every boundary; batch 1 relaunched.
    F += [
        fact("finding:usd-chain#end-to-end@body-mask-cube-render-mesh.usda-voxels.usda-voxels_delete.usda-edit-171s-10.6GB-no-ply-on-path", "2026-09-07T08:10:00Z"),
        fact("finding:dll#pxr-vs-bpy@mutually-exclusive-in-one-process-blender-bundles-usd-tbb-stub-bpy-never-import", "2026-09-07T08:10:00Z"),
        fact("design:dress-on#voxhammer-usd-shims@run_3d_rendering-voxelize_mesh-utils3d-ply-io-o3d-pointcloud-io-load_trimesh-extract_features-run_3d_editing-rebound-upstream-untouched", "2026-09-07T08:10:00Z", basis="assigned"),
        fact("design:dress-on#remaining-ply@3dgs-gaussian-splat-container-only-future-UsdGeom.Points-with-primvars", "2026-09-07T08:10:00Z", basis="assigned"),
        fact("doctrine:voxhammer.cpp#usd-adapter@beside-datasource-flow-openusd-fabric-godot-hydra-blender-unity-adapters-github-v-sekai-fabric-datasource-flow", "2026-09-07T08:10:00Z", basis="assigned"),
        fact("finding:garments#station-orientation@neckline-right-all-of-0-1-2-rotate-90ccw-unconditional-aspect-heuristic-wrong", "2026-09-07T08:10:00Z"),
        fact("finding:batch1#attempt1@failed-at-renderer-edited-under-running-batch-rank3-pass1-preserved-resumable-relaunch-2026-09-07T08:10:00Z", "2026-09-07T08:10:00Z"),
        fact("design:dress-on#judge-env@editscore-qwen3-vl-8b-installed-tools/judge_editscore.py-frozen-tuple-per-row", "2026-09-07T08:10:00Z", basis="assigned"),
    ]

    # voxhammer.cpp scoping, 2026-09-07T08:50:00Z (operator directives while batch 1 runs).
    F += [
        fact("blocklist:tinyusdz#status@banned-operator-2026-09-07", "2026-09-07T08:50:00Z", basis="assigned"),
        fact("blocklist:hand-written-openusd-parsers#status@banned-operator-2026-09-07", "2026-09-07T08:50:00Z", basis="assigned"),
        fact("doctrine:openusd#implementation@3-interactor/datasource-flow-github-v-sekai-fabric-datasource-flow-builds-openusd-26.05-exposes-flat-c-abi-flow/ports-hosts-never-see-pxr", "2026-09-07T08:50:00Z", basis="assigned"),
        fact("design:voxhammer.cpp#shape@host-binding-flow-port-ring-for-usd-plus-pixal3d.cpp-ggml-components-single-file-lib-capi-examples-tests-convert_to_gguf", "2026-09-07T08:50:00Z", basis="assigned"),
        fact("finding:pixal3d.cpp#conditioner@dinov3-vit-l16-rope-not-trellis1-dinov2-vit-l14-registers-vit-core-reusable-embedding-differs", "2026-09-07T08:50:00Z"),
        fact("finding:pixal3d.cpp#ports-present@ss_flow-ss_dec-slat_flow-slat_flow_tex-shape_enc-shape_dec-tex_dec-trellis2-variants-of-voxhammer-trellis1-lineage", "2026-09-07T08:50:00Z"),
        fact("finding:workspace#cpp-usd-readers@none-outside-datasource-flow-ufbx-to-openusd-is-python-usd-viewer-is-js", "2026-09-07T08:50:00Z"),
        fact("design:voxhammer.cpp#first-increment@walking-skeleton-usd-in-voxelize-cube-mask-preserve-set-seeded-hammersley-bit-compare-vs-python-smoke2-then-model-ports", "2026-09-07T08:50:00Z", basis="assigned"),
    ]

    # Round 5, 2026-09-07T09:30:00Z: judge proven, voxhammer.cpp skeleton, flow ring API.
    F += [
        fact("finding:judge#proven-3090@editscore-qwen3-vl-8b-loads-43s-16.3GiB-evaluate-16s-peak-17.0GiB-smoke-scores-2.8-of-25", "2026-09-07T09:30:00Z"),
        fact("repo:3-interactor/voxhammer.cpp#state@walking-skeleton-written-normalize-voxelize-sat-mask-27-sample-preserve-hammersley-mt19937-fixture-tests-unbuilt-local-only", "2026-09-07T09:30:00Z"),
        fact("repo:3-interactor/voxhammer.cpp#placement@local-not-in-manifest-not-pushed-await-operator", "2026-09-07T09:30:00Z", basis="assigned"),
        fact("finding:datasource-flow#ring-usd-entry@idtx_core_import_scene_from_usd-and-idtx_core_export_avatar_to_usd-mesh-cube-sphere-no-points-prim", "2026-09-07T09:30:00Z"),
        fact("design:voxhammer.cpp#core-addition-needed@datasource-flow-IDTX_NODE_POINTS-import-export-for-voxel-sets", "2026-09-07T09:30:00Z", basis="assigned"),
        fact("bench:magi#datasource-flow-openusd-26.05-build@started-2026-09-07T09:30:00Z-scons-template_release-vcpkg-prerun", "2026-09-07T09:30:00Z"),
        fact("design:voxhammer.cpp#toolchain@pixi-cmake-ninja-vs2022_win-64-no-system-cmake", "2026-09-07T09:30:00Z", basis="assigned"),
    ]

    # Round 6, 2026-09-07T11:10:00Z: voxhammer.cpp skeleton bit-exact; Lean library oracle linked.
    F += [
        fact("finding:voxhammer.cpp#skeleton@bit-exact-vs-python-smoke2-voxels-3378-delete-3068-cameras-max-dc2w-0-hammersley-offset-15-decimals", "2026-09-07T11:10:00Z"),
        fact("design:voxhammer.cpp#oracle@lean4-specs-lean/VoxHammer-grid-frame-mask-hammersley-constants-proof-pinned-artifacts-vh_constants.h", "2026-09-07T11:10:00Z", basis="assigned"),
        fact("design:voxhammer.cpp#oracle-binding@sigs-dispatch-table-not-bus-operator-2026-09-07-VoxHammer_oracle.dll-static-lean-runtime-3.7MB-system-dlls-only", "2026-09-07T11:10:00Z", basis="assigned"),
        fact("finding:voxhammer.cpp#self-check@8-claims-failure-mask-0x0-inside-binary-mask-rule-executed-by-proven-lean-function", "2026-09-07T11:10:00Z"),
        fact("finding:plausible-witness-dag#oracle-caught-decorative-control@grid-off-by-one-twin-absorbed-by-floor-replaced-with-strict-interior-claim", "2026-09-07T11:10:00Z"),
        fact("doctrine:lean4#usage@specs-plus-plausible-witness-dag-domain-package-soundness-theorem-completeness-exhaustive-readback-falsifiability-broken-twin-must-be-found", "2026-09-07T11:10:00Z", basis="assigned"),
        fact("finding:lake#init-symbol@package-prefixed-initialize_VoxHammer_VoxHammer-read-from-coff-symbol-table", "2026-09-07T11:10:00Z"),
        fact("finding:leanc#link@lld-prefers-libX.dll.a-over-libX.a-pass-static-archives-by-path-plus-libleanmanifest-for-lean_get_githash", "2026-09-07T11:10:00Z"),
        fact("finding:generate_stubs#msvc@emits-__attribute__-weak-compat-header-defines-it-away-under-_MSC_VER", "2026-09-07T11:10:00Z"),
        fact("finding:host#responsiveness@osquery-ryzen-3800x-16t-128GB-103GB-free-cpu-100pct-from-openusd-scons-j8-gpu4090-9pct-disk-idle-build-tree-lowered-to-BelowNormal", "2026-09-07T11:10:00Z"),
        fact("finding:datasource-flow#build@needs-cmake-ninja-git-in-pixi-env-added-openusd-26.05-building-2026-09-07T11:10:00Z", "2026-09-07T11:10:00Z"),
    ]

    # Round 7, 2026-09-07T09:40:00Z: dress-on batch 1 shipped; rest-exact gate retracted and re-measured; USD ring bound.
    F += [
        fact("dataset:anny-dress-on-stage-train#published@hf-ifire/anny-dress-on-stage-train-public-3-rows-9-candidates-576-score-rows-9-judge-rows-46-files", "2026-09-07T09:40:00Z"),
        fact("dataset:anny-dress-on-stage-train#namespace@ifire-not-chibifire-token-is-fine-grained-scoped-to-ifire-only-org-create-403-move-or-rescope-is-operator-call", "2026-09-07T09:40:00Z"),
        fact("dataset:anny-dress-on-stage-train#viewer@configs-block-required-without-it-viewer-folds-five-tables-into-one-default-config-600-rows-48-features", "2026-09-07T09:40:00Z"),
        fact("dataset:anny-dress-on-stage-train#viewer_verified@dress_on-config-3-rows-candidates-x3-scores-x64-judge-x1-throwaway-deleted", "2026-09-07T09:40:00Z"),
        fact("finding:dress-on#rest_exact_2d_gate_retracted@voxhammer-frees-any-16cube-structure-cell-not-wholly-preserved-empty-space-not-preserved-2d-outside-box-depth-was-wrong-statement", "2026-09-07T09:40:00Z"),
        fact("design:dress-on#rest_exact_basis@far-preserved-voxels-within-one-voxel-of-output-vs-rank5-floor-tol-0.02-tools/voxel_controls.py", "2026-09-07T09:40:00Z", basis="assigned"),
        fact("bench:magi-4090#dress-on_rest_exact_female-p1@floor-1.000-rank1-1.000-rank3-0.994-trellis-regeneration-control-0.751-shifted-3-voxels-control-0.644", "2026-09-07T09:40:00Z"),
        fact("finding:dress-on#regeneration_control@voxhammer-cannot-run-with-empty-preserved-set-zero-element-reshape-trellis-alone-on-same-composite-is-the-control-45s", "2026-09-07T09:40:00Z"),
        fact("finding:dress-on#judge_control@editscore-ordered-rank1-gt-rank3-gt-rank5-on-all-3-rows-2.68-2.26-0-4.20-2.53-0-2.71-1.55-0-of-25", "2026-09-07T09:40:00Z"),
        fact("bench:magi-4090#voxhammer_pass_wall@175-190s-per-pass-28s-load-11-12GB-peak-4-passes-per-row-row-15-17min-19-passes-per-hour", "2026-09-07T09:40:00Z"),
        fact("finding:voxhammer#extract_features_deadlock@loader-thread-swallows-exception-main-blocks-on-queue-forever-30min-lost-stale-voxels.ply-driver-forwards-exception", "2026-09-07T09:40:00Z"),
        fact("finding:voxhammer.cpp#usd_ring_bound@flow-idtx_core.sigs-dispatch-table-IDTX_CORE_STATIC-voxels-3378-delete-3068-identical-key-sets-roundtrip-positions-indices-bit-identical", "2026-09-07T09:40:00Z"),
        fact("finding:datasource-flow#importer@unwelds-meshes-13718-to-82260-verts-and-drops-vertex-displayColor-on-import-colours-unchecked-in-vh_usd_check", "2026-09-07T09:40:00Z"),
        fact("finding:datasource-flow#plugInfo_typo@ResourcePath-resouces-flow/core/usd/plugin/idtx/resources/plugInfo.json-102-schema-warnings-on-every-stage-open", "2026-09-07T09:40:00Z"),
        fact("design:voxhammer.cpp#dinov2_port_plan@docs/dinov2-port-plan.md-pos_embed-replaces-rope-split-qkv-eps-1e-6-518-at-patch-14-grid_sample-mean-over-views", "2026-09-07T09:40:00Z", basis="assigned"),
        fact("plan:dress-on#batch2@full-grid-4-identities-x-3-garments-9-more-rows-running-2026-09-07T09:40:00Z-then-judge-rebuild-stage-republish", "2026-09-07T09:40:00Z", basis="assigned"),
    ]

    # Round 7b, 2026-09-07T13:30:00Z: the within-one-voxel gate retracted by its own control; exact-containment midpoint gate.
    F += [
        fact("finding:dress-on#rest_exact_within1_gate_retracted@regeneration-control-scored-1.000-within-one-voxel-on-female-p50-g0-column-cannot-separate-edit-from-regeneration-tool-exited-2", "2026-09-07T13:30:00Z"),
        fact("design:dress-on#rest_exact_gate@exact-far-containment-at-least-midpoint-of-rank5-floor-and-per-row-trellis-regeneration-control-regeneration-required-not-optional", "2026-09-07T13:30:00Z", basis="assigned"),
        fact("bench:magi-4090#dress-on_far_containment_6_rows@floor-0.990-0.993-edits-0.898-0.972-regeneration-0.320-0.726-shifted-0.226-0.228-thresholds-0.655-0.859", "2026-09-07T13:30:00Z"),
        fact("finding:dress-on#planted_leak_refused@rank1-far_containment-0.5-below-threshold-0.699-writer-exit-1", "2026-09-07T13:30:00Z"),
        fact("dataset:anny-dress-on-stage-train#batch2_progress@5-rows-staged-3-judged-6th-computed-batch-resumed-2026-09-07T13:30:00Z-6-rows-remaining", "2026-09-07T13:30:00Z"),
    ]

    # Round 8, 2026-09-07T15:40:00Z: full 4x3 grid shipped.
    F += [
        fact("dataset:anny-dress-on-stage-train#batch2_complete@12-rows-36-candidates-2304-score-rows-30-judge-rows-163-files-hf-ifire-read-back-12-rows", "2026-09-07T15:40:00Z"),
        fact("bench:magi-4090#dress-on_batch2_wall@6-rows-in-5404s-about-15-min-per-row-4-edits-plus-64-view-aov-x4-plus-regeneration-control", "2026-09-07T15:40:00Z"),
        fact("finding:dress-on#judge_control_failed_on_2_of_12@female-p50-g0-rank3-0.0-equals-rank5-and-male-p99-g2-rank3-3.75-above-rank1-3.32-judge-rows-omitted-judge_control-failed-order-recorded", "2026-09-07T15:40:00Z"),
        fact("finding:dress-on#judge_scores_of_25@rank1-2.68-4.20-rank3-0.0-3.79-rank5-always-0.0-editscore-qwen3-vl-8b-frozen-tuple", "2026-09-07T15:40:00Z"),
        fact("plan:dress-on#batch2@full-grid-4-identities-x-3-garments-9-more-rows-running-2026-09-07T09:40:00Z-then-judge-rebuild-stage-republish", "2026-09-07T09:40:00Z", "2026-09-07T15:40:00Z", basis="assigned", duration_override="closed-completed-shipped-round-8"),
    ]

    # Round 8b, 2026-09-07T16:10:00Z: dataset moved to the org with the operator's HF_TOKEN.
    F += [
        fact("dataset:anny-dress-on-stage-train#moved_to_org@chibifire/anny-dress-on-stage-train-via-operator-HF_TOKEN-user-scope-env-old-ifire-url-307-redirects-163-files", "2026-09-07T16:10:00Z"),
        fact("dataset:anny-dress-on-stage-train#namespace@ifire-not-chibifire-token-is-fine-grained-scoped-to-ifire-only-org-create-403-move-or-rescope-is-operator-call", "2026-09-07T09:40:00Z", "2026-09-07T16:10:00Z", duration_override="closed-resolved-operator-moved-repo-with-own-token"),
        fact("doctrine:hf-publish#token@bao-held-token-reaches-ifire-only-org-writes-need-operator-HF_TOKEN-from-windows-user-scope-not-inherited-by-earlier-shells", "2026-09-07T16:10:00Z", basis="assigned"),
    ]

    # Round 8c, 2026-09-07T16:40:00Z: org-capable HF token moved into bao; ifire-only token revoked.
    F += [
        fact("secret:hf#hf_token_moved_to_bao@agents/magi-16739d.agents.weftspun-field-hf_token-fine-grained-repo.write-chibifire-and-ifire-verified-by-sha256-and-whoami-windows-user-scope-var-removed", "2026-09-07T16:40:00Z"),
        fact("secret:hf#ifire_only_token@revoked-by-operator-removed-from-bao-row-version-6", "2026-09-07T16:40:00Z"),
        fact("doctrine:hf-publish#token@bao-held-token-reaches-ifire-only-org-writes-need-operator-HF_TOKEN-from-windows-user-scope-not-inherited-by-earlier-shells", "2026-09-07T16:10:00Z", "2026-09-07T16:40:00Z", basis="assigned", duration_override="closed-superseded-bao-token-now-org-capable"),
    ]

    # Round 9, 2026-09-07T18:20:00Z: request-for-discussion mirrored back into v-sekai-fabric; GitHub PAT in bao.
    F += [
        fact("repo:v-sekai-fabric/request-for-discussion#mirrored_back@from-V-Sekai-archive-15-branches-2-tags-main-79fe720-default-main-301-redirect-gone-manifest-entry-2-contract/manuals-weftspun-unchanged", "2026-09-07T18:20:00Z"),
        fact("secret:github#pat_in_bao@agents-row-field-github_token-fine-grained-user-fire-resource-owner-v-sekai-fabric-contents-administration-workflows-write-user-scope-env-vars-removed", "2026-09-07T18:20:00Z"),
        fact("finding:github-pat#permissions@org-repo-create-needs-Administration-branch-with-workflow-file-needs-Workflows-both-403-with-misleading-messages-three-token-rounds", "2026-09-07T18:20:00Z"),
        fact("doctrine:github-auth#broad-gcm-token@windows-credential-manager-holds-classic-repo-scope-token-for-fire-41-orgs-not-used-org-scoped-pat-preferred", "2026-09-07T18:20:00Z", basis="assigned"),
    ]

    # Round 10, 2026-09-07T13:50:00Z: the RFD bankruptcy undone; Pages restored.
    F += [
        fact("repo:v-sekai-fabric/request-for-discussion#bankruptcy_undone@PR-1-merged-c5af22b8-reverts-ce1a5212-301-rfd-dirs-register-133-allocated-40-deleted-0-problems-six-retired-rows-moved-dead-1175-redirect-dir-dropped-litert-feature-dropped", "2026-09-07T13:50:00Z"),
        fact("finding:request-for-discussion#register_at_ce1a5212_parent@did-not-parse-duplicate-prim-Deleted-so-the-undo-keeps-the-bankruptcy-single-scope-fix", "2026-09-07T13:50:00Z"),
        fact("finding:check_goal_manifests#self_test@IndexError-when-org-has-no-archived-repo-hidden-by-pull-request-only-job-now-plants-a-victim", "2026-09-07T13:50:00Z"),
        fact("repo:v-sekai-fabric/request-for-discussion#pages@restored-from-gh-pages-root-legacy-build-HTTP-200-frozen-at-2026-09-05-deploy-publish-workflow-still-dropped", "2026-09-07T13:50:00Z"),
    ]

    # Round 11, 2026-09-07T14:30:00Z: branch backlog integrated and pruned; Publish workflow restored; git made non-interactive.
    F += [
        fact("repo:v-sekai-fabric/request-for-discussion#backlog_integrated@PR-2-merged-3d6f25c6-12-branches-folded-5-conflicts-resolved-and-stated-13-branches-deleted-remaining-main-gh-pages-drop-book-compile", "2026-09-07T14:30:00Z"),
        fact("repo:v-sekai-fabric/request-for-discussion#drop-book-compile@kept-unmerged-head-276e3832-deletes-publish.yml-and-render-scripts-contradicts-restore-directive-operator-call", "2026-09-07T14:30:00Z", basis="assigned"),
        fact("repo:v-sekai-fabric/request-for-discussion#publish_workflow@restored-verbatim-d1c8a73c-site-rebuilds-on-push-to-main-again", "2026-09-07T14:30:00Z"),
        fact("repo:v-sekai-fabric/request-for-discussion#ci@eleven-jobs-back-rfd-numbers-serials-structure-changed-site-pages-comment-ladder-trope-density-pr-description-canary-all-green-on-PR-2", "2026-09-07T14:30:00Z"),
        fact("doctrine:git-auth#non-interactive@credential.helper-empty-then-bao-script-reads-github_token-from-agents-row-GIT_TERMINAL_PROMPT-0-user-scope-GCM-helper-selector-off-pushes-use-x-access-token-url", "2026-09-07T14:30:00Z", basis="assigned"),
        fact("finding:git-windows#hang@system-gitconfig-helper-selector-runs-before-any-global-helper-and-opens-a-GUI-prompt-an-empty-global-entry-resets-the-chain", "2026-09-07T14:30:00Z"),
    ]

    # Round 12, 2026-09-07T14:50:00Z: credential blast radius audited; peer cert roles revoked by the operator.
    F += [
        fact("security:bao#peer_cert_roles_revoked@herd-anchor-hero-sidekick-deleted-by-operator-with-root-token-over-magi-mtls-all-cert-login-leases-revoked-remaining-roles-admin-v2-fdb-cluster-magi-16739d-vast-buyer", "2026-09-07T14:50:00Z", basis="observed"),
        fact("security:bao#agents_rows_read_scope@agents-rw-policy-unchanged-only-magi-cert-can-log-in-among-agents-peer-rows-remain-as-data", "2026-09-07T14:50:00Z"),
        fact("security:github-pat#effective_write_reach@v-sekai-fabric-only-blob-write-probe-refused-on-fire-and-V-Sekai-repo-listing-push-field-mirrors-user-rights-not-token-grant", "2026-09-07T14:50:00Z"),
        fact("security:audit#credential_stores@bash-history-line-with-first-revoked-pat-scrubbed-no-env-vars-no-gh-keyring-no-gcm-entry-no-scratch-files-transcript-holds-only-the-revoked-ifire-hf-token", "2026-09-07T14:50:00Z"),
        fact("peer:herd#role@parked-former-coordinator-succeeded-by-magi", "2026-09-07T05:00:00Z", "2026-09-07T14:50:00Z", basis="assigned", duration_override="closed-cert-role-revoked-2026-09-07"),
    ]

    F += [
        fact("security:bao#vast-buyer_revoked@leftover-cert-role-of-the-archived-vast-broker-deleted-by-operator-remaining-roles-admin-v2-fdb-cluster-magi-16739d", "2026-09-07T15:05:00Z"),
    ]

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
        fact("finding:g1_mjlab_obs_dim_actor@99", V),
        fact("finding:g1_mjlab_obs_dim_critic@111", V),
        fact("finding:g1_mjlab_smoke_300it_1024env_wall_s@341", V),
        fact("finding:g1_mjlab_smoke_rom_clearance@0.7986_control_shifted30deg@0.7562", V),
        fact("finding:g1_rom_envelope_binding@e3_on_all_8", V),
        fact("pin:g1_sim_to_real_host@wsl2-ubuntu-26.04-pixi-0.80", V),
        fact("pin:g1_rom_envelope#sha256_prefix@08205882", V),
    ]

    # data flows
    F += [
        fact(f"hf:chibifire/starforged-std-3001-appendix-e#read_by@{PEER}", V),
        fact(f"hf:chibifire/motionbricks-g1-mujoco-policies#published_by@{PEER}",
             HANDOFF, basis="assigned"),
        # Operator retracted the "variants -> github chibifire-characters, avoid
        # HF" rule 2026-09-07 03:19Z. Rows keep their original assertion time
        # (2026-09-06T20:35Z, when the rule was first surfaced this session)
        # rather than V=now() -- otherwise the closure end sits BEFORE start
        # on every rebuild and produces a negative-duration nonsense row.
        fact(f"github:chibifire-characters#variant_store_for@{PEER}",
             "2026-09-06T20:35:00Z", "2026-09-07T03:19:00Z",
             basis="assigned",
             duration_override="closed-retracted-operator-rescinded-github-variant-store-rule"),
        fact("github:chibifire-characters#constraint@no-git-lfs",
             "2026-09-06T20:35:00Z", "2026-09-07T03:19:00Z",
             basis="assigned",
             duration_override="closed-retracted-operator-rescinded-github-variant-store-rule"),
        fact("doctrine:storage-triage#variant_store@RETRACTED-operator-2026-09-07-await-successor-rule",
             "2026-09-07T03:19:00Z", basis="assigned"),
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

    # CoACD cost-driver result, n=57 bones, threshold=0.4
    F += [
        fact("bench:magi#coacd_cost_corr_triangles@0.111-n57-threshold0.4", V),
        fact("bench:magi#coacd_cost_corr_hullcount@0.850-n57-threshold0.4", V),
        fact("bench:magi#coacd_cost_corr_triangles_vs_hullcount@0.072-decoupled", V),
        fact("bench:magi#coacd_ms_per_triangle_spread@0.30-to-166.1-545x", V),
        fact("bench:magi#coacd_full_decomp@78-bones-176-hulls-2.3-mean", V),
        fact("bench:magi#coacd_hullset_size_on_disk@589KB-commits-without-lfs", V),
        fact("doctrine:coacd#cost_driver@hull-count-not-triangles-cap-via-max_convex_hull",
             V, basis="assigned"),
        fact("doctrine:coacd#decimation@does-not-reduce-cost-drops-coverage-0.999-to-0.656",
             V, basis="assigned"),
    ]

    # pre-park eviction state
    F += [
        fact("repo:3-interactor/motion-bricks-cpp#local_commit@7e880cf", V),
        fact("repo:3-interactor/motion-bricks-cpp#local_commit_prior@759f0bd", V),
        fact("repo:3-interactor/motion-bricks-cpp#push_status@unpushed-awaiting-operator-auth", V),
        fact("repo:3-interactor/motion-bricks-cpp#g1_stl_assets@gitignored-redownloadable-from-menagerie", V),
        fact(f"{PEER}#park_status@pending-operator-confirmation-on-own-channel", V),
        fact("repo:3-interactor/motion-bricks-cpp#pushed_branch@magi/pre-park-snapshot", V),
        fact("repo:3-interactor/motion-bricks-cpp#pushed_remote@v-sekai-fabric/motion-bricks.cpp", V),
        fact(f"{PEER}#eviction_state@complete-findings-in-rebac-code-pushed-to-wip-branch", V),

        # --- dress-on (task #176) design as it stands 2026-09-06T21:xxZ ---
        # Two pixi envs per one-pixi-env-per-usage rule (see MEMORY).
        fact("design:dress-on#env_edit@voxhammer-upstream/pixi.toml-default",
             "2026-09-06T21:00:00Z", TRELLIS_OBLITERATED,
             basis="assigned", duration_override="closed-retracted-blocklisted-model"),
        fact("design:dress-on#env_matting@voxhammer-upstream/pixi.toml-matting", V, basis="assigned"),
        fact("design:dress-on#env_edit_torch@2.4.0+cu118-python3.11",
             "2026-09-06T21:00:00Z", TRELLIS_OBLITERATED,
             basis="observed",
             duration_override="closed-retracted-model-independent-pin-may-persist-into-Pixal3D-env"),
        fact("design:dress-on#env_matting_torch@2.5.1+cu124-python3.11", V, basis="observed"),
        fact("design:dress-on#matting_model@ZhengPeng7/BiRefNet_HR-matting-MIT-2048sq", V, basis="assigned"),
        fact("design:dress-on#segmenter@rf-detr-segmentation-head-CC-BY-4-clothing-47-cats", V, basis="assigned"),
        # Shims: each raises rather than no-ops -- silent-pass is the failure mode we keep catching.
        fact("design:dress-on#shim_nvdiffrast@raises-on-call-not-registered-with-real-toolkit", V, basis="observed"),
        fact("design:dress-on#shim_kaolin@check_tensor-shape-assertion-only-verified-falsifiable", V, basis="observed"),
        fact("design:dress-on#shim_rembg@refuses-alpha-fabrication-provider-must-be-registered", V, basis="observed"),
        # Bus intent: matting env serves BiRefNet as an interactor via contract-bus,
        # so weights load once rather than per call. Roundtrip proof pending -- see status below.
        fact("design:dress-on#matting_transport_intent@iceoryx2-bus-weft_harness-persistent-interactor", V, basis="assigned"),
        fact("design:dress-on#matting_transport_fallback@localhost-socket-if-bus-blocked", V, basis="assigned"),
        # Model provider precedence (edit env picks alpha from highest-trust source):
        fact("design:dress-on#alpha_provider_order@sidecar-png_then_rgba_alpha_then_backdrop_threshold", V, basis="assigned"),
        # First-pass scope constraint from operator, carried through the design:
        fact("design:dress-on#scope_pass1@silhouette-and-palette-only-no-logos-await-#170", V, basis="assigned"),

        # --- iceoryx2 bus status ---
        fact("bench:magi#iceoryx2_wheel_cp311@abi3-loads-despite-pypi-tags-cp38-cp314-only", V, basis="observed"),
        fact("bench:magi#iceoryx2_windows_shm_dir@C:/Temp/iceoryx2/shm-required-else-NodeCreationFailure", V, basis="observed"),
        fact("bench:magi#iceoryx2_python_windows_proof@pending-node-creates-but-client-blocks-on-reply", V, basis="observed"),
        fact("doctrine:bus#python_windows@no-recorded-proof-in-repo-magi-is-first-attempt", V, basis="assigned"),

        # --- VoxHammer env work committed to git branches, safe ---
        fact("repo:3-interactor/voxhammer-upstream#branch@magi/voxhammer-windows-env-drafted-uncommitted",
             "2026-09-06T21:00:00Z", TRELLIS_OBLITERATED,
             basis="observed", duration_override="closed-retracted-blocklisted-model"),
        fact("repo:3-interactor/voxhammer-upstream#peak_vram_step4@16773MiB-of-24564-headroom-7.8GB", V, basis="observed"),
        fact("bench:magi-4090#voxhammer_step1_2_3_wall@200s-first-run-then-cached",
             "2026-09-06T21:00:00Z", TRELLIS_OBLITERATED,
             basis="observed", duration_override="closed-retracted-blocklisted-model"),
        fact("bench:magi-4090#voxhammer_step4_blocker@rembg-shim-refuses-alpha-fixed-by-driver", V, basis="observed"),

        # The hull-collision benchmark ended WITHOUT producing a result.
        fact("bench:magi-4090#coacd_hull_variant_final_status@no-result-process-exited-at-23.8-of-24.5GiB", V),
        fact("bench:magi-4090#coacd_hull_variant_landed@baseline-and-cloth_only-only-hull-variants-absent", V),
        fact("doctrine:coacd#sweep_hardware_sizing@hull-collision-4096-envs-unmeasured-suspected-24GB-ceiling-bound", V, basis="assigned"),
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
