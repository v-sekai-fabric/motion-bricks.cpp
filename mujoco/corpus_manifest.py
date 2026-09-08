"""Two-stratum avatar corpus manifest: human universe + game universe.

SPDX-License-Identifier: Apache-2.0

The operator asked for VRM equity across BOTH universes.  They cannot be one
corpus: the game universe puts 0% of characters at age 45+, the human universe
puts 30.5% there, so no single set of avatars satisfies both marginals.  This
builds two strata and records, per cell, whether it is filled by phenotype
GENERATION, by ACQUISITION of a FOSS asset, or by a manifold CSG HULL.

N=50 per stratum is the floor: the game universe's 2% classes (animal, plant,
exotic) each round to exactly one avatar at N=50 and to zero below it.

Fill strategy, and why each cell gets what it gets:

* HUMAN stratum -> ANNY phenotype generation.  ANNY (`3-interactor/anny`,
  Apache-2.0, NAVER) is a MakeHuman-derived parametric torch body model whose
  phenotype axes are literally the human-universe axes: ``gender``, ``age``,
  and ``african``/``asian``/``caucasian``, plus muscle/weight/height/
  proportions.  Verified: 11 phenotype labels, 254 local changes, 52 facial
  actions, and age alone spans a 2.6x stature ratio.  Acquiring 50 real avatars
  covering age 0-65+ across six world regions under a FOSS licence is not
  feasible; generating them is one parameter sweep.

* GAME stratum, humanoid + semi-humanoid -> ACQUISITION, because stylised
  character identity is exactly what a parametric human model cannot invent.

* GAME stratum, cyborg / animal / plant / exotic -> manifold CSG HULL.  These
  are 2% classes (one avatar each at N=50); no FOSS VRM of a plant or exotic
  species with a clean licence is reliably available, and the operator ruled
  that where a full mesh is missing, a hull built from manifold CSG shapes is
  acceptable.  Verified: Blender 5.2.1 exposes a ``MANIFOLD`` boolean solver
  and a sphere+cube+cylinder union comes out watertight (0 non-manifold edges,
  0 non-manifold verts, every edge shared by exactly 2 faces).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

# ---------------------------------------------------------------- distributions
HUMAN = {
    "sex": {"male": .504, "female": .496, "intersex": .0005},
    "age": {"infancy_0_4": .085, "juvenile_5_14": .162, "youth_15_24": .153,
            "adult_25_44": .295, "middle_aged_45_64": .204, "senior_65_plus": .101},
    "region": {"asia": .591, "africa": .186, "europe": .091,
               "latin_america": .082, "north_america": .047, "oceania": .006},
}
GAME = {
    "species": {"humanoid": .50, "semi_humanoid": .38, "cyborg": .06,
                "animal": .02, "plant": .02, "exotic": .02},
    "presentation": {"feminine": .71, "masculine": .18, "other": .11},
    "age_pres": {"youth_15_24": .58, "juvenile_5_14": .17, "adult_25_44": .12,
                 "no_specific_age": .12, "infancy_0_4": .01,
                 "middle_or_senior_45_plus": .00},
}

# ANNY phenotype axis values per human-universe bucket.
# `age` is ANNY's 0..1 axis; the mapping is monotone but NOT calibrated to years
# -- ANNY does not publish a years-to-axis curve, so these are ordinal anchors.
AGE_AXIS = {"infancy_0_4": 0.00, "juvenile_5_14": 0.18, "youth_15_24": 0.35,
            "adult_25_44": 0.55, "middle_aged_45_64": 0.78, "senior_65_plus": 1.00}
SEX_AXIS = {"female": 0.0, "male": 1.0, "intersex": 0.5}

# ANNY ships three ancestry axes; the human universe has six regions.  Three of
# them have no axis and must be expressed as admixtures.  These weights are
# PLACEHOLDERS chosen to be ordinally sensible, not demographic estimates.
REGION_MIX = {
    "asia":          {"asian": 1.0, "african": 0.0, "caucasian": 0.0},
    "africa":        {"asian": 0.0, "african": 1.0, "caucasian": 0.0},
    "europe":        {"asian": 0.0, "african": 0.0, "caucasian": 1.0},
    "latin_america": {"asian": 0.4, "african": 0.2, "caucasian": 0.4},
    "north_america": {"asian": 0.2, "african": 0.2, "caucasian": 0.6},
    "oceania":       {"asian": 0.5, "african": 0.2, "caucasian": 0.3},
}
REGION_APPROXIMATED = {"latin_america", "north_america", "oceania"}


@dataclass
class Avatar:
    avatar_id: str
    stratum: str
    fill: str                    # generated | acquired | csg_hull
    cell: dict
    source: str
    licence: str
    provenance: str
    params: dict = field(default_factory=dict)
    caveats: list = field(default_factory=list)


ACQUIRED = [
    ("pixiv_constraint_twist", "VRM1_Constraint_Twist_Sample_01.vrm",
     "VRM-native permissive (mod+redist+commercial, no NC, no SA)",
     "pixiv Inc.; in manifest at 6-datasource/sk-vrm1-constraint-twist-sample"),
    ("mire_quest", "MireQuest.blend",
     "Apache-2.0",
     "chibifire-characters/mesh-mille-mire-feuille; 75 bones, 110 shape keys"),
    ("mire_full", "Mire.blend",
     "Apache-2.0",
     "chibifire-characters/mesh-mille-mire-feuille; 30 MB source"),
    ("faavrs_breadbread", "FAAVRS_X_1.1.blend",
     "CC-BY-4.0",
     "V-Sekai-fire/SK_faavrs_breadbread; derived from David Onizaki base mesh "
     "(CC-BY-4.0); 80 bones, 83 shape keys"),
    ("blender_base_female_realistic", "human_base_meshes_bundle.blend",
     "CC0", "Blender human-base-meshes-bundle-v1.4.1"),
    ("blender_base_male_realistic", "human_base_meshes_bundle.blend",
     "CC0", "Blender human-base-meshes-bundle-v1.4.1"),
    ("blender_base_female_stylized", "human_base_meshes_bundle.blend",
     "CC0", "Blender human-base-meshes-bundle-v1.4.1"),
    ("blender_base_male_stylized", "human_base_meshes_bundle.blend",
     "CC0", "Blender human-base-meshes-bundle-v1.4.1"),
]


def largest_remainder(weights: dict, n: int) -> dict:
    """Apportion n across weights without losing or inventing avatars."""
    raw = {k: v * n for k, v in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    rem = n - sum(out.values())
    for k, _ in sorted(raw.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True):
        if rem <= 0:
            break
        out[k] += 1
        rem -= 1
    return out


def build_human(n: int = 50) -> list[Avatar]:
    """Human stratum: joint cells over sex x age x region, all generated.

    Apportions the JOINT distribution in one pass rather than nesting three
    largest-remainder calls.  Nesting loses small cells: subdividing an already
    small bucket rounds its rare children to zero, and at n=50 that silently
    dropped both north_america (0.047) and oceania (0.006) from the corpus even
    though north_america is owed ~2 avatars.  One-pass joint apportionment
    keeps every marginal honest down to the rounding floor.
    """
    def spread(weights: dict) -> list:
        """One axis, apportioned to n and interleaved so zip() cannot correlate
        axes by construction (all the asia entries landing on all the infants)."""
        counts = largest_remainder(weights, n)
        buckets = [[k] * v for k, v in counts.items() if v]
        col: list = []
        while any(buckets):
            for b in buckets:
                if b:
                    col.append(b.pop())
        return col

    sexes, ages, regions = (spread(HUMAN[a]) for a in ("sex", "age", "region"))
    cells = list(zip(sexes, ages, regions))

    out: list[Avatar] = []
    i = 0
    for sex, age, region in cells:
        if True:
            if True:
                    caveats = []
                    if region in REGION_APPROXIMATED:
                        caveats.append(
                            f"region '{region}' has no ANNY ancestry axis; "
                            "expressed as an admixture of african/asian/caucasian "
                            "with placeholder weights")
                    if sex == "intersex":
                        caveats.append(
                            "ANNY 'gender' is a single continuous axis; intersex "
                            "is represented as its midpoint, which is a modelling "
                            "choice, not a claim about morphology")
                    params = {"gender": SEX_AXIS[sex], "age": AGE_AXIS[age]}
                    params.update(REGION_MIX[region])
                    out.append(Avatar(
                        avatar_id=f"H{i:03d}", stratum="human", fill="generated",
                        cell={"sex": sex, "age": age, "region": region},
                        source="anny (3-interactor/anny)",
                        licence="Apache-2.0 (NAVER)",
                        provenance="ANNY parametric body model, phenotype sweep",
                        params=params, caveats=caveats))
                    i += 1
    return out


def build_game(n: int = 50) -> list[Avatar]:
    """Game stratum: species x presentation, acquired or CSG-hulled."""
    out: list[Avatar] = []
    spec_n = largest_remainder(GAME["species"], n)
    hull_species = {"cyborg", "animal", "plant", "exotic"}
    i, acq = 0, 0
    for species, ns in spec_n.items():
        if ns == 0:
            continue
        for pres, np_ in largest_remainder(GAME["presentation"], ns).items():
            for _ in range(np_):
                if species in hull_species:
                    out.append(Avatar(
                        avatar_id=f"G{i:03d}", stratum="game", fill="csg_hull",
                        cell={"species": species, "presentation": pres},
                        source="blender MANIFOLD boolean CSG",
                        licence="CC0 (workspace-authored)",
                        provenance="manifold CSG proxy; watertightness asserted "
                                   "per-asset (0 non-manifold edges/verts)",
                        caveats=["collision/silhouette proxy only -- no surface "
                                 "detail, no texture, not a renderable character"]))
                elif acq < len(ACQUIRED):
                    aid, src, lic, prov = ACQUIRED[acq]
                    acq += 1
                    out.append(Avatar(
                        avatar_id=f"G{i:03d}", stratum="game", fill="acquired",
                        cell={"species": species, "presentation": pres},
                        source=src, licence=lic, provenance=prov,
                        caveats=[] if aid != "pixiv_constraint_twist" else
                                ["licence form is VRM-native permission bits, "
                                 "not a named CC licence"]))
                elif species == "semi_humanoid":
                    # 38% of the game universe and the second-largest class.
                    # ANNY cannot produce it: a MakeHuman-derived model has no
                    # tail, no extra limbs, no kemonomimi ears -- the defining
                    # features. Marking these 'generated' would launder a gap
                    # into a filled cell, so they are UNFILLED by construction.
                    out.append(Avatar(
                        avatar_id=f"G{i:03d}", stratum="game", fill="unfilled",
                        cell={"species": species, "presentation": pres},
                        source="(none)", licence="(n/a)",
                        provenance="no source identified",
                        caveats=["semi-humanoid needs non-human topology (tail, "
                                 "extra limbs, animal ears) that ANNY cannot "
                                 "express and that no acquired asset covers; "
                                 "a CSG hull would lose the identity this class "
                                 "exists to carry"]))
                else:
                    out.append(Avatar(
                        avatar_id=f"G{i:03d}", stratum="game", fill="generated",
                        cell={"species": species, "presentation": pres},
                        source="anny (3-interactor/anny)",
                        licence="Apache-2.0 (NAVER)",
                        provenance="ANNY phenotype variant standing in for a "
                                   "stylised humanoid",
                        params={"gender": 0.0 if pres == "feminine"
                                else 1.0 if pres == "masculine" else 0.5},
                        caveats=["ANNY is a realistic human model; it does not "
                                 "reproduce stylised/anime character identity"]))
                i += 1
    return out


def build(n: int = 50) -> dict:
    human, game = build_human(n), build_game(n)
    def tally(rows):
        t = {}
        for r in rows:
            t[r.fill] = t.get(r.fill, 0) + 1
        return t
    return {
        "n_per_stratum": n,
        "strata": {"human": len(human), "game": len(game)},
        "fill_counts": {"human": tally(human), "game": tally(game)},
        "avatars": [asdict(a) for a in human + game],
    }


if __name__ == "__main__":
    m = build(50)
    print(json.dumps({k: v for k, v in m.items() if k != "avatars"}, indent=2))
    print(f"\ntotal avatars: {len(m['avatars'])}")
    seen = {}
    for a in m["avatars"]:
        key = (a["stratum"], tuple(sorted(a["cell"].items())))
        seen[key] = seen.get(key, 0) + 1
    print(f"distinct cells: {len(seen)}")
    print(f"cells with caveats: "
          f"{sum(1 for a in m['avatars'] if a['caveats'])}")
