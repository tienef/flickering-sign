"""Writes brain/lands/island.json from the wild land's tiles and rules (P3, slice 3a).

    python deploy/probe/make_island.py

The island (owner, 2026-09-30): land in the sea, the wild land's biomes and chains inside, fresh water
only in a pond, a few springs and marsh pools (the sea is salt), 2-3 caves in the hills (a natural roof),
the walled garden and its heavy gate kept (the first act that needs two), no shelter and no shelf;
mist banks over parts of the interior; minds wake scattered along one stretch of the shore (3m). Later slices add to it here.
"""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
base = json.loads((ROOT / "brain" / "lands" / "wild.json").read_text(encoding="utf-8"))
w = copy.deepcopy(base)
w["_doc"] = ("The island (P3, owner 2026-09-30): the wild land's tiles, creatures and chains on an island in a "
             "salt sea: beaches, meadow, forest, hills with a few caves (a natural roof), a marsh; fresh water only "
             "in a pond, a few springs and marsh pools; the walled garden whose heavy gate needs two; mist banks over "
             "parts of the interior that lift for good when someone comes near; no shelter, no shelf of papers. "
             "Minds wake scattered along one stretch of the shore, sharing its springs. None of it is told to the mind.")
w["name"] = "island"
w["land_word"] = "island"
w["size"] = [60, 60]
w["seed_note"] = "Generated per seed by Land._generate_island (deploy/probe/make_island.py writes this file)."
w["day_ticks"] = 180                      # the brain's day (owner: aligned; ~125 awake ticks, a night of ~55)
w["night_from"] = 0.7
w["weather"] = dict(base["weather"], rain_every=[240, 720], rain_lasts=[30, 90])   # the wild's, on a 180-tick day
w["asleep_feels"] = ["dark", "chill"]     # a sleeping body still feels the dark (melatonin) and the night's chill
# a year of 4 x 8 days; never named to the mind: it feels the air, sees bushes stay bare
# a generous start that thins (owner, 2026-09-30, B): minds wake in spring, when plants and mussels come back twice
# as fast (grow 0.5), then summer 0.8, autumn 1.6, winter 4: the world is easiest while the minds know least
w["seasons"] = {
    "days": 8, "order": ["spring", "summer", "autumn", "winter"],
    "grow_affects": ["sown soil", "sprouts", "bare bush", "bare apple tree", "stump", "sapling", "bare rocks"],
    "_doc": "per season: night_from (a night of 24-40% of the day), grow (x a plant's regrowth time), chill "
            "and warmth (x the night's chill outside, x the day's warmth), rain_every (x the dry spell), air (the "
            "phrase with the time of day)",
    "each": {
        "spring": {"night_from": 0.70, "grow": 0.5, "chill": 0.7, "warmth": 1.0, "rain_every": 0.8,
                   "air": "The air is fresh and green things are coming up."},
        "summer": {"night_from": 0.76, "grow": 0.8, "chill": 0.3, "warmth": 1.5, "rain_every": 1.5,
                   "air": "The air is warm."},
        "autumn": {"night_from": 0.68, "grow": 1.6, "chill": 1.0, "warmth": 0.8, "rain_every": 0.7,
                   "air": "The air is cool and the leaves are turning."},
        "winter": {"night_from": 0.60, "grow": 4.0, "chill": 2.0, "warmth": 0.3, "rain_every": 0.8,
                   "air": "The air is cold and raw."},
    },
}

t = w["tiles"]
t["~"] = {"name": "sea", "solid": True, "desc": "the sea, grey-green and restless"}
t["w"] = {"name": "water", "solid": True, "desc": "clear fresh water"}
t["j"] = {"name": "spring", "solid": True, "desc": "a spring: clear water bubbling up between stones"}
t["K"] = {"name": "cave", "roof": True, "desc": "a dry rock floor under the low roof of a cave"}
t.pop("S", None)                                                   # no shelf of papers on the island

w["generate"] = {
    "island": {"radius": 25, "coast_wobble": [0.03, 0.08], "beach": 1.6, "beach_stones": 0.03},
    "biomes": {
        "centres": 12, "spacing": 7, "centre": "meadow",
        "types": {"meadow": 4, "forest": 4, "hills": 3, "marsh": 1},
        "fill": {
            "meadow": dict(base["generate"]["biomes"]["fill"]["meadow"], B=0.03),
            "forest": dict(base["generate"]["biomes"]["fill"]["forest"], B=0.06, A=0.006),
            "hills": base["generate"]["biomes"]["fill"]["hills"],
            "marsh": {"w": 0.08, "r": 0.2, "c": 0.06, ",": 0.05},
        },
    },
    "pond": {"radius": [2, 3], "tile": "w", "rim": ["s", "c", "r", "."], "min_distance": 6},
    # isl2, isl3: bodies woke 15-19 steps from fresh water, so 8 coast springs round the island; 3m (M2): isl4-isl8, every
    # spot had its own spring and its own mussels and no mind ever saw another: now fewer, each between two spots
    "springs": {"count": 3, "biomes": ["hills", "forest", "meadow"], "apart": 12, "tile": "j",
                "coast": {"count": 3, "inland": [1, 3], "between_spawn": True}},
    # 3m (M1): the waking spots on one stretch of the shore, neighbours this many steps apart along it (out of sight
    # at 5 and of a voice at 8); before, the golden angle spread them round the island (35-65 steps apart)
    # 3m P2 (isl10: wob woke in a one-tile strip walled by the mussel rocks; seeds 1-4: walled spots): at least 2
    # free sides, fresh water within 12 steps on foot, the next spot within 2x its distance as the crow flies;
    # where the land fails it, the generator opens a way (the fewest trees and stones cleared)
    "spawn": {"spacing": 12, "open": 2, "water_within": 12, "neighbour_x": 2},
    "shore_food": {"tile": "P", "near_spawn": 8, "each": 2, "within": 3, "more": 8},
    "caves": {"count": 3, "apart": 10, "tile": "K"},
    "orchard": {"size": 7, "apple_trees": 4, "min_distance": 8, "max_distance": 18},
    "flickering_stones": 1,
}
w["fog"] = {"banks": 3, "radius": [4, 7], "near": 2, "lift": 7,
            "_doc": "mist banks over the interior (count, radius); a bank lifts within `lift` of a mind that comes "
                    "within `near` of it, for good"}

# food enough if found (3j calculation: 4 minds need ~12 berry picks a day; the wild's density gave ~7)
for g in w["grow"]:
    if g["from"] == "bare bush":
        g["after"] = 300
# the sea's rules: salt water, fishing from the shore; the spring's: water to cup or fill a bowl
take = [{"target": "sea", "text": "I cupped the sea water; it is salt and runs through my fingers."},
        {"target": "spring", "text": "I cupped the spring water; it is cold and runs through my fingers."}]
use = [{"held": "fishing line", "target": "sea", "chance": 0.3, "gives": ["fish"],
        "text": "I cast the line into the sea; something tugged, and I pulled out a fish!",
        "text_fail": "I cast the line into the sea and waited. Nothing bit.", "seen": "caught a fish",
        "seen_fail": "fished"},
       {"held": "bowl", "target": "sea", "consume": True, "gives": ["bowl of sea water"],
        "text": "I filled the bowl with sea water.", "seen": "filled a bowl"},
       {"held": "bowl", "target": "spring", "consume": True, "gives": ["bowl of water"],
        "text": "I filled the bowl at the spring.", "seen": "filled a bowl"}]
w["take"] = take + [r for r in base["take"] if r.get("target") != "shelf"]
w["use"] = use + [r for r in base["use"] if r.get("target") != "shelf"]

# verbs the island adds (the other lands keep theirs), and what can be drunk (3c)
w["verbs"] = {
    "drink": "drink from what is in front of me, or what I hold",
    "give": "give what I hold to the one in front of me",
}
# bodies pass each other (3m P1, owner 2026-09-30; isl10: sut and wob blocked each other three days in a one-tile
# strip): a move onto another's tile squeezes past; give and strike reach the one in front or on the same tile
w["bodies_block"] = False
w["drink"] = [
    {"held": "bowl of water", "amount": 0.35, "leaves": "bowl", "text": "I drank the water from the bowl.",
     "seen": "drank from a bowl"},
    {"held": "bowl of sea water", "amount": -0.1, "leaves": "bowl",
     "text": "I drank the sea water from the bowl. It is salt; my mouth is drier than before.",
     "seen": "drank from a bowl"},
    {"target": "water", "amount": 0.25, "text": "I drank the clear water from my cupped hands.", "seen": "drank water"},
    {"target": "spring", "amount": 0.3, "text": "I drank from the spring; the water is cold and clean.",
     "seen": "drank from the spring"},
    {"target": "sea", "amount": -0.1, "text": "I drank the sea water. It is salt; my mouth is drier than before.",
     "seen": "drank sea water"},
]

c = w["creatures"]
for kind, n in {"sheep": 5, "cow": 3, "chicken": 5, "rabbit": 6, "deer": 3, "frog": 4}.items():
    c[kind]["count"] = n
c["rabbit"]["biomes"] = ["meadow"]
c["frog"]["biomes"] = ["marsh"]

# strike, pain and weakness, hunting (3e; owner: minds never die, a creature struck enough does and leaves meat)
w["verbs"]["strike"] = "strike what is in front of me (with what I hold)"
w["strike"] = {
    "base": 0.25, "weakens": 0.8, "self_hurt": 0.03,
    "weapons": {"stick": 0.1, "stone": 0.1, "log": 0.15, "sharp stone": 0.15, "copper knife": 0.2, "stone axe": 0.25},
    "_doc": "a blow's strength = (base + the weapon's) x (1 - weak/2); a struck mind's pain rises by it (the `hurt` "
            "event, which wakes a sleeper) and its body's weakness by x`weakens`; striking a solid thing hurts the hand",
}
w["body"] = {
    "weak_heal": 0.004, "weak_heal_asleep": 0.008, "stumble": 0.5, "weak_carry": 5,
    "weak_felt": [[0.3, "My body is weak and bruised."], [0.7, "My body is badly hurt; every move is hard."]],
    "_doc": "weakness heals a tick (awake / asleep); a move fails with p = weak x stumble; carry_max - weak x "
            "weak_carry; the phrase is part of what the body senses",
}
# toughness (the sum of blows that kills), dodging, what is left, breeding back (every: 1 in N ticks a chance)
for kind, (tough, dodge, left, every) in {
        "sheep": (0.6, 0.0, ["raw meat", "wool"], 500), "shorn sheep": (0.6, 0.0, ["raw meat"], 0),
        "cow": (1.2, 0.0, ["raw meat", "raw meat"], 900), "chicken": (0.25, 0.3, ["raw meat"], 400),
        "rabbit": (0.25, 0.6, ["raw meat"], 400), "deer": (0.7, 0.5, ["raw meat", "raw meat"], 800),
        "frog": (0.1, 0.4, [], 600)}.items():
    c[kind].update(tough=tough, dodge=dodge, leaves=left)
    if every:
        c[kind]["breed"] = {"every": every, "max": c[kind]["count"]}
c["shorn sheep"]["family"] = "sheep"
# build (3f): what is carried -> a structure in front, on open ground; the mind is not told the recipes
w["verbs"]["build"] = "build something in front of me from what I carry"
t["L"] = {"name": "lean-to", "roof": True, "desc": "a lean-to of sticks: a low roof to shelter under"}
t["H"] = {"name": "hut", "roof": True, "desc": "a small hut of planks, with a roof and room for one or two"}
t["C"] = {"name": "cairn", "solid": True, "desc": "a cairn: stones piled up by someone's hands"}
t["X"] = {"name": "store", "solid": True, "store": 12,
          "desc": "a store of stones and planks with a lid, to keep things in"}
w["build"] = [
    {"needs": {"stone": 2, "plank": 2}, "becomes": "store",
     "text": "I set the stones and fitted the planks over them: a store with a lid, to keep things in.",
     "seen": "built a store"},
    {"needs": {"plank": 6}, "becomes": "hut",
     "text": "I stood the planks up and laid the last ones across: a small hut with a roof.", "seen": "built a hut"},
    {"needs": {"stick": 4}, "becomes": "lean-to",
     "text": "I leaned the sticks together and wove them tight: a lean-to, a low roof to shelter under.",
     "seen": "built a lean-to"},
    {"needs": {"stone": 3}, "becomes": "cairn", "text": "I piled the stones one on another: a cairn.",
     "seen": "piled up a cairn"},
]
# rare events (3g): never announced, only perceived where they happen
t["J"] = {"name": "dry spring", "solid": True, "desc": "a hollow of damp stones where a spring ran"}
t["L"]["fragile"] = {"p": 0.02, "becomes": "grass", "leaves": ["stick", "stick"]}   # a storm tears it down
w["events"] = {
    "_doc": "per_day = the chance a day that one starts (one of a kind at a time); lasts in ticks",
    "storm": {"per_day": 0.12, "lasts": [30, 80], "chill": 0.004,
              "felt": "A storm is blowing: the wind howls and hard rain lashes down."},
    "wreck": {"per_day": 0.08, "count": [2, 4],
              "things": ["planks", "bowl", "copper knife", "log", "rope", "barrel", "sailcloth", "glass bottle"]},
    "drought": {"per_day": 0.05, "lasts": [360, 900], "tile": "spring", "becomes": "dry spring"},
    "arrival": {"per_day": 0.03, "kinds": ["boar"]},
}
c["boar"] = {"desc": "a bristly boar, snuffling at the ground", "count": 0, "wander": 0.2,
             "tough": 1.5, "dodge": 0.1, "leaves": ["raw meat", "raw meat", "raw meat"],
             "charges": {"p": 0.03, "p_angry": 0.5, "angry_for": 20, "hurt": 0.2},
             "_doc": "comes ashore now and then (event `arrival`); charges one who stands beside it, more when struck"}
# objects of unclear value (3h): lying about, or brought by a wreck; some have a use to find, some none
w["generate"]["finds"] = [
    {"item": "shell", "count": 14, "where": "shore"},
    {"item": "smooth black stone", "count": 4, "biomes": ["hills"]},
    {"item": "carved figure", "count": 2, "where": "mist"},
]
t["Y"] = {"name": "barrel", "solid": True, "store": 8, "desc": "a barrel standing upright, its lid loose"}
t["Z"] = {"name": "tent", "roof": True, "desc": "a tent of sailcloth over sticks",
          "fragile": {"p": 0.01, "becomes": "grass", "leaves": ["sailcloth", "stick"]}}
w["use"] += [
    {"held": "sharp stone", "target": "item:shell", "target_becomes": "shell bead",
     "text": "I bored a hole through the shell with the sharp stone: a shell bead.", "seen": "worked a shell"},
    {"held": "string", "target": "item:shell bead", "consume": True, "target_becomes": "string of shell beads",
     "text": "I threaded the bead on the string: a string of shell beads.", "seen": "threaded a bead"},
    {"held": "glass bottle", "target": "water", "consume": True, "gives": ["bottle of water"],
     "text": "I filled the glass bottle with water.", "seen": "filled a bottle"},
    {"held": "glass bottle", "target": "spring", "consume": True, "gives": ["bottle of water"],
     "text": "I filled the glass bottle at the spring.", "seen": "filled a bottle"},
    {"held": "barrel", "target": "@ground", "consume": True, "becomes": "barrel",
     "text": "I stood the barrel upright on the ground.", "seen": "set down a barrel"},
]
w["drink"].insert(0, {"held": "bottle of water", "amount": 0.3, "leaves": "glass bottle",
                      "text": "I drank the water from the glass bottle.", "seen": "drank from a bottle"})
w["build"].insert(0, {"needs": {"sailcloth": 1, "stick": 2}, "becomes": "tent",
                      "text": "I stretched the sailcloth over the sticks: a tent.", "seen": "pitched a tent"})
w["mass_nouns"] = ["raw meat", "roasted meat", "berries", "roasted berries", "wool", "sand", "clay", "seeds", "wheat",
                   "string", "copper", "copper ore", "glass", "bread", "milk"]
w["food"]["raw meat"] = 0.1
w["food"]["roasted meat"] = 0.45
w["use"].append({"held": "raw meat", "target": "fire", "consume": True, "gives": ["roasted meat"],
                 "text": "I roasted the meat over the fire; it smells rich.", "seen": "roasted meat"})

# food on the shore and near every waking spot (isl2: the bushes near the spawn were eaten on day 0 and a body's
# nearest was 29 steps off): mussels on rocks at the sea's edge, regrowing like a bush (generate `shore_food`)
# isl3: "wet rocks crusted with dark mussels" read as rocks (take 0.61 very hungry); "mussels clinging to wet rocks" 0.96
t["P"] = {"name": "mussels on rocks", "solid": True, "desc": "mussels clinging to wet rocks"}
t["p"] = {"name": "bare rocks", "solid": True, "desc": "wet rocks, the mussels picked off"}
w["take"] = [{"target": "mussels on rocks", "gives": ["mussels"], "becomes": "bare rocks",
              "text": "I prised mussels off the rocks.", "seen": "gathered mussels"},
             {"target": "bare rocks", "text": "There are no mussels left on these rocks."}] + w["take"]
w["grow"].append({"from": "bare rocks", "to": "mussels on rocks", "after": 300})
w["food"]["mussels"] = 0.2                  # a handful: two a day feed a body (isl3)
w["food"]["roasted mussels"] = 0.35
w["mass_nouns"] += ["mussels", "roasted mussels"]
w["use"].append({"held": "mussels", "target": "fire", "consume": True, "gives": ["roasted mussels"],
                 "text": "I roasted the mussels over the fire; their shells opened.", "seen": "roasted mussels"})

# `take_to_hand` (Land._give: what was just taken is the thing in hand) is NOT set here: tried in isl5, no better,
# and it takes away something the mind can learn (switch, then eat); owner, 2026-09-30: what the minds know how to
# do at birth is a question for the global objectives

out = ROOT / "brain" / "lands" / "island.json"
out.write_text(json.dumps(w, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
print(f"{out.name}: take {len(w['take'])} use {len(w['use'])} tiles {len(t)} creatures {len(c)}")
