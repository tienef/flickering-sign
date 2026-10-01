import json, copy
from pathlib import Path

base = json.loads(Path(r'D:\Projects\jevs\brain\lands\first.json').read_text(encoding='utf-8'))
w = copy.deepcopy(base)
w['_doc'] = ("The wild land (step 10s, owner: 'un monde à la Minecraft'): the first land's tiles and rules, plus a "
             "96x96 world of biomes (meadow, forest, lake, hills, marsh, dunes) whose mist draws back as a mind comes "
             "near (fog), creatures that move and behave (sheep to shear, cows to milk, chickens that lay, rabbits and "
             "deer that bolt, frogs), rain (puts out fires, waters what was sown, fills bowls left out, chills who is "
             "outside), and deeper chains: mining with the axe (stones, copper ore in green-veined rock), copper from the "
             "fire, a copper knife, wool to string to a fishing line, fish, eggs, milk, mushrooms, reeds and a reed pipe, "
             "doors to build. None of it is told to the mind.")
w['name'] = 'wild'
w['size'] = [96, 96]
w['seed_note'] = 'Generated per seed by Land._generate_wild; the first land (first.json) is unchanged.'
w['tiles']['M'] = {"name": "mushrooms", "desc": "a cluster of brown mushrooms"}
w['tiles']['r'] = {"name": "reeds", "desc": "tall green reeds"}
w['tiles']['V'] = {"name": "veined rock", "solid": True, "desc": "a rock veined with green"}
w['generate'] = {
    "biomes": {
        "centres": 18,
        "types": {"meadow": 5, "forest": 5, "lake": 2, "hills": 3, "marsh": 2, "dunes": 1},
        "lake_radius": [3, 6],
        "fill": {
            "meadow": {",": 0.08, "f": 0.05, "B": 0.012, "T": 0.02, "o": 0.01},
            "forest": {"T": 0.36, "B": 0.03, "M": 0.03, "o": 0.01, ",": 0.02},
            "hills": {"R": 0.28, "V": 0.04, "o": 0.08, ",": 0.02},
            "marsh": {"~": 0.18, "r": 0.2, "c": 0.06, ",": 0.05},
            "dunes": {"s": 0.8, "o": 0.01},
        },
    },
    "shelter": {"size": 7, "door": "south"},
    "orchard": {"size": 9, "apple_trees": 6, "min_distance": 11, "max_distance": 22},
    "flickering_stones": 1,
    "near_door": {"B": 3, ",": 4, "o": 3, "T": 3},
}
w['fog'] = {"start": 40, "step": 12, "near": 3,
            "_doc": "revealed square at first (start), widened by `step` on a side when a mind is within `near` of it"}
w['weather'] = {"rain_every": [900, 2400], "rain_lasts": [120, 360], "quench": 0.02, "chill": 0.001,
                "fill_bowls": 0.02}
w['creature_ground'] = ["grass", "tall grass", "flowers", "dirt", "sand", "stones", "mushrooms"]
w['creatures'] = {
    "sheep": {"desc": "a woolly white sheep", "biomes": ["meadow"], "count": 10, "wander": 0.12},
    "shorn sheep": {"desc": "a sheep with its wool shorn close", "wander": 0.12},
    "cow": {"desc": "a brown cow, chewing slowly", "biomes": ["meadow"], "count": 5, "wander": 0.05},
    "chicken": {"desc": "a speckled chicken, pecking", "biomes": ["meadow", "forest"], "count": 8, "wander": 0.3,
                "lays": {"item": "egg", "every": 500}},
    "rabbit": {"desc": "a grey rabbit, ears up", "biomes": ["meadow", "dunes"], "count": 10, "wander": 0.25, "flee": 2},
    "deer": {"desc": "a deer, watching me", "biomes": ["forest"], "count": 6, "wander": 0.1, "flee": 4},
    "frog": {"desc": "a green frog", "biomes": ["marsh", "lake"], "count": 8, "wander": 0.2, "flee": 1,
             "ground": ["grass", "tall grass", "sand", "clay", "reeds"]},
}
w['creature_grow'] = [{"from": "shorn sheep", "to": "sheep", "after": 900}]
w['grow'] = base['grow'] + [
    {"from": "mushrooms", "to": "grass", "after": 99999},
]
take_new = [
    {"target": "rock", "held": "stone axe", "gives": ["stone", "stone"], "becomes": "stones",
     "text": "I struck the rock with the axe until it split: stones.", "seen": "broke a rock"},
    {"target": "veined rock", "held": "stone axe", "gives": ["stone", "copper ore"], "becomes": "stones",
     "text": "I struck the veined rock with the axe; it split, and a greenish lump came loose: copper ore.",
     "seen": "broke a veined rock"},
    {"target": "veined rock", "text": "The rock is veined with green; nothing comes loose by hand."},
    {"target": "tree", "held": "copper knife", "gives": ["stick", "stick"],
     "text": "I cut two straight sticks from the tree with the copper knife.", "seen": "cut sticks"},
    {"target": "mushrooms", "gives": ["mushroom"], "becomes": "grass", "text": "I picked a mushroom.",
     "seen": "picked a mushroom"},
    {"target": "reeds", "gives": ["reed"], "text": "I pulled a reed.", "seen": "pulled a reed"},
    {"target": "sheep", "held": "sharp stone", "gives": ["wool"], "creature_becomes": "shorn sheep",
     "text": "I cut the wool from the sheep with the sharp stone: wool. It bleats and trots off.",
     "seen": "sheared a sheep", "flees": True},
    {"target": "sheep", "held": "copper knife", "gives": ["wool", "wool"], "creature_becomes": "shorn sheep",
     "text": "I sheared the sheep with the copper knife: two bundles of wool.", "seen": "sheared a sheep"},
    {"target": "sheep", "text": "I grabbed at the sheep's wool; it bleats and pulls away.", "flees": True},
    {"target": "shorn sheep", "text": "The sheep has no wool left to take.", "flees": True},
    {"target": "cow", "text": "The cow chews and ignores me."},
    {"target": "chicken", "text": "The chicken flaps and scurries away.", "flees": True},
    {"target": "rabbit", "text": "The rabbit bolts.", "flees": True},
    {"target": "deer", "text": "The deer bounds away.", "flees": True},
    {"target": "frog", "text": "The frog slips through my fingers and hops away.", "flees": True},
]
use_new = [
    {"held": "bowl", "target": "cow", "consume": True, "gives": ["bowl of milk"],
     "text": "I milked the cow into the bowl: a bowl of milk.", "seen": "milked a cow"},
    {"target": "cow", "text": "I patted the cow. It lows softly."},
    {"target": "sheep", "text": "I patted the sheep; its wool is thick and warm."},
    {"held": "stick", "target": "item:wool", "target_becomes": "string",
     "text": "I twisted the wool around the stick into a length of string.", "seen": "spun string"},
    {"held": "string", "target": "item:stick", "consume": True, "target_becomes": "fishing line",
     "text": "I tied the string to the stick: a fishing line.", "seen": "made a fishing line"},
    {"held": "fishing line", "target": "water", "chance": 0.3, "gives": ["fish"],
     "text": "I dipped the line in the water; something tugged, and I pulled out a fish!",
     "text_fail": "I dipped the line in the water and waited. Nothing bit.", "seen": "caught a fish",
     "seen_fail": "fished"},
    {"held": "fish", "target": "fire", "consume": True, "gives": ["roasted fish"],
     "text": "I roasted the fish over the fire.", "seen": "roasted a fish"},
    {"held": "egg", "target": "fire", "consume": True, "gives": ["cooked egg"],
     "text": "I set the egg in the embers until it was cooked.", "seen": "cooked an egg"},
    {"held": "mushroom", "target": "fire", "consume": True, "gives": ["roasted mushroom"],
     "text": "I roasted the mushroom; it smells of earth.", "seen": "roasted a mushroom"},
    {"held": "copper ore", "target": "fire", "consume": True, "gives": ["copper"],
     "text": "I held the ore in the fire until bright metal ran out of it and cooled: copper.",
     "seen": "smelted copper"},
    {"held": "copper", "target": "item:stick", "consume": True, "target_becomes": "copper knife",
     "text": "I hammered the copper flat and bound it to the stick: a copper knife.", "seen": "made a knife"},
    {"held": "plank", "target": "item:plank", "consume": True, "target_becomes": "door",
     "text": "I fitted the two planks together: a door.", "seen": "made a door"},
    {"held": "door", "target": "@ground", "consume": True, "becomes": "closed door",
     "text": "I set the door upright in the ground and fixed it.", "seen": "set up a door"},
    {"held": "sharp stone", "target": "item:reed", "target_becomes": "reed pipe",
     "text": "I cut holes in the reed with the sharp stone: a reed pipe.", "seen": "made a reed pipe"},
    {"held": "reed pipe", "target": "@ground",
     "text": "I blew into the reed pipe: a clear, thin note carried over the land.",
     "seen": "played a note on a reed pipe", "heard": "From the {dir}, a thin clear note, like a pipe.",
     "heard_far": "Far away, a thin note, like a pipe."},
    {"held": "wool", "target": "item:planks", "consume": True, "target_becomes": "bed",
     "text": "I spread the wool over the planks: a soft bed.", "seen": "made a bed"},
]
# held rules must come before bare-hand rules on the same target: _rule tries held rules first anyway
w['take'] = take_new[:4] + base['take'] + take_new[4:]
w['use'] = use_new + base['use']
w['food'] = dict(base['food'], **{"mushroom": 0.08, "roasted mushroom": 0.15, "egg": 0.1, "cooked egg": 0.25,
                                    "fish": 0.1, "roasted fish": 0.35, "bowl of milk": 0.25})
w['eat_leaves'] = {"bowl of milk": "bowl"}
Path(r'D:\Projects\jevs\brain\lands\wild.json').write_text(json.dumps(w, indent=1, ensure_ascii=False), encoding='utf-8')
print('rules take', len(w['take']), 'use', len(w['use']), 'creatures', len(w['creatures']))
