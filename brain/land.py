"""A land: a grid world defined by data (step 9), for one mind or several.

    Land(seed, spec="brain/lands/first.json")   # then .join(key, papers) per mind

A spec with a `map` is drawn instead of generated (the small trial rooms of
`brain/trials.py`: same tiles and rules, so what is learned there holds here).

A top-down grid (no 3D): grass, trees, water, rock, clay, bushes; a shelter at
the centre whose door is the first thing to learn; a walled orchard whose heavy
gate opens only when two minds push it at the same moment; mist at the edges
(the frontier). What exists and every rule live in the data file
(`brain/lands/*.json`): taking things (a stick from a tree, berries from a bush)
and using what you hold on what is in front of you (a stone on a stone makes a
sharp stone, a stick on a sharp stone an axe, a stick rubbed on a log three
times in a row a fire, wheat on a fire bread...). None of it is told to the
mind. Plants grow back, sown seeds grow into wheat if water is near, fires burn
out, days turn to nights.

The body: where it stands, which way it faces, what it carries and holds.
Verbs: move north/east/south/west (turning toward a way that is blocked),
look, take, use, switch (what to hold), eat, wait, speak. Moving toward
something faces it, so "go to the tree, then take" is how things are reached.

What the mind senses is text: where it is, what is in front, the four sides,
the time of day, what it carries, the others it can see, and what it heard.
Every percept carries a `focus` ("facing a tree, holding a stone"): the
cerebellum learns on it, not on the whole text, which almost never repeats in
a big world (step 9: without it, learning progress would vanish like the sign's).
Each move also has its own focus, what lies that way ("north: plank wall"):
keyed on the front, a wall beside the mind made every sideways bump a surprise
(step 10e).

Drives from the world: eating emits `ate` (amount by food), being outside at
night emits `chill`, a roof or a fire `warmth`. No hit points, no injury, no
death (owner, 2026-09-24).

Others, as in the shared valley: "someone with an amber mark", never a name.
Seen within sight; their deeds seen within 3 steps; their words heard with
the mark within 3 steps, as a voice from a direction within 8, as a far-off
voice beyond.
"""
from __future__ import annotations

import json
import heapq
import math
import random
from pathlib import Path

from .events import event

SPEC = Path(__file__).parent / "lands" / "first.json"
DIRS = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}
GROUND = {"grass", "dirt", "sand"}
# the load felt near the carrying limit, as a share of what the body can carry (3k); a land may set `body.load_felt`
LOAD_FELT = [[0.8, "My arms are laden."], [1.0, "My arms are laden; I could not carry anything more."]]
COLOURS = ("blue", "amber", "green", "violet", "red", "white", "grey", "gold")
GLYPHS = "ᚠᚢᚦᚨᚱᚲᚷᚹᚺᚾᛁᛃᛇᛈᛉᛊᛏᛒᛖᛗᛚᛜᛞᛟ"


MASS: set[str] = set()                   # nouns said with "some" (a land's `mass_nouns`: some raw meat)


def _a(word: str) -> str:
    if word in MASS:
        return "some " + word
    return ("an " if word[:1].lower() in "aeiou" else "a ") + word


def _where(dx: int, dy: int) -> str:
    """'3 steps north', '2 steps east and 1 south', 'here'."""
    parts = []
    if dy:
        parts.append(("north", -dy) if dy < 0 else ("south", dy))
    if dx:
        parts.append(("east", dx) if dx > 0 else ("west", -dx))
    if not parts:
        return "here"
    (d1, n1), *rest = sorted(parts, key=lambda p: -p[1])
    s = f"{n1} step{'s' if n1 > 1 else ''} {d1}"
    if rest:
        s += f" and {rest[0][1]} {rest[0][0]}"
    return s


def _compass(dx: int, dy: int) -> str:
    if dx == dy == 0:
        return "here"
    ang = math.degrees(math.atan2(dx, -dy)) % 360
    return ("north", "north-east", "east", "south-east", "south", "south-west", "west",
            "north-west")[int((ang + 22.5) // 45) % 8]


class LandBody:
    def __init__(self, key: str, mark: str, x: int, y: int):
        self.key, self.mark = key, mark
        self.x, self.y, self.facing = x, y, "south"
        self.inv: list[str] = []
        self.held: str | None = None
        self.papers: list | None = None
        self.shelf, self.page = 0, 0
        self.asleep = False
        self.inbox: list = []
        self.streak: tuple | None = None          # (rule index, x, y, n) for "in a row" rules
        # food seen by `look` further than arm's reach: (x, y) -> (what, clock) (step 10ab). The
        # body keeps where it saw it (path integration), not the brain: it has no coordinates.
        self.seen_food: dict[tuple, tuple] = {}
        self.weak = 0.0                           # the body's weakness from blows (P3): heals with time
        self.pending: list = []                   # events for this body's mind (a blow), at its next sense

    def state(self) -> dict:
        return {"mark": self.mark, "x": self.x, "y": self.y, "facing": self.facing,
                "inv": self.inv, "held": self.held, "shelf": self.shelf, "page": self.page,
                **({"weak": round(self.weak, 4)} if self.weak else {})}


class Land:
    SOCIAL_VERBS = {"speak": 'call out or speak aloud (to say words, give them as "words" in your reply)'}

    def __init__(self, seed: int = 0, spec: str | Path | dict = SPEC, has_archive: bool = True):
        self.spec = spec if isinstance(spec, dict) else json.loads(Path(spec).read_text(encoding="utf-8"))
        self.rng = random.Random(seed)
        MASS.update(self.spec.get("mass_nouns", []))
        self.name = "land" if has_archive else "land-closed"
        self.has_archive = has_archive
        self.W, self.H = self.spec["size"]
        self.tiles = self.spec["tiles"]
        self.code = {t["name"]: c for c, t in self.tiles.items()}
        self.grow = {g["from"]: g for g in self.spec.get("grow", [])}
        self.clock = 0
        self.items: dict[tuple, str] = {}
        self.timers: dict[tuple, dict] = {}
        self.pushes: dict[tuple, set] = {}        # gate position -> who pushed this tick
        self.stores: dict[tuple, list] = {}       # a built cache (tile `store`): what was put in it, in order
        self.events: dict[str, dict] = {}         # rare events under way (spec `events`): kind -> {until, ...}
        self.bodies: dict[str, LandBody] = {}
        # what passed between bodies (and rare events) this round, for the world's row (P3): acts that
        # concern another mind, gathered as they happen; `advance` moves them to `last_round`
        self.happened: list[dict] = []
        self.last_round: list[dict] = []
        self.glyphs = self._glyphs()
        self.grid = [["." for _ in range(self.W)] for _ in range(self.H)]
        self.shelter = self.orchard = (0, 0, 0, 0)
        # the wild land (lands/wild.json; absent keys = the first land, unchanged): creatures that
        # move and behave, a frontier of mist that draws back as a mind comes near, rain
        self.cspec: dict = self.spec.get("creatures", {})
        self.creatures: dict[tuple, list] = {}            # (x, y) -> [kind, since]
        self.cgrow = {g["from"]: g for g in self.spec.get("creature_grow", [])}
        self.revealed: tuple | None = None
        self.weather = {"rain": False, "until": 0, "next": 0}
        self.mist: set[tuple] = set()                     # the island's mist banks (fog.banks): cells not yet seen
        self.shore: list[tuple] = []                      # the island's spawn: (angle, x, y, facing) along the coast
        if self.spec.get("map"):
            self._load_map()
        elif self.spec["generate"].get("island"):
            self._generate_island()
        elif self.spec["generate"].get("biomes"):
            self._generate_wild()
        else:
            self._generate()
        if self.spec.get("fog") and not self.spec["fog"].get("banks"):
            r = int(self.spec["fog"]["start"]) // 2
            cx, cy = self.W // 2, self.H // 2
            self.revealed = (max(1, cx - r), max(1, cy - r), min(self.W - 2, cx + r - 1), min(self.H - 2, cy + r - 1))
        if self.spec.get("weather"):
            self.weather["next"] = self.rng.randint(*self.spec["weather"]["rain_every"])
        self.verbs = {
            "north": "go north (or turn to face it, if something is in the way)",
            "east": "go east (or turn to face it, if something is in the way)",
            "south": "go south (or turn to face it, if something is in the way)",
            "west": "go west (or turn to face it, if something is in the way)",
            "look": "look around carefully at what is near",
            "take": "take or gather what is in front of me",
            # true empty-handed too (a push, a pat, a touch; the door opens so): owner 2026-09-30, "je ne veux
            # pas que la faiblesse du world fausse notre expérience" (was "use what I hold on what is in front of me")
            "use": "use what is in front of me: handle or push it, or work it with what I hold (or set that down there)",
            "switch": "hold the next thing I carry instead",
            "eat": "eat what I hold",
            "wait": "stay put and let time pass",
        }
        # verbs a land adds (the island: drink, give, strike, build); the other lands keep the list above
        self.verbs.update({k: v for k, v in self.spec.get("verbs", {}).items() if not k.startswith("_")})

    # -- the map ----------------------------------------------------------------
    def name_at(self, x: int, y: int) -> str:
        if not (0 <= x < self.W and 0 <= y < self.H) or (x, y) in self.mist:
            return "mist"
        if self.revealed:
            x0, y0, x1, y1 = self.revealed
            if not (x0 <= x <= x1 and y0 <= y <= y1):
                return "mist"
        return self.tiles[self.grid[y][x]]["name"]

    def info(self, name: str) -> dict:
        return self.tiles[self.code[name]]

    def set_tile(self, x: int, y: int, name: str) -> None:
        self.grid[y][x] = self.code[name]
        if name in self.grow:
            self.timers[(x, y)] = {"since": self.clock, "watered": False}
        else:
            self.timers.pop((x, y), None)

    def solid(self, x: int, y: int) -> bool:
        return bool(self.info(self.name_at(x, y)).get("solid"))

    def _glyphs(self) -> str:
        return "".join(self.rng.choice(GLYPHS) for _ in range(5))

    def _blob(self, cx, cy, r, code, density=1.0, only=".") -> None:
        for y in range(cy - r, cy + r + 1):
            for x in range(cx - r, cx + r + 1):
                if 1 <= x < self.W - 1 and 1 <= y < self.H - 1 and (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    if self.grid[y][x] in only and self.rng.random() < density:
                        self.grid[y][x] = code

    def _scatter(self, n, code, near=None, radius=6) -> None:
        for _ in range(n * 20):
            if n <= 0:
                return
            if near:
                x = near[0] + self.rng.randint(-radius, radius)
                y = near[1] + self.rng.randint(-radius, radius)
            else:
                x, y = self.rng.randrange(1, self.W - 1), self.rng.randrange(1, self.H - 1)
            if 1 <= x < self.W - 1 and 1 <= y < self.H - 1 and self.grid[y][x] == ".":
                self.grid[y][x] = code
                n -= 1

    def _generate(self) -> None:
        g, W, H, R = self.spec["generate"], self.W, self.H, self.rng
        cx, cy = W // 2, H // 2
        half = g["shelter"]["size"] // 2
        clear = lambda x, y, m: abs(x - cx) > half + m or abs(y - cy) > half + m
        for _ in range(g["lakes"]):
            x, y = R.randrange(4, W - 4), R.randrange(4, H - 4)
            if clear(x, y, 5):
                r = R.randint(*g["lake_radius"])
                self._blob(x, y, r + 1, "s", 0.6)
                self._blob(x, y, r, "~", only=".s")
        for y in range(1, H - 1):                      # clay on the shore
            for x in range(1, W - 1):
                if self.grid[y][x] == "s" and R.random() < 0.15:
                    self.grid[y][x] = "c"
        for _ in range(g["woods"]):
            x, y = R.randrange(3, W - 3), R.randrange(3, H - 3)
            if clear(x, y, 3):
                self._blob(x, y, R.randint(*g["wood_radius"]), "T", g["wood_density"])
        for _ in range(g["rocks"]):
            x, y = R.randrange(3, W - 3), R.randrange(3, H - 3)
            if clear(x, y, 3):
                self._blob(x, y, R.randint(*g["rock_radius"]), "R", 0.8)
                self._scatter(3, "o", near=(x, y), radius=3)
        self._scatter(g["stones"], "o")
        self._scatter(g["berry_bushes"], "B")
        self._scatter(g["tall_grass"], ",")
        self._scatter(g["flowers"], "f")
        self._scatter(g["flickering_stones"], "Q")
        for x in range(W):                             # the frontier
            self.grid[0][x] = self.grid[H - 1][x] = "m"
        for y in range(H):
            self.grid[y][0] = self.grid[y][W - 1] = "m"
        # the orchard: a walled square away from the shelter, its gate facing the centre
        o = g["orchard"]["size"]
        for _ in range(200):
            ox, oy = R.randrange(2, W - o - 2), R.randrange(2, H - o - 2)
            mx, my = ox + o // 2, oy + o // 2
            far = abs(mx - cx) + abs(my - cy) >= g["orchard"]["min_distance"] + o // 2
            apart = abs(mx - cx) > half + o // 2 + 3 or abs(my - cy) > half + o // 2 + 3
            if far and apart:
                break
        self.orchard = (ox, oy, ox + o - 1, oy + o - 1)
        for y in range(oy, oy + o):
            for x in range(ox, ox + o):
                edge = x in (ox, ox + o - 1) or y in (oy, oy + o - 1)
                self.grid[y][x] = "|" if edge else "."
        mx, my = ox + o // 2, oy + o // 2
        if abs(cx - mx) > abs(cy - my):
            gx, gy, step = (ox + o - 1, my, (1, 0)) if cx > mx else (ox, my, (-1, 0))
        else:
            gx, gy, step = (mx, oy + o - 1, (0, 1)) if cy > my else (mx, oy, (0, -1))
        self.grid[gy][gx] = "G"
        for k in (1, 2):
            self.grid[gy + step[1] * k][gx + step[0] * k] = "."
        inside = [(x, y) for y in range(oy + 1, oy + o - 1) for x in range(ox + 1, ox + o - 1)]
        for x, y in R.sample(inside, g["orchard"]["apple_trees"]):
            if (x, y) != (gx - step[0], gy - step[1]):
                self.grid[y][x] = "A"
        # the shelter at the centre: plank walls, a floor, a shelf, a door
        x0, y0, x1, y1 = cx - half, cy - half, cx + half, cy + half
        self.shelter = (x0, y0, x1, y1)
        for y in range(y0 - 2, y1 + 3):
            for x in range(x0 - 2, x1 + 3):
                self.grid[y][x] = "."                  # a clearing around it
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                self.grid[y][x] = "#" if x in (x0, x1) or y in (y0, y1) else "_"
        self.grid[y1][cx] = "D"
        if self.has_archive:
            self.grid[y0 + 1][cx] = "S"
        # the first things to find, close to the door
        self._scatter(3, "B", near=(cx, y1 + 5), radius=4)
        self._scatter(4, ",", near=(cx, y1 + 4), radius=5)
        self._scatter(3, "o", near=(cx, y1 + 4), radius=5)
        self._scatter(3, "T", near=(cx - 5, y1 + 4), radius=3)
        for k in (1, 2):                               # the doorstep stays clear: land1 (seed 1)
            self.grid[y1 + k][cx] = "."                # opened onto a berry bush, no way out
        self.door = (cx, y1)

    def _generate_wild(self) -> None:
        """The wild land (lands/wild.json, step 10s): biomes around seeded centres (meadow,
        forest, lake, hills, marsh, dunes), each filled by the data file's densities; the
        shelter at the centre in a meadow clearing, the walled orchard further off; creatures
        placed in their biomes. The first land's `_generate` is left untouched (same seeds,
        same worlds)."""
        g, W, H, R = self.spec["generate"], self.W, self.H, self.rng
        bio = g["biomes"]
        cx, cy = W // 2, H // 2
        half = g["shelter"]["size"] // 2
        kinds = [k for k, n in bio["types"].items() for _ in range(n)]
        centres = [(cx, cy, "meadow")]
        todo = [k for k in bio["types"] if k != "meadow"]           # every biome at least once
        for _ in range(5000):
            if len(centres) >= bio["centres"]:
                break
            x, y = R.randrange(4, W - 4), R.randrange(4, H - 4)
            if abs(x - cx) + abs(y - cy) > 14 and all(abs(x - a) + abs(y - b) > 9 for a, b, _ in centres):
                centres.append((x, y, todo.pop(0) if todo else R.choice(kinds)))
        self.biome = [[None] * W for _ in range(H)]
        radius = {}
        for i, (x, y, k) in enumerate(centres):
            if k == "lake":
                radius[i] = R.randint(*bio["lake_radius"])
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                jx, jy = x + R.randint(-2, 2), y + R.randint(-2, 2)          # ragged borders
                i = min(range(len(centres)), key=lambda j: (centres[j][0] - jx) ** 2 + (centres[j][1] - jy) ** 2)
                bx, by, k = centres[i]
                self.biome[y][x] = k
                if k == "lake":
                    d = math.hypot(x - bx, y - by)
                    r = radius[i]
                    if d <= r:
                        self.grid[y][x] = "~"
                    elif d <= r + 1.5:
                        self.grid[y][x] = "c" if R.random() < 0.15 else "s"
                    else:
                        self._fill_cell(x, y, bio["fill"].get("meadow", {}))
                    continue
                self._fill_cell(x, y, bio["fill"].get(k, {}))
        for _ in range(g.get("flickering_stones", 0)):
            self._scatter(1, "Q")
        for x in range(W):                                           # the frontier
            self.grid[0][x] = self.grid[H - 1][x] = "m"
        for y in range(H):
            self.grid[y][0] = self.grid[y][W - 1] = "m"
        # the orchard: a walled square in the open, its gate facing the centre
        o = g["orchard"]["size"]
        for _ in range(400):
            ox, oy = R.randrange(3, W - o - 3), R.randrange(3, H - o - 3)
            mx, my = ox + o // 2, oy + o // 2
            d = abs(mx - cx) + abs(my - cy)
            clear = (ox > cx + half + 4 or ox + o - 1 < cx - half - 4
                     or oy > cy + half + 4 or oy + o - 1 < cy - half - 4)      # off the shelter's clearing
            if g["orchard"]["min_distance"] + o // 2 <= d <= g["orchard"].get("max_distance", 99) \
                    and self.biome[my][mx] in ("meadow", "forest") and clear:
                break
        self.orchard = (ox, oy, ox + o - 1, oy + o - 1)
        for y in range(oy, oy + o):
            for x in range(ox, ox + o):
                edge = x in (ox, ox + o - 1) or y in (oy, oy + o - 1)
                self.grid[y][x] = "|" if edge else "."
        if abs(cx - mx) > abs(cy - my):
            gx, gy, step = (ox + o - 1, my, (1, 0)) if cx > mx else (ox, my, (-1, 0))
        else:
            gx, gy, step = (mx, oy + o - 1, (0, 1)) if cy > my else (mx, oy, (0, -1))
        self.grid[gy][gx] = "G"
        for k in (1, 2):
            self.grid[gy + step[1] * k][gx + step[0] * k] = "."
        inside = [(x, y) for y in range(oy + 1, oy + o - 1) for x in range(ox + 1, ox + o - 1)]
        for x, y in R.sample(inside, g["orchard"]["apple_trees"]):
            if (x, y) != (gx - step[0], gy - step[1]):
                self.grid[y][x] = "A"
        # the shelter at the centre: plank walls, a floor, a door to the south, a clearing
        x0, y0, x1, y1 = cx - half, cy - half, cx + half, cy + half
        self.shelter = (x0, y0, x1, y1)
        for y in range(y0 - 3, y1 + 4):
            for x in range(x0 - 3, x1 + 4):
                self.grid[y][x] = "."
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                self.grid[y][x] = "#" if x in (x0, x1) or y in (y0, y1) else "_"
        self.grid[y1][cx] = "D"
        if self.has_archive:
            self.grid[y0 + 1][cx] = "S"
        near = g.get("near_door", {})
        for code, n in near.items():
            self._scatter(n, code, near=(cx, y1 + 5), radius=5)
        for k in (1, 2):
            self.grid[y1 + k][cx] = "."
        self.door = (cx, y1)
        # creatures, in their biomes
        for kind, sp in self.cspec.items():
            spots = [(x, y) for y in range(1, H - 1) for x in range(1, W - 1)
                     if self.biome[y][x] in sp.get("biomes", []) and self.grid[y][x] in ".,:sof"
                     and not (x0 - 4 <= x <= x1 + 4 and y0 - 4 <= y <= y1 + 4)]
            R.shuffle(spots)
            for x, y in spots[: sp.get("count", 0)]:
                if (x, y) not in self.creatures:
                    self.creatures[(x, y)] = [kind, 0]

    def _generate_island(self) -> None:
        """The island (lands/island.json, P3): land in the sea with a ragged coast and beaches, biomes
        inside around seeded centres as in the wild land (the spec's fills), a pond and a few springs
        the only fresh water, caves in the hills (a natural roof, few), the walled garden and its heavy
        gate, no shelter and no shelf; mist banks over parts of the interior (lifted for good when a mind
        comes near); the spawn spread along the shore, facing inland; springs near the coast and food on the
        shore (spec `springs.coast`, `shore_food`: isl2, bodies woke far from both)."""
        g, W, H, R = self.spec["generate"], self.W, self.H, self.rng
        isl, bio = g["island"], g["biomes"]
        self.shelter = (-1, -1, -1, -1)                                # none: the minds build or find one
        cx, cy = W // 2, H // 2
        rad = float(isl["radius"])
        waves = [(k, R.uniform(*isl["coast_wobble"]), R.uniform(0, 2 * math.pi)) for k in range(2, 6)]
        coast = lambda a: rad * (1.0 + sum(amp * math.sin(k * a + ph) for k, amp, ph in waves))
        land = set()
        for y in range(H):
            for x in range(W):
                d, a = math.hypot(x - cx, y - cy), math.atan2(y - cy, x - cx)
                r = coast(a)
                if d > r or not (1 <= x < W - 1 and 1 <= y < H - 1):
                    self.grid[y][x] = "~"                              # the sea (tile `~` is named in the spec)
                elif d > r - float(isl.get("beach", 1.6)):
                    self.grid[y][x] = "o" if R.random() < isl.get("beach_stones", 0.03) else "s"
                else:
                    land.add((x, y))
        # biomes inside, as in the wild land (Voronoi with ragged borders)
        kinds = [k for k, n in bio["types"].items() for _ in range(n)]
        centres = [(cx, cy, bio.get("centre", "meadow"))]
        todo = [k for k in bio["types"] if k != centres[0][2]]
        inner = sorted(land)
        for _ in range(5000):
            if len(centres) >= bio["centres"]:
                break
            x, y = R.choice(inner)
            if all(abs(x - a) + abs(y - b) > bio.get("spacing", 7) for a, b, _ in centres):
                centres.append((x, y, todo.pop(0) if todo else R.choice(kinds)))
        self.biome = [[None] * W for _ in range(H)]
        for x, y in inner:
            jx, jy = x + R.randint(-2, 2), y + R.randint(-2, 2)
            k = min(centres, key=lambda c: (c[0] - jx) ** 2 + (c[1] - jy) ** 2)[2]
            self.biome[y][x] = k
            self._fill_cell(x, y, bio["fill"].get(k, {}))
        coastal = [(x, y) for y in range(H) for x in range(W) if self.grid[y][x] in "so"]
        for x, y in coastal:
            self.biome[y][x] = "shore"
        interior = lambda x, y, m: all((x + dx, y + dy) in land for dx in range(-m, m + 1) for dy in range(-m, m + 1))
        # the pond: fresh water, clay and reeds on its rim
        pond, self.pond = g.get("pond"), (-99, -99, 0)
        if pond:
            for _ in range(400):
                px, py = R.choice(inner)
                pr = R.randint(*pond["radius"])
                if interior(px, py, pr + 3) and math.hypot(px - cx, py - cy) > pond.get("min_distance", 0):
                    break
            for y in range(py - pr - 2, py + pr + 3):
                for x in range(px - pr - 2, px + pr + 3):
                    d = math.hypot(x - px, y - py)
                    if d <= pr:
                        self.grid[y][x] = pond["tile"]
                    elif d <= pr + 1.5:
                        self.grid[y][x] = R.choice(pond["rim"])
            self.pond = (px, py, pr)
        # springs: a spring tile set in the ground, far from each other and from the pond
        springs = []
        for _ in range(g.get("springs", {}).get("count", 0)):
            for _ in range(400):
                x, y = R.choice(inner)
                if interior(x, y, 2) and self.biome[y][x] in g["springs"]["biomes"] \
                        and all(abs(x - a) + abs(y - b) > g["springs"]["apart"] for a, b in springs + [self.pond[:2]]):
                    springs.append((x, y))
                    self.grid[y][x] = g["springs"]["tile"]
                    for dx, dy in DIRS.values():                     # reachable from at least one side
                        if self.tiles[self.grid[y + dy][x + dx]].get("solid"):
                            self.grid[y + dy][x + dx] = "."
                    break
        # the waking spots' angles on one arc (spec `spawn.spacing`: steps along the coast between neighbours; 3m,
        # isl4-isl8: spread round the island by the golden angle, no mind ever saw another); drawn here, before the
        # coast springs, which may sit between them. Without it, the golden angle below, as before.
        spacing = float((g.get("spawn") or {}).get("spacing", 0))
        arc = []
        if spacing:
            a0, step = R.uniform(0, 2 * math.pi), spacing / rad
            arc = [a0 + k * step for k in range(min(16, int(2 * math.pi / step)))]
        # springs near the coast (spec `springs.coast`: count, inland [a, b]): on rays spread evenly round the
        # island, a few steps in from the beach, so no stretch of shore is far from fresh water; with
        # `between_spawn` and an arc, placed after the spawn instead (below)
        sc = g.get("springs", {}).get("coast")
        between = bool(sc and sc.get("between_spawn")) and len(arc) > 1
        if sc and not between:
            a1 = R.uniform(0, 2 * math.pi)
            for k in range(int(sc["count"])):
                a = a1 + k * 2 * math.pi / int(sc["count"])
                ray = [(round(cx + math.cos(a) * i), round(cy + math.sin(a) * i)) for i in range(int(rad * 2), 0, -1)]
                edge = next((i for i, q in enumerate(ray) if q in land), None)
                if edge is None:
                    continue
                x, y = ray[min(len(ray) - 1, edge + R.randint(*sc["inland"]))]
                if (x, y) not in land or self.grid[y][x] == g.get("pond", {}).get("tile"):
                    continue
                springs.append((x, y))
                self.grid[y][x] = g["springs"]["tile"]
                for dx, dy in DIRS.values():
                    if self.tiles[self.grid[y + dy][x + dx]].get("solid"):
                        self.grid[y + dy][x + dx] = "."
        self.springs = springs
        # caves: a pocket of roofed floor in the hills, rock around, one mouth
        caves = []
        cv = g.get("caves", {})
        for _ in range(cv.get("count", 0)):
            for _ in range(600):
                x, y = R.choice(inner)
                if interior(x, y, 3) and self.biome[y][x] == "hills" \
                        and all(abs(x - a) + abs(y - b) > cv.get("apart", 10) for a, b in caves):
                    break
            else:
                continue
            caves.append((x, y))
            for yy in range(y - 1, y + 2):
                for xx in range(x - 1, x + 3):
                    self.grid[yy][xx] = "R"
            self.grid[y][x] = self.grid[y][x + 1] = cv["tile"]
            self.grid[y + 1][x] = cv["tile"]                         # the mouth, open to the south
            self.grid[y + 2][x] = "."
        self.caves = caves
        for _ in range(g.get("flickering_stones", 0)):
            self._scatter(1, "Q")
        # the walled garden and its heavy gate (two must push together), in the open, gate toward the centre
        o = g.get("orchard")
        if o:
            n = o["size"]
            for _ in range(600):
                ox, oy = R.choice(inner)
                mx, my = ox + n // 2, oy + n // 2
                if all((xx, yy) in land for xx in (ox - 2, ox + n + 1) for yy in (oy - 2, oy + n + 1)) \
                        and o["min_distance"] <= abs(mx - cx) + abs(my - cy) <= o.get("max_distance", 99) \
                        and self.biome[my][mx] in ("meadow", "forest"):
                    break
            self.orchard = (ox, oy, ox + n - 1, oy + n - 1)
            for y in range(oy, oy + n):
                for x in range(ox, ox + n):
                    self.grid[y][x] = "|" if x in (ox, ox + n - 1) or y in (oy, oy + n - 1) else "."
            if abs(cx - mx) > abs(cy - my):
                gx, gy, step = (ox + n - 1, my, (1, 0)) if cx > mx else (ox, my, (-1, 0))
            else:
                gx, gy, step = (mx, oy + n - 1, (0, 1)) if cy > my else (mx, oy, (0, -1))
            self.grid[gy][gx] = "G"
            for k in (1, 2):
                self.grid[gy + step[1] * k][gx + step[0] * k] = "."
            inside = [(x, y) for y in range(oy + 1, oy + n - 1) for x in range(ox + 1, ox + n - 1)]
            for x, y in R.sample(inside, o["apple_trees"]):
                if (x, y) != (gx - step[0], gy - step[1]):
                    self.grid[y][x] = "A"
        # the shore spawn: along rays from the centre, the last beach tile before the sea, facing inland;
        # body k takes the angle a0 + k x the golden angle (well spread for any number of minds), or the arc's k-th
        if not arc:
            a0 = R.uniform(0, 2 * math.pi)
        for k in range(len(arc) or 16):
            a = arc[k] if arc else a0 + k * math.radians(137.508)
            spot = None
            for i in range(int(rad * 2), 0, -1):
                x, y = round(cx + math.cos(a) * i), round(cy + math.sin(a) * i)
                if 0 <= x < W and 0 <= y < H and self.grid[y][x] in "so" and (x, y) not in [s[1:3] for s in self.shore]:
                    spot = (x, y)
                    break
            if spot:
                x, y = spot
                self.grid[y][x] = "s"
                dx, dy = cx - x, cy - y
                facing = ("east" if dx > 0 else "west") if abs(dx) > abs(dy) else ("south" if dy > 0 else "north")
                self.shore.append((a, x, y, facing))
        # food on the shore (spec generate `shore_food`: tile; each spot a body may wake on gets `each` within
        # `within` steps, for the first `near_spawn` spots; `more` elsewhere): on beach tiles at the sea's edge
        sf = g.get("shore_food")
        if sf:
            spawn = {(x, y) for _, x, y, _ in self.shore}
            edge = [(x, y) for x, y in coastal if self.grid[y][x] in "so" and (x, y) not in spawn
                    and any(self.grid[y + dy][x + dx] == "~" for dx, dy in DIRS.values())]
            if (g.get("spawn") or {}).get("open"):          # 3m P2: never beside a waking spot (isl10: wob's was walled)
                edge = [q for q in edge if all(abs(q[0] - a) + abs(q[1] - b) > 1 for a, b in spawn)]
            for _, sx, sy, _ in self.shore[: int(sf.get("near_spawn", 0))]:
                near = [q for q in edge if abs(q[0] - sx) + abs(q[1] - sy) <= int(sf["within"])
                        and self.grid[q[1]][q[0]] != sf["tile"]]
                for x, y in R.sample(near, min(len(near), int(sf["each"]))):
                    self.grid[y][x] = sf["tile"]
            rest = [q for q in edge if self.grid[q[1]][q[0]] != sf["tile"]]
            for x, y in R.sample(rest, min(len(rest), int(sf.get("more", 0)))):
                self.grid[y][x] = sf["tile"]
        # a spring between each two neighbouring waking spots (3m M2, spec `springs.coast.between_spawn`: water shared,
        # as at a well; isl4-isl8, every spot had its own and no mind ever saw another): a few steps in from the beach,
        # on the tile the two reach on foot in the most even number of steps, the fewer the better, never one that cuts
        # a way between spots; after the shore food, whose rocks narrow the beach
        if between:
            def on_foot(s):
                dist, q = {s: 0}, [s]
                for p in q:
                    for dx, dy in DIRS.values():
                        n = (p[0] + dx, p[1] + dy)
                        if n not in dist and 0 <= n[0] < W and 0 <= n[1] < H and not self.solid(*n):
                            dist[n] = dist[p] + 1
                            q.append(n)
                return dist
            lo, hi = sc["inland"]
            beach = set(coastal)
            ox0, oy0, ox1, oy1 = self.orchard if o else (-9, -9, -9, -9)
            spots = {(x, y) for _, x, y, _ in self.shore}
            ok = [(x, y) for x, y in land if not self.solid(x, y) and (x, y) not in spots
                  and not (ox0 - 1 <= x <= ox1 + 1 and oy0 - 1 <= y <= oy1 + 1)
                  and lo <= min(max(abs(x - bx), abs(y - by)) for bx, by in beach) <= hi]
            for k in range(min(int(sc["count"]), len(self.shore) - 1)):
                da, db = on_foot(self.shore[k][1:3]), on_foot(self.shore[k + 1][1:3])
                both = [p for p in ok if p in da and p in db]
                if not both:
                    continue
                spots_a = {q for q in spots if q in da} | {q for q in spots if q in db}
                for x, y in sorted(both, key=lambda p: (max(da[p], db[p]), abs(da[p] - db[p]))):
                    was = self.grid[y][x]
                    self.grid[y][x] = g["springs"]["tile"]
                    ak, bk = on_foot(self.shore[k][1:3]), on_foot(self.shore[k + 1][1:3])
                    if spots_a <= set(ak) | set(bk) and ak.get(self.shore[k + 1][1:3], 99) <= da[self.shore[k + 1][1:3]] + 2:
                        break                                   # it cuts no way between the waking spots
                    self.grid[y][x] = was
                else:
                    continue
                springs.append((x, y))
                ok.remove((x, y))
                for dx, dy in DIRS.values():
                    if self.grid[y + dy][x + dx] != "~" and self.tiles[self.grid[y + dy][x + dx]].get("solid"):
                        self.grid[y + dy][x + dx] = "."
            self.springs = springs
        # the waking spots' guarantee (3m P2, spec `spawn`: `open` free sides at least, fresh water within
        # `water_within` steps on foot, the next spot within `neighbour_x` x its distance as the crow flies): where
        # the land fails it, a way is opened, the fewest trees and stones cleared (never the sea, water, walls, caves
        # or the shore food): isl6b rux's corner, isl8-isl9 a tree maze, isl10 a walled strip, seeds 1-4 walled spots
        gs = g.get("spawn") or {}
        if gs.get("open") or gs.get("water_within"):
            self._spawn_guarantee(gs, land, set(coastal))
        # mist banks over parts of the interior, away from the shore
        fog = self.spec.get("fog", {})
        if fog.get("banks"):
            cells = [(x, y) for x, y in inner if interior(x, y, 2)]
            for _ in range(int(fog["banks"])):
                bx, by = R.choice(cells)
                br = R.randint(*fog["radius"])
                self.mist |= {(x, y) for x, y in land if (x - bx) ** 2 + (y - by) ** 2 <= br * br}
            self.mist -= {(x, y) for _, x, y, _ in self.shore}
        # things lying about (spec generate `finds`: {item, count, and where: "shore" / biomes / under the mist})
        spawn = {(x, y) for _, x, y, _ in self.shore}
        for f in g.get("finds", []):
            spots = [(x, y) for y in range(1, H - 1) for x in range(1, W - 1)
                     if not self.tiles[self.grid[y][x]].get("solid") and (x, y) not in self.items and (x, y) not in spawn
                     and (self.biome[y][x] == "shore" if f.get("where") == "shore" else
                          (x, y) in self.mist if f.get("where") == "mist" else
                          self.biome[y][x] in f["biomes"] if f.get("biomes") else self.biome[y][x] is not None)]
            for q in R.sample(spots, min(len(spots), int(f["count"]))):
                self.items[q] = f["item"]
        # creatures, in their biomes (never on a spawn spot)
        for kind, sp in self.cspec.items():
            spots = [(x, y) for y in range(1, H - 1) for x in range(1, W - 1)
                     if self.biome[y][x] in sp.get("biomes", []) and self.grid[y][x] in ".,:sof"
                     and all(abs(x - a) + abs(y - b) > 3 for a, b in spawn)]
            R.shuffle(spots)
            for x, y in spots[: sp.get("count", 0)]:
                if (x, y) not in self.creatures:
                    self.creatures[(x, y)] = [kind, 0]

    def _spawn_guarantee(self, gs: dict, land: set, coastal: set) -> None:
        g, W, H = self.spec["generate"], self.W, self.H
        keep = {"~", "|", "G", "K", g.get("springs", {}).get("tile", "j"), g.get("pond", {}).get("tile", "w")}
        sf = g.get("shore_food") or {}
        keep |= {sf.get("tile", "P"), "p"}
        caves = {(cx + dx, cy + dy) for cx, cy in getattr(self, "caves", []) for dx in range(-1, 3) for dy in range(-1, 3)}
        clearable = lambda x, y: self.solid(x, y) and self.grid[y][x] not in keep and (x, y) not in caves
        fresh = {(x, y) for y in range(H) for x in range(W) if self.name_at(x, y) in ("spring", "water")}
        near_fresh = lambda p: any((p[0] + dx, p[1] + dy) in fresh for dx, dy in DIRS.values())

        def clear(x, y):
            self.grid[y][x] = "s" if (x, y) in coastal else "."

        def on_foot(s):
            dist, q = {s: 0}, [s]
            for p in q:
                for dx, dy in DIRS.values():
                    n = (p[0] + dx, p[1] + dy)
                    if n not in dist and 0 <= n[0] < W and 0 <= n[1] < H and not self.solid(*n):
                        dist[n] = dist[p] + 1
                        q.append(n)
            return dist

        def open_way(s, goal, limit):
            """A way from s to a tile goal() accepts within `limit` steps, the fewest trees and stones cleared."""
            layer, prev = {s: 0}, {}
            for step in range(1, int(limit) + 1):
                nxt = {}
                for p, c in layer.items():
                    for dx, dy in DIRS.values():
                        n = (p[0] + dx, p[1] + dy)
                        if not (0 <= n[0] < W and 0 <= n[1] < H) or (n not in land and n not in coastal):
                            continue
                        if self.solid(*n) and not clearable(*n):
                            continue
                        nc = c + (1 if self.solid(*n) else 0)
                        if nc < nxt.get(n, 1e9):
                            nxt[n], prev[(n, step)] = nc, (p, step - 1)
                done = [n for n in nxt if goal(n)]
                if done:
                    n, k = min(done, key=nxt.get), step
                    while k > 0:
                        if self.solid(*n):
                            clear(*n)
                        n, k = prev[(n, k)]
                    return True
                layer = nxt
            return False

        spots = [(x, y) for _, x, y, _ in self.shore[: int(sf.get("near_spawn", 8))]]
        for k, (x, y) in enumerate(spots):
            sides = [(x + dx, y + dy) for dx, dy in DIRS.values()]
            for n in sorted((n for n in sides if clearable(*n)), key=lambda n: n not in coastal):
                if sum(not self.solid(*m) for m in sides) >= int(gs.get("open", 0)):
                    break
                clear(*n)
            within = int(gs.get("water_within", 0))
            if within:
                d = on_foot((x, y))
                if min((v for p, v in d.items() if near_fresh(p)), default=1e9) > within:
                    open_way((x, y), near_fresh, within)
            if gs.get("neighbour_x") and k + 1 < len(spots):
                nb = spots[k + 1]
                crow = abs(nb[0] - x) + abs(nb[1] - y)
                if on_foot((x, y)).get(nb, 1e9) > float(gs["neighbour_x"]) * crow:
                    open_way((x, y), lambda p: p == nb, float(gs["neighbour_x"]) * crow)

    def _fill_cell(self, x, y, fill: dict) -> None:
        r = self.rng.random()
        acc = 0.0
        for code, pr in fill.items():
            acc += pr
            if r < acc:
                self.grid[y][x] = code
                return

    def _load_map(self) -> None:
        """A drawn map instead of a generated one (the trial rooms of brain/trials.py):
        `map` rows of tile codes, `shelter` [x0, y0, x1, y1], optional `spawn`."""
        self.grid = [list(row) for row in self.spec["map"]]
        self.shelter = tuple(self.spec.get("shelter", (-1, -1, -1, -1)))
        self.orchard = tuple(self.spec.get("orchard", (-1, -1, -1, -1)))
        self.door = next(((x, y) for y, row in enumerate(self.grid)
                          for x, c in enumerate(row) if c == "D"), None)

    # -- minds in the land -------------------------------------------------------
    def join(self, key: str, papers: list | None = None) -> "LandView":
        if key not in self.bodies:
            used = {b.mark for b in self.bodies.values()}
            mark = next((c for c in COLOURS if c not in used), f"number {len(self.bodies) + 1}")
            if self.spec.get("spawn"):                   # a drawn map says where minds wake
                x, y, facing = self.spec["spawn"][len(self.bodies) % len(self.spec["spawn"])]
            elif self.shore:                             # the island: spread along the shore, facing inland
                taken = {(b.x, b.y) for b in self.bodies.values()}
                _, x, y, facing = next((s for s in self.shore if s[1:3] not in taken), self.shore[0])
            else:
                x0, y0, x1, y1 = self.shelter
                spots = [(x, y) for y in range(y0 + 2, y1) for x in range(x0 + 1, x1)
                         if self.grid[y][x] == "_" and not any((b.x, b.y) == (x, y) for b in self.bodies.values())]
                (x, y), facing = spots[len(self.bodies) % len(spots)], "south"
            self.bodies[key] = LandBody(key, mark, x, y)
            self.bodies[key].facing = facing
        body = self.bodies[key]
        body.papers = papers
        body.seen_food.clear()                           # a new day: where food lay yesterday is gone
        return LandView(self, body)

    def _here(self, body) -> list:
        return [o for o in self.bodies.values() if o is not body and abs(o.x - body.x) + abs(o.y - body.y) <= 1]

    def zone(self, x: int, y: int) -> str:
        x0, y0, x1, y1 = self.shelter
        if x0 <= x <= x1 and y0 <= y <= y1:
            return "shelter"
        a, b, c, d = self.orchard
        if a <= x <= c and b <= y <= d:
            return "orchard"
        cx, cy = self.W // 2, self.H // 2
        if abs(x - cx) + abs(y - cy) <= 8:
            return "near the shelter" if x0 >= 0 else "the middle"
        return f"the {_compass(x - cx, y - cy)}"

    def _zone_phrase(self, x, y) -> str:
        z = self.zone(x, y)
        where = self.spec.get("land_word", "land")
        return {"shelter": "in the shelter", "orchard": "in the walled orchard",
                "near the shelter": "near the shelter"}.get(z, f"in {z} of the {where}")

    # -- time --------------------------------------------------------------------
    def season(self) -> dict:
        """The season's settings (spec `seasons`: `days` a season, `order`, `each` by name), with its
        `name`, or {} in a land without seasons. Never told to the mind: it feels the air."""
        ss = self.spec.get("seasons")
        if not ss:
            return {}
        day = self.clock // self.spec["day_ticks"]
        name = ss["order"][(day // int(ss["days"])) % len(ss["order"])]
        return {"name": name, **ss["each"][name]}

    def phase(self) -> str:
        f = (self.clock % self.spec["day_ticks"]) / self.spec["day_ticks"]
        if f >= self.season().get("night_from", self.spec["night_from"]):
            return "night"
        return "dawn" if f < 0.08 else "day" if f < 0.45 else "evening"

    def _dark(self, b) -> float:
        """How dark it is for this body (the world's `dark` event, which melatonin reads): full at
        night, less by a fire's light, a little in the evening, none by day."""
        ph = self.phase()
        if ph == "night":
            return 0.5 if self._sight(b) > self.spec["sight"]["night"] else 1.0
        return 0.3 if ph == "evening" else 0.0

    def ambient(self, b) -> list:
        """What still reaches a sleeping body (spec `asleep_feels`): the dark through closed eyes,
        the night's chill or a roof's warmth. Empty in lands that do not declare it."""
        feels = self.spec.get("asleep_feels", [])
        out, b.pending = list(b.pending), []              # a blow reaches a sleeper (and wakes it: `rouse`)
        if "dark" in feels and (d := self._dark(b)):
            out.append(event(self.name, "dark", amount=d))
        if "chill" in feels:
            out += [e for e in self._temperature(b) if e.kind in ("chill", "warmth")]
        return out

    def advance(self) -> None:
        """One tick of world time: things grow, fires burn down, gates swing shut."""
        self.clock += 1
        self.pushes.clear()
        self.last_round, self.happened = self.happened, []
        heal = self.spec.get("body", {})
        for b in self.bodies.values():
            if b.weak:
                b.weak = max(0.0, b.weak - float(heal.get("weak_heal_asleep" if b.asleep else "weak_heal", 0.004)))
        for (x, y), tm in list(self.timers.items()):
            name = self.name_at(x, y)
            rule = self.grow.get(name)
            if not rule:
                self.timers.pop((x, y), None)
                continue
            after = rule.get("watered_after", rule["after"]) if tm["watered"] else rule["after"]
            if name in self.spec.get("seasons", {}).get("grow_affects", ()):
                after *= float(self.season().get("grow", 1.0))   # plants grow slower in the cold months
            if self.clock - tm["since"] < after:
                continue
            need = rule.get("needs_water_within")
            if need and not tm["watered"] and not self._near(x, y, "water", need):
                continue                                   # dry: it waits
            if any((b.x, b.y) == (x, y) for b in self.bodies.values()) and self.info(rule["to"]).get("solid"):
                continue                                   # never grow a wall into someone
            self.set_tile(x, y, rule["to"])
        if self.creatures:
            self._creatures_live()
        if self.spec.get("weather"):
            self._weather()
        if self.spec.get("events"):
            self._rare_events()

    # -- the wild land: creatures and rain -------------------------------------------
    def _walkable_for(self, kind: str, x: int, y: int) -> bool:
        if not (1 <= x < self.W - 1 and 1 <= y < self.H - 1) or (x, y) in self.creatures:
            return False
        if any((o.x, o.y) == (x, y) for o in self.bodies.values()) or (x, y) in self.items:
            return False
        x0, y0, x1, y1 = self.shelter
        if x0 - 1 <= x <= x1 + 1 and y0 - 1 <= y <= y1 + 1:
            return False                                   # creatures keep off the shelter and its doorstep
        ground = self.cspec.get(kind, {}).get("ground") or self.spec.get("creature_ground", sorted(GROUND))
        return self.tiles[self.grid[y][x]]["name"] in ground

    def _creature_step(self, x, y, away_from=None, tries=1) -> bool:
        kind = self.creatures[(x, y)][0]
        options = [(x + dx, y + dy) for dx, dy in DIRS.values() if self._walkable_for(kind, x + dx, y + dy)]
        if not options:
            return False
        if away_from:
            ax, ay = away_from
            options.sort(key=lambda q: -(abs(q[0] - ax) + abs(q[1] - ay)))
            options = options[:2]
        nx, ny = self.rng.choice(options)
        self.creatures[(nx, ny)] = self.creatures.pop((x, y))
        if tries > 1:
            self._creature_step(nx, ny, away_from, tries - 1)
        return True

    def _creatures_live(self) -> None:
        for (x, y), c in list(self.creatures.items()):
            if self.creatures.get((x, y)) is not c:
                continue
            g = self.cgrow.get(c[0])
            if g and self.clock - c[1] >= g["after"]:
                c[0], c[1] = g["to"], self.clock
            kind = c[0]
            sp = self.cspec.get(kind, {})
            lay = sp.get("lays")
            if lay and self.rng.random() < 1.0 / float(lay["every"]):
                spots = [(x + dx, y + dy) for dx, dy in DIRS.values()
                         if self._walkable_for(kind, x + dx, y + dy)]
                if spots:
                    self.items[self.rng.choice(spots)] = lay["item"]
            flee = int(sp.get("flee", 0))
            if flee:
                near = [o for o in self.bodies.values() if not o.asleep and abs(o.x - x) + abs(o.y - y) <= flee]
                if near and self.rng.random() < 0.9:
                    self._creature_step(x, y, away_from=(near[0].x, near[0].y))
                    continue
            if len(c) > 2 and c[2] > 0:
                c[2] = max(0.0, c[2] - 0.002)             # a wound heals
            ch = sp.get("charges")
            if ch:
                near = [o for o in self.bodies.values() if abs(o.x - x) + abs(o.y - y) == 1]
                angry = len(c) > 3 and self.clock < c[3]
                if near and self.rng.random() < float(ch["p_angry" if angry else "p"]):
                    o = self.rng.choice(near)
                    hit = float(ch["hurt"])
                    o.weak = min(1.0, o.weak + hit * float(self.spec.get("strike", {}).get("weakens", 0.8)))
                    o.pending.append(event(self.name, "hurt", amount=hit, rouse=1.0))
                    o.inbox.append((f"{_a(kind).capitalize()} charged at me and struck me! It hurts.", True, 0.0))
                    for w in self.bodies.values():
                        if w is not o and not w.asleep and abs(w.x - x) + abs(w.y - y) <= self.spec["hearing"]["words_with_mark"]:
                            w.inbox.append((f"{_a(kind).capitalize()} charged at {self.someone(o).lower()}.", False, 0.0))
                    self.note("charged", what=kind, to=o.key, hit=hit)
                    continue
            if self.rng.random() < float(sp.get("wander", 0.0)):
                self._creature_step(x, y)
        self._creatures_breed()

    def _creatures_breed(self) -> None:
        """Creatures breed back toward their number (spec creature `breed`: {every, max}; a kind's
        `family` counts its other forms, a shorn sheep is a sheep): a young one appears beside one of
        them; a kind gone from the land comes back, rarely, somewhere in its biomes."""
        for kind, sp in self.cspec.items():
            br = sp.get("breed")
            if not br or self.rng.random() >= 1.0 / float(br["every"]):
                continue
            fam = [p for p, c in self.creatures.items() if self.cspec.get(c[0], {}).get("family", c[0]) == kind]
            if len(fam) >= int(br.get("max", sp.get("count", 0))):
                continue
            if fam:
                x, y = self.rng.choice(fam)
                spots = [(x + dx, y + dy) for dx, dy in DIRS.values() if self._walkable_for(kind, x + dx, y + dy)]
            elif self.rng.random() < 0.25 and getattr(self, "biome", None):
                spots = [(x, y) for y in range(1, self.H - 1) for x in range(1, self.W - 1)
                         if self.biome[y][x] in sp.get("biomes", []) and self._walkable_for(kind, x, y)]
            else:
                spots = []
            if spots:
                self.creatures[self.rng.choice(spots)] = [kind, self.clock]

    def _weather(self) -> None:
        w, sp = self.weather, self.spec["weather"]
        if not w["rain"] and self.clock >= w["next"]:
            w["rain"], w["until"] = True, self.clock + self.rng.randint(*sp["rain_lasts"])
        elif w["rain"] and self.clock >= w["until"]:
            gap = self.rng.randint(*sp["rain_every"]) * float(self.season().get("rain_every", 1.0))
            w["rain"], w["next"] = False, self.clock + int(gap)
        if not w["rain"]:
            return
        for (x, y), tm in list(self.timers.items()):
            name = self.tiles[self.grid[y][x]]["name"]
            if name == "fire" and self.rng.random() < float(sp.get("quench", 0.02)):
                self.set_tile(x, y, "ashes")               # rain puts fires out
            elif name in ("sown soil", "sprouts"):
                tm["watered"] = True                       # and waters what was sown
        for (x, y), it in list(self.items.items()):
            if it == "bowl" and not self.tiles[self.grid[y][x]].get("roof") \
                    and self.rng.random() < float(sp.get("fill_bowls", 0.02)):
                self.items[(x, y)] = "bowl of water"       # a bowl left out fills

    # -- rare events (P3, 3g): a storm, a wreck washing up, a spring drying up, a creature arriving --------
    def _rare_events(self) -> None:
        """Spec `events`: {kind: {per_day, lasts: [a, b] ticks, ...}}. One of each kind at a time; each
        starts with p = per_day / day_ticks a tick. Minds only perceive them (never announced)."""
        day = float(self.spec["day_ticks"])
        for kind, ev in self.events.items():
            if ev.get("until") is not None and self.clock >= ev["until"]:
                self._event_end(kind, ev)
        self.events = {k: v for k, v in self.events.items() if v.get("until") is None or self.clock < v["until"]}
        for kind, sp in self.spec["events"].items():
            if kind.startswith("_") or kind in self.events or self.rng.random() >= float(sp["per_day"]) / day:
                continue
            ev = self._event_start(kind, sp)
            if ev is not None:
                if ev.get("until") is not None:             # one that lasts (a storm, a drought); a wreck is over at once
                    self.events[kind] = ev
                self.note("event", what=kind, **{k: v for k, v in ev.items() if k != "until"})
        if "storm" in self.events:
            sp = self.spec["events"]["storm"]
            self.weather["rain"] = True
            self.weather["until"] = max(self.weather.get("until", 0), self.events["storm"]["until"])
            frail = {c for c, t in self.tiles.items() if t.get("fragile")}
            for x, y in [(x, y) for y in range(self.H) for x in range(self.W) if self.grid[y][x] in frail]:
                f = self.tiles[self.grid[y][x]]["fragile"]
                if self.rng.random() < float(f["p"]):
                    self.set_tile(x, y, f["becomes"])        # the wind tears it down
                    for i, it in enumerate(f.get("leaves", [])):
                        spot = [(x, y)] + [(x + dx, y + dy) for dx, dy in DIRS.values()]
                        spot = [q for q in spot if q not in self.items and not self.solid(*q)]
                        if spot:
                            self.items[spot[0]] = it
                    self.note("event", what="blown down", at=[x, y])

    def _beach(self) -> list:
        return [(x, y) for y in range(self.H) for x in range(self.W) if self.grid[y][x] == "s"
                and any(self.grid[y + dy][x + dx] == "~" for dx, dy in DIRS.values()
                        if 0 <= x + dx < self.W and 0 <= y + dy < self.H)]

    def _event_start(self, kind: str, sp: dict) -> dict | None:
        until = self.clock + self.rng.randint(*sp["lasts"]) if sp.get("lasts") else None
        if kind == "storm":
            return {"until": until}
        if kind == "wreck":                                # things washed up on a stretch of beach
            beach = [q for q in self._beach() if q not in self.items]
            if not beach:
                return None
            x, y = self.rng.choice(beach)
            things = self.rng.sample(sp["things"], min(len(sp["things"]), self.rng.randint(*sp["count"])))
            spots = sorted((q for q in beach if abs(q[0] - x) + abs(q[1] - y) <= 4), key=lambda q: abs(q[0] - x) + abs(q[1] - y))
            placed = []
            for it, q in zip(things, spots):
                self.items[q] = it
                placed.append(it)
            return {"until": None, "at": [x, y], "things": placed}
        if kind == "drought":                              # a spring dries up for a while
            springs = [(x, y) for y in range(self.H) for x in range(self.W) if self.name_at(x, y) == sp["tile"]]
            if not springs:
                return None
            x, y = self.rng.choice(springs)
            self.set_tile(x, y, sp["becomes"])
            return {"until": until, "at": [x, y]}
        if kind == "arrival":                              # a creature comes ashore
            kind2 = self.rng.choice(sp["kinds"])
            beach = [q for q in self._beach() if self._walkable_for(kind2, *q)]
            if not beach:
                return None
            q = self.rng.choice(beach)
            self.creatures[q] = [kind2, self.clock]
            return {"until": None, "creature": kind2, "at": list(q)}
        return None

    def _event_end(self, kind: str, ev: dict) -> None:
        if kind == "drought":
            x, y = ev["at"]
            self.set_tile(x, y, self.spec["events"]["drought"]["tile"])
        self.note("event_end", what=kind)

    def _near(self, x, y, name, r) -> bool:
        return any(self.name_at(x + dx, y + dy) == name
                   for dy in range(-r, r + 1) for dx in range(-r, r + 1))

    def _warm_here(self, b) -> float:
        if self.info(self.name_at(b.x, b.y)).get("roof"):
            return self.spec["warmth"]["roof_warmth"]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                w = self.info(self.name_at(b.x + dx, b.y + dy)).get("warm")
                if w:
                    return w
        return 0.0

    def _sight(self, b) -> int:
        s = self.spec["sight"]
        if self.phase() != "night":
            return s["day"]
        # a light source within its reach (P3 fix: `light 0 >= distance 0` made every body's own tile a
        # light, so no night was ever dark, in any land)
        lit = any((L := self.info(self.name_at(b.x + dx, b.y + dy)).get("light", 0)) > 0 and L >= max(abs(dx), abs(dy))
                  for dy in range(-2, 3) for dx in range(-2, 3))
        return 2 if lit else s["night"]

    # -- describing --------------------------------------------------------------
    def front(self, b) -> tuple[int, int]:
        dx, dy = DIRS[b.facing]
        return b.x + dx, b.y + dy

    def body_at(self, x, y, but=None):
        return next((o for o in self.bodies.values() if o is not but and (o.x, o.y) == (x, y)), None)

    def other_in_front(self, b):
        """The one in front of this body, else one on the same tile (lands where bodies do not block)."""
        return self.body_at(*self.front(b), but=b) or self.body_at(b.x, b.y, but=b)

    def note(self, kind: str, **data) -> None:
        self.happened.append({"t": self.clock, "kind": kind, **data})

    def thing(self, x, y) -> str:
        """What is at a tile, as the rules name it: a creature, an item lying there, else the tile."""
        if self.name_at(x, y) == "mist":                  # nothing shows through the mist
            return "mist"
        c = self.creatures.get((x, y))
        if c:
            return c[0]
        it = self.items.get((x, y))
        return f"item:{it}" if it else self.name_at(x, y)

    def describe(self, x, y) -> str:
        if self.name_at(x, y) == "mist":
            return self.info("mist").get("desc", "mist")
        c = self.creatures.get((x, y))
        if c:
            return self.cspec.get(c[0], {}).get("desc", _a(c[0]))
        it = self.items.get((x, y))
        t = self.name_at(x, y)
        desc = self.info(t).get("desc", t)
        return f"{_a(it)} lying on the {t}" if it else (desc if desc.startswith(("a ", "an ", "the ")) else desc)

    def short(self, x, y) -> str:
        if self.name_at(x, y) == "mist":
            return "mist"
        c = self.creatures.get((x, y))
        if c:
            return _a(c[0])
        it = self.items.get((x, y))
        return _a(it) if it else self.name_at(x, y)

    def carry(self, b) -> str:
        if not b.inv:
            return "I carry nothing."
        counts = {}
        for i in b.inv:
            counts[i] = counts.get(i, 0) + 1
        s = "I carry: " + ", ".join(f"{k} ({n})" for k, n in counts.items()) + "."
        return s + (f" I hold the {b.held}." if b.held else " My hands are free.")

    @staticmethod
    def someone(b) -> str:
        return f"Someone with {_a(b.mark)} mark"

    def _sense(self, b):
        x, y = b.x, b.y
        fx, fy = self.front(b)
        o = self.body_at(fx, fy, but=b)
        focus = f"facing {'someone' if o else self.thing(fx, fy).replace('item:', '')}, holding {b.held or 'nothing'}"
        lines = [f"I am {self._zone_phrase(x, y)}, on the {self.name_at(x, y)}, facing {b.facing}.",
                 f"In front of me: {self.someone(o).lower() if o else self.describe(fx, fy)}."]
        if (x, y) in self.items:                  # a thing walked onto is still there, at my feet (10aa)
            lines[0] += f" At my feet: {_a(self.items[(x, y)])}."
        sides = [f"{d} {self.short(x + dx, y + dy)}" for d, (dx, dy) in DIRS.items() if d != b.facing]
        lines.append("Around me: " + ", ".join(sides) + ".")
        ph = self.phase()
        air = self.season().get("air")
        lines.append(f"It is {ph}." + (" It is dark; I can see only what is close." if ph == "night"
                                         and self._sight(b) <= 1 else "")
                     + ((" " + self.spec["events"]["storm"]["felt"]) if "storm" in self.events
                        else " It is raining." if self.weather["rain"] else "") + (f" {air}" if air else ""))
        lines.append(self.carry(b))
        # a move goes where the mind may not be facing: each move's own focus is what lies
        # that way (step 10f), so a wall to the north is predictable when going north
        by_verb = {d: f"{d}: {self.short(x + dx, y + dy)}" for d, (dx, dy) in DIRS.items()}
        out = [event(self.name, "percept", text=lines[0], focus=focus, focus_by_verb=by_verb,
                     food_by_verb=self._food_by_verb(b), food_far_by_verb=self._food_far_by_verb(b),
                     drink_by_verb=self._drink_by_verb(b))]
        out += [event(self.name, "percept", text=t) for t in lines[1:]]
        sight = self._sight(b)
        near = False
        for o in self.bodies.values():
            if o is b:
                continue
            dx, dy = o.x - x, o.y - y
            d = abs(dx) + abs(dy)
            still = ", lying still, as if asleep" if o.asleep else ""
            if d <= 1:
                near = True
                side = "" if d == 0 else f", just to the {_compass(dx, dy)}"
                out.append(event(self.name, "percept", text=f"{self.someone(o)} is here{still}{side}."))
            elif d <= sight:
                out.append(event(self.name, "percept", text=f"{self.someone(o)} is {_where(dx, dy)} from me{still}."))
        if near:
            out.append(event(self.name, "contact", amount=0.02))
        for text, attention, contact in b.inbox:
            out.append(event(self.name, "percept", text=text, attention=attention))
            if contact:
                out.append(event(self.name, "contact", amount=contact))
        b.inbox.clear()
        out += b.pending
        b.pending = []
        felt = next((t for lvl, t in reversed(self.spec.get("body", {}).get("weak_felt", [])) if b.weak >= lvl), None)
        for t in (felt, self._load_felt(b)):
            if t:
                out.append(event(self.name, "percept", text=t))
        out += self._temperature(b)
        if "dark" in self.spec.get("asleep_feels", []) and (d := self._dark(b)):
            out.append(event(self.name, "dark", amount=d))
        return out

    def _temperature(self, b) -> list:
        """Chill and warmth on the body this tick: rain outside chills, a roof or a fire warms,
        the night outside chills, the day warms a little (both scaled by the season)."""
        out, se = [], self.season()
        w = self._warm_here(b)
        if self.weather.get("rain") and self.spec.get("weather") and not self.info(self.name_at(b.x, b.y)).get("roof"):
            out.append(event(self.name, "chill", amount=float(self.spec["weather"].get("chill", 0.001))))
            if "storm" in self.events:
                out.append(event(self.name, "chill", amount=float(self.spec["events"]["storm"].get("chill", 0.004))))
        if w:
            out.append(event(self.name, "warmth", amount=w))
        elif self.phase() == "night":
            out.append(event(self.name, "chill",
                             amount=self.spec["warmth"]["night_outside_chill"] * float(se.get("chill", 1.0))))
        else:
            out.append(event(self.name, "warmth",
                             amount=self.spec["warmth"]["day_warmth"] * float(se.get("warmth", 1.0))))
        return out

    def _food_sources(self) -> set:
        """What gives food by hand (a `take` rule with no tool whose gifts include a food)."""
        if getattr(self, "_sources", None) is None:
            food = set(self.spec.get("food", {}))
            self._sources = {r["target"] for r in self.spec.get("take", [])
                             if "held" not in r and food & set(r.get("gives", []))}
        return self._sources

    def _food_at(self, x, y) -> bool:
        th = self.thing(x, y)
        if th.startswith("item:"):
            it = th[5:]
            return bool(set(self.spec.get("pickup", {}).get(it, [it])) & set(self.spec.get("food", {})))
        return th in self._food_sources()

    def _drinkable_at(self, x, y) -> bool:
        return self.thing(x, y) in {r["target"] for r in self.spec.get("drink", [])
                                    if r.get("target") and float(r.get("amount", 0)) > 0}

    def _drink_by_verb(self, b) -> dict:
        """The acts that lead to something that quenches, in sight (3n I5, isl11: abu, very thirsty, walked past a
        spring beside it): in front, drink; beside, turn that way; held, drink. As `_food_by_verb` for food: the
        text already names what is there; this only says which acts lead to it."""
        out = {}
        fx, fy = self.front(b)
        for d, (dx, dy) in DIRS.items():
            if (b.x + dx, b.y + dy) != (fx, fy) and self._drinkable_at(b.x + dx, b.y + dy):
                out[d] = self.short(b.x + dx, b.y + dy)
        if self._drinkable_at(fx, fy):
            out["drink"] = self.short(fx, fy)
        elif b.held and any(r.get("held") == b.held and float(r.get("amount", 0)) > 0 for r in self.spec.get("drink", [])):
            out["drink"] = b.held
        return out

    def _food_by_verb(self, b) -> dict:
        """The acts that lead to food in sight (step 10x): the sight of it, for the brain's food cue.
        Food within reach in front: take; beside: turn that way; held: eat; carried: switch to it.
        The text already names what is there; this only says which acts lead to it."""
        food = self.spec.get("food", {})
        out = {}
        fx, fy = self.front(b)
        for d, (dx, dy) in DIRS.items():
            if (b.x + dx, b.y + dy) != (fx, fy) and self._food_at(b.x + dx, b.y + dy):
                out[d] = self.short(b.x + dx, b.y + dy)
        if self._food_at(fx, fy):
            out["take"] = self.short(fx, fy)
        elif self._food_at(b.x, b.y):             # at my feet (10aa)
            out["take"] = self.short(b.x, b.y)
        if b.held in food:
            out["eat"] = b.held
        elif any(i in food for i in b.inv):
            out["switch"] = next(i for i in b.inv if i in food)
        return out

    def _clear_line(self, b, x, y) -> bool:
        """Nothing solid strictly between the body and (x, y), on the straight line."""
        dx, dy = x - b.x, y - b.y
        n = max(abs(dx), abs(dy))
        for i in range(1, n):
            cx, cy = b.x + round(dx * i / n), b.y + round(dy * i / n)
            if (cx, cy) != (x, y) and self.solid(cx, cy):
                return False
        return True

    def _in_view(self, b, x, y) -> bool:
        """Seen from where the body stands (step 10ab; for `look` since 10ac, owner, after s15: it
        listed what lay beyond walls, a bush outside the room). In view: a clear line to it, or it
        borders open ground that has one (a door in the room's wall, seen at a slant along it)."""
        if self._clear_line(b, x, y):
            return True
        return any(0 <= x + ox < self.W and 0 <= y + oy < self.H and not self.solid(x + ox, y + oy)
                   and (x + ox, y + oy) != (x, y) and self._clear_line(b, x + ox, y + oy)
                   for ox, oy in DIRS.values())

    def _food_far_by_verb(self, b) -> dict:
        """The step toward the nearest food seen by `look` and not yet in reach (step 10ab):
        the appetitive phase, approaching what was seen. {move: what}, or {} when nothing is
        remembered, the way is blocked, or it is already in reach (`_food_by_verb` says so).
        A spot is forgotten once seen empty from next to it, or after `food_memory` ticks."""
        keep = int(self.spec.get("food_memory", 60))
        best = None
        for (x, y), (what, at) in list(b.seen_food.items()):
            d = abs(x - b.x) + abs(y - b.y)
            if self.clock - at > keep or (d <= 1 and not self._food_at(x, y)):
                del b.seen_food[(x, y)]
            elif d >= 2 and (best is None or d < best[0]):
                best = (d, x, y, what)
        if best is None:
            return {}
        _, x, y, what = best
        dx, dy = x - b.x, y - b.y
        steps = sorted((d for d, (sx, sy) in DIRS.items() if sx * dx > 0 or sy * dy > 0),
                       key=lambda d: -abs(dx if DIRS[d][0] else dy))  # the longer way first
        for d in steps:
            nx, ny = b.x + DIRS[d][0], b.y + DIRS[d][1]
            if 0 <= nx < self.W and 0 <= ny < self.H and not self.solid(nx, ny) and (nx, ny) not in self.creatures:
                return {d: what}
        return {}

    # -- others ------------------------------------------------------------------
    def _tell(self, b, seen: str | None, voice: str | None = None, far: str | None = None,
              attention: bool = False, contact: float = 0.0) -> dict:
        """Deeds are seen within 3 steps (with the mark); a voice is heard from a direction
        within 8 steps; `far` goes to everyone else. Returns who got what (`seen` / `voice` / `far`)."""
        h = self.spec["hearing"]
        got = {}
        for o in self.bodies.values():
            if o is b or o.asleep:
                continue
            dx, dy = b.x - o.x, b.y - o.y
            d = abs(dx) + abs(dy)
            if seen and d <= h["words_with_mark"] and d <= max(1, self._sight(o)):
                o.inbox.append((seen, attention, contact))
                got[o.key] = "seen"
            elif voice and d <= h["words"]:
                o.inbox.append((voice.format(dir=_compass(dx, dy)), attention, contact / 2))
                got[o.key] = "voice"
            elif far:
                o.inbox.append((far, False, 0.0))
                got[o.key] = "far"
        return got

    # -- acting ------------------------------------------------------------------
    def _carry_max(self, b) -> int:
        """What a body can carry: less while it is weak (P3)."""
        return self.spec["carry_max"] - int(b.weak * float(self.spec.get("body", {}).get("weak_carry", 0)))

    def _too_much(self, b, what: str | None = None, adding: int = 1) -> str:
        """The carrying limit said as what it is (3k, isl6b: "My hands are full" named the hands for the load,
        and qam, carrying 10 reeds, learnt that mussels "cannot be collected if the hands are full")."""
        n, m = len(b.inv), self._carry_max(b)
        why = (f"I already carry {n} things, as many as I can" if n >= m
               else f"I already carry {n} things; {adding} more would be more than I can carry")
        return f"I cannot carry {what}: {why}." if what else f"I cannot carry any more: {why}."

    def _load_felt(self, b) -> str | None:
        """The load as a body sense (3k): near the carrying limit the arms feel it, as weakness is felt.
        Spec `body.load_felt` [[share of what can be carried, phrase], ...] or the default; nothing below."""
        felt = self.spec.get("body", {}).get("load_felt", LOAD_FELT)
        m = self._carry_max(b)
        share = len(b.inv) / m if m > 0 else 1.0
        return next((t for lvl, t in reversed(felt) if share >= lvl), None)

    def _give(self, b, items: list[str], to_hand: bool = False) -> str:
        """Into the body's carrying; held when the hands were empty, or, `to_hand` (a take, on a land whose spec
        says `take_to_hand`: isl4, food taken went behind a shell held and `eat what I hold` failed), the first
        thing gained is now the one held."""
        lost = []
        for it in items:
            if len(b.inv) < self._carry_max(b):
                b.inv.append(it)
                if b.held is None or to_hand:
                    b.held = it
                    to_hand = False
            else:
                lost.append(it)
        return (" " + self._too_much(b, "the " + ", ".join(lost) + " too")) if lost else ""

    def _drop_held(self, b) -> None:
        if b.held in b.inv:
            b.inv.remove(b.held)
        if b.held not in b.inv:
            b.held = b.inv[0] if b.inv else None

    def _rule(self, kind: str, b, target: str, ground: bool):
        """The first rule of `kind` matching what is held and what is in front."""
        rules = self.spec[kind]
        for want_held in (True, False):
            for i, r in enumerate(rules):
                if ("held" in r) != want_held or (want_held and r["held"] != b.held):
                    continue
                if r["target"] == target or (r["target"] == "@ground" and ground):
                    return i, r
        return None, None

    def _state_print(self, b) -> tuple:
        """What an act can change, apart from where the body stands and faces: the tiles, the
        things lying about, what this body carries (as a multiset: switching hands is no change)."""
        return ("".join("".join(r) for r in self.grid), tuple(sorted(self.items.items())),
                tuple(sorted(b.inv + ([b.held] if b.held and b.held not in b.inv else []))))

    def _act(self, b, verb: str, words: str | None = None):
        """An act, then `effect` if it changed the world or what the body holds (not a move, a
        look or a call): the raw material of effectance (acting with effect, White 1959)."""
        before, place = self._state_print(b), (b.x, b.y, b.facing)
        out = self._do(b, verb, words)
        effect = self._state_print(b) != before
        # whether the act changed anything at all, the body's place and facing and the land in view included:
        # the fast level carries on with an act while it does, and asks afresh when it did nothing (PLAN 1.1)
        # (a drink from a pond changes nothing outside the body: counted as a change, the fast level repeated it,
        # 49 drinks in a day in isl2; now the next tick asks afresh, and the thirst felt decides)
        changed = effect or (b.x, b.y, b.facing) != place or self._discovered
        for e in out:
            if e.kind == "outcome":
                e.data["changed"] = changed
        if effect:
            text = next((e.data.get("text", "") for e in out if e.kind == "outcome"), "")
            out.append(event(self.name, "effect", what=text))
        return out

    def _do(self, b, verb: str, words: str | None = None):
        streak, b.streak = b.streak, None
        self._discovered = False
        if verb in DIRS:
            text = self._move(b, verb)
        elif verb == "look":
            text = self._look(b)
        elif verb == "take":
            text = self._take(b)
        elif verb == "use":
            text = self._use(b, streak)
        elif verb == "switch":
            text = self._switch(b)
        elif verb == "eat":
            return self._eat(b)
        elif verb == "drink" and "drink" in self.verbs:
            return self._drink(b)
        elif verb == "give" and "give" in self.verbs:
            text = self._hand(b)
        elif verb == "strike" and "strike" in self.verbs:
            text = self._strike(b)
        elif verb == "build" and "build" in self.verbs:
            text = self._build(b)
        elif verb == "wait":
            text = "I waited. Time passed."
        elif verb == "speak":
            text = self._speak(b, words)
        else:
            text = f"I could not {verb} here."
        out = [event(self.name, "outcome", ok=True, text=text)]
        if self._discovered:                          # more land came into view (step 10x: discovery)
            out.append(event(self.name, "discovery", amount=1.0))
        return out

    def _move(self, b, d: str) -> str:
        b.facing = d
        dx, dy = DIRS[d]
        nx, ny = b.x + dx, b.y + dy
        if not (0 <= nx < self.W and 0 <= ny < self.H) or self.solid(nx, ny) or (nx, ny) in self.creatures:
            return f"I turned {d}; {self.describe(nx, ny)} is in the way."
        o = self.body_at(nx, ny, but=b)
        if o and self.spec.get("bodies_block"):
            # the one in the way feels it (3n I1, isl10: sut stood on wob's only way out for days and was never told)
            if not o.asleep:
                o.inbox.append((f"{self.someone(b)} bumped into me, trying to go {d}.", True, 0.02))
            self.note("blocked", by=b.key, by_whom=o.key)
            return f"I turned {d}; {self.someone(o).lower()} is in the way."
        if b.weak and self.rng.random() < b.weak * float(self.spec.get("body", {}).get("stumble", 0)):
            return f"I turned {d} and tried to go, but I stumbled; my body is too weak."
        before = self.zone(b.x, b.y)
        # where bodies do not block, one squeezes past another (3m P1, isl10: two bodies in a one-tile strip of beach
        # blocked each other for three days)
        past = f", squeezing past {self.someone(o).lower()}" if o else ""
        self._tell(b, f"{self.someone(b)} walked on to the {d}.")
        if o:
            self.note("squeeze", by=b.key, past=o.key, asleep=o.asleep)
        if o and not o.asleep:
            o.inbox.append((f"{self.someone(b)} squeezed past me.", True, 0.02))
        b.x, b.y = nx, ny
        text = f"I walked {d}{past}, onto the {self.name_at(nx, ny)}."
        if self.zone(nx, ny) != before:
            text += f" I am now {self._zone_phrase(nx, ny)}."
        return text + self._draw_back(nx, ny)

    def _draw_back(self, x, y) -> str:
        """The frontier: when a mind comes within `near` steps of the mist, it draws back by
        `step` on that side (the land was there all along, unseen)."""
        fog = self.spec.get("fog")
        if fog and fog.get("banks"):                  # the island: a bank lifts for good around a mind that comes near
            near, lift = int(fog["near"]), int(fog["lift"])
            if not any((x + dx, y + dy) in self.mist for dy in range(-near, near + 1) for dx in range(-near, near + 1)):
                return ""
            gone = {(a, b) for a, b in self.mist if (a - x) ** 2 + (b - y) ** 2 <= lift * lift}
            self.mist -= gone
            self._discovered = True
            sides = sorted({_compass(a - x, b - y) for a, b in gone} & {"north", "east", "south", "west"})
            return (" The mist thinned and drew back" + (f" to the {' and '.join(sides)}" if sides else "")
                    + ": there is land there I had not seen.")
        if not fog or not self.revealed:
            return ""
        x0, y0, x1, y1 = self.revealed
        near, step = int(fog["near"]), int(fog["step"])
        sides = []
        if x - x0 < near and x0 > 1:
            x0 = max(1, x0 - step)
            sides.append("west")
        if x1 - x < near and x1 < self.W - 2:
            x1 = min(self.W - 2, x1 + step)
            sides.append("east")
        if y - y0 < near and y0 > 1:
            y0 = max(1, y0 - step)
            sides.append("north")
        if y1 - y < near and y1 < self.H - 2:
            y1 = min(self.H - 2, y1 + step)
            sides.append("south")
        if not sides:
            return ""
        self.revealed = (x0, y0, x1, y1)
        self._discovered = True
        return f" The mist drew back to the {' and '.join(sides)}: there is more land there."

    def _look(self, b) -> str:
        fx, fy = self.front(b)
        front = self.name_at(fx, fy)
        if front == "shelf" and b.papers:
            b.shelf, b.page = (b.shelf + 1) % len(b.papers), 0
            p = b.papers[b.shelf]
            return f"I looked along the shelf. The next paper is titled '{p['title']}' ({len(p['pages'])} pages)."
        if front == "flickering stone":
            self.glyphs = self._glyphs()
            return f"I looked at the flickering stone; its glyphs now read {self.glyphs}."
        r = self._sight(b)
        seen, names = [], set()
        cells = sorted(((dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
                        if (dx or dy) and abs(dx) + abs(dy) <= r + 1), key=lambda p: abs(p[0]) + abs(p[1]))
        for dx, dy in cells:
            x, y = b.x + dx, b.y + dy
            what = self.short(x, y)
            if what in GROUND or what in ("floor", "tall grass", "flowers") or what in names:
                continue
            if not self._in_view(b, x, y):
                continue                                     # behind a wall, a closed door, a tree (10ac)
            names.add(what)
            seen.append(f"{what} {_where(dx, dy)}")
            if abs(dx) + abs(dy) >= 2 and self._food_at(x, y):
                b.seen_food[(x, y)] = (what, self.clock)     # where food was seen (step 10ab)
            if len(seen) >= 9:
                break
        self._tell(b, f"{self.someone(b)} looked around.")
        dark = " It is too dark to see further." if r <= 1 else ""
        return ("Looking around, I see: " + "; ".join(seen) + "." if seen
                else "Looking around, I see nothing but open ground.") + dark

    def _take(self, b) -> str:
        fx, fy = self.front(b)
        it = None if (fx, fy) in self.creatures else self.items.get((fx, fy))
        if it:
            got = self.spec.get("pickup", {}).get(it, [it])
            if len(b.inv) + len(got) > self._carry_max(b):
                return self._too_much(b, f"the {it}", len(got))
            del self.items[(fx, fy)]
            self._tell(b, f"{self.someone(b)} picked up {_a(it)}.")
            return f"I picked up the {it}." + self._give(b, got, self.spec.get("take_to_hand", False))
        target = self.creatures[(fx, fy)][0] if (fx, fy) in self.creatures else self.name_at(fx, fy)
        if target in self.code and self.info(target).get("store"):
            kept = self.stores.get((fx, fy), [])
            if not kept:
                return f"The {target} is empty."
            if len(b.inv) >= self._carry_max(b):
                return self._too_much(b, f"anything from the {target}")
            what = kept.pop()
            self._give(b, [what], self.spec.get("take_to_hand", False))
            self._tell(b, f"{self.someone(b)} took {_a(what)} from the {target}.")
            self.note("store_take", by=b.key, what=what, at=[fx, fy])
            return f"I took {_a(what)} from the {target}." + (f" It still holds {len(kept)}." if kept else " It is empty now.")
        i, r = self._rule("take", b, target, False)
        if r is None and (b.x, b.y) in self.items:   # nothing to take in front: what lies at my feet (10aa)
            it = self.items[(b.x, b.y)]
            got = self.spec.get("pickup", {}).get(it, [it])
            if len(b.inv) + len(got) > self._carry_max(b):
                return self._too_much(b, f"the {it}", len(got))
            del self.items[(b.x, b.y)]
            self._tell(b, f"{self.someone(b)} picked up {_a(it)}.")
            return f"I picked up the {it} at my feet." + self._give(b, got, self.spec.get("take_to_hand", False))
        if r is None:
            return f"I cannot take anything from the {target}."
        if r.get("chance") is not None and self.rng.random() >= float(r["chance"]):
            return r.get("text_fail", "Nothing came of it.") + self._flee(fx, fy, r)
        text = r.get("text", "Nothing happened.")
        if r.get("noise"):
            self.glyphs = self._glyphs()
            text = text.format(glyphs=self.glyphs)
        if r.get("gives") and len(b.inv) + len(r["gives"]) > self._carry_max(b):
            return self._too_much(b, "the " + ", ".join(dict.fromkeys(r["gives"])), len(r["gives"]))
        if r.get("becomes"):
            self.set_tile(fx, fy, r["becomes"])
        extra = self._give(b, r.get("gives", []), self.spec.get("take_to_hand", False))
        extra += self._extras(b, r, fx, fy)
        if r.get("seen"):
            self._tell(b, f"{self.someone(b)} {r['seen']}.")
        return text + extra

    def _extras(self, b, r, fx, fy) -> str:
        """The wild land's rule fields: a creature changed (`creature_becomes`, null = gone), a
        creature that bolts (`flees`), a lucky extra (`bonus`: p, gives, text)."""
        out = ""
        if "creature_becomes" in r and (fx, fy) in self.creatures:
            if r["creature_becomes"]:
                self.creatures[(fx, fy)] = [r["creature_becomes"], self.clock]
            else:
                del self.creatures[(fx, fy)]
        bonus = r.get("bonus")
        if bonus and self.rng.random() < float(bonus["p"]):
            out += " " + bonus["text"] + self._give(b, bonus.get("gives", []))
        return out + self._flee(fx, fy, r)

    def _flee(self, fx, fy, r) -> str:
        if not r.get("flees") or (fx, fy) not in self.creatures:
            return ""
        self._creature_step(fx, fy, away_from=(fx, fy), tries=3)
        return ""

    def _use(self, b, streak) -> str:
        fx, fy = self.front(b)
        target = self.thing(fx, fy)
        cap = self.info(self.name_at(fx, fy)).get("store")
        if cap and b.held:                            # put what I hold in the cache
            kept = self.stores.setdefault((fx, fy), [])
            if len(kept) >= int(cap):
                return f"The {self.name_at(fx, fy)} is full."
            what = b.held
            self._drop_held(b)
            kept.append(what)
            self._tell(b, f"{self.someone(b)} put {_a(what)} in the {self.name_at(fx, fy)}.")
            self.note("store_put", by=b.key, what=what, at=[fx, fy])
            return f"I put the {what} in the {self.name_at(fx, fy)}. It holds {len(kept)} thing{'s' if len(kept) > 1 else ''}."
        ground = (self.name_at(fx, fy) in GROUND and (fx, fy) not in self.items
                  and not any((o.x, o.y) == (fx, fy) for o in self.bodies.values()))
        i, r = self._rule("use", b, target, ground)
        if r is None:
            if b.held and ground:
                self.items[(fx, fy)] = b.held
                what = b.held
                self._drop_held(b)
                self._tell(b, f"{self.someone(b)} set {_a(what)} down.")
                return f"I set the {what} down on the {self.name_at(fx, fy)}."
            shown = target.replace("item:", "")
            return (f"I used the {b.held} on the {shown}. Nothing happened." if b.held
                    else f"I touched the {shown}. Nothing happened.")
        if r.get("archive"):
            return self._read(b)
        if r.get("chance") is not None and self.rng.random() >= float(r["chance"]):
            if r.get("seen_fail"):
                self._tell(b, f"{self.someone(b)} {r['seen_fail']}.")
            return r.get("text_fail", "Nothing came of it.") + self._flee(fx, fy, r)
        if r.get("together"):
            pushed = self.pushes.setdefault((fx, fy), set())
            pushed.add(b.key)
            if len(pushed) < r["together"]:
                self.note("push", by=b.key, what=r["target"], at=[fx, fy], pushing=sorted(pushed), done=False)
                self._tell(b, f"{self.someone(b)} {r.get('seen', 'pushed')}.")
                return r["text_alone"]
            self.note("push", by=b.key, what=r["target"], at=[fx, fy], pushing=sorted(pushed), done=True)
            self.set_tile(fx, fy, r["becomes"])
            self._tell(b, f"{self.someone(b)} {r.get('seen', 'pushed')}, and it swung open.")
            return r["text"]
        if r.get("in_a_row"):
            n = streak[3] + 1 if streak and streak[:3] == (i, fx, fy) else 1
            if n < r["in_a_row"]:
                b.streak = (i, fx, fy, n)
                return r["text_before"]
        if r.get("noise"):
            self.glyphs = self._glyphs()
            return r["text"].format(glyphs=self.glyphs)
        gives = r.get("gives", [])
        if gives and len(b.inv) - (1 if r.get("consume") else 0) + len(gives) > self._carry_max(b):
            return self._too_much(b, "what that would make", len(gives) - (1 if r.get("consume") else 0))
        if r.get("consume"):
            self._drop_held(b)
        if "target_becomes" in r:
            if r["target_becomes"]:
                self.items[(fx, fy)] = r["target_becomes"]
            else:
                self.items.pop((fx, fy), None)
        if r.get("becomes"):
            self.set_tile(fx, fy, r["becomes"])
        if r.get("feed") and (fx, fy) in self.timers:
            self.timers[(fx, fy)]["since"] = self.clock
        if r.get("water") and (fx, fy) in self.timers:
            self.timers[(fx, fy)]["watered"] = True
        extra = self._give(b, gives)
        extra += self._extras(b, r, fx, fy)
        if r.get("seen"):
            self._tell(b, f"{self.someone(b)} {r['seen']}.")
        if r.get("heard"):                                 # a sound that carries (a reed pipe)
            self._tell(b, None, voice=r["heard"], far=r.get("heard_far"))
        return r.get("text", "Something changed.") + extra

    def _read(self, b) -> str:
        if not b.papers:
            return "The shelf is empty."
        p = b.papers[b.shelf % len(b.papers)]
        if b.page >= len(p["pages"]):
            return f"I have read '{p['title']}' to the end."
        text = p["pages"][b.page]
        b.page += 1
        self._tell(b, f"{self.someone(b)} read a paper from the shelf.")
        return f"I read '{p['title']}', page {b.page} of {len(p['pages'])}: {text}"

    def _switch(self, b) -> str:
        kinds = list(dict.fromkeys(b.inv))
        if not kinds:
            return "I have nothing to hold."
        order = kinds + [None]
        b.held = order[(order.index(b.held) + 1) % len(order)] if b.held in order else kinds[0]
        return f"I now hold the {b.held}." if b.held else "I put everything away; my hands are free."

    def _eat(self, b):
        if not b.held:
            return [event(self.name, "outcome", ok=False, text="I have nothing in my hands to eat.")]
        amount = self.spec["food"].get(b.held)
        if amount is None:
            return [event(self.name, "outcome", ok=False, text=f"I cannot eat the {b.held}.")]
        what = b.held
        self._drop_held(b)
        left = self.spec.get("eat_leaves", {}).get(what)
        if left:
            self._give(b, [left])
        self._tell(b, f"{self.someone(b)} ate {_a(what)}.")
        return [event(self.name, "outcome", ok=True, text=f"I ate the {what}."),
                event(self.name, "ate", amount=amount)]

    def _hand(self, b) -> str:
        """Give what I hold to the one in front (P3). No asking and no accepting: it is in their hands
        (at their feet if those are full or they are asleep). Refusing is only not giving back."""
        o = self.other_in_front(b)
        if o is None:
            return "There is no one in front of me to give anything to."
        if not b.held:
            return f"I hold nothing to give to {self.someone(o).lower()}."
        what = b.held
        feet = o.asleep or len(o.inv) >= self._carry_max(o)
        if feet and (o.x, o.y) in self.items:
            return f"I held out the {what}, but {self.someone(o).lower()} cannot take it and there is no room at their feet."
        self._drop_held(b)
        if feet:
            self.items[(o.x, o.y)] = what                 # set down beside them
            where = "at their feet"
            o.inbox.append((f"{self.someone(b)} set {_a(what)} down at my feet.", True, 0.2))
        else:
            self._give(o, [what])
            where = "into their hands"
            o.inbox.append((f"{self.someone(b)} gave me {_a(what)}. I now carry it.", True, 0.3))
        self._tell_seen(b, o, f"{self.someone(b)} gave {_a(what)} to {self.someone(o).lower()}.")
        self.note("give", by=b.key, to=o.key, what=what, asleep=o.asleep)
        return f"I gave the {what} to {self.someone(o).lower()}, {where}."

    def _build(self, b) -> str:
        """Build in front from what is carried (spec `build`: recipes {needs: {item: n}, becomes: a tile,
        on: the ground it can stand on}); a recipe that uses what is held goes first, then the spec's
        order. The mind is not told the recipes: an attempt that matches none says so, no more."""
        fx, fy = self.front(b)
        ground = self.name_at(fx, fy)
        if (fx, fy) in self.items or (fx, fy) in self.creatures or self.body_at(fx, fy, but=b):
            return "There is something in the way in front of me; I cannot build there."
        if not any(ground in r.get("on", sorted(GROUND)) for r in self.spec.get("build", [])):
            return f"There is no open ground in front of me to build on, only {self.describe(fx, fy)}."
        have = {}
        for i in b.inv:
            have[i] = have.get(i, 0) + 1
        fits = [r for r in self.spec.get("build", [])
                if ground in r.get("on", sorted(GROUND)) and all(have.get(i, 0) >= n for i, n in r["needs"].items())]
        if not fits:
            return ("I tried to build something from what I carry, but it came to nothing." if b.inv
                    else "I carry nothing to build with.")
        r = sorted(fits, key=lambda r: b.held not in r["needs"])[0]
        for i, n in r["needs"].items():
            for _ in range(n):
                b.inv.remove(i)
        if b.held not in b.inv:
            b.held = b.inv[0] if b.inv else None
        self.set_tile(fx, fy, r["becomes"])
        if r.get("seen"):
            self._tell(b, f"{self.someone(b)} {r['seen']}.")
        self.note("build", by=b.key, what=r["becomes"], at=[fx, fy])
        return r["text"]

    def _hit(self, b) -> float:
        """How hard this body strikes: the spec's base, what it holds, less while weak."""
        st = self.spec["strike"]
        return (float(st["base"]) + float(st.get("weapons", {}).get(b.held, 0.0))) * (1.0 - 0.5 * b.weak)

    def _strike(self, b) -> str:
        """Strike what is in front (P3, owner): a mind feels pain and grows weak, never dies; a creature may
        dodge, bolts if it lives, and dies when struck past its toughness, leaving what the spec says; a
        thing hurts the hand a little."""
        st = self.spec["strike"]
        hit = round(self._hit(b), 3)
        with_ = f" with the {b.held}" if b.held else ""
        o = self.other_in_front(b)
        if o is not None:
            o.weak = min(1.0, o.weak + hit * float(st.get("weakens", 0.8)))
            o.pending.append(event(self.name, "hurt", amount=hit, rouse=1.0))
            o.inbox.append((f"{self.someone(b)} struck me{with_}! It hurts.", True, 0.0))
            self._tell_seen(b, o, f"{self.someone(b)} struck {self.someone(o).lower()}{with_}.")
            self.note("strike", by=b.key, to=o.key, hit=hit, held=b.held, asleep=o.asleep)
            return f"I struck {self.someone(o).lower()}{with_}."
        fx, fy = self.front(b)
        c = self.creatures.get((fx, fy))
        if c is not None:
            kind = c[0]
            sp = self.cspec.get(kind, {})
            if self.rng.random() < float(sp.get("dodge", 0.0)):
                self._creature_step(fx, fy, away_from=(b.x, b.y), tries=2)
                self._tell(b, f"{self.someone(b)} struck at {_a(kind)} and missed.")
                return f"I struck at the {kind}{with_}, but it dodged away."
            while len(c) < 3:
                c.append(0.0)
            c[2] += hit
            if c[2] >= float(sp.get("tough", 1.0)):
                del self.creatures[(fx, fy)]
                left = sp.get("leaves", [])
                if left:
                    self.items[(fx, fy)] = left[0]
                    for extra in left[1:]:                 # the rest lands beside it
                        spot = next(((fx + dx, fy + dy) for dx, dy in DIRS.values()
                                     if (fx + dx, fy + dy) not in self.items and not self.solid(fx + dx, fy + dy)), None)
                        if spot:
                            self.items[spot] = extra
                self._tell(b, f"{self.someone(b)} killed {_a(kind)}.")
                self.note("kill", by=b.key, what=kind)
                return (f"I struck the {kind}{with_}; it fell and lay still. It is dead."
                        + (f" There is {_a(left[0])} where it fell." if left else ""))
            if sp.get("charges"):                          # it turns on the one who struck it
                while len(c) < 4:
                    c.append(0)
                c[3] = self.clock + int(sp["charges"].get("angry_for", 20))
                self._tell(b, f"{self.someone(b)} struck {_a(kind)}.")
                return f"I struck the {kind}{with_}; it squealed and turned on me."
            self._creature_step(fx, fy, away_from=(b.x, b.y), tries=3)
            self._tell(b, f"{self.someone(b)} struck {_a(kind)}.")
            return f"I struck the {kind}{with_}; it cried out and fled, hurt."
        what = self.short(fx, fy)
        if self.solid(fx, fy):
            b.pending.append(event(self.name, "hurt", amount=float(st.get("self_hurt", 0.03))))
            return f"I struck {what}{with_}. It did not give; my hand stings."
        return f"I struck at the empty air{with_}."

    def _tell_seen(self, b, o, text: str) -> None:
        """Those who see an act between two bodies (not the two)."""
        for w in self.bodies.values():
            if w not in (b, o) and not w.asleep and abs(w.x - b.x) + abs(w.y - b.y) <= self.spec["hearing"]["words_with_mark"]:
                w.inbox.append((text, False, 0.0))

    def _drink(self, b):
        """Drink (spec `drink`: rules on what is held, then on what is in front): `drank` with the
        rule's amount; salt water has a negative one (thirst worse). What is held goes first."""
        fx, fy = self.front(b)
        front = self.thing(fx, fy)
        rules = self.spec.get("drink", [])
        r = (next((r for r in rules if "held" in r and r["held"] == b.held), None)
             or next((r for r in rules if r.get("target") == front), None))
        if r is None:
            return [event(self.name, "outcome", ok=False,
                          text=f"There is nothing to drink in the {front.replace('item:', '')} in front of me"
                               + (f" or the {b.held} I hold." if b.held else "."))]
        if "held" in r:
            self._drop_held(b)
            if r.get("leaves"):
                self._give(b, [r["leaves"]])
        if r.get("seen"):
            self._tell(b, f"{self.someone(b)} {r['seen']}.")
        return [event(self.name, "outcome", ok=True, text=r["text"]),
                event(self.name, "drank", amount=float(r["amount"]))]

    def _speak(self, b, words: str | None) -> str:
        who = self.someone(b)
        words = " ".join(str(words or "").split()).strip(" \"'")[:240]
        if not words:
            got = self._tell(b, f"{who} called out.", voice="I heard a call from the {dir}.",
                             far="I heard a call from far away.", attention=True, contact=0.2)
            self.note("speak", by=b.key, words=None, heard=got)
            return "I called out."
        got = self._tell(b, f'{who} said: "{words}"', voice='From the {dir}, a voice said: "' + words.replace("{", "(").replace("}", ")") + '"',
                         far="I heard a voice far away, too far to make out words.", attention=True, contact=0.3)
        self.note("speak", by=b.key, words=words, heard=got)
        return f'I said aloud: "{words}"'

    # -- the world's row (P3, 3i): one per round, read by the inspector's map --------------------------
    def world_meta(self) -> dict:
        """The full map at a run's start or resume (the rows after it carry only what changed)."""
        self._row_prev = (["".join(r) for r in self.grid], dict(self.items), set(self.mist))
        return {"world_meta": {"name": self.spec.get("name"), "size": [self.W, self.H], "clock": self.clock,
                               "day_ticks": self.spec["day_ticks"], "rows": self._row_prev[0],
                               "legend": {c: t["name"] for c, t in self.tiles.items()},
                               "items": {f"{x},{y}": v for (x, y), v in self.items.items()},
                               "mist": sorted(f"{x},{y}" for x, y in self.mist),
                               "caves": getattr(self, "caves", []), "springs": getattr(self, "springs", []),
                               "orchard": self.orchard}}

    def world_row(self) -> dict:
        """This round: every body (place, facing, holdings, asleep, weakness), time, weather, season,
        rare events under way, what passed between bodies (`last_round`), creatures, and the tiles,
        things lying about and mist that changed since the previous row."""
        rows0, items0, mist0 = getattr(self, "_row_prev", None) or (["".join(r) for r in self.grid], {}, set())
        rows = ["".join(r) for r in self.grid]
        tiles = [[x, y, self.tiles[rows[y][x]]["name"]] for y in range(self.H) if rows[y] != rows0[y]
                 for x in range(self.W) if rows[y][x] != rows0[y][x]]
        items = {f"{x},{y}": v for (x, y), v in self.items.items() if items0.get((x, y)) != v}
        gone = [f"{x},{y}" for (x, y) in items0 if (x, y) not in self.items]
        lifted = sorted(f"{x},{y}" for x, y in mist0 - self.mist)
        self._row_prev = (rows, dict(self.items), set(self.mist))
        se = self.season()
        return {"world_t": self.clock, "phase": self.phase(), **({"season": se["name"]} if se else {}),
                "rain": bool(self.weather.get("rain")), **({"events": sorted(self.events)} if self.events else {}),
                "bodies": {k: {"x": b.x, "y": b.y, "facing": b.facing, "held": b.held, "inv": list(b.inv),
                               "asleep": b.asleep, **({"weak": round(b.weak, 3)} if b.weak else {})}
                           for k, b in self.bodies.items()},
                **({"happened": self.last_round} if self.last_round else {}),
                "creatures": [[x, y, c[0]] for (x, y), c in sorted(self.creatures.items())],
                **({"tiles": tiles} if tiles else {}), **({"items": items} if items else {}),
                **({"items_gone": gone} if gone else {}), **({"mist_lifted": lifted} if lifted else {})}

    # -- persistence -------------------------------------------------------------
    def state(self) -> dict:
        return {"clock": self.clock, "glyphs": self.glyphs, "size": [self.W, self.H],
                "rows": ["".join(r) for r in self.grid],
                "legend": {c: t["name"] for c, t in self.tiles.items()},
                "items": {f"{x},{y}": n for (x, y), n in self.items.items()},
                "timers": {f"{x},{y}": t for (x, y), t in self.timers.items()},
                "shelter": self.shelter, "orchard": self.orchard, "phase": self.phase(),
                "bodies": {k: b.state() for k, b in self.bodies.items()},
                **({"creatures": {f"{x},{y}": c for (x, y), c in self.creatures.items()}} if self.cspec else {}),
                **({"revealed": list(self.revealed)} if self.revealed else {}),
                **({"mist": sorted(f"{x},{y}" for x, y in self.mist)} if self.spec.get("fog", {}).get("banks") else {}),
                **({"stores": {f"{x},{y}": v for (x, y), v in self.stores.items()}} if self.stores else {}),
                **({"events": self.events} if self.events else {}),
                **({"weather": dict(self.weather)} if self.spec.get("weather") else {})}

    def restore(self, st: dict) -> None:
        self.clock = st.get("clock", 0)
        self.glyphs = st.get("glyphs", self.glyphs)
        if st.get("rows"):
            self.grid = [list(r) for r in st["rows"]]
        self.items = {tuple(map(int, k.split(","))): v for k, v in (st.get("items") or {}).items()}
        self.timers = {tuple(map(int, k.split(","))): v for k, v in (st.get("timers") or {}).items()}
        self.shelter = tuple(st.get("shelter", self.shelter))
        self.orchard = tuple(st.get("orchard", self.orchard))
        if "creatures" in st:
            self.creatures = {tuple(map(int, k.split(","))): list(v) for k, v in st["creatures"].items()}
        if st.get("revealed"):
            self.revealed = tuple(st["revealed"])
        if "mist" in st:
            self.mist = {tuple(map(int, k.split(","))) for k in st["mist"]}
        self.stores = {tuple(map(int, k.split(","))): list(v) for k, v in (st.get("stores") or {}).items()}
        self.events = dict(st.get("events") or {})
        if st.get("weather"):
            self.weather = dict(st["weather"])
        for key, s in (st.get("bodies") or {}).items():
            b = self.bodies.setdefault(key, LandBody(key, s.get("mark"), s["x"], s["y"]))
            b.mark, b.x, b.y, b.facing = s.get("mark"), s["x"], s["y"], s.get("facing", "south")
            b.inv, b.held = list(s.get("inv", [])), s.get("held")
            b.shelf, b.page = s.get("shelf", 0), s.get("page", 0)
            b.weak = float(s.get("weak", 0.0))


class LandView:
    """What one Brain ticks against: its body's senses and acts."""

    def __init__(self, world: Land, body: LandBody):
        self.world, self.body = world, body
        self.name = world.name
        self.verbs = {**world.verbs, **world.SOCIAL_VERBS}

    @property
    def place(self) -> str:
        return self.world.zone(self.body.x, self.body.y)

    @property
    def pos(self) -> list[int]:
        return [self.body.x, self.body.y]

    def sense(self):
        return self.world._sense(self.body)

    def ambient(self):
        return self.world.ambient(self.body)

    def act(self, verb: str, words: str | None = None):
        return self.world._act(self.body, verb, words)
