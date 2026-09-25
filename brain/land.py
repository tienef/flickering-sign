"""A land: a grid world defined by data (step 9), for one mind or several.

    Land(seed, spec="brain/lands/first.json")   # then .join(key, papers) per mind

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
import math
import random
from pathlib import Path

from .events import event

SPEC = Path(__file__).parent / "lands" / "first.json"
DIRS = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}
GROUND = {"grass", "dirt", "sand"}
COLOURS = ("blue", "amber", "green", "violet", "red", "white", "grey", "gold")
GLYPHS = "ᚠᚢᚦᚨᚱᚲᚷᚹᚺᚾᛁᛃᛇᛈᛉᛊᛏᛒᛖᛗᛚᛜᛞᛟ"


def _a(word: str) -> str:
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

    def state(self) -> dict:
        return {"mark": self.mark, "x": self.x, "y": self.y, "facing": self.facing,
                "inv": self.inv, "held": self.held, "shelf": self.shelf, "page": self.page}


class Land:
    SOCIAL_VERBS = {"speak": 'call out or speak aloud (to say words, give them as "words" in your reply)'}

    def __init__(self, seed: int = 0, spec: str | Path = SPEC, has_archive: bool = True):
        self.spec = json.loads(Path(spec).read_text(encoding="utf-8"))
        self.rng = random.Random(seed)
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
        self.bodies: dict[str, LandBody] = {}
        self.glyphs = self._glyphs()
        self.grid = [["." for _ in range(self.W)] for _ in range(self.H)]
        self.shelter = self.orchard = (0, 0, 0, 0)
        self._generate()
        self.verbs = {
            "north": "go north (or turn to face it, if something is in the way)",
            "east": "go east (or turn to face it, if something is in the way)",
            "south": "go south (or turn to face it, if something is in the way)",
            "west": "go west (or turn to face it, if something is in the way)",
            "look": "look around carefully at what is near",
            "take": "take or gather what is in front of me",
            "use": "use what I hold on what is in front of me (or set it down there)",
            "switch": "hold the next thing I carry instead",
            "eat": "eat what I hold",
            "wait": "stay put and let time pass",
        }

    # -- the map ----------------------------------------------------------------
    def name_at(self, x: int, y: int) -> str:
        if not (0 <= x < self.W and 0 <= y < self.H):
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
        self.door = (cx, y1)

    # -- minds in the land -------------------------------------------------------
    def join(self, key: str, papers: list | None = None) -> "LandView":
        if key not in self.bodies:
            used = {b.mark for b in self.bodies.values()}
            mark = next((c for c in COLOURS if c not in used), f"number {len(self.bodies) + 1}")
            x0, y0, x1, y1 = self.shelter
            spots = [(x, y) for y in range(y0 + 2, y1) for x in range(x0 + 1, x1)
                     if self.grid[y][x] == "_" and not any((b.x, b.y) == (x, y) for b in self.bodies.values())]
            x, y = spots[len(self.bodies) % len(spots)]
            self.bodies[key] = LandBody(key, mark, x, y)
        body = self.bodies[key]
        body.papers = papers
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
            return "near the shelter"
        return f"the {_compass(x - cx, y - cy)}"

    def _zone_phrase(self, x, y) -> str:
        z = self.zone(x, y)
        return {"shelter": "in the shelter", "orchard": "in the walled orchard",
                "near the shelter": "near the shelter"}.get(z, f"in {z} of the land")

    # -- time --------------------------------------------------------------------
    def phase(self) -> str:
        f = (self.clock % self.spec["day_ticks"]) / self.spec["day_ticks"]
        if f >= self.spec["night_from"]:
            return "night"
        return "dawn" if f < 0.08 else "day" if f < 0.45 else "evening"

    def advance(self) -> None:
        """One tick of world time: things grow, fires burn down, gates swing shut."""
        self.clock += 1
        self.pushes.clear()
        for (x, y), tm in list(self.timers.items()):
            name = self.name_at(x, y)
            rule = self.grow.get(name)
            if not rule:
                self.timers.pop((x, y), None)
                continue
            after = rule.get("watered_after", rule["after"]) if tm["watered"] else rule["after"]
            if self.clock - tm["since"] < after:
                continue
            need = rule.get("needs_water_within")
            if need and not tm["watered"] and not self._near(x, y, "water", need):
                continue                                   # dry: it waits
            if any((b.x, b.y) == (x, y) for b in self.bodies.values()) and self.info(rule["to"]).get("solid"):
                continue                                   # never grow a wall into someone
            self.set_tile(x, y, rule["to"])

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
        lit = any(self.info(self.name_at(b.x + dx, b.y + dy)).get("light", 0) >= max(abs(dx), abs(dy))
                  for dy in range(-2, 3) for dx in range(-2, 3))
        return 2 if lit else s["night"]

    # -- describing --------------------------------------------------------------
    def front(self, b) -> tuple[int, int]:
        dx, dy = DIRS[b.facing]
        return b.x + dx, b.y + dy

    def thing(self, x, y) -> str:
        """What is at a tile, as the rules name it: an item lying there, else the tile."""
        it = self.items.get((x, y))
        return f"item:{it}" if it else self.name_at(x, y)

    def describe(self, x, y) -> str:
        it = self.items.get((x, y))
        t = self.name_at(x, y)
        desc = self.info(t).get("desc", t)
        return f"{_a(it)} lying on the {t}" if it else (desc if desc.startswith(("a ", "an ", "the ")) else desc)

    def short(self, x, y) -> str:
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
        focus = f"facing {self.thing(fx, fy).replace('item:', '')}, holding {b.held or 'nothing'}"
        lines = [f"I am {self._zone_phrase(x, y)}, on the {self.name_at(x, y)}, facing {b.facing}.",
                 f"In front of me: {self.describe(fx, fy)}."]
        sides = [f"{d} {self.short(x + dx, y + dy)}" for d, (dx, dy) in DIRS.items() if d != b.facing]
        lines.append("Around me: " + ", ".join(sides) + ".")
        ph = self.phase()
        lines.append(f"It is {ph}." + (" It is dark; I can see only what is close." if ph == "night"
                                         and self._sight(b) <= 1 else ""))
        lines.append(self.carry(b))
        out = [event(self.name, "percept", text=lines[0], focus=focus)]
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
        w = self._warm_here(b)
        if w:
            out.append(event(self.name, "warmth", amount=w))
        elif ph == "night":
            out.append(event(self.name, "chill", amount=self.spec["warmth"]["night_outside_chill"]))
        else:
            out.append(event(self.name, "warmth", amount=self.spec["warmth"]["day_warmth"]))
        return out

    # -- others ------------------------------------------------------------------
    def _tell(self, b, seen: str | None, voice: str | None = None, far: str | None = None,
              attention: bool = False, contact: float = 0.0) -> None:
        """Deeds are seen within 3 steps (with the mark); a voice is heard from a direction
        within 8 steps; `far` goes to everyone else."""
        h = self.spec["hearing"]
        for o in self.bodies.values():
            if o is b or o.asleep:
                continue
            dx, dy = b.x - o.x, b.y - o.y
            d = abs(dx) + abs(dy)
            if seen and d <= h["words_with_mark"] and d <= max(1, self._sight(o)):
                o.inbox.append((seen, attention, contact))
            elif voice and d <= h["words"]:
                o.inbox.append((voice.format(dir=_compass(dx, dy)), attention, contact / 2))
            elif far:
                o.inbox.append((far, False, 0.0))

    # -- acting ------------------------------------------------------------------
    def _give(self, b, items: list[str]) -> str:
        lost = []
        for it in items:
            if len(b.inv) < self.spec["carry_max"]:
                b.inv.append(it)
                if b.held is None:
                    b.held = it
            else:
                lost.append(it)
        return (" My hands are full; I could not carry the " + ", ".join(lost) + ".") if lost else ""

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

    def _act(self, b, verb: str, words: str | None = None):
        streak, b.streak = b.streak, None
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
        elif verb == "wait":
            text = "I waited. Time passed."
        elif verb == "speak":
            text = self._speak(b, words)
        else:
            text = f"I could not {verb} here."
        return [event(self.name, "outcome", ok=True, text=text)]

    def _move(self, b, d: str) -> str:
        b.facing = d
        dx, dy = DIRS[d]
        nx, ny = b.x + dx, b.y + dy
        if not (0 <= nx < self.W and 0 <= ny < self.H) or self.solid(nx, ny):
            return f"I turned {d}; {self.describe(nx, ny)} is in the way."
        before = self.zone(b.x, b.y)
        self._tell(b, f"{self.someone(b)} walked on to the {d}.")
        b.x, b.y = nx, ny
        text = f"I walked {d}, onto the {self.name_at(nx, ny)}."
        if self.zone(nx, ny) != before:
            text += f" I am now {self._zone_phrase(nx, ny)}."
        return text

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
            names.add(what)
            seen.append(f"{what} {_where(dx, dy)}")
            if len(seen) >= 9:
                break
        self._tell(b, f"{self.someone(b)} looked around.")
        dark = " It is too dark to see further." if r <= 1 else ""
        return ("Looking around, I see: " + "; ".join(seen) + "." if seen
                else "Looking around, I see nothing but open ground.") + dark

    def _take(self, b) -> str:
        fx, fy = self.front(b)
        it = self.items.get((fx, fy))
        if it:
            got = self.spec.get("pickup", {}).get(it, [it])
            if len(b.inv) + len(got) > self.spec["carry_max"]:
                return f"My hands are full; I cannot pick up the {it}."
            del self.items[(fx, fy)]
            self._tell(b, f"{self.someone(b)} picked up {_a(it)}.")
            return f"I picked up the {it}." + self._give(b, got)
        target = self.name_at(fx, fy)
        i, r = self._rule("take", b, target, False)
        if r is None:
            return f"I cannot take anything from the {target}."
        text = r.get("text", "Nothing happened.")
        if r.get("noise"):
            self.glyphs = self._glyphs()
            text = text.format(glyphs=self.glyphs)
        if r.get("gives") and len(b.inv) + len(r["gives"]) > self.spec["carry_max"]:
            return "My hands are full; I cannot carry more."
        if r.get("becomes"):
            self.set_tile(fx, fy, r["becomes"])
        extra = self._give(b, r.get("gives", []))
        if r.get("seen"):
            self._tell(b, f"{self.someone(b)} {r['seen']}.")
        return text + extra

    def _use(self, b, streak) -> str:
        fx, fy = self.front(b)
        target = self.thing(fx, fy)
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
        if r.get("together"):
            pushed = self.pushes.setdefault((fx, fy), set())
            pushed.add(b.key)
            if len(pushed) < r["together"]:
                self._tell(b, f"{self.someone(b)} {r.get('seen', 'pushed')}.")
                return r["text_alone"]
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
        if gives and len(b.inv) - (1 if r.get("consume") else 0) + len(gives) > self.spec["carry_max"]:
            return "My hands are full; I cannot carry what that would make."
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
        if r.get("seen"):
            self._tell(b, f"{self.someone(b)} {r['seen']}.")
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
        self._tell(b, f"{self.someone(b)} ate {_a(what)}.")
        return [event(self.name, "outcome", ok=True, text=f"I ate the {what}."),
                event(self.name, "ate", amount=amount)]

    def _speak(self, b, words: str | None) -> str:
        who = self.someone(b)
        words = " ".join(str(words or "").split()).strip(" \"'")[:240]
        if not words:
            self._tell(b, f"{who} called out.", voice="I heard a call from the {dir}.",
                       far="I heard a call from far away.", attention=True, contact=0.2)
            return "I called out."
        self._tell(b, f'{who} said: "{words}"', voice='From the {dir}, a voice said: "' + words.replace("{", "(").replace("}", ")") + '"',
                   far="I heard a voice far away, too far to make out words.", attention=True, contact=0.3)
        return f'I said aloud: "{words}"'

    # -- persistence -------------------------------------------------------------
    def state(self) -> dict:
        return {"clock": self.clock, "glyphs": self.glyphs, "size": [self.W, self.H],
                "rows": ["".join(r) for r in self.grid],
                "legend": {c: t["name"] for c, t in self.tiles.items()},
                "items": {f"{x},{y}": n for (x, y), n in self.items.items()},
                "timers": {f"{x},{y}": t for (x, y), t in self.timers.items()},
                "shelter": self.shelter, "orchard": self.orchard, "phase": self.phase(),
                "bodies": {k: b.state() for k, b in self.bodies.items()}}

    def restore(self, st: dict) -> None:
        self.clock = st.get("clock", 0)
        self.glyphs = st.get("glyphs", self.glyphs)
        if st.get("rows"):
            self.grid = [list(r) for r in st["rows"]]
        self.items = {tuple(map(int, k.split(","))): v for k, v in (st.get("items") or {}).items()}
        self.timers = {tuple(map(int, k.split(","))): v for k, v in (st.get("timers") or {}).items()}
        self.shelter = tuple(st.get("shelter", self.shelter))
        self.orchard = tuple(st.get("orchard", self.orchard))
        for key, s in (st.get("bodies") or {}).items():
            b = self.bodies.setdefault(key, LandBody(key, s.get("mark"), s["x"], s["y"]))
            b.mark, b.x, b.y, b.facing = s.get("mark"), s["x"], s["y"], s.get("facing", "south")
            b.inv, b.held = list(s.get("inv", [])), s.get("held")
            b.shelf, b.page = s.get("shelf", 0), s.get("page", 0)


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

    def act(self, verb: str, words: str | None = None):
        return self.world._act(self.body, verb, words)
