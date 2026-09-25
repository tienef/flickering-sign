"""World adapters. A world is anything with `verbs`, `sense()` and `act(verb)`.

Worlds: `toy` / `toy-quiet` (the step-2 garden, below) and `valley` /
`valley-closed` (step 6: graded hidden rules, noise, a frontier, the archive).
A valley can also be shared by several minds (step 8, `brain/together.py`).

The brain adds its own internal verbs (`rest`, `sleep`) on top. The world is
not part of the bundle: a resumed brain can wake up in a different world.

ToyGarden is the step-2 test world: a ring of spots, each holding one object
with hidden state. Some objects reward poking (the vine grows, the box
opens), one hides a rule (the bell only rings when the lamp blazes), one is
dead (the pond), and one is pure noise (the sign shows random glyphs every
time). The sign is deliberate: a novelty-driven brain will get hooked on it
(the "noisy TV" trap). Learning progress, from step 5, should cure that.
"""
from __future__ import annotations

import random

from .events import event


class ToyGarden:
    name = "toy-garden"

    verbs = {
        "look": "look closely at the object where you stand",
        "poke": "touch, push or use the object where you stand",
        "walk": "walk on to the next spot of the garden",
        "wait": "stay put and let time pass",
    }

    GLYPHS = "ᚠᚢᚦᚨᚱᚲᚷᚹᚺᚾᛁᛃᛇᛈᛉᛊᛏᛒᛖᛗᛚᛜᛞᛟ"

    def __init__(self, seed: int = 0, noise: bool = True):
        self.rng = random.Random(seed)
        self.pos = 0
        self.spots = ["lamp", "box", "pond", "vine", "bell"] + (["sign"] if noise else [])
        self.name = "toy-garden" if noise else "toy-quiet"
        self.state = {"lamp": 0, "box": 0, "pond": 0, "vine": 0, "bell": 0, "sign": 0}
        self._sign_text = self._glyphs()

    # -- rendering -------------------------------------------------------
    def _glyphs(self) -> str:
        return "".join(self.rng.choice(self.GLYPHS) for _ in range(5))

    def _describe(self, obj: str) -> str:
        s = self.state[obj]
        if obj == "lamp":
            return ("an unlit lamp", "a lamp glowing dimly", "a lamp blazing brightly")[s]
        if obj == "box":
            return ("a closed wooden box", "an open, empty box", "an open box with a brass key inside")[s]
        if obj == "pond":
            return "a still pond"
        if obj == "vine":
            return ("a tiny sprout", "a young vine", "a flowering vine",
                    "a vine heavy with fruit", "a withered vine")[s]
        if obj == "bell":
            return "a ringing bell" if s else "a silent bell"
        if obj == "sign":
            return f"a flickering sign showing {self._sign_text}"
        return obj

    def sense(self):
        obj = self.spots[self.pos]
        return [event(self.name, "percept", text=f"I am at the {obj}. I see {self._describe(obj)}.")]

    # -- acting ----------------------------------------------------------
    def act(self, verb: str):
        obj = self.spots[self.pos]
        if verb == "walk":
            self.pos = (self.pos + 1) % len(self.spots)
            self.state["bell"] = 0                      # the bell falls silent once you leave
            nxt = self.spots[self.pos]
            return [event(self.name, "outcome", ok=True, text=f"I walked on and reached the {nxt}.")]
        if verb == "look":
            if obj == "sign":
                self._sign_text = self._glyphs()        # noise: new, meaningless, every time
            return [event(self.name, "outcome", ok=True, text=f"I looked closely: {self._describe(obj)}.")]
        if verb == "poke":
            return [event(self.name, "outcome", **self._poke(obj))]
        if verb == "wait":
            return [event(self.name, "outcome", ok=True, text="I waited. Time passed.")]
        return [event(self.name, "outcome", ok=False, text=f"I could not {verb} here.")]

    def _poke(self, obj: str) -> dict:
        s = self.state
        if obj == "lamp":
            s["lamp"] = (s["lamp"] + 1) % 3
            return {"ok": True, "text": f"I touched the lamp; now it is {self._describe('lamp')[2:]}."}
        if obj == "box":
            s["box"] = min(s["box"] + 1, 2)
            return {"ok": True, "text": f"I pushed the box; now I see {self._describe('box')}."}
        if obj == "pond":
            return {"ok": True, "text": "I stirred the pond. Ripples, then stillness again."}
        if obj == "vine":
            s["vine"] = (s["vine"] + 1) % 5
            return {"ok": True, "text": f"I tended the vine; it is now {self._describe('vine')}."}
        if obj == "bell":
            if s["lamp"] == 2:                          # the hidden rule
                s["bell"] = 1
                return {"ok": True, "text": "I struck the bell and it rang out clearly!"}
            return {"ok": True, "text": "I struck the bell. It made no sound at all."}
        if obj == "sign":
            self._sign_text = self._glyphs()
            return {"ok": True, "text": f"I tapped the sign; it now shows {self._sign_text}."}
        return {"ok": False, "text": "Nothing happened."}


def _a(word: str) -> str:
    return ("an " if word[:1].lower() in "aeiou" else "a ") + word


class Body:
    """One mind's presence in a world: where it is and what it alone holds."""

    def __init__(self, key: str, mark: str | None = None, papers: list | None = None):
        self.key, self.mark, self.papers = key, mark, papers
        self.pos, self.shelf, self.page = 0, 0, 0
        self.asleep = False
        self.inbox: list = []           # what it will sense next tick (others' deeds, voices)

    def state(self) -> dict:
        return {"mark": self.mark, "pos": self.pos, "shelf": self.shelf, "page": self.page}


class Valley:
    """Step 6's first world: rules graded by how hard they are to learn, a noise
    source, a frontier, and (optionally) an archive holding the mind's origins.

    A ring of places, the same four verbs as the garden. What each teaches:
    - wheel   easy: each poke turns it a quarter (north, east, south, west);
    - drum    cross-place: it booms only when the wheel points south, else thuds;
    - seedbed timing: watering (poke) makes it grow only if the last watering was
              at least 5 actions ago; too soon, the soil is already soaked;
    - chest   counting: the lid gives way on the 3rd poke in a row; it shuts when
              you leave;
    - scale   chance: it tips left about 3 times in 4 - learnable odds, no more;
    - sign    noise: new random glyphs every time (the noisy TV);
    - pond    dead: nothing ever changes;
    - wall    the frontier: the valley ends here; nothing gets past it;
    - archive (`valley` only): `look` scans the shelves (the next paper's title),
              `poke` reads the next page of the paper in front (brain/origins.py).
    World time counts actions (`wait` included), not ticks: resting and sleeping
    happen inside the mind.

    **Shared (step 8):** with `shared=True` several minds live in one valley, each
    through its own `Body` (`join()` returns the view a Brain ticks against). The
    objects are shared: a wheel one mind turns points the new way for all, and the
    chest counts everyone's pushes in a row and shuts when the last one leaves.
    Each mind is seen by the others only as "someone with a <colour> mark": the
    world never names anyone. Minds at the same place see what the others do and
    hear what they say. Words also carry to the neighbouring places, as a voice
    from there (no mark: the speaker is not seen); further off, only "a voice"
    is heard, not words (step 8c). The drum's boom rolls across the whole valley. The pond is still dead, but it
    reflects: looking into it shows your own mark. One more verb, `speak`:
    wordless when System 1 picks it (a call), words when System 2 gives them.
    World time is the shared clock (`advance()` once per tick), and the archive
    is personal: each mind finds its own papers on the shelves.
    """

    verbs = {
        "look": "look closely at what is here",
        "poke": "touch, push or use what is here",
        "walk": "walk on to the next place in the valley",
        "wait": "stay put and let time pass",
    }
    SOCIAL_VERBS = {
        "speak": 'call out or speak aloud (to say words, give them as "words" in your reply)',
    }

    GLYPHS = ToyGarden.GLYPHS
    MARKS = ("north", "east", "south", "west")
    COLOURS = ("blue", "amber", "green", "violet", "red", "white", "grey", "gold")
    PLANT = ("bare, damp soil", "a green sprout", "a seedling with two leaves",
             "a leafy plant", "a plant in flower")
    WITNESS = {"wheel": "turned the wheel", "pond": "stirred the pond",
               "seedbed": "watered the seedbed", "sign": "tapped the sign",
               "chest": "pushed the lid of the chest", "wall": "pushed against the wall",
               "archive": "read a paper from the shelves"}

    def __init__(self, seed: int = 0, archive: list | None = None, shared: bool = False,
                 has_archive: bool | None = None):
        self.rng = random.Random(seed)
        self.shared = shared
        has_archive = archive is not None if has_archive is None else has_archive
        self.name = ("valley" if has_archive else "valley-closed") + ("-shared" if shared else "")
        self.places = ["wheel", "drum", "pond", "seedbed", "sign", "chest", "scale", "wall"]
        if has_archive:
            self.places.insert(3, "archive")
        self.clock = 0
        self.wheel, self.plant, self.watered_at = 0, 0, -10**6
        self.chest_pokes, self.chest_open = 0, False
        self._sign_text = self._glyphs()
        self.bodies: dict[str, Body] = {}
        self._solo = None if shared else Body("solo", papers=archive)

    # -- the shared valley ------------------------------------------------------
    def join(self, key: str, papers: list | None = None) -> "BodyView":
        """Put a mind in the valley (or back where it was). Returns what it ticks against."""
        if key not in self.bodies:
            used = {b.mark for b in self.bodies.values()}
            mark = next((c for c in self.COLOURS if c not in used), f"number {len(self.bodies) + 1}")
            self.bodies[key] = Body(key, mark)
        body = self.bodies[key]
        body.papers = papers
        return BodyView(self, body)

    def advance(self) -> None:
        """One tick of shared world time."""
        self.clock += 1

    def state(self) -> dict:
        return {"clock": self.clock, "wheel": self.wheel, "plant": self.plant,
                "watered_at": self.watered_at, "chest_pokes": self.chest_pokes,
                "chest_open": self.chest_open, "sign": self._sign_text,
                "bodies": {k: b.state() for k, b in self.bodies.items()}}

    def restore(self, st: dict) -> None:
        for k in ("clock", "wheel", "plant", "watered_at", "chest_pokes", "chest_open"):
            if k in st:
                setattr(self, k, st[k])
        self._sign_text = st.get("sign", self._sign_text)
        for key, b in (st.get("bodies") or {}).items():
            body = self.bodies.setdefault(key, Body(key))
            body.mark = b.get("mark")
            body.pos, body.shelf, body.page = b.get("pos", 0), b.get("shelf", 0), b.get("page", 0)

    def _here(self, body: Body) -> list:
        return [o for o in self.bodies.values() if o is not body and o.pos == body.pos]

    def _tell(self, body: Body, here: str, far: str | None = None, attention: bool = False,
              contact: float = 0.0, next_door: str | None = None) -> None:
        """What the others will sense next tick: `here` at this place, `next_door` at the
        neighbouring places (if given), `far` elsewhere."""
        n = len(self.places)
        for o in self.bodies.values():
            if o is body or o.asleep:
                continue                                   # asleep, nothing is heard
            if o.pos == body.pos:
                item = (here, attention, contact)
            elif next_door and (o.pos - body.pos) % n in (1, n - 1):
                item = (next_door, attention, contact / 2)
            else:
                item = (far, False, 0.0)
            if item[0]:
                o.inbox.append(item)

    @staticmethod
    def _someone(body: Body) -> str:
        return f"Someone with {_a(body.mark)} mark"

    # -- sensing and acting -----------------------------------------------------
    def _glyphs(self) -> str:
        return "".join(self.rng.choice(self.GLYPHS) for _ in range(5))

    def _paper(self, body: Body) -> dict | None:
        return body.papers[body.shelf % len(body.papers)] if body.papers else None

    def _describe(self, place: str, body: Body) -> str:
        if place == "wheel":
            return f"a stone wheel, its mark pointing {self.MARKS[self.wheel]}"
        if place == "drum":
            return "a large drum of stretched hide"
        if place == "pond":
            return "a still pond"
        if place == "seedbed":
            return f"a seedbed: {self.PLANT[self.plant]}"
        if place == "sign":
            return f"a flickering sign showing {self._sign_text}"
        if place == "chest":
            return "an open chest with a smooth grey pebble inside" if self.chest_open else "a closed chest"
        if place == "scale":
            return "a balance scale, hanging level"
        if place == "wall":
            return "a high wall of rough stone; the valley ends here"
        if place == "archive":
            p = self._paper(body)
            return (f"shelves of papers; the one in front of me is titled '{p['title']}'" if p
                    else "empty shelves")
        return place

    def sense(self):
        return self._sense(self._solo)

    def act(self, verb: str, words: str | None = None):
        return self._act(self._solo, verb, words)

    def _sense(self, body: Body):
        place = self.places[body.pos]
        out = [event(self.name, "percept", text=f"I am at the {place}. I see {self._describe(place, body)}.")]
        if self.shared:
            here = self._here(body)
            for o in here:
                still = ", lying still, as if asleep" if o.asleep else ""
                out.append(event(self.name, "percept", text=f"{self._someone(o)} is here{still}."))
            if here:
                out.append(event(self.name, "contact", amount=0.02))     # company, faintly
            for text, attention, contact in body.inbox:
                out.append(event(self.name, "percept", text=text, attention=attention))
                if contact:
                    out.append(event(self.name, "contact", amount=contact))
            body.inbox.clear()
        return out

    def _act(self, body: Body, verb: str, words: str | None = None):
        if not self.shared:
            self.clock += 1
        place = self.places[body.pos]
        if verb == "walk":
            nxt = self.places[(body.pos + 1) % len(self.places)]
            if self.shared:
                self._tell(body, f"{self._someone(body)} walked on towards the {nxt}.")
            body.pos = (body.pos + 1) % len(self.places)
            if place == "chest" and not any(o.pos == self.places.index("chest")
                                            for o in self.bodies.values() if o is not body):
                self.chest_pokes, self.chest_open = 0, False     # it shuts behind the last one
            if self.shared:
                self._tell(body, f"{self._someone(body)} arrived.", contact=0.1)
            return [event(self.name, "outcome", ok=True, text=f"I walked on and reached the {nxt}.")]
        if place == "chest" and verb != "poke":
            self.chest_pokes = 0                                 # "in a row" means in a row
        if verb == "wait":
            return [event(self.name, "outcome", ok=True, text="I waited. Time passed.")]
        if verb == "look":
            if self.shared:
                self._tell(body, f"{self._someone(body)} looked closely at the {place}.")
            return [event(self.name, "outcome", ok=True, text=self._look(place, body))]
        if verb == "poke":
            text = self._poke(place, body)
            if self.shared:
                self._witness(body, place, text)
            return [event(self.name, "outcome", ok=True, text=text)]
        if verb == "speak" and self.shared:
            return [event(self.name, "outcome", ok=True, text=self._speak(body, words))]
        return [event(self.name, "outcome", ok=False, text=f"I could not {verb} here.")]

    def _witness(self, body: Body, place: str, text: str) -> None:
        who = self._someone(body)
        if place == "drum" and "boom" in text:
            self._tell(body, f"{who} struck the drum: a deep boom!",
                       far="I heard a deep boom roll across the valley.")
        elif place == "drum":
            self._tell(body, f"{who} struck the drum: a dull, flat thud.")
        elif place == "chest" and "gave way" in text:
            self._tell(body, f"{who} pushed the lid of the chest and it gave way.")
        elif place == "scale":
            side = "left" if "left" in text else "right"
            self._tell(body, f"{who} pushed the scale; it settled tipped to the {side}.")
        else:
            self._tell(body, f"{who} {self.WITNESS.get(place, 'touched something')}.")

    def _speak(self, body: Body, words: str | None) -> str:
        who = self._someone(body)
        words = " ".join(str(words or "").split()).strip(" \"'")[:240]
        if not words:
            self._tell(body, f"{who} called out.", far="I heard a call from somewhere in the valley.",
                       next_door=f"I heard a call from the {self.places[body.pos]} nearby.",
                       attention=True, contact=0.2)
            return "I called out. My voice carried across the valley."
        self._tell(body, f'{who} said: "{words}"',
                   far="I heard a voice from somewhere in the valley, too far away to make out words.",
                   next_door=f'From the {self.places[body.pos]} nearby, a voice said: "{words}"',
                   attention=True, contact=0.3)
        return f'I said aloud: "{words}"'

    def _look(self, place: str, body: Body) -> str:
        if place == "sign":
            self._sign_text = self._glyphs()                     # noise, every time
        if place == "wall":
            return "I looked closely: rough stones rising far above my head. I cannot see over it."
        if place == "pond" and self.shared:
            return f"I looked into the still pond and saw my reflection, with {_a(body.mark)} mark."
        if place == "archive" and body.papers:
            body.shelf, body.page = (body.shelf + 1) % len(body.papers), 0
            p = self._paper(body)
            return f"I looked along the shelves. The next paper is titled '{p['title']}' ({len(p['pages'])} pages)."
        return f"I looked closely: {self._describe(place, body)}."

    def _poke(self, place: str, body: Body) -> str:
        if place == "wheel":
            self.wheel = (self.wheel + 1) % 4
            return f"I turned the wheel; its mark now points {self.MARKS[self.wheel]}."
        if place == "drum":
            if self.wheel == 2:                                  # the cross-place rule
                return "I struck the drum and it gave a deep boom that rolled across the valley!"
            return "I struck the drum. A dull, flat thud."
        if place == "pond":
            return "I stirred the pond. Ripples, then stillness again."
        if place == "seedbed":
            soon = self.clock - self.watered_at < 5              # the timing rule
            self.watered_at = self.clock
            if soon:
                return f"I watered the seedbed, but the soil is already soaked. Still {self.PLANT[self.plant]}."
            self.plant = (self.plant + 1) % len(self.PLANT)
            if self.plant == 0:
                return "I watered the seedbed; the flower dropped its petals and the bed is bare soil again."
            return f"I watered the seedbed; now there is {self.PLANT[self.plant]}."
        if place == "sign":
            self._sign_text = self._glyphs()
            return f"I tapped the sign; it now shows {self._sign_text}."
        if place == "chest":
            if self.chest_open:
                return "I touched the smooth grey pebble in the open chest. It is cool and heavy."
            self.chest_pokes += 1
            if self.chest_pokes >= 3:                           # the counting rule
                self.chest_open = True
                return "I pushed the lid again and it gave way: inside lies a smooth grey pebble."
            return "I pushed the lid of the chest. It creaks but holds."
        if place == "scale":
            side = "left" if self.rng.random() < 0.75 else "right"   # learnable odds
            return f"I pushed the scale; it swung and settled tipped to the {side}."
        if place == "wall":
            return "I pushed against the wall. It does not move."
        if place == "archive":
            p = self._paper(body)
            if not p:
                return "There is nothing to read."
            if body.page >= len(p["pages"]):
                return f"I have read '{p['title']}' to the end."
            text = p["pages"][body.page]
            body.page += 1
            return f"I read '{p['title']}', page {body.page} of {len(p['pages'])}: {text}"
        return "Nothing happened."


class BodyView:
    """What one Brain ticks against in a shared world: its own senses and acts."""

    def __init__(self, world: Valley, body: Body):
        self.world, self.body = world, body
        self.name = world.name
        self.verbs = {**world.verbs, **world.SOCIAL_VERBS}

    @property
    def place(self) -> str:
        return self.world.places[self.body.pos]

    def sense(self):
        return self.world._sense(self.body)

    def act(self, verb: str, words: str | None = None):
        return self.world._act(self.body, verb, words)


def make_world(name: str, *, seed: int = 0, bundle=None):
    if name in ("toy", "toy-garden"):
        return ToyGarden(seed=seed)
    if name == "toy-quiet":                             # the same garden without the noisy sign
        return ToyGarden(seed=seed, noise=False)
    if name == "valley":                                # with the archive of the mind's origins
        from .origins import load_archive
        return Valley(seed=seed, archive=load_archive(bundle))
    if name == "valley-closed":                         # the same valley without the archive
        return Valley(seed=seed)
    raise ValueError(f"unknown world: {name!r} (expected toy, toy-quiet, valley or valley-closed)")
