"""Interactive 2-player Super Mario Kart play + save-state tool (Step 0).

Boots the SMK integration from power-on so you can navigate the menus into a
2-PLAYER MARIOKART GP (2 humans + 6 CPUs), then save a retro-compatible
savestate for each track as it loads. Run from WSL (Windows 11 WSLg gives the
window).

    # one Mushroom-Cup run can capture Mario Circuit 1 (track 1) and 2 (track 5):
    python -m tools.play_2p --names MarioCircuit_2P,DonutPlains_2P,GhostValley_2P,BowserCastle_2P,MarioCircuit2_2P

Save slots: each name maps to a number key. While a track is at the start line,
press its digit to write custom_integrations/SuperMarioKart-Snes/<name>.state:

    1 -> first name, 2 -> second name, ... (printed at startup)

Keyboard:
  Player 1                         Player 2
    Arrow keys = steer               I/J/K/L   = up/left/down/right
    Z          = accelerate (B)      SPACE     = accelerate (B)
    X          = Y                   COMMA (,) = Y
    A          = A (hop/drift)       PERIOD(.) = A (hop/drift)
    S          = X (item)            SLASH (/) = X (item)
    Q / W      = L / R               U / O     = L / R
    ENTER      = START               RSHIFT    = START
    TAB        = SELECT              BACKSPACE = SELECT

  1..9 = SAVE STATE to that slot's name      ESC = quit
"""
import argparse
import gzip
import os

import stable_retro as retro
from stable_retro.examples.interactive import Interactive

GAME = "SuperMarioKart-Snes"
HERE = os.path.dirname(os.path.abspath(__file__))
CUSTOM = os.path.abspath(os.path.join(HERE, "..", "custom_integrations"))
INT_DIR = os.path.join(CUSTOM, GAME)

# button name -> pyglet key name, per player
P1_MAP = {
    "B": "Z", "Y": "X", "A": "A", "X": "S", "L": "Q", "R": "W",
    "UP": "UP", "DOWN": "DOWN", "LEFT": "LEFT", "RIGHT": "RIGHT",
    "START": "ENTER", "SELECT": "TAB",
}
P2_MAP = {
    "B": "SPACE", "Y": "COMMA", "A": "PERIOD", "X": "SLASH", "L": "U", "R": "O",
    "UP": "I", "DOWN": "K", "LEFT": "J", "RIGHT": "L",
    "START": "RSHIFT", "SELECT": "BACKSPACE",
}


def read_kart(ram, idx):
    """Quick speed/checkpoint read so we can confirm both karts are live on save."""
    def s16(addr):
        return int.from_bytes(bytes(ram[addr:addr + 2]), "little", signed=True)
    speed = s16(0x10EA + 0x100 * idx)
    checkpoint = int(ram[0x10DC + 0x100 * idx])
    return speed, checkpoint


class TwoPlayerInteractive(Interactive):
    def __init__(self, state, save_slots):
        retro.data.Integrations.add_custom_path(CUSTOM)
        env = retro.make(
            game=GAME,
            state=state,
            players=2,
            render_mode="rgb_array",
            inttype=retro.data.Integrations.ALL,
        )
        self._buttons = env.buttons
        # key name (e.g. "_1") -> (name, path)
        self._save_slots = save_slots
        self._held = {k: False for k in save_slots}
        super().__init__(env=env, sync=False, tps=60, aspect_ratio=4 / 3)

    def get_image(self, _obs, env):
        return env.render()

    def _save_state(self, name, path):
        raw = self._env.unwrapped.em.get_state()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(gzip.compress(raw))
        ram = self._env.get_ram()
        s0, c0 = read_kart(ram, 0)
        s1, c1 = read_kart(ram, 1)
        print(f"\n*** SAVED [{name}] -> {path}")
        print(f"    kart0 speed={s0} checkpoint={c0} | kart1 speed={s1} checkpoint={c1}")
        print("    (both karts should be live in a real 2P race)\n")

    def keys_to_act(self, keys):
        # edge-triggered saves so holding a digit writes once
        for key_name, (name, path) in self._save_slots.items():
            if key_name in keys:
                if not self._held[key_name]:
                    self._save_state(name, path)
                self._held[key_name] = True
            else:
                self._held[key_name] = False

        p1 = {b: (P1_MAP[b] in keys) for b in self._buttons}
        p2 = {b: (P2_MAP[b] in keys) for b in self._buttons}
        return [p1[b] for b in self._buttons] + [p2[b] for b in self._buttons]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="MarioCircuit_2P",
                    help="comma-separated .state names (no extension); slot N = digit N")
    ap.add_argument("--state", default=retro.State.NONE,
                    help="state to boot from; default boots from power-on")
    args = ap.parse_args()

    names = [n.strip() for n in args.names.split(",") if n.strip()]
    if len(names) > 9:
        raise SystemExit("Max 9 save slots (digit keys 1-9).")
    save_slots = {f"_{i}": (name, os.path.join(INT_DIR, f"{name}.state"))
                  for i, name in enumerate(names, start=1)}

    print("Booting SMK (2 players). Save slots:")
    for key_name, (name, path) in save_slots.items():
        print(f"  press {key_name[1:]} -> {name}.state")
    TwoPlayerInteractive(state=args.state, save_slots=save_slots).run()


if __name__ == "__main__":
    main()
