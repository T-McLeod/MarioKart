"""Step-0 RAM probe for the 2-player integration work.

Loads the Super Mario Kart integration, steps the emulator with scripted
throttle inputs, and prints named RAM values for kart index 0 (Player 1) and
kart index 1 (Player 2 / first CPU). Used to validate the per-kart struct
strides documented in custom_integrations/SuperMarioKart-Snes/Notes.txt:

    * $10xx physics page: stride 0x100 per kart (P1 speed 0x10EA -> P2 0x11EA)
    * X/Y position:       interleaved 16-bit array, stride 2 (P1 X 0x88 -> P2 0x8A)

Run inside the mariokart-rl container, e.g.:

    docker run --rm -v "${PWD}:/workspace/MarioKart" mariokart-rl \
        python -u -m tools.ram_probe --state MarioCircuit_M --players 1 --frames 240
"""
import argparse
import os

import numpy as np
import stable_retro

GAME_NAME = "SuperMarioKart-Snes"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# name -> (address, n_bytes, signed). Addresses are WRAM offsets.
# Kart-indexed fields are given for kart 0; kart i is derived via the strides below.
PAGE10_STRIDE = 0x100   # $10xx physics page, per kart
POS_STRIDE = 0x02       # X/Y interleaved 16-bit array, per kart

KART0_FIELDS = {
    "X":          (0x88,   2, True),    # 136
    "Y":          (0x8C,   2, True),    # 140
    "speed":      (0x10EA, 2, True),    # 4330
    "checkpoint": (0x10DC, 1, False),   # 4316
    "lap":        (0x10C1, 1, False),   # 4289
    "surface":    (0x10AE, 1, False),   # 4270
    "rank":       (0x1040, 1, False),   # 4160
}
# which stride each field uses
FIELD_STRIDE = {
    "X": POS_STRIDE, "Y": POS_STRIDE,
    "speed": PAGE10_STRIDE, "checkpoint": PAGE10_STRIDE, "lap": PAGE10_STRIDE,
    "surface": PAGE10_STRIDE, "rank": PAGE10_STRIDE,
}


def read(ram, addr, n, signed):
    val = int.from_bytes(bytes(ram[addr:addr + n]), "little", signed=signed)
    return val


def kart_fields(ram, kart_idx):
    out = {}
    for name, (base, n, signed) in KART0_FIELDS.items():
        addr = base + FIELD_STRIDE[name] * kart_idx
        out[name] = read(ram, addr, n, signed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="MarioCircuit_M")
    ap.add_argument("--players", type=int, default=1)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--print-every", type=int, default=30)
    args = ap.parse_args()

    custom_path = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "custom_integrations"))
    stable_retro.data.Integrations.add_custom_path(custom_path)

    env = stable_retro.make(
        game=GAME_NAME,
        state=args.state,
        players=args.players,
        render_mode="rgb_array",
        inttype=stable_retro.data.Integrations.ALL,
    )
    print(f"retro action_space: {env.action_space}")
    obs, info = env.reset()
    print(f"obs shape: {np.asarray(obs).shape}")

    n_buttons = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
    # Hold B (accelerate, index 0) for every player block of 12 buttons.
    action = np.zeros(n_buttons, dtype=np.int8)
    for p in range(args.players):
        action[p * 12 + 0] = 1

    for frame in range(args.frames):
        obs, reward, terminated, truncated, info = env.step(action)
        if frame % args.print_every == 0 or frame == args.frames - 1:
            ram = env.get_ram()
            k0 = kart_fields(ram, 0)
            k1 = kart_fields(ram, 1)
            print(f"--- frame {frame} | reward={reward} done={terminated or truncated} ---")
            print(f"  kart0 (P1): {k0}")
            print(f"  kart1 (P2): {k1}")
        if terminated or truncated:
            print(f"episode ended at frame {frame}")
            break

    env.close()


if __name__ == "__main__":
    main()
