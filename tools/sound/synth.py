"""Synthesise the FD-25's looping sounds (mono 44.1 kHz Ogg Vorbis).

Every layer is built to be exactly periodic over the loop length (integer cycle
counts, circular convolution, spectrally shaped noise), so the loops are
seamless without crossfades.  Pitch 1.0 of each loop corresponds to the
mid-point of the controller's pitch range (see fd25Hydraulics.lua updateSounds):

  fd25_diesel_loop.ogg  6-cyl diesel at ~1530 rpm (firing 76.5 Hz), turbo whine
  fd25_relief_loop.ogg  relief-valve squeal + hiss + pump ripple
  fd25_track_loop.ogg   track-link clank at ~1.05 m/s (5.5 links/s) + roller grind

Run:  python -m tools.sound.synth
"""
import os

import numpy as np
import soundfile as sf

from tools.spec import machine as M

SR = 44100
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "mod", "vehicles", M.MODEL_ID, "sounds")


# ----------------------------------------------------------------------------- periodic building blocks
def periodic_noise(n, lo, hi, rng, tilt=0.0):
    """Band-limited noise that loops perfectly (random-phase spectrum, 1/f^tilt slope)."""
    freqs = np.fft.rfftfreq(n, 1 / SR)
    mag = ((freqs >= lo) & (freqs <= hi)).astype(float)
    # soft band edges
    edge = 0.15
    mag = np.where(freqs < lo, np.exp(-((lo - freqs) / (lo * edge + 1)) ** 2), mag)
    mag = np.where(freqs > hi, np.exp(-((freqs - hi) / (hi * edge + 1)) ** 2), mag)
    mag *= np.where(freqs > 0, (np.maximum(freqs, 1) / max(lo, 1)) ** (-tilt), 0)
    spec = mag * np.exp(1j * rng.uniform(0, 2 * np.pi, len(freqs)))
    x = np.fft.irfft(spec, n)
    return x / (np.abs(x).max() + 1e-12)


def circ_conv(x, h):
    n = len(x)
    hp = np.zeros(n)
    hp[:min(n, len(h))] = h[:n]
    return np.real(np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(hp), n))


def place(n, times, amps):
    """Impulse train at given times (seconds), wrapped into the loop (fractional delays via linear split)."""
    x = np.zeros(n)
    for t, a in zip(times, amps):
        p = (t * SR) % n
        i = int(p)
        f = p - i
        x[i % n] += a * (1 - f)
        x[(i + 1) % n] += a * f
    return x


def modal(freqs, decays, amps, length):
    t = np.arange(int(length * SR)) / SR
    y = np.zeros_like(t)
    for f, d, a in zip(freqs, decays, amps):
        y += a * np.exp(-t / d) * np.sin(2 * np.pi * f * t)
    return y


def periodic_freq(f, seconds):
    """Snap a frequency so it completes an integer number of cycles in the loop."""
    return max(1, round(f * seconds)) / seconds


def onepole_lp(x, fc):
    """Zero-phase-ish periodic low-pass via FFT (Butterworth-like magnitude)."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    H = 1 / np.sqrt(1 + (freqs / fc) ** 4)
    return np.fft.irfft(np.fft.rfft(x) * H, n)


def bandpass(x, lo, hi, order=2):
    n = len(x)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    H = 1 / np.sqrt(1 + (lo / np.maximum(freqs, 1e-3)) ** (2 * order)) / np.sqrt(1 + (freqs / hi) ** (2 * order))
    return np.fft.irfft(np.fft.rfft(x) * H, n)


def soft_clip(x, drive=1.0):
    return np.tanh(x * drive) / np.tanh(drive)


def normalise(x, peak=0.89):
    x = x - x.mean()
    return x * (peak / (np.abs(x).max() + 1e-12))


def write(name, x):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    sf.write(path, x.astype(np.float32), SR, format="OGG", subtype="VORBIS")
    return path


# ----------------------------------------------------------------------------- diesel
def diesel(seconds=4.0, seed=1):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    cycle_hz = periodic_freq(1530 / 60 / 2, seconds)        # 4-stroke cycle (12.75 Hz)
    fire_hz = cycle_hz * 6                                    # 6 cylinders
    n_fire = int(round(fire_hz * seconds))
    # firing order with per-cylinder strength / timing quirks (repeat every cycle -> periodic)
    cyl_amp = 1.0 + 0.10 * rng.standard_normal(6)
    cyl_dt = 0.00035 * rng.standard_normal(6)
    times = [k / fire_hz + cyl_dt[k % 6] for k in range(n_fire)]
    amps = [cyl_amp[k % 6] for k in range(n_fire)]
    pulses = place(n, times, amps)
    # combustion: knock resonance + low thump; exhaust: heavy low-passed version
    knock = modal([620, 940, 1380, 2150, 3100], [0.010, 0.008, 0.006, 0.004, 0.003],
                  [0.9, 0.7, 0.5, 0.35, 0.2], 0.05)
    thump = modal([fire_hz, fire_hz * 2, 140], [0.025, 0.015, 0.012], [1.0, 0.5, 0.6], 0.08)
    comb = circ_conv(pulses, knock)
    exhaust = onepole_lp(circ_conv(pulses, thump), 300)
    # gas-flow noise modulated by firing (breathing)
    env = onepole_lp(pulses, 60)
    env = env / (np.abs(env).max() + 1e-9)
    breath = periodic_noise(n, 80, 1800, rng, tilt=0.8) * (0.35 + 0.65 * np.clip(env, 0, None))
    # mechanical clatter (valvetrain / injectors), higher band noise gated at 2x cycle
    clat_t = [k / (cycle_hz * 12) for k in range(int(round(cycle_hz * 12 * seconds)))]
    clat = circ_conv(place(n, clat_t, 0.5 + 0.2 * rng.random(len(clat_t))),
                     modal([2600, 3700, 5200], [0.0025, 0.002, 0.0015], [1, 0.7, 0.4], 0.02))
    # turbo whine and cooling-fan hum (exactly periodic tones)
    turbo_f = periodic_freq(3150, seconds)
    lfo = np.sin(2 * np.pi * periodic_freq(0.5, seconds) * t)
    turbo = np.sin(2 * np.pi * turbo_f * t + 2.5 * lfo) * 0.035
    fan = (np.sin(2 * np.pi * periodic_freq(205, seconds) * t) * 0.03 +
           periodic_noise(n, 150, 700, rng, 0.5) * 0.05)
    # crank-order rumble (1st and 1/2 orders) gives the "big engine" body
    rumble = (0.20 * np.sin(2 * np.pi * cycle_hz * 2 * t) + 0.12 * np.sin(2 * np.pi * cycle_hz * t + 0.7))
    mix = (normalise(exhaust) * 0.95 + normalise(comb) * 0.42 + breath * 0.22 + normalise(clat) * 0.10 +
           turbo + fan + rumble * 0.35)
    mix = soft_clip(normalise(mix, 1.0), 1.6)
    return normalise(mix, 0.89)


# ----------------------------------------------------------------------------- relief valve
def relief(seconds=3.0, seed=2):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    # pump ripple: 9-piston pump at ~1530 rpm -> 229.5 Hz + harmonics, stressed (louder, harsher)
    ripple_f = periodic_freq(229.5, seconds)
    ripple = sum(a * np.sin(2 * np.pi * ripple_f * h * t + p)
                 for h, a, p in ((1, 1.0, 0), (2, 0.6, 1.1), (3, 0.35, 2.0), (4, 0.2, 0.4), (6, 0.12, 2.6)))
    # valve squeal: resonant whistle with slow warble and flutter (exactly periodic modulators)
    warble = np.sin(2 * np.pi * periodic_freq(5.3, seconds) * t)
    flutter = np.sin(2 * np.pi * periodic_freq(37.0, seconds) * t)
    squeal_f = periodic_freq(1460, seconds)
    phase = 2 * np.pi * squeal_f * t + 4.5 * warble + 0.4 * flutter      # FM: +-24 Hz warble, light flutter
    squeal = (np.sin(phase) + 0.45 * np.sin(2 * phase + 0.3) + 0.2 * np.sin(3 * phase + 1.0))
    squeal *= 0.75 + 0.25 * np.sin(2 * np.pi * periodic_freq(2.0, seconds) * t + 0.3)
    # high-velocity oil jet: hiss band
    hiss = periodic_noise(n, 2200, 9000, rng, 0.3)
    hiss *= 0.8 + 0.2 * flutter
    # structural groan (low, rough)
    groan = bandpass(periodic_noise(n, 60, 260, rng, 0.0), 70, 240) * (0.8 + 0.2 * warble)
    mix = 0.35 * normalise(ripple) + 0.42 * normalise(squeal) + 0.30 * hiss + 0.25 * normalise(groan)
    return normalise(soft_clip(normalise(mix, 1.0), 1.4), 0.85)


# ----------------------------------------------------------------------------- track clank
def track(seconds=4.0, seed=3):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    link_hz = periodic_freq(1.05 / M.UC.shoe_pitch, seconds)       # ~5.5 links/s
    n_links = int(round(link_hz * seconds))
    out = np.zeros(n)
    # sprocket engagement clanks (sharp, metallic) and idler clunks (duller, offset half a pitch)
    for k in range(n_links):
        for offset, bright in ((0.0, 1.0), (0.5, 0.55)):
            t0 = (k + offset + 0.04 * rng.standard_normal()) / link_hz
            nm = 6
            freqs = rng.uniform(380, 3600, nm) * (0.7 + 0.3 * bright)
            dec = rng.uniform(0.012, 0.06, nm)
            amp = rng.uniform(0.2, 1.0, nm) * bright
            ev = modal(freqs, dec, amp, 0.25) + modal([75 + 25 * rng.random(), 140], [0.05, 0.03], [1.4, 0.6], 0.25)
            imp = place(n, [t0], [0.6 + 0.4 * rng.random()])
            out += circ_conv(imp, ev)
    # continuous roller grind / sand crunch (amplitude follows link passing)
    grind = periodic_noise(n, 180, 2400, rng, 0.6)
    lm = 0.6 + 0.4 * np.sin(2 * np.pi * link_hz * t) ** 2
    # occasional roller squeak (two short periodic chirps per loop)
    squeak = np.zeros(n)
    for c in (0.9, 2.7):
        env = np.exp(-((t - c) / 0.07) ** 2)
        squeak += env * np.sin(2 * np.pi * periodic_freq(1900, seconds) * t + 6.0 * np.sin(2 * np.pi * 3 * t))
    mix = normalise(out) * 0.85 + grind * lm * 0.20 + squeak * 0.06
    return normalise(soft_clip(normalise(mix, 1.0), 1.3), 0.87)


def main():
    files = {
        "fd25_diesel_loop.ogg": diesel(),
        "fd25_relief_loop.ogg": relief(),
        "fd25_track_loop.ogg": track(),
    }
    for name, x in files.items():
        p = write(name, x)
        # seamless check: the wrap-around step must look like an ordinary sample step
        jump = abs(x[0] - x[-1])
        typical = np.percentile(np.abs(np.diff(x)), 99.5)
        print(f"  {name}: {len(x) / SR:.1f}s  {os.path.getsize(p) / 1024:.0f} KB  wrap step {jump:.3f} (p99.5 {typical:.3f})")


if __name__ == "__main__":
    main()
