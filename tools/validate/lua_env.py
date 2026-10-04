"""LuaJIT sandbox that mimics the BeamNG vehicle-Lua globals the FD-25 controller uses.

The physics side (node positions, beam lengths, stresses, wheel speeds, vehicle
velocity) is supplied by Python callbacks, so the same environment serves the
unit tests (simple fakes) and the physics co-simulation.
"""
import os

from lupa import luajit21 as lupa

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CTRL_PATH = os.path.join(ROOT, "mod", "vehicles", "ferrovore_fd25", "lua", "controller", "fd25Hydraulics.lua")

PRELUDE = r"""
electrics = {values = {}}
messages = {}
hud = {}
guihooks = {
  message = function(msg, ttl, cat) table.insert(messages, tostring(msg)) end,
  trigger = function(name, data) hud[name] = data end,
}
log = function(...) end
function newTemporalSmoothing(inRate, outRate, autoCenterRate, start)
  local s = {v = start or 0, inRate = inRate or 1e9, outRate = outRate or inRate or 1e9}
  function s:getCapped(x, dt)
    local d = x - self.v
    local rate
    if math.abs(x) > math.abs(self.v) then rate = self.outRate else rate = self.inRate end
    local stepMax = rate * dt
    if d > stepMax then d = stepMax elseif d < -stepMax then d = -stepMax end
    self.v = self.v + d
    return self.v
  end
  s.get = s.getCapped
  function s:reset() self.v = start or 0 end
  function s:value() return self.v end
  return s
end
input = {state = {}, steering = 0, throttle = 0, brake = 0, parkingbrake = 0}
function input.event(itype, ivalue, filter)
  if input.state[itype] == nil then error("undefined input " .. tostring(itype)) end
  input.state[itype].val = ivalue
  input.state[itype].filter = filter or 0
end
function input.updateGFX(dt)
  for k, e in pairs(input.state) do
    local ival = math.max(math.min(e.val or 0, e.maxLimit), e.minLimit)
    if e.filter == 1 then ival = e.smootherPAD:getCapped(ival, dt) else ival = e.smootherKBD:getCapped(ival, dt) end
    input[k] = ival
    electrics.values[k .. "_input"] = ival
  end
end
"""


class LuaVehicle:
    def __init__(self, phys):
        """phys: object providing node_pos(cid)->(x,y,z), beam_length(cid), beam_stress(cid),
        velocity()->(x,y,z), direction()->(x,y,z), node_cids (name->cid), hydros (list of dict),
        beams (cid->dict), wheel_ids (name->cid), wheel_speed(cid)."""
        self.L = lupa.LuaRuntime(unpack_returned_tuples=True)
        self.phys = phys
        L = self.L
        L.execute(PRELUDE)
        g = L.globals()
        lt = L.table_from

        def vec(t):
            return lt({"x": t[0], "y": t[1], "z": t[2]})

        sounds = {}

        def create_sfx(_self, src, desc, profile, node, *rest):
            sid = len(sounds) + 1
            sounds[sid] = dict(src=src, vol=0.0, pitch=1.0, playing=False)
            return sid

        def set_vp(_self, sid, vol, pitch, *rest):
            if sid in sounds:
                sounds[sid]["vol"], sounds[sid]["pitch"] = vol, pitch

        def play(_self, sid):
            if sid in sounds:
                sounds[sid]["playing"] = True

        def stop(_self, sid):
            if sid in sounds:
                sounds[sid]["playing"] = False

        self.sounds = sounds
        objt = lt({
            "getNodePosition": lambda _s, cid: vec(phys.node_pos(int(cid))),
            "getBeamLength": lambda _s, cid: float(phys.beam_length(int(cid))),
            "getBeamStress": lambda _s, cid: float(phys.beam_stress(int(cid))),
            "getVelocity": lambda _s: vec(phys.velocity()),
            "getDirectionVector": lambda _s: vec(phys.direction()),
            "createSFXSource2": create_sfx,
            "createSFXSource": create_sfx,
            "setVolumePitchCT": set_vp,
            "playSFX": play,
            "stopSFX": stop,
            "cutSFX": stop,
        })
        g.obj = objt
        g.beamstate = lt({"nodeNameMap": lt(dict(phys.node_cids))})
        hyd = lt({i + 1: lt(dict(h)) for i, h in enumerate(phys.hydros)})
        beams = lt({int(cid): lt(dict(b)) for cid, b in phys.beams.items()})
        g.v = lt({"data": lt({"hydros": hyd, "beams": beams, "nodes": lt({})})})
        rot = lt({int(cid): lt({"wheelSpeed": 0.0, "name": n}) for n, cid in phys.wheel_ids.items()})
        self._rot = rot
        g.wheels = lt({"wheelRotatorIDs": lt(dict(phys.wheel_ids)), "wheelRotators": rot})
        with open(CTRL_PATH) as f:
            src = f.read()
        self.ctrl = L.execute(src)
        self.g = g

    # ------------------------------------------------------------------
    def init(self, params):
        data = self._to_lua(params)
        self.ctrl.init(data)
        self.ctrl.initSecondStage(data)
        self.ctrl.initLastStage(data)
        self.ctrl.initSounds(data)
        self.params = data

    def _to_lua(self, x):
        if isinstance(x, dict):
            return self.L.table_from({k: self._to_lua(v) for k, v in x.items()})
        if isinstance(x, (list, tuple)):
            return self.L.table_from({i + 1: self._to_lua(v) for i, v in enumerate(x)})
        return x

    def press(self, name, value, filt=0):
        self.L.globals().input.event(name, value, filt)

    def set_drive(self, steering=0.0, throttle=0.0, brake=0.0):
        inp = self.L.globals().input
        inp.steering, inp.throttle, inp.brake = steering, throttle, brake

    def step(self, dt):
        for n, cid in self.phys.wheel_ids.items():
            self._rot[cid].wheelSpeed = float(self.phys.wheel_speed(cid))
        self.L.globals().input.updateGFX(dt)
        self.ctrl.updateGFX(dt)

    def ev(self, key, default=0.0):
        v = self.L.globals().electrics["values"][key]
        return default if v is None else v

    def call(self, fn):
        getattr(self.ctrl, fn)()

    def messages(self):
        m = self.L.globals().messages
        return [m[i] for i in range(1, len(m) + 1)]

    def state(self):
        st, ch, cfg = self.ctrl.getState()
        return st, ch, cfg
