-- Ferrovore FD-25 car dismantler: hydraulic, slew, rotator and travel control.
-- Part of the CarDestroyer BeamNG.drive mod.
--
-- Every function of the machine is a "channel":
--   linear  : boom, stick, tilt, jaw, clamp   (hydraulic cylinders = jbeam hydros)
--   angular : slew, rot                       (swing ring / claw rotator = jbeam torsionHydros)
-- Operator inputs set *velocities*; the controller integrates them into target
-- cylinder lengths / angles, with valve-spool lag, pump-flow sharing between
-- functions, engine auto-idle, end-of-stroke cushioning and relief-valve
-- behaviour (anti-windup against the measured cylinder length).  Hydros and
-- torsion hydros then servo to those targets at the 2 kHz physics rate.

local M = {}
M.type = "auxiliary"
M.relevantDevice = nil
M.defaultOrder = 900

local min, max, abs, sqrt, exp, floor = math.min, math.max, math.abs, math.sqrt, math.exp, math.floor
local atan2 = math.atan2 or math.atan
local pi = math.pi

local function clampf(x, a, b)
  if x < a then return a elseif x > b then return b end
  return x
end
local function sgn(x)
  if x > 0 then return 1 elseif x < 0 then return -1 end
  return 0
end

local LINEAR = {"boom", "stick", "tilt", "jaw", "clamp"}
local ANGULAR = {"slew", "rot"}
local ALL = {"boom", "stick", "tilt", "jaw", "clamp", "slew", "rot"}
local TITLES = {boom = "Boom", stick = "Stick", tilt = "Tilt", jaw = "Jaw", clamp = "Clamp arm", slew = "Swing", rot = "Rotator"}
local POWER_ORDER = {"eco", "standard", "power"}
local POWER_FLOW = {eco = 0.72, standard = 0.86, power = 1.0}
local POWER_RPM = {eco = 0.84, standard = 0.93, power = 1.0}

-- input names (registered in input.state so BeamNG applies keyboard / gamepad smoothing)
local IN = {}
for _, c in ipairs(ALL) do
  local cap = c:sub(1, 1):upper() .. c:sub(2)
  IN[c] = {plus = "fd25" .. cap .. "Plus", minus = "fd25" .. cap .. "Minus", axis = "fd25" .. cap .. "Axis"}
end
local IN_TRACK_L, IN_TRACK_R = "fd25TrackLAxis", "fd25TrackRAxis"

local cfg = {}
local ch = {}
local st = {}
local nodeCid = {}
local snd = {}
local hudTimer, msgCategory = 0, "fd25"
local wheelCids = {L = {}, R = {}}

-- --------------------------------------------------------------------------- helpers
local function msg(text, ttl)
  if guihooks and guihooks.message then
    pcall(guihooks.message, text, ttl or 3, msgCategory)
  end
end

local function npos(name)
  local cid = nodeCid[name]
  if not cid then return nil end
  local p = obj:getNodePosition(cid)
  return p.x, p.y, p.z
end

local function v3(name)
  local x, y, z = npos(name)
  if not x then return nil end
  return {x, y, z}
end

local function sub(a, b) return {a[1] - b[1], a[2] - b[2], a[3] - b[3]} end
local function dot(a, b) return a[1] * b[1] + a[2] * b[2] + a[3] * b[3] end
local function cross(a, b) return {a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1]} end
local function len(a) return sqrt(dot(a, a)) end
local function scale(a, s) return {a[1] * s, a[2] * s, a[3] * s} end
local function norm(a)
  local l = len(a)
  if l < 1e-9 then return {0, 0, 0} end
  return scale(a, 1 / l)
end

-- signed angle of refB relative to refA around the axis axA -> axB (right-hand rule)
local function axisAngle(axA, axB, refA, refB)
  local a0, a1, ra, rb = v3(axA), v3(axB), v3(refA), v3(refB)
  if not (a0 and a1 and ra and rb) then return nil end
  local ax = norm(sub(a1, a0))
  local u, w = sub(ra, a0), sub(rb, a0)
  u = sub(u, scale(ax, dot(u, ax)))
  w = sub(w, scale(ax, dot(w, ax)))
  return atan2(dot(ax, cross(u, w)), dot(u, w))
end

local function readIn(name)
  return electrics.values[name .. "_input"] or 0
end

-- deadzone + gentle expo so analog sticks give fine control near centre
local function shape(x)
  local a = abs(x)
  if a < 0.04 then return 0 end
  a = (a - 0.04) / 0.96
  return sgn(x) * a * (0.4 + 0.6 * a)
end

local function defineInput(name, centered)
  if input.state[name] then return end
  input.state[name] = {
    val = 0, filter = 0,
    smootherKBD = newTemporalSmoothing(7, 10, nil, 0),
    smootherPAD = newTemporalSmoothing(40, 40, nil, 0),
    minLimit = centered and -1 or 0,
    maxLimit = 1
  }
end

-- --------------------------------------------------------------------------- channel model
local function linearOutput(c)
  local L = c.target + (c.comp or 0)
  if L >= c.spawn then
    return clampf((L - c.spawn) / max(c.max - c.spawn, 1e-6), 0, 1)
  end
  return clampf((L - c.spawn) / max(c.spawn - c.min, 1e-6), -1, 0)
end

local function measuredLength(c)
  if #c.cids == 0 then return nil end
  local s = 0
  for _, cid in ipairs(c.cids) do
    s = s + obj:getBeamLength(cid)
  end
  return s / #c.cids
end

local function measuredPressure(c)
  if #c.cids == 0 or not obj.getBeamStress then return 0 end
  local f = 0
  for _, cid in ipairs(c.cids) do
    f = f + abs(obj:getBeamStress(cid))
  end
  return f / max(c.pistonArea, 1e-6)
end

local function measuredAngle(c)
  if not c.refs then return nil end
  return axisAngle(c.refs[1], c.refs[2], c.refs[3], c.refs[4])
end

-- attachment pitch: 0 = hanging straight down, positive = claw tip forward
local function wrapAngle(a)
  while a > pi do a = a - 2 * pi end
  while a < -pi do a = a + 2 * pi end
  return a
end

-- pitch of the direction a0 -> a1 in the arm plane (0 = pointing straight down, + = towards the front)
local function planePitch(n0, n1)
  local a0, a1 = v3(n0), v3(n1)
  local pl, pr = v3(cfg.attAxis[1]), v3(cfg.attAxis[2])
  if not (a0 and a1 and pl and pr) then return nil end
  local axis = norm(sub(a1, a0))
  local n = norm(sub(pl, pr))
  local zw = {0, 0, 1}
  local f = norm(cross(n, zw))
  return atan2(dot(axis, f), -dot(axis, zw))
end

local function attachmentPitch()
  return planePitch(cfg.attPitchNodes[1], cfg.attPitchNodes[2])
end

local function resetState()
  st.mode = st.mode or "work"
  st.precision = false
  st.power = st.power or "standard"
  st.level = false
  st.levelHold = nil
  st.travelHigh = false
  st.macro = nil
  st.macroTime = 0
  st.rpm = cfg.idleRPM
  st.idleTimer = 10
  st.trackDir = cfg.trackDirection or 1
  st.trackCal = 0
  st.trackPhase = {L = 0, R = 0}
  st.idlerAngle = {L = 0, R = 0}
  st.sprocketAngle = {L = 0, R = 0}
  st.flowUse = 0
  st.pressure = 0
  st.stalledAny = false
  st.beaconT = 0
  st.time = 0
  st.vfInt = 0
  for _, name in ipairs(LINEAR) do
    local c = ch[name]
    if c then
      c.target = c.spawn
      c.vel, c.spool, c.stalled, c.comp = 0, 0, false, 0
    end
  end
  for _, name in ipairs(ANGULAR) do
    local c = ch[name]
    if c then
      c.target, c.vel, c.stalled = 0, 0, false
      c.calT, c.calStart, c.calCmd = 0, nil, nil
    end
  end
end

-- --------------------------------------------------------------------------- sounds
local function makeSound(kind, src, nodeName, profile)
  local nid = nodeCid[nodeName] or 0
  local ok, id = pcall(function()
    if kind == "event" then
      return obj:createSFXSource2(src, "AudioDefaultLoop3D", profile, nid, 1)
    end
    return obj:createSFXSource(src, "AudioDefaultLoop3D", profile, nid)
  end)
  if ok and id and id ~= -1 then
    pcall(obj.setVolumePitchCT, obj, id, 0, 1, 0, 0)
    pcall(obj.playSFX, obj, id)
    return id
  end
  return nil
end

local function setSound(id, vol, pitch)
  if id then
    pcall(obj.setVolumePitchCT, obj, id, vol, pitch, 0, 0)
  end
end

-- --------------------------------------------------------------------------- public toggles
local function setMode(m)
  st.mode = m
  st.macro = nil
  msg(m == "travel" and "FD-25: TRAVEL mode (arrows drive the tracks)" or "FD-25: WORK mode (arrows swing / boom)", 3)
end

local function toggleMode()
  setMode(st.mode == "work" and "travel" or "work")
end

local function togglePrecision()
  st.precision = not st.precision
  msg(st.precision and "FD-25: precision control ON" or "FD-25: precision control OFF", 2)
end

local function cyclePower()
  local idx = 1
  for i, p in ipairs(POWER_ORDER) do
    if p == st.power then idx = i end
  end
  st.power = POWER_ORDER[idx % #POWER_ORDER + 1]
  msg("FD-25: power mode " .. st.power:upper(), 2)
end

local function toggleLevelAssist()
  st.level = not st.level
  st.levelHold = nil
  msg(st.level and "FD-25: claw level assist ON" or "FD-25: claw level assist OFF", 2)
end

local function toggleTravelSpeed()
  st.travelHigh = not st.travelHigh
  msg(st.travelHigh and "FD-25: travel speed HIGH" or "FD-25: travel speed LOW", 2)
end

local function toggleBeacon()
  st.beacon = not st.beacon
  msg(st.beacon and "FD-25: beacon ON" or "FD-25: beacon OFF", 2)
end

local function startPose(name)
  if not (cfg.poses and cfg.poses[name]) then return end
  st.macro = name
  st.macroTime = 0
  msg("FD-25: moving to " .. name .. " pose (touch any control to cancel)", 3)
end

local function poseStow() startPose("stow") end
local function poseWork() startPose("work") end

-- --------------------------------------------------------------------------- main update
local function gatherCommands()
  local cmd = {}
  for _, name in ipairs(ALL) do
    local i = IN[name]
    cmd[name] = clampf(readIn(i.axis) + readIn(i.plus) - readIn(i.minus), -1, 1)
  end
  if st.mode == "work" then
    -- standard drive inputs double as swing (steer) and boom (throttle/brake)
    cmd.slew = clampf(cmd.slew - (input.steering or 0), -1, 1)
    cmd.boom = clampf(cmd.boom + (input.throttle or 0) - (input.brake or 0), -1, 1)
  end
  for k, v in pairs(cmd) do
    cmd[k] = shape(v)
  end
  return cmd
end

local function macroCommands(cmd, dt)
  if not st.macro then return end
  st.macroTime = st.macroTime + dt
  if st.macroTime > 0.4 then
    for _, name in ipairs(ALL) do
      if abs(cmd[name]) > 0.25 then
        st.macro = nil
        msg("FD-25: pose cancelled", 2)
        return
      end
    end
  end
  local pose = cfg.poses[st.macro]
  local done = true
  for _, name in ipairs(ALL) do
    local c = ch[name]
    local goal = pose[name]
    if c and goal then
      local err
      if c.kind == "linear" then
        err = (c.min + goal * (c.max - c.min)) - c.target
        if abs(err) > 0.01 then done = false end
        cmd[name] = clampf(err * 4.0 / max(c.maxSpeed, 1e-3), -0.85, 0.85)
      else
        err = goal * c.limit - c.target
        if abs(err) > 0.02 then done = false end
        cmd[name] = clampf(err * 2.5 / max(c.maxSpeed, 1e-3), -0.8, 0.8)
      end
    end
  end
  if done or st.macroTime > 25 then
    st.macro = nil
    msg("FD-25: pose reached", 2)
  end
end

local function levelAssist(cmd, dt)
  if not st.level or st.macro or dt <= 0 then
    st.levelHold, st.stickPrev = nil, nil
    return
  end
  local c = ch.tilt
  local phi = attachmentPitch()
  local theta = planePitch(cfg.stickAxis[1], cfg.stickAxis[2])
  if not (c and phi and theta) then return end
  -- feed-forward: the claw turns with the stick, so cancel the stick's measured rotation rate
  -- (boom, stick and machine motion all show up in it); feedback trims the remaining claw error
  local rate = 0
  if st.stickPrev then
    rate = wrapAngle(theta - st.stickPrev) / dt
  end
  st.stickPrev = theta
  st.stickRate = (st.stickRate or 0) + (rate - (st.stickRate or 0)) * (1 - exp(-dt / cfg.levelRateTau))
  if abs(cmd.tilt) > 0.02 or not st.levelHold then
    -- operator is tilting by hand (or first frame): (re)capture the angle to hold
    st.levelHold = phi
    return
  end
  local err = wrapAngle(st.levelHold - phi)
  local L = cfg.tiltLengthPerRad or -0.5          -- tilt cylinder metres per radian of claw pitch
  local v = clampf(L * (cfg.levelGain * err - st.stickRate), -c.maxSpeed, c.maxSpeed)
  cmd.tilt = v / max(c.maxSpeed, 1e-3)
end

local function updateEngine(cmd, travelDemand, dt)
  local demand = travelDemand or st.macro ~= nil
  for _, name in ipairs(ALL) do
    if abs(cmd[name]) > 0.03 then demand = true end
  end
  if demand then st.idleTimer = 0 else st.idleTimer = st.idleTimer + dt end
  local target = st.idleTimer > 4.0 and cfg.idleRPM or cfg.workRPM * POWER_RPM[st.power]
  local rate = target > st.rpm and 1300 or 650
  st.rpm = st.rpm + clampf(target - st.rpm, -rate * dt, rate * dt)
  return cfg.pumpFlowMax * POWER_FLOW[st.power] * clampf(st.rpm / cfg.maxRPM, 0, 1)
end

local function updateChannels(cmd, pumpQ, dt)
  local speedScale = st.precision and cfg.precisionScale or 1
  local spoolA = 1 - exp(-dt / cfg.spoolTau)
  -- 1) desired velocities (after valve spool lag) and flow demand
  local demandQ = 0
  for _, name in ipairs(ALL) do
    local c = ch[name]
    if c then
      local vd = cmd[name] * c.maxSpeed * speedScale
      c.spool = c.spool + (vd - c.spool) * spoolA
      if c.kind == "linear" then
        local area = c.spool >= 0 and c.pistonArea or c.rodArea
        demandQ = demandQ + abs(c.spool) * area
      else
        demandQ = demandQ + abs(c.spool) * c.flowCoef
      end
    end
  end
  local share = 1
  if demandQ > pumpQ and demandQ > 1e-9 then share = pumpQ / demandQ end
  st.flowUse = pumpQ > 0 and clampf(demandQ / pumpQ, 0, 9) or 0

  st.stalledAny = false
  local pmax = 0
  for _, name in ipairs(ALL) do
    local c = ch[name]
    if c then
      local want = c.spool * share
      local accel = c.accel
      if c.kind == "angular" and (abs(want) < abs(c.vel) or (c.vel ~= 0 and sgn(want) ~= sgn(c.vel))) then
        accel = c.decel
      elseif c.kind == "linear" and abs(want) < abs(c.vel) then
        accel = c.accel * 2
      end
      c.vel = c.vel + clampf(want - c.vel, -accel * dt, accel * dt)

      if c.kind == "linear" then
        -- end-of-stroke cushioning
        local stroke = c.max - c.min
        local zone = cfg.cushion * stroke
        local dist = c.vel > 0 and (c.max - c.target) or (c.target - c.min)
        if dist < zone then
          c.vel = c.vel * max(0.12, dist / zone)
        end
        c.target = clampf(c.target + c.vel * dt, c.min, c.max)
        if c.target <= c.min or c.target >= c.max then c.vel = 0 end
        -- relief valve / anti-windup against the real cylinder length
        local m = measuredLength(c)
        c.stalled = false
        if m then
          c.measured = m
          local err = c.target - m
          if abs(c.vel) > 1e-4 and err * sgn(c.vel) > c.windup then
            c.target = m + sgn(c.vel) * c.windup
            c.stalled = true
            st.stalledAny = true
          end
          -- load holding: oil is (almost) incompressible, so trim the command until the
          -- measured length matches the target; bounded so it can never exceed relief force
          if cfg.loadHolding and not c.stalled and abs(err) < c.windup then
            c.comp = clampf((c.comp or 0) + err * dt / cfg.holdTau, -c.compMax, c.compMax)
          else
            c.comp = (c.comp or 0) * max(0, 1 - dt / 0.3)
          end
        end
        c.pressure = measuredPressure(c)
        pmax = max(pmax, c.stalled and c.relief or min(c.pressure, c.relief))
        electrics.values[c.electrics] = linearOutput(c)
      else
        local limit = c.limit
        local cushionZone = 0.12
        local dist = c.vel > 0 and (limit - c.target) or (c.target + limit)
        if dist < cushionZone then
          c.vel = c.vel * max(0.15, dist / cushionZone)
        end
        c.target = clampf(c.target + c.vel * dt, -limit, limit)
        if abs(c.target) >= limit then c.vel = 0 end
        local m = measuredAngle(c)
        c.stalled = false
        if m then
          c.measured = m
          -- one-time direction auto-calibration (engine sign convention safety net)
          if cfg.autoCalibrate and not c.calibrated then
            if abs(c.vel) > 0.1 then
              if not c.calStart then
                c.calStart, c.calCmd, c.calT = m, c.target, 0
              end
              c.calT = c.calT + dt
              if c.calT > 0.5 then
                local dm, dc = m - c.calStart, c.target - c.calCmd
                if abs(dm) > 0.025 and abs(dc) > 0.04 then
                  if sgn(dm) ~= sgn(dc) then
                    c.sign = -c.sign
                    c.target = m
                    c.vel = 0
                    msg("FD-25: " .. TITLES[c.name] .. " direction calibrated", 2)
                  end
                  c.calibrated = true
                end
                c.calStart = nil
              end
            else
              c.calStart = nil
            end
          end
          local err = c.target - m
          if c.calibrated and abs(c.vel) > 1e-4 and err * sgn(c.vel) > c.windup then
            c.target = m + sgn(c.vel) * c.windup
            c.stalled = true
            st.stalledAny = true
          end
        end
        electrics.values[c.electrics] = clampf(c.sign * c.target / limit, -1, 1)
      end
    end
  end
  st.pressure = pmax
end

-- heading (CCW positive, world frame) from the undercarriage forward vector
local function heading()
  if not obj.getDirectionVector then return 0 end
  local d = obj:getDirectionVector()
  return atan2(d.x, -d.y)
end

local function wrapPi(a)
  while a > pi do a = a - 2 * pi end
  while a < -pi do a = a + 2 * pi end
  return a
end

local function forwardSpeed()
  if not (obj.getVelocity and obj.getDirectionVector) then return 0 end
  local v = obj:getVelocity()
  local d = obj:getDirectionVector()
  return v.x * d.x + v.y * d.y + v.z * d.z
end

local function sideWheelSpeed(side)
  local list = wheelCids[side]
  if not list or #list == 0 or not (wheels and wheels.wheelRotators) then return 0 end
  local s, n = 0, 0
  for _, cid in ipairs(list) do
    local wd = wheels.wheelRotators[cid]
    if wd and wd.wheelSpeed then
      s = s + wd.wheelSpeed
      n = n + 1
    end
  end
  return n > 0 and s / n or 0
end

local function updateTracks(dt)
  local fwd, turn = 0, 0
  if st.mode == "travel" then
    fwd = (input.throttle or 0) - (input.brake or 0)
    turn = input.steering or 0
  end
  -- optional pedal-style per-track inputs (work in both modes, like a real excavator)
  local pl, pr = readIn(IN_TRACK_L), readIn(IN_TRACK_R)
  local l = clampf(fwd + 0.9 * turn + pl, -1, 1)
  local r = clampf(fwd - 0.9 * turn + pr, -1, 1)
  local vmax = st.travelHigh and cfg.travelSpeedHigh or cfg.travelSpeedLow
  if st.precision then vmax = vmax * 0.5 end
  local vf = forwardSpeed()
  local common = (l + r) * 0.5
  local diff = (l - r) * 0.5
  local target = common * vmax
  local moving = abs(l) > 0.03 or abs(r) > 0.03
  local thrC = 0
  if moving then
    st.vfInt = clampf(st.vfInt + (target - vf) * dt, -1.5, 1.5)
    thrC = clampf(0.65 * common + 1.4 * (target - vf) + 0.35 * st.vfInt, -1, 1)
  else
    st.vfInt = 0
  end
  -- heading hold: keep straight when no steering is requested (load / slope asymmetry)
  local psi = heading()
  local psiRate = st.prevPsi and wrapPi(psi - st.prevPsi) / dt or 0
  st.prevPsi = psi
  if moving and abs(turn) < 0.02 and abs(pl - pr) < 0.05 and abs(common) > 0.05 then
    if not st.headingHold then st.headingHold = psi end
    local err = wrapPi(psi - st.headingHold)
    diff = diff + clampf(1.6 * err + 0.5 * psiRate, -0.4, 0.4) * sgn(common)
  else
    st.headingHold = nil
  end
  local thrL = clampf(thrC + 0.95 * diff, -1, 1) * st.trackDir
  local thrR = clampf(thrC - 0.95 * diff, -1, 1) * st.trackDir
  electrics.values.fd_trackL = thrL
  electrics.values.fd_trackR = thrR
  electrics.values.fd_trackFactorL = 1
  electrics.values.fd_trackFactorR = 1
  -- spring-applied travel brakes: on whenever a side is not commanded
  electrics.values.fd_brakeL = abs(l) > 0.03 and 0 or 1
  electrics.values.fd_brakeR = abs(r) > 0.03 and 0 or 1

  -- travel-direction auto calibration (straight driving only)
  if cfg.autoCalibrate and not st.trackCalibrated and abs(target) > 0.3 and abs(diff) < 0.2 then
    st.trackCal = st.trackCal + dt
    if st.trackCal > 0.7 and abs(vf) > 0.12 then
      if sgn(vf) ~= sgn(target) then
        st.trackDir = -st.trackDir
        msg("FD-25: travel direction calibrated", 2)
      end
      st.trackCalibrated = true
    end
  else
    st.trackCal = 0
  end

  -- visual track motion (props): phase in metres, wrapped at the shoe pitch
  for _, side in ipairs({"L", "R"}) do
    local v = sideWheelSpeed(side) * st.trackDir
    local pitch = cfg.shoePitch
    local ph = (st.trackPhase[side] + v * dt) % pitch
    st.trackPhase[side] = ph
    st.idlerAngle[side] = (st.idlerAngle[side] + v * dt / cfg.idlerRadius * 180 / pi) % 360
    st.sprocketAngle[side] = (st.sprocketAngle[side] + v * dt / cfg.sprocketRadius * 180 / pi) % 360
    electrics.values["fd_trackPhase" .. side] = ph
    electrics.values["fd_idlerAngle" .. side] = st.idlerAngle[side]
    electrics.values["fd_sprocketAngle" .. side] = st.sprocketAngle[side]
  end
  return moving, abs(vf)
end

local function updateSounds(trackSpeed, dt)
  local rpmN = st.rpm / cfg.maxRPM
  setSound(snd.engine, 0.35 + 0.45 * rpmN, 0.55 + 0.6 * rpmN)
  local flow = clampf(st.flowUse, 0, 1.2)
  setSound(snd.pump, clampf(0.15 + 0.85 * flow, 0, 1) * (0.4 + 0.6 * rpmN), 0.7 + 0.5 * rpmN)
  setSound(snd.relief, st.stalledAny and 0.55 or 0, 0.9 + 0.2 * rpmN)
  setSound(snd.track, clampf(trackSpeed / 1.2, 0, 1) * 0.9, 0.6 + 0.6 * clampf(trackSpeed / 1.6, 0, 1))
  -- travel alarm (beeps while the tracks move)
  if snd.alarmEvent then
    local want = trackSpeed > 0.15
    if want and not st.alarmOn then
      snd.alarm = snd.alarm or makeSound("event", snd.alarmEvent, cfg.soundNodeTrack, "fd25TravelAlarm")
      if snd.alarm then pcall(obj.playSFX, obj, snd.alarm) end
      st.alarmOn = true
    elseif not want and st.alarmOn then
      if snd.alarm then pcall(obj.stopSFX, obj, snd.alarm) end
      st.alarmOn = false
    end
  end
end

local function updateHud(dt)
  hudTimer = hudTimer + dt
  if hudTimer < 0.1 then return end
  hudTimer = 0
  local d = {
    mode = st.mode, precision = st.precision, power = st.power, level = st.level,
    travelHigh = st.travelHigh, macro = st.macro, beacon = st.beacon or false, rpm = floor(st.rpm),
    pressureMPa = floor(st.pressure / 1e5) / 10, flow = floor(clampf(st.flowUse, 0, 9) * 100),
    cyl = {}
  }
  for _, name in ipairs(LINEAR) do
    local c = ch[name]
    if c then
      d.cyl[name] = floor(100 * (c.target - c.min) / max(c.max - c.min, 1e-6))
      d.cyl[name .. "Stall"] = c.stalled
    end
  end
  for _, name in ipairs(ANGULAR) do
    local c = ch[name]
    if c then
      d[name .. "Deg"] = floor((c.measured or c.target) * 180 / pi)
    end
  end
  if guihooks and guihooks.trigger then
    pcall(guihooks.trigger, "FD25HudUpdate", d)
  end
end

local function updateGFX(dt)
  if dt <= 0 then return end
  st.time = st.time + dt
  local cmd = gatherCommands()
  macroCommands(cmd, dt)
  levelAssist(cmd, dt)
  local moving, trackSpeed = updateTracks(dt)
  local pumpQ = updateEngine(cmd, moving, dt)
  updateChannels(cmd, pumpQ, dt)

  electrics.values.fd_rpm = st.rpm
  electrics.values.fd_pressure = st.pressure / 1e6
  electrics.values.fd_flowUse = st.flowUse
  electrics.values.fd_modeTravel = st.mode == "travel" and 1 or 0
  electrics.values.fd_precision = st.precision and 1 or 0
  -- amber beacon pulses while switched on (own toggle, or the vehicle's lightbar electric)
  local lb = electrics.values[cfg.beaconElectrics]
  local beaconOn = st.beacon or lb == true or (type(lb) == "number" and lb > 0)
  if beaconOn then
    st.beaconT = st.beaconT + dt
    local s = math.sin(st.beaconT * 2 * pi * 1.3)
    electrics.values.fd_beacon = s > 0.2 and s or 0
  else
    st.beaconT = 0
    electrics.values.fd_beacon = 0
  end
  updateSounds(trackSpeed, dt)
  updateHud(dt)
end

-- --------------------------------------------------------------------------- init / reset
local function buildChannels(data)
  ch = {}
  local hydrosBySource = {}
  for _, h in pairs((v and v.data and v.data.hydros) or {}) do
    local src = h.inputSource
    if src then
      hydrosBySource[src] = hydrosBySource[src] or {}
      table.insert(hydrosBySource[src], h)
    end
  end
  for name, c in pairs(data.channels or {}) do
    local s = {name = name, kind = c.type, electrics = c.electrics, maxSpeed = c.maxSpeed or 0.3,
               accel = c.accel or 0.5, vel = 0, spool = 0, stalled = false, sign = 1}
    if c.type == "linear" then
      s.min, s.max, s.spawn = c.lengthMin, c.lengthMax, c.lengthSpawn
      s.pistonArea, s.rodArea = c.pistonArea or 0.01, c.rodArea or 0.006
      s.cids, s.relief, s.windup = {}, 34e6, 0.02
      local springSum, flimit = 0, 0
      for _, h in ipairs(hydrosBySource[c.electrics] or {}) do
        local cid = h.beamCID or h.cid or (h.beam and h.beam.cid)
        if cid then
          table.insert(s.cids, cid)
          local b = v.data.beams and v.data.beams[cid]
          springSum = springSum + ((b and b.beamSpring) or h.beamSpring or 3e7)
        end
        flimit = flimit + (h.forceLimit or 0)
      end
      if #s.cids > 0 and flimit > 0 then
        -- stall when the command leads the cylinder by roughly the relief force
        s.windup = clampf(1.15 * flimit / springSum, 0.004, data.antiWindup or 0.05)
        s.relief = flimit / max(s.pistonArea, 1e-6)
      end
      s.comp, s.compMax = 0, 0.6 * s.windup
    else
      s.limit = c.angleLimit or 2.96
      s.decel = c.decel or s.accel
      s.flowCoef = c.flowCoef or 0.001
      s.windup = 0.10
      if name == "slew" then
        s.sign = data.slewSign or 1
        s.refs = {data.slewAxis[1], data.slewAxis[2], data.slewRefUC, data.slewRefSS}
      else
        s.sign = data.rotSign or 1
        s.refs = {data.rotAxis[1], data.rotAxis[2], data.rotRefStator, data.rotRefHead}
      end
    end
    ch[name] = s
  end
end

local function init(jbeamData)
  cfg = {}
  for k, val in pairs(jbeamData or {}) do cfg[k] = val end
  cfg.idleRPM = cfg.idleRPM or 900
  cfg.workRPM = cfg.workRPM or 1950
  cfg.maxRPM = cfg.maxRPM or 2050
  cfg.pumpFlowMax = cfg.pumpFlowMax or 0.0072
  cfg.precisionScale = cfg.precisionScale or 0.35
  cfg.spoolTau = cfg.spoolTau or 0.12
  cfg.cushion = cfg.cushion or 0.05
  cfg.travelSpeedLow = cfg.travelSpeedLow or 0.83
  cfg.travelSpeedHigh = cfg.travelSpeedHigh or 1.53
  cfg.shoePitch = cfg.shoePitch or 0.19
  cfg.idlerRadius = cfg.idlerRadius or 0.36
  cfg.sprocketRadius = cfg.sprocketRadius or 0.39
  cfg.attAxis = cfg.attAxis or {"snL", "snR"}
  cfg.attPitchNodes = cfg.attPitchNodes or {"rt0", "rt1"}
  cfg.beaconElectrics = cfg.beaconElectrics or "lightbar"
  if cfg.autoCalibrate == nil then cfg.autoCalibrate = true end
  if cfg.loadHolding == nil then cfg.loadHolding = false end
  cfg.holdTau = cfg.holdTau or 0.45
  cfg.levelGain = cfg.levelGain or 3.0
  cfg.levelRateTau = cfg.levelRateTau or 0.08
  cfg.stickAxis = cfg.stickAxis or {"bnL", "snL"}

  nodeCid = {}
  if beamstate and beamstate.nodeNameMap then
    for k, cid in pairs(beamstate.nodeNameMap) do nodeCid[k] = cid end
  end
  if v and v.data and v.data.nodes then
    for cid, n in pairs(v.data.nodes) do
      if n.name and not nodeCid[n.name] then nodeCid[n.name] = n.cid or cid end
    end
  end

  for _, name in ipairs(ALL) do
    defineInput(IN[name].plus, false)
    defineInput(IN[name].minus, false)
    defineInput(IN[name].axis, true)
  end
  defineInput(IN_TRACK_L, true)
  defineInput(IN_TRACK_R, true)

  buildChannels(cfg)
  st = {}
  resetState()
  for _, side in ipairs({"L", "R"}) do
    electrics.values["fd_track" .. side] = 0
    electrics.values["fd_brake" .. side] = 1
    electrics.values["fd_trackFactor" .. side] = 1
  end
  for _, c in pairs(ch) do electrics.values[c.electrics] = 0 end
  electrics.values.fd_beacon = 0
end

local function initSecondStage(jbeamData)
  -- wheels are fully initialised now: map rotator names to cids for track speeds
  wheelCids = {L = {}, R = {}}
  if wheels and wheels.wheelRotatorIDs then
    for name, cid in pairs(wheels.wheelRotatorIDs) do
      local side = name:match("^RW_([LR])")
      if side then table.insert(wheelCids[side], cid) end
    end
  end
end

local function initLastStage(jbeamData)
  -- hydros are initialised after controllers; rebuild beam cid lookups now
  buildChannels(cfg)
  resetState()
end

local function initSounds(jbeamData)
  snd = {}
  local dir = cfg.soundDir or "vehicles/ferrovore_fd25/sounds/"
  snd.engine = makeSound("file", dir .. "fd25_diesel_loop.ogg", cfg.soundNodeEngine, "fd25DieselLoop")
  snd.pump = makeSound("event", cfg.pumpEvent or "event:>Vehicle>Hydraulics>Pump_Big", cfg.soundNodePump, "fd25PumpLoop")
  snd.relief = makeSound("file", dir .. "fd25_relief_loop.ogg", cfg.soundNodePump, "fd25ReliefLoop")
  snd.track = makeSound("file", dir .. "fd25_track_loop.ogg", cfg.soundNodeTrack, "fd25TrackLoop")
  snd.alarmEvent = cfg.travelAlarmEvent or "event:>Vehicle>Electrics>Reverse>Beep_01"
  st.alarmOn = false
end

local function resetSounds(jbeamData)
  for _, id in pairs({snd.engine, snd.pump, snd.relief, snd.track}) do setSound(id, 0, 1) end
  if snd.alarm then pcall(obj.stopSFX, obj, snd.alarm) end
  st.alarmOn = false
end

local function reset(jbeamData)
  local keepMode, keepPower = st.mode, st.power
  buildChannels(cfg)
  st = {}
  st.mode, st.power = keepMode, keepPower
  resetState()
  for _, c in pairs(ch) do electrics.values[c.electrics] = 0 end
end

-- debug / test access
local function getState()
  return st, ch, cfg
end

M.init = init
M.initSecondStage = initSecondStage
M.initLastStage = initLastStage
M.initSounds = initSounds
M.resetSounds = resetSounds
M.reset = reset
M.updateGFX = updateGFX

M.setMode = setMode
M.toggleMode = toggleMode
M.togglePrecision = togglePrecision
M.cyclePower = cyclePower
M.toggleLevelAssist = toggleLevelAssist
M.toggleTravelSpeed = toggleTravelSpeed
M.toggleBeacon = toggleBeacon
M.poseStow = poseStow
M.poseWork = poseWork
M.getState = getState

return M
