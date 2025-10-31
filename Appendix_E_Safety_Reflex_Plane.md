# Appendix E — Safety & Reflex Plane (SRP) Logic

## E.1 Objective and Requirements

The Safety & Reflex Plane (SRP) provides **hardware-enforced deterministic safety control** independent of inference workload and thermal conditions.

### E.1.1 Design Requirements

| Requirement | Specification | Source Standard |
|-------------|---------------|-----------------|
| **Deterministic timing** | <5 ms worst-case control loop | ISO 13849-1 |
| **Fault detection** | >90% diagnostic coverage | IEC 61508 SIL 3 |
| **Fail-safe behavior** | Actuators → safe state within 10 ms | ISO 10218-1 (robotics) |
| **Independence** | SRP operational during IP/AOP failure | IEC 61508 (independence req) |
| **Power isolation** | Separate V_DD, thermal domain | Design choice |
| **Redundancy** | Dual lock-step execution | IEC 61508 SIL 3 |

---

## E.2 Core Finite State Machine (FSM)

### E.2.1 State Diagram

```
                    ┌──────────────┐
              ┌────►│    INIT      │◄────┐
              │     │  (Power-on)  │     │
              │     └──────┬───────┘     │
              │            │             │
              │            ↓             │
              │     ┌──────────────┐     │
              │  ┌─►│   BALANCE    │─┐   │
              │  │  │ (Normal ops) │ │   │
              │  │  └──────┬───────┘ │   │
              │  │         │         │   │
              │  │    Sensor OK      │   │
              │  └─────────┘         │   │
              │                      │   │
              │         Fault/Watchdog│   │
              │                      ↓   │
              │              ┌──────────────┐
              │              │     HALT     │
              │              │ (Emergency)  │
              │              └──────┬───────┘
              │                     │
              │              Manual/Auto    │
              │                     ↓       │
              │              ┌──────────────┐
              └──────────────│   RECOVER    │
                             │ (Step back)  │
                             └──────┬───────┘
                                    │
                             Unrecoverable
                                    ↓
                             ┌──────────────┐
                             │    ERROR     │
                             │  (Latched)   │
                             └──────────────┘
```

### E.2.2 State Descriptions

| State | Function | Entry Condition | Exit Condition | Outputs |
|-------|----------|----------------|----------------|---------|
| **INIT** | System startup, self-test | Power-on reset | Self-test pass | All actuators disabled |
| **BALANCE** | Normal operation, maintain posture | INIT complete | Fault detected | Computed torques |
| **HALT** | Emergency stop | Watchdog timeout, fault | Manual/auto recovery | Zero torque |
| **RECOVER** | Return to stable pose | User command | Stable state reached | Safe motion commands |
| **ERROR** | Unrecoverable fault | Critical failure | System reset | All actuators locked |

### E.2.3 Transition Timing Constraints

All state transitions must complete within **bounded time**:

| Transition | Max Latency | Determinism Source |
|------------|-------------|-------------------|
| BALANCE → HALT | 1 ms | Single-well MVL ensures immediate response |
| HALT → RECOVER | 100 ms | Requires operator confirmation (not time-critical) |
| Any → ERROR | 10 ms | Fail-safe default on unrecognized condition |
| INIT → BALANCE | 500 ms | Self-test completion |

**Critical property**: BALANCE → HALT transition is **hardware-guaranteed deterministic** through MVL single-well biasing (Appendix A, Section A.3.2).

---

## E.3 Control Algorithm

### E.3.1 Balance Control Loop

**Executed every 1-5 ms** in BALANCE state:

```python
def balance_control_loop():
    """
    Deterministic reflex control for postural stability.
    Executes on SRP hardware in <1 ms.
    """
    # Read sensors (gyro, accelerometer, force sensors)
    sensor_data = read_sensors()  # <100 µs
    
    # Validate sensor data
    if not validate_sensors(sensor_data):
        transition_to_HALT()
        return
    
    # Compute center-of-mass (COM) position
    com_x, com_y = compute_com(sensor_data)  # <200 µs
    
    # PD controller for COM stabilization
    error_x = target_com_x - com_x
    error_y = target_com_y - com_y
    
    torque_x = K_P * error_x + K_D * d_error_x/dt
    torque_y = K_P * error_y + K_D * d_error_y/dt
    
    # Limit torques to safe range
    torque_x = clip(torque_x, -MAX_TORQUE, MAX_TORQUE)
    torque_y = clip(torque_y, -MAX_TORQUE, MAX_TORQUE)
    
    # Apply to actuators
    set_joint_torques(torque_x, torque_y)  # <100 µs
    
    # Update watchdog
    kick_watchdog()
```

**Timing breakdown**:
- Sensor read: 100 µs
- COM computation: 200 µs
- PD control: 50 µs
- Actuator command: 100 µs
- Overhead: 50 µs
- **Total**: 500 µs (well under 1 ms budget)

### E.3.2 Collision Avoidance

**Runs in parallel** with balance control:

```python
def collision_avoidance():
    """
    Check for imminent collisions and override actuator commands.
    """
    # Read proximity sensors
    distances = read_proximity_sensors()  # <50 µs
    
    # Check critical zones
    for zone in CRITICAL_ZONES:
        if distances[zone] < COLLISION_THRESHOLD:
            # Immediate halt
            transition_to_HALT()
            log_event("Collision avoidance triggered", zone)
            return
    
    # Check joint limits
    joint_positions = read_joint_encoders()
    for joint, pos in enumerate(joint_positions):
        if pos < JOINT_MIN[joint] or pos > JOINT_MAX[joint]:
            # Stop motion in that joint
            override_joint_torque(joint, 0)
            log_event("Joint limit exceeded", joint)
```

**Timing**: <100 µs (sensor read + comparison)

### E.3.3 Watchdog Supervision

**Independent timer** monitoring control loop health:

```python
class WatchdogTimer:
    def __init__(self, timeout_ms=10):
        self.timeout = timeout_ms
        self.last_kick = time.now()
    
    def kick(self):
        """Reset watchdog timer (called by control loop)."""
        self.last_kick = time.now()
    
    def check(self):
        """Check if watchdog has expired."""
        if (time.now() - self.last_kick) > self.timeout:
            # Watchdog timeout - force HALT
            self.trigger_fault()
            return EXPIRED
        return OK
    
    def trigger_fault(self):
        """Hardware failsafe: cut power to actuators."""
        set_actuator_enable(False)
        transition_to_HALT()
```

**Watchdog period**: 10 ms (2× worst-case control loop time for safety margin)

---

## E.4 Dual Redundancy and Lock-Step Execution

### E.4.1 Architecture

```
┌──────────────────────────────────────┐
│           SRP-C Chiplet              │
│                                      │
│   ┌────────────┐    ┌────────────┐  │
│   │  Core A    │    │  Core B    │  │
│   │  (Primary) │    │(Redundant) │  │
│   └─────┬──────┘    └──────┬─────┘  │
│         │                  │        │
│         └────────┬─────────┘        │
│                  │                  │
│           ┌──────▼───────┐          │
│           │   Comparator │          │
│           │  (Cycle-by-  │          │
│           │   cycle vote)│          │
│           └──────┬───────┘          │
│                  │                  │
│                  ▼                  │
│           [Actuator Output]        │
└──────────────────────────────────────┘
```

### E.4.2 Lock-Step Operation

Both cores execute **identical** instructions on **identical** data:

```
Clock cycle N:
  Core A: Read sensor → COM compute → Torque compute → Output
  Core B: Read sensor → COM compute → Torque compute → Output
  
  Comparator: if (output_A == output_B) then
                  valid = TRUE
              else
                  transition_to_ERROR()
```

**Skew tolerance**: ±100 ns (cores must produce identical results within this window)

### E.4.3 Fault Detection Coverage

| Fault Type | Detection Method | Coverage | Response Time |
|------------|------------------|----------|---------------|
| **Transient upset** | Output mismatch | >99% | <1 µs |
| **Stuck-at fault** | Self-test pattern | >95% | At INIT |
| **Timing violation** | Watchdog timeout | 100% | <10 ms |
| **Sensor failure** | Range check, redundancy | >90% | <1 ms |
| **Power glitch** | Brown-out detector | 100% | <10 µs |

**Total diagnostic coverage**: >90% (meets IEC 61508 SIL 3 requirement)

---

## E.5 Formal Verification Framework

### E.5.1 Safety Properties (Temporal Logic)

**Property 1: Bounded response time**
```
□ (fault_detected → ◇≤10ms HALT_state)
"Always, if a fault is detected, the system eventually reaches HALT within 10 ms"
```

**Property 2: No conflicting commands**
```
□ ¬(actuator_A_active ∧ actuator_A_disable)
"Always, actuator A cannot be commanded active and disabled simultaneously"
```

**Property 3: Fail-safe default**
```
□ (unrecognized_input → ◇≤1ms ERROR_state)
"Always, unrecognized inputs lead to ERROR state within 1 ms"
```

**Property 4: Watchdog guarantees**
```
□ (last_kick > 10ms_ago → actuators_disabled)
"Always, if watchdog not kicked for >10 ms, actuators are disabled"
```

### E.5.2 NuSMV Model (Simplified)

```smv
MODULE main
VAR
    state : {INIT, BALANCE, HALT, RECOVER, ERROR};
    sensor_valid : boolean;
    watchdog_ok : boolean;
    fault_detected : boolean;
    actuator_enabled : boolean;

ASSIGN
    init(state) := INIT;
    init(actuator_enabled) := FALSE;
    
    next(state) := case
        state = INIT & sensor_valid : BALANCE;
        state = BALANCE & watchdog_ok & !fault_detected : BALANCE;
        state = BALANCE & (!watchdog_ok | fault_detected) : HALT;
        state = HALT : {HALT, RECOVER};  -- Non-deterministic recovery
        state = RECOVER & sensor_valid : BALANCE;
        state = RECOVER & !sensor_valid : ERROR;
        state = ERROR : ERROR;  -- Latched
        TRUE : ERROR;  -- Default fail-safe
    esac;
    
    next(actuator_enabled) := case
        next(state) = BALANCE : TRUE;
        next(state) = HALT : FALSE;
        next(state) = RECOVER : TRUE;
        TRUE : FALSE;
    esac;

-- Safety properties
SPEC AG (fault_detected -> AF state = HALT)
SPEC AG (state = HALT -> !actuator_enabled)
SPEC AG (state = ERROR -> AX state = ERROR)  -- Latch
SPEC AG (!sensor_valid -> AX (state != BALANCE))
```

**Verification result** (expected):
- All properties PASS for binary abstraction
- Timing bounds proven via separate WCET analysis

### E.5.3 Coq Formalization (Illustrative)

```coq
(* State type *)
Inductive SRP_state : Type :=
  | INIT
  | BALANCE
  | HALT
  | RECOVER
  | ERROR.

(* Transition function *)
Definition srp_transition (s : SRP_state) (fault : bool) (watchdog : bool) : SRP_state :=
  match s with
  | INIT => BALANCE
  | BALANCE => if fault || negb watchdog then HALT else BALANCE
  | HALT => RECOVER
  | RECOVER => if fault then ERROR else BALANCE
  | ERROR => ERROR
  end.

(* Safety property: fault always leads to HALT *)
Lemma fault_leads_to_halt :
  forall (s : SRP_state),
    s = BALANCE ->
    srp_transition s true true = HALT.
Proof.
  intros. subst. simpl. reflexivity.
Qed.

(* ERROR state is absorbing *)
Lemma error_is_latched :
  forall (fault watchdog : bool),
    srp_transition ERROR fault watchdog = ERROR.
Proof.
  intros. simpl. reflexivity.
Qed.

(* Actuators disabled in HALT *)
Definition actuators_enabled (s : SRP_state) : bool :=
  match s with
  | BALANCE => true
  | RECOVER => true
  | _ => false
  end.

Lemma halt_disables_actuators :
  actuators_enabled HALT = false.
Proof.
  simpl. reflexivity.
Qed.
```

---

## E.6 Hardware Implementation Concepts

### E.6.1 MVL Logic Block

**SRP-C chiplet integrates 64-128 MVL cells** for state storage and transition logic.

**Example: 4-level MVL cell encoding states**:
```
State encoding (2 bits per state):
  INIT    = 00
  BALANCE = 01
  HALT    = 10
  RECOVER = 11
  ERROR   = (stuck-at-fault, not explicitly encoded)
```

**Transition logic** (conceptual):
```
Current state (MVL cell):   01 (BALANCE)
Input (fault bit):          1  (fault detected)
Bias voltage applied:       V_HALT
Next state (MVL cell):      10 (HALT)
Transition time:            <50 ns (write pulse width)
```

**Single-well biasing ensures** that given current state + input, only ONE next state is energetically favorable → **deterministic** transition.

### E.6.2 Clock and Timing

**Primary clock**: 1 MHz (1 µs period)
- Control loop: 1000 cycles = 1 ms
- Sensor read: 100 cycles = 100 µs
- State transition: 50 cycles = 50 µs

**Watchdog clock**: Independent 100 Hz (10 ms period)
- Runs on separate oscillator
- Cannot be stopped by software

**Clock domain crossing**: Dual-flop synchronizers for signals between domains

### E.6.3 Power Domain Isolation

```
┌─────────────────────────────────────┐
│  Main Power Domain (IP, AOP)        │
│  V_DD_main = 1.0 V                  │
│  Max current: 25 A                  │
└─────────────────────────────────────┘
        │ (Electrically isolated)
        │
┌───────▼─────────────────────────────┐
│  SRP Power Domain                   │
│  V_DD_srp = 1.1 V (independent LDO) │
│  Max current: 2 A                   │
│  Thermal isolation: >10 mm spacing  │
└─────────────────────────────────────┘
```

**Benefits**:
- SRP continues operation during IP/AOP power-down
- Voltage droop on main domain does not affect SRP
- Thermal runaway in IP does not impact SRP timing

---

## E.7 Sensor Interfacing

### E.7.1 Supported Interfaces

| Interface | Speed | Use Case | Latency |
|-----------|-------|----------|---------|
| **SPI** | 10 MHz | IMU (gyro, accel) | <10 µs |
| **CAN** | 1 Mbps | Joint encoders | <50 µs |
| **TSN** | 100 Mbps | High-speed sensors | <10 µs |
| **GPIO** | N/A | Limit switches | <1 µs |

### E.7.2 Sensor Validation

**Range check** (every sample):
```python
def validate_sensor(value, min, max):
    if value < min or value > max:
        log_fault("Sensor out of range")
        return INVALID
    return VALID

# Example: Gyroscope validation
gyro_x = read_gyro_x()
if validate_sensor(gyro_x, -500, +500) == INVALID:
    transition_to_HALT()
```

**Redundancy check** (dual sensors):
```python
def validate_redundant_sensors(sensor_A, sensor_B, tolerance=0.05):
    delta = abs(sensor_A - sensor_B) / sensor_A
    if delta > tolerance:
        log_fault("Redundant sensor mismatch")
        return INVALID
    return VALID
```

**Stuck-value detection** (temporal):
```python
class StuckDetector:
    def __init__(self, threshold=10):
        self.last_value = None
        self.stuck_count = 0
        self.threshold = threshold
    
    def check(self, current_value):
        if current_value == self.last_value:
            self.stuck_count += 1
            if self.stuck_count > self.threshold:
                return STUCK
        else:
            self.stuck_count = 0
        self.last_value = current_value
        return OK
```

---

## E.8 Actuator Safety Limits

### E.8.1 Joint Limits

**Software limits** (enforced in BALANCE state):
```python
JOINT_LIMITS = {
    'hip_pitch':   (-45, +90),   # degrees
    'hip_roll':    (-30, +30),
    'knee':        (-120, 0),
    'ankle_pitch': (-30, +30),
}

def enforce_joint_limits(joint_name, position):
    min_pos, max_pos = JOINT_LIMITS[joint_name]
    if position < min_pos:
        return min_pos
    if position > max_pos:
        return max_pos
    return position
```

**Hardware limits** (mechanical end-stops):
- Provide absolute guarantee even if software fails
- Trip limit switches → immediate transition to HALT

### E.8.2 Torque Limits

**Per-joint maximum torque**:
```python
MAX_TORQUE = {
    'hip_pitch':   50,  # Nm
    'hip_roll':    40,
    'knee':        60,
    'ankle_pitch': 30,
}

def limit_torque(joint_name, torque_command):
    max_t = MAX_TORQUE[joint_name]
    return clip(torque_command, -max_t, max_t)
```

**Rate limiter** (prevent jerky motions):
```python
def rate_limit(new_torque, last_torque, max_delta=10):
    """Limit torque change per control cycle."""
    delta = new_torque - last_torque
    if abs(delta) > max_delta:
        return last_torque + sign(delta) * max_delta
    return new_torque
```

---

## E.9 Testing and Validation

### E.9.1 Unit Tests

**State transition testing**:
```python
def test_balance_to_halt_on_fault():
    srp = SRP_FSM()
    srp.set_state(BALANCE)
    srp.inject_fault()
    assert srp.get_state() == HALT
    assert srp.actuators_enabled == False

def test_watchdog_timeout():
    srp = SRP_FSM()
    srp.set_state(BALANCE)
    time.sleep(0.015)  # Exceed 10 ms watchdog
    assert srp.get_state() == HALT
```

### E.9.2 Fault Injection

**Simulated faults** during testing:
- Sensor stuck-at-zero
- Sensor out-of-range
- Actuator command conflict
- Watchdog timeout
- Power glitch (voltage drop)

**Expected response**: All faults lead to HALT within 10 ms

### E.9.3 Long-Duration Testing

**Endurance test** (minimum requirements):
- Duration: 1000 hours continuous operation
- Fault injection rate: 1 per minute
- Expected HALT transitions: >60,000
- Allowable ERROR states: <10 (non-recoverable)
- **Pass criterion**: >99.99% fault detection coverage

---

## E.10 Safety Certification Alignment

### E.10.1 IEC 61508 SIL 3 Requirements

| Requirement | HG-FCCA SRP Implementation | Status |
|-------------|----------------------------|--------|
| **Hardware fault tolerance** | Dual lock-step cores | ✓ Designed |
| **Diagnostic coverage >90%** | >90% via comparison + self-test | ✓ Modeled |
| **Safe failure fraction** | >90% (faults lead to HALT) | ✓ Designed |
| **Systematic capability** | SC2 (structured development) | ✓ Planned |
| **Proof of correctness** | Formal verification (NuSMV, Coq) | ⧗ J3 deliverable |

### E.10.2 ISO 13849-1 Performance Level d Requirements

| Requirement | HG-FCCA SRP Implementation | Status |
|-------------|----------------------------|--------|
| **Category 3** | Single fault does not lead to loss of safety | ✓ Redundancy |
| **MTTF_d >100 years** | Estimated >150 years (solid-state) | ✓ Calculated |
| **DC_avg >90%** | >90% via dual-core voting | ✓ Designed |
| **CCF avoidance** | Separate power, thermal, clock domains | ✓ Designed |

---

## E.11 References for Appendix E

1. **IEC 61508** (2010). "Functional Safety of Electrical/Electronic/Programmable Electronic Safety-Related Systems."

2. **ISO 13849-1** (2015). "Safety of Machinery: Safety-Related Parts of Control Systems."

3. **ISO 10218-1** (2011). "Robots and Robotic Devices — Safety Requirements for Industrial Robots — Part 1: Robots."

4. **Clarke, E. M., et al.** (2018). "Model Checking." MIT Press.

5. **Bertot, Y. & Castéran, P.** (2004). "Interactive Theorem Proving and Program Development: Coq'Art." Springer.

---

**End of Appendix E**

*This appendix provides the safety control architecture, formal verification framework, and compliance roadmap for the SRP. Full safety case documentation and certification artifacts are developed during Deliverable J3.*
