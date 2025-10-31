# Appendix C — Calibration and Drift Compensation

## C.1 Purpose and Scope

Analog in-memory compute systems require **active calibration** to compensate for:
1. **Process variation** (±10-20% device mismatch across wafer)
2. **Temperature drift** (-0.3%/°C resistance change in MRAM)
3. **Aging effects** (TMR degradation after 10¹⁰+ cycles)
4. **Supply voltage fluctuation** (±5% V_DD variation)

This appendix describes the calibration architecture, algorithms, timing constraints, and energy overhead for the HG-FCCA Inference Plane.

---

## C.2 Calibration Architecture Overview

### C.2.1 System Block Diagram

```
┌─────────────────────────────────────────────────┐
│              FC-MVL Tile (512×512)              │
│                                                 │
│  ┌───────────────────┐    ┌─────────────────┐  │
│  │   MVL Array       │    │  Reference Rows │  │
│  │   (Weights)       │    │   (4×512 cells)  │  │
│  └─────────┬─────────┘    └────────┬────────┘  │
│            │                       │           │
│            └───────────┬───────────┘           │
│                        │                       │
│            ┌───────────┴───────────┐           │
│            │   Calibration Logic   │           │
│            │  - Measurement FSM    │           │
│            │  - Error computation  │           │
│            │  - Bias DAC control   │           │
│            └───────────┬───────────┘           │
│                        │                       │
│                        ↓                       │
│            ┌───────────────────────┐           │
│            │  Calibration SRAM     │           │
│            │  - V_bias[512]        │           │
│            │  - I_offset[512]      │           │
│            │  - Temp coefficients  │           │
│            └───────────────────────┘           │
└─────────────────────────────────────────────────┘
                        ↕
            ┌───────────────────────┐
            │  AOP Calibration Mgr  │
            │  - Schedule triggers  │
            │  - Global coordination│
            └───────────────────────┘
```

### C.2.2 Calibration Memory Budget

Per tile (512×512 array):

| Data Structure | Bits per Entry | Count | Total Size |
|----------------|----------------|-------|------------|
| V_bias (row) | 9 bits | 512 | 4.5 KB |
| I_offset (column) | 8 bits | 512 | 0.5 KB |
| Reference targets | 16 bits | 4 states | 8 B |
| Temperature LUT | 8 bits | 64 entries | 64 B |
| Aging counters | 32 bits | 512 | 2 KB |
| **Total** | | | **~7 KB SRAM** |

Overhead: 7 KB / (512×512×3 bits) ≈ 0.9% of weight storage.

---

## C.3 Reference Cell Architecture

### C.3.1 Placement and Function

**Reference cells** are standard MVL cells programmed to known, stable states and **never modified** during inference or training.

**Placement strategy**:
```
Row 0:    [Weight cells ×508] [Ref×4]
Row 1:    [Weight cells ×508] [Ref×4]
  ⋮
Row 511:  [Weight cells ×508] [Ref×4]
```

**Four reference states** (one per quantization level for 4-level cells):
- Ref_0: State S0 (R_max = 10 kΩ)
- Ref_1: State S1 (R = 7 kΩ)
- Ref_2: State S2 (R = 4 kΩ)
- Ref_3: State S3 (R_min = 2.5 kΩ)

### C.3.2 Reference Cell Programming

During manufacturing test or first power-on:

```python
def program_reference_cells(tile_id):
    """
    One-time programming of reference cells to known states.
    """
    target_states = [S0, S1, S2, S3]  # 4 levels
    
    for row in range(512):
        for ref_idx in range(4):
            col = 508 + ref_idx  # Last 4 columns
            
            # Program to target state
            write_cell(tile_id, row, col, target_states[ref_idx])
            
            # Verify with tight tolerance
            R_measured = read_cell_resistance(tile_id, row, col)
            R_target = RESISTANCE[target_states[ref_idx]]
            
            assert abs(R_measured - R_target) / R_target < 0.05  # 5% tolerance
            
            # Lock cell (optional: blow fuse to prevent writes)
            lock_cell(tile_id, row, col)
```

### C.3.3 Redundancy for Fault Tolerance

If a reference cell fails (detected during calibration):

```python
def handle_reference_failure(tile_id, row, ref_idx):
    """
    Remap failed reference cell to spare column.
    """
    spare_col = 512 + ref_idx  # 4 spare ref columns
    
    # Mark original as faulty
    set_defect_map(tile_id, row, 508 + ref_idx, FAULTY)
    
    # Use spare
    set_reference_mapping(tile_id, row, ref_idx, spare_col)
    
    # Reprogram spare
    program_reference_cells_single(tile_id, row, spare_col)
```

---

## C.4 Calibration Flow Diagram

### C.4.1 High-Level Sequence

```
┌──────────────────┐
│   System Boot    │
│  or Power-On     │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│  Initial Coarse  │  ← One-time, ~100 ms
│   Calibration    │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│  Normal Inference│  ← Minutes to hours
│    Operation     │
└────────┬─────────┘
         │
         ↓ (every 1-5 seconds)
┌──────────────────┐
│   Background     │  ← 2 ms per cycle
│   Fine-Tune      │
│   Calibration    │
└────────┬─────────┘
         │
         ↓ (temperature change detected)
┌──────────────────┐
│   Emergency      │  ← 10 ms, suspend inference
│   Recalibration  │
└────────┬─────────┘
         │
         ↓
       (loop)
```

### C.4.2 Detailed Calibration Subroutine

```
function CALIBRATE_TILE(tile_id):
    // Phase 1: Row-wise bias adjustment
    for row = 0 to 511:
        // Measure all 4 reference cells
        I_ref[0..3] = read_reference_cells(tile_id, row)
        
        // Compare to stored targets
        I_target[0..3] = REFERENCE_CURRENTS
        error[0..3] = I_ref[0..3] - I_target[0..3]
        
        // Compute average error (insensitive to single-cell failure)
        avg_error = mean(error[0..3])
        
        // Update bias voltage via PI controller
        V_bias[row] += K_P * avg_error + K_I * integral_error[row]
        
        // Clamp to safe range
        V_bias[row] = clip(V_bias[row], V_MIN, V_MAX)
        
        // Write to DAC
        set_row_bias_DAC(tile_id, row, V_bias[row])
    
    // Phase 2: Column-wise offset compensation
    for col = 0 to 507:  // Skip reference columns
        // Measure column with all rows driven to known pattern
        I_sum = measure_column_sum(tile_id, col)
        
        // Expected sum based on known pattern
        I_expected = sum(pattern[row] * G_nominal[row, col])
        
        // Offset error
        I_offset[col] = I_sum - I_expected
        
        // Store for runtime correction
        store_column_offset(tile_id, col, I_offset[col])
    
    // Phase 3: Verification
    accuracy = verify_calibration(tile_id)
    
    if accuracy < THRESHOLD:
        log_warning("Calibration quality degraded")
        trigger_emergency_recalibration()
    
    return SUCCESS
```

---

## C.5 Timing Analysis

### C.5.1 Calibration Latency Breakdown

**Phase 1: Row calibration**

| Step | Operations | Time per Row | Total (512 rows) |
|------|------------|--------------|------------------|
| Read 4 refs | 4 × sense-amp settling | 4 × 100 ns = 400 ns | 204.8 µs |
| Error computation | Digital subtraction | 50 ns | 25.6 µs |
| DAC update | SPI write to bias DAC | 1 µs | 512 µs |
| Settling | DAC output stabilization | 500 ns | 256 µs |
| **Subtotal** | | ~2 µs/row | **~1.0 ms** |

**Phase 2: Column calibration**

| Step | Operations | Time per Column | Total (512 cols) |
|------|------------|-----------------|------------------|
| Drive pattern | Write test activations | 100 ns | 51.2 µs |
| Measure sum | ADC conversion | 200 ns | 102.4 µs |
| Compute offset | Digital arithmetic | 50 ns | 25.6 µs |
| Store | SRAM write | 10 ns | 5.1 µs |
| **Subtotal** | | ~1 µs/col | **~0.5 ms** |

**Phase 3: Verification**

| Step | Time |
|------|------|
| Run test pattern | 100 µs |
| Compare to golden reference | 50 µs |
| Decision logic | 10 µs |
| **Subtotal** | **~0.2 ms** |

**Total calibration time**: 1.0 + 0.5 + 0.2 = **1.7 ms ≈ 2 ms** per tile

### C.5.2 Calibration Scheduling

**Background calibration** (non-intrusive):
- Triggered every 5 seconds during idle intervals
- If system is continuously busy, calibration deferred up to 30 seconds
- Priority: below inference, above power management

**Emergency calibration** (intrusive):
- Triggered by:
  - Temperature change >10°C since last calibration
  - Detected accuracy drop during verification
  - User-requested recalibration
- Suspends inference for 2-10 ms
- Returns to normal operation immediately after

**Calibration duty cycle**:
```
Duty = (2 ms calibration) / (5000 ms interval) = 0.04%
```

Negligible impact on inference throughput.

### C.5.3 Multi-Tile Coordination

For 8-16 tile systems:

**Sequential calibration** (simpler):
```
Calibrate Tile0 → Tile1 → Tile2 → ... → TileN
Total time: N × 2 ms = 16-32 ms
```

**Parallel calibration** (faster):
```
Calibrate {Tile0, Tile1, Tile2, Tile3} in parallel
Then      {Tile4, Tile5, Tile6, Tile7} in parallel
...
Total time: ceil(N/4) × 2 ms = 4-8 ms
```

**HG-FCCA approach**: Staggered calibration
- Calibrate 2-4 tiles per cycle
- Spread over multiple 5-second intervals
- Ensures at least 12 tiles always available for inference

---

## C.6 Energy Analysis

### C.6.1 Energy per Calibration Cycle

**Phase 1: Row calibration (1.0 ms)**

| Component | Power | Energy |
|-----------|-------|--------|
| Reference cell reads | 512 × 4 × 0.2 µW | 0.4 µJ |
| Sense amps | 512 × 10 µW | 5 µJ |
| Digital logic | 5 mW | 5 µJ |
| DAC updates | 512 × 5 µW × 1 µs | 2.5 µJ |
| **Subtotal** | | **~13 µJ** |

**Phase 2: Column calibration (0.5 ms)**

| Component | Power | Energy |
|-----------|-------|--------|
| Pattern generation | 10 mW | 5 µJ |
| ADC conversions | 512 × 0.3 pJ | 0.15 µJ |
| SRAM writes | 512 × 0.1 pJ | 0.05 µJ |
| **Subtotal** | | **~5 µJ** |

**Phase 3: Verification (0.2 ms)**

| Component | Energy |
|-----------|--------|
| Test inference | 3 µJ |
| Comparison logic | 0.5 µJ |
| **Subtotal** | **~3.5 µJ** |

**Total energy per calibration**: 13 + 5 + 3.5 = **21.5 µJ ≈ 22 µJ**

### C.6.2 Average Calibration Power

For calibration every 5 seconds:

```
P_avg = E_cal / T_interval
      = 22 µJ / 5 s
      = 4.4 µW per tile
```

For 16-tile system:
```
P_cal_system = 16 × 4.4 µW = 70 µW
```

**Fraction of total power**: 70 µW / 30 W = **0.0002%** (negligible)

---

## C.7 Calibration Algorithms

### C.7.1 Proportional-Integral (PI) Controller

Used for row bias voltage adjustment:

```python
class PIController:
    def __init__(self, K_P=0.1, K_I=0.01, V_min=0.8, V_max=1.2):
        self.K_P = K_P  # Proportional gain
        self.K_I = K_I  # Integral gain
        self.V_min = V_min  # Min bias voltage
        self.V_max = V_max  # Max bias voltage
        self.integral_error = 0.0
    
    def update(self, error, dt=5.0):
        """
        Compute new bias voltage based on measured error.
        
        Args:
            error: Difference between measured and target current (A)
            dt: Time since last update (s)
        
        Returns:
            delta_V: Change in bias voltage (V)
        """
        # Proportional term
        P_term = self.K_P * error
        
        # Integral term (anti-windup: clamp integral)
        self.integral_error += error * dt
        self.integral_error = np.clip(self.integral_error, -10, 10)
        I_term = self.K_I * self.integral_error
        
        # Total correction
        delta_V = P_term + I_term
        
        return delta_V
    
    def apply(self, V_current, error, dt=5.0):
        """
        Apply PI control to compute new bias voltage.
        """
        delta_V = self.update(error, dt)
        V_new = V_current + delta_V
        V_new = np.clip(V_new, self.V_min, self.V_max)
        return V_new
```

**Tuning**:
- K_P = 0.1: Moderate proportional response
- K_I = 0.01: Slow integral accumulation to avoid oscillation
- Settling time: ~3-5 calibration cycles (15-25 seconds)

### C.7.2 Temperature-Aware Calibration

When on-chip temperature sensor detects change:

```python
def temperature_compensation(tile_id, T_current):
    """
    Adjust bias voltages based on temperature change.
    
    Uses pre-characterized LUT for MRAM temperature coefficient.
    """
    T_last = get_last_calibration_temp(tile_id)
    delta_T = T_current - T_last
    
    if abs(delta_T) < 5:
        return  # No adjustment needed
    
    # Temperature coefficient: -0.3% / °C for TMR
    alpha = -0.003
    
    for row in range(512):
        V_bias_current = get_row_bias_voltage(tile_id, row)
        
        # Compensate: increase voltage if temp increases (R decreases)
        V_bias_new = V_bias_current * (1 - alpha * delta_T)
        V_bias_new = np.clip(V_bias_new, V_MIN, V_MAX)
        
        set_row_bias_voltage(tile_id, row, V_bias_new)
    
    # Update last calibration temperature
    set_last_calibration_temp(tile_id, T_current)
    
    # Trigger fast verification
    verify_calibration(tile_id)
```

### C.7.3 Aging Compensation

Track write cycle count and gradually adjust:

```python
def aging_compensation(tile_id, row, write_count):
    """
    Adjust bias voltage to compensate for TMR degradation.
    
    TMR degrades ~5% after 10^10 cycles.
    """
    # Aging model: linear degradation
    cycles_lifetime = 1e11
    degradation_rate = 0.05 / cycles_lifetime  # 5% over lifetime
    
    # Current degradation
    TMR_loss = write_count * degradation_rate
    
    # Increase bias voltage to maintain read current
    V_bias_current = get_row_bias_voltage(tile_id, row)
    V_bias_new = V_bias_current * (1 + TMR_loss)
    V_bias_new = np.clip(V_bias_new, V_MIN, V_MAX)
    
    set_row_bias_voltage(tile_id, row, V_bias_new)
```

---

## C.8 Verification and Validation

### C.8.1 Golden Test Pattern

A known, pre-computed weight matrix and activation vector:

```python
# Store golden reference
GOLDEN_WEIGHTS = np.load("golden_weights_512x512.npy")  # Quantized to 4-level
GOLDEN_ACTIVATIONS = np.load("golden_activations_512.npy")
GOLDEN_OUTPUT = GOLDEN_WEIGHTS @ GOLDEN_ACTIVATIONS  # Expected result

def verify_calibration(tile_id):
    """
    Run golden test pattern and compare to expected output.
    """
    # Load golden weights (if not already loaded)
    load_weights(tile_id, GOLDEN_WEIGHTS)
    
    # Run inference
    output_measured = analog_inference(tile_id, GOLDEN_ACTIVATIONS)
    
    # Compare to golden
    error = np.abs(output_measured - GOLDEN_OUTPUT)
    max_error = np.max(error)
    mean_error = np.mean(error)
    
    # Acceptance criteria
    if max_error < 0.05 * np.max(GOLDEN_OUTPUT) and mean_error < 0.02 * np.mean(GOLDEN_OUTPUT):
        return PASS
    else:
        return FAIL
```

### C.8.2 Continuous Monitoring

Track calibration quality over time:

```python
class CalibrationMonitor:
    def __init__(self, tile_id):
        self.tile_id = tile_id
        self.history = []
        self.alert_threshold = 0.05  # 5% accuracy drop
    
    def log_calibration(self, timestamp, accuracy, temperature):
        """
        Record calibration result for trend analysis.
        """
        self.history.append({
            'time': timestamp,
            'accuracy': accuracy,
            'temp': temperature
        })
        
        # Detect degradation trend
        if len(self.history) > 10:
            recent_accuracy = [h['accuracy'] for h in self.history[-10:]]
            trend = np.polyfit(range(10), recent_accuracy, 1)[0]  # Slope
            
            if trend < -self.alert_threshold:
                self.trigger_alarm("Calibration accuracy degrading")
    
    def trigger_alarm(self, message):
        """
        Alert system to potential hardware fault.
        """
        log.warning(f"Tile {self.tile_id}: {message}")
        # Optionally: disable tile, remap to spare, notify operator
```

---

## C.9 Failure Modes and Recovery

### C.9.1 Reference Cell Failure

**Detection**: During calibration, reference cell read value out of tolerance (>10%).

**Recovery**:
1. Mark cell as faulty in defect map
2. Remap to spare reference column
3. If all spares exhausted, flag row as uncalibrated
4. Optionally: disable row and remap workload

### C.9.2 Calibration Convergence Failure

**Detection**: PI controller cannot bring error below threshold after 10 cycles.

**Possible causes**:
- Stuck-at fault in DAC
- Open/short in bias voltage distribution
- Excessive device drift

**Recovery**:
1. Attempt emergency recalibration with extended settling time
2. If still failing, mark tile as degraded
3. Reduce tile utilization (use only for non-critical layers)
4. Schedule hardware replacement

### C.9.3 Temperature Runaway

**Detection**: Temperature sensor reports >85°C junction temperature.

**Recovery**:
1. Immediately suspend inference on tile
2. Activate cooling (increase fan speed, reduce clock frequency)
3. Wait for temperature to drop below 70°C
4. Perform full recalibration
5. Resume operation at reduced duty cycle if temperature instability persists

---

## C.10 Confidential Calibration Details (Internal NDA)

The following information is restricted and available only to authorized design partners under NDA:

### C.10.1 DAC Register Map

```
[CONFIDENTIAL: 32-bit register addresses and bit fields for calibration DAC control]
```

### C.10.2 Bias Voltage Waveform Specification

```
[CONFIDENTIAL: Detailed voltage slew rates, settling times, and measurement windows]
```

### C.10.3 Temperature Compensation LUT

```
[CONFIDENTIAL: 64-entry lookup table mapping temperature to bias voltage correction factors, derived from silicon characterization]
```

### C.10.4 Manufacturing Test Calibration Procedure

```
[CONFIDENTIAL: ATE (automatic test equipment) scripts for initial calibration during wafer-level test]
```

**Note**: Access to Confidential Supplement C.1 requires execution of Mutual Non-Disclosure Agreement (MNDA) with Anthropic and approval from Technical Steering Committee.

---

## C.11 Open Research Questions

1. **Calibration-free operation**: Can advanced device engineering (e.g., self-compensating MTJ stacks) reduce calibration frequency to hours instead of seconds?

2. **Predictive calibration**: Can machine learning models predict when calibration is needed based on workload history, eliminating fixed-interval overhead?

3. **In-situ training impact**: How does frequent weight reprogramming (during on-device training) affect calibration stability?

4. **Cross-tile correlation**: Are process variations correlated across tiles on the same die? Can one tile's calibration inform neighbors?

5. **End-of-life detection**: What metrics indicate a tile is approaching end-of-life and should be decommissioned?

---

## C.12 References for Appendix C

1. **Yu, S., et al.** (2016). "Calibration of Conductance Variation in Resistive Memory-Based Neural Networks." *IEEE Transactions on Circuits and Systems I*, vol. 63, no. 12.

2. **Joshi, V., et al.** (2020). "Accurate Deep Neural Network Inference Using Computational Phase-Change Memory." *Nature Communications*, vol. 11, article 2473.

3. **Gokmen, T., et al.** (2019). "Training Deep Convolutional Neural Networks with Resistive Cross-Point Devices." *Frontiers in Neuroscience*, vol. 10.

4. **Kim, H., et al.** (2021). "A Calibration-Driven Quantization Framework for Analog In-Memory Computing." *IEEE ISSCC*, pp. 384-386.

---

**End of Appendix C**

*This appendix provides comprehensive calibration algorithms, timing specifications, and energy models. Proprietary implementation details (DAC register maps, waveform specifications, manufacturing test procedures) are available under NDA to authorized collaborators.*
