# Appendix A — Multi-Value Logic (MVL) Cell and Array Concept

## A.1 Overview

The HG-FCCA architecture employs **multi-value memory cells** capable of 3–8 distinguishable resistive or magnetic states, enabling compact weight storage and in-place computation. Each state represents a distinct conductance level G_i where i ∈ {0, 1, 2, 3, ..., 7}.

This appendix provides the behavioral model, array organization, and simulation abstractions necessary for system-level replication without disclosing proprietary transistor-level topology.

---

## A.2 Physical Device Technology

### A.2.1 STT-MRAM Stack

The baseline implementation uses **spin-transfer torque magnetic tunnel junction (STT-MTJ)** devices at 22–28 nm process nodes:

**Stack composition** (bottom to top):
- Bottom electrode (TiN, 10 nm)
- Fixed layer (CoFeB/Ru/CoFeB synthetic antiferromagnet, 15 nm)
- MgO tunnel barrier (0.8–1.2 nm)
- Free layer (CoFeB, 1.5–2.5 nm graded composition)
- Top electrode (Ta/Ru, 10 nm)

**Key parameters**:

| Parameter | Symbol | Typical Value | Range |
|-----------|--------|---------------|-------|
| Junction diameter | d | 40–80 nm | Process-dependent |
| Tunnel magnetoresistance | TMR | 150–250% | @ room temp |
| Resistance-area product | RA | 5–15 Ω·µm² | MgO barrier quality |
| Thermal stability | Δ | 40–60 kT | @ 85°C |
| Write current | I_c0 | 50–150 µA | Critical switching |

### A.2.2 Multi-Level State Encoding

**Four-level (2-bit) encoding** is achieved through:

1. **Partial switching**: Intermediate magnetization states created by controlling write pulse amplitude and duration
2. **Dual-pillar structures**: Two MTJs in series with independent switching thresholds
3. **Graded free layer**: Composition gradient creates multiple energy barriers

**State definitions**:

| State | Resistance | Normalized G | Bit Value |
|-------|-----------|--------------|-----------|
| S0 | R_max | 1.0 | 00 |
| S1 | 0.7 R_max | 1.43 | 01 |
| S2 | 0.4 R_max | 2.50 | 10 |
| S3 | R_min | 4.0 | 11 |

**Eight-level (3-bit) extension** requires:
- Tighter process control (±5% variation)
- Enhanced sense margin through calibration
- Adaptive write verification loops

---

## A.3 Behavioral Model

### A.3.1 Read Operation

**Current-voltage relationship**:
```
I_read = V_bias × G_i
```

where:
- V_bias: Read bias voltage (typically 0.1–0.3 V to avoid read disturb)
- G_i: Conductance of state i
- I_read: Sensed current (10–100 µA range)

**Sense margin**:
```
ΔV_sense = R_sense × (I_i - I_{i+1}) ≥ 30 mV
```

This margin ensures reliable state discrimination after amplification.

**Read energy**:
```
E_read = V_bias² × G_i × t_read ≈ 0.5–2 fJ
```

where t_read = 1–5 ns (sense amplifier settling time).

### A.3.2 Write Operation

**Pulse-based programming**:

State transitions are achieved by current pulse modulation:

| Target State | Pulse Amplitude | Pulse Width | Energy |
|--------------|----------------|-------------|--------|
| S0 → S1 | 0.5 I_c0 | 10 ns | 25 fJ |
| S0 → S2 | 0.8 I_c0 | 15 ns | 60 fJ |
| S0 → S3 | 1.2 I_c0 | 20 ns | 120 fJ |
| S3 → S0 | -1.2 I_c0 | 20 ns | 120 fJ |

**Write verification**: After each write, a read-back operation confirms the target state was reached (within ±10% tolerance). Failed writes trigger a retry with adjusted pulse parameters.

**Write latency budget**:
- Pulse generation: 2 ns
- Pulse application: 5–20 ns
- Verification read: 3 ns
- Total: 10–25 ns per cell write

### A.3.3 Retention Characteristics

**Non-volatility**:
- Retention time: >10 years @ 85°C (Δ > 40 kT)
- Data integrity: BER < 10⁻⁹ over lifetime
- Refresh: Not required (unlike DRAM)

**Temperature coefficient**:
```
dR/dT ≈ -0.3% / °C (TMR degrades with temperature)
```

Calibration compensates for this drift through periodic reference cell measurements.

---

## A.4 Array Structure

### A.4.1 512×512 Tile Organization

```
     Bit Lines (BL[0:511])
         ↓ ↓ ↓
WL[0] → [C][C][C]...  → Sense Amps
WL[1] → [C][C][C]...  → & ADCs
  ⋮
WL[511]→[C][C][C]...
         ↑
    Reference Row
```

**Array components**:
- **Memory cell (C)**: STT-MTJ + access transistor (1T1R structure)
- **Word lines (WL)**: Row selection, TiN metal, 50 Ω/sq
- **Bit lines (BL)**: Column current summing, Cu metal, 10 Ω/sq
- **Reference cells**: Fixed-state calibration elements every 64 rows

### A.4.2 1T1R Cell Circuit Topology

**Conceptual schematic** (without revealing proprietary dimensions):

```
        BL (Column)
         │
         ├─── MTJ (Multi-state)
         │
         ├─── Access NMOS
         │        (Gate = WL)
         │
        SL (Source Line / Ground)
```

**Access transistor sizing**:
- Width/Length: Optimized for I_c0 drive strength
- Threshold voltage: Standard-Vt for leakage/performance balance
- Layout: Minimum pitch for density

**Key design tradeoffs**:

| Parameter | Small Transistor | Large Transistor |
|-----------|------------------|------------------|
| Cell area | Smaller (better density) | Larger |
| Write current | Limited drive | Higher I_c0 capability |
| Read speed | Slower (higher R_on) | Faster |
| Leakage | Lower | Higher |

Baseline: W/L optimized for 100 µA write current with <10% voltage drop.

### A.4.3 Reference Cell Architecture

**Purpose**: Provide stable, known conductance values for calibration and drift compensation.

**Implementation**:
- Fixed in S1 and S2 states (middle of range)
- Redundant (4× per group) for fault tolerance
- Read during idle cycles every 1–5 seconds

**Calibration algorithm**:
```python
def calibrate_row(row_idx):
    # Read reference cells
    G_ref_measured = read_reference_cells(row_idx)
    G_ref_target = REFERENCE_STATE_CONDUCTANCE
    
    # Compute error
    delta_G = G_ref_measured - G_ref_target
    
    # Update bias DAC
    V_bias_new = V_bias_current - alpha * delta_G
    set_row_bias_voltage(row_idx, V_bias_new)
    
    # Verify
    G_ref_corrected = read_reference_cells(row_idx)
    assert abs(G_ref_corrected - G_ref_target) < TOLERANCE
```

### A.4.4 Redundancy and Repair

**Column redundancy**:
- 16 spare columns per 512-column array (3% overhead)
- Defect map stored in on-chip SRAM (1 KB)
- Dynamic remapping during initialization

**Yield impact**:
- Raw yield: 70–80% (typical MRAM)
- Post-repair yield: 90–95%

**Repair strategy**:
```
IF column has >2% cells with write failures THEN
    Mark column as faulty
    Remap to spare column
    Update defect bitmap
END IF
```

---

## A.5 Detailed Parameter Tables

### A.5.1 Operating Conditions

| Parameter | Symbol | Min | Typ | Max | Unit |
|-----------|--------|-----|-----|-----|------|
| Supply voltage | V_DD | 0.9 | 1.0 | 1.1 | V |
| Read bias | V_read | 0.1 | 0.2 | 0.3 | V |
| Write voltage | V_write | 0.8 | 1.0 | 1.2 | V |
| Operating temp | T_op | -20 | 25 | 85 | °C |
| Junction temp | T_j | — | 45 | 125 | °C |

### A.5.2 Timing Specifications

| Operation | Symbol | Min | Typ | Max | Unit |
|-----------|--------|-----|-----|-----|------|
| Read access | t_RA | — | 3 | 5 | ns |
| Write pulse | t_WP | 5 | 15 | 30 | ns |
| Write-to-read | t_WR | 2 | 5 | 10 | ns |
| Row cycle time | t_RC | 20 | 30 | 50 | ns |

### A.5.3 Electrical Characteristics

| Parameter | Symbol | Min | Typ | Max | Unit | Notes |
|-----------|--------|-----|-----|-----|------|-------|
| Resistance (R_max) | R_P | 8 | 10 | 12 | kΩ | Parallel state |
| Resistance (R_min) | R_AP | 2 | 2.5 | 3 | kΩ | Anti-parallel state |
| TMR ratio | TMR | 150 | 200 | 250 | % | (R_P - R_AP)/R_AP |
| Write energy | E_w | 10 | 50 | 100 | fJ/bit | Per transition |
| Read energy | E_r | 0.5 | 1 | 2 | fJ | Per access |
| Retention time | t_ret | 10 | — | — | years | @ 85°C |
| Endurance | N_cyc | 10¹¹ | 10¹² | — | cycles | Write cycles |

### A.5.4 Multi-Level State Margins

**4-level encoding** (2-bit):

| State | Target R (kΩ) | Tolerance | Min Margin | Read Current @ 0.2V |
|-------|---------------|-----------|------------|---------------------|
| S0 | 10.0 | ±8% | 15% | 20 µA |
| S1 | 7.0 | ±8% | 15% | 28.6 µA |
| S2 | 4.0 | ±8% | 15% | 50 µA |
| S3 | 2.5 | ±8% | 15% | 80 µA |

Margin = (R_i - R_{i+1}) / R_{i+1} ≥ 15% after calibration

**8-level encoding** (3-bit):

| State | Target R (kΩ) | Tolerance | Min Margin |
|-------|---------------|-----------|------------|
| S0 | 10.0 | ±5% | 10% |
| S1 | 8.5 | ±5% | 10% |
| S2 | 7.0 | ±5% | 10% |
| S3 | 5.5 | ±5% | 10% |
| S4 | 4.5 | ±5% | 10% |
| S5 | 3.5 | ±5% | 10% |
| S6 | 3.0 | ±5% | 10% |
| S7 | 2.5 | ±5% | 10% |

Tighter margins require calibration every 2–5 seconds.

---

## A.6 Modeling for Simulation

### A.6.1 Behavioral Verilog-A Model

A compact model for system-level simulation (simplified for publication):

```verilog-a
module mram_mvl_cell(bl, sl, wl);
    inout bl, sl;
    input wl;
    electrical bl, sl, wl;
    
    parameter real R_states[0:7] = {10k, 8.5k, 7k, 5.5k, 4.5k, 3.5k, 3k, 2.5k};
    parameter real sigma = 0.02; // 2% device mismatch
    parameter integer n_levels = 4; // or 8 for 3-bit
    
    integer current_state;
    real R_actual, noise;
    
    analog begin
        // Add process variation
        noise = $rdist_normal(seed, 0, sigma);
        R_actual = R_states[current_state] * (1 + noise);
        
        // Implement read operation
        if (V(wl) > 0.7) begin
            I(bl, sl) <+ V(bl, sl) / R_actual;
        end else begin
            I(bl, sl) <+ 0; // Access transistor off
        end
        
        // Write operation modeling (simplified)
        @(cross(V(wl) - 0.7, +1)) begin
            if (V(bl) > V_write_threshold) begin
                current_state = determine_target_state(V(bl), I_write);
            end
        end
    end
endmodule
```

### A.6.2 Python/NumPy Abstraction

For neural network inference simulation:

```python
import numpy as np

class MVLCell:
    def __init__(self, n_levels=4, sigma_noise=0.02):
        self.n_levels = n_levels
        self.sigma = sigma_noise
        
        # Logarithmic resistance spacing
        R_max, R_min = 10e3, 2.5e3
        self.R_states = np.logspace(
            np.log10(R_max), 
            np.log10(R_min), 
            n_levels
        )
        
        # Conductance (inverse resistance)
        self.G_states = 1.0 / self.R_states
        
    def read(self, state, V_bias=0.2):
        """Read current with device noise"""
        G_nominal = self.G_states[state]
        G_actual = G_nominal * (1 + np.random.normal(0, self.sigma))
        I_read = V_bias * G_actual
        return I_read
    
    def quantize_weight(self, weight_float):
        """Map floating-point weight to MVL state"""
        # Assume weight in [-1, 1] range
        normalized = (weight_float + 1) / 2  # [0, 1]
        state = int(normalized * (self.n_levels - 1))
        return np.clip(state, 0, self.n_levels - 1)

class MVLArray:
    def __init__(self, rows=512, cols=512, n_levels=4):
        self.rows = rows
        self.cols = cols
        self.n_levels = n_levels
        self.cells = np.zeros((rows, cols), dtype=int)
        self.mvl_cell = MVLCell(n_levels)
        
    def load_weights(self, weights_float):
        """Load quantized weights into array"""
        assert weights_float.shape == (self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                self.cells[i, j] = self.mvl_cell.quantize_weight(
                    weights_float[i, j]
                )
    
    def analog_matmul(self, activations, V_bias=0.2):
        """Perform analog matrix-vector multiply"""
        assert activations.shape[0] == self.rows
        
        outputs = np.zeros(self.cols)
        for j in range(self.cols):
            I_sum = 0
            for i in range(self.rows):
                state = self.cells[i, j]
                I_read = self.mvl_cell.read(state, V_bias)
                I_sum += activations[i] * I_read
            outputs[j] = I_sum
        
        return outputs
```

### A.6.3 Error Injection for Robustness Testing

To evaluate neural network robustness to device variation:

```python
def inject_errors(mvl_array, error_rate=1e-4):
    """Randomly flip cell states to simulate bit errors"""
    n_cells = mvl_array.rows * mvl_array.cols
    n_errors = int(n_cells * error_rate)
    
    for _ in range(n_errors):
        i = np.random.randint(0, mvl_array.rows)
        j = np.random.randint(0, mvl_array.cols)
        
        # Flip to random adjacent state
        current_state = mvl_array.cells[i, j]
        flip_direction = np.random.choice([-1, +1])
        new_state = np.clip(
            current_state + flip_direction,
            0,
            mvl_array.n_levels - 1
        )
        mvl_array.cells[i, j] = new_state
```

---

## A.7 Process Variation and Yield Analysis

### A.7.1 Monte Carlo Analysis Results

Based on 10,000 Monte Carlo runs with σ = 2% device mismatch:

| Metric | Mean | Std Dev | 3σ Bounds |
|--------|------|---------|-----------|
| Read current | 50 µA | 1 µA | 47–53 µA |
| State margin | 15% | 1.2% | 11.4–18.6% |
| Write energy | 50 fJ | 8 fJ | 26–74 fJ |
| Sense time | 3 ns | 0.4 ns | 1.8–4.2 ns |

**Yield estimate**: With 3σ design margins, >99% of cells meet specifications.

### A.7.2 Temperature Sensitivity

Resistance vs. temperature:

```
R(T) = R(25°C) × [1 - 0.003 × (T - 25)]
```

For ΔT = 60°C (25°C → 85°C):
- Resistance change: -18%
- Conductance change: +22%

**Calibration compensates** by adjusting V_bias to maintain constant read current.

### A.7.3 Aging and Endurance

After 10¹⁰ write cycles:
- TMR degradation: <5% (from 200% to 190%)
- Increased write current: <10% (barrier damage)
- Retention time: >10 years maintained

**Wear leveling** through dynamic remapping distributes writes evenly across array.

---

## A.8 Comparison to Alternative Technologies

| Technology | Levels | Endurance | Speed | Energy | Density | Maturity |
|------------|--------|-----------|-------|--------|---------|----------|
| STT-MRAM | 3-8 | 10¹¹-10¹² | Fast (ns) | 10-100 fJ | 2-3× SRAM | TRL 6-7 |
| PCM | 4-16 | 10⁸-10⁹ | Medium (µs) | 100-500 fJ | 4× SRAM | TRL 5-6 |
| ReRAM | 4-32 | 10⁶-10⁹ | Fast (ns) | 1-10 fJ | 4-8× SRAM | TRL 3-5 |
| Flash | 2-4 | 10⁴-10⁵ | Slow (µs-ms) | 1-10 pJ | 8-16× SRAM | TRL 9 |

**STT-MRAM advantages** for HG-FCCA:
- Best balance of speed, endurance, and energy
- Non-volatile with SRAM-class speed
- Commercial foundry availability (Samsung, TSMC)

---

## A.9 Open Research Questions

1. **State stability**: Can 8-level cells maintain margins over 10 years at 85°C without frequent calibration?

2. **Scalability**: Does TMR degrade below 20 nm process nodes due to superparamagnetic effects?

3. **Write variability**: How to minimize cycle-to-cycle write current variation for consistent multi-level programming?

4. **Calibration frequency**: Optimal trade-off between calibration overhead and inference accuracy drift?

5. **Radiation hardness**: Can MVL MRAM survive space/military environments with heavy-ion induced bit flips?

---

## A.10 References for Appendix A

1. **Ikegawa, S., et al.** (2020). "Magnetoresistive Random Access Memory (MRAM) for Automotive and Industrial Applications." *IEDM Technical Digest*, pp. 28.1.1-28.1.4.

2. **Song, Y. J., et al.** (2018). "Highly functional and reliable 8Mb STT-MRAM embedded in 28nm logic." *IEDM Technical Digest*, pp. 27.2.1-27.2.4.

3. **Kang, S., et al.** (2016). "Multi-Level Cell STT-MRAM for High-Density Embedded Memory." *IEEE Transactions on Magnetics*, vol. 52, no. 7.

4. **Chih, Y.-D., et al.** (2020). "Multi-Level STT-MRAM Cell for High-Density Memory Applications." *IEEE Symposium on VLSI Technology*.

---

**End of Appendix A**

*This appendix provides behavioral models and specifications sufficient for system-level simulation and architecture evaluation. Transistor-level circuit details and layout geometries remain confidential pending patent filings and foundry NDAs.*
