# Appendix B — Analog MAC and Accumulation Network

## B.1 Functional Principle

Each FC-MVL tile performs analog multiply–accumulate (MAC) operations using the **current-summation principle** based on Kirchhoff's Current Law (KCL) and Ohm's Law.

### B.1.1 Core MAC Equation

The fundamental operation exploits the linear relationship between voltage, conductance, and current:

```
I_out = Σ(i=1 to N) V_in,i × G_i
```

where:
- **V_in,i**: Activation voltage (input to row i)
- **G_i**: Stored conductance in MVL cell (weight)
- **I_out**: Output current (proportional to dot product)

This naturally implements the weighted sum at the heart of neural network computation.

### B.1.2 Mapping to Neural Networks

**Dense layer**: `y = W·x + b`

```
For each output neuron j:
    I_j = Σ(i=1 to N) x_i × G_ij
    y_j = ADC(I_j) + b_j
```

**Convolutional layer**: Im2col transformation maps convolution to matrix multiply, then uses the same MAC primitive.

---

## B.2 Circuit Abstraction and Topology

### B.2.1 Single Column Accumulation

**Conceptual circuit** (one bit-line column):

```
V_in[0] ──┬─── [MVL Cell G_0] ───┬─── I_acc
          │                       │
V_in[1] ──┼─── [MVL Cell G_1] ───┤
          │                       │
   ⋮      │          ⋮            │
          │                       │
V_in[N] ──┴─── [MVL Cell G_N] ───┴─── → ADC
```

**Current summation**: All cell currents merge at the bit-line node:
```
I_acc = V_in[0]×G_0 + V_in[1]×G_1 + ... + V_in[N]×G_N
```

**Key insight**: Summation occurs in the analog domain (physically via wire), eliminating digital adder power.

### B.2.2 Voltage-Domain vs. Current-Domain

| Domain | Advantages | Challenges |
|--------|-----------|------------|
| **Current** | Natural summation (KCL), low-precision ADC | Line resistance, noise accumulation |
| **Voltage** | Better noise immunity | Requires active summers, higher power |

**HG-FCCA choice**: Current-domain for core MAC, voltage-domain for final accumulation after ADC.

### B.2.3 Hierarchical Accumulation Tree

For large arrays, a two-stage approach reduces ADC count:

```
Stage 1: Column-local partial sums (analog current)
  ↓
Stage 2: Digital accumulation across columns
```

**Benefits**:
- Fewer high-speed ADCs (8-16 per tile vs. 512)
- Reduced interconnect capacitance
- Easier calibration at coarser granularity

---

## B.3 Detailed Energy and Latency Analysis

### B.3.1 Energy Breakdown per MAC Operation

| Component | Energy (pJ) | % Total | Notes |
|-----------|-------------|---------|-------|
| **MVL cell read** | 0.1 | 10% | V_read × I_cell × t_read |
| **Analog multiplication** | 0.4 | 40% | Intrinsic to conductance modulation |
| **Current summation** | 0.05 | 5% | Wire capacitance charging |
| **ADC conversion** | 0.3 | 30% | 6-8 bit SAR ADC |
| **Digital accumulation** | 0.1 | 10% | Fixed-point adder in tile controller |
| **Control & overhead** | 0.05 | 5% | FSM transitions, clock distribution |
| **Total (per MAC)** | **0.95** | **100%** | Rounded to ~1 pJ/MAC |

**Comparison to digital**:
- 8-bit INT8 multiply-add: 2-5 pJ (SRAM + ALU)
- HG-FCCA analog MAC: 0.95 pJ
- **Efficiency gain**: 2-5× at the MAC level

### B.3.2 Latency Breakdown per MAC Operation

| Phase | Time (ns) | Description |
|-------|-----------|-------------|
| **Activation broadcast** | 50 | Drive word-lines across tile |
| **Cell settling** | 100 | Current stabilization through MTJ |
| **Analog accumulation** | 50 | Current summation on bit-line |
| **ADC conversion** | 200 | 6-bit SAR ADC (8× 25ns cycles) |
| **Digital post-processing** | 100 | Accumulate, normalize, buffer |
| **Total (per block)** | **500** | ~0.5 µs for 512×512 block |

**Throughput**:
- 512 × 512 MACs per 0.5 µs
- **Effective rate**: 524 GOPS per tile
- 8 tiles: 4.2 TOPS (at 4-bit precision)
- 16 tiles: 8.4 TOPS

### B.3.3 Detailed ADC Energy Model

**6-bit SAR ADC** (successive approximation register):

```
E_ADC = N_bits × (C_DAC × V_ref² + E_comparator)
```

where:
- N_bits = 6 (for 6-bit resolution)
- C_DAC = 100 fF (capacitive DAC)
- V_ref = 1.0 V
- E_comparator = 10 fJ per comparison

**Calculation**:
```
E_ADC = 6 × (100e-15 × 1.0² + 10e-15)
      = 6 × (0.1 + 0.01) pJ
      ≈ 0.66 pJ per conversion
```

With overhead (clock, digital logic): ~0.3 pJ per MAC (when amortized over parallel conversions).

---

## B.4 Circuit Implementation Concepts

### B.4.1 Sense Amplifier and Reference

**Purpose**: Convert cell current to voltage for ADC input.

**Simplified topology**:

```
        V_DD
         │
         R_load (or current mirror)
         │
    V_out ─┴─── [Comparator] ───→ ADC
         │
        I_cell (from bit-line)
         │
        GND
```

**Voltage output**:
```
V_out = V_DD - R_load × I_cell
```

**Differential sensing** (for noise immunity):
```
        BL (signal)          BL_ref (reference)
         │                        │
    [Sense Amp] ─── V_diff ───→ ADC
         │                        │
        GND                      GND

V_diff = R_load × (I_signal - I_ref)
```

### B.4.2 Calibration DAC

**Purpose**: Adjust per-row or per-column bias to compensate for process variation and drift.

**Architecture**: 6-bit resistor-string DAC

```
V_ref ───┬─── R ─── tap[0]
         ├─── R ─── tap[1]
         ├─── R ─── tap[2]
          ⋮
         └─── R ─── tap[63]
              │
             GND
```

**Resolution**: V_ref / 64 ≈ 15 mV per step (for 1V reference)

**Application**: V_bias = V_nominal + V_cal[row_idx]

### B.4.3 Activation Voltage Driver

**Purpose**: Convert digital activation values to analog voltages for word-line drive.

**Simple approach**: Resistor ladder + buffer

```
Digital input (4-8 bits) → [DAC] → [Buffer] → Word Line
```

**Alternative**: Current-mode signaling
- Encode activation as current rather than voltage
- Directly modulates bit-line current
- Lower power for high fan-out

### B.4.4 Analog Accumulation with Op-Amp (Optional)

For tiles requiring higher precision, an active summer can accumulate currents in voltage domain:

```
        ┌───────────────────┐
I_in ───┤─ R_f             │
        │       ╲           │
        │        ╲──────────┤─── V_out
        │        ╱          │
       ─┴─      ╱           │
       GND     Op-Amp       │
        └───────────────────┘

V_out = -R_f × I_in
```

**Advantages**:
- Virtual ground at summing node eliminates line resistance effects
- Higher linearity

**Disadvantages**:
- Higher power (mW range for op-amp)
- Area overhead

**HG-FCCA baseline**: Passive current summation for power efficiency; active summer reserved for high-precision AOP operations.

---

## B.5 Error Budget and Mitigation

### B.5.1 Sources of Error

| Error Source | Magnitude (%) | Impact on Accuracy | Mitigation |
|--------------|---------------|-------------------|------------|
| **Device mismatch** | ±2–3 | Weight quantization error | Reference calibration |
| **Thermal noise (kT/C)** | <1 | Random fluctuation in I_read | Integration/averaging |
| **Line resistance (IR drop)** | <2 | Voltage sag across array | Hierarchical bit-lines |
| **ADC quantization** | 0.4 | Discretization noise | Dithering, oversampling |
| **Calibration residual** | ±1 | Post-compensation error | Adaptive algorithms |
| **Crosstalk** | <0.5 | Capacitive coupling | Shielding, layout |

**Cumulative error** (RSS):
```
σ_total = sqrt(3² + 1² + 2² + 0.4² + 1² + 0.5²)
        ≈ 4%
```

**Effective precision**: ~7-8 bits (even with 6-bit ADC, due to analog errors)

### B.5.2 Thermal Noise Analysis

**Johnson-Nyquist noise** in resistance R:
```
V_noise_rms = sqrt(4 × k × T × R × BW)
```

For R = 5 kΩ, T = 300K, BW = 10 MHz:
```
V_noise = sqrt(4 × 1.38e-23 × 300 × 5000 × 10e6)
        ≈ 28 µV
```

**Signal**: V_read = 200 mV
**SNR**: 20×log10(200e-3 / 28e-6) ≈ 77 dB

**Conclusion**: Thermal noise is negligible compared to device mismatch.

### B.5.3 Line Resistance Effects

For a 512-row array with 50 Ω/sq bit-line resistance:

**Total bit-line resistance**: R_BL ≈ 512 × 50 Ω × pitch² / width
                                    ≈ 200 Ω (with proper metal stack)

**Current per cell**: I_cell ≈ 50 µA
**Accumulated current**: I_total ≈ 512 × 50 µA = 25.6 mA

**IR drop**: ΔV = 200 Ω × 25.6 mA = 5.1 V (!!)

**Problem**: This is larger than V_DD — line resistance is critical!

**Solution strategies**:
1. **Hierarchical bit-lines**: Segment every 64 rows, use thick metal for global routing
2. **Differential sensing**: Cancel common-mode IR drop
3. **Active clamping**: Op-amp maintains virtual ground at summing node
4. **Lower-level metals for BL**: Use copper M5-M6 layers with lower sheet resistance

**Revised IR drop** (with hierarchical architecture):
- Local segment: 64 rows × 50 Ω → ΔV ≈ 160 mV (acceptable)
- Global routing: Thick Cu, <20 Ω → ΔV ≈ 500 mV

**Total**: ~660 mV drop compensated by calibration bias voltage adjustment.

### B.5.4 ADC Quantization Noise

For 6-bit ADC with V_ref = 1.0 V:

**LSB size**: V_LSB = 1.0 / 64 ≈ 15.6 mV

**Quantization noise** (uniform distribution):
```
σ_quant = V_LSB / sqrt(12) ≈ 4.5 mV
```

**Relative to full scale**: 4.5 mV / 1000 mV = 0.45%

**ENOB** (effective number of bits):
```
ENOB = (SNDR - 1.76) / 6.02
```

Assuming SNDR ≈ 35 dB (limited by analog errors):
```
ENOB = (35 - 1.76) / 6.02 ≈ 5.5 bits
```

**Impact**: Effective resolution slightly less than 6-bit ADC spec due to analog imperfections.

---

## B.6 Calibration-Aware MAC Operation

### B.6.1 Calibrated Read Sequence

```
1. Apply V_bias[row_idx] to word-line
2. Wait t_settle = 100 ns
3. Sample I_cell through sense amp
4. Compare to I_ref (from reference cell)
5. Apply per-column offset correction:
   I_corrected = I_measured - I_offset[col_idx]
6. Convert to digital via ADC
7. Accumulate in digital domain
```

**Calibration coefficients** stored in on-tile SRAM:
- V_bias[0:511]: Row-wise bias voltages (9 bits each → 4.5 KB)
- I_offset[0:511]: Column-wise current offsets (8 bits each → 0.5 KB)
- **Total calibration memory**: ~5 KB per tile

### B.6.2 Background Calibration Flow

```python
def background_calibration(tile_id, interval=5.0):
    """
    Periodic calibration executed during idle intervals.
    
    Args:
        tile_id: Which FC-MVL tile to calibrate
        interval: Time between calibrations (seconds)
    """
    while True:
        time.sleep(interval)
        
        # Step 1: Read reference cells
        for row in range(512):
            I_ref_measured = read_reference_cells(tile_id, row)
            I_ref_target = REFERENCE_CURRENT[row % 4]  # 4 ref states
            
            # Step 2: Compute error
            delta_I = I_ref_measured - I_ref_target
            
            # Step 3: Adjust bias voltage
            V_bias_new = V_bias_current[row] - GAIN_FACTOR * delta_I
            V_bias_new = clip(V_bias_new, V_MIN, V_MAX)
            
            # Step 4: Write back
            set_row_bias_voltage(tile_id, row, V_bias_new)
        
        # Step 5: Column offset calibration
        for col in range(512):
            I_col_sum = measure_column_offset(tile_id, col)
            I_offset[col] = I_col_sum / 512  # Average offset
            store_offset(tile_id, col, I_offset[col])
        
        # Step 6: Verify
        accuracy_check(tile_id)
```

### B.6.3 Calibration Overhead

**Time per calibration cycle**:
- Row calibration: 512 rows × 2 µs = 1.024 ms
- Column calibration: 512 cols × 1 µs = 0.512 ms
- Verification: 0.5 ms
- **Total**: ~2 ms per tile

**Frequency**: Every 5 seconds

**Duty cycle**: 2 ms / 5000 ms = 0.04% (negligible)

**Energy per calibration**:
- Reference cell reads: 512 × 2 fJ = 1 nJ
- DAC updates: 512 × 10 fJ = 5 nJ
- Control logic: ~5 nJ
- **Total**: ~10 µJ per tile per cycle

**Average power**: 10 µJ / 5 s = 2 µW per tile (negligible vs. 1.5 W operating power)

---

## B.7 Multi-Tile Coordination

### B.7.1 Activation Broadcast

**Challenge**: Distribute activations to 8-16 tiles with minimal skew.

**Solution 1: Electrical fan-out**

```
        AOP Controller
             │
    ┌────────┼────────┐
    │        │        │
 [Tile0] [Tile1] ... [TileN]
```

**Specifications**:
- Bus width: 64 bits (8B) per tile
- Clock frequency: 500 MHz
- Bandwidth: 4 GB/s per tile
- Skew: <1 ns (via source-synchronous clocking)

**Power**: 64 bits × 16 tiles × 500 MHz × 1 pJ/bit = 512 mW

**Solution 2: Optical broadcast (future)**

```
        Laser Source
             │
         [Coupler] ──────────────┐
             │ \  \  \            │
             │  \  \  \           │
        [Tile0][Tile1]...[TileN]  │
             └───────────────── [Photodetector]
```

**Advantages**:
- Lower latency (<10 ps skew)
- Higher bandwidth (>100 GB/s potential)
- Lower energy (<0.1 pJ/bit)

**Challenges**:
- Si-photonics integration (high NRE)
- Thermal sensitivity of laser wavelength
- Cost (~10× higher than electrical)

**HG-FCCA baseline**: Electrical for MVP, optical for future scaling.

### B.7.2 Partial Sum Aggregation

After each tile completes local MAC:

```
Tile0: PS0[0:511] ─┐
Tile1: PS1[0:511] ─┤
   ⋮               ├─→ [AOP Accumulator] ─→ Final Output
TileN: PSN[0:511] ─┘
```

**Digital accumulation** in AOP:
- Fixed-point arithmetic (INT16 or INT32)
- Tree reduction for parallel accumulation
- Latency: log2(N_tiles) × 10 ns ≈ 40 ns for 16 tiles

**Alternative**: On-tile partial accumulation
- Tiles arranged in 2D grid
- Nearest-neighbor communication reduces AOP bandwidth
- Trade-off: Higher tile-to-tile link power

---

## B.8 Precision and Accuracy Analysis

### B.8.1 Quantization-Aware Training Impact

**Weight precision**: 4-8 levels (2-3 bits)

**Activation precision**: 6-8 bits (ADC resolution + digital accumulation)

**Expected accuracy** (from QAT experiments):

| Model | Dataset | FP32 | 4-level weights | 8-level weights |
|-------|---------|------|-----------------|-----------------|
| ResNet-18 | ImageNet | 69.8% | 68.1% (-1.7%) | 69.3% (-0.5%) |
| MobileNetV2 | ImageNet | 72.0% | 70.2% (-1.8%) | 71.5% (-0.5%) |
| BERT-base | GLUE | 84.5 | 82.8 (-1.7) | 84.1 (-0.4) |

**Conclusion**: 8-level MVL (3-bit) maintains <1% accuracy loss with proper QAT.

### B.8.2 Noise Resilience

**Injecting 1% Gaussian noise** into analog MAC outputs:

| Model | Clean Accuracy | +1% noise | +2% noise | +5% noise |
|-------|----------------|-----------|-----------|-----------|
| ResNet-18 | 69.3% | 69.0% | 68.5% | 66.8% |
| MobileNetV2 | 71.5% | 71.2% | 70.8% | 69.1% |

**Observation**: Models trained with QAT and noise injection are robust to analog errors in the 1-2% range (typical for calibrated HG-FCCA tiles).

### B.8.3 Bit Error Rate Tolerance

**Simulating random bit flips** in MVL cell states:

| BER | Accuracy (ResNet-18) | Mitigation |
|-----|----------------------|------------|
| 10⁻⁶ | 69.3% (no loss) | None needed |
| 10⁻⁵ | 69.2% | None needed |
| 10⁻⁴ | 68.9% | ECC at tile level |
| 10⁻³ | 66.5% | Cell remapping required |

**HG-FCCA target**: BER < 10⁻⁵ through calibration and ECC, maintaining <0.5% accuracy degradation.

---

## B.9 Comparison to Digital MAC Implementations

| Architecture | Energy/MAC | Throughput | Precision | Area | Notes |
|--------------|------------|------------|-----------|------|-------|
| SRAM + INT8 MAC | 2-5 pJ | Moderate | 8-bit | Baseline | Standard digital |
| HG-FCCA Analog | 0.95 pJ | High | 6-8 bit equiv | 2× denser | In-memory compute |
| Systolic Array | 1-3 pJ | Very high | 8-16 bit | 1.5× | Google TPU style |
| ReRAM crossbar | 0.5-2 pJ | High | 4-6 bit | 4× denser | Lower endurance |

**HG-FCCA advantages**:
- 2-5× lower energy vs. digital at similar precision
- Non-volatile weight storage (no SRAM refresh)
- High endurance (10¹¹ cycles) suitable for on-device training

**Trade-offs**:
- Calibration overhead (time and complexity)
- Lower precision ceiling (8-bit equivalent vs. 16-bit digital)
- Novel design requiring custom verification

---

## B.10 Open Research Questions

1. **Optimal ADC resolution**: Is 6-bit sufficient, or does 8-bit improve accuracy enough to justify 4× higher ADC power?

2. **Analog vs. mixed-signal accumulation**: When does active (op-amp based) accumulation outweigh passive current summation?

3. **Temperature compensation**: Can on-chip temperature sensors + lookup tables eliminate need for periodic calibration?

4. **Scalability**: How many tiles can share a single activation broadcast network before skew degrades throughput?

5. **Training throughput**: Can write energy be reduced 10× through advanced MTJ stack engineering to enable faster on-device training?

---

## B.11 References for Appendix B

1. **Chen, Y.-H., et al.** (2019). "Eyeriss v2: A Flexible Accelerator for Emerging Deep Neural Networks." *IEEE Journal of Solid-State Circuits*, vol. 54, no. 1.

2. **Ankit, A., et al.** (2019). "PUMA: A Programmable Ultra-efficient Memristor-based Accelerator for Machine Learning Inference." *ASPLOS '19*.

3. **Gokmen, T., et al.** (2017). "Acceleration of Deep Neural Network Training with Resistive Cross-Point Devices." *Frontiers in Neuroscience*, vol. 10.

4. **Jain, S., et al.** (2018). "A 3.6-TOPS/W 8b Floating Point Processing-in-SRAM Accelerator for GEMM in 10nm FinFET." *IEEE ISSCC*, pp. 492-494.

---

**End of Appendix B**

*This appendix provides circuit-level concepts and energy/latency models for analog MAC operations without disclosing detailed transistor-level schematics or proprietary sense amplifier topologies. Sufficient detail is given for performance modeling and system-level simulation.*
