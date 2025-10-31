# Appendix D — Tile Controller and Interconnect Specification

## D.1 Overview

Each FC-MVL tile requires a **tile controller** to:
1. Receive commands from the AOP
2. Manage weight loading and storage
3. Execute analog MAC operations
4. Coordinate calibration cycles
5. Report status and partial sums

This appendix specifies the controller finite state machine (FSM), command set, interconnect protocols, and performance characteristics.

---

## D.2 Tile Controller Architecture

### D.2.1 Block Diagram

```
┌─────────────────────────────────────────────────┐
│              Tile Controller                     │
│                                                  │
│  ┌──────────────┐        ┌──────────────┐      │
│  │  Command FSM │◄──────►│ Weight SRAM  │      │
│  │              │        │  (Staging)   │      │
│  └──────┬───────┘        └──────────────┘      │
│         │                                        │
│         ↓                                        │
│  ┌──────────────┐        ┌──────────────┐      │
│  │   Sequencer  │◄──────►│ MVL Array    │      │
│  │              │        │  Interface   │      │
│  └──────┬───────┘        └──────────────┘      │
│         │                                        │
│         ↓                                        │
│  ┌──────────────┐        ┌──────────────┐      │
│  │  ADC/Result  │◄──────►│ Calibration  │      │
│  │   Manager    │        │  Controller  │      │
│  └──────────────┘        └──────────────┘      │
│         │                                        │
│         ↓                                        │
│  ┌──────────────────────────────────────┐      │
│  │         Status & Debug               │      │
│  └──────────────────────────────────────┘      │
└───────────────────┬──────────────────────────────┘
                    │
              UCIe Interconnect
                    │
              ┌─────▼──────┐
              │    AOP     │
              └────────────┘
```

### D.2.2 Key Components

| Component | Function | Complexity |
|-----------|----------|------------|
| **Command FSM** | Decode and execute AOP commands | ~500 gates |
| **Weight SRAM** | Buffer for weight loading (512 KB) | SRAM macro |
| **Sequencer** | Generate word-line and bit-line timing | ~1K gates |
| **MVL Array IF** | Drive word lines, sense bit lines | Analog/mixed-signal |
| **ADC Manager** | Collect and buffer analog results | ~2K gates |
| **Cal Controller** | Execute calibration subroutines | ~1K gates |
| **Status/Debug** | Error reporting, performance counters | ~500 gates |

**Total digital logic**: ~5K gates ≈ 0.01 mm² @ 22 nm

---

## D.3 Finite State Machine (FSM)

### D.3.1 State Diagram

```
        ┌──────────┐
    ┌──►│   IDLE   │◄──┐
    │   └────┬─────┘   │
    │        │ LOAD_CMD│
    │        ↓         │
    │   ┌──────────┐   │
    │   │  LOADING │   │ DONE
    │   │ (Weights)│   │
    │   └────┬─────┘   │
    │        │ RUN_CMD │
    │        ↓         │
    │   ┌──────────┐   │
    │   │ COMPUTE  │   │
    │   │ (Analog) │───┘
    │   └────┬─────┘
    │        │ CAL_CMD
    │        ↓
    │   ┌──────────┐
    └───│CALIBRATE │
        │          │
        └──────────┘
            ↓ ERROR
        ┌──────────┐
        │  ERROR   │
        │  (Halt)  │
        └──────────┘
```

### D.3.2 State Descriptions

| State | Function | Duration | Power |
|-------|----------|----------|-------|
| **IDLE** | Await commands, minimal power | Indefinite | <10 mW |
| **LOADING** | Transfer weights from AOP to tile SRAM | 100 µs - 10 ms | 200 mW |
| **COMPUTE** | Execute analog MAC, collect results | 0.5 - 5 ms | 1.5 W |
| **CALIBRATE** | Run calibration subroutine | 2 ms | 100 mW |
| **ERROR** | Fault condition, await reset | Indefinite | <5 mW |

### D.3.3 State Transition Logic

```verilog
// Simplified FSM (Verilog pseudocode)
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
    end else begin
        case (state)
            IDLE: begin
                if (cmd_valid && cmd_opcode == LOAD_WEIGHTS)
                    state <= LOADING;
                else if (cmd_valid && cmd_opcode == RUN_BLOCK)
                    state <= COMPUTE;
                else if (cmd_valid && cmd_opcode == CALIBRATE)
                    state <= CALIBRATE;
            end
            
            LOADING: begin
                if (load_complete)
                    state <= IDLE;
                else if (load_error)
                    state <= ERROR;
            end
            
            COMPUTE: begin
                if (compute_done)
                    state <= IDLE;
                else if (compute_timeout)
                    state <= ERROR;
            end
            
            CALIBRATE: begin
                if (cal_done)
                    state <= IDLE;
                else if (cal_failed)
                    state <= ERROR;
            end
            
            ERROR: begin
                if (error_clear)
                    state <= IDLE;
            end
        endcase
    end
end
```

---

## D.4 Command Set Specification

### D.4.1 Command Format

**32-bit command word**:

```
┌─────────┬────────┬─────────────────────┬──────────┐
│ Opcode  │ TileID │      Address        │  Length  │
│ (8-bit) │ (4-bit)│      (12-bit)       │  (8-bit) │
└─────────┴────────┴─────────────────────┴──────────┘
```

### D.4.2 Command Opcodes

| Opcode | Mnemonic | Function | Arguments | Latency |
|--------|----------|----------|-----------|---------|
| **0x01** | LOAD_WEIGHTS | Load weight data to tile SRAM | addr, len | Variable (10 µs - 10 ms) |
| **0x02** | RUN_BLOCK | Execute analog MAC operation | N/A | 0.5 - 5 ms |
| **0x03** | READ_PARTIAL | Read accumulated partial sums | addr, len | 100 ns per word |
| **0x04** | CALIBRATE | Execute calibration cycle | N/A | 2 ms |
| **0x05** | SLEEP | Enter low-power mode | N/A | Immediate |
| **0x06** | WAKE | Exit low-power mode | N/A | <1 µs |
| **0x07** | RESET | Software reset of tile | N/A | 10 µs |
| **0x08** | STATUS | Query tile status register | N/A | <100 ns |
| **0x09** | CONFIG | Set tile configuration | config_word | 1 µs |

### D.4.3 LOAD_WEIGHTS Command Details

**Payload format**:
- 32-bit command word
- Followed by N × 32-bit data words
- Each data word contains 10-11 quantized weights (2-3 bits each)

**Example**: Load 512×512 4-level weights
```
Total weights: 512 × 512 = 262,144 weights
Bits per weight: 2 bits
Total bits: 524,288 bits = 64 KB
Transfer time @ 1 GB/s: 64 µs
Verification time: ~50 µs
Total: ~100 µs
```

**Pseudo-protocol**:
```python
def load_weights(tile_id, weights_array):
    """
    Load quantized weights into tile.
    
    Args:
        tile_id: Target tile (0-15)
        weights_array: np.array of shape (512, 512) with values 0-3 (4-level)
    """
    # Pack weights into 32-bit words (16 weights × 2 bits each)
    packed_data = pack_weights(weights_array)
    
    # Send LOAD command
    cmd = (0x01 << 24) | (tile_id << 20) | (0 << 8) | len(packed_data)
    send_command(cmd)
    
    # Stream weight data
    for word in packed_data:
        send_data(word)
    
    # Wait for acknowledgment
    status = wait_for_ack(tile_id, timeout_ms=100)
    
    if status != LOAD_COMPLETE:
        raise TileLoadError(f"Tile {tile_id} load failed: {status}")
```

### D.4.4 RUN_BLOCK Command Details

**Operation**:
1. AOP broadcasts activations to all tiles simultaneously
2. Each tile performs local analog MAC
3. Tiles report partial sums back to AOP
4. AOP accumulates and applies post-processing

**Timing**:
```
┌────────────┬──────────┬─────────┬────────────┐
│ Activation │  Analog  │   ADC   │  Readback  │
│  Broadcast │   MAC    │ Convert │  to AOP    │
├────────────┼──────────┼─────────┼────────────┤
│   50 ns    │  100 ns  │ 200 ns  │   100 ns   │
└────────────┴──────────┴─────────┴────────────┘
Total per block: ~0.5 µs
```

### D.4.5 CALIBRATE Command

**Automatic sequence** (no parameters needed):
1. Read reference cells
2. Compute errors
3. Adjust bias DACs
4. Verify
5. Report status

**Duration**: Fixed 2 ms

**Return value**: Calibration quality metric (0-100)

---

## D.5 Inter-Tile Interconnect

### D.5.1 Physical Layer: UCIe-Compatible Links

**UCIe (Universal Chiplet Interconnect Express)** specification compliance:

| Parameter | HG-FCCA Implementation | UCIe Standard |
|-----------|------------------------|---------------|
| Protocol | UCIe PHY + custom link layer | UCIe 1.0 |
| Lane width | 64 lanes (per tile pair) | 16-256 lanes |
| Signaling | Single-ended or differential | Both supported |
| Data rate | 8-16 Gb/s per lane | Up to 32 Gb/s |
| Total BW | 64 GB/s (bidirectional) | Scalable |
| Latency | <10 ns per hop | ~1 ns/mm |
| Energy | <1 pJ/bit | 0.5-2 pJ/bit |

### D.5.2 Logical Topology

**Star topology** (baseline):
```
        ┌─────────┐
        │   AOP   │ (Central hub)
        └────┬────┘
    ┌────────┼────────┐
    │        │        │
┌───▼───┐ ┌─▼──┐ ┌───▼───┐
│ Tile0 │ │ T1 │ │ Tile2 │ ...
└───────┘ └────┘ └───────┘
```

**Advantages**:
- Simple routing
- Low latency to AOP
- Easy fault isolation

**Disadvantages**:
- High AOP I/O bandwidth requirement
- No direct tile-to-tile communication

**Mesh topology** (future scaling):
```
┌────┐──┌────┐──┌────┐──┌────┐
│ T0 │──│ T1 │──│ T2 │──│ T3 │
└─┬──┘  └─┬──┘  └─┬──┘  └─┬──┘
  │       │       │       │
┌─▼──┐  ┌─▼──┐  ┌─▼──┐  ┌─▼──┐
│ T4 │──│ T5 │──│ T6 │──│ T7 │
└─┬──┘  └─┬──┘  └─┬──┘  └─┬──┘
  │       │       │       │
  └───────┴───────┴───────┴──► AOP
```

**Advantages**:
- Direct tile-to-tile partial sum exchange
- Reduced AOP bandwidth
- Better scalability to 32+ tiles

**Disadvantages**:
- Higher latency (multi-hop)
- More complex routing

**HG-FCCA choice**: Star for 8-16 tiles, mesh for >16 tiles.

### D.5.3 Protocol Stack

```
┌─────────────────────────────────┐
│     Application Layer           │  ← Neural network workload
│  (Model partitioning, scheduling)│
├─────────────────────────────────┤
│      Transport Layer            │  ← Reliable data transfer
│   (Error detection, retries)    │
├─────────────────────────────────┤
│       Link Layer                │  ← Framing, flow control
│   (Packet headers, CRC)         │
├─────────────────────────────────┤
│      Physical Layer             │  ← UCIe electrical signaling
│   (SerDes, equalization)        │
└─────────────────────────────────┘
```

### D.5.4 Packet Format

**Activation broadcast packet** (256 bytes):

```
┌──────────┬─────────┬──────────────────┬─────────┐
│  Header  │ Seq ID  │  Payload (252B)  │   CRC   │
│  (2B)    │  (2B)   │  (Activations)   │   (4B)  │
└──────────┴─────────┴──────────────────┴─────────┘
```

**Header fields**:
- Packet type (4 bits): DATA, COMMAND, STATUS, ACK
- Destination (4 bits): Tile ID or broadcast (0xF)
- Priority (4 bits): 0=background, 15=realtime
- Flags (4 bits): EOP, ERROR, RETRY, etc.

**Partial sum return packet** (64 bytes):

```
┌──────────┬─────────┬──────────────────┬─────────┐
│  Header  │ Tile ID │  Partial Sums    │   CRC   │
│  (2B)    │  (2B)   │  (56B = 14×INT32)│   (4B)  │
└──────────┴─────────┴──────────────────┴─────────┘
```

---

## D.6 Performance Analysis

### D.6.1 Bandwidth Requirements

**Activation broadcast** (per inference):
- Activations per layer: 512 values × 8 bits = 512 B
- Frequency: ~100 Hz (10 ms per inference)
- Bandwidth: 512 B × 100 Hz = 51.2 KB/s per tile
- For 16 tiles: 819 KB/s (negligible vs. 64 GB/s link capacity)

**Weight loading** (one-time):
- Weights per tile: 512×512 × 2 bits = 64 KB
- Load time @ 1 GB/s: 64 µs
- Rare operation (only during model swap)

**Partial sum readback** (per inference):
- Results per tile: 512 INT16 values = 1 KB
- Frequency: ~100 Hz
- Bandwidth: 1 KB × 100 Hz × 16 tiles = 1.6 MB/s

**Total sustained bandwidth**: ~2 MB/s (<<1% of 64 GB/s capacity)

**Conclusion**: Interconnect bandwidth is not a bottleneck.

### D.6.2 Latency Budget

**End-to-end data path**:

| Stage | Latency | Notes |
|-------|---------|-------|
| AOP command generation | 100 ns | Digital logic |
| Serialization | 50 ns | Pack into UCIe packet |
| Transmission (electrical) | 1 ns/mm × 10 mm | 10 ns | Die-to-die |
| Deserialization | 50 ns | Unpack at tile |
| Tile FSM dispatch | 50 ns | State transition |
| **Total (command)** | **260 ns** | Negligible |

**Analog MAC execution** (dominant):
- Setup: 50 ns
- Computation: 100-200 ns
- ADC: 200 ns
- **Total (MAC)**: 350-450 ns

**Interconnect contributes <10%** of total latency.

### D.6.3 Energy per Transfer

**UCIe energy** (~1 pJ/bit):
- 64-byte activation packet: 512 bits × 1 pJ = 512 pJ = 0.5 nJ
- 16 tiles: 16 × 0.5 nJ = 8 nJ per broadcast

**Compared to MAC energy**:
- One 512×512 MAC block: 512² × 1 pJ = 262 µJ
- Interconnect: 8 nJ
- **Ratio**: 8 nJ / 262 µJ = 0.003% (negligible)

---

## D.7 Fault Tolerance and Error Handling

### D.7.1 Error Detection

**CRC-32** on all packets:
- Detects burst errors up to 32 bits
- Overhead: 4 bytes per packet (~2%)

**Timeout mechanisms**:
- Command acknowledgment timeout: 10 ms
- Calibration timeout: 5 ms
- If timeout expires → mark tile as faulty

### D.7.2 Retry Logic

```python
def send_command_with_retry(tile_id, cmd, max_retries=3):
    """
    Send command with automatic retry on failure.
    """
    for attempt in range(max_retries):
        send_command(tile_id, cmd)
        
        try:
            ack = wait_for_ack(tile_id, timeout_ms=10)
            if ack == SUCCESS:
                return SUCCESS
        except TimeoutError:
            log.warning(f"Tile {tile_id} timeout, retry {attempt+1}/{max_retries}")
            continue
    
    # All retries failed
    log.error(f"Tile {tile_id} unresponsive, marking as faulty")
    mark_tile_faulty(tile_id)
    return FAILURE
```

### D.7.3 Graceful Degradation

If tile fails:
1. Redistribute workload to remaining tiles
2. Notify AOP scheduler
3. Continue operation at reduced throughput

Example: 16 tiles → 15 working tiles
- Throughput reduction: 1/16 = 6.25%
- Still meets latency requirements for most workloads

---

## D.8 Power Management States

### D.8.1 Power State Table

| State | Power | Wake Latency | Use Case |
|-------|-------|--------------|----------|
| **ACTIVE** | 1.5 W | N/A | Normal inference |
| **IDLE** | 50 mW | <1 µs | Between operations |
| **SLEEP** | 5 mW | 10 µs | Extended idle (>1 s) |
| **DEEP_SLEEP** | <1 mW | 100 µs | System suspend |
| **OFF** | <100 µW | ~10 ms | Tile disabled |

### D.8.2 Dynamic Voltage and Frequency Scaling (DVFS)

**Three operating points**:

| Mode | V_DD | Frequency | Performance | Power |
|------|------|-----------|-------------|-------|
| **High** | 1.1 V | 500 MHz | 100% | 1.5 W |
| **Medium** | 1.0 V | 400 MHz | 80% | 1.0 W |
| **Low** | 0.9 V | 300 MHz | 60% | 0.7 W |

**Switching time**: <10 µs (voltage regulator settling)

**Use case**: Throttle tiles during thermal stress while maintaining SRP at full power.

---

## D.9 Debug and Observability

### D.9.1 Performance Counters

Each tile exposes 32-bit counters:

| Counter | Description |
|---------|-------------|
| `CYCLES` | Total clock cycles since reset |
| `MAC_OPS` | Number of MAC operations executed |
| `CAL_COUNT` | Number of calibration cycles |
| `ERRORS` | Error event counter |
| `TIMEOUTS` | Command timeout counter |
| `POWER_ON_TIME` | Cumulative time in ACTIVE state |
| `TEMP_PEAK` | Maximum recorded junction temperature |

**Access**: Read via STATUS command, logged to AOP for analysis.

### D.9.2 Debug Interface

**JTAG boundary scan**:
- Standard IEEE 1149.1 compliant
- Allows low-level access to tile registers
- Used during manufacturing test and bringup

**Performance trace buffer** (optional):
- 4 KB circular buffer logging events
- Captures last 1000 commands with timestamps
- Useful for debugging protocol issues

---

## D.10 Optical Interconnect Extension (Future)

### D.10.1 Motivation

For >32 tiles, electrical links face:
- High capacitance (>10 pF)
- Signal integrity challenges
- Energy penalty (>2 pJ/bit)

**Silicon photonics** offers:
- Energy: <0.1 pJ/bit
- Bandwidth: >100 Gb/s per wavelength
- Low latency: ~10 ps skew

### D.10.2 Hybrid Electrical-Optical Architecture

```
        ┌─────────┐
        │   AOP   │
        └────┬────┘
             │ Electrical (control)
    ┌────────┼────────┐
    │    Optical      │
    │   Broadcast     │
    │    Network      │
    └────┬──┬──┬──┬───┘
         │  │  │  │
      [T0][T1][T2][T3]...
```

**Electrical**: Low-bandwidth control commands
**Optical**: High-bandwidth activation broadcast

### D.10.3 Photonic Component Requirements

| Component | Function | Specs |
|-----------|----------|-------|
| **Laser** | Light source | 1310 nm, 10 mW |
| **Modulator** | Encode data | 25 Gb/s, MZM or ring |
| **Waveguide** | Distribute light | Si waveguide, <1 dB/cm loss |
| **Photodetector** | Receive data | >25 GHz BW, Ge-on-Si |
| **Grating coupler** | Chip I/O | >-3 dB coupling |

**Challenges**:
- Thermal sensitivity (λ drift ~0.1 nm/°C)
- High NRE cost ($10M+ for photonic mask set)
- Integration complexity (separate photonic die)

**HG-FCCA strategy**: Electrical for MVP, photonic for future scaling.

---

## D.11 References for Appendix D

1. **UCIe Consortium** (2022). "Universal Chiplet Interconnect Express Specification v1.0."

2. **Salman, E., et al.** (2020). "3D Integration and Through-Silicon Vias in VLSI." *Synthesis Lectures on Emerging Engineering Technologies*.

3. **Beamer, S., et al.** (2017). "Re-architecting DRAM Memory Systems with Monolithically Integrated Silicon Photonics." *ISCA '17*.

4. **Ahn, J., et al.** (2019). "Scalable On-Package Optical Interconnects for High-Performance Computing." *IEEE Micro*, vol. 39, no. 5.

---

**End of Appendix D**

*This appendix provides complete tile controller FSM, command set, and interconnect specifications. Proprietary RTL code and timing closure reports are available under NDA to design partners.*
