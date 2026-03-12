# Multicast Load Test

Tests for the `CLUSTER_LOAD` instruction on supported cluster-load capable targets.

## Overview

`CLUSTER_LOAD` is a global memory load instruction that can broadcast data to multiple workgroups within a cluster, reducing redundant memory traffic when multiple workgroups need the same data.

### Clusters

A cluster is a group of up to 16 Workgroup Processors (WGPs) that can share data via multicast. When multiple workgroups within a cluster request the same address, the hardware fetches the data once and broadcasts it to all requesters.

### Broadcasting

Broadcasting is controlled by the M0 register:
- Bits `M0[15:0]` form a bitmask indicating which WGPs should receive the data
- All waves requesting the same data must set identical M0 values
- If `M0[15:0] == 0`, the load behaves as a normal non-multicast load
- `M0[16]` is an early-timeout bit: when set, the instruction completes without waiting for all masked WGPs to participate, preventing deadlock when fewer WGPs are launched than the mask implies

### Variants

| Instruction | Data Size |
|-------------|-----------|
| `cluster_load_b32` | 32-bit |
| `cluster_load_b64` | 64-bit |
| `cluster_load_b128` | 128-bit |

**Wait instruction:** `s_wait_loadcnt` (increments LOADcnt counter)

## Tests

`test_cluster_load_multicast` covers five groups:

| Group | Description |
|-------|-------------|
| `SingleWGP` | B32/B64/B128 correctness with a single WGP, mask=0x1 |
| `M0Mask` | mask=0x0 (non-multicast path) and mask=0x1 (single-WGP multicast) |
| `MultiWGP` | 2–6 WGP cluster broadcasts for B32, B64, B128 |
| `PartialBroadcast` | Non-contiguous mask (0x5): only WGPs 0 and 2 issue cluster load, others use a plain load |
| `ConcurrentGroups` | Two independent broadcast groups within the same 4-WGP cluster |
| `EarlyTimeout` | M0[16] early-timeout bit prevents deadlock when fewer WGPs are launched than the mask claims |
