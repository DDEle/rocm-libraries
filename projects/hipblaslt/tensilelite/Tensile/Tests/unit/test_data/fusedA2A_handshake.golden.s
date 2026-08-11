
/******************************************/
/* fused-A2A cross-card handshake (design spec 2.3): counter election + SDMA packet submit + DRAIN */
/******************************************/
s_mov_b64 exec, -1                                 // fused-A2A: full exec before wave-0 election
// loadKernArg 10 KernArgAddress dword=1 sgprOffset=0x60
s_waitcnt lgkmcnt(0)                               // wait FusedAM
s_lshr_b32 s10, s10, 8                             // AM_tiles = FusedAM >> log2(MT0=256)
s_cmp_gt_u32 s10, s[sgprWorkGroup0]                // AM_tiles > WorkGroup0? (this WG in PUSH region)
s_cbranch_scc0 label_fusedA2A_local_tally          // WorkGroup0 >= AM_tiles -> not a PUSH WG, skip the store wait + barrier
s_waitcnt vmcnt(0)                                 // fused-A2A: my A2A stores (sc1) are in HBM before the counter (spec 2.3 step 2)
s_barrier                                          // fused-A2A: all waves done before counter election
v_readfirstlane_b32 s21, v[vgprSerial]             // wave 0 elects the WG's single counter writer
s_cmp_eq_u32 s21, 0                                // wave 0?
s_cbranch_scc0 label_fusedA2A_handshake_after      // non-wave-0 -> skip (single writer per WG)
s_mov_b64 exec, 1                                  // fused-A2A: isolate lane 0 for the once-per-WG counter atomic + flag store
// loadKernArg 10 KernArgAddress dword=1 sgprOffset=0x50
// loadKernArg 11 KernArgAddress dword=1 sgprOffset=0x64
// loadKernArg 12 KernArgAddress dword=1 sgprOffset=0x58
// loadKernArg 13 KernArgAddress dword=1 sgprOffset=0x68
// loadKernArg 14 KernArgAddress dword=2 sgprOffset=0x40
s_waitcnt lgkmcnt(0)                               // wait FusedMyRank/TilesPerRank/NShard/TokenTiles/counter_ptr
s_mul_i32 s19, s[sgprWorkGroup0], 256              // n_col_base_wg = WorkGroup0 * MT0
s_mov_b32 s18, 0                                   // dst_rank = 0 (default)
s_mul_i32 s20, s12, 1                              // cand shard_lo = 1 * n_shard
s_cmp_gt_u32 s20, s19                              // shard_lo > n_col_base_wg? (WG below rank 1)
s_cbranch_scc1 label_fusedA2A_flag_skip1           // below rank 1: keep current winner
s_mov_b32 s18, 1                                   // dst_rank = 1
label_fusedA2A_flag_skip1:  /// n_col_base_wg below rank 1
s_mul_i32 s20, s12, 2                              // cand shard_lo = 2 * n_shard
s_cmp_gt_u32 s20, s19                              // shard_lo > n_col_base_wg? (WG below rank 2)
s_cbranch_scc1 label_fusedA2A_flag_skip2           // below rank 2: keep current winner
s_mov_b32 s18, 2                                   // dst_rank = 2
label_fusedA2A_flag_skip2:  /// n_col_base_wg below rank 2
s_mul_i32 s20, s12, 3                              // cand shard_lo = 3 * n_shard
s_cmp_gt_u32 s20, s19                              // shard_lo > n_col_base_wg? (WG below rank 3)
s_cbranch_scc1 label_fusedA2A_flag_skip3           // below rank 3: keep current winner
s_mov_b32 s18, 3                                   // dst_rank = 3
label_fusedA2A_flag_skip3:  /// n_col_base_wg below rank 3
s_mul_i32 s20, s12, 4                              // cand shard_lo = 4 * n_shard
s_cmp_gt_u32 s20, s19                              // shard_lo > n_col_base_wg? (WG below rank 4)
s_cbranch_scc1 label_fusedA2A_flag_skip4           // below rank 4: keep current winner
s_mov_b32 s18, 4                                   // dst_rank = 4
label_fusedA2A_flag_skip4:  /// n_col_base_wg below rank 4
s_mul_i32 s20, s12, 5                              // cand shard_lo = 5 * n_shard
s_cmp_gt_u32 s20, s19                              // shard_lo > n_col_base_wg? (WG below rank 5)
s_cbranch_scc1 label_fusedA2A_flag_skip5           // below rank 5: keep current winner
s_mov_b32 s18, 5                                   // dst_rank = 5
label_fusedA2A_flag_skip5:  /// n_col_base_wg below rank 5
s_mul_i32 s20, s12, 6                              // cand shard_lo = 6 * n_shard
s_cmp_gt_u32 s20, s19                              // shard_lo > n_col_base_wg? (WG below rank 6)
s_cbranch_scc1 label_fusedA2A_flag_skip6           // below rank 6: keep current winner
s_mov_b32 s18, 6                                   // dst_rank = 6
label_fusedA2A_flag_skip6:  /// n_col_base_wg below rank 6
s_mul_i32 s20, s12, 7                              // cand shard_lo = 7 * n_shard
s_cmp_gt_u32 s20, s19                              // shard_lo > n_col_base_wg? (WG below rank 7)
s_cbranch_scc1 label_fusedA2A_flag_skip7           // below rank 7: keep current winner
s_mov_b32 s18, 7                                   // dst_rank = 7
label_fusedA2A_flag_skip7:  /// n_col_base_wg below rank 7
s_lshl_b32 s20, s18, 3                             // dst_rank * 8 (byte offset into peer_ptr[] array)
s_add_u32 s20, s20, 0                              // kernarg offset = fusedBase + peer_ptr_0 + dst_rank*8
// loadKernArg 16 KernArgAddress dword=2 sgprOffset=s20
s_waitcnt lgkmcnt(0)                               // wait peer_ptr[dst_rank] load
s_mul_i32 s19, s18, s13                            // dst_rank * tokenTiles
s_add_u32 s19, s19, s[sgprWorkGroup1]              // counter index = dst_rank*tokenTiles + j (j = WorkGroup1)
s_lshl_b32 s19, s19, 2                             // * 4 (u32 counter byte offset)
s_add_u32 s14, s14, s19                            // counter[dst_rank][j] lo = counter_ptr + (dst_rank*tokenTiles+j)*4
s_addc_u32 s15, s15, 0                             // counter[dst_rank][j] hi (carry)
v_mov_b32 v2, s14                                  // counter addr lo -> vgpr
v_mov_b32 v3, s15                                  // counter addr hi -> vgpr
v_mov_b32 v1, 1                                    // counter increment = 1
global_atomic_add v4, v[2:3], v1, off sc0          // old = atomic_add(counter[dst_rank][j], 1) device scope, return pre-op (sc0)
s_waitcnt vmcnt(0)                                 // fused-A2A: wait counter atomic return (load counter)
v_readfirstlane_b32 s19, v4                        // old -> sgpr
s_add_u32 s19, s19, 1                              // old + 1
s_cmp_eq_u32 s19, s11                              // old+1 == FusedTilesPerRank? (last WG for (dst_rank, j))
s_cbranch_scc0 label_fusedA2A_handshake_notlast    // not the last WG -> skip the SDMA submit

/* fused-A2A: build + submit the SDMA COPY_SUBWIN + ATOMIC packet pair */
// loadKernArg 22 KernArgAddress dword=2 sgprOffset=0x48
s_waitcnt lgkmcnt(0)                               // wait FusedSdmaQueues
s_mul_i32 s19, s18, 56                             // dst_rank * sizeof(SdmaQueueDeviceHandle)
s_add_u32 s22, s22, s19                            // handle lo = FusedSdmaQueues + dst_rank*56
s_addc_u32 s23, s23, 0                             // handle hi (carry)
s_load_dwordx2 s[24:25], s[22:23], 0x30            // seed cachedHwReadIndex from handle+48 (private, never stored back)
s_waitcnt lgkmcnt(0)                               // wait cachedHwReadIndex seed
s_mul_i32 s19, s[sgprWorkGroup0], 256              // n_col_base_wg = WorkGroup0 * MT0
s_mov_b32 s20, 0                                   // dst_rank = 0 (default)
s_mov_b32 s21, 0                                   // shard_base = 0 (rank 0)
s_mul_i32 s21, s12, 1                              // cand shard_lo = 1 * n_shard
s_cmp_gt_u32 s21, s19                              // shard_lo > n_col_base_wg? (WG below rank 1)
s_cbranch_scc1 label_fusedA2A_recv_skip1           // below rank 1: keep current winner
s_mov_b32 s20, 1                                   // dst_rank = 1
label_fusedA2A_recv_skip1:  /// n_col_base_wg below rank 1
s_mul_i32 s21, s12, 2                              // cand shard_lo = 2 * n_shard
s_cmp_gt_u32 s21, s19                              // shard_lo > n_col_base_wg? (WG below rank 2)
s_cbranch_scc1 label_fusedA2A_recv_skip2           // below rank 2: keep current winner
s_mov_b32 s20, 2                                   // dst_rank = 2
label_fusedA2A_recv_skip2:  /// n_col_base_wg below rank 2
s_mul_i32 s21, s12, 3                              // cand shard_lo = 3 * n_shard
s_cmp_gt_u32 s21, s19                              // shard_lo > n_col_base_wg? (WG below rank 3)
s_cbranch_scc1 label_fusedA2A_recv_skip3           // below rank 3: keep current winner
s_mov_b32 s20, 3                                   // dst_rank = 3
label_fusedA2A_recv_skip3:  /// n_col_base_wg below rank 3
s_mul_i32 s21, s12, 4                              // cand shard_lo = 4 * n_shard
s_cmp_gt_u32 s21, s19                              // shard_lo > n_col_base_wg? (WG below rank 4)
s_cbranch_scc1 label_fusedA2A_recv_skip4           // below rank 4: keep current winner
s_mov_b32 s20, 4                                   // dst_rank = 4
label_fusedA2A_recv_skip4:  /// n_col_base_wg below rank 4
s_mul_i32 s21, s12, 5                              // cand shard_lo = 5 * n_shard
s_cmp_gt_u32 s21, s19                              // shard_lo > n_col_base_wg? (WG below rank 5)
s_cbranch_scc1 label_fusedA2A_recv_skip5           // below rank 5: keep current winner
s_mov_b32 s20, 5                                   // dst_rank = 5
label_fusedA2A_recv_skip5:  /// n_col_base_wg below rank 5
s_mul_i32 s21, s12, 6                              // cand shard_lo = 6 * n_shard
s_cmp_gt_u32 s21, s19                              // shard_lo > n_col_base_wg? (WG below rank 6)
s_cbranch_scc1 label_fusedA2A_recv_skip6           // below rank 6: keep current winner
s_mov_b32 s20, 6                                   // dst_rank = 6
label_fusedA2A_recv_skip6:  /// n_col_base_wg below rank 6
s_mul_i32 s21, s12, 7                              // cand shard_lo = 7 * n_shard
s_cmp_gt_u32 s21, s19                              // shard_lo > n_col_base_wg? (WG below rank 7)
s_cbranch_scc1 label_fusedA2A_recv_skip7           // below rank 7: keep current winner
s_mov_b32 s20, 7                                   // dst_rank = 7
label_fusedA2A_recv_skip7:  /// n_col_base_wg below rank 7
s_mul_i32 s21, s20, s12                            // shard_base = dst_rank * n_shard
s_lshl_b32 s20, s20, 3                             // dst_rank * 8 (byte offset into peer_ptr[] array)
s_add_u32 s20, s20, 0                              // kernarg offset = fusedBase + peer_ptr_0 + dst_rank*8
// loadKernArg 26 KernArgAddress dword=2 sgprOffset=s20
s_waitcnt lgkmcnt(0)                               // wait peer_ptr[dst_rank] load
s_add_u32 s26, s26, 0x1000                         // recv base = peer_ptr[dst_rank] + recv offset
s_addc_u32 s27, s27, 0                             // recv base hi carry
s_mul_i32 s28, s[sgprWorkGroup1], 256              // src_y = j * MT1 (folded into the base, not a field)
s_mul_i32 s29, s[sgprSizesFree+0], s[sgprSizesFree+1] // src_slice = M * N (single-plane, don't-care)
s_mul_i32 s19, s18, s12                            // src_x = p * nShard (folded into the base, not a field)
s_mul_hi_u32 s35, s28, s[sgprStrideD1J]            // src row offset = j*MT1 * ldd (64-bit: unbounded in N and ldd) (hi)
s_mul_i32 s34, s28, s[sgprStrideD1J]               // src row offset = j*MT1 * ldd (64-bit: unbounded in N and ldd) (lo)
s_add_u32 s34, s34, s19                            // + p*nShard (feature offset)
s_addc_u32 s35, s35, 0                             // propagate carry into the high word
s_lshl_b64 s[34:35], s[34:35], 1                   // src offset: elements -> bytes (sizeof(bf16))
s_add_u32 s32, s8, s34                             // srcBase = D + src offset (src_x/src_y now 0)

s_addc_u32 s33, s9, s35                            // srcBase = D + src offset (src_x/src_y now 0)
s_mul_i32 s19, s10, s[sgprSizesFree+1]             // myRank * N
s_add_u32 s19, s19, s28                            // dst row = myRank*N + j*MT1 (folded, not a field)
s_mul_hi_u32 s35, s19, s12                         // dst row offset = dst row * nShard (64-bit: unbounded in W and N) (hi)
s_mul_i32 s34, s19, s12                            // dst row offset = dst row * nShard (64-bit: unbounded in W and N) (lo)
s_lshl_b64 s[34:35], s[34:35], 1                   // dst offset: elements -> bytes (sizeof(bf16))
s_add_u32 s26, s26, s34                            // dstBase = recv slot + dst offset (dst_x/dst_y now 0)

s_addc_u32 s27, s27, s35                           // dstBase = recv slot + dst offset (dst_x/dst_y now 0)
s_mul_i32 s30, s12, 256                            // dst_slice = MT1 * nShard (one band's plane)
s_sub_u32 s31, s[sgprSizesFree+1], s28             // N - j*MT1 (tokens left in this tile)
s_min_u32 s31, s31, 256                            // rect_y = min(MT1, N - j*MT1) (clamp tail tile)
v_mov_b32 v5, 0x80000401                           // SUBWIN DW0: op=COPY sub_op=RECT elementsize=log2(16B)
v_mov_b32 v6, s32                                  // SUBWIN DW1: srcBase lo
v_mov_b32 v7, s33                                  // SUBWIN DW2: srcBase hi
v_mov_b32 v8, 0x0                                  // SUBWIN DW3: src_x=0|src_y=0 (folded into srcBase)
s_lshr_b32 s19, s[sgprStrideD1J], 3                // SUBWIN DW4: src_pitch-1 (bf16 elems -> packet elems)
s_sub_u32 s19, s19, 1                              // SUBWIN DW4: src_pitch-1 (pitch - 1)
s_lshl_b32 s19, s19, 13                            // SUBWIN DW4: src_pitch-1 (<< 13)
v_mov_b32 v9, s19                                  // SUBWIN DW4: src_pitch-1
s_lshr_b32 s19, s29, 3                             // SUBWIN DW5: src_slice-1 (bf16 elems -> packet elems)
s_sub_u32 s19, s19, 1                              // SUBWIN DW5: src_slice-1 (slice - 1)
v_mov_b32 v10, s19                                 // SUBWIN DW5: src_slice-1
v_mov_b32 v11, s26                                 // SUBWIN DW6: dstBase lo
v_mov_b32 v12, s27                                 // SUBWIN DW7: dstBase hi
v_mov_b32 v13, 0x0                                 // SUBWIN DW8: dst_x=0|dst_y=0 (folded into dstBase)
s_lshr_b32 s19, s12, 3                             // SUBWIN DW9: dst_pitch-1 (bf16 elems -> packet elems)
s_sub_u32 s19, s19, 1                              // SUBWIN DW9: dst_pitch-1 (pitch - 1)
s_lshl_b32 s19, s19, 13                            // SUBWIN DW9: dst_pitch-1 (<< 13)
v_mov_b32 v14, s19                                 // SUBWIN DW9: dst_pitch-1
s_lshr_b32 s19, s30, 3                             // SUBWIN DW10: dst_slice-1 (bf16 elems -> packet elems)
s_sub_u32 s19, s19, 1                              // SUBWIN DW10: dst_slice-1 (slice - 1)
v_mov_b32 v15, s19                                 // SUBWIN DW10: dst_slice-1
s_lshr_b32 s19, s12, 3                             // SUBWIN DW11: rect_x-1|rect_y-1 (rectX) (bf16 elems -> packet elems)
s_sub_u32 s19, s19, 1                              // SUBWIN DW11: rect_x-1|rect_y-1 (rectX - 1)
s_sub_u32 s20, s31, 1                              // SUBWIN DW11: rect_x-1|rect_y-1 (rectY - 1, rows: NOT scaled)
s_lshl_b32 s20, s20, 16                            // SUBWIN DW11: rect_x-1|rect_y-1 ((rectY-1) << 16)
s_or_b32 s19, s19, s20                             // SUBWIN DW11: rect_x-1|rect_y-1 | (rectY-1) << 16
v_mov_b32 v16, s19                                 // SUBWIN DW11: rect_x-1|rect_y-1
v_mov_b32 v17, 0x0                                 // SUBWIN DW12: rect_z=0, default cache/swizzle
s_lshl_b32 s19, s10, 2                             // myRank * 4 (u32 flag-slot byte offset: the ATOMIC is an ADD_RTN_32)
s_add_u32 s34, s16, s19                            // flag addr lo = peer_ptr[p] + myRank*4
s_addc_u32 s35, s17, 0                             // flag addr hi (carry)
v_mov_b32 v18, 0x1e00000a                          // ATOMIC DW0: op=ATOMIC operation=ADD_RTN_32
v_mov_b32 v19, s34                                 // ATOMIC DW1: addr lo
v_mov_b32 v20, s35                                 // ATOMIC DW2: addr hi
v_mov_b32 v21, 0x1                                 // ATOMIC DW3: src_data lo (addend)
v_mov_b32 v22, 0x0                                 // ATOMIC DW4: src_data hi (unused by ADD_RTN_32)
v_mov_b32 v23, 0x0                                 // ATOMIC DW5: cmp_data lo (unused)
v_mov_b32 v24, 0x0                                 // ATOMIC DW6: cmp_data hi (unused)
v_mov_b32 v25, 0x0                                 // ATOMIC DW7: loop_interval=0
  s_load_dwordx2 s[30:31], s[22:23], 0x20
s_waitcnt lgkmcnt(0)                               // wait cachedWptr pointer load
label_sdma_reserve_loop:  /// ReserveQueueSpace: CAS retry loop
v_mov_b32 v26, s30                                 // cachedWptr addr lo
v_mov_b32 v27, s31                                 // cachedWptr addr hi
global_load_dwordx2 v[30:31], v[26:27], off sc1    // cur = load cachedWptr (AGENT scope, sc1)
s_waitcnt vmcnt(0)                                 // wait cachedWptr load
v_readfirstlane_b32 s26, v30                       // cur lo -> sgpr
v_readfirstlane_b32 s27, v31                       // cur hi -> sgpr
s_mov_b32 s21, 0                                   // offset = 0 (no pad)
s_and_b32 s34, s26, 262143                         // WrapIntoRing(cur)
s_add_u32 s36, s34, 84                             // WrapIntoRing(cur) + size
s_cmp_lt_u32 s36, 262145                           // wrap+size <= queueSize? (fits without wrap)
s_cbranch_scc1 label_sdma_reserve_nopad            // fits -> no padding
s_sub_u32 s21, 262144, s34                         // offset = queueSize - WrapIntoRing(cur) (pad ring tail)
label_sdma_reserve_nopad:  /// ReserveQueueSpace: no wrap padding
s_add_u32 s32, s26, 84                             // new lo = cur + size
s_addc_u32 s33, s27, 0                             // new hi (carry)
s_add_u32 s32, s32, s21                            // new lo += offset
s_addc_u32 s33, s33, 0                             // new hi (carry)
s_mov_b32 s35, 0                                   // CanWriteUpto = false (default)
s_sub_u32 s36, s32, s24                            // CanWriteUpto: upto - cachedHwReadIndex (lo)
s_subb_u32 s37, s33, s25                           // CanWriteUpto: upto - cachedHwReadIndex (hi, borrow)
s_cmp_eq_u32 s37, 0                                // diff hi == 0? (gap < 2^32)
s_cbranch_scc0 label_sdma_canwrite_full            // hi != 0 -> gap huge -> full path
s_cmp_lt_u32 s36, 262144                           // diff < queueSize? (fast-path room check)
s_cbranch_scc1 label_sdma_canwrite_ok              // room via cached index
label_sdma_canwrite_full:  /// CanWriteUpto: cache says full -> read rptr
  s_load_dwordx2 s[38:39], s[22:23], 0x8
s_waitcnt lgkmcnt(0)                               // wait rptr pointer load
v_mov_b32 v32, s38                                 // rptr addr lo
v_mov_b32 v33, s39                                 // rptr addr hi
global_load_dwordx2 v[34:35], v[32:33], off sc0 sc1 // load hardware rptr (SYSTEM scope, sc0 sc1)
s_waitcnt vmcnt(0)                                 // wait rptr load
v_readfirstlane_b32 s24, v34                       // cachedHwReadIndex lo = rptr
v_readfirstlane_b32 s25, v35                       // cachedHwReadIndex hi = rptr
s_sub_u32 s36, s32, s24                            // CanWriteUpto: upto - refreshed rptr (lo)
s_subb_u32 s37, s33, s25                           // CanWriteUpto: upto - refreshed rptr (hi, borrow)
s_cmp_eq_u32 s37, 0                                // diff hi == 0?
s_cbranch_scc0 label_sdma_canwrite_done            // hi != 0 -> full (result already defaulted to 0)
s_cmp_lt_u32 s36, 262144                           // diff < queueSize?
s_cbranch_scc1 label_sdma_canwrite_ok              // room after refresh
s_mov_b32 s35, 0                                   // CanWriteUpto = false (full)
s_branch label_sdma_canwrite_done                  // -> done
label_sdma_canwrite_ok:  /// CanWriteUpto: room in ring
s_mov_b32 s35, 1                                   // CanWriteUpto = true (room)
label_sdma_canwrite_done:  /// CanWriteUpto: done
s_cmp_eq_u32 s35, 0                                // CanWriteUpto == false?
s_cbranch_scc1 label_sdma_reserve_loop             // full -> retry
v_mov_b32 v28, s32                                 // swap lo = new
v_mov_b32 v29, s33                                 // swap hi = new
v_mov_b32 v26, s30                                 // cachedWptr addr lo
v_mov_b32 v27, s31                                 // cachedWptr addr hi
global_atomic_cmpswap_x2 v[28:29], v[26:27], v[28:31], off sc0 // CAS cachedWptr cur->new (device scope, return pre-op: sc0)
s_waitcnt vmcnt(0)                                 // wait CAS return
v_readfirstlane_b32 s36, v28                       // CAS pre-op lo
s_cmp_eq_u32 s36, s26                              // CAS pre-op lo == cur lo? (won the slot)
s_cbranch_scc0 label_sdma_reserve_loop             // lost race -> retry
v_readfirstlane_b32 s37, v29                       // CAS pre-op hi
s_cmp_eq_u32 s37, s27                              // pre-op hi == cur hi?
s_cbranch_scc0 label_sdma_reserve_loop             // lost race -> retry
s_branch label_sdma_reserve_done                   // won -> reserved
label_sdma_reserve_done:  /// ReserveQueueSpace: reserved
s_mov_b32 s28, s26                                 // pending = reserved base lo
s_mov_b32 s29, s27                                 // pending = reserved base hi
  s_load_dwordx2 s[30:31], s[22:23], 0x0
s_waitcnt lgkmcnt(0)                               // wait queueBuf pointer load
v_mov_b32 v28, 0                                   // padding NOP value = 0
s_lshr_b32 s33, s21, 2                             // numOffsetDwords = offset / 4
s_cmp_eq_u32 s33, 0                                // no padding?
s_cbranch_scc1 label_sdma_pp_paddone               // offset==0 -> skip pad
label_sdma_pp_padloop:  /// placePacket: zero-pad ring tail
s_and_b32 s32, s28, 262143                         // WrapIntoRing(pending) (pad)
v_mov_b32 v26, s30                                 // queueBuf lo
v_mov_b32 v27, s31                                 // queueBuf hi
v_add_co_u32 v26, vcc, s32, v26                    // addr lo = queueBuf + WrapIntoRing(pending)
v_addc_co_u32 v27, vcc, v27, 0, vcc                // addr hi (carry)
global_store_dword v[26:27], v28, off sc1          // ring[wrap] = 0 padding NOP (AGENT scope, sc1)
s_add_u32 s28, s28, 4                              // pending += 4 (one padded dword)
s_addc_u32 s29, s29, 0                             // pending hi carry
s_sub_u32 s33, s33, 1                              // numOffsetDwords -= 1
s_cmp_eq_u32 s33, 0                                // pad done?
s_cbranch_scc0 label_sdma_pp_padloop               // more padding
label_sdma_pp_paddone:  /// placePacket: padding done
s_and_b32 s32, s28, 262143                         // WrapIntoRing(pending) (packet base)
v_mov_b32 v26, s30                                 // queueBuf lo
v_mov_b32 v27, s31                                 // queueBuf hi
v_add_co_u32 v26, vcc, s32, v26                    // addr lo = queueBuf + WrapIntoRing(pending)
v_addc_co_u32 v27, vcc, v27, 0, vcc                // addr hi (carry)
global_store_dword v[26:27], v5, off sc1           // ring[base + 0] = packet dword 0 (AGENT scope, sc1)
global_store_dword v[26:27], v6, off offset:4 sc1  // ring[base + 1] = packet dword 1 (AGENT scope, sc1)
global_store_dword v[26:27], v7, off offset:8 sc1  // ring[base + 2] = packet dword 2 (AGENT scope, sc1)
global_store_dword v[26:27], v8, off offset:12 sc1 // ring[base + 3] = packet dword 3 (AGENT scope, sc1)
global_store_dword v[26:27], v9, off offset:16 sc1 // ring[base + 4] = packet dword 4 (AGENT scope, sc1)
global_store_dword v[26:27], v10, off offset:20 sc1 // ring[base + 5] = packet dword 5 (AGENT scope, sc1)
global_store_dword v[26:27], v11, off offset:24 sc1 // ring[base + 6] = packet dword 6 (AGENT scope, sc1)
global_store_dword v[26:27], v12, off offset:28 sc1 // ring[base + 7] = packet dword 7 (AGENT scope, sc1)
global_store_dword v[26:27], v13, off offset:32 sc1 // ring[base + 8] = packet dword 8 (AGENT scope, sc1)
global_store_dword v[26:27], v14, off offset:36 sc1 // ring[base + 9] = packet dword 9 (AGENT scope, sc1)
global_store_dword v[26:27], v15, off offset:40 sc1 // ring[base + 10] = packet dword 10 (AGENT scope, sc1)
global_store_dword v[26:27], v16, off offset:44 sc1 // ring[base + 11] = packet dword 11 (AGENT scope, sc1)
global_store_dword v[26:27], v17, off offset:48 sc1 // ring[base + 12] = packet dword 12 (AGENT scope, sc1)
s_add_u32 s28, s28, 52                             // pending += packet size
s_addc_u32 s29, s29, 0                             // pending hi carry
s_mov_b32 s21, 0                                   // ATOMIC follows the COPY: no further padding
  s_load_dwordx2 s[30:31], s[22:23], 0x0
s_waitcnt lgkmcnt(0)                               // wait queueBuf pointer load
v_mov_b32 v28, 0                                   // padding NOP value = 0
s_lshr_b32 s33, s21, 2                             // numOffsetDwords = offset / 4
s_cmp_eq_u32 s33, 0                                // no padding?
s_cbranch_scc1 label_sdma_pp_paddone_1             // offset==0 -> skip pad
label_sdma_pp_padloop_1:  /// placePacket: zero-pad ring tail
s_and_b32 s32, s28, 262143                         // WrapIntoRing(pending) (pad)
v_mov_b32 v26, s30                                 // queueBuf lo
v_mov_b32 v27, s31                                 // queueBuf hi
v_add_co_u32 v26, vcc, s32, v26                    // addr lo = queueBuf + WrapIntoRing(pending)
v_addc_co_u32 v27, vcc, v27, 0, vcc                // addr hi (carry)
global_store_dword v[26:27], v28, off sc1          // ring[wrap] = 0 padding NOP (AGENT scope, sc1)
s_add_u32 s28, s28, 4                              // pending += 4 (one padded dword)
s_addc_u32 s29, s29, 0                             // pending hi carry
s_sub_u32 s33, s33, 1                              // numOffsetDwords -= 1
s_cmp_eq_u32 s33, 0                                // pad done?
s_cbranch_scc0 label_sdma_pp_padloop_1             // more padding
label_sdma_pp_paddone_1:  /// placePacket: padding done
s_and_b32 s32, s28, 262143                         // WrapIntoRing(pending) (packet base)
v_mov_b32 v26, s30                                 // queueBuf lo
v_mov_b32 v27, s31                                 // queueBuf hi
v_add_co_u32 v26, vcc, s32, v26                    // addr lo = queueBuf + WrapIntoRing(pending)
v_addc_co_u32 v27, vcc, v27, 0, vcc                // addr hi (carry)
global_store_dword v[26:27], v18, off sc1          // ring[base + 0] = packet dword 0 (AGENT scope, sc1)
global_store_dword v[26:27], v19, off offset:4 sc1 // ring[base + 1] = packet dword 1 (AGENT scope, sc1)
global_store_dword v[26:27], v20, off offset:8 sc1 // ring[base + 2] = packet dword 2 (AGENT scope, sc1)
global_store_dword v[26:27], v21, off offset:12 sc1 // ring[base + 3] = packet dword 3 (AGENT scope, sc1)
global_store_dword v[26:27], v22, off offset:16 sc1 // ring[base + 4] = packet dword 4 (AGENT scope, sc1)
global_store_dword v[26:27], v23, off offset:20 sc1 // ring[base + 5] = packet dword 5 (AGENT scope, sc1)
global_store_dword v[26:27], v24, off offset:24 sc1 // ring[base + 6] = packet dword 6 (AGENT scope, sc1)
global_store_dword v[26:27], v25, off offset:28 sc1 // ring[base + 7] = packet dword 7 (AGENT scope, sc1)
s_add_u32 s28, s28, 32                             // pending += packet size
s_addc_u32 s29, s29, 0                             // pending hi carry
  s_load_dwordx2 s[30:31], s[22:23], 0x28
s_waitcnt lgkmcnt(0)                               // wait committedWptr pointer load
v_mov_b32 v26, s30                                 // committedWptr addr lo
v_mov_b32 v27, s31                                 // committedWptr addr hi
label_sdma_submit_spin:  /// submitPacket: wait committedWptr == base
s_sleep 1                                          // submitPacket: backoff between polls (must stay INSIDE the spin body)
global_load_dwordx2 v[28:29], v[26:27], off sc1    // load committedWptr (AGENT scope, sc1)
s_waitcnt vmcnt(0)                                 // wait committedWptr load
v_readfirstlane_b32 s32, v28                       // committedWptr lo
s_cmp_eq_u32 s32, s26                              // committedWptr lo == base lo?
s_cbranch_scc0 label_sdma_submit_spin              // not our turn -> spin
v_readfirstlane_b32 s32, v29                       // committedWptr hi
s_cmp_eq_u32 s32, s27                              // committedWptr hi == base hi?
s_cbranch_scc0 label_sdma_submit_spin              // not our turn -> spin
label_sdma_submit_ready:  /// submitPacket: our turn
s_waitcnt vmcnt(0)                                 // ensure our packet stores are globally visible before wptr
v_mov_b32 v28, s28                                 // publish value = pending lo
v_mov_b32 v29, s29                                 // publish value = pending hi
  s_load_dwordx2 s[34:35], s[22:23], 0x10
s_waitcnt lgkmcnt(0)                               // wait wptr pointer load
v_mov_b32 v26, s34                                 // wptr addr lo
v_mov_b32 v27, s35                                 // wptr addr hi
global_store_dwordx2 v[26:27], v[28:29], off sc1   // store wptr = pending (AGENT scope, sc1)
s_waitcnt vmcnt(0)                                 // s_waitcnt vmcnt(0): wptr store visible before doorbell
  s_load_dwordx2 s[34:35], s[22:23], 0x18
s_waitcnt lgkmcnt(0)                               // wait doorbell pointer load
v_mov_b32 v26, s34                                 // doorbell addr lo
v_mov_b32 v27, s35                                 // doorbell addr hi
global_store_dwordx2 v[26:27], v[28:29], off sc0 sc1 // ring doorbell = pending (SYSTEM scope, sc0 sc1)
s_waitcnt vmcnt(0)                                 // wait doorbell store issued
v_mov_b32 v26, s30                                 // committedWptr addr lo
v_mov_b32 v27, s31                                 // committedWptr addr hi
global_store_dwordx2 v[26:27], v[28:29], off sc1   // store committedWptr = pending (AGENT scope, sc1)
s_waitcnt vmcnt(0)                                 // wait committedWptr store issued
// loadKernArg 22 KernArgAddress dword=2 sgprOffset=0x40
// loadKernArg 21 KernArgAddress dword=1 sgprOffset=0x54
s_waitcnt lgkmcnt(0)                               // wait counter_ptr/FusedW
s_mul_i32 s19, s21, s13                            // W * tokenTiles (counter2 base index, past the (p,j) counter)
s_add_u32 s19, s19, s18                            // counter2 index = W*tokenTiles + dst_rank
s_lshl_b32 s19, s19, 2                             // * 4 (u32 counter byte offset)
s_add_u32 s22, s22, s19                            // counter2[dst_rank] lo = counter_ptr + (W*tokenTiles+dst_rank)*4
s_addc_u32 s23, s23, 0                             // counter2[dst_rank] hi (carry)
v_mov_b32 v2, s22                                  // counter2 addr lo -> vgpr
v_mov_b32 v3, s23                                  // counter2 addr hi -> vgpr
v_mov_b32 v1, 1                                    // counter2 increment = 1
global_atomic_add v4, v[2:3], v1, off sc0          // old2 = atomic_add(counter2[dst_rank], 1) device scope, return pre-op (sc0)
s_waitcnt vmcnt(0)                                 // fused-A2A: wait counter2 atomic return
v_readfirstlane_b32 s19, v4                        // old2 -> sgpr
s_add_u32 s19, s19, 1                              // old2 + 1
s_cmp_eq_u32 s19, s13                              // old2+1 == FusedTokenTiles? (this card's last packet to dst_rank)
s_cbranch_scc0 label_fusedA2A_handshake_notlast    // not the last submitter for dst_rank -> skip ahead (inert: counter3 elects the DRAIN owner)
label_fusedA2A_handshake_notlast:  /// fused-A2A: not the last WG for (dst_rank, token-tile) -> skip release
s_branch label_fusedA2A_counter3                   // PUSH path: already elected, skip the local-path election
label_fusedA2A_local_tally:  /// fused-A2A: local WG -> wave-0 election only, then the tally
v_readfirstlane_b32 s21, v[vgprSerial]             // wave 0 elects the WG's single tally writer (local path)
s_cmp_eq_u32 s21, 0                                // wave 0?
s_cbranch_scc0 label_fusedA2A_handshake_after      // non-wave-0 -> skip (single writer per WG)
s_mov_b64 exec, 1                                  // fused-A2A: isolate lane 0 so the tally fires once per WG
label_fusedA2A_counter3:  /// fused-A2A: grid-wide counter3 tally (every WG, PUSH and local)
s_sub_u32 s22, s[sgprFusedTotalWGs], 1             // wrap limit = FusedTotalWGs - 1 (what the last WG reads back)
s_mov_b32 s21, s22                                 // SDATA in = limit; the atomic overwrites it with the pre-op value
s_atomic_inc s21, s[sgprFusedCounter3Ptr:sgprFusedCounter3Ptr+1], 0 glc // old3 = atomic_inc(counter3), wrap at FusedTotalWGs-1, return pre-op
s_waitcnt lgkmcnt(0)                               // fused-A2A: wait counter3 atomic return (SMEM -> lgkmcnt)
s_cmp_eq_u32 s21, s22                              // pre-op == FusedTotalWGs-1? (globally last WG)
s_cbranch_scc0 label_fusedA2A_handshake_after      // not the last WG -> done (only the globally last WG drains)
// loadKernArg 21 KernArgAddress dword=1 sgprOffset=0x5c
s_waitcnt lgkmcnt(0)                               // wait FusedDrain
s_cmp_eq_u32 s21, 0                                // FusedDrain == 0?
s_cbranch_scc1 label_fusedA2A_drain_skip           // FusedDrain==0 -> skip drain barrier
// loadKernArg 21 KernArgAddress dword=1 sgprOffset=0x50
// loadKernArg 26 KernArgAddress dword=1 sgprOffset=0x54
// loadKernArg 27 KernArgAddress dword=1 sgprOffset=0x68
s_waitcnt lgkmcnt(0)                               // wait FusedMyRank/FusedW/FusedTokenTiles
s_lshl_b32 s24, s21, 3                             // rank * 8 (byte offset into peer_ptr[] array)
s_add_u32 s24, s24, 0                              // kernarg offset = fusedBase + peer_ptr_0 + rank*8
// loadKernArg 22 KernArgAddress dword=2 sgprOffset=s24
s_waitcnt lgkmcnt(0)                               // wait peer_ptr[my_rank] load
s_bfm_b64 s[24:25], s26, 0                         // fused-A2A: (1 << W) - 1, one lane per peer flag slot
s_mov_b64 exec, s[24:25]                           // fused-A2A: widen EXEC to W lanes for the DRAIN poll
v_lshlrev_b32 v2, 2, v[vgprSerial]                 // lane j -> self flag slot byte offset j*4
label_fusedA2A_drain_poll:  /// fused-A2A: DRAIN poll all W self flags until each == tokenTiles
global_load_dword v4, v2, s[22:23] sc0 sc1         // poll self flag[lane] low dword (system scope, sc0 sc1)
s_waitcnt vmcnt(0)                                 // fused-A2A: wait poll load
v_cmp_ne_u32 vcc, v4, s27                          // any lane's flag != FusedTokenTiles? (peer still sending)
s_cbranch_vccnz label_fusedA2A_drain_poll          // some peer incomplete -> spin (poll again)
s_mov_b64 exec, 1                                  // fused-A2A: back to lane 0 after the DRAIN poll
label_fusedA2A_drain_skip:  /// fused-A2A: FusedDrain==0 -> no drain barrier
label_fusedA2A_handshake_after:  /// fused-A2A: after handshake (non-wave-0 skips)
s_mov_b64 exec, -1                                 // fused-A2A: restore full exec after single-lane handshake
