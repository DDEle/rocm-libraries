### 5. run-rocdtif.sh hardcode 了 r5.01 path

`~/workspace/rocdtif/run-rocdtif.sh` 内部 hardcode `pkgroot=.../rocdtif-7.13-am+ffmlite-mi400-r5.01/...`。换 r5.04 必须 sed 替换 path 做新 wrapper (`run-rocdtif-r5.04.sh`)。

### 6. fmha_fwd binary 在 AM 上必须 `-timer=cpu` (else 跳过 validation)

`tile_example_fmha_fwd` 默认 `-timer=gpu` → AM 上 HIP gpu timer query 返回负数 → `fmha_fwd_runner.hpp:1648-1652` 的 `if(fwd_ave_time < 0.0f) { std::cout << ", not supported yet"; return no_instance; }` short-circuit → **kernel 真跑了但 validation 被跳过, 拿不到 valid:y/n 也拿不到 max_err**。

→ **AM 跑 fmha 必须显式 `-timer=cpu`** (CPU wall timer 给一个垃圾但正数的 ms 数避开 short-circuit, 让 runner 进 validation block)。

```sh
docker exec -e HIP_VISIBLE_DEVICES=0 yiding12-gfx1250 bash -c "
  source .../enable_perftools.sh 2>/dev/null
  ~/workspace/rocdtif/run-rocdtif-r5.04.sh am <abs-fmha-bin> \
    -prec=fp16 -b=1 -h=1 -s=128 -d=128 -mask=0 -mode=0 \
    -iperm=0 -operm=0 -warmup=0 -repeat=1 -v=1 \
    -timer=cpu -kname=1
"
```

CPU timer 在 AM 上报的 ms 数 (e.g. 7095 ms for s=128 single-head dense) 是 host wall 不是 kernel cycle, **不是 perf 数据**, 仅用作避开 short-circuit + 让 validation 跑出来。

Wall expectation: fmha s=128 single-head dense = ~3-4min wall (vs gemm m=n=k=128 = ~4-5min)。 fmha case A (s=1023) 估 ~60min 没实测过 (一直只跑过 s=128)。

`gemm` example binary 没 `-timer=gpu` short-circuit 问题, default `-timer=gpu` 跑得通 (走另外一段不依赖 ave_time 正负的 perf eval 代码)。这条只 hit fmha_fwd_runner。
