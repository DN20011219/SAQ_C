# 项目说明

此项目是 SAQ 量化算法中 CAQ 量化算法的 C 语言风格版本。算法原仓库为：https://github.com/howarlii/SAQ

## 运行测试说明

### 宏参数说明：

1. QUERY_QUANTIZER_PROFILE
   - 含义：开启 Query 量化阶段的轻量级性能分析（计时钩子）。默认关闭。
   - 影响：仅增加少量计时开销，不改变算法结果。
   - 配置方式：在 `query_quantizer.h` 中取消注释 `#define QUERY_QUANTIZER_PROFILE`。
     - 也可用编译器宏开启，但为避免重定义告警，建议直接编辑头文件。

2. **QUERY_QUANTIZER_NUM_BITS（重要）**
   - 含义：Query 量化位数（≤ 8），与数据库编码进行匹配以加速距离计算。仓库默认 `6` 位。
   - 取值建议：
     - 4：更快，精度略降；
     - 6：速度/精度均衡（默认，但在部分数据集上难以实现 recall@P99 ）；
     - 8：更高精度，速度略慢（可实现 recall@P99 ）。
   - 配置方式：在 `query_quantizer.h` 顶部修改对应的宏定义（示例有 4/6/8 三档）。
     - 注意：该宏在头文件中直接定义，使用 `-DQUERY_QUANTIZER_NUM_BITS=...` 可能产生重定义告警，推荐直接编辑头文件。

3. Householder_Random_Orthogonal_Matrix 与 Haar_Random_Orthogonal_Matrix
   - 含义：正交矩阵生成方式的二选一开关（见 `rotator.h`）。
   - 对比：
     - Householder（默认）：L2 范数误差更低（~1e-6），旋转幅度较小；
     - Haar：服从 Haar 分布更均匀，L2 误差略高（~1e-5）。
   - 配置方式：编辑 `rotator.h` 顶部宏，注释/取消注释其中一项。
     - 或使用编译器宏（需同时“去定义”另一项）：
       ```bash
       cmake -DCMAKE_C_FLAGS="-DHaar_Random_Orthogonal_Matrix -UHouseholder_Random_Orthogonal_Matrix" ..
       ```

4. ADJUST_ROUND_LIMIT 及 ADJUST_EPSILON
   - 含义：编码器（`encoder.h`）中量化后“编码调整”过程的控制参数。
     - `ADJUST_ROUND_LIMIT`：最大调整轮数（默认 6），更大值→更强的编码优化→更慢；
     - `ADJUST_EPSILON`：调整判定阈值（默认 1e-8），更小值→更敏感→可能更慢但略提精度。
   - 配置方式：在 `encoder.h` 顶部修改宏定义。
     - 若用编译器宏覆盖，需确保不与头文件值冲突以避免重定义告警。

### 编译：

```bash
set -e
rm -rf build && mkdir -p build && cd build
cmake -DSIMD=AVX ..
cmake -DSIMD=NEON ..
cmake -DSIMD=NONE .. # 或弃用 SIMD 优化
cmake --build . -j
rm -rf build && mkdir -p build && cd build && cmake -DSIMD=AVX .. && cmake --build . -j && cd ../
cd ../
```

本项目预计支持 AVX 和 NEON 两个 SIMD 版本，以实现低配机器可运行。

由于 _mm256_shuffle_ps 仅支持静态查表，因此无法支持 FastScan ，只能选择单向量计算。

此外，针对是否使用分离存储，本项目提供了两种量化和估算路径：

* estimator.h 文件使用分离存储版本进行距离估算。此外，为了加速计算，该版本必须对 query 进行量化操作。

* estimator_easy.h 文件使用非分离存储版本进行距离估算。为了尽可能减少精度影响，该版本不支持对 query 进行量化，以体现真实的量化精度。

### 运行

```bash
./build/encoder_example
./build/rotator_example
./build/estimator_example 100 100 256 1234 1                  # data数量 query数量(用于控制transpose成本) 维度 随机种子 重复次数，B固定为9
./build/estimator_easy_example 10000 1 256 1234 1 8           # data数量 query数量 维度 随机种子 重复次数 B
./estimator_easy_zero_centroid_example 10000 1 256 1234 1 8   # data数量 query数量 维度 随机种子 重复次数 B

# dataset C [numBits] [nprobe] [topK]（详细定义请阅读data目录下的README.md）
./build/ivf_ann_test wiki1m 5533 9 600 100

# u8u8 内积跨平台调优
./build/u8u8_perf
```

预计结果：

```bash
(base) dn@ubun:~/projects/SAQ_C$ ./build/estimator 2000 1 256 1234 10
Data count: 2000
Query count: 1
Loop count: 10
Dim: 256, numBits: 9
Seed: 1234
Init time (sum): 0.050 ms
Estimate time (sum): 8.384 ms
Rest estimate time (sum): 0.442 ms
Loop unstable pairs: 0
Loop max abs diff: 0.000000
Rest mean abs error: 0.061395
Rest mean rel error: 0.000351 # 1+8 bit 估算的平均相对误差
Rest max abs error: 0.317078
Rest max rel error: 0.001731
Mean abs error: 16.058422 # 1bit估算的平均绝对误差
Mean rel error: 0.091338  # 1bit估算的平均相对误差
Max abs error: 42.528992  # 1bit估算的最大绝对误差
Max rel error: 0.254242   # 1bit估算的最大相对误差
```

IVF 测试预计结果:

```bash
(base) dn@ubun:~/projects/SAQ_C$ ./build/ivf_ann_test wiki1m 5533 9 700 100
Config:
  dataset=wiki1m C=5533 numBits=9 nprobe=700 topK=100
Paths:
  base=data/wiki1m/wiki1m_base.fvecs
  query=data/wiki1m/wiki1m_query.fvecs
  groundtruth=data/wiki1m/wiki1m_groundtruth.ivecs
  centroids=data/wiki1m/wiki1m_centroid_5533.fvecs
  assignments=data/wiki1m/wiki1m_cluster_id_5533.ivecs
Loaded:
  base: N=1000000 D=384
  queries: N=14722 D=384
  centroids: K=5533 D=384
Assignments loaded: N=1000000 D=1
Building IVF using assignments mapping.
Building IVF: 1000000/1000000 (100.0%)
IVF built: N=1000000 K=5533 D=384 in 28884.678 ms
  recall@100 = 0.9964
Queries: 147/14722 (1.0%)  recall@100 = 0.9950
Queries: 294/14722 (2.0%)  recall@100 = 0.9937
Queries: 441/14722 (3.0%)  recall@100 = 0.9937
Queries: 588/14722 (4.0%)  recall@100 = 0.9930
Queries: 735/14722 (5.0%)  recall@100 = 0.9925
Queries: 882/14722 (6.0%)  recall@100 = 0.9923
Queries: 1029/14722 (7.0%)  recall@100 = 0.9922
Queries: 1176/14722 (8.0%)  recall@100 = 0.9917
Queries: 1323/14722 (9.0%)  recall@100 = 0.9907
Queries: 1470/14722 (10.0%)  recall@100 = 0.9899
Queries: 1617/14722 (11.0%)  recall@100 = 0.9900
Queries: 1764/14722 (12.0%)  recall@100 = 0.9899
Queries: 1911/14722 (13.0%)  recall@100 = 0.9899
Queries: 2058/14722 (14.0%)  recall@100 = 0.9901
Queries: 2205/14722 (15.0%)  recall@100 = 0.9903
Queries: 2352/14722 (16.0%)  recall@100 = 0.9906
Queries: 2499/14722 (17.0%)  recall@100 = 0.9906
```

# 移植差异说明

## 1. 编码器差异

注意：encoder 部分可能与 SAQ 量化结果存在微弱区别，原因主要是，浮点加法使用结合律可能会吞掉部分精度，而 CPP 版本 double vec_sum = o.sum() 使用了 Eigen 的 SIMD 加法，使用了浮点结合律，因此可能会有微小差异。具体代码：

```C
    originalVectorSum = static_cast<double>(sum);
```

区别如下示例：

```bash
原始：
vec_sum: 0.102713
ip_o_code: 858.692, code_l2sqr: 25802011, code_sum: 98193
  v_mi: -0.307509, v_mx: 0.307509, delta: 0.00120121
  ip_o_oa: 0.999945
  oa_l2sqr: 0.999936
新：
originalVectorSum: 0.102712
oriQuantCodeIp: 858.692, quantCodeL2Sqr: 25802011, quantCodeSum: 98193
  Max: 0.307509, Min: -0.307509, Delta: 0.00120121
  OriginalQuantizedIp: 0.999945
  QuantizedL2Sqr: 0.999936
  OrigL2Sqr: 1
Mismatch at index 182: 177 != 176 Code mismatch for vector 653
Value mismatch for vector 653
Expected: Max=1, Min=-1, Delta=0.00390625, QuantizedL2Sqr=10.5744, OriginalQuantizedIp=3.25176, OrigL2Sqr=1, OrigL2Norm=1, RescaleFactor=0.307526
Got: Max=1, Min=-1, Delta=0.00390625, QuantizedL2Sqr=10.572, OriginalQuantizedIp=3.25138, OrigL2Sqr=1, OrigL2Norm=1, RescaleFactor=0.307561
Mismatch at index 335: 323 != 322 Mismatch at index 336: 209 != 210 Mismatch at index 365: 269 != 268 Code mismatch for vector 7226
Value mismatch for vector 7226
Expected: Max=1, Min=-1, Delta=0.00390625, QuantizedL2Sqr=8.55286, OriginalQuantizedIp=2.92443, OrigL2Sqr=1, OrigL2Norm=1, RescaleFactor=0.341947
Got: Max=1, Min=-1, Delta=0.00390625, QuantizedL2Sqr=8.5567, OriginalQuantizedIp=2.92509, OrigL2Sqr=1, OrigL2Norm=1, RescaleFactor=0.34187
Mismatch at index 107: 217 != 218 Code mismatch for vector 7994
Value mismatch for vector 7994
Expected: Max=1, Min=-1, Delta=0.00390625, QuantizedL2Sqr=13.7857, OriginalQuantizedIp=3.71285, OrigL2Sqr=1, OrigL2Norm=1, RescaleFactor=0.269335
Got: Max=1, Min=-1, Delta=0.00390625, QuantizedL2Sqr=13.7869, OriginalQuantizedIp=3.713, OrigL2Sqr=1, OrigL2Norm=1, RescaleFactor=0.269324
Total code mismatches: 3
Total value mismatches: 3
```

## 2. 正交矩阵差异

由于 SAQ 使用的 Eigen 难以直接嵌入 C 环境，我们重写了一版本正交矩阵生成算法，提供了两个版本用于测试。

默认使用 Householder_Random_Orthogonal_Matrix 以贴近 Eigen 版本的生成算法。

在 GIST 数据集上测试结果与 SAQ 的 Eigen::HouseholderQR 随机正交矩阵效果接近，下面是 GIST 数据集上的 IvfErrorTestL2Sqr.SAQ_GIST_AllBits 测试结果：

```bash
原始：
SAQ 8-bit (gist)    | Error Acc: 4.00437e-05 (diff=0.03%)    Fast: 1.1454e-01 (diff=0.37%)      Vars: 8.3301e-01
Householder_Random_Orthogonal_Matrix：
SAQ 8-bit (gist)    | Error Acc: 4.00800e-05 (diff=0.12%)    Fast: 1.1457e-01 (diff=0.39%)      Vars: 8.3301e-01

原始：
SAQ 4-bit (gist)    | Error Acc: 5.77080e-04 (diff=-0.01%)   Fast: 1.4479e-01 (diff=0.19%)      Vars: 7.6837e-01
Householder_Random_Orthogonal_Matrix：
SAQ 4-bit (gist)    | Error Acc: 5.77456e-04 (diff=0.05%)    Fast: 1.4492e-01 (diff=0.27%)      Vars: 7.6837e-01

原始：
SAQ 1-bit (gist)    | Error Acc: 5.88650e-03 (diff=0.02%)    Fast: 1.5131e-01 (diff=0.12%)      Vars: 7.5919e-01
Householder_Random_Orthogonal_Matrix：
SAQ 1-bit (gist)    | Error Acc: 5.88780e-03 (diff=0.04%)    Fast: 1.5130e-01 (diff=0.12%)      Vars: 7.5919e-01
```

## 3. Query 编码器差异

经过测试，本 query 量化器与原始 SAQ 仓库中量化结果完全一致。

## 4. 估算器差异

对于精确距离估算器（即 1 + 8 bit 估算器），不同于原始仓库中使用无量化版本query进行距离重算：

```c++
float ip_oa1_q = utils::mask_ip_x0_q(curr_query_.data(), short_code, num_dim_padded_);
```

我们使用量化后的 query 与 short_code 的内积缓存作为 1bit 内积数据源，因此存在少量差异。

此外对于原始仓库中使用 float query 与 database vector 量化编码直接内积：

```c++
IP_FUNC(curr_query_.data(), long_code, num_dim_padded_);
```

我们使用量化后的 query 与 database vector 量化编码做内积，因此可能存在少量差异。