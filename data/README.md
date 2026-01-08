
# 数据目录与命名规范

每个数据集在 `data/` 下建立同名子目录，文件使用统一前缀与后缀，示例（以 `wiki1m` 为例）：

- data/
	- wiki1m/
		- wiki1m_base.fvecs
		- wiki1m_query.fvecs
		- wiki1m_groundtruth.ivecs
		- wiki1m_centroid_10000.fvecs
		- wiki1m_cluster_id_10000.ivecs

## 文件含义与格式
- base.fvecs: 训练/检索库向量，fvecs 格式（每向量前置 `int32 dim`，后随 `dim` 个 `float`）。
- query.fvecs: 查询向量，fvecs 格式（同上）。
- groundtruth.ivecs: 每个查询的真值最近邻 ID 列表，ivecs 格式（每行前置 `int32 K`，后随 `K` 个 `int32`）。
- centroid_C.fvecs: 粗量化质心（C 个），fvecs 格式，每向量长度 = 维度。
- cluster_id_C.ivecs: 向量到簇的映射，ivecs 格式（每行 `dim=1`，单个 `int32` 代表该向量所属簇 ID）。

## 运行示例
使用简化版 CLI，只需传入数据集名称与簇数 C：

```bash
./build/ivf_ann_test wiki1m 10000
```

程序会自动查找以下文件：
- data/wiki1m/wiki1m_base.fvecs
- data/wiki1m/wiki1m_query.fvecs
- data/wiki1m/wiki1m_groundtruth.ivecs（若存在则计算 recall@K）
- data/wiki1m/wiki1m_centroid_10000.fvecs
- data/wiki1m/wiki1m_cluster_id_10000.ivecs（若存在则按映射分配；否则按最近质心分配）

## 可选参数与默认值
- 量化位数 `numBits`: 默认 9。
- `nprobe`: 默认 200（查询时探测的簇数）。
- `topK`: 默认 100（返回的候选数）。

如需调整这些默认值，可在后续版本中扩展命令行或改源码常量。