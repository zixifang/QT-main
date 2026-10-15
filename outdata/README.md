# outdata

界面只读这个目录。每个数据集一个子文件夹。

```text
outdata/<dataset_id>/index.json
outdata/<dataset_id>/<SYMBOL>.json
```

当前示例是 `outdata/equity-2018/`。`index.json` 列出股票和算法，每只股票一个 JSON。新算法算完后，把同样结构的文件放进对应的数据集目录。
