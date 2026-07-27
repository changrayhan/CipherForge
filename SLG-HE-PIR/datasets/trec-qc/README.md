# TREC-QC 问题分类数据集

## 来源

TREC-QC（Text REtrieval Conference - Question Classification）源自 TREC 的问答（QA）Track，由 Xin Li 和 Dan Roth 在 COLING'02 上提出。原始数据来自 USENET 新闻组、TREC 主题和约 5,000 个众包问题。

## 数据概览

| 项目 | 数值 |
|:---|:---:|
| 任务类型 | 多分类文本分类 |
| 粗粒度类别 | 6 类 |
| 细粒度类别 | 50 类 |
| 总样本数 | 5,952 |
| 许可证 | Apache-2.0 |

## 文件结构

```
trec-qc/
├── train.jsonl          (1,123 KB / 4,909 条) — 训练集
├── val.jsonl            (124 KB / 543 条)     — 验证集
├── test.jsonl           (107 KB / 500 条)     — 测试集
├── question_types.txt   (1.3 KB)              — 类别说明
└── README.md            (本文件)
```

## 划分方式

从原始训练集（5,452 条）中按**粗粒度标签分层抽样 10%** 作为验证集，保证每个大类的比例一致。原始测试集（500 条）保持不变。

| 粗粒度类别 | 训练集 | 验证集 | 测试集 |
|:---|---:|---:|---:|
| DESC | 1,046 | 116 | 138 |
| ENTY | 1,125 | 125 | 94 |
| ABBR | 78 | 8 | 9 |
| HUM | 1,101 | 122 | 65 |
| NUM | 807 | 89 | 113 |
| LOC | 752 | 83 | 81 |
| **总计** | **4,909** | **543** | **500** |

- 测试集所有 42 个细粒度类别均在训练集或验证集中出现
- 验证集覆盖 46/50 个细粒度类别

## 数据格式

JSON Lines（每行一个 JSON 对象）：

```json
{
  "text":                "How did serfdom develop in and then leave Russia?",
  "label":               0,
  "label_text":          "manner of an action",
  "label_original":      "DESC:manner",
  "label_coarse":        0,
  "label_coarse_text":   "description and abstract concepts",
  "label_coarse_original": "DESC"
}
```

| 字段 | 类型 | 说明 |
|:---|---:|:---|
| `text` | str | 问题文本 |
| `label` | int | 细粒度标签 ID (0-49) |
| `label_text` | str | 细粒度标签文本描述 |
| `label_original` | str | 原始细粒度编码，如 `DESC:manner` |
| `label_coarse` | int | 粗粒度标签 ID (0-5) |
| `label_coarse_text` | str | 粗粒度标签文本描述 |
| `label_coarse_original` | str | 粗粒度编码，如 `DESC` |

## 标签体系

### 粗粒度（6 类）

| ID | 编码 | 描述 | 样本量 |
|:---:|:---:|:---|---:|
| 0 | DESC | description and abstract concepts（描述与抽象概念） | 1,300 |
| 1 | ENTY | entities（实体） | 1,344 |
| 2 | ABBR | abbreviation（缩写） | 95 |
| 3 | HUM | human beings（人类） | 1,288 |
| 4 | NUM | numeric values（数值） | 1,009 |
| 5 | LOC | locations（地点） | 916 |

### 细粒度（50 类）

细粒度标签由 `粗粒度编码:子类名` 构成，例如：
- `DESC:def` — 定义
- `DESC:manner` — 方式
- `ENTY:animal` — 动物
- `HUM:ind` — 个人
- `NUM:date` — 日期
- `LOC:city` — 城市

完整类别列表见 `question_types.txt`。

**注意：** 类别分布呈长尾，15 个细粒度类别的样本数少于 20 条（如 `ENTY:religion` 仅 4 条），在 50 类微调时可能需要关注。

## 数据特征

| 特征 | 统计 |
|:---|---:|
| 文本长度（最短） | 13 字符 |
| 文本长度（最长） | 196 字符 |
| 文本长度（平均） | ~50 字符 |
| 语种 | 英文 |
| 可用任务粒度 | 6 类（粗粒度）或 50 类（细粒度） |

## 许可

Apache-2.0 License

## 引用

```
@inproceedings{li-roth-2002-learning,
  title     = "Learning Question Classifiers",
  author    = "Li, Xin and Roth, Dan",
  booktitle = "Proceedings of COLING 2002",
  year      = 2002,
  address   = "Taipei, Taiwan",
}
```

## 关联数据

同一目录下还有来自 TREC-34 (2025) Product Search and Recommendation Track 的数据：
- `../product-recommendation-2025/` — 产品推荐数据集（S/C 二分类）
