# OPD On-Policy Distillation Evolution

在线策略蒸馏方法演进：从 OPD 到 SDFT

## 方法概览

| 方法 | 全称 | 核心思想 |
|------|------|----------|
| OPD | Online Policy Distillation | KL 散度蒸馏 |
| SDPO | Score-based Direct Preference Optimization | 得分匹配蒸馏 |
| OPSD | On-Policy Score Distillation | 重要性加权得分蒸馏 |
| SDFT | Score Distillation Fine-Tuning | 自适应混合蒸馏 |

## 文件结构

```
├── README.md               # 本文件
├── losses.py               # 四种方法的损失函数实现
├── theory.md               # 理论推导与公式
```

## 公式统一视角

所有方法可以统一表示为：

$$\mathcal{L}(\theta) = \mathbb{E}_{x,y \sim q}\left[ w(x,y) \cdot \mathcal{D}\big(\pi_\theta \| \pi_{\text{ref}}\big) \right]$$

其中 $(q, w, \mathcal{D})$ 三元组决定了具体方法。
