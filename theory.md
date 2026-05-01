# OPD On-Policy Distillation: Theoretical Derivations

## Unified Framework

All on-policy distillation methods can be unified under:

$$ \mathcal{L}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim q}\left[ w(x,y) \cdot \mathcal{D}\big(\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)\big) \right] $$

where the triple $(q, w, \mathcal{D})$ determines the specific method.

---

## 1. OPD (Online Policy Distillation)

### Objective

$$ \mathcal{L}_{\text{OPD}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta}\left[ D_{KL}\big(\pi_{\text{ref}} \| \pi_\theta\big) \right] $$

### Gradient

$$ \nabla_\theta \mathcal{L}_{\text{OPD}}(\theta) = -\mathbb{E}_{x,y \sim \pi_\theta}\left[ \sum_t \nabla_\theta \log \pi_\theta(y_t|x,y_{<t}) \cdot \frac{\pi_{\text{ref}}(y_t|x,y_{<t})}{\pi_\theta(y_t|x,y_{<t})} \right] $$

### Key Insight

On-policy sampling ensures the student learns on its own distribution, avoiding distribution mismatch.

---

## 2. SDPO (Score-based Direct Preference Optimization)

### Objective

$$ \mathcal{L}_{\text{SDPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta}\left[ \frac{1}{2} \left\| \nabla_y \log \pi_\theta(y|x) - \nabla_y \log \pi_{\text{ref}}(y|x) \right\|^2 \right] $$

### Relationship to KL

For small distribution differences:

$$ D_{KL}(p\|q) \approx \frac{1}{2} \int p(x) \|\nabla_x \log p(x) - \nabla_x \log q(x)\|^2 dx $$

### Key Insight

Score matching provides more robust learning in low-probability regions.

---

## 3. OPSD (On-Policy Score Distillation)

### Objective

$$ \mathcal{L}_{\text{OPSD}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\text{ref}}}\left[ \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \cdot \frac{1}{2} \left\| \nabla_y \log \pi_\theta - \nabla_y \log \pi_{\text{ref}} \right\|^2 \right] $$

### Key Insight

Importance ratio $r(x,y) = \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ corrects the distribution mismatch between teacher sampling and student distribution.

---

## 4. SDFT (Score Distillation Fine-Tuning)

### Objective

$$ \mathcal{L}_{\text{SDFT}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta}\left[ \alpha_t \cdot D_F(\pi_\theta \| \pi_{\text{ref}}) + \beta_t \cdot D_{KL}(\pi_{\text{ref}} \| \pi_\theta) \right] $$

### Adaptive Weight

$$ \alpha_t = \sigma\left( \frac{\text{score\_gap}_t - \tau_1}{\tau_2} \right), \quad \beta_t = 1 - \alpha_t $$

where:

$$ \text{score\_gap}_t = \mathbb{E}_{y \sim \pi_\theta}\left[ \left\| \nabla_y \log \pi_\theta(y|x) - \nabla_y \log \pi_{\text{ref}}(y|x) \right\|^2 \right] $$

### Key Insight

When score gap is large → focus on score matching; when small → focus on KL fine-tuning.

---

## Method Summary Table

| Method | $q$ | $w$ | $\mathcal{D}$ |
|--------|:---:|:---:|:---:|
| OPD | $\pi_\theta$ | $1$ | $D_{KL}$ |
| SDPO | $\pi_\theta$ | $1$ | $D_F$ |
| OPSD | $\pi_{\text{ref}}$ | $\frac{\pi_\theta}{\pi_{\text{ref}}}$ | $D_F$ |
| SDFT | $\pi_\theta$ | $(\alpha_t, \beta_t)$ | $D_F + D_{KL}$ |
