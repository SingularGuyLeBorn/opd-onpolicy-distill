"""
OPD On-Policy Distillation Methods - Loss Implementations

Methods covered:
- OPD: Online Policy Distillation (KL divergence)
- SDPO: Score-based Direct Preference Optimization (Fisher divergence)
- OPSD: On-Policy Score Distillation (importance weighted)
- SDFT: Score Distillation Fine-Tuning (adaptive hybrid)
"""

import torch
import torch.nn.functional as F


# ============================================================
# 1. OPD: Online Policy Distillation
# ============================================================

def opd_loss(logits_student, logits_teacher, temperature=1.0):
    """
    OPD Loss: On-Policy KL Divergence Distillation

    L_OPD(theta) = E_{x~D, y~pi_theta}[ D_KL(pi_ref || pi_theta) ]

    Args:
        logits_student: [batch_size, seq_len, vocab_size]
        logits_teacher: [batch_size, seq_len, vocab_size]
        temperature: distillation temperature
    """
    # Temperature scaling
    student_logits = logits_student / temperature
    teacher_logits = logits_teacher / temperature

    # Teacher probabilities (target distribution)
    teacher_probs = F.softmax(teacher_logits, dim=-1)

    # Student log-probabilities
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    # KL divergence: KL(teacher || student)
    kl_loss = F.kl_div(student_log_probs, teacher_probs,
                        reduction='batchmean')

    return kl_loss * (temperature ** 2)


# ============================================================
# 2. SDPO: Score-based Direct Preference Optimization
# ============================================================

def sdpo_loss(logits_student, logits_teacher, temperature=1.0):
    """
    SDPO Loss: Score-based Direct Preference Optimization

    L_SDPO(theta) = E_{x~D, y~pi_theta}[
        0.5 * || grad_y log pi_theta - grad_y log pi_ref ||^2
    ]

    Uses Fisher divergence (score matching) instead of KL divergence.

    Args:
        logits_student: [batch_size, seq_len, vocab_size]
        logits_teacher: [batch_size, seq_len, vocab_size]
        temperature: distillation temperature
    """
    log_prob_s = F.log_softmax(logits_student / temperature, dim=-1)
    log_prob_t = F.log_softmax(logits_teacher / temperature, dim=-1)

    # Score matching loss: MSE of log-probabilities
    # (approximation of score function difference)
    score_mse = F.mse_loss(log_prob_s, log_prob_t, reduction='mean')

    return score_mse


# ============================================================
# 3. OPSD: On-Policy Score Distillation
# ============================================================

def opsd_loss(logits_student, logits_teacher, temperature=1.0):
    """
    OPSD Loss: On-Policy Score Distillation

    L_OPSD(theta) = E_{x~D, y~pi_ref}[
        (pi_theta / pi_ref) * 0.5 * || score_theta - score_ref ||^2
    ]

    Importance-weighted score matching with correction ratio.

    Args:
        logits_student: [batch_size, seq_len, vocab_size]
        logits_teacher: [batch_size, seq_len, vocab_size]
        temperature: distillation temperature
    """
    log_prob_s = F.log_softmax(logits_student / temperature, dim=-1)
    log_prob_t = F.log_softmax(logits_teacher / temperature, dim=-1)

    # Importance ratio: pi_theta / pi_ref
    importance_ratio = torch.exp(log_prob_s - log_prob_t).detach()

    # Score difference (squared)
    score_diff = (log_prob_s - log_prob_t) ** 2

    # Weighted loss
    weighted_loss = (importance_ratio * score_diff).mean()

    return weighted_loss


# ============================================================
# 4. SDFT: Score Distillation Fine-Tuning
# ============================================================

def sdft_loss(logits_student, logits_teacher, temperature=1.0,
              tau1=0.5, tau2=0.1):
    """
    SDFT Loss: Score Distillation Fine-Tuning

    L_SDFT(theta) = E_{x~D, y~pi_theta}[
        alpha_t * D_F(pi_theta || pi_ref) + beta_t * D_KL(pi_ref || pi_theta)
    ]

    Adaptive fusion of KL divergence and score matching.
    alpha_t = sigmoid((score_gap - tau1) / tau2)
    beta_t = 1 - alpha_t

    Args:
        logits_student: [batch_size, seq_len, vocab_size]
        logits_teacher: [batch_size, seq_len, vocab_size]
        temperature: distillation temperature
        tau1: threshold for adaptive weight
        tau2: temperature for sigmoid
    """
    teacher_probs = F.softmax(logits_teacher / temperature, dim=-1)
    log_probs_s = F.log_softmax(logits_student / temperature, dim=-1)
    log_probs_t = F.log_softmax(logits_teacher / temperature, dim=-1)

    # Score gap
    score_gap = ((log_probs_s - log_probs_t) ** 2).mean().detach()

    # Adaptive weights
    alpha = torch.sigmoid((score_gap - tau1) / tau2)
    beta = 1.0 - alpha

    # KL divergence term
    kl_term = F.kl_div(log_probs_s, teacher_probs, reduction='batchmean')

    # Score matching term
    score_term = ((log_probs_s - log_probs_t) ** 2).mean()

    # Hybrid loss
    loss = alpha * score_term + beta * kl_term * (temperature ** 2)

    meta = {
        "alpha": alpha.item(),
        "beta": beta.item(),
        "score_gap": score_gap.item()
    }

    return loss, meta


# ============================================================
# Unified Framework
# ============================================================

METHODS = {
    "OPD": opd_loss,
    "SDPO": sdpo_loss,
    "OPSD": opsd_loss,
    "SDFT": sdft_loss,
}


def get_loss(method_name, logits_student, logits_teacher, **kwargs):
    """
    Unified interface for all distillation methods.

    L(theta) = E_{x,y~q}[ w(x,y) * D(pi_theta || pi_ref) ]

    Args:
        method_name: one of "OPD", "SDPO", "OPSD", "SDFT"
        logits_student: student model logits
        logits_teacher: teacher model logits
        **kwargs: additional arguments for specific methods

    Returns:
        loss tensor (and meta dict for SDFT)
    """
    if method_name not in METHODS:
        raise ValueError(f"Unknown method: {method_name}. "
                         f"Available: {list(METHODS.keys())}")

    loss_fn = METHODS[method_name]
    return loss_fn(logits_student, logits_teacher, **kwargs)


if __name__ == "__main__":
    # Quick test
    batch_size, seq_len, vocab_size = 2, 4, 16
    student = torch.randn(batch_size, seq_len, vocab_size)
    teacher = torch.randn(batch_size, seq_len, vocab_size)

    print(f"{'Method':<8} {'Loss':<10} {'Alpha':<8} {'Beta':<8}")
    print("-" * 40)

    for name, fn in METHODS.items():
        result = fn(student, teacher)
        if isinstance(result, tuple):
            loss, meta = result
            print(f"{name:<8} {loss.item():<10.4f} "
                  f"{meta['alpha']:<8.4f} {meta['beta']:<8.4f}")
        else:
            print(f"{name:<8} {result.item():<10.4f}")
