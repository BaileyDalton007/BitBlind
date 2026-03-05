import Levenshtein
import os
import sys
import numpy as np

# Add the ability to import modules from the ocr/utils directory
sys.path.append(os.path.abspath("../utils"))
from image_tools import render_text



def normalize_text(text):
    return ''.join(c.lower() for c in text if c.isalnum())



def compute_loss(
    pred_text,
    truth_text,
    symbol_confs,
    word_count,
    alpha=0.7,
    beta=0.2,
    gamma=0.1
):
    pred_norm = normalize_text(pred_text)
    truth_norm = normalize_text(truth_text)

    # confidence term (primary driver)
    if len(symbol_confs) > 0:
        mean_conf = sum(symbol_confs) / len(symbol_confs)
        conf_term = 1.0 - mean_conf
    else:
        conf_term = 1.0  # nothing detected

    # soft normalized edit distance
    if len(truth_norm) == 0:
        ned = 1.0
    else:
        ned = Levenshtein.distance(pred_norm, truth_norm) / len(truth_norm)

    # soft detection term
    detection_term = 1.0 / (1.0 + word_count)

    loss = (
        alpha * conf_term
        + beta * ned
        + gamma * detection_term
    )

    return loss



def nes_step(
    theta,
    sigma,
    lr,
    n_samples,
    truth_text,
    query_function,
    loss_function,
    rendering_function,
    param_mapping_function,
):
    """
    Performs one NES update step.
    """

    d = theta.shape[0]
    grad_est = np.zeros_like(theta)
    losses = []

    half = n_samples // 2

    for _ in range(half):
        eps = np.random.randn(d)

        # ----- Positive direction -----
        theta_pos = theta + sigma * eps
        params_pos = param_mapping_function(theta_pos)

        img_pos = rendering_function(
            text=truth_text,
            params=params_pos
        )

        res_pos = query_function(img_pos)

        loss_pos = loss_function(
            res_pos["text"],
            truth_text,
            res_pos["symbol_confs"],
            res_pos["word_count"]
        )

        # ----- Negative direction -----
        theta_neg = theta - sigma * eps
        params_neg = param_mapping_function(theta_neg)

        img_neg = render_text(
            text=truth_text,
            params=params_neg
        )

        res_neg = query_function(img_neg)

        loss_neg = compute_loss(
            res_neg["text"],
            truth_text,
            res_neg["symbol_confs"],
            res_neg["word_count"]
        )

        # NES gradient estimate
        grad_est += (loss_pos - loss_neg) * eps

        losses.append(loss_pos)
        losses.append(loss_neg)

    grad_est /= (2 * half * sigma)

    theta_new = theta + lr * grad_est

    return theta_new, np.mean(losses)




def optimize(
    theta_init,
    truth_text,

    query_function,
    loss_function,
    rendering_function,
    param_mapping_function,

    steps=200,
    sigma=0.1,
    lr=0.05,
    n_samples=20,
):
    theta = theta_init.copy()

    for step in range(steps):
        theta, loss = nes_step(
            theta,
            sigma,
            lr,
            n_samples,
            truth_text,
            query_function,
            loss_function,
            rendering_function,
            param_mapping_function
        )

        print(f"Step {step:03d} | Loss: {loss:.4f} | Theta: {theta}")

    return theta