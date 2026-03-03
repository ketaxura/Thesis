import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import mpcc


def subplots(x_state_hist, u_hist, cont_err_hist, lag_err_hist, mu_hist):
        
    steps = np.arange(len(cont_err_hist))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # -----------------------------
    # Top Left — States
    # -----------------------------
    axs[0, 0].plot(steps, x_state_hist[:, 0], label="x")
    axs[0, 0].plot(steps, x_state_hist[:, 1], label="y")
    axs[0, 0].plot(steps, x_state_hist[:, 2], label="theta")
    axs[0, 0].set_title("State Evolution")
    axs[0, 0].set_xlabel("Step k")
    axs[0, 0].set_ylabel("State Value")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # -----------------------------
    # Top Right — Controls
    # -----------------------------
    axs[0, 1].plot(steps, u_hist[:, 0], label="v")
    axs[0, 1].plot(steps, u_hist[:, 1], label="omega")
    axs[0, 1].set_title("Control Inputs")
    axs[0, 1].set_xlabel("Step k")
    axs[0, 1].set_ylabel("Control Value")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # -----------------------------
    # Bottom Left — Errors
    # -----------------------------
    axs[1, 0].plot(steps, cont_err_hist, label="Contouring Error")
    axs[1, 0].plot(steps, lag_err_hist, label="Lag Error")
    axs[1, 0].set_title("MPCC Errors")
    axs[1, 0].set_xlabel("Step k")
    axs[1, 0].set_ylabel("Error")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # -----------------------------
    # Bottom Right — (Optional empty or extra metric)
    # -----------------------------
    axs[1, 1].plot(steps, mu_hist, label="s progress")
    axs[1, 1].set_title("Progress Variable (s)")
    axs[1, 1].set_xlabel("Step k")
    axs[1, 1].set_ylabel("Progress")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()