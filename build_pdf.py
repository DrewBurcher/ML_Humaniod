"""
Build methodology PDF from content using fpdf2.
Generates a formatted academic-style PDF with tables, equations (as text), and sections.
"""
import os
from fpdf import FPDF

class MethodologyPDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section(self, title, level=1):
        if level == 1:
            self.set_font("Helvetica", "B", 14)
            self.ln(4)
        elif level == 2:
            self.set_font("Helvetica", "B", 12)
            self.ln(3)
        else:
            self.set_font("Helvetica", "B", 11)
            self.ln(2)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10.5)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def equation(self, text):
        self.set_font("Courier", "", 10)
        self.set_x(self.l_margin + 10)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bullet(self, text, bold_prefix=None):
        self.set_font("Helvetica", "", 10.5)
        x = self.get_x()
        self.set_x(x + 8)
        if bold_prefix:
            self.cell(3, 5.5, "- ")
            self.set_font("Helvetica", "B", 10.5)
            self.write(5.5, bold_prefix + ": ")
            self.set_font("Helvetica", "", 10.5)
            self.multi_cell(0, 5.5, text)
        else:
            self.cell(3, 5.5, "- ")
            self.multi_cell(0, 5.5, text)
        self.ln(0.5)

    def numbered(self, num, text, bold_prefix=None):
        self.set_font("Helvetica", "", 10.5)
        x = self.get_x()
        self.set_x(x + 8)
        if bold_prefix:
            self.cell(8, 5.5, f"{num}. ")
            self.set_font("Helvetica", "B", 10.5)
            self.write(5.5, bold_prefix + ": ")
            self.set_font("Helvetica", "", 10.5)
            self.multi_cell(0, 5.5, text)
        else:
            self.cell(8, 5.5, f"{num}. ")
            self.multi_cell(0, 5.5, text)
        self.ln(0.5)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            w = (self.w - self.l_margin - self.r_margin) / len(headers)
            col_widths = [w] * len(headers)

        # Header
        self.set_font("Helvetica", "B", 9.5)
        self.set_fill_color(230, 230, 230)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()

        # Rows
        self.set_font("Helvetica", "", 9.5)
        for row in rows:
            max_h = 7
            for i, cell in enumerate(row):
                # Estimate if wrapping is needed
                if self.get_string_width(cell) > col_widths[i] - 2:
                    lines = max(1, int(self.get_string_width(cell) / (col_widths[i] - 2)) + 1)
                    max_h = max(max_h, lines * 5.5)

            y_start = self.get_y()
            x_start = self.get_x()
            for i, cell in enumerate(row):
                self.set_xy(x_start + sum(col_widths[:i]), y_start)
                # Draw cell border
                self.rect(x_start + sum(col_widths[:i]), y_start, col_widths[i], max_h)
                self.set_xy(x_start + sum(col_widths[:i]) + 1, y_start + 1)
                self.multi_cell(col_widths[i] - 2, 5, cell)
            self.set_y(y_start + max_h)
        self.ln(3)


def build():
    pdf = MethodologyPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Methodology: Deep Reinforcement Learning", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "for Bipedal Humanoid Locomotion", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # ── 1. Data ──
    pdf.section("1  Methodology")
    pdf.section("1.1  Data", level=2)

    pdf.section("1.1.1  Data Source and Generation", level=3)
    pdf.body_text(
        "Unlike supervised learning tasks that rely on a static, pre-collected dataset, "
        "reinforcement learning generates training data online through agent-environment "
        "interaction. In our setting, the agent (policy network) observes the state of a "
        "simulated humanoid robot, selects an action (target joint positions), and receives "
        "a scalar reward signal. This cycle produces transitions of the form (s_t, a_t, r_t, s_{t+1}) "
        "that serve as training data. Because the data distribution depends on the current policy, "
        "it shifts over the course of training -- a key challenge that both of our chosen algorithms "
        "address in different ways."
    )
    pdf.body_text(
        "Our environment simulates a T1 bipedal humanoid robot in PyBullet, an open-source "
        "rigid-body physics engine. The robot model is defined by a URDF (Unified Robot Description "
        "Format) file exported from SolidWorks CAD software. The T1 model has 23 revolute joints "
        "organized into a head (2 joints), two arms (4 joints each), a waist (1 joint), and two "
        "legs (6 joints each). For the locomotion task, we actuate the 13 joints comprising the "
        "waist and both legs; arm and head joints are locked in place via position control."
    )
    pdf.body_text(
        "The physics simulation runs at 240 Hz, while the policy issues actions at 60 Hz "
        "(4 simulation sub-steps per policy step). Gravity is set to -9.81 m/s^2. Each episode "
        "begins with the robot spawned at a height of 0.85 m in a neutral standing pose and runs "
        "for up to 1,000 policy steps (~16.7 s of simulated time)."
    )

    pdf.section("1.1.2  Observation Space", level=3)
    pdf.body_text("The observation at each timestep is a 36-dimensional continuous vector:")
    pdf.equation(
        "o_t = [ z, phi, theta, psi, x_dot, y_dot, z_dot, "
        "phi_dot, theta_dot, psi_dot, q_1, ..., q_13, q_1_dot, ..., q_13_dot ]"
    )
    pdf.body_text(
        "where z is the torso height, (phi, theta, psi) are Euler angles (roll, pitch, yaw) of "
        "the torso, (x_dot, y_dot, z_dot) and (phi_dot, theta_dot, psi_dot) are the linear and "
        "angular velocities of the torso, and q_i, q_i_dot are the position and velocity of each "
        "actuated joint. All observations are normalized online using a running mean and variance "
        "estimate (VecNormalize), with values clipped to +/-10 standard deviations."
    )

    pdf.section("1.1.3  Action Space", level=3)
    pdf.body_text(
        "The action space is a 13-dimensional continuous vector a_t in [-1, 1]^13, with one "
        "component per actuated joint. Each normalized action is mapped to a target joint position:"
    )
    pdf.equation("q_i_target = q_i_mid + a_i * (q_i_upper - q_i_lower) / 2")
    pdf.body_text(
        "where q_i_mid is the midpoint and q_i_upper, q_i_lower are the joint limits from the "
        "URDF. Target positions are applied via PD position control with gains K_p = 0.2 and "
        "K_d = 0.5, and forces clamped to each joint's maximum torque."
    )

    pdf.section("1.1.4  Reward Function", level=3)
    pdf.body_text(
        "We use a multi-component shaped reward to guide learning. At each timestep the reward is:"
    )
    pdf.equation("r_t = r_vel + r_surv + r_energy + r_orient + r_jlim + r_height + r_zvel")
    pdf.body_text(
        "with a large penalty r_fall = -100 applied on termination. The individual components are:"
    )

    pdf.add_table(
        ["Component", "Formula", "Weight"],
        [
            ["Forward velocity", "1.5 * exp(-2(v_x - 0.5)^2)", "+1.5"],
            ["Survival", "Constant bonus per step", "+2.0"],
            ["Energy", "-0.0033 * mean(|tau_i * q_i_dot|)", "-0.0033"],
            ["Orientation", "-1.0 * (phi^2 + theta^2)", "-1.0"],
            ["Joint limit", "-2.0 * sum(max(|q_bar_i| - 0.5, 0)^2)", "-2.0"],
            ["Height", "1.0 * exp(-5(z - 0.85)^2)", "+1.0"],
            ["Z-velocity", "-0.5 * max(-z_dot, 0)", "-0.5"],
            ["Fall penalty", "Applied at episode termination", "-100"],
        ],
        col_widths=[32, 100, 18],
    )
    pdf.body_text(
        "The forward velocity term uses a Gaussian kernel so that the agent is rewarded for "
        "matching the target speed (0.5 m/s) rather than simply moving as fast as possible. "
        "The energy penalty encourages efficient gaits, the orientation and height terms "
        "incentivize an upright posture, and the joint-limit penalty discourages extreme "
        "configurations."
    )

    pdf.section("1.1.5  Episode Termination", level=3)
    pdf.body_text("An episode terminates early (fallen) if:")
    pdf.bullet("torso height z < 0.35 m or z > 2.0 m")
    pdf.bullet("roll or pitch exceeds 1.2 rad (~69 degrees)")
    pdf.body_text("Otherwise, the episode truncates at 1,000 steps.")

    # ── 2. Models ──
    pdf.section("1.2  Models", level=2)
    pdf.body_text(
        "We compare two state-of-the-art deep reinforcement learning algorithms: Proximal "
        "Policy Optimization (PPO) and Soft Actor-Critic (SAC). Both are trained for 2,000,000 "
        "environment timesteps with identical observation and action spaces."
    )

    pdf.section("1.2.1  Proximal Policy Optimization (PPO)", level=3)
    pdf.body_text(
        "PPO (Schulman et al., 2017) is an on-policy, actor-critic algorithm that updates the "
        "policy using data from the most recent rollouts. It addresses the instability of large "
        "policy updates by clipping the probability ratio between the new and old policies."
    )
    pdf.body_text(
        "The policy pi_theta(a|s) is optimized by maximizing the clipped surrogate objective:"
    )
    pdf.equation(
        "L_CLIP(theta) = E_t[ min( r_t(theta) * A_hat_t,  clip(r_t(theta), 1-eps, 1+eps) * A_hat_t ) ]"
    )
    pdf.body_text(
        "where r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t) is the importance sampling "
        "ratio, A_hat_t is the Generalized Advantage Estimate (GAE), and eps = 0.2 is the clipping "
        "parameter. The advantage is computed using GAE with lambda = 0.95 and discount gamma = 0.99:"
    )
    pdf.equation(
        "A_hat_t = sum_{l=0}^inf (gamma * lambda)^l * delta_{t+l},   "
        "delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)"
    )
    pdf.body_text(
        "A separate value network V_phi(s) is trained to minimize the squared error against "
        "the observed returns. The full loss includes an entropy bonus H[pi_theta] to encourage "
        "exploration:"
    )
    pdf.equation(
        "L(theta, phi) = L_CLIP(theta) - 0.5 * L_VF(phi) + 0.01 * H[pi_theta]"
    )

    pdf.add_table(
        ["Hyperparameter", "Value"],
        [
            ["Learning rate", "3e-4"],
            ["Rollout length (n_steps)", "2,048"],
            ["Mini-batch size", "64"],
            ["Epochs per update", "10"],
            ["Discount (gamma)", "0.99"],
            ["GAE lambda", "0.95"],
            ["Clip range (epsilon)", "0.2"],
            ["Entropy coefficient", "0.01"],
            ["Value function coefficient", "0.5"],
            ["Max gradient norm", "0.5"],
            ["Network architecture", "MLP [256, 256]"],
        ],
        col_widths=[75, 75],
    )

    pdf.section("1.2.2  Soft Actor-Critic (SAC)", level=3)
    pdf.body_text(
        "SAC (Haarnoja et al., 2018) is an off-policy, maximum entropy actor-critic algorithm. "
        "It augments the standard RL objective with an entropy term, encouraging the agent to "
        "find policies that are both high-reward and maximally stochastic:"
    )
    pdf.equation(
        "J(pi) = sum_t E_{(s_t,a_t)~rho_pi} [ r(s_t, a_t) + alpha * H[pi(.|s_t)] ]"
    )
    pdf.body_text(
        "where alpha is the temperature parameter that controls the entropy-reward trade-off."
    )
    pdf.body_text(
        "SAC maintains two Q-networks Q_psi1, Q_psi2 (to mitigate overestimation), a policy "
        "network pi_theta, and target Q-networks. The Q-functions are trained on the soft "
        "Bellman residual:"
    )
    pdf.equation(
        "L(psi_i) = E_{(s,a,r,s')~D} [ (Q_psi_i(s,a) - Q_hat(s,a))^2 ]"
    )
    pdf.equation(
        "Q_hat(s,a) = r + gamma * ( min_j Q_bar_psi_j(s', a_tilde') - alpha * log pi_theta(a_tilde'|s') )"
    )
    pdf.body_text(
        "where D is the replay buffer and Q_bar are the target network parameters. The policy "
        "is updated to minimize:"
    )
    pdf.equation(
        "L(theta) = E_{s~D} [ alpha * log pi_theta(a_tilde|s) - min_j Q_psi_j(s, a_tilde) ]"
    )
    pdf.body_text(
        "The temperature alpha is tuned automatically with target entropy H_bar = -5 "
        "(reflecting the 13-dimensional action space). Target networks are updated via "
        "Polyak averaging with coefficient tau = 0.005."
    )

    pdf.add_table(
        ["Hyperparameter", "Value"],
        [
            ["Learning rate", "3e-4"],
            ["Replay buffer size", "1,000,000"],
            ["Learning starts", "1,000"],
            ["Batch size", "256"],
            ["Soft update coefficient (tau)", "0.005"],
            ["Discount (gamma)", "0.99"],
            ["Entropy coefficient (alpha)", "auto (learned)"],
            ["Target entropy", "-5"],
            ["Network architecture", "MLP [256, 256]"],
        ],
        col_widths=[75, 75],
    )

    pdf.section("1.2.3  Comparison Rationale", level=3)
    pdf.body_text(
        "PPO and SAC represent fundamentally different paradigms in deep RL. PPO is on-policy: "
        "it discards data after each update, which can be sample-inefficient but tends to produce "
        "stable training. SAC is off-policy: it stores all transitions in a replay buffer and "
        "reuses them across many updates, offering better sample efficiency at the cost of "
        "additional memory and potential instability from stale data. By training both for "
        "2,000,000 timesteps under identical conditions, we aim to determine which paradigm "
        "is better suited for high-dimensional humanoid locomotion."
    )

    # ── 3. Implementation ──
    pdf.section("1.3  Implementation", level=2)
    pdf.body_text(
        "Both algorithms are implemented using Stable-Baselines3 (SB3), a widely-used "
        "PyTorch-based reinforcement learning library. The simulation environment is built "
        "on PyBullet and wrapped in a Gymnasium-compatible interface (registered as T1Walking-v0)."
    )
    pdf.body_text(
        "Both the actor and critic networks for each algorithm use a two-hidden-layer MLP "
        "with 256 units per layer and ReLU activations. Training is performed on CPU, as the "
        "small network size and single-environment setup do not benefit from GPU acceleration."
    )
    pdf.body_text("Key implementation details:")
    pdf.bullet(
        "A VecNormalize wrapper maintains a running estimate of observation mean and variance, "
        "normalizing inputs to approximately zero mean and unit variance. Reward normalization "
        "is also applied during training.",
        bold_prefix="Observation normalization"
    )
    pdf.bullet(
        "The policy is evaluated every 5,000 timesteps on a separate environment instance "
        "using 5 deterministic episodes (no exploration noise). The best-performing checkpoint "
        "is saved automatically.",
        bold_prefix="Evaluation"
    )
    pdf.bullet(
        "Full model checkpoints are saved every 50,000 timesteps. For SAC, the replay buffer "
        "is also saved periodically to enable seamless training resumption.",
        bold_prefix="Checkpointing"
    )
    pdf.bullet(
        "Policy outputs in [-1, 1] are linearly mapped to joint position targets within "
        "URDF-defined limits, then applied via PD control.",
        bold_prefix="Action scaling"
    )

    pdf.section("1.3.1  Challenges and Limitations", level=3)
    pdf.bullet(
        "The agent is highly sensitive to relative reward weights. Without careful tuning, "
        "the robot may learn degenerate strategies such as standing still (maximizing survival) "
        "or falling forward (brief velocity reward).",
        bold_prefix="Reward shaping"
    )
    pdf.bullet(
        "Policies trained in simulation may not transfer directly to physical hardware due to "
        "differences in contact dynamics, actuator modeling, and sensor noise.",
        bold_prefix="Sim-to-real gap"
    )
    pdf.bullet(
        "Even at 2,000,000 timesteps, humanoid locomotion is a difficult exploration problem. "
        "The agent must discover coordinated multi-joint gaits in a 13-dimensional continuous "
        "action space.",
        bold_prefix="Sample efficiency"
    )
    pdf.bullet(
        "Both algorithms may converge to suboptimal gaits (e.g., shuffling rather than walking) "
        "that are difficult to escape without curriculum learning or domain randomization.",
        bold_prefix="Local optima"
    )

    # ── 4. Evaluation ──
    pdf.section("1.4  Evaluation Metrics", level=2)
    pdf.body_text(
        "We assess model performance using the following quantitative metrics, measured over "
        "10 deterministic evaluation episodes after training:"
    )
    pdf.numbered(1, "The sum of per-step rewards over a full episode. Higher values indicate better overall locomotion quality.", bold_prefix="Cumulative episode reward")
    pdf.numbered(2, "Total displacement along the x-axis (meters). Directly measures whether the robot walks.", bold_prefix="Forward distance")
    pdf.numbered(3, "Number of steps before termination. Longer episodes indicate the robot avoids falling.", bold_prefix="Episode length")
    pdf.numbered(4, "Ratio of total energy penalty to forward distance traveled (energy per meter). Lower values indicate more efficient gaits.", bold_prefix="Energy efficiency")
    pdf.numbered(5, "Cumulative reward as a function of training timesteps, plotted with smoothing and confidence bands across evaluation checkpoints.", bold_prefix="Learning curve")

    pdf.ln(2)
    pdf.body_text(
        "We expect SAC to exhibit faster initial learning due to its off-policy replay buffer "
        "enabling more data reuse per environment step. PPO may show more stable convergence but "
        "potentially at lower final performance given the fixed 2,000,000 timestep budget. Both "
        "algorithms should produce policies that keep the robot upright and produce forward motion, "
        "though the quality of the learned gait (smoothness, symmetry, efficiency) will vary."
    )
    pdf.body_text(
        "The desired output of each trained model is a policy pi_theta(a|s) that, given the "
        "current robot state, produces joint position targets that result in stable, energy-efficient "
        "forward walking at approximately 0.5 m/s. Success is defined as achieving sustained forward "
        "locomotion without falling for the full 1,000-step episode, with a cumulative reward "
        "significantly above the random-policy baseline."
    )

    # ── References ──
    pdf.section("References", level=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.body_text(
        "[1] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). "
        "Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347."
    )
    pdf.body_text(
        "[2] Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft actor-critic: "
        "Off-policy maximum entropy deep reinforcement learning with a stochastic actor. "
        "Proceedings of the 35th International Conference on Machine Learning (ICML)."
    )

    out_path = os.path.join(os.path.dirname(__file__), "methodology.pdf")
    pdf.output(out_path)
    print(f"PDF saved to: {out_path}")


if __name__ == "__main__":
    build()
