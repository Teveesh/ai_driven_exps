"""
Experiment 07 - Video Activity Recognition Using RNN and LSTM
=============================================================
Recognizes activity (Running / Fast Movement vs. Walking / Normal
Movement) from a video using temporal visual features and two recurrent
networks - an RNN and an LSTM - both implemented manually with NumPy.

Pipeline:
    traffic.mp4
        -> 20 equally spaced frames (BGR -> RGB)
        -> visual features (mean, standard deviation, edge density, motion)
        -> min-max normalization
        -> 64-D feature sequence (np.tile expansion)
        -> SimpleRNN  (20, 64) -> (20, 32)
        -> SimpleLSTM (20, 64) -> (20, 32)
        -> activity classification + visualization
        -> saved .npy features (outputs/)

Only OpenCV, NumPy and Matplotlib are used (no deep-learning framework).
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# traffic.mp4 lives in the same directory as this script, so the path is
# built from __file__ and works regardless of the current working directory.
VIDEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traffic.mp4")

NUM_FRAMES = 20          # number of equally spaced frames to sample
FEATURE_SIZE = 64        # dimension of each time step after feature expansion
RNN_HIDDEN_SIZE = 32     # hidden units of the SimpleRNN
LSTM_HIDDEN_SIZE = 32    # hidden units of the SimpleLSTM
RANDOM_SEED = 42         # seed for reproducible random weights

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


# ---------------------------------------------------------------------------
# 1. Video handling
# ---------------------------------------------------------------------------
def check_video(video_path):
    """Check that the video file exists and can be opened.

    Returns an opened ``cv2.VideoCapture`` object.
    Raises FileNotFoundError / IOError with a friendly message otherwise.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video not found: {video_path}\n"
            "Place 'traffic.mp4' in the same directory as 7exp.py."
        )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise IOError(f"Could not open video file: {video_path}")
    return cap


def _count_frames(video_path):
    """Fallback frame counter for files with an unreliable frame-count tag."""
    cap = cv2.VideoCapture(video_path)
    total = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        total += 1
    cap.release()
    return total


def get_video_info(video_path):
    """Read FPS, width, height, total frames and duration of the video."""
    cap = check_video(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Some encoders report 0 / negative frames -> count them manually.
    if total_frames <= 0:
        total_frames = _count_frames(video_path)

    duration = total_frames / fps if fps > 0 else 0.0

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration": duration,
    }


def extract_frames(video_path, num_frames=NUM_FRAMES):
    """Extract ``num_frames`` equally spaced frames and convert BGR -> RGB.

    Frame indices are spread evenly across the whole video (np.linspace)
    so the sampled sequence represents the full clip duration.
    """
    cap = check_video(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total <= 0:
        total = _count_frames(video_path)
    if total < num_frames:
        raise ValueError(
            f"Video has only {total} frames, but {num_frames} are required."
        )

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    wanted = set(int(i) for i in indices)

    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in wanted:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            wanted.remove(frame_idx)
        frame_idx += 1
    cap.release()

    if len(frames) < num_frames:
        raise ValueError(
            f"Could only extract {len(frames)}/{num_frames} frames from the video."
        )
    return frames
# ---------------------------------------------------------------------------
# 2. Visual feature extraction, normalization and expansion
# ---------------------------------------------------------------------------
def extract_frame_features(frames):
    """Compute 4 visual features per frame from the sampled RGB frames.

    Each frame is converted to grayscale first (as in the experiment), then:

      1. Mean intensity          - overall brightness
      2. Standard deviation      - contrast / texture variation
      3. Edge density (Canny)    - fraction of edge pixels
      4. Motion (frame diff)     - mean absolute difference vs previous frame

    Returns an array of shape (num_frames, 4).
    """
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]

    features = []
    for i, g in enumerate(gray):
        mean_intensity = float(np.mean(g))
        std_intensity = float(np.std(g))

        edges = cv2.Canny(g, 100, 200)
        edge_density = float(np.count_nonzero(edges) / edges.size)

        if i == 0:
            motion = 0.0  # no previous frame available
        else:
            motion = float(np.mean(cv2.absdiff(gray[i], gray[i - 1])))

        features.append([mean_intensity, std_intensity, edge_density, motion])

    return np.array(features, dtype=np.float64)  # (20, 4)


def normalize_features(features):
    """Min-max normalize each feature column into the [0, 1] range.

        (features - min) / (max - min + 1e-8)
    """
    min_vals = features.min(axis=0)
    max_vals = features.max(axis=0)
    return (features - min_vals) / (max_vals - min_vals + 1e-8)


def expand_features(features, feature_size=FEATURE_SIZE):
    """Expand the (num_frames, 4) normalized features to (num_frames, 64).

    The 4 features are repeated/tiled using np.tile so that each of the 64
    dimensions is a copy of one of the 4 visual features (16 ticks each).
    """
    tiles = feature_size // features.shape[1]  # 64 / 4 = 16
    return np.tile(features, (1, tiles))       # (20, 4) -> (20, 64)
# ---------------------------------------------------------------------------
# 3. SimpleRNN implemented from scratch with NumPy
# ---------------------------------------------------------------------------
class SimpleRNN:
    """Minimal RNN:  h_t = tanh(x_t @ Wx + h_{t-1} @ Wh + b)."""

    def __init__(self, input_size, hidden_size, seed=RANDOM_SEED):
        self.input_size = input_size
        self.hidden_size = hidden_size

        np.random.seed(seed)
        # Small random initializations avoid saturating the tanh.
        self.Wx = np.random.randn(input_size, hidden_size) * 0.01
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b = np.zeros((1, hidden_size))
        self.hidden_states = None

    def forward(self, x):
        """Forward pass.

        Input:  x of shape (seq_len, input_size)     e.g. (20, 64)
        Output: hidden states (seq_len, hidden_size) e.g. (20, 32)
        """
        seq_len = x.shape[0]
        h = np.zeros((1, self.hidden_size))
        outputs = []

        for t in range(seq_len):
            h = np.tanh(x[t:t + 1] @ self.Wx + h @ self.Wh + self.b)
            outputs.append(h)

        self.hidden_states = np.vstack(outputs)
        return self.hidden_states


# ---------------------------------------------------------------------------
# 4. SimpleLSTM implemented from scratch with NumPy
# ---------------------------------------------------------------------------
def sigmoid(x):
    """Numerically stable sigmoid (clipped to avoid exp overflow)."""
    x = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-x))


class SimpleLSTM:
    """Minimal LSTM with the standard four gates.

        f    = sigmoid(x @ Wf + h @ Uf + bf)      forget gate
        i    = sigmoid(x @ Wi + h @ Ui + bi)      input gate
        c~   = tanh(x @ Wc + h @ Uc + bc)         candidate cell state
        o    = sigmoid(x @ Wo + h @ Uo + bo)      output gate
        c    = f * c + i * c~                     cell state update
        h    = o * tanh(c)                        hidden state
    """

    def __init__(self, input_size, hidden_size, seed=RANDOM_SEED):
        self.input_size = input_size
        self.hidden_size = hidden_size

        rng = np.random.RandomState(seed)
        scale = 0.01  # small init keeps signals away from saturating regions

        # Forget gate
        self.Wf = rng.randn(input_size, hidden_size) * scale
        self.Uf = rng.randn(hidden_size, hidden_size) * scale
        self.bf = np.zeros((1, hidden_size))

        # Input gate
        self.Wi = rng.randn(input_size, hidden_size) * scale
        self.Ui = rng.randn(hidden_size, hidden_size) * scale
        self.bi = np.zeros((1, hidden_size))

        # Candidate cell state
        self.Wc = rng.randn(input_size, hidden_size) * scale
        self.Uc = rng.randn(hidden_size, hidden_size) * scale
        self.bc = np.zeros((1, hidden_size))

        # Output gate
        self.Wo = rng.randn(input_size, hidden_size) * scale
        self.Uo = rng.randn(hidden_size, hidden_size) * scale
        self.bo = np.zeros((1, hidden_size))

        self.hidden_states = None
        self.cell_states = None

    def forward(self, x):
        """Forward pass.

        Input:  x of shape (seq_len, input_size)     e.g. (20, 64)
        Output: hidden states (seq_len, hidden_size) e.g. (20, 32)
        """
        seq_len = x.shape[0]
        h = np.zeros((1, self.hidden_size))
        c = np.zeros((1, self.hidden_size))
        hidden_out, cell_out = [], []

        for t in range(seq_len):
            xt = x[t:t + 1]

            f = sigmoid(xt @ self.Wf + h @ self.Uf + self.bf)
            i = sigmoid(xt @ self.Wi + h @ self.Ui + self.bi)
            c_tilde = np.tanh(xt @ self.Wc + h @ self.Uc + self.bc)
            o = sigmoid(xt @ self.Wo + h @ self.Uo + self.bo)

            c = f * c + i * c_tilde
            h = o * np.tanh(c)

            hidden_out.append(h)
            cell_out.append(c)

        self.hidden_states = np.vstack(hidden_out)
        self.cell_states = np.vstack(cell_out)
        return self.hidden_states
# ---------------------------------------------------------------------------
# 5. Activity classification (demonstration rule from the experiment)
# ---------------------------------------------------------------------------
def classify_activity(motion_values, lstm_output):
    """Classify the activity using the experiment's demonstration rule.

    - average_motion      : mean of the normalized motion feature
    - temporal_variation  : mean absolute change between consecutive LSTM
                            hidden states (how fast the representation shifts)
    - score               : average_motion + temporal_variation * 50
    - score > 5           -> "Running / Fast Movement"
    - otherwise           -> "Walking / Normal Movement"

    This is a heuristic/demonstration classifier, not a trained model.
    """
    average_motion = float(np.mean(motion_values))
    temporal_variation = float(np.mean(np.abs(np.diff(lstm_output, axis=0))))

    score = average_motion + temporal_variation * 50

    if score > 5:
        activity = "Running / Fast Movement"
    else:
        activity = "Walking / Normal Movement"

    return activity, score, average_motion, temporal_variation


# ---------------------------------------------------------------------------
# 6. Visualization (Matplotlib)
# ---------------------------------------------------------------------------
def display_frames(frames):
    """Show the 20 selected video frames in a 4x5 grid."""
    cols = 5
    rows = (len(frames) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.2 * rows))
    fig.suptitle("Selected Video Frames", fontsize=14)

    for i, ax in enumerate(axes.flat):
        if i < len(frames):
            ax.imshow(frames[i])
            ax.set_title(f"Frame {i + 1}", fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    plt.show()


def plot_motion(motion_values):
    """Plot 1: Motion variation across video frames."""
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(motion_values) + 1), motion_values,
             marker="o", color="tab:red")
    plt.xlabel("Frame Number")
    plt.ylabel("Normalized Motion (mean |frame diff|)")
    plt.title("Motion Variation Across Video Frames")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_rnn_output(rnn_output):
    """Plot 2: RNN temporal feature representation over time steps."""
    plt.figure(figsize=(10, 5))
    for d in range(rnn_output.shape[1]):
        plt.plot(rnn_output[:, d], linewidth=0.8)
    plt.xlabel("Time Step")
    plt.ylabel("RNN Hidden Activation")
    plt.title("RNN Temporal Feature Representation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_lstm_output(lstm_output):
    """Plot 3: LSTM temporal feature representation over time steps."""
    plt.figure(figsize=(10, 5))
    for d in range(lstm_output.shape[1]):
        plt.plot(lstm_output[:, d], linewidth=0.8)
    plt.xlabel("Time Step")
    plt.ylabel("LSTM Hidden Activation")
    plt.title("LSTM Temporal Feature Representation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
# ---------------------------------------------------------------------------
# 7. Saving .npy features
# ---------------------------------------------------------------------------
def save_features(sequence, rnn_output, lstm_output, output_dir=OUTPUTS_DIR):
    """Save the feature sequence and both model outputs as .npy files."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "video_features.npy"), sequence)
    np.save(os.path.join(output_dir, "rnn_features.npy"), rnn_output)
    np.save(os.path.join(output_dir, "lstm_features.npy"), lstm_output)
    print(f"Saved: {output_dir}\\video_features.npy  shape {sequence.shape}")
    print(f"Saved: {output_dir}\\rnn_features.npy    shape {rnn_output.shape}")
    print(f"Saved: {output_dir}\\lstm_features.npy   shape {lstm_output.shape}")


# ---------------------------------------------------------------------------
# 8. Pipeline coordination
# ---------------------------------------------------------------------------
def main():
    """Run the complete Experiment 07 pipeline step by step."""
    print("=" * 70)
    print("Experiment 07 - Video Activity Recognition Using RNN and LSTM")
    print("=" * 70)

    # 1. Check video
    check_video(VIDEO_PATH)
    print("[1] Video check passed ->", VIDEO_PATH)

    # 2. Read video information
    info = get_video_info(VIDEO_PATH)
    print("[2] Video information:")
    print(f"    FPS         : {info['fps']:.2f}")
    print(f"    Resolution  : {info['width']} x {info['height']}")
    print(f"    Total frames: {info['total_frames']}")
    print(f"    Duration    : {info['duration']:.2f} s")

    # 3. Extract 20 equally spaced frames
    frames = extract_frames(VIDEO_PATH, NUM_FRAMES)
    print(f"[3] Extracted {len(frames)} equally spaced frames")

    # 4. Display selected frames
    print("[4] Displaying selected frames (close window to continue)...")
    display_frames(frames)

    # 5. Extract visual features
    raw_features = extract_frame_features(frames)
    print(f"[5] Extracted visual features -> shape {raw_features.shape}")

    # 6. Normalize features (min-max)
    normalized = normalize_features(raw_features)
    print(f"[6] Normalized features -> shape {normalized.shape}")

    # 7. Expand to 64-dimensional sequence (np.tile)
    sequence = expand_features(normalized, FEATURE_SIZE)
    print(f"[7] Expanded feature sequence -> shape {sequence.shape}")

    # 8. Run SimpleRNN
    rnn = SimpleRNN(FEATURE_SIZE, RNN_HIDDEN_SIZE, seed=RANDOM_SEED)
    rnn_output = rnn.forward(sequence)
    print(f"[8] SimpleRNN output -> shape {rnn_output.shape}")

    # 9. Run SimpleLSTM
    lstm = SimpleLSTM(FEATURE_SIZE, LSTM_HIDDEN_SIZE, seed=RANDOM_SEED)
    lstm_output = lstm.forward(sequence)
    print(f"[9] SimpleLSTM output -> shape {lstm_output.shape}")

    # 10. Classify activity (uses the normalized motion feature column)
    motion_values = normalized[:, 3]
    activity, score, avg_motion, temporal_var = classify_activity(
        motion_values, lstm_output
    )

    # 11. Display result
    print("[10] Activity classification:")
    print(f"     average_motion      : {avg_motion:.4f}")
    print(f"     temporal_variation  : {temporal_var:.4f}")
    print(f"     activity_score      : {score:.4f}")
    print(f"     => {activity}")

    # 12-14. Plots
    print("[11] Plotting motion variation... (close window to continue)")
    plot_motion(motion_values)
    print("[12] Plotting RNN output... (close window to continue)")
    plot_rnn_output(rnn_output)
    print("[13] Plotting LSTM output... (close window to continue)")
    plot_lstm_output(lstm_output)

    # 15. Save .npy files
    print("[14] Saving .npy feature files...")
    save_features(sequence, rnn_output, lstm_output)

    print("=" * 70)
    print("Experiment 07 completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, IOError, ValueError) as err:
        print("\nERROR:", err)
        raise SystemExit(1)