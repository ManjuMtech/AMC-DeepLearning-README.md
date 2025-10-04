import numpy as np
import pandas as pd
import random
from scipy.signal import upfirdn
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# Define Modulation Types (Only 3 Modulations: BPSK, QPSK, 16QAM)
MODULATIONS = ["BPSK", "QPSK", "16QAM"]

# Dataset Parameters
TOTAL_SAMPLES = 12000  # Increased total samples for better balance
NO_SIGNAL_SAMPLES = 4000  # Balanced no-signal samples
SIGNAL_SAMPLES = TOTAL_SAMPLES - NO_SIGNAL_SAMPLES  # Remaining samples contain signals
SIGNAL_LENGTH = 512  # Number of I/Q samples per signal
MAX_MODULATIONS = 3  # Maximum number of modulations per signal
SNR_RANGE = (-5, 20)  # SNR in dB
TEST_SIZE = 0.2  # 20% of the dataset for testing

def add_noise(iq_signal, snr_db):
    """Adds AWGN noise based on SNR value."""
    signal_power = np.mean(np.abs(iq_signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*iq_signal.shape) + 1j * np.random.randn(*iq_signal.shape))
    return iq_signal + noise

def generate_signal(mod_type, snr_db):
    """Generates an I/Q signal with specified modulation and SNR."""
    num_symbols = SIGNAL_LENGTH
    if mod_type == "BPSK":
        symbols = 2 * np.random.randint(0, 2, num_symbols) - 1
    elif mod_type == "QPSK":
        symbols = (2 * np.random.randint(0, 2, num_symbols) - 1) + 1j * (2 * np.random.randint(0, 2, num_symbols) - 1)
    elif mod_type == "16QAM":
        real = (2 * np.random.randint(0, 4, num_symbols) - 3) / np.sqrt(10)
        imag = (2 * np.random.randint(0, 4, num_symbols) - 3) / np.sqrt(10)
        symbols = real + 1j * imag
    else:
        raise ValueError("Unsupported Modulation Type")
    
    iq_signal = add_noise(symbols, snr_db)
    return iq_signal

#  **Store Data as a List Instead of Appending to DataFrame**
data = []

#  **Generate exactly 4000 "No Signal" (Signal_Count = 0) samples**
for _ in range(NO_SIGNAL_SAMPLES):
    real_part = np.random.normal(0, 0.1, SIGNAL_LENGTH)  # Background noise
    imag_part = np.random.normal(0, 0.1, SIGNAL_LENGTH)
    
    mod_labels = [0] * len(MODULATIONS)  # No active modulations
    snr_value = random.uniform(*SNR_RANGE)
    signal_count_label = 0  # No signal present
    
    data.append(np.concatenate([real_part, imag_part, mod_labels, [snr_value, signal_count_label]]))

#  **Generate exactly 8000 Samples with 1, 2, or 3 Modulations**
for _ in range(SIGNAL_SAMPLES):
    num_signals = random.choice([1, 2, 3])  # Pick a random number of modulations
    selected_mods = random.sample(MODULATIONS, k=num_signals)
    snr_value = random.uniform(*SNR_RANGE)

    mixed_signal = np.zeros(SIGNAL_LENGTH, dtype=complex)
    mod_labels = [1 if mod in selected_mods else 0 for mod in MODULATIONS]
    for mod in selected_mods:
        mixed_signal += generate_signal(mod, snr_value) / num_signals  # Normalize mixture
    
    real_part, imag_part = np.real(mixed_signal), np.imag(mixed_signal)
    signal_count_label = num_signals  # Number of signals present
    data.append(np.concatenate([real_part, imag_part, mod_labels, [snr_value, signal_count_label]]))

#  **Convert List to DataFrame in One Step**
columns = [f"Real_{i}" for i in range(SIGNAL_LENGTH)] + \
          [f"Imag_{i}" for i in range(SIGNAL_LENGTH)] + \
          [f"Mod_{mod}" for mod in MODULATIONS] + ["SNR", "Signal_Count"]

df = pd.DataFrame(data, columns=columns)

#  **Ensure Exactly 12000 Rows Are Generated**
assert len(df) == TOTAL_SAMPLES, f"Error: Expected {TOTAL_SAMPLES} rows, but got {len(df)}"

#  **Balanced Train-Test Split Using `StratifiedShuffleSplit`**
splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=42)
for train_idx, test_idx in splitter.split(df, df["Signal_Count"]):
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

#  **Final Checks Before Saving**
print("Training Set Signal Count Distribution:\n", train_df["Signal_Count"].value_counts(normalize=True))
print("Testing Set Signal Count Distribution:\n", test_df["Signal_Count"].value_counts(normalize=True))

#  **Save Train & Test Datasets**
train_df.to_csv("train_multi_signal_count_dataset.csv", index=False)
test_df.to_csv("test_multi_signal_count_dataset.csv", index=False)

print(f"✅ Train dataset with {len(train_df)} samples and Test dataset with {len(test_df)} samples saved successfully!")
