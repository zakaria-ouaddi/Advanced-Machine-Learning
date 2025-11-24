import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf
from mne import concatenate_raws, events_from_annotations
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---------------------------------------------------
# 1. Basic parameters
# ---------------------------------------------------
runs = [6, 10, 14]      # Task 4: Motor Imagery Fists vs Feet
tmin, tmax = 0.0, 4.0   # 4s window per trial (entire imagery phase)

# All subjects – EEGBCI has up to 109, but we'll just start with 1..109.
all_subjects = [1]

X_list = []
y_list = []
subject_ids = []   # to later know which person a trial came from

# ---------------------------------------------------
# 2. Loop over all subjects
# ---------------------------------------------------
for subject in all_subjects:
    print(f"=== Subject {subject} ===")
    try:
        # 2.1 Load data for this subject and the desired runs
        fnames = eegbci.load_data(subject, runs)
    except Exception as e:
        print(f"  -> Could not load data for subject {subject}: {e}")
        continue

    try:
        raws = [read_raw_edf(f, preload=True, stim_channel='auto') for f in fnames]
        raw = concatenate_raws(raws)
    except Exception as e:
        print(f"  -> Error reading/concatenating EDF files: {e}")
        continue

    # 2.2 Clean channel names (remove dots)
    channel_mapping = {old_name: old_name.rstrip('.') for old_name in raw.ch_names}
    raw.rename_channels(channel_mapping)

    # 2.3 Set montage
    montage = mne.channels.make_standard_montage('standard_1005')
    try:
        raw.set_montage(montage, match_case=False)
    except Exception as e:
        print(f"  -> Could not set montage: {e}")
        continue

    # 2.5 Events & Event-IDs from annotations
    try:
        events, event_id_annot = events_from_annotations(raw)
    except Exception as e:
        print(f"  -> Konnte Events aus Annotations nicht extrahieren: {e}")
        continue

    # Check if T1/T2 are even present
    if 'T1' not in event_id_annot or 'T2' not in event_id_annot:
        print(f"  -> T1/T2 not in event_id_annot for subject {subject}, skipping.")
        continue

    event_id = {
        'fists': event_id_annot['T1'],  # Imagery both fists
        'feet':  event_id_annot['T2'],  # Imagery both feet
    }

    # 2.6 Create epochs for T1/T2
    try:
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False
        )
    except Exception as e:
        print(f"  -> Error creating epochs: {e}")
        continue

    if len(epochs) == 0:
        print(f"  -> No epochs for T1/T2 for subject {subject}, skipping.")
        continue

    print(f"  -> {len(epochs)} Trials found (fists/feet).")

    # 2.7 Extract X_subj and y_subj
    X_subj = epochs.get_data()          # (n_trials, n_channels, n_samples)
    event_codes = epochs.events[:, 2]   # e.g. 2 and 3

    code_to_label = {
        event_id['fists']: 0,   # Fists -> 0
        event_id['feet']:  1,   # Feet  -> 1
    }
    y_subj = np.vectorize(code_to_label.get)(event_codes)

    # 2.8 Add to the overall lists
    X_list.append(X_subj)
    y_list.append(y_subj)
    subject_ids.extend([subject] * len(y_subj))

# ---------------------------------------------------
# 3. Combine everything into one big dataset
# ---------------------------------------------------
if len(X_list) == 0:
    raise RuntimeError("No valid trials found – check if dataset is accessible.")

X_all = np.concatenate(X_list, axis=0)
y_all = np.concatenate(y_list, axis=0)
subject_ids = np.array(subject_ids)

print("Overall dataset:")
print("  X_all shape:", X_all.shape)   # (total_trials, n_channels, n_samples)
print("  y_all shape:", y_all.shape)
print("  Number of subjects in dataset:", len(np.unique(subject_ids)))
print("  Label distribution:", np.unique(y_all, return_counts=True))
# ---------------------------------------------------
# 4. Train/Test-Split over ALL trials
# ---------------------------------------------------
X_train, X_test, y_train, y_test, subj_train, subj_test = train_test_split(
    X_all,
    y_all,
    subject_ids,
    test_size=0.2,
    random_state=42,
    stratify=y_all
)

print("Train:", X_train.shape, y_train.shape)
print("Test:",  X_test.shape,  y_test.shape)

# ---------------------------------------------------
# 5. Shape for EEGNet / ShallowConvNet
# ---------------------------------------------------
# Many CNN-EEG models expect (batch, 1, n_channels, n_samples)
X_train_cnn = X_train[:, np.newaxis, :, :]
X_test_cnn  = X_test[:,  np.newaxis, :, :]

print("X_train_cnn shape:", X_train_cnn.shape)
print("X_test_cnn shape:",  X_test_cnn.shape)




# 6. Example output
# ---------------------------------------------------
i = 2  # Example number
example = X_train[i]  # shape: (64, 641)
label = y_train[i]

print("Shape of an example:", example.shape)
print("Label:", label)

trial = X_train[i]
channel = 10  # e.g. Channel 10
plt.plot(trial[channel])
plt.title(f"Trial 0 – Channel {channel} – Label = {y_train[i]}")
plt.xlabel("Time (Samples)")
plt.ylabel("Amplitude (µV)")
plt.show()

example = X_train[i]      # shape: (64, 641)
label = y_train[i]        # 0 or 1

print("Label of X_train[0]:", label)