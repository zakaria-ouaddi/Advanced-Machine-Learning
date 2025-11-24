import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf
from mne import concatenate_raws, events_from_annotations
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---------------------------------------------------
# 1. Grundparameter
# ---------------------------------------------------
runs = [6, 10, 14]      # Task 4: Motor Imagery Fäuste vs Füße
tmin, tmax = 0.0, 4.0   # 4s-Fenster pro Trial (ganze Imagery-Phase)

# Alle Subjects – EEGBCI hat bis zu 109, aber wir fangen einfach 1..109 an.
all_subjects = [1]

X_list = []
y_list = []
subject_ids = []   # um später zu wissen, von welcher Person ein Trial stammt

# ---------------------------------------------------
# 2. Schleife über alle Subjects
# ---------------------------------------------------
for subject in all_subjects:
    print(f"=== Subject {subject} ===")
    try:
        # 2.1 Daten für diesen Subject und die gewünschten Runs laden
        fnames = eegbci.load_data(subject, runs)
    except Exception as e:
        print(f"  -> Konnte Daten für Subject {subject} nicht laden: {e}")
        continue

    try:
        raws = [read_raw_edf(f, preload=True, stim_channel='auto') for f in fnames]
        raw = concatenate_raws(raws)
    except Exception as e:
        print(f"  -> Fehler beim Lesen/Konkat. der EDF-Dateien: {e}")
        continue

    # 2.2 Kanalnamen säubern (Punkte entfernen)
    channel_mapping = {old_name: old_name.rstrip('.') for old_name in raw.ch_names}
    raw.rename_channels(channel_mapping)

    # 2.3 Montage setzen
    montage = mne.channels.make_standard_montage('standard_1005')
    try:
        raw.set_montage(montage, match_case=False)
    except Exception as e:
        print(f"  -> Konnte Montage nicht setzen: {e}")
        continue

    # 2.5 Events & Event-IDs aus Annotationen
    try:
        events, event_id_annot = events_from_annotations(raw)
    except Exception as e:
        print(f"  -> Konnte Events aus Annotations nicht extrahieren: {e}")
        continue

    # Prüfen, ob T1/T2 überhaupt vorhanden sind
    if 'T1' not in event_id_annot or 'T2' not in event_id_annot:
        print(f"  -> T1/T2 nicht in event_id_annot für Subject {subject}, überspringe.")
        continue

    event_id = {
        'fists': event_id_annot['T1'],  # Imagery beide Fäuste
        'feet':  event_id_annot['T2'],  # Imagery beide Füße
    }

    # 2.6 Epochs für T1/T2 erstellen
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
        print(f"  -> Fehler beim Erzeugen der Epochs: {e}")
        continue

    if len(epochs) == 0:
        print(f"  -> Keine Epochs für T1/T2 bei Subject {subject}, überspringe.")
        continue

    print(f"  -> {len(epochs)} Trials gefunden (fists/feet).")

    # 2.7 X_subj und y_subj extrahieren
    X_subj = epochs.get_data()          # (n_trials, n_channels, n_samples)
    event_codes = epochs.events[:, 2]   # z.B. 2 und 3

    code_to_label = {
        event_id['fists']: 0,   # Fäuste -> 0
        event_id['feet']:  1,   # Füße  -> 1
    }
    y_subj = np.vectorize(code_to_label.get)(event_codes)

    # 2.8 In die gesamt-Listen einfügen
    X_list.append(X_subj)
    y_list.append(y_subj)
    subject_ids.extend([subject] * len(y_subj))

# ---------------------------------------------------
# 3. Alles zu einem großen Dataset zusammenfügen
# ---------------------------------------------------
if len(X_list) == 0:
    raise RuntimeError("Es wurden keine gültigen Trials gefunden – prüfen, ob Dataset erreichbar ist.")

X_all = np.concatenate(X_list, axis=0)
y_all = np.concatenate(y_list, axis=0)
subject_ids = np.array(subject_ids)

print("Gesamt-Dataset:")
print("  X_all shape:", X_all.shape)   # (gesamt_trials, n_channels, n_samples)
print("  y_all shape:", y_all.shape)
print("  Anzahl Subjects im Datensatz:", len(np.unique(subject_ids)))
print("  Label-Verteilung:", np.unique(y_all, return_counts=True))

# ---------------------------------------------------
# 4. Train/Test-Split über ALLE Trials
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
# 5. Shape für EEGNet / ShallowConvNet vorbereiten
# ---------------------------------------------------
# Viele CNN-EEG-Modelle erwarten (batch, 1, n_channels, n_samples)
X_train_cnn = X_train[:, np.newaxis, :, :]
X_test_cnn  = X_test[:,  np.newaxis, :, :]

print("X_train_cnn shape:", X_train_cnn.shape)
print("X_test_cnn shape:",  X_test_cnn.shape)




# 6. Beispiel-Ausgabe
# ---------------------------------------------------
i = 2  # Beispielnummer
example = X_train[i]  # shape: (64, 641)
label = y_train[i]

print("Shape eines Beispiels:", example.shape)
print("Label:", label)

trial = X_train[i]
channel = 10  # z. B. Kanal 10
plt.plot(trial[channel])
plt.title(f"Trial 0 – Channel {channel} – Label = {y_train[i]}")
plt.xlabel("Time (Samples)")
plt.ylabel("Amplitude (µV)")
plt.show()

example = X_train[i]      # shape: (64, 641)
label = y_train[i]        # 0 or 1

print("Label of X_train[0]:", label)