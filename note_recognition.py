import argparse
from pydub import AudioSegment
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from utils import (
    frequency_spectrum,
    calculate_distance,
    classify_note_attempt_3,
)


def main(file, note_file=None, note_starts_file=None, plot_starts=False, plot_fft_indices=[]):
    # Load actual notes / starts if provided
    actual_starts = []
    if note_starts_file:
        with open(note_starts_file) as f:
            actual_starts = [float(line.strip()) for line in f]

    actual_notes = []
    if note_file:
        with open(note_file) as f:
            actual_notes = [line.strip() for line in f]

    song = AudioSegment.from_file(file)
    song = song.high_pass_filter(80)  # Remove low-frequency noise

    # Detect note starts
    starts = predict_note_starts(song, plot_starts, actual_starts)

    # Predict notes
    predicted_notes = predict_notes(song, starts, actual_notes, plot_fft_indices)

    print("\n--- RESULTS ---")
    if actual_notes:
        print("Actual Notes:")
        print(actual_notes)
    print("Predicted Notes:")
    print(predicted_notes)

    if actual_notes:
        lev_distance = calculate_distance(predicted_notes, actual_notes)
        print(f"Levenshtein distance: {lev_distance}/{len(actual_notes)}")


def predict_note_starts(song, plot=False, actual_starts=[]):
    SEGMENT_MS = 50
    MIN_MS_BETWEEN = 100

    # Use RMS for adaptive thresholding
    volume = np.array([segment.rms for segment in song[::SEGMENT_MS]])
    threshold = np.mean(volume) + 0.5 * np.std(volume)

    predicted_starts = []
    for i in range(1, len(volume)):
        if volume[i] > threshold and volume[i] - volume[i - 1] > 0:
            ms = i * SEGMENT_MS
            if len(predicted_starts) == 0 or ms - predicted_starts[-1] >= MIN_MS_BETWEEN:
                predicted_starts.append(ms)

    if len(actual_starts) > 0:
        print(f"Actual starts ({len(actual_starts)}): {['{:.2f}'.format(s) for s in actual_starts]}")
        print(f"Predicted starts ({len(predicted_starts)}): {['{:.2f}'.format(ms / 1000) for ms in predicted_starts]}")

    if plot:
        x_axis = np.arange(len(volume)) * (SEGMENT_MS / 1000)
        plt.plot(x_axis, volume)
        for s in actual_starts:
            plt.axvline(x=s, color="r", linewidth=0.5)
        for ms in predicted_starts:
            plt.axvline(x=ms / 1000, color="g", linewidth=0.5)
        plt.xlabel("Time (s)")
        plt.ylabel("RMS volume")
        plt.show()

    return predicted_starts


def predict_notes(song, starts, actual_notes, plot_fft_indices):
    predicted_notes = []
    for i, start in enumerate(starts):
        sample_from = start
        sample_to = start + 500  # 500 ms window
        if i < len(starts) - 1:
            sample_to = min(starts[i + 1], sample_to)

        segment = song[sample_from:sample_to]
        freqs, freq_magnitudes = frequency_spectrum(segment)

        # Classify note
        try:
            predicted = classify_note_attempt_3(freqs, freq_magnitudes)
        except IndexError:
            predicted = "U"
        predicted_notes.append(predicted or "U")

        # Info output
        print(f"\nNote {i + 1}: Predicted={predicted}")
        if i < len(actual_notes):
            print(f"Actual={actual_notes[i]}")
        print(f"Start={start} ms, Window={sample_from}-{sample_to} ms")

        # Show peaks
        peaks, props = scipy.signal.find_peaks(freq_magnitudes, height=0.015)
        for j, p in enumerate(peaks):
            print(f"Peak {j + 1}: {freqs[p]:.1f} Hz, magnitude {props['peak_heights'][j]:.3f}")

        if i in plot_fft_indices:
            plt.plot(freqs, freq_magnitudes)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Normalized magnitude")
            plt.title(f"FFT for note {i + 1}")
            plt.show()

    return predicted_notes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Input audio file (wav, mp3, etc.)")
    parser.add_argument("--note-file", type=str, help="Ground truth note labels")
    parser.add_argument("--note-starts-file", type=str, help="Ground truth note start times")
    parser.add_argument("--plot-starts", action="store_true", help="Plot predicted vs actual starts")
    parser.add_argument("--plot-fft-index", type=int, nargs="*", help="Plot FFT for specific note indices")
    args = parser.parse_args()

    main(
        args.file,
        note_file=args.note_file,
        note_starts_file=args.note_starts_file,
        plot_starts=args.plot_starts,
        plot_fft_indices=args.plot_fft_index or [],
    )