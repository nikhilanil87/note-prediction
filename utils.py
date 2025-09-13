import array
from collections import Counter

import numpy as np
from scipy.fft import fft
from pydub.utils import get_array_type
from Levenshtein import distance

# Note frequencies (A4 = 440 Hz)
NOTES = {
    "A": 440.00,
    "A#": 466.16,
    "B": 493.88,
    "C": 523.25,
    "C#": 554.37,
    "D": 587.33,
    "D#": 622.25,
    "E": 659.26,
    "F": 698.46,
    "F#": 739.99,
    "G": 783.99,
    "G#": 830.61,
}

def frequency_spectrum(sample, max_frequency=800):
    """
    Derive frequency spectrum of a pydub.AudioSegment.
    Returns frequencies and normalized magnitudes.
    """
    bit_depth = sample.sample_width * 8
    array_type = get_array_type(bit_depth)
    raw_audio_data = array.array(array_type, sample._data)
    n = len(raw_audio_data)

    raw_audio_data = np.array(raw_audio_data)
    raw_audio_data = raw_audio_data - np.mean(raw_audio_data)  # zero-centering

    freq_magnitude = fft(raw_audio_data)
    freq_magnitude = freq_magnitude[: n // 2]
    freq_magnitude = np.abs(freq_magnitude)
    freq_magnitude = freq_magnitude / np.sum(freq_magnitude)

    freq_array = np.arange(n // 2) * (sample.frame_rate / n)
    if max_frequency:
        max_index = int(max_frequency * n / sample.frame_rate) + 1
        freq_array = freq_array[:max_index]
        freq_magnitude = freq_magnitude[:max_index]

    return freq_array, freq_magnitude


def get_note_for_freq(f, tolerance=33):
    """
    Return the closest note to frequency f, or None.
    Tolerance in cents (1/100 of a semitone)
    """
    tolerance_multiplier = 2 ** (tolerance / 1200)
    note_ranges = {k: (v / tolerance_multiplier, v * tolerance_multiplier) for k, v in NOTES.items()}

    # Map f to A4 octave range
    range_min = note_ranges["A"][0]
    range_max = note_ranges["G#"][1]
    while f < range_min:
        f *= 2
    while f > range_max:
        f /= 2

    for note, (low, high) in note_ranges.items():
        if low <= f <= high:
            return note
    return None


def classify_note_attempt_1(freq_array, freq_magnitude):
    """Return note with maximum magnitude"""
    i = np.argmax(freq_magnitude)
    return get_note_for_freq(freq_array[i])


def classify_note_attempt_2(freq_array, freq_magnitude):
    """Weighted by magnitude"""
    note_counter = Counter()
    for f, mag in zip(freq_array, freq_magnitude):
        if mag < 0.01:
            continue
        note = get_note_for_freq(f)
        if note:
            note_counter[note] += mag
    if note_counter:
        return note_counter.most_common(1)[0][0]
    return None


def classify_note_attempt_3(freq_array, freq_magnitude):
    """Weighted with harmonic multiples"""
    note_counter = Counter()
    min_freq = 82  # low E on guitar
    for f, mag in zip(freq_array, freq_magnitude):
        if mag < 0.01:
            continue
        for mult, credit in [(1, 1), (1 / 3, 0.75), (1 / 5, 0.5), (1 / 6, 0.5), (1 / 7, 0.5)]:
            freq = f * mult
            if freq < min_freq:
                continue
            note = get_note_for_freq(freq)
            if note:
                note_counter[note] += mag * credit
    if note_counter:
        return note_counter.most_common(1)[0][0]
    return None


def calculate_distance(predicted, actual):
    """
    Levenshtein distance between predicted and actual notes
    Natural notes lowercase, sharps uppercase
    """
    def transform(n):
        if "#" in n:
            return n[0].upper()
        return n.lower()

    pred_str = "".join([transform(n) for n in predicted])
    actual_str = "".join([transform(n) for n in actual])
    return distance(pred_str, actual_str)
