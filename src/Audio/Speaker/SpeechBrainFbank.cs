namespace AiDotNet.Audio.Speaker;

/// <summary>
/// speechbrain's <c>Fbank</c> (a 25 ms Hamming-windowed STFT every 10 ms, centred with zero padding, the power spectrum
/// through triangular HTK-mel filters, 10·log10 floored at 80 dB below the maximum), shared by its speaker extractors.
/// </summary>
internal static class SpeechBrainFbank
{
    private const int FftSize = 400, Window = 400, Hop = 160;

    // speechbrain Filterbank: triangles centred at HTK-mel-spaced frequencies over 0 … sr/2, on the 201 STFT bins.
    public static double[,] Filters(int mels, int sampleRate)
    {
        int bins = FftSize / 2 + 1;
        double ToMel(double hz) => 2595 * Math.Log10(1 + hz / 700);
        double ToHz(double mel) => 700 * (Math.Pow(10, mel / 2595) - 1);
        double lo = ToMel(0), hi = ToMel(sampleRate / 2.0);
        var hz = Enumerable.Range(0, mels + 2).Select(i => ToHz(lo + (hi - lo) * i / (mels + 1))).ToArray();
        var filters = new double[bins, mels];
        for (int m = 0; m < mels; m++)
        {
            double central = hz[m + 1], band = hz[m + 1] - hz[m];
            for (int b = 0; b < bins; b++)
            {
                double frequency = (sampleRate / 2.0) * b / (bins - 1);
                double slope = (frequency - central) / band;
                filters[b, m] = Math.Max(0, Math.Min(slope + 1, -slope + 1));
            }
        }
        return filters;
    }

    /// <summary>speechbrain's <c>Fbank</c> of a 16 kHz waveform through <paramref name="filters"/>, the sentence mean removed
    /// (<c>InputNormalization</c> by sentence without the deviation): <c>[frames, mels]</c>.</summary>
    public static double[,] Features(IReadOnlyList<double> wave, double[,] filters)
    {
        int bins = FftSize / 2 + 1, half = FftSize / 2, mels = filters.GetLength(1);
        int frames = 1 + wave.Count / Hop;
        var window = Enumerable.Range(0, Window).Select(n => 0.54 - 0.46 * Math.Cos(2 * Math.PI * n / Window)).ToArray();
        var db = new double[frames, mels];
        double max = double.NegativeInfinity;
        var cos = new double[FftSize, bins];
        var sin = new double[FftSize, bins];
        for (int n = 0; n < FftSize; n++)
            for (int b = 0; b < bins; b++)
            {
                double angle = 2 * Math.PI * n * b / FftSize;
                cos[n, b] = Math.Cos(angle);
                sin[n, b] = Math.Sin(angle);
            }
        var power = new double[bins];
        for (int f = 0; f < frames; f++)
        {
            Array.Clear(power, 0, bins);
            var frame = new double[FftSize];
            for (int n = 0; n < FftSize; n++)
            {
                int i = f * Hop + n - half;
                frame[n] = i >= 0 && i < wave.Count ? wave[i] * window[n] : 0.0;
            }
            for (int b = 0; b < bins; b++)
            {
                double re = 0, im = 0;
                for (int n = 0; n < FftSize; n++)
                {
                    re += frame[n] * cos[n, b];
                    im -= frame[n] * sin[n, b];
                }
                power[b] = re * re + im * im;
            }
            for (int m = 0; m < mels; m++)
            {
                double s = 0;
                for (int b = 0; b < bins; b++) s += power[b] * filters[b, m];
                db[f, m] = 10 * Math.Log10(Math.Max(s, 1e-10));
                max = Math.Max(max, db[f, m]);
            }
        }
        for (int m = 0; m < mels; m++)
        {
            double mean = 0;
            for (int f = 0; f < frames; f++)
            {
                db[f, m] = Math.Max(db[f, m], max - 80);
                mean += db[f, m];
            }
            mean /= frames;
            for (int f = 0; f < frames; f++) db[f, m] -= mean;
        }
        return db;
    }
}
