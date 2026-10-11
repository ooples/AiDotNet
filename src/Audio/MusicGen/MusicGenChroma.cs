using System.Numerics;

namespace AiDotNet.Audio.MusicGen;

/// <summary>
/// MusicGen's melody representation (Copet et al. 2023, §2.3 and §3.1): the chromagram of the conditioning audio with
/// a 2¹⁴-sample window and a 2¹² hop, quantized to its dominant pitch class per frame (one-hot), an information
/// bottleneck against copying the input.
/// </summary>
/// <remarks>
/// Follows audiocraft's <c>ChromaExtractor</c> (argmax mode): a centred, reflect-padded power spectrogram with a
/// periodic Hann window, normalized by √Σw² (torchaudio <c>Spectrogram(normalized=True, power=2)</c>), weighted by
/// librosa's chroma filterbank (<c>filters.chroma</c>: tuning 0, 12 bins, centre octave 5, octave width 2, L2-normed,
/// C first), each frame divided by its largest bin, then the argmax set to one. Audio shorter than the window is
/// zero-padded to it on both sides. audiocraft also removes drums and bass with Demucs before this step; the paper does
/// not, so a caller that wants that passes the separated stem.
/// </remarks>
internal static class MusicGenChroma
{
    public const int Bins = 12;

    /// <summary>The one-hot chromagram <c>[frames, 12]</c> of a mono waveform.</summary>
    public static double[,] Extract(double[] audio, int sampleRate, int window = 1 << 14, int hop = 1 << 12)
    {
        if (audio.Length < window)
        {
            int pad = window - audio.Length, left = pad / 2;
            var padded = new double[window];
            Array.Copy(audio, 0, padded, left, audio.Length);
            audio = padded;
        }
        int half = window / 2, bins = window / 2 + 1;
        // Centre with reflect padding (torch.stft center=True, pad_mode="reflect").
        var signal = new double[audio.Length + window];
        for (int i = 0; i < signal.Length; i++)
        {
            int j = i - half;
            if (j < 0) j = -j;
            if (j >= audio.Length) j = 2 * (audio.Length - 1) - j;
            signal[i] = audio[j];
        }
        int frames = 1 + audio.Length / hop;
        var hann = new double[window];
        double energy = 0;
        for (int i = 0; i < window; i++)
        {
            hann[i] = 0.5 - 0.5 * Math.Cos(2 * Math.PI * i / window);
            energy += hann[i] * hann[i];
        }
        var filters = Filterbank(sampleRate, window);
        var chroma = new double[frames, Bins];
        var buffer = new Complex[window];
        for (int f = 0; f < frames; f++)
        {
            for (int i = 0; i < window; i++) buffer[i] = new Complex(signal[f * hop + i] * hann[i], 0);
            Fft(buffer);
            var raw = new double[Bins];
            for (int k = 0; k < bins; k++)
            {
                double power = (buffer[k].Real * buffer[k].Real + buffer[k].Imaginary * buffer[k].Imaginary) / energy;
                for (int c = 0; c < Bins; c++) raw[c] += filters[c, k] * power;
            }
            double max = Math.Max(raw.Max(r => Math.Abs(r)), 1e-6);
            int best = 0;
            for (int c = 1; c < Bins; c++) if (raw[c] / max > raw[best] / max) best = c;
            chroma[f, best] = 1;
        }
        return chroma;
    }

    /// <summary>librosa <c>filters.chroma(sr, n_fft, tuning=0, n_chroma=12)</c>: <c>[12, n_fft/2 + 1]</c>.</summary>
    public static double[,] Filterbank(int sampleRate, int fftSize)
    {
        const double centreOctave = 5.0, octaveWidth = 2.0;
        int n = Bins;
        // frqbins: n · octaves of each FFT frequency above 0 (A440 / 16 as octave 0), with bin 0 placed 1.5 octaves below.
        var frq = new double[fftSize];
        for (int i = 1; i < fftSize; i++)
            frq[i] = n * Math.Log(((double)i * sampleRate / fftSize) / (440.0 / 16), 2);
        frq[0] = frq[1] - 1.5 * n;
        var width = new double[fftSize];
        for (int i = 0; i < fftSize - 1; i++) width[i] = Math.Max(frq[i + 1] - frq[i], 1.0);
        width[fftSize - 1] = 1;
        double half = Math.Round(n / 2.0);
        var weights = new double[n, fftSize];
        for (int c = 0; c < n; c++)
            for (int i = 0; i < fftSize; i++)
            {
                double d = frq[i] - c;
                d = ((d + half + 10 * n) % n + n) % n - half;                  // np.remainder (non-negative)
                weights[c, i] = Math.Exp(-0.5 * Math.Pow(2 * d / width[i], 2));
            }
        for (int i = 0; i < fftSize; i++)
        {
            double norm = 0;
            for (int c = 0; c < n; c++) norm += weights[c, i] * weights[c, i];
            norm = Math.Sqrt(norm);
            if (norm > 1e-300) for (int c = 0; c < n; c++) weights[c, i] /= norm;
            double octave = Math.Exp(-0.5 * Math.Pow((frq[i] / n - centreOctave) / octaveWidth, 2));
            for (int c = 0; c < n; c++) weights[c, i] *= octave;
        }
        // base_c: roll by −3 so the first bin is C; keep the non-negative frequencies.
        int kept = fftSize / 2 + 1;
        var result = new double[n, kept];
        for (int c = 0; c < n; c++)
            for (int i = 0; i < kept; i++) result[c, i] = weights[(c + 3) % n, i];
        return result;
    }

    // In-place radix-2 FFT (length a power of two).
    private static void Fft(Complex[] a)
    {
        int n = a.Length;
        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) (a[i], a[j]) = (a[j], a[i]);
        }
        for (int length = 2; length <= n; length <<= 1)
        {
            double angle = -2 * Math.PI / length;
            var step = new Complex(Math.Cos(angle), Math.Sin(angle));
            for (int i = 0; i < n; i += length)
            {
                var w = Complex.One;
                for (int k = 0; k < length / 2; k++)
                {
                    var u = a[i + k];
                    var v = a[i + k + length / 2] * w;
                    a[i + k] = u + v;
                    a[i + k + length / 2] = u - v;
                    w *= step;
                }
            }
        }
    }
}
