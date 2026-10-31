using AiDotNet.Helpers;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// GPT-SoVITS's resampler for the reference audio (<c>inference_webui.resample(..., match_librosa=True)</c>): resampy's
/// <c>kaiser_best</c> band-limited interpolation (a 50-zero-crossing Kaiser-windowed sinc, β 12.9846, roll-off 0.91735,
/// tabulated at 8,192 points per zero crossing and linearly interpolated), with the release's float32 table and float32
/// time arithmetic.
/// </summary>
internal static class GptSoVitsAudio
{
    private const int ZeroCrossings = 50, Precision = 8192;
    private const double Rolloff = 0.9173473712608761, Beta = 12.984585247040012;

    private static readonly Lazy<(float[] Window, float[] Delta)> Table = new(() =>
    {
        // resampy.filters.sinc_window: rolloff·sinc(rolloff·x) over [0, 50] tapered by the right half of a Kaiser window.
        int n = Precision * ZeroCrossings;
        var window = new float[n + 1];
        double i0Beta = AudioHelper<double>.BesselI0(Beta);
        for (int k = 0; k <= n; k++)
        {
            double x = ZeroCrossings * (double)k / n;
            double sinc = x == 0 ? 1 : Math.Sin(Math.PI * Rolloff * x) / (Math.PI * Rolloff * x);
            double ratio = (double)k / n;
            double taper = AudioHelper<double>.BesselI0(Beta * Math.Sqrt(Math.Max(0, 1 - ratio * ratio))) / i0Beta;
            window[k] = (float)(taper * Rolloff * sinc);
        }
        // torch.diff(interp_win, append=interp_win[-1:]) in float32.
        var delta = new float[n + 1];
        for (int k = 0; k < n; k++) delta[k] = window[k + 1] - window[k];
        return (window, delta);
    }, isThreadSafe: true);

    /// <summary>The release's <c>resample(audio, sr0, sr1, match_librosa=True)</c> of mono audio: ⌊samples·sr1/sr0⌋
    /// samples.</summary>
    public static double[] Resample(double[] audio, int from, int to)
    {
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        if (from <= 0 || to <= 0) throw new ArgumentOutOfRangeException(nameof(from), "Sample rates are positive.");
        if (from == to) return (double[])audio.Clone();
        var (table, tableDelta) = Table.Value;
        double ratio = (double)to / from;
        double scale = Math.Min(1.0, ratio);
        int nOrig = audio.Length, nOut = (int)(nOrig * ratio);
        int nwin = table.Length;
        // The scaled tables stay float32 (interp_win * ratio on a float32 tensor).
        float[] window = table, delta = tableDelta;
        if (ratio < 1.0)
        {
            float r = (float)ratio;
            window = table.Select(v => v * r).ToArray();
            delta = tableDelta.Select(v => v * r).ToArray();
        }
        int indexStep = (int)(scale * Precision);
        float scale32 = (float)scale, ratio32 = (float)ratio;
        var output = new double[nOut];
        for (int i = 0; i < nOut; i++)
        {
            float t = i / ratio32;
            int n = (int)Math.Floor(t);
            float frac = scale32 * (t - n);
            output[i] = Side(audio, window, delta, frac, n, nwin, indexStep, left: true)
                + Side(audio, window, delta, scale32 - frac, n, nwin, indexStep, left: false);
        }
        return output;
    }

    private static double Side(double[] audio, float[] window, float[] delta, float frac, int n, int nwin, int indexStep, bool left)
    {
        float indexFrac = frac * Precision;
        int offset = (int)Math.Floor(indexFrac);
        float eta = indexFrac - offset;
        int limit = Math.Min(left ? n + 1 : audio.Length - n - 1, (nwin - offset) / indexStep);
        double sum = 0;
        for (int k = 0; k < limit; k++)
        {
            int index = offset + k * indexStep;
            float weight = window[index] + eta * delta[index];
            sum += audio[left ? n - k : n + k + 1] * (double)weight;
        }
        return sum;
    }
}
