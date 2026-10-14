namespace AiDotNet.Audio.Features;

/// <summary>
/// The two mel filterbanks published audio models are trained with, computed exactly as their reference
/// libraries compute them: librosa's (Slaney scale and area normalization, the default of
/// <c>librosa.filters.mel</c>) and torchaudio's (<c>melscale_fbanks</c> with the HTK scale and no normalization,
/// the default of <c>torchaudio.transforms.MelSpectrogram</c>).
/// </summary>
/// <remarks>
/// <para>
/// Both build continuous triangles over the FFT bin frequencies, not the bin-rounded triangles of the classic
/// HTK recipe, so the weights differ from those wherever a band edge falls between bins. A model loaded from a
/// checkpoint was trained on one of these two exactly; feeding it the other moves every band.
/// </para>
/// <para>Both return <c>[mels, frequency bins]</c> with <c>nFft / 2 + 1</c> bins, in double precision.</para>
/// <para><b>For Beginners:</b> A mel filterbank turns the hundreds of FFT frequencies into a few dozen bands spaced
/// the way human hearing is. Different libraries draw those bands slightly differently, and a trained model
/// only understands the ones it was trained on.</para>
/// </remarks>
public static class MelFilterbanks
{
    /// <summary>librosa <c>filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False, norm='slaney')</c>.</summary>
    /// <param name="sampleRate">Sample rate in Hz.</param>
    /// <param name="nFft">FFT size.</param>
    /// <param name="mels">Number of bands.</param>
    /// <param name="fMin">Lowest band edge in Hz.</param>
    /// <param name="fMax">Highest band edge in Hz.</param>
    public static double[,] Slaney(int sampleRate, int nFft, int mels, double fMin, double fMax)
    {
        Validate(sampleRate, nFft, mels, fMin, fMax);
        int bins = nFft / 2 + 1;
        var fftFrequencies = new double[bins];
        for (int k = 0; k < bins; k++) fftFrequencies[k] = k * (sampleRate / 2.0) / (bins - 1);
        var edges = MelPoints(mels, fMin, fMax, SlaneyHzToMel, SlaneyMelToHz);
        var weights = Triangles(edges, fftFrequencies, mels);
        for (int m = 0; m < mels; m++)
        {
            double norm = 2.0 / (edges[m + 2] - edges[m]);
            for (int k = 0; k < bins; k++) weights[m, k] *= norm;
        }
        return weights;
    }

    /// <summary>torchaudio <c>functional.melscale_fbanks(n_freqs, f_min, f_max, n_mels, sample_rate,
    /// norm=None, mel_scale="htk")</c>, transposed to <c>[mels, bins]</c>.</summary>
    /// <param name="sampleRate">Sample rate in Hz.</param>
    /// <param name="nFft">FFT size.</param>
    /// <param name="mels">Number of bands.</param>
    /// <param name="fMin">Lowest band edge in Hz.</param>
    /// <param name="fMax">Highest band edge in Hz.</param>
    public static double[,] Htk(int sampleRate, int nFft, int mels, double fMin, double fMax)
    {
        Validate(sampleRate, nFft, mels, fMin, fMax);
        int bins = nFft / 2 + 1;
        // torch.linspace(0, sample_rate // 2, n_freqs): the integer Nyquist, as torchaudio writes it.
        double nyquist = Math.Floor(sampleRate / 2.0);
        var fftFrequencies = new double[bins];
        for (int k = 0; k < bins; k++) fftFrequencies[k] = k * nyquist / (bins - 1);
        var edges = MelPoints(mels, fMin, fMax, HtkHzToMel, HtkMelToHz);
        return Triangles(edges, fftFrequencies, mels);
    }

    /// <summary>HTK mel scale: <c>2595 · log10(1 + f / 700)</c>.</summary>
    public static double HtkHzToMel(double hz) => 2595.0 * Math.Log10(1.0 + hz / 700.0);

    /// <summary>Inverse of <see cref="HtkHzToMel"/>.</summary>
    public static double HtkMelToHz(double mel) => 700.0 * (Math.Pow(10.0, mel / 2595.0) - 1.0);

    private const double SlaneyStep = 200.0 / 3;
    private const double SlaneyLogHz = 1000.0;
    private const double SlaneyLogMel = SlaneyLogHz / SlaneyStep;
    private static readonly double SlaneyLogStep = Math.Log(6.4) / 27.0;

    /// <summary>Slaney's Auditory Toolbox scale (librosa <c>hz_to_mel(htk=False)</c>): linear below 1 kHz,
    /// logarithmic above.</summary>
    public static double SlaneyHzToMel(double hz) =>
        hz >= SlaneyLogHz ? SlaneyLogMel + Math.Log(hz / SlaneyLogHz) / SlaneyLogStep : hz / SlaneyStep;

    /// <summary>Inverse of <see cref="SlaneyHzToMel"/>.</summary>
    public static double SlaneyMelToHz(double mel) =>
        mel >= SlaneyLogMel ? SlaneyLogHz * Math.Exp(SlaneyLogStep * (mel - SlaneyLogMel)) : SlaneyStep * mel;

    private static double[] MelPoints(int mels, double fMin, double fMax, Func<double, double> toMel, Func<double, double> toHz)
    {
        double low = toMel(fMin), high = toMel(fMax);
        var points = new double[mels + 2];
        for (int i = 0; i < mels + 2; i++) points[i] = toHz(low + (high - low) * i / (mels + 1));
        return points;
    }

    // weights[m, k] = max(0, min((f_k - e_m) / (e_{m+1} - e_m), (e_{m+2} - f_k) / (e_{m+2} - e_{m+1})))
    private static double[,] Triangles(double[] edges, double[] frequencies, int mels)
    {
        var weights = new double[mels, frequencies.Length];
        for (int m = 0; m < mels; m++)
        {
            double lowWidth = edges[m + 1] - edges[m], highWidth = edges[m + 2] - edges[m + 1];
            for (int k = 0; k < frequencies.Length; k++)
            {
                double rising = (frequencies[k] - edges[m]) / lowWidth;
                double falling = (edges[m + 2] - frequencies[k]) / highWidth;
                weights[m, k] = Math.Max(0.0, Math.Min(rising, falling));
            }
        }
        return weights;
    }

    private static void Validate(int sampleRate, int nFft, int mels, double fMin, double fMax)
    {
        if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));
        if (nFft <= 1) throw new ArgumentOutOfRangeException(nameof(nFft));
        if (mels <= 0) throw new ArgumentOutOfRangeException(nameof(mels));
        if (fMin < 0 || fMax <= fMin) throw new ArgumentException($"The band [{fMin}, {fMax}] Hz is empty.", nameof(fMax));
    }
}
