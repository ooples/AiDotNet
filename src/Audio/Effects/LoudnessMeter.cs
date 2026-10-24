namespace AiDotNet.Audio.Effects;

/// <summary>
/// Integrated loudness by ITU-R BS.1770-4 as pyloudnorm 0.1 computes it (<c>pyloudnorm.Meter</c> with K-weighting):
/// the K-weighting filter (a high shelf of +4 dB at 1500 Hz, Q 1/√2, then a high pass at 38 Hz, Q 0.5, both the RBJ
/// biquads pyloudnorm derives for the sample rate), the mean square of 400 ms blocks overlapping by 75 %, an absolute gate
/// at −70 LUFS and a relative gate 10 LU below the gated mean; and pyloudnorm's loudness and peak normalization.
/// </summary>
/// <remarks><para><b>For Beginners:</b> Loudness in LUFS measures how loud audio sounds, weighting frequencies roughly as
/// the ear does; normalizing to a target loudness makes recordings sound equally loud.</para></remarks>
internal static class LoudnessMeter
{
    /// <summary><c>Meter(rate, block_size).integrated_loudness(data)</c> of mono audio; −∞ for silence.</summary>
    public static double IntegratedLoudness(double[] data, int rate, double blockSize = 0.4, double overlap = 0.75)
    {
        if (data is null) throw new ArgumentNullException(nameof(data));
        if (data.Length < blockSize * rate)
            throw new ArgumentException("Audio must have length greater than the block size.", nameof(data));
        var filtered = Filter(Biquad(4.0, 1 / Math.Sqrt(2), 1500.0, rate, highShelf: true), data);
        filtered = Filter(Biquad(0.0, 0.5, 38.0, rate, highShelf: false), filtered);

        double step = 1.0 - overlap;
        double duration = (double)data.Length / rate;
        int blocks = (int)(Math.Round((duration - blockSize) / (blockSize * step), MidpointRounding.ToEven) + 1);
        var z = new double[blocks];
        for (int j = 0; j < blocks; j++)
        {
            int lower = (int)(blockSize * (j * step) * rate);
            int upper = (int)(blockSize * (j * step + 1) * rate);
            double sum = 0;
            for (int i = lower; i < Math.Min(upper, filtered.Length); i++) sum += filtered[i] * filtered[i];
            z[j] = 1.0 / (blockSize * rate) * sum;
        }
        var loudness = z.Select(v => -0.691 + 10.0 * Math.Log10(v)).ToArray();
        double Mean(IEnumerable<double> values)
        {
            var list = values.ToList();
            return list.Count == 0 ? double.NaN : list.Sum() / list.Count;
        }
        var absolute = Enumerable.Range(0, blocks).Where(j => loudness[j] >= -70.0);
        double relative = -0.691 + 10.0 * Math.Log10(Mean(absolute.Select(j => z[j]))) - 10.0;
        double gated = Mean(Enumerable.Range(0, blocks).Where(j => loudness[j] > relative && loudness[j] > -70.0).Select(j => z[j]));
        if (double.IsNaN(gated)) gated = 0;   // np.nan_to_num
        return -0.691 + 10.0 * Math.Log10(gated);
    }

    /// <summary><c>pyloudnorm.normalize.loudness</c>: the gain that moves the measured loudness to the target.</summary>
    public static double[] NormalizeLoudness(double[] data, double inputLoudness, double targetLoudness)
    {
        double gain = Math.Pow(10.0, (targetLoudness - inputLoudness) / 20.0);
        return data.Select(v => gain * v).ToArray();
    }

    /// <summary><c>pyloudnorm.normalize.peak</c>: the gain that moves the peak to the target (dBFS).</summary>
    public static double[] NormalizePeak(double[] data, double target)
    {
        double peak = data.Max(Math.Abs);
        double gain = Math.Pow(10.0, target / 20.0) / peak;
        return data.Select(v => gain * v).ToArray();
    }

    // IIRfilter.generate_coefficients: b and a, divided by a0.
    private static (double[] B, double[] A) Biquad(double gainDb, double q, double frequency, int rate, bool highShelf)
    {
        double a = Math.Pow(10, gainDb / 40.0);
        double w0 = 2.0 * Math.PI * (frequency / rate);
        double alpha = Math.Sin(w0) / (2.0 * q);
        double b0, b1, b2, a0, a1, a2;
        if (highShelf)
        {
            b0 = a * (a + 1 + (a - 1) * Math.Cos(w0) + 2 * Math.Sqrt(a) * alpha);
            b1 = -2 * a * (a - 1 + (a + 1) * Math.Cos(w0));
            b2 = a * (a + 1 + (a - 1) * Math.Cos(w0) - 2 * Math.Sqrt(a) * alpha);
            a0 = a + 1 - (a - 1) * Math.Cos(w0) + 2 * Math.Sqrt(a) * alpha;
            a1 = 2 * (a - 1 - (a + 1) * Math.Cos(w0));
            a2 = a + 1 - (a - 1) * Math.Cos(w0) - 2 * Math.Sqrt(a) * alpha;
        }
        else
        {
            b0 = (1 + Math.Cos(w0)) / 2;
            b1 = -(1 + Math.Cos(w0));
            b2 = (1 + Math.Cos(w0)) / 2;
            a0 = 1 + alpha;
            a1 = -2 * Math.Cos(w0);
            a2 = 1 - alpha;
        }
        return (new[] { b0 / a0, b1 / a0, b2 / a0 }, new[] { a0 / a0, a1 / a0, a2 / a0 });
    }

    // scipy.signal.lfilter: the transposed direct form II.
    private static double[] Filter((double[] B, double[] A) coefficients, double[] x)
    {
        var (b, a) = coefficients;
        var y = new double[x.Length];
        double z1 = 0, z2 = 0;
        for (int n = 0; n < x.Length; n++)
        {
            double output = b[0] * x[n] + z1;
            z1 = b[1] * x[n] + z2 - a[1] * output;
            z2 = b[2] * x[n] - a[2] * output;
            y[n] = output;
        }
        return y;
    }
}
