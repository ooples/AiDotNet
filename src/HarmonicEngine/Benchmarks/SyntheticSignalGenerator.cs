using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.HarmonicEngine.Benchmarks;

/// <summary>
/// Generates controlled synthetic signals for benchmarking and validating HRE components.
/// Each generator produces signals with known spectral properties for rigorous testing.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SyntheticSignalGenerator<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Random _rng;

    /// <summary>
    /// Initializes a new SyntheticSignalGenerator.
    /// </summary>
    /// <param name="seed">Random seed for reproducibility. Null for non-deterministic.</param>
    public SyntheticSignalGenerator(int? seed = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Generates a K-sparse signal in the frequency domain: exactly K nonzero frequency components.
    /// </summary>
    /// <param name="length">Signal length (must be power of 2).</param>
    /// <param name="k">Number of nonzero frequency components.</param>
    /// <param name="snr">Signal-to-noise ratio in dB. Set to double.PositiveInfinity for no noise.</param>
    /// <returns>The generated signal and the ground-truth frequency bins.</returns>
    public (Vector<T> signal, int[] frequencyBins) GenerateKSparse(int length, int k, double snr = 40.0)
    {
        var signal = new Vector<T>(length);
        var bins = new int[k];

        // Pick K random frequency bins
        var available = Enumerable.Range(1, length / 2 - 1).ToList();
        for (int i = 0; i < k && available.Count > 0; i++)
        {
            int idx = _rng.Next(available.Count);
            bins[i] = available[idx];
            available.RemoveAt(idx);
        }

        Array.Sort(bins);

        // Generate signal as sum of K sinusoids
        double signalPower = 0;
        for (int t = 0; t < length; t++)
        {
            double val = 0;
            for (int j = 0; j < k; j++)
            {
                double amplitude = 1.0 + _rng.NextDouble() * 2.0;
                double phase = _rng.NextDouble() * 2.0 * Math.PI;
                val += amplitude * Math.Cos(2 * Math.PI * bins[j] * t / length + phase);
            }
            signal[t] = _numOps.FromDouble(val);
            signalPower += val * val;
        }
        signalPower /= length;

        // Add noise based on SNR
        if (!double.IsPositiveInfinity(snr))
        {
            double noisePower = signalPower / Math.Pow(10, snr / 10);
            double noiseStd = Math.Sqrt(noisePower);
            for (int t = 0; t < length; t++)
            {
                double noise = noiseStd * NextGaussian();
                signal[t] = _numOps.Add(signal[t], _numOps.FromDouble(noise));
            }
        }

        return (signal, bins);
    }

    /// <summary>
    /// Generates a composite time-series with known periodicity: sum of sinusoids + trend + noise.
    /// </summary>
    /// <param name="length">Signal length.</param>
    /// <param name="frequencies">Frequencies (in cycles per signal length) to include.</param>
    /// <param name="amplitudes">Amplitude for each frequency. Must be same length as frequencies.</param>
    /// <param name="trendSlope">Linear trend slope (set to 0 for no trend).</param>
    /// <param name="noiseLevel">Standard deviation of additive Gaussian noise.</param>
    /// <returns>The generated time series.</returns>
    public Vector<T> GenerateComposite(int length, double[] frequencies, double[] amplitudes,
        double trendSlope = 0, double noiseLevel = 0.1)
    {
        var signal = new Vector<T>(length);

        for (int t = 0; t < length; t++)
        {
            double val = trendSlope * t / length;
            for (int j = 0; j < frequencies.Length; j++)
            {
                val += amplitudes[j] * Math.Sin(2 * Math.PI * frequencies[j] * t / length);
            }
            val += noiseLevel * NextGaussian();
            signal[t] = _numOps.FromDouble(val);
        }

        return signal;
    }

    /// <summary>
    /// Generates a periodic character sequence for testing HRESequenceModel.
    /// </summary>
    /// <param name="length">Sequence length.</param>
    /// <param name="period">Repetition period.</param>
    /// <param name="vocabSize">Vocabulary size (characters 0 to vocabSize-1).</param>
    /// <returns>Array of character codes.</returns>
    public int[] GeneratePeriodicSequence(int length, int period, int vocabSize = 26)
    {
        var seq = new int[length];
        for (int i = 0; i < length; i++)
        {
            seq[i] = i % period;
            if (seq[i] >= vocabSize) seq[i] = seq[i] % vocabSize;
        }
        return seq;
    }

    /// <summary>
    /// Generates an AR(p) process with known coefficients for testing Hebbian learning.
    /// </summary>
    /// <param name="length">Signal length.</param>
    /// <param name="coefficients">AR coefficients [a1, a2, ..., ap].</param>
    /// <param name="noiseLevel">Innovation noise standard deviation.</param>
    /// <returns>The generated AR process.</returns>
    public Vector<T> GenerateAR(int length, double[] coefficients, double noiseLevel = 0.1)
    {
        int order = coefficients.Length;
        var signal = new Vector<T>(length);

        for (int t = 0; t < length; t++)
        {
            double val = noiseLevel * NextGaussian();
            for (int j = 0; j < order && t - j - 1 >= 0; j++)
            {
                val += coefficients[j] * _numOps.ToDouble(signal[t - j - 1]);
            }
            signal[t] = _numOps.FromDouble(val);
        }

        return signal;
    }

    private double NextGaussian()
    {
        // Box-Muller transform
        double u1 = 1.0 - _rng.NextDouble();
        double u2 = _rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }
}
