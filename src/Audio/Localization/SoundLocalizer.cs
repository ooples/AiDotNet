using System;
using System.Collections.Generic;
using System.Numerics;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Localization;

/// <summary>
/// Sound source localization using microphone arrays.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Estimates the direction of arrival (DOA) of sound sources using multiple microphones.
/// Supports various algorithms: GCC-PHAT, MUSIC, SRP-PHAT.
/// </para>
/// <para><b>For Beginners:</b> Sound localization finds where sounds come from:
/// <list type="bullet">
/// <item>Uses 2+ microphones to detect time differences in sound arrival</item>
/// <item>Calculates direction (azimuth/elevation) or 3D position of sound sources</item>
/// <item>Works like how human ears localize sounds</item>
/// </list>
///
/// Usage:
/// <code>
/// // Define microphone array geometry (2 mics, 10cm apart)
/// var micPositions = new double[,] { { -0.05, 0, 0 }, { 0.05, 0, 0 } };
/// var localizer = new SoundLocalizer&lt;float&gt;(micPositions);
///
/// // Audio from each microphone
/// var audioChannels = new[] { ImageHelper&lt;float&gt;.LoadAudio("mic1.wav"), ImageHelper&lt;float&gt;.LoadAudio("mic2.wav") };
/// var result = localizer.Localize(audioChannels);
///
/// // Result is available in the returned value
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.SignalProcessing)]
[ModelTask(ModelTask.Detection)]
[ModelTask(ModelTask.SignalProcessing)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("The Generalized Correlation Method for Estimation of Time Delay", "https://doi.org/10.1109/TASSP.1976.1162830", Year = 1976, Authors = "Charles Knapp, G. Clifford Carter")]
public class SoundLocalizer<T> : IDisposable
{
    private readonly INumericOperations<T> _numOps;
    private readonly SoundLocalizerOptions _options;
    private readonly double[,] _microphonePositions;
    private readonly int _numMicrophones;
    private readonly OnnxModel<T>? _model;
    private bool _disposed;

    /// <summary>
    /// Gets the number of microphones in the array.
    /// </summary>
    public int NumMicrophones => _numMicrophones;

    /// <summary>
    /// Gets the microphone positions (meters).
    /// </summary>
    public double[,] MicrophonePositions => _microphonePositions;

    /// <summary>
    /// Creates a new SoundLocalizer with specified microphone positions.
    /// </summary>
    /// <param name="microphonePositions">Microphone positions as [numMics, 3] array (x, y, z in meters).</param>
    /// <param name="options">Localization options.</param>
    public SoundLocalizer(double[,] microphonePositions, SoundLocalizerOptions? options = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _options = options ?? new SoundLocalizerOptions();
        _microphonePositions = microphonePositions;
        _numMicrophones = microphonePositions.GetLength(0);

        if (_numMicrophones < 2)
            throw new ArgumentException("At least 2 microphones are required.");

        if (_options.ModelPath is not null && _options.ModelPath.Length > 0)
        {
            _model = new OnnxModel<T>(_options.ModelPath, _options.OnnxOptions);
        }
    }

    /// <summary>
    /// Creates a SoundLocalizer for a linear microphone array.
    /// </summary>
    /// <param name="numMicrophones">Number of microphones.</param>
    /// <param name="spacing">Spacing between microphones in meters.</param>
    /// <param name="options">Localization options.</param>
    public static SoundLocalizer<T> CreateLinearArray(
        int numMicrophones,
        double spacing,
        SoundLocalizerOptions? options = null)
    {
        var positions = new double[numMicrophones, 3];
        double startX = -spacing * (numMicrophones - 1) / 2;

        for (int i = 0; i < numMicrophones; i++)
        {
            positions[i, 0] = startX + i * spacing; // X
            positions[i, 1] = 0; // Y
            positions[i, 2] = 0; // Z
        }

        return new SoundLocalizer<T>(positions, options);
    }

    /// <summary>
    /// Creates a SoundLocalizer for a circular microphone array.
    /// </summary>
    /// <param name="numMicrophones">Number of microphones.</param>
    /// <param name="radius">Radius of the circle in meters.</param>
    /// <param name="options">Localization options.</param>
    public static SoundLocalizer<T> CreateCircularArray(
        int numMicrophones,
        double radius,
        SoundLocalizerOptions? options = null)
    {
        var positions = new double[numMicrophones, 3];

        for (int i = 0; i < numMicrophones; i++)
        {
            double angle = 2 * Math.PI * i / numMicrophones;
            positions[i, 0] = radius * Math.Cos(angle); // X
            positions[i, 1] = radius * Math.Sin(angle); // Y
            positions[i, 2] = 0; // Z
        }

        return new SoundLocalizer<T>(positions, options);
    }

    /// <summary>
    /// Localizes sound sources from multi-channel audio.
    /// </summary>
    /// <param name="audioChannels">Audio from each microphone.</param>
    /// <returns>Localization result with estimated direction.</returns>
    public LocalizationResult Localize(Tensor<T>[] audioChannels)
    {
        ThrowIfDisposed();

        if (audioChannels.Length != _numMicrophones)
            throw new ArgumentException($"Expected {_numMicrophones} channels, got {audioChannels.Length}");

        if (_model is not null)
        {
            return LocalizeWithModel(audioChannels);
        }

        return _options.Algorithm switch
        {
            LocalizationAlgorithm.GCCPHAT => LocalizeGccPhat(audioChannels),
            LocalizationAlgorithm.MUSIC => LocalizeMusic(audioChannels),
            LocalizationAlgorithm.SRPPHAT => LocalizeSrpPhat(audioChannels),
            _ => LocalizeGccPhat(audioChannels)
        };
    }

    /// <summary>
    /// Localizes sound sources asynchronously.
    /// </summary>
    public Task<LocalizationResult> LocalizeAsync(
        Tensor<T>[] audioChannels,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Localize(audioChannels), cancellationToken);
    }

    /// <summary>
    /// Estimates time difference of arrival (TDOA) between two channels.
    /// </summary>
    public double EstimateTdoa(Tensor<T> channel1, Tensor<T> channel2)
    {
        return ComputeTdoaGccPhat(channel1, channel2);
    }

    private LocalizationResult LocalizeGccPhat(Tensor<T>[] audioChannels)
    {
        // GCC-PHAT (Generalized Cross-Correlation with Phase Transform)
        // Works well for 2 microphones, estimates azimuth

        // Compute TDOA for first microphone pair
        var tdoa = ComputeTdoaGccPhat(audioChannels[0], audioChannels[1]);

        // Convert TDOA to angle
        double micDistance = GetMicrophoneDistance(0, 1);
        double maxTdoa = micDistance / _options.SpeedOfSound;

        // Clamp to valid range
        double normalizedTdoa = Math.Max(-1.0, Math.Min(1.0, tdoa / maxTdoa));
        double azimuthRad = Math.Asin(normalizedTdoa);
        double azimuthDeg = azimuthRad * 180 / Math.PI;

        // For linear array, we have 180-degree ambiguity
        // Return front hemisphere (-90 to 90)

        return new LocalizationResult
        {
            AzimuthDegrees = azimuthDeg,
            ElevationDegrees = 0, // Cannot estimate with linear array
            TdoaSamples = (int)(tdoa * _options.SampleRate),
            TdoaSeconds = tdoa,
            Confidence = ComputeConfidence(audioChannels[0], audioChannels[1]),
            Algorithm = "GCC-PHAT"
        };
    }

    private LocalizationResult LocalizeMusic(Tensor<T>[] audioChannels)
    {
        // MUSIC (Multiple Signal Classification)
        // Works for multiple sources with more microphones

        int frameSize = _options.FrameSize;
        int numFrames = audioChannels[0].Length / frameSize;
        var covarianceMatrix = new Complex[_numMicrophones, _numMicrophones];

        // Compute spatial covariance matrix
        for (int frame = 0; frame < numFrames; frame++)
        {
            var fftOutputs = new Complex[_numMicrophones][];

            for (int mic = 0; mic < _numMicrophones; mic++)
            {
                fftOutputs[mic] = ComputeFrameFft(audioChannels[mic], frame * frameSize, frameSize);
            }

            // Update covariance matrix
            for (int i = 0; i < _numMicrophones; i++)
            {
                for (int j = 0; j < _numMicrophones; j++)
                {
                    for (int k = 0; k < fftOutputs[i].Length; k++)
                    {
                        covarianceMatrix[i, j] += fftOutputs[i][k] * Complex.Conjugate(fftOutputs[j][k]);
                    }
                }
            }
        }

        // Normalize
        for (int i = 0; i < _numMicrophones; i++)
        {
            for (int j = 0; j < _numMicrophones; j++)
            {
                covarianceMatrix[i, j] /= (numFrames * frameSize);
            }
        }

        // Scan for peak in MUSIC pseudo-spectrum
        double bestAzimuth = 0;
        double bestPower = 0;

        for (double azimuth = -90; azimuth <= 90; azimuth += _options.AngleResolution)
        {
            double power = ComputeMusicSpectrum(covarianceMatrix, azimuth, 0);
            if (power > bestPower)
            {
                bestPower = power;
                bestAzimuth = azimuth;
            }
        }

        return new LocalizationResult
        {
            AzimuthDegrees = bestAzimuth,
            ElevationDegrees = 0,
            TdoaSamples = 0,
            TdoaSeconds = 0,
            Confidence = Math.Min(bestPower / 100, 1.0),
            Algorithm = "MUSIC"
        };
    }

    private LocalizationResult LocalizeSrpPhat(Tensor<T>[] audioChannels)
    {
        // SRP-PHAT (Steered Response Power with Phase Transform)
        // Robust but computationally intensive

        double bestAzimuth = 0;
        double bestPower = 0;

        // Precompute GCC-PHAT for all mic pairs
        var gccResults = new Dictionary<(int, int), double[]>();
        for (int i = 0; i < _numMicrophones; i++)
        {
            for (int j = i + 1; j < _numMicrophones; j++)
            {
                gccResults[(i, j)] = ComputeGccPhat(audioChannels[i], audioChannels[j]);
            }
        }

        // Scan angles
        for (double azimuth = -90; azimuth <= 90; azimuth += _options.AngleResolution)
        {
            double power = 0;

            // Sum contributions from all mic pairs
            foreach (var pair in gccResults)
            {
                int i = pair.Key.Item1;
                int j = pair.Key.Item2;
                double[] gcc = pair.Value;

                double expectedTdoa = ComputeExpectedTdoa(i, j, azimuth, 0);
                int sampleDelay = (int)(expectedTdoa * _options.SampleRate);
                int centerIdx = gcc.Length / 2;
                int idx = centerIdx + sampleDelay;

                if (idx >= 0 && idx < gcc.Length)
                {
                    power += gcc[idx];
                }
            }

            if (power > bestPower)
            {
                bestPower = power;
                bestAzimuth = azimuth;
            }
        }

        return new LocalizationResult
        {
            AzimuthDegrees = bestAzimuth,
            ElevationDegrees = 0,
            TdoaSamples = 0,
            TdoaSeconds = 0,
            Confidence = Math.Min(bestPower / _numMicrophones, 1.0),
            Algorithm = "SRP-PHAT"
        };
    }

    private LocalizationResult LocalizeWithModel(Tensor<T>[] audioChannels)
    {
        if (_model is null)
            throw new InvalidOperationException("Model not loaded.");

        // Concatenate all channels
        var engine = AiDotNetEngine.Current;
        int length = audioChannels[0].Length;
        var channels = new Tensor<T>[_numMicrophones];
        for (int mic = 0; mic < _numMicrophones; mic++)
        {
            if (audioChannels[mic].Length < length)
                throw new ArgumentOutOfRangeException("flatIndex", "Every channel must have at least as many samples as the first.");
            channels[mic] = AudioEngineOps.FitLength(engine, audioChannels[mic], length);
        }
        // The constructor guarantees at least two microphones.
        var input = engine.Reshape(engine.TensorStack(channels, 0), [1, _numMicrophones, length]);

        var output = _model.Run(input);

        // Assume output is [azimuth, elevation, confidence]
        double[] head = output.Length == 0
            ? new double[0]
            : AudioEngineOps.FitLength(engine, output, Math.Min(3, output.Length)).Cast<double>().ToArray();
        double azimuth = head.Length > 0 ? head[0] : 0;
        double elevation = head.Length > 1 ? head[1] : 0;
        double confidence = head.Length > 2 ? head[2] : 0.5;

        return new LocalizationResult
        {
            AzimuthDegrees = azimuth,
            ElevationDegrees = elevation,
            TdoaSamples = 0,
            TdoaSeconds = 0,
            Confidence = confidence,
            Algorithm = "Neural Network"
        };
    }

    private double ComputeTdoaGccPhat(Tensor<T> channel1, Tensor<T> channel2)
    {
        var gcc = ComputeGccPhat(channel1, channel2);

        // Find peak
        int peakIdx = 0;
        double peakVal = double.MinValue;

        for (int i = 0; i < gcc.Length; i++)
        {
            if (gcc[i] > peakVal)
            {
                peakVal = gcc[i];
                peakIdx = i;
            }
        }

        // Convert to TDOA
        int centerIdx = gcc.Length / 2;
        int delaySamples = peakIdx - centerIdx;

        return (double)delaySamples / _options.SampleRate;
    }

    private double[] ComputeGccPhat(Tensor<T> channel1, Tensor<T> channel2)
    {
        int n = channel1.Length;
        int fftSize = NextPowerOf2(n * 2);
        if (channel2.Length < n)
            throw new ArgumentOutOfRangeException("flatIndex", "The second channel is shorter than the first.");

        var engine = AiDotNetEngine.Current;

        // FFT of both channels, zero-padded to fftSize
        var fft1 = engine.NativeComplexFFT(AudioEngineOps.FitLength(engine, channel1, fftSize));
        var fft2 = engine.NativeComplexFFT(AudioEngineOps.FitLength(engine, AudioEngineOps.FitLength(engine, channel2, n), fftSize));
        var re1 = engine.TensorReal(fft1);
        var im1 = engine.TensorImag(fft1);
        var re2 = engine.TensorReal(fft2);
        var im2 = engine.TensorImag(fft2);

        // Cross-power spectrum X1 * conj(X2) with PHAT weighting (bins with |cross| <= 1e-10 stay 0)
        var crossRe = engine.TensorAdd(engine.TensorMultiply(re1, re2), engine.TensorMultiply(im1, im2));
        var crossIm = engine.TensorSubtract(engine.TensorMultiply(im1, re2), engine.TensorMultiply(re1, im2));
        var mag = engine.TensorSqrt(engine.TensorAdd(engine.TensorSquare(crossRe), engine.TensorSquare(crossIm)));
        var zeros = new Tensor<T>([fftSize]);
        var keep = engine.TensorGreaterThan(mag, _numOps.FromDouble(1e-10));
        var phatRe = engine.TensorWhere(keep, engine.TensorDivide(crossRe, mag), zeros);
        var phatIm = engine.TensorWhere(keep, engine.TensorDivide(crossIm, mag), zeros);

        // Inverse FFT (1/N normalised), real part. The IFFT puts zero lag at index 0 and negative lags
        // at the end; rotate by fftSize / 2 so zero lag sits at the center index every consumer reads.
        var correlation = engine.NativeComplexIFFTReal(engine.TensorComplex(phatRe, phatIm));
        return engine.TensorRoll(correlation, [fftSize / 2], [0]).Cast<double>().ToArray();
    }

    private Complex[] ComputeFrameFft(Tensor<T> channel, int start, int size)
    {
        var fft = new Complex[size];

        for (int i = 0; i < size && start + i < channel.Length; i++)
        {
            fft[i] = new Complex(_numOps.ToDouble(channel[start + i]), 0);
        }

        Fft(fft, false);
        return fft;
    }

    private double ComputeMusicSpectrum(Complex[,] covMatrix, double azimuth, double elevation)
    {
        // Simplified MUSIC spectrum calculation
        // In practice, would need eigendecomposition

        double azimuthRad = azimuth * Math.PI / 180;
        double elevationRad = elevation * Math.PI / 180;

        // Unit direction vector
        double dx = Math.Cos(elevationRad) * Math.Sin(azimuthRad);
        double dy = Math.Cos(elevationRad) * Math.Cos(azimuthRad);
        double dz = Math.Sin(elevationRad);

        // Compute steering vector correlation
        double power = 0;

        for (int i = 0; i < _numMicrophones; i++)
        {
            for (int j = 0; j < _numMicrophones; j++)
            {
                double dist_i = dx * _microphonePositions[i, 0] +
                               dy * _microphonePositions[i, 1] +
                               dz * _microphonePositions[i, 2];
                double dist_j = dx * _microphonePositions[j, 0] +
                               dy * _microphonePositions[j, 1] +
                               dz * _microphonePositions[j, 2];

                double phase = 2 * Math.PI * _options.CenterFrequency *
                             (dist_i - dist_j) / _options.SpeedOfSound;

                power += covMatrix[i, j].Magnitude * Math.Cos(phase);
            }
        }

        return power;
    }

    private double ComputeExpectedTdoa(int mic1, int mic2, double azimuth, double elevation)
    {
        double azimuthRad = azimuth * Math.PI / 180;
        double elevationRad = elevation * Math.PI / 180;

        // Unit direction vector
        double dx = Math.Cos(elevationRad) * Math.Sin(azimuthRad);
        double dy = Math.Cos(elevationRad) * Math.Cos(azimuthRad);
        double dz = Math.Sin(elevationRad);

        // Distance difference
        double dist1 = dx * _microphonePositions[mic1, 0] +
                      dy * _microphonePositions[mic1, 1] +
                      dz * _microphonePositions[mic1, 2];
        double dist2 = dx * _microphonePositions[mic2, 0] +
                      dy * _microphonePositions[mic2, 1] +
                      dz * _microphonePositions[mic2, 2];

        return (dist1 - dist2) / _options.SpeedOfSound;
    }

    private double GetMicrophoneDistance(int mic1, int mic2)
    {
        double dx = _microphonePositions[mic1, 0] - _microphonePositions[mic2, 0];
        double dy = _microphonePositions[mic1, 1] - _microphonePositions[mic2, 1];
        double dz = _microphonePositions[mic1, 2] - _microphonePositions[mic2, 2];
        return Math.Sqrt(dx * dx + dy * dy + dz * dz);
    }

    private double ComputeConfidence(Tensor<T> channel1, Tensor<T> channel2)
    {
        // Confidence based on cross-correlation peak sharpness
        var gcc = ComputeGccPhat(channel1, channel2);

        double peak = gcc.Max();
        double mean = gcc.Average();

        return Math.Min(peak / (Math.Abs(mean) + 0.01), 1.0);
    }

    private static void Fft(Complex[] data, bool inverse)
    {
        int n = data.Length;
        if (n == 0) return;

        // Bit reversal
        int j = 0;
        for (int i = 0; i < n - 1; i++)
        {
            if (i < j)
            {
                (data[i], data[j]) = (data[j], data[i]);
            }
            int k = n >> 1;
            while (k <= j)
            {
                j -= k;
                k >>= 1;
            }
            j += k;
        }

        // Cooley-Tukey FFT
        double sign = inverse ? 1 : -1;
        for (int len = 2; len <= n; len <<= 1)
        {
            double angle = sign * 2 * Math.PI / len;
            Complex w = new(Math.Cos(angle), Math.Sin(angle));

            for (int i = 0; i < n; i += len)
            {
                Complex wn = Complex.One;
                for (int k = 0; k < len / 2; k++)
                {
                    Complex u = data[i + k];
                    Complex v = data[i + k + len / 2] * wn;
                    data[i + k] = u + v;
                    data[i + k + len / 2] = u - v;
                    wn *= w;
                }
            }
        }

        if (inverse)
        {
            for (int i = 0; i < n; i++)
            {
                data[i] /= n;
            }
        }
    }

    private static int NextPowerOf2(int n)
    {
        int power = 1;
        while (power < n) power <<= 1;
        return power;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName);
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;

        if (disposing)
        {
            _model?.Dispose();
        }

        _disposed = true;
    }
}
