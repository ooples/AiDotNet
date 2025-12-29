using System;
using System.Collections.Generic;
using AiDotNet.Diffusion.Audio;
using AiDotNet.LinearAlgebra;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.SourceSeparation;

/// <summary>
/// Music source separation model for separating audio into stems (vocals, drums, bass, other).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This implements a U-Net based source separation approach similar to Spleeter/Demucs.
/// The model separates mixed audio into individual instrument stems using spectral masking.
/// </para>
/// <para><b>For Beginners:</b> Source separation is like unmixing a smoothie:
/// <list type="bullet">
/// <item>Input: Mixed audio with multiple instruments and vocals</item>
/// <item>Output: Separate tracks for vocals, drums, bass, and other instruments</item>
/// <item>Uses neural networks to predict which parts of the spectrum belong to each source</item>
/// </list>
///
/// Usage:
/// <code>
/// var separator = await MusicSourceSeparator&lt;float&gt;.CreateAsync();
///
/// var mixedAudio = LoadAudio("song.wav");
/// var stems = separator.Separate(mixedAudio);
///
/// SaveAudio(stems.Vocals, "vocals.wav");
/// SaveAudio(stems.Drums, "drums.wav");
/// SaveAudio(stems.Bass, "bass.wav");
/// SaveAudio(stems.Other, "other.wav");
/// </code>
/// </para>
/// </remarks>
public class MusicSourceSeparator<T> : IDisposable
{
    private readonly INumericOperations<T> _numOps;
    private readonly SourceSeparationOptions _options;
    private readonly ShortTimeFourierTransform<T> _stft;
    private readonly OnnxModel<T>? _model;
    private bool _disposed;

    /// <summary>
    /// Gets the model options.
    /// </summary>
    public SourceSeparationOptions Options => _options;

    /// <summary>
    /// Gets whether the model is ready for separation.
    /// </summary>
    public bool IsReady => _model?.IsLoaded == true;

    /// <summary>
    /// Creates a new MusicSourceSeparator instance.
    /// </summary>
    private MusicSourceSeparator(
        SourceSeparationOptions options,
        OnnxModel<T>? model)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _options = options;
        _model = model;

        _stft = new ShortTimeFourierTransform<T>(
            nFft: options.FftSize,
            hopLength: options.HopLength);
    }

    /// <summary>
    /// Creates a MusicSourceSeparator asynchronously, downloading models if needed.
    /// </summary>
    public static async Task<MusicSourceSeparator<T>> CreateAsync(
        SourceSeparationOptions? options = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new SourceSeparationOptions();
        OnnxModel<T>? model = null;

        try
        {
            if (options.ModelPath is not null && options.ModelPath.Length > 0)
            {
                model = new OnnxModel<T>(options.ModelPath, options.OnnxOptions);
            }
            else
            {
                var downloader = new OnnxModelDownloader();
                var modelRepo = GetModelRepository(options.StemCount);
                var path = await downloader.DownloadAsync(
                    modelRepo,
                    "model.onnx",
                    progress: progress,
                    cancellationToken);
                model = new OnnxModel<T>(path, options.OnnxOptions);
            }

            return new MusicSourceSeparator<T>(options, model);
        }
        catch
        {
            model?.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Creates a MusicSourceSeparator for CPU-based spectral processing without neural network.
    /// </summary>
    /// <remarks>
    /// This uses traditional signal processing techniques like harmonic-percussive separation.
    /// Less accurate than neural network-based separation but works without external models.
    /// </remarks>
    public static MusicSourceSeparator<T> CreateCpuOnly(SourceSeparationOptions? options = null)
    {
        options ??= new SourceSeparationOptions();
        return new MusicSourceSeparator<T>(options, null);
    }

    /// <summary>
    /// Separates audio into individual stems.
    /// </summary>
    /// <param name="audio">Mixed audio waveform.</param>
    /// <returns>Separated stems.</returns>
    public SeparationResult<T> Separate(Tensor<T> audio)
    {
        ThrowIfDisposed();

        if (_model is not null)
        {
            return SeparateWithModel(audio);
        }
        else
        {
            return SeparateSpectral(audio);
        }
    }

    /// <summary>
    /// Separates audio asynchronously.
    /// </summary>
    public Task<SeparationResult<T>> SeparateAsync(
        Tensor<T> audio,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Separate(audio), cancellationToken);
    }

    private SeparationResult<T> SeparateWithModel(Tensor<T> audio)
    {
        if (_model is null)
            throw new InvalidOperationException("Model not loaded.");

        // Compute STFT of input
        var stft = _stft.Forward(audio);
        var magnitude = ComputeMagnitude(stft);

        // Prepare input for model (add batch dimension)
        var modelInput = new Tensor<T>([1, magnitude.Shape[0], magnitude.Shape[1]]);
        for (int t = 0; t < magnitude.Shape[0]; t++)
        {
            for (int f = 0; f < magnitude.Shape[1]; f++)
            {
                modelInput[0, t, f] = magnitude[t, f];
            }
        }

        // Run model to get masks for each stem
        var masks = _model.Run(modelInput);

        // Apply masks and inverse STFT
        return ApplyMasksAndReconstruct(stft, masks);
    }

    private SeparationResult<T> SeparateSpectral(Tensor<T> audio)
    {
        // Spectral-based Harmonic-Percussive Source Separation (HPSS)
        var stft = _stft.Forward(audio);
        var magnitude = ComputeMagnitude(stft);
        var phase = ComputePhase(stft);

        // Perform HPSS to separate harmonic and percussive
        var (harmonicMag, percussiveMag) = HarmonicPercussiveSeparation(magnitude);

        // Reconstruct signals
        var harmonic = ReconstructFromMagnitudePhase(harmonicMag, phase);
        var percussive = ReconstructFromMagnitudePhase(percussiveMag, phase);

        // For 4-stem, further separate harmonic into vocals and other
        Tensor<T> vocals, other;
        if (_options.StemCount >= 4)
        {
            (vocals, other) = SeparateVocals(harmonic, harmonicMag, phase);
        }
        else
        {
            vocals = harmonic;
            other = new Tensor<T>(audio.Shape);
        }

        return new SeparationResult<T>
        {
            Vocals = vocals,
            Drums = percussive,
            Bass = ExtractBassline(harmonic, harmonicMag, phase),
            Other = other,
            SampleRate = _options.SampleRate
        };
    }

    private Tensor<T> ComputeMagnitude(Tensor<Complex<T>> stft)
    {
        int numFrames = stft.Shape[0];
        int numBins = stft.Shape[1];
        var magnitude = new Tensor<T>([numFrames, numBins]);

        for (int t = 0; t < numFrames; t++)
        {
            for (int f = 0; f < numBins; f++)
            {
                var complex = stft[t, f];
                double real = _numOps.ToDouble(complex.Real);
                double imag = _numOps.ToDouble(complex.Imaginary);
                magnitude[t, f] = _numOps.FromDouble(Math.Sqrt(real * real + imag * imag));
            }
        }

        return magnitude;
    }

    private Tensor<T> ComputePhase(Tensor<Complex<T>> stft)
    {
        int numFrames = stft.Shape[0];
        int numBins = stft.Shape[1];
        var phase = new Tensor<T>([numFrames, numBins]);

        for (int t = 0; t < numFrames; t++)
        {
            for (int f = 0; f < numBins; f++)
            {
                var complex = stft[t, f];
                double real = _numOps.ToDouble(complex.Real);
                double imag = _numOps.ToDouble(complex.Imaginary);
                phase[t, f] = _numOps.FromDouble(Math.Atan2(imag, real));
            }
        }

        return phase;
    }

    private (Tensor<T> harmonic, Tensor<T> percussive) HarmonicPercussiveSeparation(Tensor<T> magnitude)
    {
        int numFrames = magnitude.Shape[0];
        int numBins = magnitude.Shape[1];
        int kernelSize = _options.HpssKernelSize;

        // Median filtering along time (for harmonic) and frequency (for percussive)
        var harmonicEnhanced = MedianFilterTime(magnitude, kernelSize);
        var percussiveEnhanced = MedianFilterFrequency(magnitude, kernelSize);

        // Soft masking based on enhanced magnitudes
        var harmonicMag = new Tensor<T>([numFrames, numBins]);
        var percussiveMag = new Tensor<T>([numFrames, numBins]);

        for (int t = 0; t < numFrames; t++)
        {
            for (int f = 0; f < numBins; f++)
            {
                double h = _numOps.ToDouble(harmonicEnhanced[t, f]);
                double p = _numOps.ToDouble(percussiveEnhanced[t, f]);
                double m = _numOps.ToDouble(magnitude[t, f]);

                double sum = h + p + 1e-10;
                double hMask = h / sum;
                double pMask = p / sum;

                harmonicMag[t, f] = _numOps.FromDouble(m * hMask);
                percussiveMag[t, f] = _numOps.FromDouble(m * pMask);
            }
        }

        return (harmonicMag, percussiveMag);
    }

    private Tensor<T> MedianFilterTime(Tensor<T> input, int kernelSize)
    {
        int numFrames = input.Shape[0];
        int numBins = input.Shape[1];
        var output = new Tensor<T>([numFrames, numBins]);
        int halfKernel = kernelSize / 2;

        for (int f = 0; f < numBins; f++)
        {
            var window = new double[kernelSize];

            for (int t = 0; t < numFrames; t++)
            {
                int count = 0;
                for (int k = -halfKernel; k <= halfKernel; k++)
                {
                    int ti = Math.Max(0, Math.Min(numFrames - 1, t + k));
                    window[count++] = _numOps.ToDouble(input[ti, f]);
                }

                Array.Sort(window, 0, count);
                output[t, f] = _numOps.FromDouble(window[count / 2]);
            }
        }

        return output;
    }

    private Tensor<T> MedianFilterFrequency(Tensor<T> input, int kernelSize)
    {
        int numFrames = input.Shape[0];
        int numBins = input.Shape[1];
        var output = new Tensor<T>([numFrames, numBins]);
        int halfKernel = kernelSize / 2;

        for (int t = 0; t < numFrames; t++)
        {
            var window = new double[kernelSize];

            for (int f = 0; f < numBins; f++)
            {
                int count = 0;
                for (int k = -halfKernel; k <= halfKernel; k++)
                {
                    int fi = Math.Max(0, Math.Min(numBins - 1, f + k));
                    window[count++] = _numOps.ToDouble(input[t, fi]);
                }

                Array.Sort(window, 0, count);
                output[t, f] = _numOps.FromDouble(window[count / 2]);
            }
        }

        return output;
    }

    private Tensor<T> ReconstructFromMagnitudePhase(Tensor<T> magnitude, Tensor<T> phase)
    {
        int numFrames = magnitude.Shape[0];
        int numBins = magnitude.Shape[1];

        // Create complex STFT
        var stft = new Tensor<Complex<T>>([numFrames, numBins]);
        for (int t = 0; t < numFrames; t++)
        {
            for (int f = 0; f < numBins; f++)
            {
                double mag = _numOps.ToDouble(magnitude[t, f]);
                double ph = _numOps.ToDouble(phase[t, f]);
                T real = _numOps.FromDouble(mag * Math.Cos(ph));
                T imag = _numOps.FromDouble(mag * Math.Sin(ph));
                stft[t, f] = new Complex<T>(real, imag);
            }
        }

        // Inverse STFT
        return _stft.Inverse(stft);
    }

    private (Tensor<T> vocals, Tensor<T> other) SeparateVocals(
        Tensor<T> harmonic, Tensor<T> harmonicMag, Tensor<T> phase)
    {
        // Simple vocal separation using frequency band filtering
        // Vocals typically in 300Hz-4000Hz range
        int numFrames = harmonicMag.Shape[0];
        int numBins = harmonicMag.Shape[1];

        double vocalLowBin = 300.0 * _options.FftSize / _options.SampleRate;
        double vocalHighBin = 4000.0 * _options.FftSize / _options.SampleRate;

        var vocalMag = new Tensor<T>([numFrames, numBins]);
        var otherMag = new Tensor<T>([numFrames, numBins]);

        for (int t = 0; t < numFrames; t++)
        {
            for (int f = 0; f < numBins; f++)
            {
                double mag = _numOps.ToDouble(harmonicMag[t, f]);

                // Soft boundary for vocal range
                double vocalWeight = 0;
                if (f >= vocalLowBin && f <= vocalHighBin)
                {
                    vocalWeight = 0.7; // 70% to vocals in range
                }
                else if (f < vocalLowBin && f >= vocalLowBin - 50)
                {
                    vocalWeight = 0.3 * (f - (vocalLowBin - 50)) / 50;
                }
                else if (f > vocalHighBin && f <= vocalHighBin + 100)
                {
                    vocalWeight = 0.3 * (1 - (f - vocalHighBin) / 100);
                }

                vocalMag[t, f] = _numOps.FromDouble(mag * vocalWeight);
                otherMag[t, f] = _numOps.FromDouble(mag * (1 - vocalWeight));
            }
        }

        var vocals = ReconstructFromMagnitudePhase(vocalMag, phase);
        var other = ReconstructFromMagnitudePhase(otherMag, phase);

        return (vocals, other);
    }

    private Tensor<T> ExtractBassline(Tensor<T> harmonic, Tensor<T> harmonicMag, Tensor<T> phase)
    {
        // Bass is typically below 250Hz
        int numFrames = harmonicMag.Shape[0];
        int numBins = harmonicMag.Shape[1];
        double bassMaxBin = 250.0 * _options.FftSize / _options.SampleRate;

        var bassMag = new Tensor<T>([numFrames, numBins]);

        for (int t = 0; t < numFrames; t++)
        {
            for (int f = 0; f < numBins; f++)
            {
                double mag = _numOps.ToDouble(harmonicMag[t, f]);
                double bassWeight = f <= bassMaxBin ? 1.0 : 0.0;
                if (f > bassMaxBin && f <= bassMaxBin + 20)
                {
                    bassWeight = 1.0 - (f - bassMaxBin) / 20;
                }

                bassMag[t, f] = _numOps.FromDouble(mag * bassWeight);
            }
        }

        return ReconstructFromMagnitudePhase(bassMag, phase);
    }

    private SeparationResult<T> ApplyMasksAndReconstruct(Tensor<Complex<T>> stft, Tensor<T> masks)
    {
        // masks expected shape: [batch, stems, time, freq] or [batch, stems, freq, time]
        int numFrames = stft.Shape[0];
        int numBins = stft.Shape[1];
        var phase = ComputePhase(stft);
        var magnitude = ComputeMagnitude(stft);

        var results = new Tensor<T>[4];
        for (int stem = 0; stem < 4; stem++)
        {
            var stemMag = new Tensor<T>([numFrames, numBins]);

            for (int t = 0; t < numFrames; t++)
            {
                for (int f = 0; f < numBins; f++)
                {
                    double mag = _numOps.ToDouble(magnitude[t, f]);
                    double mask = 0;

                    // Try to get mask from model output
                    if (stem < masks.Shape[1] && t < masks.Shape[2] && f < masks.Shape[3])
                    {
                        mask = _numOps.ToDouble(masks[0, stem, t, f]);
                    }

                    stemMag[t, f] = _numOps.FromDouble(mag * Math.Max(0.0, Math.Min(1.0, mask)));
                }
            }

            results[stem] = ReconstructFromMagnitudePhase(stemMag, phase);
        }

        return new SeparationResult<T>
        {
            Vocals = results[0],
            Drums = results[1],
            Bass = results[2],
            Other = results[3],
            SampleRate = _options.SampleRate
        };
    }

    private static string GetModelRepository(int stemCount)
    {
        return stemCount switch
        {
            2 => "deezer/spleeter-2stems-onnx",
            4 => "deezer/spleeter-4stems-onnx",
            5 => "deezer/spleeter-5stems-onnx",
            _ => "deezer/spleeter-4stems-onnx"
        };
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

/// <summary>
/// Options for music source separation.
/// </summary>
public class SourceSeparationOptions
{
    /// <summary>Audio sample rate. Default: 44100.</summary>
    public int SampleRate { get; set; } = 44100;

    /// <summary>FFT size. Default: 4096.</summary>
    public int FftSize { get; set; } = 4096;

    /// <summary>Hop length between frames. Default: 1024.</summary>
    public int HopLength { get; set; } = 1024;

    /// <summary>Number of stems to separate (2, 4, or 5). Default: 4.</summary>
    public int StemCount { get; set; } = 4;

    /// <summary>HPSS kernel size for spectral separation. Default: 31.</summary>
    public int HpssKernelSize { get; set; } = 31;

    /// <summary>Path to ONNX model file (optional).</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX model options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
}

/// <summary>
/// Result of music source separation containing individual stems.
/// </summary>
/// <typeparam name="T">The numeric type used for audio samples.</typeparam>
public class SeparationResult<T>
{
    /// <summary>Isolated vocal track.</summary>
    public required Tensor<T> Vocals { get; init; }

    /// <summary>Isolated drums/percussion track.</summary>
    public required Tensor<T> Drums { get; init; }

    /// <summary>Isolated bass track.</summary>
    public required Tensor<T> Bass { get; init; }

    /// <summary>Other instruments (guitar, piano, etc.).</summary>
    public required Tensor<T> Other { get; init; }

    /// <summary>Sample rate of output stems.</summary>
    public int SampleRate { get; init; }

    /// <summary>Gets all stems as a dictionary.</summary>
    public Dictionary<string, Tensor<T>> ToDictionary()
    {
        return new Dictionary<string, Tensor<T>>
        {
            ["vocals"] = Vocals,
            ["drums"] = Drums,
            ["bass"] = Bass,
            ["other"] = Other
        };
    }
}
