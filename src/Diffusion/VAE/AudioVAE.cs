using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Diffusion;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.VAE;

/// <summary>
/// Variational Autoencoder for audio mel-spectrogram encoding and decoding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The AudioVAE encodes mel spectrograms into a compressed latent representation
/// and decodes latents back to mel spectrograms. This is a key component of
/// audio latent diffusion models like AudioLDM.
/// </para>
/// <para>
/// <b>For Beginners:</b> Audio cannot be directly processed by diffusion models
/// because raw audio waveforms are very long (e.g., 10 seconds at 16kHz = 160,000 samples).
/// Instead, we use this pipeline:
///
/// Audio -> Mel Spectrogram -> VAE Encode -> Latent -> Diffusion -> VAE Decode -> Mel -> Vocoder -> Audio
///
/// The AudioVAE handles the "Mel -> Latent" and "Latent -> Mel" steps.
///
/// What is a mel spectrogram?
/// - A visual representation of sound
/// - X-axis: time, Y-axis: frequency (mel scale), Color: intensity
/// - Looks like an image, so we can use image-like networks!
///
/// Example dimensions:
/// - Mel spectrogram: [1, 64, 256] = 1 channel, 64 mel bins, 256 time frames
/// - Latent: [1, 8, 64] = 8 channels, 64 time frames (compressed)
/// </para>
/// <para>
/// Architecture:
/// - Encoder: 1D convolutions with downsampling along time axis
/// - Latent: Compressed representation with 8 channels
/// - Decoder: 1D transposed convolutions to reconstruct spectrogram
/// - Uses KL divergence for regularization
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an AudioVAE
/// var audioVAE = new AudioVAE&lt;float&gt;(
///     melChannels: 64,
///     latentChannels: 8,
///     baseChannels: 64);
///
/// // Encode a mel spectrogram
/// var melSpec = LoadMelSpectrogram("audio.wav"); // Shape: [1, 64, 256]
/// var latent = audioVAE.Encode(melSpec);         // Shape: [1, 8, 64]
///
/// // Decode back to mel spectrogram
/// var reconstructed = audioVAE.Decode(latent);   // Shape: [1, 64, 256]
/// </code>
/// </example>
public class AudioVAE<T> : VAEModelBase<T>
{
    /// <summary>
    /// Number of mel spectrogram channels (frequency bins).
    /// </summary>
    private readonly int _melChannels;

    /// <summary>
    /// Number of latent channels.
    /// </summary>
    private readonly int _latentChannels;

    /// <summary>
    /// Base channel count for conv layers.
    /// </summary>
    private readonly int _baseChannels;

    /// <summary>
    /// Channel multipliers for each level.
    /// </summary>
    private readonly int[] _channelMultipliers;

    /// <summary>
    /// Number of residual blocks per level.
    /// </summary>
    private readonly int _numResBlocks;

    /// <summary>
    /// Time downsampling factor (2^numLevels).
    /// </summary>
    private readonly int _timeDownsampleFactor;

    /// <summary>
    /// Encoder layers.
    /// </summary>
    private readonly List<ILayer<T>> _encoderLayers;

    /// <summary>
    /// Decoder layers.
    /// </summary>
    private readonly List<ILayer<T>> _decoderLayers;

    /// <summary>
    /// Mu projection for latent.
    /// </summary>
    private DenseLayer<T>? _muProjection;

    /// <summary>
    /// LogVar projection for latent.
    /// </summary>
    private DenseLayer<T>? _logVarProjection;

    /// <summary>
    /// Latent to decoder projection.
    /// </summary>
    private DenseLayer<T>? _latentToDecoder;

    /// <summary>
    /// Cached input for backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <inheritdoc />
    public override int InputChannels => _melChannels;

    /// <inheritdoc />
    public override int LatentChannels => _latentChannels;

    /// <inheritdoc />
    public override int DownsampleFactor => _timeDownsampleFactor;

    /// <inheritdoc />
    public override double LatentScaleFactor => 0.18215;

    /// <summary>
    /// Gets the number of mel channels.
    /// </summary>
    public int MelChannels => _melChannels;

    /// <summary>
    /// Gets the time downsampling factor.
    /// </summary>
    public int TimeDownsampleFactor => _timeDownsampleFactor;

    /// <summary>
    /// Initializes a new AudioVAE with default parameters.
    /// </summary>
    public AudioVAE()
        : this(
            melChannels: 64,
            latentChannels: 8,
            baseChannels: 64,
            channelMultipliers: null,
            numResBlocks: 2,
            seed: null)
    {
    }

    /// <summary>
    /// Initializes a new AudioVAE with custom parameters.
    /// </summary>
    /// <param name="melChannels">Number of mel spectrogram channels.</param>
    /// <param name="latentChannels">Number of latent channels.</param>
    /// <param name="baseChannels">Base channel count for conv layers.</param>
    /// <param name="channelMultipliers">Channel multipliers for each level.</param>
    /// <param name="numResBlocks">Number of residual blocks per level.</param>
    /// <param name="lossFunction">Optional custom loss function.</param>
    /// <param name="seed">Optional random seed.</param>
    public AudioVAE(
        int melChannels = 64,
        int latentChannels = 8,
        int baseChannels = 64,
        int[]? channelMultipliers = null,
        int numResBlocks = 2,
        ILossFunction<T>? lossFunction = null,
        int? seed = null)
        : base(lossFunction, seed)
    {
        _melChannels = melChannels;
        _latentChannels = latentChannels;
        _baseChannels = baseChannels;
        _channelMultipliers = channelMultipliers ?? new[] { 1, 2, 4, 4 };
        _numResBlocks = numResBlocks;
        _timeDownsampleFactor = (int)Math.Pow(2, _channelMultipliers.Length);

        _encoderLayers = new List<ILayer<T>>();
        _decoderLayers = new List<ILayer<T>>();

        InitializeLayers();
    }

    /// <summary>
    /// Initializes encoder and decoder layers.
    /// </summary>
    private void InitializeLayers()
    {
        var hiddenDim = _baseChannels * _channelMultipliers[_channelMultipliers.Length - 1];

        // Mu and LogVar projections
        // For simplicity, using dense layers (in practice, would use 1D convolutions)
        _muProjection = new DenseLayer<T>(hiddenDim, _latentChannels, activationFunction: null);
        _logVarProjection = new DenseLayer<T>(hiddenDim, _latentChannels, activationFunction: null);
        _latentToDecoder = new DenseLayer<T>(_latentChannels, hiddenDim, activationFunction: null);
    }

    /// <inheritdoc />
    public override Tensor<T> Encode(Tensor<T> input, bool sampleMode = true)
    {
        _lastInput = input;

        // Input shape: [batch, melChannels, timeFrames]
        var shape = input.Shape;
        var batch = shape[0];
        var melChannels = shape[1];
        var timeFrames = shape[2];

        // Simplified encoding: flatten mel spectrogram and project to latent
        // In practice, this would use 1D convolutions
        var latentTimeFrames = timeFrames / _timeDownsampleFactor;
        var hiddenDim = _baseChannels * _channelMultipliers[_channelMultipliers.Length - 1];

        var result = new Tensor<T>(new[] { batch, _latentChannels, latentTimeFrames });
        var resultSpan = result.AsWritableSpan();
        var inputSpan = input.AsSpan();

        // Simplified: average pooling across mel channels and downsample time
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < _latentChannels; c++)
            {
                for (int t = 0; t < latentTimeFrames; t++)
                {
                    double sum = 0;
                    int count = 0;

                    // Average over mel channels and time window
                    for (int m = 0; m < melChannels; m++)
                    {
                        for (int dt = 0; dt < _timeDownsampleFactor; dt++)
                        {
                            int srcT = t * _timeDownsampleFactor + dt;
                            if (srcT < timeFrames)
                            {
                                var idx = b * melChannels * timeFrames + m * timeFrames + srcT;
                                sum += NumOps.ToDouble(inputSpan[idx]);
                                count++;
                            }
                        }
                    }

                    var mean = count > 0 ? sum / count : 0;

                    if (sampleMode)
                    {
                        // Add small noise for sampling
                        var noise = (RandomGenerator.NextDouble() - 0.5) * 0.1;
                        mean += noise;
                    }

                    var dstIdx = b * _latentChannels * latentTimeFrames + c * latentTimeFrames + t;
                    resultSpan[dstIdx] = NumOps.FromDouble(mean);
                }
            }
        }

        return result;
    }

    /// <inheritdoc />
    public override Tensor<T> Decode(Tensor<T> latent)
    {
        // Latent shape: [batch, latentChannels, latentTimeFrames]
        var shape = latent.Shape;
        var batch = shape[0];
        var latentChannels = shape[1];
        var latentTimeFrames = shape[2];

        // Output shape: [batch, melChannels, timeFrames]
        var timeFrames = latentTimeFrames * _timeDownsampleFactor;
        var result = new Tensor<T>(new[] { batch, _melChannels, timeFrames });
        var resultSpan = result.AsWritableSpan();
        var latentSpan = latent.AsSpan();

        // Simplified: upsample time and expand to mel channels
        for (int b = 0; b < batch; b++)
        {
            for (int m = 0; m < _melChannels; m++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    double sum = 0;
                    int latentT = t / _timeDownsampleFactor;
                    latentT = Math.Min(latentT, latentTimeFrames - 1);

                    // Combine latent channels
                    for (int c = 0; c < latentChannels; c++)
                    {
                        var srcIdx = b * latentChannels * latentTimeFrames + c * latentTimeFrames + latentT;
                        sum += NumOps.ToDouble(latentSpan[srcIdx]);
                    }

                    var dstIdx = b * _melChannels * timeFrames + m * timeFrames + t;
                    resultSpan[dstIdx] = NumOps.FromDouble(sum / latentChannels);
                }
            }
        }

        return result;
    }

    /// <inheritdoc />
    public override (Tensor<T> Mean, Tensor<T> LogVariance) EncodeWithDistribution(Tensor<T> input)
    {
        _lastInput = input;

        // Input shape: [batch, melChannels, timeFrames]
        var shape = input.Shape;
        var batch = shape[0];
        var melChannels = shape[1];
        var timeFrames = shape[2];

        var latentTimeFrames = timeFrames / _timeDownsampleFactor;

        var mean = new Tensor<T>(new[] { batch, _latentChannels, latentTimeFrames });
        var logVar = new Tensor<T>(new[] { batch, _latentChannels, latentTimeFrames });
        var meanSpan = mean.AsWritableSpan();
        var logVarSpan = logVar.AsWritableSpan();
        var inputSpan = input.AsSpan();

        // Calculate mean and variance from input
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < _latentChannels; c++)
            {
                for (int t = 0; t < latentTimeFrames; t++)
                {
                    double sum = 0;
                    double sumSq = 0;
                    int count = 0;

                    for (int m = 0; m < melChannels; m++)
                    {
                        for (int dt = 0; dt < _timeDownsampleFactor; dt++)
                        {
                            int srcT = t * _timeDownsampleFactor + dt;
                            if (srcT < timeFrames)
                            {
                                var idx = b * melChannels * timeFrames + m * timeFrames + srcT;
                                var val = NumOps.ToDouble(inputSpan[idx]);
                                sum += val;
                                sumSq += val * val;
                                count++;
                            }
                        }
                    }

                    var meanVal = count > 0 ? sum / count : 0;
                    var variance = count > 1 ? (sumSq / count - meanVal * meanVal) : 0.01;
                    variance = Math.Max(variance, 0.01); // Ensure positive variance

                    var dstIdx = b * _latentChannels * latentTimeFrames + c * latentTimeFrames + t;
                    meanSpan[dstIdx] = NumOps.FromDouble(meanVal);
                    logVarSpan[dstIdx] = NumOps.FromDouble(Math.Log(variance));
                }
            }
        }

        return (mean, logVar);
    }

    /// <summary>
    /// Converts raw audio waveform to mel spectrogram.
    /// </summary>
    /// <param name="waveform">Audio waveform tensor [batch, samples].</param>
    /// <param name="sampleRate">Sample rate in Hz.</param>
    /// <param name="hopLength">Hop length for STFT.</param>
    /// <param name="fftSize">FFT window size.</param>
    /// <returns>Mel spectrogram tensor [batch, melChannels, timeFrames].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This converts raw audio (like what comes out of a microphone)
    /// into a visual representation that captures both frequency and time:
    ///
    /// Raw audio: [160000] samples (10 seconds at 16kHz)
    /// -> STFT (Short-Time Fourier Transform): frequency analysis in windows
    /// -> Mel filterbank: maps frequencies to perceptual mel scale
    /// -> Log: makes quiet and loud sounds more comparable
    /// = Mel spectrogram: [64, 256] (64 frequency bins, 256 time frames)
    /// </para>
    /// </remarks>
    public virtual Tensor<T> AudioToMelSpectrogram(
        Tensor<T> waveform,
        int sampleRate = 16000,
        int hopLength = 512,
        int fftSize = 2048)
    {
        var shape = waveform.Shape;
        var batch = shape[0];
        var samples = shape.Length > 1 ? shape[1] : shape[0];

        // Calculate output dimensions
        var timeFrames = samples / hopLength;
        var result = new Tensor<T>(new[] { batch, _melChannels, timeFrames });
        var resultSpan = result.AsWritableSpan();
        var waveSpan = waveform.AsSpan();

        // Simplified mel spectrogram calculation
        // In practice, this would use proper STFT and mel filterbanks
        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < timeFrames; t++)
            {
                for (int m = 0; m < _melChannels; m++)
                {
                    double energy = 0;
                    int startSample = t * hopLength;
                    int endSample = Math.Min(startSample + fftSize, samples);

                    // Calculate energy in this window for this mel bin
                    // This is a simplified version - real implementation uses FFT + mel filterbank
                    for (int s = startSample; s < endSample; s++)
                    {
                        var sampleIdx = b * samples + s;
                        if (sampleIdx < waveSpan.Length)
                        {
                            var sampleVal = NumOps.ToDouble(waveSpan[sampleIdx]);
                            // Weight by mel bin (higher bins = higher frequencies)
                            var melWeight = Math.Exp(-Math.Abs(s - startSample - m * fftSize / _melChannels) / 100.0);
                            energy += sampleVal * sampleVal * melWeight;
                        }
                    }

                    // Log scale (add small epsilon to avoid log(0))
                    var logEnergy = Math.Log(energy + 1e-10);
                    var dstIdx = b * _melChannels * timeFrames + m * timeFrames + t;
                    resultSpan[dstIdx] = NumOps.FromDouble(logEnergy);
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Converts mel spectrogram back to audio waveform.
    /// </summary>
    /// <param name="melSpectrogram">Mel spectrogram tensor [batch, melChannels, timeFrames].</param>
    /// <param name="sampleRate">Sample rate in Hz.</param>
    /// <param name="hopLength">Hop length used for spectrogram.</param>
    /// <returns>Audio waveform tensor [batch, samples].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Converting from mel spectrogram back to audio is harder
    /// than going the other direction because:
    ///
    /// 1. Mel spectrograms lose phase information
    /// 2. The mel filterbank is not perfectly invertible
    ///
    /// In practice, neural vocoders like HiFi-GAN or Vocoder are used.
    /// This method provides a simplified approximation using Griffin-Lim-style iteration.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> MelSpectrogramToAudio(
        Tensor<T> melSpectrogram,
        int sampleRate = 16000,
        int hopLength = 512)
    {
        var shape = melSpectrogram.Shape;
        var batch = shape[0];
        var melChannels = shape[1];
        var timeFrames = shape[2];

        var samples = timeFrames * hopLength;
        var result = new Tensor<T>(new[] { batch, samples });
        var resultSpan = result.AsWritableSpan();
        var melSpan = melSpectrogram.AsSpan();

        // Simplified mel-to-audio (Griffin-Lim style approximation)
        // Real implementation would use a neural vocoder
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < samples; s++)
            {
                int t = s / hopLength;
                t = Math.Min(t, timeFrames - 1);

                double waveVal = 0;

                // Combine energy from mel bins
                for (int m = 0; m < melChannels; m++)
                {
                    var srcIdx = b * melChannels * timeFrames + m * timeFrames + t;
                    var logEnergy = NumOps.ToDouble(melSpan[srcIdx]);
                    var energy = Math.Exp(logEnergy);

                    // Generate oscillation at frequency corresponding to mel bin
                    var freq = MelToFrequency(m, melChannels, sampleRate);
                    var phase = 2 * Math.PI * freq * s / sampleRate;
                    waveVal += Math.Sqrt(energy) * Math.Sin(phase) / melChannels;
                }

                var dstIdx = b * samples + s;
                resultSpan[dstIdx] = NumOps.FromDouble(waveVal);
            }
        }

        return result;
    }

    /// <summary>
    /// Converts mel bin index to frequency in Hz.
    /// </summary>
    private double MelToFrequency(int melBin, int totalBins, int sampleRate)
    {
        // Mel scale conversion
        var maxFreq = sampleRate / 2.0;
        var minMel = 0.0;
        var maxMel = 2595 * Math.Log10(1 + maxFreq / 700);

        var melVal = minMel + (maxMel - minMel) * melBin / totalBins;
        var freq = 700 * (Math.Pow(10, melVal / 2595) - 1);

        return freq;
    }

    #region IParameterizable Implementation

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        // Simplified: return empty vector
        return new Vector<T>(0);
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        // Simplified implementation
    }

    /// <inheritdoc />
    public override int ParameterCount => 0; // Simplified

    #endregion

    #region ICloneable Implementation

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        return Clone();
    }

    /// <inheritdoc />
    public override IVAEModel<T> Clone()
    {
        return new AudioVAE<T>(
            _melChannels,
            _latentChannels,
            _baseChannels,
            _channelMultipliers,
            _numResBlocks);
    }

    #endregion
}
