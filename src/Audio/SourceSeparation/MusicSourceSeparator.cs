using AiDotNet.Diffusion.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
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
/// Usage with ONNX model:
/// <code>
/// var separator = await MusicSourceSeparator&lt;float&gt;.CreateAsync();
/// var stems = separator.Separate(mixedAudio);
/// var vocals = stems.GetSource("vocals");
/// </code>
///
/// Usage for training:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 1025, outputSize: 4*1025);
/// var separator = new MusicSourceSeparator&lt;float&gt;(architecture);
/// separator.Train(mixed, stems);
/// </code>
/// </para>
/// </remarks>
public class MusicSourceSeparator<T> : AudioNeuralNetworkBase<T>, IMusicSourceSeparator<T>
{
    #region Fields

    private readonly SourceSeparationOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private readonly ShortTimeFourierTransform<T> _stft;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly bool _useNativeMode;
    private bool _disposed;

    /// <summary>Standard source names for 4-stem separation.</summary>
    public static readonly string[] StandardSources = ["vocals", "drums", "bass", "other"];

    /// <summary>Source names for 2-stem separation.</summary>
    public static readonly string[] TwoStemSources = ["vocals", "accompaniment"];

    /// <summary>Source names for 5-stem separation.</summary>
    public static readonly string[] FiveStemSources = ["vocals", "drums", "bass", "piano", "other"];

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a MusicSourceSeparator for ONNX inference mode.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="options">Optional configuration options.</param>
    public MusicSourceSeparator(string modelPath, SourceSeparationOptions? options = null)
        : base(CreateMinimalArchitecture(options))
    {
        _options = options ?? new SourceSeparationOptions();
        _useNativeMode = false;
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

        // Set base class properties
        base.SampleRate = _options.SampleRate;

        // Initialize ONNX model
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);

        // Initialize STFT
        _stft = CreateStft();
    }

    /// <summary>
    /// Creates a MusicSourceSeparator for native training mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture.</param>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="optimizer">Optional custom optimizer.</param>
    public MusicSourceSeparator(
        NeuralNetworkArchitecture<T> architecture,
        SourceSeparationOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new SourceSeparationOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

        // Set base class properties
        base.SampleRate = _options.SampleRate;

        // Initialize STFT
        _stft = CreateStft();

        // Initialize layers
        InitializeLayers();
    }

    /// <summary>
    /// Creates a MusicSourceSeparator for CPU-based spectral processing.
    /// </summary>
    /// <param name="options">Optional configuration options.</param>
    public MusicSourceSeparator(SourceSeparationOptions? options = null)
        : base(CreateMinimalArchitecture(options))
    {
        _options = options ?? new SourceSeparationOptions();
        _useNativeMode = false;
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

        // Set base class properties
        base.SampleRate = _options.SampleRate;

        // Initialize ONNX if path provided
        if (_options.ModelPath is string modelPath && !string.IsNullOrEmpty(modelPath))
        {
            OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        }

        // Initialize STFT
        _stft = CreateStft();
    }

    private static NeuralNetworkArchitecture<T> CreateMinimalArchitecture(SourceSeparationOptions? options)
    {
        var opts = options ?? new SourceSeparationOptions();
        // Frequency bins = FFT size / 2 + 1
        int freqBins = opts.FftSize / 2 + 1;
        // Output: masks for each stem
        int outputSize = opts.StemCount * freqBins;
        return new NeuralNetworkArchitecture<T>(inputFeatures: freqBins, outputSize: outputSize);
    }

    private ShortTimeFourierTransform<T> CreateStft()
    {
        return new ShortTimeFourierTransform<T>(
            nFft: _options.FftSize,
            hopLength: _options.HopLength);
    }

    #endregion

    #region Static Factory Methods

    /// <summary>
    /// Creates a MusicSourceSeparator asynchronously, downloading models if needed.
    /// </summary>
    public static async Task<MusicSourceSeparator<T>> CreateAsync(
        SourceSeparationOptions? options = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new SourceSeparationOptions();
        string modelPath = options.ModelPath ?? string.Empty;

        if (string.IsNullOrEmpty(modelPath))
        {
            var downloader = new OnnxModelDownloader();
            var modelRepo = GetModelRepository(options.StemCount);
            modelPath = await downloader.DownloadAsync(
                modelRepo,
                "model.onnx",
                progress: progress,
                cancellationToken);
            options.ModelPath = modelPath;
        }

        return new MusicSourceSeparator<T>(modelPath, options);
    }

    /// <summary>
    /// Creates a MusicSourceSeparator for CPU-based spectral processing without neural network.
    /// </summary>
    public static MusicSourceSeparator<T> CreateCpuOnly(SourceSeparationOptions? options = null)
    {
        return new MusicSourceSeparator<T>(options);
    }

    #endregion

    #region IMusicSourceSeparator Properties

    /// <summary>
    /// Gets the sources this model can separate.
    /// </summary>
    public IReadOnlyList<string> SupportedSources => _options.StemCount switch
    {
        2 => TwoStemSources,
        5 => FiveStemSources,
        _ => StandardSources
    };

    /// <summary>
    /// Gets the number of stems/sources this model produces.
    /// </summary>
    public int NumStems => _options.StemCount;

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the neural network layers.
    /// </summary>
    protected override void InitializeLayers()
    {
        // Use architecture layers if provided
        if (Architecture.Layers is not null && Architecture.Layers.Any())
        {
            foreach (var layer in Architecture.Layers)
            {
                Layers.Add(layer);
            }
            return;
        }

        // Create default source separation layers (U-Net style)
        // numMels is FFT bins which is FftSize/2 + 1
        int numMels = _options.FftSize / 2 + 1;
        var layers = LayerHelper<T>.CreateDefaultSourceSeparationLayers(
            numMels: numMels,
            baseChannels: 32,
            numSources: _options.StemCount,
            maxFrames: 512,
            dropoutRate: 0.1);
        foreach (var layer in layers)
        {
            Layers.Add(layer);
        }
    }

    #endregion

    #region IMusicSourceSeparator Methods

    /// <summary>
    /// Separates all sources from the audio mix.
    /// </summary>
    public SourceSeparationResult<T> Separate(Tensor<T> audio)
    {
        ThrowIfDisposed();

        if (IsOnnxMode && OnnxEncoder is not null)
        {
            return SeparateWithModel(audio);
        }
        else if (_useNativeMode && Layers.Count > 0)
        {
            return SeparateWithNativeNetwork(audio);
        }
        else
        {
            return SeparateSpectral(audio);
        }
    }

    /// <summary>
    /// Separates all sources asynchronously.
    /// </summary>
    public Task<SourceSeparationResult<T>> SeparateAsync(
        Tensor<T> audio,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Separate(audio), cancellationToken);
    }

    /// <summary>
    /// Extracts a specific source from the mix.
    /// </summary>
    public Tensor<T> ExtractSource(Tensor<T> audio, string source)
    {
        var result = Separate(audio);
        return result.GetSource(source);
    }

    /// <summary>
    /// Removes a specific source from the mix.
    /// </summary>
    public Tensor<T> RemoveSource(Tensor<T> audio, string source)
    {
        var result = Separate(audio);
        var sources = result.Sources;

        // Sum all sources except the one to remove
        Tensor<T>? output = null;
        foreach (var kvp in sources)
        {
            if (kvp.Key == source) continue;

            if (output is null)
            {
                output = new Tensor<T>(kvp.Value.Shape);
                for (int i = 0; i < output.Length; i++)
                {
                    output[i] = kvp.Value[i];
                }
            }
            else
            {
                for (int i = 0; i < output.Length; i++)
                {
                    output[i] = NumOps.Add(output[i], kvp.Value[i]);
                }
            }
        }

        return output ?? new Tensor<T>(audio.Shape);
    }

    /// <summary>
    /// Gets the soft mask for a specific source.
    /// </summary>
    public Tensor<T> GetSourceMask(Tensor<T> audio, string source)
    {
        ThrowIfDisposed();

        // Compute STFT
        var stft = _stft.Forward(audio);
        var magnitude = ComputeMagnitude(stft);

        // Prepare input and get all masks
        var modelInput = new Tensor<T>([1, magnitude.Shape[0], magnitude.Shape[1]]);
        for (int t = 0; t < magnitude.Shape[0]; t++)
        {
            for (int f = 0; f < magnitude.Shape[1]; f++)
            {
                modelInput[0, t, f] = magnitude[t, f];
            }
        }

        // Get masks from prediction
        var masks = Predict(modelInput);

        // Find the index of the requested source
        int sourceIndex = SupportedSources.ToList().IndexOf(source);
        if (sourceIndex < 0)
        {
            throw new ArgumentException($"Unknown source: {source}. Supported: {string.Join(", ", SupportedSources)}");
        }

        // Extract mask for the specific source
        var sourceMask = new Tensor<T>([magnitude.Shape[0], magnitude.Shape[1]]);
        int numBins = magnitude.Shape[1];

        for (int t = 0; t < magnitude.Shape[0]; t++)
        {
            for (int f = 0; f < numBins; f++)
            {
                if (masks.Shape.Length >= 4 && sourceIndex < masks.Shape[1])
                {
                    sourceMask[t, f] = masks[0, sourceIndex, t, f];
                }
                else
                {
                    // Fallback: equal mask
                    sourceMask[t, f] = NumOps.FromDouble(1.0 / NumStems);
                }
            }
        }

        return sourceMask;
    }

    /// <summary>
    /// Remixes the separated sources with custom volumes.
    /// </summary>
    public Tensor<T> Remix(SourceSeparationResult<T> separationResult, IReadOnlyDictionary<string, double> sourceVolumes)
    {
        var sources = separationResult.Sources;
        Tensor<T>? output = null;

        foreach (var kvp in sources)
        {
            string sourceName = kvp.Key;
            double volume = sourceVolumes.TryGetValue(sourceName, out var v) ? v : 1.0;

            if (Math.Abs(volume) < 1e-10) continue;

            if (output is null)
            {
                output = new Tensor<T>(kvp.Value.Shape);
            }

            for (int i = 0; i < output.Length && i < kvp.Value.Length; i++)
            {
                double val = NumOps.ToDouble(kvp.Value[i]) * volume;
                output[i] = NumOps.Add(output[i], NumOps.FromDouble(val));
            }
        }

        return output ?? new Tensor<T>(separationResult.OriginalMix.Shape);
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Predicts source masks from spectrogram magnitude.
    /// </summary>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        if (IsOnnxMode && OnnxEncoder is not null)
        {
            return OnnxEncoder.Run(input);
        }

        // Native mode - use layers
        if (!_useNativeMode || Layers.Count == 0)
        {
            // Return uniform masks as fallback
            return CreateUniformMasks(input);
        }

        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }
        return current;
    }

    /// <summary>
    /// Trains the model on mixed audio and ground truth stems.
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (!_useNativeMode)
        {
            throw new InvalidOperationException(
                "Training is not supported in ONNX inference mode. " +
                "Create the model with NeuralNetworkArchitecture for training.");
        }

        SetTrainingMode(true);

        // Forward pass
        var output = Predict(input);

        // Calculate loss gradient
        var gradient = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());
        var gradientTensor = Tensor<T>.FromVector(gradient);

        // Backward pass through layers
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradientTensor = Layers[i].Backward(gradientTensor);
        }

        // Update parameters
        _optimizer.UpdateParameters(Layers);

        SetTrainingMode(false);
    }

    /// <summary>
    /// Updates parameters from a flattened parameter vector.
    /// </summary>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("UpdateParameters is not supported in ONNX mode.");

        int index = 0;
        foreach (var layer in Layers)
        {
            int count = layer.ParameterCount;
            var layerParams = parameters.SubVector(index, count);
            layer.UpdateParameters(layerParams);
            index += count;
        }
    }

    /// <summary>
    /// Preprocesses raw audio into spectrogram format.
    /// </summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        var stft = _stft.Forward(rawAudio);
        return ComputeMagnitude(stft);
    }

    /// <summary>
    /// Postprocesses model output (applies sigmoid to mask values).
    /// </summary>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        var result = new Tensor<T>(modelOutput.Shape);
        for (int i = 0; i < modelOutput.Length; i++)
        {
            double val = NumOps.ToDouble(modelOutput[i]);
            // Sigmoid to ensure mask values are in [0, 1]
            result[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-val)));
        }
        return result;
    }

    /// <summary>
    /// Gets model metadata for serialization.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "MusicSourceSeparator-Native" : "MusicSourceSeparator-ONNX",
            Description = "Music source separation model (Spleeter/Demucs-style)",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = _options.FftSize / 2 + 1,
            Complexity = 1
        };
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        metadata.AdditionalInfo["FftSize"] = _options.FftSize.ToString();
        metadata.AdditionalInfo["StemCount"] = _options.StemCount.ToString();
        return metadata;
    }

    /// <summary>
    /// Serializes network-specific data.
    /// </summary>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_options.SampleRate);
        writer.Write(_options.FftSize);
        writer.Write(_options.HopLength);
        writer.Write(_options.StemCount);
        writer.Write(_options.HpssKernelSize);
        writer.Write(_useNativeMode);
    }

    /// <summary>
    /// Deserializes network-specific data.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Simplified - options would need to be reconstructed
    }

    /// <summary>
    /// Creates a new instance of this network type.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new MusicSourceSeparator<T>(Architecture, _options);
    }

    #endregion

    #region Separation Methods

    private SourceSeparationResult<T> SeparateWithModel(Tensor<T> audio)
    {
        if (OnnxEncoder is null)
            throw new InvalidOperationException("Model not loaded.");

        // Compute STFT of input
        var stft = _stft.Forward(audio);
        var magnitude = ComputeMagnitude(stft);
        var phase = ComputePhase(stft);

        // Prepare input for model
        var modelInput = new Tensor<T>([1, magnitude.Shape[0], magnitude.Shape[1]]);
        for (int t = 0; t < magnitude.Shape[0]; t++)
        {
            for (int f = 0; f < magnitude.Shape[1]; f++)
            {
                modelInput[0, t, f] = magnitude[t, f];
            }
        }

        // Run model to get masks
        var masks = OnnxEncoder.Run(modelInput);

        // Apply masks and reconstruct
        return ApplyMasksAndReconstruct(audio, stft, magnitude, phase, masks);
    }

    private SourceSeparationResult<T> SeparateWithNativeNetwork(Tensor<T> audio)
    {
        var stft = _stft.Forward(audio);
        var magnitude = ComputeMagnitude(stft);
        var phase = ComputePhase(stft);

        var modelInput = new Tensor<T>([1, magnitude.Shape[0], magnitude.Shape[1]]);
        for (int t = 0; t < magnitude.Shape[0]; t++)
        {
            for (int f = 0; f < magnitude.Shape[1]; f++)
            {
                modelInput[0, t, f] = magnitude[t, f];
            }
        }

        var masks = Predict(modelInput);
        masks = PostprocessOutput(masks);

        return ApplyMasksAndReconstruct(audio, stft, magnitude, phase, masks);
    }

    private SourceSeparationResult<T> SeparateSpectral(Tensor<T> audio)
    {
        var stft = _stft.Forward(audio);
        var magnitude = ComputeMagnitude(stft);
        var phase = ComputePhase(stft);

        // Perform HPSS
        var (harmonicMag, percussiveMag) = HarmonicPercussiveSeparation(magnitude);

        // Reconstruct signals
        var harmonic = ReconstructFromMagnitudePhase(harmonicMag, phase);
        var percussive = ReconstructFromMagnitudePhase(percussiveMag, phase);

        Tensor<T> vocals, other;
        if (_options.StemCount >= 4)
        {
            (vocals, other) = SeparateVocals(harmonicMag, phase);
        }
        else
        {
            vocals = harmonic;
            other = new Tensor<T>(audio.Shape);
        }

        var bass = ExtractBassline(harmonicMag, phase);

        var sources = new Dictionary<string, Tensor<T>>
        {
            ["vocals"] = vocals,
            ["drums"] = percussive,
            ["bass"] = bass,
            ["other"] = other
        };

        return new SourceSeparationResult<T>
        {
            Sources = sources,
            OriginalMix = audio,
            SampleRate = _options.SampleRate,
            Duration = (double)audio.Length / _options.SampleRate
        };
    }

    private SourceSeparationResult<T> ApplyMasksAndReconstruct(
        Tensor<T> audio,
        Tensor<Complex<T>> stft,
        Tensor<T> magnitude,
        Tensor<T> phase,
        Tensor<T> masks)
    {
        int numFrames = stft.Shape[0];
        int numBins = stft.Shape[1];

        var sources = new Dictionary<string, Tensor<T>>();
        var sourceNames = SupportedSources;

        for (int stem = 0; stem < NumStems && stem < sourceNames.Count; stem++)
        {
            var stemMag = new Tensor<T>([numFrames, numBins]);

            for (int t = 0; t < numFrames; t++)
            {
                for (int f = 0; f < numBins; f++)
                {
                    double mag = NumOps.ToDouble(magnitude[t, f]);
                    double mask = 0;

                    if (masks.Shape.Length >= 4 && stem < masks.Shape[1] && t < masks.Shape[2] && f < masks.Shape[3])
                    {
                        mask = NumOps.ToDouble(masks[0, stem, t, f]);
                    }
                    else if (masks.Shape.Length >= 3 && stem < masks.Shape[0])
                    {
                        mask = NumOps.ToDouble(masks[stem, t, f]);
                    }

                    stemMag[t, f] = NumOps.FromDouble(mag * Math.Max(0.0, Math.Min(1.0, mask)));
                }
            }

            sources[sourceNames[stem]] = ReconstructFromMagnitudePhase(stemMag, phase);
        }

        return new SourceSeparationResult<T>
        {
            Sources = sources,
            OriginalMix = audio,
            SampleRate = _options.SampleRate,
            Duration = (double)audio.Length / _options.SampleRate
        };
    }

    #endregion

    #region Signal Processing Helpers

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
                double real = NumOps.ToDouble(complex.Real);
                double imag = NumOps.ToDouble(complex.Imaginary);
                magnitude[t, f] = NumOps.FromDouble(Math.Sqrt(real * real + imag * imag));
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
                double real = NumOps.ToDouble(complex.Real);
                double imag = NumOps.ToDouble(complex.Imaginary);
                phase[t, f] = NumOps.FromDouble(Math.Atan2(imag, real));
            }
        }

        return phase;
    }

    private (Tensor<T> harmonic, Tensor<T> percussive) HarmonicPercussiveSeparation(Tensor<T> magnitude)
    {
        int numFrames = magnitude.Shape[0];
        int numBins = magnitude.Shape[1];
        int kernelSize = _options.HpssKernelSize;

        var harmonicEnhanced = MedianFilterTime(magnitude, kernelSize);
        var percussiveEnhanced = MedianFilterFrequency(magnitude, kernelSize);

        var harmonicMag = new Tensor<T>([numFrames, numBins]);
        var percussiveMag = new Tensor<T>([numFrames, numBins]);

        for (int t = 0; t < numFrames; t++)
        {
            for (int f = 0; f < numBins; f++)
            {
                double h = NumOps.ToDouble(harmonicEnhanced[t, f]);
                double p = NumOps.ToDouble(percussiveEnhanced[t, f]);
                double m = NumOps.ToDouble(magnitude[t, f]);

                double sum = h + p + 1e-10;
                harmonicMag[t, f] = NumOps.FromDouble(m * h / sum);
                percussiveMag[t, f] = NumOps.FromDouble(m * p / sum);
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
        int windowSize = (2 * halfKernel) + 1;

        for (int f = 0; f < numBins; f++)
        {
            var window = new double[windowSize];

            for (int t = 0; t < numFrames; t++)
            {
                int count = 0;
                for (int k = -halfKernel; k <= halfKernel; k++)
                {
                    int ti = Math.Max(0, Math.Min(numFrames - 1, t + k));
                    window[count++] = NumOps.ToDouble(input[ti, f]);
                }

                Array.Sort(window, 0, count);
                output[t, f] = NumOps.FromDouble(window[count / 2]);
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
        int windowSize = (2 * halfKernel) + 1;

        for (int t = 0; t < numFrames; t++)
        {
            var window = new double[windowSize];

            for (int f = 0; f < numBins; f++)
            {
                int count = 0;
                for (int k = -halfKernel; k <= halfKernel; k++)
                {
                    int fi = Math.Max(0, Math.Min(numBins - 1, f + k));
                    window[count++] = NumOps.ToDouble(input[t, fi]);
                }

                Array.Sort(window, 0, count);
                output[t, f] = NumOps.FromDouble(window[count / 2]);
            }
        }

        return output;
    }

    private Tensor<T> ReconstructFromMagnitudePhase(Tensor<T> magnitude, Tensor<T> phase)
    {
        int numFrames = magnitude.Shape[0];
        int numBins = magnitude.Shape[1];

        var stft = new Tensor<Complex<T>>([numFrames, numBins]);
        for (int t = 0; t < numFrames; t++)
        {
            for (int f = 0; f < numBins; f++)
            {
                double mag = NumOps.ToDouble(magnitude[t, f]);
                double ph = NumOps.ToDouble(phase[t, f]);
                T real = NumOps.FromDouble(mag * Math.Cos(ph));
                T imag = NumOps.FromDouble(mag * Math.Sin(ph));
                stft[t, f] = new Complex<T>(real, imag);
            }
        }

        return _stft.Inverse(stft);
    }

    private (Tensor<T> vocals, Tensor<T> other) SeparateVocals(Tensor<T> harmonicMag, Tensor<T> phase)
    {
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
                double mag = NumOps.ToDouble(harmonicMag[t, f]);
                double vocalWeight = 0;

                if (f >= vocalLowBin && f <= vocalHighBin)
                    vocalWeight = 0.7;
                else if (f < vocalLowBin && f >= vocalLowBin - 50)
                    vocalWeight = 0.3 * (f - (vocalLowBin - 50)) / 50;
                else if (f > vocalHighBin && f <= vocalHighBin + 100)
                    vocalWeight = 0.3 * (1 - (f - vocalHighBin) / 100);

                vocalMag[t, f] = NumOps.FromDouble(mag * vocalWeight);
                otherMag[t, f] = NumOps.FromDouble(mag * (1 - vocalWeight));
            }
        }

        return (ReconstructFromMagnitudePhase(vocalMag, phase), ReconstructFromMagnitudePhase(otherMag, phase));
    }

    private Tensor<T> ExtractBassline(Tensor<T> harmonicMag, Tensor<T> phase)
    {
        int numFrames = harmonicMag.Shape[0];
        int numBins = harmonicMag.Shape[1];
        double bassMaxBin = 250.0 * _options.FftSize / _options.SampleRate;

        var bassMag = new Tensor<T>([numFrames, numBins]);

        for (int t = 0; t < numFrames; t++)
        {
            for (int f = 0; f < numBins; f++)
            {
                double mag = NumOps.ToDouble(harmonicMag[t, f]);
                double bassWeight = f <= bassMaxBin ? 1.0 : 0.0;
                if (f > bassMaxBin && f <= bassMaxBin + 20)
                    bassWeight = 1.0 - (f - bassMaxBin) / 20;

                bassMag[t, f] = NumOps.FromDouble(mag * bassWeight);
            }
        }

        return ReconstructFromMagnitudePhase(bassMag, phase);
    }

    private Tensor<T> CreateUniformMasks(Tensor<T> input)
    {
        // Create uniform masks for fallback
        int numFrames = input.Shape.Length > 2 ? input.Shape[1] : input.Shape[0];
        int numBins = input.Shape.Length > 2 ? input.Shape[2] : input.Shape[1];

        var masks = new Tensor<T>([1, NumStems, numFrames, numBins]);
        double uniformValue = 1.0 / NumStems;

        for (int stem = 0; stem < NumStems; stem++)
        {
            for (int t = 0; t < numFrames; t++)
            {
                for (int f = 0; f < numBins; f++)
                {
                    masks[0, stem, t, f] = NumOps.FromDouble(uniformValue);
                }
            }
        }

        return masks;
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

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(GetType().FullName ?? nameof(MusicSourceSeparator<T>));
        }
    }

    /// <summary>
    /// Disposes of managed resources.
    /// </summary>
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;

        if (disposing)
        {
            OnnxEncoder?.Dispose();
        }

        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
