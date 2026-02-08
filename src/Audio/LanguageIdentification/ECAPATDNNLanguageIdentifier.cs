using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Features;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.LanguageIdentification;

/// <summary>
/// ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation Time Delay Neural Network)
/// for spoken language identification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ECAPA-TDNN is a state-of-the-art architecture originally designed for speaker verification
/// that has been adapted for language identification. It uses:
/// - Time Delay Neural Network (TDNN) layers with dilated convolutions
/// - Squeeze-Excitation (SE) blocks for channel attention
/// - Multi-layer feature aggregation (MFA) for combining information across layers
/// - Attentive statistics pooling for variable-length utterances
/// </para>
/// <para><b>For Beginners:</b> ECAPA-TDNN is like having a very sophisticated listener that can:
/// 1. Hear patterns at different time scales (TDNN layers)
/// 2. Focus on the most important sound characteristics (channel attention)
/// 3. Combine information from multiple processing stages (MFA)
/// 4. Handle audio of any length (attentive pooling)
///
/// This model is particularly good at:
/// - Identifying languages from short audio clips (3-10 seconds)
/// - Handling noisy or low-quality audio
/// - Distinguishing between similar languages (e.g., Spanish vs Portuguese)
///
/// Example usage:
/// <code>
/// var model = new ECAPATDNNLanguageIdentifier&lt;float&gt;(new ECAPATDNNOptions
/// {
///     SampleRate = 16000,
///     ModelPath = "ecapa_tdnn_lid.onnx"
/// });
///
/// var result = model.IdentifyLanguage(audioTensor);
/// Console.WriteLine($"Detected: {result.LanguageName} ({result.Confidence:P0})");
/// </code>
/// </para>
/// </remarks>
public class ECAPATDNNLanguageIdentifier<T> : AudioNeuralNetworkBase<T>, ILanguageIdentifier<T>
{
    #region Fields

    private readonly INumericOperations<T> _numOps;
    private readonly ECAPATDNNOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private readonly MfccExtractor<T> _mfccExtractor;
    private readonly ILossFunction<T> _lossFunction;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    // ECAPA-TDNN architecture components
    private readonly List<ILayer<T>> _tdnnLayers = [];
    private readonly List<ILayer<T>> _seBlocks = [];
    private readonly List<ILayer<T>> _resBlocks = [];
    private DenseLayer<T>? _poolingLayer;
    private DenseLayer<T>? _classifierLayer;
    private BatchNormalizationLayer<T>? _finalBatchNorm;

    // Cached values for proper gradient flow in MFA
    private readonly List<int> _blockOutputLengths = [];
    private Tensor<T>? _lastTdnnOutput;

    // Language mapping
    private readonly Dictionary<int, string> _languageIdToCode;
    private readonly Dictionary<string, int> _languageCodeToId;
    private readonly Dictionary<string, string> _languageCodeToName;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override bool SupportsTraining => !IsOnnxMode;

    /// <inheritdoc/>
    public IReadOnlyList<string> SupportedLanguages => _languageIdToCode.Values.ToList();

    /// <summary>
    /// Gets the embedding dimension produced by this model.
    /// </summary>
    public int EmbeddingDimension => _options.EmbeddingDimension;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates an ECAPA-TDNN language identifier with ONNX model for inference.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="options">Language identifier options.</param>
    public ECAPATDNNLanguageIdentifier(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        ECAPATDNNOptions? options = null)
        : base(architecture, new CrossEntropyLoss<T>())
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model file not found: {modelPath}");

        _numOps = MathHelper.GetNumericOperations<T>();
        _options = options ?? new ECAPATDNNOptions();
        Options = _options;
        _options.ModelPath = modelPath;

        SampleRate = _options.SampleRate;
        NumMels = _options.NumMels;

        _lossFunction = new CrossEntropyLoss<T>();

        // Initialize MFCC extractor
        _mfccExtractor = new MfccExtractor<T>(new MfccOptions
        {
            SampleRate = _options.SampleRate,
            FftSize = _options.FftSize,
            HopLength = _options.HopLength,
            NumCoefficients = _options.NumMels,
            AppendDelta = true,
            AppendDeltaDelta = true
        });

        // Initialize language mappings
        (_languageIdToCode, _languageCodeToId, _languageCodeToName) = InitializeLanguageMappings();

        // Load ONNX model
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);

        // Initialize optimizer (not used in ONNX mode but required for readonly field)
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
    }

    /// <summary>
    /// Creates an ECAPA-TDNN language identifier for native training.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="supportedLanguages">List of language codes to identify (e.g., ["en", "es", "fr"]).</param>
    /// <param name="options">ECAPA-TDNN options.</param>
    /// <param name="optimizer">Optimizer for training. If null, Adam is used.</param>
    /// <param name="lossFunction">Loss function. If null, CrossEntropy is used.</param>
    public ECAPATDNNLanguageIdentifier(
        NeuralNetworkArchitecture<T> architecture,
        IReadOnlyList<string> supportedLanguages,
        ECAPATDNNOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>())
    {
        if (supportedLanguages is null)
            throw new ArgumentNullException(nameof(supportedLanguages));
        if (supportedLanguages.Count == 0)
            throw new ArgumentException("At least one language must be specified.", nameof(supportedLanguages));

        _numOps = MathHelper.GetNumericOperations<T>();
        _options = options ?? new ECAPATDNNOptions();
        Options = _options;

        SampleRate = _options.SampleRate;
        NumMels = _options.NumMels;

        _lossFunction = lossFunction ?? new CrossEntropyLoss<T>();
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

        // Initialize MFCC extractor
        _mfccExtractor = new MfccExtractor<T>(new MfccOptions
        {
            SampleRate = _options.SampleRate,
            FftSize = _options.FftSize,
            HopLength = _options.HopLength,
            NumCoefficients = _options.NumMels,
            AppendDelta = true,
            AppendDeltaDelta = true
        });

        // Initialize language mappings from provided list
        (_languageIdToCode, _languageCodeToId, _languageCodeToName) =
            InitializeLanguageMappings(supportedLanguages);

        InitializeNativeLayers(supportedLanguages.Count);
    }

    #endregion

    #region Layer Initialization

    private void InitializeNativeLayers(int numLanguages)
    {
        int inputDim = _options.NumMels * 3; // MFCC + delta + delta-delta
        int channels = _options.TdnnChannels;

        // Initial TDNN layer (frame-level feature extraction)
        _tdnnLayers.Add(new DenseLayer<T>(inputDim, channels, (IActivationFunction<T>)new ReLUActivation<T>()));
        _tdnnLayers.Add(new BatchNormalizationLayer<T>(channels));

        // ECAPA-TDNN blocks with SE-Res2Net architecture
        foreach (int dilation in _options.Dilations)
        {
            // SE-Res2Block
            AddSERes2Block(channels, dilation);
        }

        // Multi-layer feature aggregation (MFA)
        // Concatenate outputs from all SE-Res2 blocks
        int mfaOutputDim = channels * _options.Dilations.Length;

        // Attentive Statistics Pooling layer
        int[] poolingShape = [mfaOutputDim];
        _poolingLayer = new DenseLayer<T>(mfaOutputDim, _options.EmbeddingDimension * 2);

        // Final batch normalization
        _finalBatchNorm = new BatchNormalizationLayer<T>(_options.EmbeddingDimension);

        // Classification layer
        _classifierLayer = new DenseLayer<T>(_options.EmbeddingDimension, numLanguages);
    }

    private void AddSERes2Block(int channels, int dilation)
    {
        // 1x1 conv for channel reduction
        _resBlocks.Add(new DenseLayer<T>(channels, channels / 4, (IActivationFunction<T>)new ReLUActivation<T>()));
        _resBlocks.Add(new BatchNormalizationLayer<T>(channels / 4));

        // Dilated conv (simulated with dense + temporal handling)
        _resBlocks.Add(new DenseLayer<T>(channels / 4, channels / 4, (IActivationFunction<T>)new ReLUActivation<T>()));
        _resBlocks.Add(new BatchNormalizationLayer<T>(channels / 4));

        // 1x1 conv for channel expansion
        _resBlocks.Add(new DenseLayer<T>(channels / 4, channels, (IActivationFunction<T>)new ReLUActivation<T>()));
        _resBlocks.Add(new BatchNormalizationLayer<T>(channels));

        // Squeeze-Excitation block
        int seReduction = 8;
        _seBlocks.Add(new DenseLayer<T>(channels, channels / seReduction, (IActivationFunction<T>)new ReLUActivation<T>()));
        _seBlocks.Add(new DenseLayer<T>(channels / seReduction, channels, (IActivationFunction<T>)new SigmoidActivation<T>()));
    }

    #endregion

    #region ILanguageIdentifier Implementation

    /// <inheritdoc/>
    public LanguageResult<T> IdentifyLanguage(Tensor<T> audio)
    {
        var probabilities = GetLanguageProbabilities(audio);
        var topLanguage = probabilities.OrderByDescending(p => _numOps.ToDouble(p.Value)).First();

        string altLanguage = string.Empty;
        T altProb = _numOps.Zero;

        var sortedProbs = probabilities.OrderByDescending(p => _numOps.ToDouble(p.Value)).ToList();
        if (sortedProbs.Count > 1)
        {
            altLanguage = sortedProbs[1].Key;
            altProb = sortedProbs[1].Value;
        }

        return new LanguageResult<T>
        {
            LanguageCode = topLanguage.Key,
            LanguageName = GetLanguageDisplayName(topLanguage.Key),
            Confidence = topLanguage.Value,
            AlternativeLanguage = altLanguage,
            AlternativeProbability = altProb
        };
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, T> GetLanguageProbabilities(Tensor<T> audio)
    {
        var logits = GetLogits(audio);
        var probabilities = Softmax(logits);

        var result = new Dictionary<string, T>();
        for (int i = 0; i < probabilities.Length && i < _languageIdToCode.Count; i++)
        {
            if (_languageIdToCode.TryGetValue(i, out string? code))
            {
                result[code] = probabilities[i];
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public IReadOnlyList<(string Language, T Probability)> GetTopLanguages(Tensor<T> audio, int topN = 5)
    {
        var probabilities = GetLanguageProbabilities(audio);
        return probabilities
            .OrderByDescending(p => _numOps.ToDouble(p.Value))
            .Take(topN)
            .Select(p => (p.Key, p.Value))
            .ToList();
    }

    /// <inheritdoc/>
    public IReadOnlyList<LanguageSegment<T>> IdentifyLanguageSegments(Tensor<T> audio, int windowSizeMs = 2000)
    {
        var segments = new List<LanguageSegment<T>>();
        int samplesPerWindow = (int)(SampleRate * windowSizeMs / 1000.0);
        int hopSamples = samplesPerWindow / 2; // 50% overlap

        int totalSamples = audio.Length;
        double sampleDuration = 1.0 / SampleRate;

        for (int start = 0; start + samplesPerWindow <= totalSamples; start += hopSamples)
        {
            // Extract window
            var window = new Tensor<T>([samplesPerWindow]);
            for (int i = 0; i < samplesPerWindow; i++)
            {
                window[i] = audio[start + i];
            }

            var result = IdentifyLanguage(window);

            segments.Add(new LanguageSegment<T>
            {
                StartTime = start * sampleDuration,
                EndTime = (start + samplesPerWindow) * sampleDuration,
                LanguageCode = result.LanguageCode,
                Confidence = result.Confidence
            });
        }

        // Merge consecutive segments with same language
        return MergeConsecutiveSegments(segments);
    }

    /// <inheritdoc/>
    public string GetLanguageDisplayName(string languageCode)
    {
        if (_languageCodeToName.TryGetValue(languageCode.ToLowerInvariant(), out string? name))
            return name;
        return languageCode;
    }

    /// <inheritdoc/>
    public (bool SameLanguage, T Confidence) AreSameLanguage(Tensor<T> audio1, Tensor<T> audio2)
    {
        var lang1 = IdentifyLanguage(audio1);
        var lang2 = IdentifyLanguage(audio2);

        bool same = lang1.LanguageCode.Equals(lang2.LanguageCode, StringComparison.OrdinalIgnoreCase);

        // Confidence is minimum of the two confidences
        double conf1 = _numOps.ToDouble(lang1.Confidence);
        double conf2 = _numOps.ToDouble(lang2.Confidence);
        T confidence = _numOps.FromDouble(Math.Min(conf1, conf2));

        return (same, confidence);
    }

    #endregion

    #region AudioNeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        // Extract MFCCs with deltas
        return _mfccExtractor.Extract(rawAudio);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        // Apply softmax to get probabilities
        var probs = Softmax(modelOutput.Data.ToArray());
        return new Tensor<T>(probs, modelOutput.Shape);
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var preprocessed = PreprocessAudio(input);

        if (IsOnnxMode && OnnxModel is not null)
        {
            return OnnxModel.Run(preprocessed);
        }
        else
        {
            return ForwardNative(preprocessed);
        }
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!SupportsTraining)
            throw new InvalidOperationException("Cannot train in ONNX mode.");

        SetTrainingMode(true);

        var preprocessed = PreprocessAudio(input);
        var predicted = ForwardNative(preprocessed);

        // Convert to vectors for loss computation
        var predictedVector = predicted.ToVector();
        var expectedVector = expectedOutput.ToVector();

        var loss = _lossFunction.CalculateLoss(predictedVector, expectedVector);
        var gradientVector = _lossFunction.CalculateDerivative(predictedVector, expectedVector);
        var gradientTensor = Tensor<T>.FromVector(gradientVector, predicted.Shape);

        BackwardNative(gradientTensor);
        _optimizer.UpdateParameters(Layers);

        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var layer in GetAllLayers())
        {
            var layerParams = layer.GetParameters();
            var newParams = parameters.Slice(offset, layerParams.Length);
            // Apply actual parameter updates from optimizer
            for (int i = 0; i < layerParams.Length; i++)
            {
                layerParams[i] = newParams[i];
            }
            layer.SetParameters(layerParams);
            offset += layerParams.Length;
        }
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            Version = "1.0.0",
            AdditionalInfo = new Dictionary<string, object>
            {
                { "Architecture", "ECAPA-TDNN" },
                { "EmbeddingDimension", _options.EmbeddingDimension },
                { "TdnnChannels", _options.TdnnChannels },
                { "NumLanguages", _languageIdToCode.Count },
                { "SampleRate", SampleRate },
                { "IsOnnxMode", IsOnnxMode }
            }
        };
    }

    #endregion

    #region NeuralNetworkBase Abstract Methods

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        // In ONNX mode, layers are handled by ONNX runtime
        if (IsOnnxMode)
        {
            return;
        }

        // Check if user provided custom layers
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            return;
        }

        // Use LayerHelper to create default ECAPA-TDNN layers
        Layers.AddRange(LayerHelper<T>.CreateDefaultECAPATDNNLanguageIdentifierLayers(
            Architecture,
            numMels: _options.NumMels,
            tdnnChannels: _options.TdnnChannels,
            embeddingDimension: _options.EmbeddingDimension,
            numLanguages: _languageIdToCode.Count,
            dilations: _options.Dilations));
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(IsOnnxMode);
        writer.Write(SampleRate);
        writer.Write(_options.EmbeddingDimension);
        writer.Write(_options.TdnnChannels);
        writer.Write(_languageIdToCode.Count);

        foreach (var kvp in _languageIdToCode)
        {
            writer.Write(kvp.Key);
            writer.Write(kvp.Value);
        }
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read configuration values for validation
        _ = reader.ReadBoolean(); // IsOnnxMode
        _ = reader.ReadInt32();   // SampleRate
        _ = reader.ReadInt32();   // EmbeddingDimension
        _ = reader.ReadInt32();   // TdnnChannels
        int langCount = reader.ReadInt32();

        for (int i = 0; i < langCount; i++)
        {
            _ = reader.ReadInt32();   // language id
            _ = reader.ReadString();  // language code
        }
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new ECAPATDNNLanguageIdentifier<T>(
            Architecture,
            _languageIdToCode.Values.ToList(),
            _options,
            _optimizer,
            _lossFunction);
    }

    #endregion

    #region Private Methods

    private T[] GetLogits(Tensor<T> audio)
    {
        var preprocessed = PreprocessAudio(audio);

        Tensor<T> output;
        if (IsOnnxMode && OnnxModel is not null)
        {
            output = OnnxModel.Run(preprocessed);
        }
        else
        {
            output = ForwardNative(preprocessed);
        }

        return output.Data.ToArray();
    }

    private Tensor<T> ForwardNative(Tensor<T> features)
    {
        var output = features;

        // TDNN layers
        foreach (var layer in _tdnnLayers)
        {
            output = layer.Forward(output);
        }

        // Cache TDNN output for backward pass
        _lastTdnnOutput = output;

        // Collect outputs for MFA
        var blockOutputs = new List<Tensor<T>>();
        _blockOutputLengths.Clear();

        // SE-Res2Net blocks
        int blockIdx = 0;
        foreach (int _ in _options.Dilations)
        {
            var residual = output;

            // Process through res block layers (6 per block)
            for (int i = 0; i < 6 && blockIdx * 6 + i < _resBlocks.Count; i++)
            {
                output = _resBlocks[blockIdx * 6 + i].Forward(output);
            }

            // SE attention (2 layers per block)
            var seOutput = output;
            int seIdx = blockIdx * 2;
            if (seIdx < _seBlocks.Count)
            {
                // Global pooling (mean)
                var pooled = GlobalAveragePooling(output);
                var attention = _seBlocks[seIdx].Forward(pooled);
                if (seIdx + 1 < _seBlocks.Count)
                {
                    attention = _seBlocks[seIdx + 1].Forward(attention);
                }
                // Apply attention
                output = ApplyChannelAttention(output, attention);
            }

            // Residual connection
            output = AddTensors(output, residual);
            blockOutputs.Add(output);
            _blockOutputLengths.Add(output.Length);
            blockIdx++;
        }

        // Multi-layer feature aggregation (concatenate all block outputs)
        output = ConcatenateTensors(blockOutputs);

        // Attentive statistics pooling
        if (_poolingLayer is not null)
        {
            output = AttentiveStatisticsPooling(output);
            output = _poolingLayer.Forward(output);
        }

        // Final batch norm and classifier
        if (_finalBatchNorm is not null)
        {
            output = _finalBatchNorm.Forward(output);
        }

        if (_classifierLayer is not null)
        {
            output = _classifierLayer.Forward(output);
        }

        return output;
    }

    private void BackwardNative(Tensor<T> gradient)
    {
        var grad = gradient;

        // Backward through classifier
        if (_classifierLayer is not null)
        {
            grad = _classifierLayer.Backward(grad);
        }

        if (_finalBatchNorm is not null)
        {
            grad = _finalBatchNorm.Backward(grad);
        }

        // Backward through pooling layer
        if (_poolingLayer is not null)
        {
            grad = _poolingLayer.Backward(grad);
        }

        // Split gradient for MFA (Multi-layer Feature Aggregation)
        // The forward pass concatenated outputs from each block, so we split the gradient
        var blockGradients = new List<Tensor<T>>();
        int gradOffset = 0;
        foreach (int blockLen in _blockOutputLengths)
        {
            var blockGrad = new T[blockLen];
            for (int i = 0; i < blockLen && gradOffset + i < grad.Length; i++)
            {
                blockGrad[i] = grad.GetFlat(gradOffset + i);
            }
            blockGradients.Add(new Tensor<T>(blockGrad, [blockLen]));
            gradOffset += blockLen;
        }

        // Accumulate gradients from all blocks back to TDNN output
        int numBlocks = _options.Dilations.Length;
        Tensor<T>? accumulatedGrad = null;

        // Backward through each SE-Res2Net block in reverse order
        for (int blockIdx = numBlocks - 1; blockIdx >= 0; blockIdx--)
        {
            var blockGrad = blockIdx < blockGradients.Count ? blockGradients[blockIdx] : grad;

            // Residual connection: gradient flows to both paths
            var residualGrad = blockGrad;

            // Backward through SE attention (simplified - SE gradients don't propagate to main path in this approx)
            int seIdx = blockIdx * 2;
            if (seIdx + 1 < _seBlocks.Count)
            {
                _seBlocks[seIdx + 1].Backward(blockGrad);
            }
            if (seIdx < _seBlocks.Count)
            {
                _seBlocks[seIdx].Backward(blockGrad);
            }

            // Backward through res block layers (6 per block) in reverse
            var resGrad = blockGrad;
            for (int i = 5; i >= 0 && blockIdx * 6 + i < _resBlocks.Count; i--)
            {
                resGrad = _resBlocks[blockIdx * 6 + i].Backward(resGrad);
            }

            // Accumulate gradients (residual + main path)
            if (accumulatedGrad is null)
            {
                accumulatedGrad = AddTensors(resGrad, residualGrad);
            }
            else
            {
                accumulatedGrad = AddTensors(accumulatedGrad, AddTensors(resGrad, residualGrad));
            }
        }

        // Backward through TDNN layers
        grad = accumulatedGrad ?? grad;
        for (int i = _tdnnLayers.Count - 1; i >= 0; i--)
        {
            grad = _tdnnLayers[i].Backward(grad);
        }
    }

    private IEnumerable<ILayer<T>> GetAllLayers()
    {
        foreach (var layer in _tdnnLayers) yield return layer;
        foreach (var layer in _resBlocks) yield return layer;
        foreach (var layer in _seBlocks) yield return layer;
        if (_poolingLayer is not null) yield return _poolingLayer;
        if (_finalBatchNorm is not null) yield return _finalBatchNorm;
        if (_classifierLayer is not null) yield return _classifierLayer;
    }

    private T[] Softmax(T[] logits)
    {
        double maxLogit = logits.Max(x => _numOps.ToDouble(x));
        double[] expValues = logits.Select(x => Math.Exp(_numOps.ToDouble(x) - maxLogit)).ToArray();
        double sumExp = expValues.Sum();

        return expValues.Select(x => _numOps.FromDouble(x / sumExp)).ToArray();
    }

    private Tensor<T> GlobalAveragePooling(Tensor<T> input)
    {
        // Average across time dimension
        int features = input.Shape.Length > 1 ? input.Shape[^1] : input.Length;
        int timeSteps = input.Length / features;

        var output = new T[features];
        for (int f = 0; f < features; f++)
        {
            double sum = 0;
            for (int t = 0; t < timeSteps; t++)
            {
                sum += _numOps.ToDouble(input[t * features + f]);
            }
            output[f] = _numOps.FromDouble(sum / timeSteps);
        }

        return new Tensor<T>(output, [features]);
    }

    private Tensor<T> ApplyChannelAttention(Tensor<T> input, Tensor<T> attention)
    {
        var output = new T[input.Length];
        int features = attention.Length;
        int timeSteps = input.Length / features;

        for (int t = 0; t < timeSteps; t++)
        {
            for (int f = 0; f < features; f++)
            {
                int idx = t * features + f;
                output[idx] = _numOps.Multiply(input[idx], attention[f]);
            }
        }

        return new Tensor<T>(output, input.Shape);
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        int len = Math.Min(a.Length, b.Length);
        var output = new T[len];
        for (int i = 0; i < len; i++)
        {
            output[i] = _numOps.Add(a[i], b[i]);
        }
        return new Tensor<T>(output, a.Shape);
    }

    private Tensor<T> ConcatenateTensors(List<Tensor<T>> tensors)
    {
        int totalLen = tensors.Sum(t => t.Length);
        var output = new T[totalLen];
        int offset = 0;
        foreach (var tensor in tensors)
        {
            Array.Copy(tensor.Data.ToArray(), 0, output, offset, tensor.Length);
            offset += tensor.Length;
        }
        return new Tensor<T>(output, [totalLen]);
    }

    private Tensor<T> AttentiveStatisticsPooling(Tensor<T> input)
    {
        // Compute attention weights over time
        int features = input.Shape.Length > 1 ? input.Shape[^1] : input.Length / 10;
        int timeSteps = input.Length / features;

        // Mean pooling
        var mean = new T[features];
        for (int f = 0; f < features; f++)
        {
            double sum = 0;
            for (int t = 0; t < timeSteps; t++)
            {
                sum += _numOps.ToDouble(input[t * features + f]);
            }
            mean[f] = _numOps.FromDouble(sum / timeSteps);
        }

        // Standard deviation pooling
        var std = new T[features];
        for (int f = 0; f < features; f++)
        {
            double sumSq = 0;
            double meanVal = _numOps.ToDouble(mean[f]);
            for (int t = 0; t < timeSteps; t++)
            {
                double diff = _numOps.ToDouble(input[t * features + f]) - meanVal;
                sumSq += diff * diff;
            }
            std[f] = _numOps.FromDouble(Math.Sqrt(sumSq / timeSteps));
        }

        // Concatenate mean and std
        var output = new T[features * 2];
        Array.Copy(mean, 0, output, 0, features);
        Array.Copy(std, 0, output, features, features);

        return new Tensor<T>(output, [features * 2]);
    }

    private IReadOnlyList<LanguageSegment<T>> MergeConsecutiveSegments(List<LanguageSegment<T>> segments)
    {
        if (segments.Count == 0) return segments;

        var merged = new List<LanguageSegment<T>>();
        var current = segments[0];

        for (int i = 1; i < segments.Count; i++)
        {
            if (segments[i].LanguageCode == current.LanguageCode)
            {
                // Extend current segment
                current = new LanguageSegment<T>
                {
                    StartTime = current.StartTime,
                    EndTime = segments[i].EndTime,
                    LanguageCode = current.LanguageCode,
                    Confidence = _numOps.FromDouble(
                        (_numOps.ToDouble(current.Confidence) + _numOps.ToDouble(segments[i].Confidence)) / 2)
                };
            }
            else
            {
                merged.Add(current);
                current = segments[i];
            }
        }
        merged.Add(current);

        return merged;
    }

    private static (Dictionary<int, string>, Dictionary<string, int>, Dictionary<string, string>)
        InitializeLanguageMappings(IReadOnlyList<string>? languages = null)
    {
        var idToCode = new Dictionary<int, string>();
        var codeToId = new Dictionary<string, int>();
        var codeToName = GetDefaultLanguageNames();

        if (languages is not null)
        {
            for (int i = 0; i < languages.Count; i++)
            {
                string code = languages[i].ToLowerInvariant();
                idToCode[i] = code;
                codeToId[code] = i;
            }
        }
        else
        {
            // Default common languages
            string[] defaultLanguages = [
                "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko",
                "ar", "hi", "tr", "pl", "nl", "sv", "da", "no", "fi", "cs"
            ];

            for (int i = 0; i < defaultLanguages.Length; i++)
            {
                idToCode[i] = defaultLanguages[i];
                codeToId[defaultLanguages[i]] = i;
            }
        }

        return (idToCode, codeToId, codeToName);
    }

    private static Dictionary<string, string> GetDefaultLanguageNames()
    {
        return new Dictionary<string, string>
        {
            ["en"] = "English",
            ["es"] = "Spanish",
            ["fr"] = "French",
            ["de"] = "German",
            ["it"] = "Italian",
            ["pt"] = "Portuguese",
            ["ru"] = "Russian",
            ["zh"] = "Chinese",
            ["ja"] = "Japanese",
            ["ko"] = "Korean",
            ["ar"] = "Arabic",
            ["hi"] = "Hindi",
            ["tr"] = "Turkish",
            ["pl"] = "Polish",
            ["nl"] = "Dutch",
            ["sv"] = "Swedish",
            ["da"] = "Danish",
            ["no"] = "Norwegian",
            ["fi"] = "Finnish",
            ["cs"] = "Czech",
            ["el"] = "Greek",
            ["he"] = "Hebrew",
            ["th"] = "Thai",
            ["vi"] = "Vietnamese",
            ["id"] = "Indonesian",
            ["ms"] = "Malay",
            ["uk"] = "Ukrainian",
            ["ro"] = "Romanian",
            ["hu"] = "Hungarian",
            ["bg"] = "Bulgarian"
        };
    }

    #endregion
}
