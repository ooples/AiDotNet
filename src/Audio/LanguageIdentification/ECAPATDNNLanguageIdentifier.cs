using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Audio.Features;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Enums;
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
/// // Result is available in the returned value
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification", "https://arxiv.org/abs/2005.07143", Year = 2020, Authors = "Brecht Desplanques, Jenthe Thienpondt, Kris Demuynck")]
public partial class ECAPATDNNLanguageIdentifier<T> : AudioNeuralNetworkBase<T>, ILanguageIdentifier<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// Traced from output construction: PredictCore returns ForwardNative, whose last step is
    /// <c>_classifierLayer.Forward(...)</c> - the final layer of the shared ECAPA-TDNN factory,
    /// sized by <c>numLanguages: _languageIdToCode.Count</c>. A class count, not an embedding size:
    /// EmbeddingDimension is the pooling width one layer earlier.
    /// </remarks>
    protected override int OutputFeatureWidth => _languageIdToCode.Count;

    #region Fields

    private readonly INumericOperations<T> _numOps;
    private readonly ECAPATDNNOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private readonly MfccExtractor<T> _mfccExtractor;
    private readonly ILossFunction<T> _lossFunction;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    // ECAPA-TDNN architecture components
    private readonly List<ILayer<T>> _tdnnLayers = [];
    private readonly List<ILayer<T>> _seBlocks = [];
    private readonly List<ILayer<T>> _resBlocks = [];
    private DenseLayer<T>? _poolingLayer;
    private DenseLayer<T>? _classifierLayer;
    private BatchNormalizationLayer<T>? _finalBatchNorm;

    // Cached values for proper gradient flow in MFA
    private readonly List<int> _blockOutputLengths = [];
    [Scratch]
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
        : base(architecture, new CrossEntropyWithLogitsLoss<T>())
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

        _lossFunction = new CrossEntropyWithLogitsLoss<T>();

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
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>())
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

        _lossFunction = lossFunction ?? new CrossEntropyWithLogitsLoss<T>();
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

        // The head is sized from this list, so an architecture declaring a different output width
        // describes a different model: say so instead of silently building one of the two.
        LanguageIdentificationDefaults.ValidateHeadWidth(architecture, _languageIdToCode.Count);

        // One stack. InitializeLayers builds it through the shared factory and publishes those exact
        // instances through Layers, which training, serialization and clone all walk.
        InitializeLayers();
    }

    #endregion

    #region Layer Initialization

    private void InitializeNativeLayers()
    {
        // Build the default ECAPA-TDNN stack from the shared factory - the one VoxLingua107Identifier
        // also uses - and partition the flat list back into the typed roles the forward needs. This
        // used to build a second, private copy of the stack by hand, so the forward ran on layers
        // Layers never held and Train's optimizer step reached none of them.
        var built = LayerHelper<T>.CreateDefaultECAPATDNNLanguageIdentifierLayers(
            Architecture,
            numMels: _options.NumMels,
            tdnnChannels: _options.TdnnChannels,
            embeddingDimension: _options.EmbeddingDimension,
            numLanguages: _languageIdToCode.Count,
            dilations: _options.Dilations).ToList();

        int index = 0;

        // Initial TDNN: DenseLayer + BatchNormalizationLayer.
        _tdnnLayers.Add(built[index++]);
        _tdnnLayers.Add(built[index++]);

        // One SE-Res2 block per dilation: six residual-path layers, then two squeeze-excitation
        // layers. The forward indexes residual layers 6-per-block and SE layers 2-per-block.
        foreach (int _ in _options.Dilations)
        {
            for (int i = 0; i < 6; i++)
            {
                _resBlocks.Add(built[index++]);
            }

            _seBlocks.Add(built[index++]);
            _seBlocks.Add(built[index++]);
        }

        // Attentive-statistics-pooling projection, final BatchNorm, classifier head.
        _poolingLayer = (DenseLayer<T>)built[index++];
        _finalBatchNorm = (BatchNormalizationLayer<T>)built[index++];
        _classifierLayer = (DenseLayer<T>)built[index++];

        if (index != built.Count)
        {
            throw new InvalidOperationException(
                $"The ECAPA-TDNN factory produced {built.Count} layers but the role partition consumed " +
                $"{index}; the factory layout and this partition have drifted apart.");
        }
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
        return new Tensor<T>(probs, modelOutput._shape);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
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
        try
        {
            // TrainWithTape runs the forward, loss, backward and the configured optimizer step. The
            // previous body computed a loss, discarded it, and asked the optimizer to update Layers -
            // which was empty, and which had received no gradient.
            TrainWithTape(input, expectedOutput, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Runs the same MFCC front end prediction runs, so the objective optimizes the function
    /// inference evaluates.
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => ForwardNative(PreprocessAudio(input));

    // UpdateParameters restated the base verbatim; ModelBase routes it to SetParameters.
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
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

        // Build the topology once and publish those exact instances through Layers. ForwardNative
        // needs their block roles; parameters, gradients, the optimizer, serialization and clone need
        // the same instances.
        InitializeNativeLayers();
        Layers.AddRange(GetAllLayers());
    }

    /// <inheritdoc/>


    /// <inheritdoc/>


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
        // A caller-supplied architecture is an ordinary custom layer chain; the role-aware
        // traversal below applies only to the default topology.
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            var customOutput = features;
            foreach (var layer in Layers)
            {
                customOutput = layer.Forward(customOutput);
            }

            return customOutput;
        }

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

    // GlobalAveragePooling, ApplyChannelAttention and AttentiveStatisticsPooling are engine ops, the
    // same bodies VoxLingua107Identifier runs on this identical ECAPA-TDNN forward. They were NumOps
    // scalar loops writing into fresh tensors, which the gradient tape cannot see through: the SE gate
    // and everything upstream of statistics pooling received no gradient.
    private Tensor<T> GlobalAveragePooling(Tensor<T> input)
    {
        // ECAPA activations are time-major [time, channels].
        if (input.Rank <= 1)
            return input;

        int[] timeAxes = new int[input.Rank - 1];
        for (int axis = 0; axis < timeAxes.Length; axis++)
            timeAxes[axis] = axis;
        return Engine.ReduceMean(input, timeAxes, keepDims: false);
    }

    private Tensor<T> ApplyChannelAttention(Tensor<T> input, Tensor<T> attention)
    {
        if (input.Rank <= 1)
            return Engine.TensorMultiply(input, attention);

        var broadcastShape = new int[input.Rank];
        for (int axis = 0; axis < broadcastShape.Length; axis++)
            broadcastShape[axis] = 1;
        broadcastShape[broadcastShape.Length - 1] = attention.Length;
        var channelGate = Engine.Reshape(attention, broadcastShape);
        return Engine.TensorMultiply(input, channelGate);
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorAdd(a, b);
    }

    private Tensor<T> ConcatenateTensors(List<Tensor<T>> tensors)
    {
        return Engine.TensorConcatenate(tensors.ToArray(), axis: 0);
    }

    private Tensor<T> AttentiveStatisticsPooling(Tensor<T> input)
    {
        if (input.Rank <= 1)
            return Engine.TensorConcatenate([input, input], axis: 0);

        int[] timeAxes = new int[input.Rank - 1];
        for (int axis = 0; axis < timeAxes.Length; axis++)
            timeAxes[axis] = axis;

        var meanKeepDims = Engine.ReduceMean(input, timeAxes, keepDims: true);
        var centered = Engine.TensorSubtract(input, meanKeepDims);
        var variance = Engine.ReduceMean(
            Engine.TensorMultiply(centered, centered), timeAxes, keepDims: false);
        var std = Engine.TensorSqrt(
            Engine.TensorAddScalar(variance, NumericalStabilityHelper.GetEpsilon<T>()));
        var mean = Engine.ReduceMean(input, timeAxes, keepDims: false);
        return Engine.TensorConcatenate([mean, std], axis: 0);
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
            var defaultLanguages = LanguageIdentificationDefaults.CommonLanguageCodes;

            for (int i = 0; i < defaultLanguages.Count; i++)
            {
                idToCode[i] = defaultLanguages[i];
                codeToId[defaultLanguages[i]] = i;
            }
        }

        return (idToCode, codeToId, codeToName);
    }

    private static Dictionary<string, string> GetDefaultLanguageNames()
        => LanguageIdentificationDefaults.CreateDisplayNameMap();

    #endregion
}
