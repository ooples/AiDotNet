using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.LanguageIdentification;

/// <summary>
/// Wav2Vec2 model fine-tuned for spoken language identification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Wav2Vec2 is Meta's self-supervised speech representation learning model that learns
/// powerful representations directly from raw audio waveforms. When fine-tuned for
/// language identification, it achieves state-of-the-art performance on many benchmarks.
/// </para>
/// <para>
/// Architecture overview:
/// - Feature Encoder: 7 temporal convolution layers that process raw waveform
/// - Transformer Encoder: 12-24 transformer blocks for contextual representations
/// - Classification Head: Linear projection to language classes
/// </para>
/// <para><b>For Beginners:</b> Wav2Vec2 is like a very attentive listener that:
/// 1. First breaks down the raw sound wave into small pieces (feature encoder)
/// 2. Then looks at how all these pieces relate to each other (transformer)
/// 3. Finally makes a decision about what language is being spoken (classifier)
///
/// Key advantages:
/// - Works directly on raw audio (no need for handcrafted features like MFCCs)
/// - Pre-trained on massive amounts of unlabeled speech data
/// - Can recognize languages even with limited labeled training data
///
/// Example usage:
/// <code>
/// var model = new Wav2Vec2LanguageIdentifier&lt;float&gt;(architecture, "wav2vec2_lid.onnx");
/// var result = model.IdentifyLanguage(audioTensor);
/// // Result is available in the returned value
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations", "https://arxiv.org/abs/2006.11477", Year = 2020, Authors = "Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, Michael Auli")]
public partial class Wav2Vec2LanguageIdentifier<T> : AudioNeuralNetworkBase<T>, ILanguageIdentifier<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// Traced from output construction: PredictCore returns ForwardNative, whose last step is
    /// <c>_classifierLayer.Forward(...)</c> - the final layer of the shared Wav2Vec2 factory,
    /// sized by <c>numLanguages: _languageIdToCode.Count</c>. A class count - HiddenSize is the
    /// pooling-projection width one layer earlier, not the output width.
    /// </remarks>
    protected override int OutputFeatureWidth => _languageIdToCode.Count;

    #region Fields

    private readonly INumericOperations<T> _numOps;
    private readonly Wav2Vec2LidOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private readonly ILossFunction<T> _lossFunction;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    // Every layer of the default topology, in the factory's order.
    private readonly List<ILayer<T>> _defaultLayers = [];

    // Feature encoder: stage 0's conv, GroupNorm and GELU, then the later stages' convs.
    private readonly List<ILayer<T>> _featureEncoder = [];

    // Feature projection and the convolutional positional embedding.
    private LayerNormalizationLayer<T>? _projectionNorm;
    private DenseLayer<T>? _projection;
    private DropoutLayer<T>? _projectionDropout;
    private Conv1DLayer<T>? _positionalConv;
    private LayerNormalizationLayer<T>? _encoderNorm;
    private DropoutLayer<T>? _encoderDropout;

    // Transformer encoder: views over Layers, rebuilt by PartitionDefaultLayers.
    private readonly List<EncoderBlock> _blocks = [];

    // Classification head
    private DenseLayer<T>? _poolingProjection;
    private DenseLayer<T>? _classifierLayer;

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
    /// Gets the hidden size of the transformer.
    /// </summary>
    public int HiddenSize => _options.HiddenSize;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a Wav2Vec2 language identifier with ONNX model for inference.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="options">Wav2Vec2 LID options.</param>
    public Wav2Vec2LanguageIdentifier(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        Wav2Vec2LidOptions? options = null)
        : base(architecture, new CrossEntropyWithLogitsLoss<T>())
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model file not found: {modelPath}");

        _numOps = MathHelper.GetNumericOperations<T>();
        _options = options ?? new Wav2Vec2LidOptions();
        Options = _options;
        _options.ModelPath = modelPath;

        SampleRate = _options.SampleRate;

        _lossFunction = new CrossEntropyWithLogitsLoss<T>();

        // Initialize language mappings
        (_languageIdToCode, _languageCodeToId, _languageCodeToName) = InitializeLanguageMappings();

        // Load ONNX model
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);

    }

    /// <summary>
    /// Creates a Wav2Vec2 language identifier for native training.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="supportedLanguages">List of language codes to identify.</param>
    /// <param name="options">Wav2Vec2 LID options.</param>
    /// <param name="optimizer">Optimizer for training.</param>
    /// <param name="lossFunction">Loss function.</param>
    public Wav2Vec2LanguageIdentifier(
        NeuralNetworkArchitecture<T> architecture,
        IReadOnlyList<string> supportedLanguages,
        Wav2Vec2LidOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>())
    {
        if (supportedLanguages is null)
            throw new ArgumentNullException(nameof(supportedLanguages));
        if (supportedLanguages.Count == 0)
            throw new ArgumentException("At least one language must be specified.", nameof(supportedLanguages));

        _numOps = MathHelper.GetNumericOperations<T>();
        _options = options ?? new Wav2Vec2LidOptions();
        Options = _options;

        SampleRate = _options.SampleRate;

        _lossFunction = lossFunction ?? new CrossEntropyWithLogitsLoss<T>();
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

        // Initialize language mappings
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
        // Build the default stack from the shared factory and partition it back into the roles the
        // forward needs, in the factory's documented order. The partition fails loudly if the two
        // ever drift apart, and Layers publishes exactly these instances.
        var built = LayerHelper<T>.CreateDefaultWav2Vec2LanguageIdentifierLayers(
            Architecture,
            hiddenSize: _options.HiddenSize,
            numLayers: _options.NumLayers,
            numAttentionHeads: _options.NumAttentionHeads,
            intermediateSize: _options.IntermediateSize,
            numLanguages: _languageIdToCode.Count,
            dropoutRate: _options.HiddenDropout,
            featureEncoderDim: _options.FeatureEncoderDim,
            featureProjectionDropout: _options.FeatureProjectionDropout,
            featureEncoderKernels: _options.FeatureEncoderKernels,
            featureEncoderStrides: _options.FeatureEncoderStrides,
            positionalConvKernel: _options.PositionalConvKernel,
            positionalConvGroups: _options.PositionalConvGroups).ToList();
        _defaultLayers.AddRange(built);
        PartitionDefaultLayers(built);
    }

    /// <summary>
    /// Assigns every role from <paramref name="built"/>, in the factory's documented order. Run at
    /// construction and again whenever a deserialize or eager clone has replaced the layer instances,
    /// since the transformer blocks are views this model holds, not layer-typed members it rebinds.
    /// </summary>
    private void PartitionDefaultLayers(IReadOnlyList<ILayer<T>> built)
    {
        _featureEncoder.Clear();
        _blocks.Clear();
        _projectionDropout = null;
        _encoderDropout = null;
        int index = 0;
        TLayer Next<TLayer>() where TLayer : class, ILayer<T>
        {
            if (index >= built.Count || built[index] is not TLayer typed)
            {
                throw new InvalidOperationException(
                    $"The Wav2Vec2 factory layout and this partition have drifted apart at layer {index}: " +
                    $"expected {typeof(TLayer).Name}, found {(index < built.Count ? built[index].GetType().Name : "the end")}.");
            }

            index++;
            return typed;
        }

        // Feature encoder: stage 0's conv, GroupNorm and GELU, then one GELU conv per later stage.
        _featureEncoder.Add(Next<Conv1DLayer<T>>());
        _featureEncoder.Add(Next<GroupNormalizationLayer<T>>());
        _featureEncoder.Add(Next<ActivationLayer<T>>());
        for (int i = 1; i < LayerHelper<T>.Wav2Vec2FeatureEncoderStages; i++)
        {
            _featureEncoder.Add(Next<Conv1DLayer<T>>());
        }

        _projectionNorm = Next<LayerNormalizationLayer<T>>();
        _projection = Next<DenseLayer<T>>();
        if (_options.FeatureProjectionDropout > 0) _projectionDropout = Next<DropoutLayer<T>>();

        _positionalConv = Next<Conv1DLayer<T>>();
        _encoderNorm = Next<LayerNormalizationLayer<T>>();
        if (_options.HiddenDropout > 0) _encoderDropout = Next<DropoutLayer<T>>();

        for (int i = 0; i < _options.NumLayers; i++)
        {
            _blocks.Add(new EncoderBlock(
                Next<MultiHeadAttentionLayer<T>>(),
                Next<LayerNormalizationLayer<T>>(),
                Next<DenseLayer<T>>(),
                Next<DenseLayer<T>>(),
                Next<LayerNormalizationLayer<T>>(),
                _options.HiddenDropout > 0 ? Next<DropoutLayer<T>>() : null));
        }

        _poolingProjection = Next<DenseLayer<T>>();
        _classifierLayer = Next<DenseLayer<T>>();

        if (index != built.Count)
        {
            throw new InvalidOperationException(
                $"The Wav2Vec2 factory produced {built.Count} layers but the role partition consumed " +
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
        int samplesPerWindow = (int)((double)SampleRate * windowSizeMs / 1000.0);
        int hopSamples = samplesPerWindow / 2;

        int totalSamples = audio.Length;
        double sampleDuration = 1.0 / SampleRate;

        for (int start = 0; start + samplesPerWindow <= totalSamples; start += hopSamples)
        {
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
        // Wav2Vec2 works on raw waveform - just normalize
        var normalized = new T[rawAudio.Length];

        // Compute mean and std for normalization
        double sum = 0;
        for (int i = 0; i < rawAudio.Length; i++)
        {
            sum += _numOps.ToDouble(rawAudio[i]);
        }
        double mean = sum / rawAudio.Length;

        double sumSq = 0;
        for (int i = 0; i < rawAudio.Length; i++)
        {
            double diff = _numOps.ToDouble(rawAudio[i]) - mean;
            sumSq += diff * diff;
        }
        double std = Math.Sqrt(sumSq / rawAudio.Length);
        if (std < 1e-7) std = 1e-7;

        for (int i = 0; i < rawAudio.Length; i++)
        {
            normalized[i] = _numOps.FromDouble((_numOps.ToDouble(rawAudio[i]) - mean) / std);
        }

        return new Tensor<T>(normalized, rawAudio._shape);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
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
    /// Runs the same waveform normalization prediction runs, so the objective optimizes the function
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
                { "Architecture", "Wav2Vec2-LID" },
                { "HiddenSize", _options.HiddenSize },
                { "NumLayers", _options.NumLayers },
                { "NumAttentionHeads", _options.NumAttentionHeads },
                { "NumLanguages", _languageIdToCode.Count },
                { "SampleRate", SampleRate },
                { "IsOnnxMode", IsOnnxMode }
            }
        };
    }

    #endregion

    #region NeuralNetworkBase Abstract Methods

    private bool _lazyShapesProbed;

    /// <inheritdoc/>
    /// <remarks>
    /// The base walk feeds the architecture's input shape through Layers as one sequential chain, so
    /// the first convolution would resolve its input channels from the fixture's [1, 64, 32] shape
    /// instead of the single waveform channel the real forward feeds it. Resolve through the real
    /// topology with the shortest waveform that leaves one frame after every strided stage.
    /// </remarks>
    protected override void ResolveLazyLayerShapes()
    {
        if (_lazyShapesProbed || IsOnnxMode || _featureEncoder.Count == 0) return;
        _lazyShapesProbed = true;

        int samples = 1;
        for (int i = _options.FeatureEncoderKernels.Length - 1; i >= 0; i--)
        {
            samples = (samples - 1) * _options.FeatureEncoderStrides[i] + _options.FeatureEncoderKernels[i];
        }

        bool wasTraining = IsTrainingMode;
        if (wasTraining) SetTrainingMode(false);
        try
        {
            _ = ForwardNative(new Tensor<T>(new[] { samples }));
        }
        finally
        {
            if (wasTraining) SetTrainingMode(true);
        }
    }

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
        Layers.AddRange(_defaultLayers);
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

    private Tensor<T> ForwardNative(Tensor<T> input)
    {
        // A caller-supplied architecture is an ordinary custom layer chain; the role-aware
        // traversal below applies only to the default topology.
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            var customOutput = input;
            foreach (var layer in Layers)
            {
                customOutput = layer.Forward(customOutput);
            }

            return customOutput;
        }

        // A deserialize or eager clone replaces some or all of Layers; every role view must follow.
        if (!DefaultLayersMatch())
        {
            _defaultLayers.Clear();
            _defaultLayers.AddRange(Layers);
            PartitionDefaultLayers(_defaultLayers);
        }

        bool unbatched = input.Shape.Length != 2;
        int batch = unbatched ? 1 : input.Shape[0];
        int samples = unbatched ? input.Length : input.Shape[1];

        // Feature encoder over the raw waveform [B, 1, L]. Stage 0's GroupNorm (one group per
        // channel) is a 4-D layer, so its [B, C, T] input is viewed as [B, C, 1, T].
        var x = Engine.Reshape(input, new[] { batch, 1, samples });
        x = _featureEncoder[0].Forward(x);
        int channels = x.Shape[1], frames = x.Shape[2];
        x = Engine.Reshape(_featureEncoder[1].Forward(Engine.Reshape(x, new[] { batch, channels, 1, frames })),
            new[] { batch, channels, frames });
        for (int i = 2; i < _featureEncoder.Count; i++)
        {
            x = _featureEncoder[i].Forward(x);
        }

        // Feature projection on time-major frames [B, T, C].
        frames = x.Shape[2];
        x = RequireLayer(_projectionNorm).Forward(Engine.TensorPermute(x, new[] { 0, 2, 1 }));
        int hidden = _options.HiddenSize;
        x = Engine.Reshape(
            RequireLayer(_projection).Forward(Engine.Reshape(x, new[] { batch * frames, x.Shape[2] })),
            new[] { batch, frames, hidden });
        if (_projectionDropout is not null) x = _projectionDropout.Forward(x);

        // Convolutional relative positional embedding. The even kernel with kernel/2 padding yields
        // one extra frame, which the paper's SamePad drops; the embedding is added to the frames.
        var positional = RequireLayer(_positionalConv).Forward(Engine.TensorPermute(x, new[] { 0, 2, 1 }));
        positional = Engine.TensorNarrow(positional, dim: 2, start: 0, length: frames);
        x = Engine.TensorAdd(x, Engine.TensorPermute(positional, new[] { 0, 2, 1 }));
        x = RequireLayer(_encoderNorm).Forward(x);
        if (_encoderDropout is not null) x = _encoderDropout.Forward(x);

        // Post-LN transformer blocks: x = LN(x + Attn(x)); x = LN(x + FFN(x)).
        foreach (var block in _blocks)
        {
            x = block.Forward(Engine, x);
        }

        // Mean over time, the tanh projection and the per-language logits.
        var pooled = Engine.ReduceMean(x, new[] { 1 }, keepDims: false);
        var logits = RequireLayer(_classifierLayer).Forward(RequireLayer(_poolingProjection).Forward(pooled));
        return unbatched ? Engine.Reshape(logits, new[] { logits.Shape[logits.Shape.Length - 1] }) : logits;
    }

    /// <summary>
    /// Whether every role view still points at the instance Layers holds at its position. The views
    /// themselves are compared, not a saved copy of the list: the generated alias rebinding updates
    /// layer-typed members and layer lists after a clone, but not the EncoderBlock views.
    /// </summary>
    private bool DefaultLayersMatch()
    {
        int index = 0;
        foreach (var layer in RoleLayersInFactoryOrder())
        {
            if (index >= Layers.Count || !ReferenceEquals(Layers[index], layer)) return false;
            index++;
        }

        return index == Layers.Count;
    }

    private IEnumerable<ILayer<T>> RoleLayersInFactoryOrder()
    {
        foreach (var layer in _featureEncoder) yield return layer;
        if (_projectionNorm is not null) yield return _projectionNorm;
        if (_projection is not null) yield return _projection;
        if (_projectionDropout is not null) yield return _projectionDropout;
        if (_positionalConv is not null) yield return _positionalConv;
        if (_encoderNorm is not null) yield return _encoderNorm;
        if (_encoderDropout is not null) yield return _encoderDropout;
        foreach (var block in _blocks)
        {
            yield return block.Attention;
            yield return block.AttentionNorm;
            yield return block.FeedForwardUp;
            yield return block.FeedForwardDown;
            yield return block.FeedForwardNorm;
            if (block.Dropout is not null) yield return block.Dropout;
        }

        if (_poolingProjection is not null) yield return _poolingProjection;
        if (_classifierLayer is not null) yield return _classifierLayer;
    }

    private static TLayer RequireLayer<TLayer>(TLayer? layer) where TLayer : class        => layer ?? throw new InvalidOperationException("The Wav2Vec2 network has not been initialized.");

    /// <summary>One post-LN Wav2Vec2 transformer block.</summary>
    private sealed class EncoderBlock
    {
        public EncoderBlock(
            MultiHeadAttentionLayer<T> attention, LayerNormalizationLayer<T> attentionNorm,
            DenseLayer<T> feedForwardUp, DenseLayer<T> feedForwardDown, LayerNormalizationLayer<T> feedForwardNorm,
            DropoutLayer<T>? dropout)
        {
            Attention = attention;
            AttentionNorm = attentionNorm;
            FeedForwardUp = feedForwardUp;
            FeedForwardDown = feedForwardDown;
            FeedForwardNorm = feedForwardNorm;
            Dropout = dropout;
        }

        public MultiHeadAttentionLayer<T> Attention { get; }
        public LayerNormalizationLayer<T> AttentionNorm { get; }
        public DenseLayer<T> FeedForwardUp { get; }
        public DenseLayer<T> FeedForwardDown { get; }
        public LayerNormalizationLayer<T> FeedForwardNorm { get; }
        public DropoutLayer<T>? Dropout { get; }

        public Tensor<T> Forward(IEngine engine, Tensor<T> x)
        {
            var attended = Attention.Forward(x);
            if (Dropout is not null) attended = Dropout.Forward(attended);
            x = AttentionNorm.Forward(engine.TensorAdd(x, attended));

            int batch = x.Shape[0], frames = x.Shape[1], width = x.Shape[2];
            var fed = FeedForwardDown.Forward(FeedForwardUp.Forward(engine.Reshape(x, new[] { batch * frames, width })));
            fed = engine.Reshape(fed, new[] { batch, frames, width });
            if (Dropout is not null) fed = Dropout.Forward(fed);
            return FeedForwardNorm.Forward(engine.TensorAdd(x, fed));
        }
    }

    private T[] Softmax(T[] logits)
    {
        double maxLogit = logits.Max(x => _numOps.ToDouble(x));
        double[] expValues = logits.Select(x => Math.Exp(_numOps.ToDouble(x) - maxLogit)).ToArray();
        double sumExp = expValues.Sum();

        return expValues.Select(x => _numOps.FromDouble(x / sumExp)).ToArray();
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
