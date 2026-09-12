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

    // Feature encoder (CNN layers)
    private readonly List<ILayer<T>> _featureEncoder = [];
    private readonly List<ILayer<T>> _featureProjection = [];

    // Transformer encoder
    private readonly List<ILayer<T>> _transformerLayers = [];

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
        // forward needs. This used to build a second, private copy by hand, so the forward ran on
        // layers Layers never held and Train's optimizer step reached none of them. The partition
        // mirrors the factory's layout exactly, including the dropout layers it omits when a rate
        // is zero, and fails loudly if the two ever drift apart.
        var built = LayerHelper<T>.CreateDefaultWav2Vec2LanguageIdentifierLayers(
            Architecture,
            hiddenSize: _options.HiddenSize,
            numLayers: _options.NumLayers,
            numAttentionHeads: _options.NumAttentionHeads,
            intermediateSize: _options.IntermediateSize,
            numLanguages: _languageIdToCode.Count,
            dropoutRate: _options.HiddenDropout,
            featureEncoderDim: _options.FeatureEncoderDim,
            featureProjectionDropout: _options.FeatureProjectionDropout).ToList();

        int index = 0;

        // Feature encoder: a DenseLayer + LayerNormalizationLayer pair per stage.
        for (int i = 0; i < LayerHelper<T>.Wav2Vec2FeatureEncoderStages * 2; i++)
        {
            _featureEncoder.Add(built[index++]);
        }

        // Feature projection, plus its dropout only when that rate is non-zero.
        _featureProjection.Add(built[index++]);
        if (_options.FeatureProjectionDropout > 0)
        {
            _featureProjection.Add(built[index++]);
        }

        // Transformer blocks: attention stand-in + norm, feed-forward pair + norm, and a dropout when
        // the hidden rate is non-zero.
        int perBlock = _options.HiddenDropout > 0 ? 6 : 5;
        for (int i = 0; i < _options.NumLayers * perBlock; i++)
        {
            _transformerLayers.Add(built[index++]);
        }

        // Classification head: tanh pooling projection, then the per-language logits.
        _poolingProjection = (DenseLayer<T>)built[index++];
        _classifierLayer = (DenseLayer<T>)built[index++];

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
        int samplesPerWindow = (int)(SampleRate * windowSizeMs / 1000.0);
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

        var output = input;

        // Feature encoder
        foreach (var layer in _featureEncoder)
        {
            output = layer.Forward(output);
        }

        // Feature projection
        foreach (var layer in _featureProjection)
        {
            output = layer.Forward(output);
        }

        // Transformer layers
        foreach (var layer in _transformerLayers)
        {
            output = layer.Forward(output);
        }

        // Mean pooling across time
        output = MeanPooling(output);

        // Classification head
        if (_poolingProjection is not null)
        {
            output = _poolingProjection.Forward(output);
        }

        if (_classifierLayer is not null)
        {
            output = _classifierLayer.Forward(output);
        }

        return output;
    }

    private IEnumerable<ILayer<T>> GetAllLayers()
    {
        foreach (var layer in _featureEncoder) yield return layer;
        foreach (var layer in _featureProjection) yield return layer;
        foreach (var layer in _transformerLayers) yield return layer;
        if (_poolingProjection is not null) yield return _poolingProjection;
        if (_classifierLayer is not null) yield return _classifierLayer;
    }

    private Tensor<T> MeanPooling(Tensor<T> input)
    {
        int hiddenSize = _options.HiddenSize;
        if (input.Length == 0 || hiddenSize <= 0)
        {
            return new Tensor<T>(new T[Math.Max(hiddenSize, 1)], [Math.Max(hiddenSize, 1)]);
        }

        // The mean over time of HiddenSize-wide frames, as an engine reduction. This was a NumOps
        // scalar loop into a fresh tensor, which the gradient tape cannot see through: the feature
        // encoder, projection and transformer upstream of it received no gradient at all.
        var frames = Engine.Reshape(input, new[] { input.Length / hiddenSize, hiddenSize });
        return Engine.ReduceMean(frames, new[] { 0 }, keepDims: false);
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
