using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Audio.Features;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.LanguageIdentification;

/// <summary>
/// VoxLingua107 language identifier supporting 107 languages.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// VoxLingua107 is a language identification model trained on the VoxLingua107 dataset,
/// which contains speech samples from 107 languages collected from YouTube videos.
/// The model uses the ECAPA-TDNN architecture and is specifically optimized for
/// large-scale multilingual language identification.
/// </para>
/// <para>
/// Supported language families include:
/// - Indo-European (English, Spanish, French, German, Russian, Hindi, etc.)
/// - Sino-Tibetan (Mandarin, Cantonese, etc.)
/// - Afro-Asiatic (Arabic, Hebrew, Amharic, etc.)
/// - Austronesian (Indonesian, Tagalog, Malay, etc.)
/// - Niger-Congo (Swahili, Yoruba, Zulu, etc.)
/// - Altaic (Turkish, Korean, Japanese, Mongolian, etc.)
/// - And many more...
/// </para>
/// <para><b>For Beginners:</b> VoxLingua107 is like having a polyglot friend who can
/// recognize 107 different languages just by listening.
///
/// Key features:
/// - Covers most of the world's major languages
/// - Trained on real-world YouTube audio (diverse accents and recording conditions)
/// - Can identify languages even from short clips (3-10 seconds)
/// - Handles code-switching and multilingual speakers
///
/// Example usage:
/// <code>
/// var model = new VoxLingua107Identifier&lt;float&gt;(architecture, "voxlingua107.onnx");
/// var result = model.IdentifyLanguage(audioTensor);
/// // Result is available in the returned value
/// // Output: Language: Swedish (85%)
///
/// // Get top 5 predictions
/// var topLanguages = model.GetTopLanguages(audioTensor, 5);
/// foreach (var (lang, prob) in topLanguages)
///     Console.WriteLine($"  {lang}: {prob:P1}");
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("VoxLingua107: A Dataset for Spoken Language Recognition", "https://arxiv.org/abs/2011.12998", Year = 2021, Authors = "Jörgen Valk, Tanel Alumäe")]
public partial class VoxLingua107Identifier<T> : AudioNeuralNetworkBase<T>, ILanguageIdentifier<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// Traced from output construction: PredictCore returns ForwardNative, whose last step is
    /// <c>_classifierLayer.Forward(...)</c>. That layer is the final entry of
    /// CreateDefaultVoxLingua107Layers, which sizes its classifier head as
    /// <c>architecture.OutputSize &gt; 0 ? architecture.OutputSize : 107</c> - the identical
    /// expression this model caches in <c>_numLanguages</c>, so the field tracks the head exactly.
    /// A class count (107 by paper default), not EmbeddingDimension, which is the pooling width.
    /// </remarks>
    protected override int OutputFeatureWidth => _numLanguages;

    #region Constants

    /// <summary>
    /// The 107 language codes supported by VoxLingua107 (ISO 639-1/3).
    /// </summary>
    public static readonly string[] VoxLingua107Languages =
    [
        "ab", "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn",
        "bo", "br", "bs", "ca", "ceb", "cs", "cy", "da", "de", "el",
        "en", "eo", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl",
        "gn", "gu", "ha", "haw", "he", "hi", "hr", "ht", "hu", "hy",
        "ia", "id", "is", "it", "ja", "jv", "ka", "kk", "km", "kn",
        "ko", "la", "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk",
        "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn", "no",
        "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si",
        "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw", "ta",
        "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz",
        "vi", "war", "xh", "yi", "yo", "zh", "zu"
    ];

    #endregion

    #region Fields

    private readonly INumericOperations<T> _numOps;
    private readonly VoxLingua107Options _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private readonly MfccExtractor<T> _mfccExtractor;
    private readonly ILossFunction<T> _lossFunction;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    // ECAPA-TDNN architecture (same as ECAPATDNNLanguageIdentifier)
    private readonly List<ILayer<T>> _tdnnLayers = [];
    private readonly List<ILayer<T>> _seBlocks = [];
    private readonly List<ILayer<T>> _resBlocks = [];
    private DenseLayer<T>? _poolingLayer;
    private DenseLayer<T>? _classifierLayer;
    private BatchNormalizationLayer<T>? _finalBatchNorm;

    // The classifier head width. VoxLingua107's paper label set is 107 languages
    // (the default), but the head honours a caller-configured Architecture.OutputSize
    // so the identifier can target any label set. Hardcoding 107 overrode a smaller
    // configured head and CrossEntropyWithLogitsLoss.ClassIndicesToOneHot then indexed
    // past its one-hot buffer.
    private readonly int _numLanguages;

    // Language mapping for 107 languages
    private readonly Dictionary<int, string> _languageIdToCode;
    private readonly Dictionary<string, int> _languageCodeToId;
    private readonly Dictionary<string, string> _languageCodeToName;

    // MFA gradient flow tracking
    private readonly List<int> _blockOutputLengths = [];
    [Scratch]
    private Tensor<T>? _lastTdnnOutput;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override bool SupportsTraining => !IsOnnxMode;

    /// <inheritdoc/>
    public IReadOnlyList<string> SupportedLanguages => VoxLingua107Languages.ToList();

    /// <summary>
    /// Gets the number of languages the classifier head predicts. Defaults to the
    /// paper's 107-language VoxLingua107 label set, or the configured
    /// <see cref="NeuralNetworkArchitecture{T}.OutputSize"/> when one is given.
    /// </summary>
    public int NumLanguages => _numLanguages;

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    public int EmbeddingDimension => _options.EmbeddingDimension;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a VoxLingua107 identifier with ONNX model for inference.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="options">VoxLingua107 options.</param>
    public VoxLingua107Identifier(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        VoxLingua107Options? options = null)
        : base(architecture, new CrossEntropyWithLogitsLoss<T>())
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model file not found: {modelPath}");

        _numOps = MathHelper.GetNumericOperations<T>();
        _options = options ?? new VoxLingua107Options();
        Options = _options;
        _options.ModelPath = modelPath;

        // The ONNX graph's output width is the paper's 107 by default; honour a
        // configured output size for a re-headed export.
        _numLanguages = architecture.OutputSize > 0 ? architecture.OutputSize : 107;

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

        // Initialize all 107 language mappings
        (_languageIdToCode, _languageCodeToId, _languageCodeToName) = InitializeVoxLingua107Mappings();

        // Load ONNX model
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);

    }

    /// <summary>
    /// Creates a VoxLingua107 identifier for native training.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="options">VoxLingua107 options.</param>
    /// <param name="optimizer">Optimizer for training.</param>
    /// <param name="lossFunction">Loss function.</param>
    public VoxLingua107Identifier(
        NeuralNetworkArchitecture<T> architecture,
        VoxLingua107Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>())
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _options = options ?? new VoxLingua107Options();
        Options = _options;

        // Honour a configured output size; fall back to the paper's 107 languages.
        _numLanguages = architecture.OutputSize > 0 ? architecture.OutputSize : 107;

        SampleRate = _options.SampleRate;
        NumMels = _options.NumMels;

        _lossFunction = lossFunction ?? new CrossEntropyWithLogitsLoss<T>();
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                EnableGradientClipping = true,
                MaxGradientNorm = 1.0
            });

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

        // Initialize all 107 language mappings
        (_languageIdToCode, _languageCodeToId, _languageCodeToName) = InitializeVoxLingua107Mappings();

        InitializeLayers();
    }

    #endregion

    #region Layer Initialization

    private void InitializeNativeLayers()
    {
        // Build the default ECAPA-TDNN stack from the shared factory rather than
        // inline, so this model follows the same "custom layers or LayerHelper
        // defaults" contract as the rest of the framework. The factory yields the
        // layers in a fixed order; the role-aware forward keeps them in typed
        // groups, so partition the flat list back into those roles here.
        var built = LayerHelper<T>.CreateDefaultVoxLingua107Layers(
            Architecture,
            numMels: _options.NumMels,
            tdnnChannels: _options.TdnnChannels,
            embeddingDimension: _options.EmbeddingDimension,
            dilations: _options.Dilations).ToList();

        int index = 0;

        // Initial TDNN: DenseLayer + BatchNormalizationLayer.
        _tdnnLayers.Add(built[index++]);
        _tdnnLayers.Add(built[index++]);

        // One SE-Res2 block per dilation: the factory emits six residual-path
        // layers followed by two squeeze-excitation layers. The forward keeps
        // residual and SE layers in separate lists indexed 6-per-block and
        // 2-per-block, so de-interleave them here.
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
        _classifierLayer = (DenseLayer<T>)built[index];
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
        for (int i = 0; i < probabilities.Length && i < VoxLingua107Languages.Length; i++)
        {
            result[VoxLingua107Languages[i]] = probabilities[i];
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

    /// <summary>
    /// Gets all languages in a specific language family.
    /// </summary>
    /// <param name="family">Language family (e.g., "germanic", "romance", "slavic").</param>
    /// <returns>List of language codes in that family.</returns>
    public IReadOnlyList<string> GetLanguagesByFamily(string family)
    {
        return family.ToLowerInvariant() switch
        {
            "germanic" => ["en", "de", "nl", "sv", "da", "no", "nn", "is", "af", "yi", "lb", "fo"],
            "romance" => ["es", "fr", "it", "pt", "ro", "ca", "gl", "oc", "la"],
            "slavic" => ["ru", "uk", "be", "pl", "cs", "sk", "bg", "mk", "sr", "hr", "bs", "sl"],
            "sino-tibetan" => ["zh", "bo", "my"],
            "semitic" => ["ar", "he", "am"],
            "indic" => ["hi", "bn", "pa", "gu", "mr", "ne", "si", "sa", "ur", "sd", "as"],
            "turkic" => ["tr", "az", "uz", "kk", "tt", "tk", "ba"],
            "austronesian" => ["id", "ms", "tl", "jv", "su", "ceb", "war", "haw", "mi", "mg"],
            "dravidian" => ["ta", "te", "kn", "ml"],
            "japonic" => ["ja"],
            "koreanic" => ["ko"],
            "uralic" => ["fi", "et", "hu"],
            _ => []
        };
    }

    #endregion

    #region AudioNeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        return _mfccExtractor.Extract(rawAudio);
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
            // TrainWithTape performs the complete forward, loss, backward, and
            // configured-optimizer step.  The previous implementation calculated
            // a loss derivative and discarded it, then asked the optimizer to
            // update layers that had never received gradients.
            TrainWithTape(PreprocessAudio(input), expectedOutput, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => ForwardNative(input);

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Version = "1.0.0",
            AdditionalInfo = new Dictionary<string, object>
            {
                { "Architecture", "VoxLingua107 (ECAPA-TDNN)" },
                { "EmbeddingDimension", _options.EmbeddingDimension },
                { "NumLanguages", _numLanguages },
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

        // Build the paper-faithful ECAPA-TDNN topology and publish those exact
        // instances through Layers.  ForwardNative needs their block roles to
        // apply SE gates, residuals, MFA concatenation, and attentive statistics
        // pooling, while the framework needs the same instances for parameter,
        // gradient, optimizer, serialization, and clone discovery.
        InitializeNativeLayers();
        Layers.AddRange(GetAllLayers());
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(IsOnnxMode);
        writer.Write(SampleRate);
        writer.Write(_options.EmbeddingDimension);
        writer.Write(_options.TdnnChannels);
        writer.Write(_numLanguages); // classifier head width (paper default 107)
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read configuration values for validation
        _ = reader.ReadBoolean(); // IsOnnxMode
        _ = reader.ReadInt32();   // SampleRate
        _ = reader.ReadInt32();   // EmbeddingDimension
        _ = reader.ReadInt32();   // TdnnChannels
        _ = reader.ReadInt32();   // NumLanguages
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
        // A caller-supplied architecture is an ordinary custom layer chain; the
        // ECAPA role-aware traversal below applies only to the default topology.
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

        // Collect MFA outputs and track lengths for gradient flow
        var blockOutputs = new List<Tensor<T>>();
        _blockOutputLengths.Clear();

        int blockIdx = 0;
        foreach (int _ in _options.Dilations)
        {
            var residual = output;

            for (int i = 0; i < 6 && blockIdx * 6 + i < _resBlocks.Count; i++)
            {
                output = _resBlocks[blockIdx * 6 + i].Forward(output);
            }

            int seIdx = blockIdx * 2;
            if (seIdx < _seBlocks.Count)
            {
                var pooled = GlobalAveragePooling(output);
                var attention = _seBlocks[seIdx].Forward(pooled);
                if (seIdx + 1 < _seBlocks.Count)
                {
                    attention = _seBlocks[seIdx + 1].Forward(attention);
                }
                output = ApplyChannelAttention(output, attention);
            }

            output = AddTensors(output, residual);
            blockOutputs.Add(output);
            _blockOutputLengths.Add(output.Length);
            blockIdx++;
        }

        output = ConcatenateTensors(blockOutputs);

        if (_poolingLayer is not null)
        {
            output = AttentiveStatisticsPooling(output);
            output = _poolingLayer.Forward(output);
        }

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

    private Tensor<T> GlobalAveragePooling(Tensor<T> input)
    {
        // ECAPA activations are time-major [time, channels]. Keep the
        // reduction on the engine so the SE gate remains part of the tape.
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
        broadcastShape[^1] = attention.Length;
        var channelGate = Engine.Reshape(attention, broadcastShape);
        return Engine.TensorBroadcastMultiply(input, channelGate);
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
        var centered = Engine.TensorBroadcastSubtract(input, meanKeepDims);
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
        InitializeVoxLingua107Mappings()
    {
        var idToCode = new Dictionary<int, string>();
        var codeToId = new Dictionary<string, int>();
        var codeToName = GetVoxLingua107LanguageNames();

        for (int i = 0; i < VoxLingua107Languages.Length; i++)
        {
            idToCode[i] = VoxLingua107Languages[i];
            codeToId[VoxLingua107Languages[i]] = i;
        }

        return (idToCode, codeToId, codeToName);
    }

    private static Dictionary<string, string> GetVoxLingua107LanguageNames()
    {
        return new Dictionary<string, string>
        {
            ["ab"] = "Abkhazian",
            ["af"] = "Afrikaans",
            ["am"] = "Amharic",
            ["ar"] = "Arabic",
            ["as"] = "Assamese",
            ["az"] = "Azerbaijani",
            ["ba"] = "Bashkir",
            ["be"] = "Belarusian",
            ["bg"] = "Bulgarian",
            ["bn"] = "Bengali",
            ["bo"] = "Tibetan",
            ["br"] = "Breton",
            ["bs"] = "Bosnian",
            ["ca"] = "Catalan",
            ["ceb"] = "Cebuano",
            ["cs"] = "Czech",
            ["cy"] = "Welsh",
            ["da"] = "Danish",
            ["de"] = "German",
            ["el"] = "Greek",
            ["en"] = "English",
            ["eo"] = "Esperanto",
            ["es"] = "Spanish",
            ["et"] = "Estonian",
            ["eu"] = "Basque",
            ["fa"] = "Persian",
            ["fi"] = "Finnish",
            ["fo"] = "Faroese",
            ["fr"] = "French",
            ["gl"] = "Galician",
            ["gn"] = "Guarani",
            ["gu"] = "Gujarati",
            ["ha"] = "Hausa",
            ["haw"] = "Hawaiian",
            ["he"] = "Hebrew",
            ["hi"] = "Hindi",
            ["hr"] = "Croatian",
            ["ht"] = "Haitian Creole",
            ["hu"] = "Hungarian",
            ["hy"] = "Armenian",
            ["ia"] = "Interlingua",
            ["id"] = "Indonesian",
            ["is"] = "Icelandic",
            ["it"] = "Italian",
            ["ja"] = "Japanese",
            ["jv"] = "Javanese",
            ["ka"] = "Georgian",
            ["kk"] = "Kazakh",
            ["km"] = "Khmer",
            ["kn"] = "Kannada",
            ["ko"] = "Korean",
            ["la"] = "Latin",
            ["lb"] = "Luxembourgish",
            ["ln"] = "Lingala",
            ["lo"] = "Lao",
            ["lt"] = "Lithuanian",
            ["lv"] = "Latvian",
            ["mg"] = "Malagasy",
            ["mi"] = "Maori",
            ["mk"] = "Macedonian",
            ["ml"] = "Malayalam",
            ["mn"] = "Mongolian",
            ["mr"] = "Marathi",
            ["ms"] = "Malay",
            ["mt"] = "Maltese",
            ["my"] = "Burmese",
            ["ne"] = "Nepali",
            ["nl"] = "Dutch",
            ["nn"] = "Norwegian Nynorsk",
            ["no"] = "Norwegian",
            ["oc"] = "Occitan",
            ["pa"] = "Punjabi",
            ["pl"] = "Polish",
            ["ps"] = "Pashto",
            ["pt"] = "Portuguese",
            ["ro"] = "Romanian",
            ["ru"] = "Russian",
            ["sa"] = "Sanskrit",
            ["sd"] = "Sindhi",
            ["si"] = "Sinhala",
            ["sk"] = "Slovak",
            ["sl"] = "Slovenian",
            ["sn"] = "Shona",
            ["so"] = "Somali",
            ["sq"] = "Albanian",
            ["sr"] = "Serbian",
            ["su"] = "Sundanese",
            ["sv"] = "Swedish",
            ["sw"] = "Swahili",
            ["ta"] = "Tamil",
            ["te"] = "Telugu",
            ["tg"] = "Tajik",
            ["th"] = "Thai",
            ["tk"] = "Turkmen",
            ["tl"] = "Tagalog",
            ["tr"] = "Turkish",
            ["tt"] = "Tatar",
            ["uk"] = "Ukrainian",
            ["ur"] = "Urdu",
            ["uz"] = "Uzbek",
            ["vi"] = "Vietnamese",
            ["war"] = "Waray",
            ["xh"] = "Xhosa",
            ["yi"] = "Yiddish",
            ["yo"] = "Yoruba",
            ["zh"] = "Chinese",
            ["zu"] = "Zulu"
        };
    }

    #endregion
}
