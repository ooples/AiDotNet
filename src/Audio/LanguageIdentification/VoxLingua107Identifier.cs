using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Features;
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
/// Console.WriteLine($"Language: {result.LanguageName} ({result.Confidence:P0})");
/// // Output: Language: Swedish (85%)
///
/// // Get top 5 predictions
/// var topLanguages = model.GetTopLanguages(audioTensor, 5);
/// foreach (var (lang, prob) in topLanguages)
///     Console.WriteLine($"  {lang}: {prob:P1}");
/// </code>
/// </para>
/// </remarks>
public class VoxLingua107Identifier<T> : AudioNeuralNetworkBase<T>, ILanguageIdentifier<T>
{
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
    private readonly MfccExtractor<T> _mfccExtractor;
    private readonly ILossFunction<T> _lossFunction;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    // ECAPA-TDNN architecture (same as ECAPATDNNLanguageIdentifier)
    private readonly List<ILayer<T>> _tdnnLayers = [];
    private readonly List<ILayer<T>> _seBlocks = [];
    private readonly List<ILayer<T>> _resBlocks = [];
    private DenseLayer<T>? _poolingLayer;
    private DenseLayer<T>? _classifierLayer;
    private BatchNormalizationLayer<T>? _finalBatchNorm;

    // Language mapping for 107 languages
    private readonly Dictionary<int, string> _languageIdToCode;
    private readonly Dictionary<string, int> _languageCodeToId;
    private readonly Dictionary<string, string> _languageCodeToName;

    // MFA gradient flow tracking
    private readonly List<int> _blockOutputLengths = [];
    private Tensor<T>? _lastTdnnOutput;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override bool SupportsTraining => !IsOnnxMode;

    /// <inheritdoc/>
    public IReadOnlyList<string> SupportedLanguages => VoxLingua107Languages.ToList();

    /// <summary>
    /// Gets the number of supported languages (107).
    /// </summary>
    public int NumLanguages => 107;

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
        : base(architecture, new CrossEntropyLoss<T>())
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model file not found: {modelPath}");

        _numOps = MathHelper.GetNumericOperations<T>();
        _options = options ?? new VoxLingua107Options();
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

        // Initialize all 107 language mappings
        (_languageIdToCode, _languageCodeToId, _languageCodeToName) = InitializeVoxLingua107Mappings();

        // Load ONNX model
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);

        // Initialize optimizer (not used in ONNX mode but required for readonly field)
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
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
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>())
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _options = options ?? new VoxLingua107Options();

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

        // Initialize all 107 language mappings
        (_languageIdToCode, _languageCodeToId, _languageCodeToName) = InitializeVoxLingua107Mappings();

        InitializeNativeLayers();
    }

    #endregion

    #region Layer Initialization

    private void InitializeNativeLayers()
    {
        int inputDim = _options.NumMels * 3;
        int channels = _options.TdnnChannels;

        // Initial TDNN layer
        _tdnnLayers.Add(new DenseLayer<T>(inputDim, channels, (IActivationFunction<T>)new ReLUActivation<T>()));
        _tdnnLayers.Add(new BatchNormalizationLayer<T>(channels));

        // ECAPA-TDNN SE-Res2 blocks
        foreach (int dilation in _options.Dilations)
        {
            AddSERes2Block(channels, dilation);
        }

        // MFA output dimension
        int mfaOutputDim = channels * _options.Dilations.Length;

        // Attentive Statistics Pooling
        _poolingLayer = new DenseLayer<T>(mfaOutputDim, _options.EmbeddingDimension * 2);

        // Final layers
        _finalBatchNorm = new BatchNormalizationLayer<T>(_options.EmbeddingDimension);
        _classifierLayer = new DenseLayer<T>(_options.EmbeddingDimension, 107); // 107 languages
    }

    private void AddSERes2Block(int channels, int dilation)
    {
        // 1x1 reduction
        _resBlocks.Add(new DenseLayer<T>(channels, channels / 4, (IActivationFunction<T>)new ReLUActivation<T>()));
        _resBlocks.Add(new BatchNormalizationLayer<T>(channels / 4));

        // Dilated conv
        _resBlocks.Add(new DenseLayer<T>(channels / 4, channels / 4, (IActivationFunction<T>)new ReLUActivation<T>()));
        _resBlocks.Add(new BatchNormalizationLayer<T>(channels / 4));

        // 1x1 expansion
        _resBlocks.Add(new DenseLayer<T>(channels / 4, channels, (IActivationFunction<T>)new ReLUActivation<T>()));
        _resBlocks.Add(new BatchNormalizationLayer<T>(channels));

        // SE block
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
                { "Architecture", "VoxLingua107 (ECAPA-TDNN)" },
                { "EmbeddingDimension", _options.EmbeddingDimension },
                { "NumLanguages", 107 },
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

        // Use LayerHelper to create default VoxLingua107 layers (ECAPA-TDNN with 107 languages)
        Layers.AddRange(LayerHelper<T>.CreateDefaultVoxLingua107Layers(
            Architecture,
            numMels: _options.NumMels,
            tdnnChannels: _options.TdnnChannels,
            embeddingDimension: _options.EmbeddingDimension,
            dilations: _options.Dilations));
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(IsOnnxMode);
        writer.Write(SampleRate);
        writer.Write(_options.EmbeddingDimension);
        writer.Write(_options.TdnnChannels);
        writer.Write(107); // NumLanguages is always 107 for VoxLingua107
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

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new VoxLingua107Identifier<T>(
            Architecture,
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

            var seOutput = output;
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

    private void BackwardNative(Tensor<T> gradient)
    {
        var grad = gradient;

        // Backward through classifier
        if (_classifierLayer is not null)
            grad = _classifierLayer.Backward(grad);

        // Backward through final batch norm
        if (_finalBatchNorm is not null)
            grad = _finalBatchNorm.Backward(grad);

        // Backward through pooling layer
        if (_poolingLayer is not null)
            grad = _poolingLayer.Backward(grad);

        // MFA gradient splitting: the forward pass concatenated outputs from each block
        // We need to split the gradient back to each block's portion
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

        // Backward through each SE-Res2 block (in reverse order)
        Tensor<T>? tdnnGrad = null;
        for (int blockIdx = _blockOutputLengths.Count - 1; blockIdx >= 0; blockIdx--)
        {
            var blockGrad = blockGradients[blockIdx];

            // Backward through SE block for this block
            int seIdx = blockIdx * 2;
            if (seIdx + 1 < _seBlocks.Count)
            {
                blockGrad = _seBlocks[seIdx + 1].Backward(blockGrad);
            }
            if (seIdx < _seBlocks.Count)
            {
                blockGrad = _seBlocks[seIdx].Backward(blockGrad);
            }

            // Backward through res block layers (6 layers per block)
            for (int i = 5; i >= 0; i--)
            {
                int layerIdx = blockIdx * 6 + i;
                if (layerIdx < _resBlocks.Count)
                {
                    blockGrad = _resBlocks[layerIdx].Backward(blockGrad);
                }
            }

            // Accumulate gradient for TDNN output (residual connection)
            if (tdnnGrad is null)
            {
                tdnnGrad = blockGrad;
            }
            else
            {
                // Add gradients from multiple blocks
                var combined = new T[Math.Max(tdnnGrad.Length, blockGrad.Length)];
                for (int i = 0; i < combined.Length; i++)
                {
                    T val1 = i < tdnnGrad.Length ? tdnnGrad.GetFlat(i) : _numOps.Zero;
                    T val2 = i < blockGrad.Length ? blockGrad.GetFlat(i) : _numOps.Zero;
                    combined[i] = _numOps.Add(val1, val2);
                }
                tdnnGrad = new Tensor<T>(combined, [combined.Length]);
            }
        }

        // Backward through TDNN layers
        if (tdnnGrad is not null)
        {
            for (int i = _tdnnLayers.Count - 1; i >= 0; i--)
            {
                tdnnGrad = _tdnnLayers[i].Backward(tdnnGrad);
            }
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
        int features = input.Shape.Length > 1 ? input.Shape[^1] : input.Length;
        int timeSteps = input.Length / features;

        var output = new T[features];
        for (int f = 0; f < features; f++)
        {
            double sum = 0;
            for (int t = 0; t < timeSteps; t++)
                sum += _numOps.ToDouble(input[t * features + f]);
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
            output[i] = _numOps.Add(a[i], b[i]);
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
        int features = input.Shape.Length > 1 ? input.Shape[^1] : input.Length / 10;
        int timeSteps = input.Length / features;

        var mean = new T[features];
        for (int f = 0; f < features; f++)
        {
            double sum = 0;
            for (int t = 0; t < timeSteps; t++)
                sum += _numOps.ToDouble(input[t * features + f]);
            mean[f] = _numOps.FromDouble(sum / timeSteps);
        }

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
