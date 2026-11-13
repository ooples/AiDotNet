using AiDotNet.Optimizers;
using AiDotNet.LearningRateSchedulers;
using System.IO;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using Microsoft.ML.OnnxRuntime;
using AiDotNet.Validation;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

using System.Collections.Generic;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Flamingo neural network for in-context visual learning and few-shot tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Flamingo is a visual language model that excels at few-shot learning. It uses a Perceiver
/// Resampler to compress visual features and gated cross-attention layers to integrate
/// visual information into a frozen language model.
/// </para>
/// <para><b>For Beginners:</b> Flamingo is a visual AI that can answer questions about images
/// with just a few examples. Show it 2-3 examples of image-answer pairs, and it can handle
/// similar questions about new images. It works by feeding image features into a language
/// model through special cross-attention layers, letting the language model "see" the image
/// while generating text responses.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 8, outputSize: 4);
/// var image = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 3, 224, 224 });
/// var trainX = Tensor&lt;float&gt;.CreateRandom(4, 8);
/// var trainY = Tensor&lt;float&gt;.CreateRandom(4, 2);
/// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(new FlamingoNeuralNetwork&lt;float&gt;(architecture))
///     .Build(trainX, trainY);
/// var output = result.Predict(image);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.Multimodal)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Flamingo: a Visual Language Model for Few-Shot Learning", "https://arxiv.org/abs/2204.14198", Year = 2022, Authors = "Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, Karen Simonyan")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, WeightDecay = 0.1,
                WarmupSteps = 5000, MaxGradientNorm = 1.0,
                Schedule = LearningRateSchedulerType.LinearWarmup,
                PostWarmupDecay = LinearWarmupScheduler.DecayMode.Constant,
                Source = "Alayrac et al. 2022, Sec. 3.4: AdamW with global norm clipping of 1 and a "
                        + "weight decay of 0.1 on the trainable parameters other than the Perceiver "
                        + "Resampler, which uses none. The learning rate rises linearly from 0 to 1e-4 "
                        + "over the first 5000 steps and is then held constant -- the paper states it "
                        + "observed no improvement from decaying it, so the constant tail is the stated "
                        + "choice rather than an omission.")]
public partial class FlamingoNeuralNetwork<T> : MultimodalModelLayoutBase<T>, IFlamingoModel<T>
{
    private readonly FlamingoOptions _options;

    #region Native Mode Fields

    private readonly List<ILayer<T>> _visionEncoderLayers = [];
    private readonly List<ILayer<T>> _perceiverLayers = [];
    private readonly List<ILayer<T>> _gatedCrossAttentionLayers = [];
    private readonly List<ILayer<T>> _languageModelLayers = [];
    private Tensor<T>? _perceiverQueries;
    private Tensor<T>? _visionPositionalEmbeddings;
    private ILayer<T>? _patchEmbedding;
    private ILayer<T>? _textTokenEmbedding;
    private Tensor<T>? _textPositionalEmbeddings;
    private ILayer<T>? _outputProjection;

    #endregion

    #region Shared Fields

    private readonly ITokenizer _tokenizer;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly int _embeddingDimension;
    private readonly int _maxSequenceLength;
    private readonly int _imageSize;

    /// <summary>
    /// The input channel count the native layers were built for.
    /// </summary>
    /// <remarks>
    /// The constructor took this and forwarded it to InitializeNativeLayers without keeping it, so
    /// two places in the clone path assumed 3. The shape probe built a 3-channel tensor, which
    /// makes ResolveShapes throw for a 1- or 4-channel model and drops the copy into the fallback
    /// -- copying from unresolved lazy projections, the exact failure that probe was added to
    /// prevent -- and CreateNewInstance passed the literal 3, silently rebuilding the clone with an
    /// RGB patch embedding whatever the original used.
    /// </remarks>
    private readonly int _channels;
    private readonly int _visionHiddenDim;
    private readonly int _lmHiddenDim;
    private readonly int _numVisionLayers;
    private readonly int _numLmLayers;
    private readonly int _numHeads;
    private readonly int _patchSize;
    private readonly int _vocabularySize;
    private readonly LanguageModelBackbone _languageModelBackbone;
    private readonly int _numPerceiverTokens;
    private readonly int _maxImagesInContext;
    private readonly int _numPerceiverLayers;
    private readonly double _learningRate;

    #endregion

    #region IMultimodalEmbedding Properties

    /// <inheritdoc/>
    public int EmbeddingDimension => _embeddingDimension;

    /// <inheritdoc/>
    public int MaxSequenceLength => _maxSequenceLength;

    /// <inheritdoc/>
    public int ImageSize => _imageSize;

    #endregion

    #region IFlamingoModel Properties

    /// <inheritdoc/>
    public int NumPerceiverTokens => _numPerceiverTokens;

    /// <inheritdoc/>
    public int MaxImagesInContext => _maxImagesInContext;

    /// <inheritdoc/>
    public LanguageModelBackbone LanguageModelBackbone => _languageModelBackbone;

    #endregion

    #region Constructors

    /// <summary>
    /// ONNX construction is not supported: a vision encoder and a language model alone cannot run Flamingo.
    /// </summary>
    /// <remarks>
    /// Flamingo (Alayrac et al. 2022) conditions a frozen language model on images through a Perceiver Resampler
    /// and gated cross-attention layers interleaved between the LM blocks. This two-file overload received only a
    /// vision encoder and a plain language model, so it validated the graphs, reported success, and then failed on
    /// every image-embedding, prediction and generation call because neither the resampler nor the cross-attention
    /// path existed. It now fails at construction, before any session is opened. Use the native constructor.
    /// </remarks>
    /// <exception cref="NotSupportedException">Always.</exception>
    public FlamingoNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        string visionEncoderPath,
        string languageModelPath,
        ITokenizer tokenizer,
        FlamingoOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        throw new NotSupportedException(
            "FlamingoNeuralNetwork cannot run from a vision encoder and a language model alone. ONNX inference would " +
            "also need the exported Perceiver Resampler and a language model with gated cross-attention layers " +
            "(inputs for the resampled visual tokens), which this constructor does not accept. " +
            "Use the native FlamingoNeuralNetwork(architecture, options, ...) constructor instead.");
    }

    /// <summary>
    /// Initializes a new instance using native layers.
    /// </summary>
    public FlamingoNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        FlamingoOptions? options = null,
        ITokenizer? tokenizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new FlamingoOptions();
        _options.Validate();

        Options = _options;
        _embeddingDimension = _options.EmbeddingDimension;
        _maxSequenceLength = _options.MaxSequenceLength;
        _imageSize = _options.ImageSize;
        _numPerceiverTokens = _options.NumPerceiverTokens;
        _maxImagesInContext = _options.MaxImagesInContext;
        _visionHiddenDim = _options.VisionDim;
        _lmHiddenDim = _options.LmHiddenDim;
        _numVisionLayers = _options.VisionLayers;
        _numLmLayers = _options.NumLmLayers;
        _numHeads = _options.NumHeads;
        _patchSize = _options.PatchSize;
        _vocabularySize = _options.VocabSize;
        _languageModelBackbone = _options.LanguageModelBackbone;
        _numPerceiverLayers = _options.NumPerceiverLayers;
        _learningRate = _options.LearningRate;

        // Use factory to create appropriate tokenizer for the backbone, or use provided tokenizer
        _tokenizer = tokenizer ?? Tokenization.LanguageModelTokenizerFactory.CreateForBackbone(_options.LanguageModelBackbone);
        // Hand-built on purpose (options learning rate, gradient clipping), so it is verified against
        // the declared recipe rather than replaced by it. This is the only construction left now the
        // ONNX constructor is gone, so it is also what keeps [PaperOptimizer] wired (AIDN104).
        _optimizer = optimizer ?? PaperOptimizerFactory.VerifyHandBuilt(this,
            new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
                this,
                new AiDotNet.Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = _options.LearningRate,
                    MaxGradientNorm = 1.0
                }));
        _lossFunction = lossFunction ?? new CrossEntropyWithLogitsLoss<T>();

        _channels = _options.Channels;
        InitializeNativeLayers(_options.Channels);
    }

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        // Flamingo has one execution mode. Build the same graph the constructor builds, so a base path
        // that re-initializes layers ends with a bound model rather than an empty one. Idempotent:
        // InitializeNativeLayers clears Layers before rebuilding.
        InitializeNativeLayers(_channels);
    }

    private void InitializeNativeLayers(int channels)
    {
        int numPatches = (_imageSize / _patchSize) * (_imageSize / _patchSize);
        int gatedCrossAttnCount = _numLmLayers / 4;

        Layers.Clear();

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateFlamingoLayers(
                _imageSize, channels, _patchSize, _visionHiddenDim, _lmHiddenDim,
                _numVisionLayers, _numPerceiverLayers, _numPerceiverTokens,
                _numLmLayers, _numHeads, _vocabularySize, _maxSequenceLength));
        }

        RebindNativeLayerReferences();

        // Initialize vision positional embeddings
        _visionPositionalEmbeddings = new Tensor<T>([numPatches + 1, _visionHiddenDim]);
        InitializePositionalEmbeddings(_visionPositionalEmbeddings);

        // Initialize perceiver queries
        _perceiverQueries = new Tensor<T>([_numPerceiverTokens, _lmHiddenDim]);
        InitializePerceiverQueries(_perceiverQueries);

        // Text positional embeddings
        _textPositionalEmbeddings = new Tensor<T>([_maxSequenceLength, _lmHiddenDim]);
        InitializePositionalEmbeddings(_textPositionalEmbeddings);
    }

    /// <summary>
    /// Rebinds Flamingo's branch-specific layer references to the canonical base
    /// <see cref="NeuralNetworkBase{T}.Layers"/> collection.
    /// </summary>
    /// <remarks>
    /// The base clone paths replace or COW-share the canonical layer collection. Flamingo executes
    /// through its branch lists rather than walking that collection sequentially, so those lists must
    /// be rebound after cloning; copying parameters into the constructor's stale branch objects would
    /// duplicate storage and bypass the base COW contract.
    /// </remarks>
    private void RebindNativeLayerReferences()
    {
        int gatedCrossAttnCount = _numLmLayers / 4;
        int requiredLayers = 1 + _numVisionLayers + (3 * _numPerceiverLayers)
            + gatedCrossAttnCount + 1 + _numLmLayers + 1;
        // EXACTLY requiredLayers, NOT "at least". The distribution loop below consumes exactly
        // requiredLayers entries, so a surplus passed this guard and was then bound to nothing --
        // yet stayed in the canonical Layers collection, so it was still serialized, still counted
        // toward ParameterCount, and still updated by training, while contributing nothing to any
        // forward path. A caller who supplied one layer too many got a model that trained weights
        // it never used and reported a parameter count it could not explain.
        if (Layers.Count != requiredLayers)
        {
            throw new InvalidOperationException(
                $"Flamingo layer graph contains {Layers.Count} layers but exactly {requiredLayers} are "
                + "required; every layer is bound to a named execution branch, so a surplus would be "
                + "trained and serialized without ever being used.");
        }

        // Distribute canonical layers to the private execution branches.
        int idx = 0;

        // Patch embedding
        _patchEmbedding = Layers[idx++];

        // Vision encoder transformer layers
        _visionEncoderLayers.Clear();
        for (int i = 0; i < _numVisionLayers; i++)
            _visionEncoderLayers.Add(Layers[idx++]);

        // Perceiver Resampler layers: (CrossAttn + FFN_expand + FFN_contract) × numPerceiverLayers
        _perceiverLayers.Clear();
        for (int i = 0; i < _numPerceiverLayers; i++)
        {
            _perceiverLayers.Add(Layers[idx++]); // CrossAttention
            _perceiverLayers.Add(Layers[idx++]); // FFN expand
            _perceiverLayers.Add(Layers[idx++]); // FFN contract
        }

        // Gated cross-attention layers
        _gatedCrossAttentionLayers.Clear();
        for (int i = 0; i < gatedCrossAttnCount; i++)
            _gatedCrossAttentionLayers.Add(Layers[idx++]);

        // Text token embedding
        _textTokenEmbedding = Layers[idx++];

        // Language model transformer layers
        _languageModelLayers.Clear();
        for (int i = 0; i < _numLmLayers; i++)
            _languageModelLayers.Add(Layers[idx++]);

        // Output projection
        _outputProjection = Layers[idx++];
    }

    private void InitializePositionalEmbeddings(Tensor<T> embeddings)
    {
        int positions = embeddings.Shape[0];
        int dim = embeddings.Shape[1];
        if (positions == 0 || dim == 0) return;

        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            // angle[i, j] = i / 10000^(2*(j/2) / dim): column j uses its sin/cos pair index. The table is
            // a constant built in double (angles reach the position count in radians, beyond what float or
            // Half carries to sin/cos) and rounded to T once, as the per-element double math was.
            var pairStart = Engine.TensorSlice(
                Engine.TensorRepeatInterleave(Engine.TensorArange<double>(0, dim, 2), 2, 0), [0], [dim]);
            var tenThousands = new Tensor<double>([dim]);
            tenThousands.Fill(10000.0);
            var denominator = Engine.TensorPower(tenThousands, Engine.TensorDivideScalar(pairStart, (double)dim));
            var angle = Engine.TensorDivide(
                Engine.TensorBroadcastTo(Engine.Reshape(Engine.TensorArange<double>(0, positions), [positions, 1]), [positions, dim]),
                Engine.TensorBroadcastTo(Engine.Reshape(denominator, [1, dim]), [positions, dim]));

            // Even columns take sin, odd columns cos
            var evenColumn = new bool[positions * dim];
            for (int k = 0; k < evenColumn.Length; k++)
            {
                evenColumn[k] = (k % dim) % 2 == 0;
            }
            Engine.TensorWhere(evenColumn, Engine.TensorSin(angle), Engine.TensorCos(angle)).Cast<T>()
                .CopyTo(embeddings.AsWritableSpan());
        }
    }

    private void InitializePerceiverQueries(Tensor<T> queries)
    {
        var rand = RandomHelper.CreateSeededRandom(42);
        double scale = 1.0 / Math.Sqrt(queries.Shape[1]);
        for (int i = 0; i < queries.Shape[0]; i++)
        {
            for (int j = 0; j < queries.Shape[1]; j++)
            {
                double value = (rand.NextDouble() * 2 - 1) * scale;
                queries[i, j] = NumOps.FromDouble(value);
            }
        }
    }

    #endregion

    #region IMultimodalEmbedding Implementation

    /// <inheritdoc/>
    public Vector<T> GetImageEmbedding(Tensor<T> image)
    {
        return GetImageEmbeddings([image]).First();
    }

    /// <inheritdoc/>
    public IEnumerable<Vector<T>> GetImageEmbeddings(IEnumerable<Tensor<T>> images)
    {
        var results = new List<Vector<T>>();
        foreach (var image in images)
        {
            var features = ExtractPerceiverFeatures(image);
            var embedding = MeanPool(features);
            var normalized = Normalize(embedding);
            results.Add(normalized);
        }
        return results;
    }

    /// <inheritdoc/>
    public Vector<T> GetTextEmbedding(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            throw new ArgumentException("Text cannot be null or empty.", nameof(text));

        return GetTextEmbeddings([text]).First();
    }

    /// <inheritdoc/>
    public IEnumerable<Vector<T>> GetTextEmbeddings(IEnumerable<string> texts)
    {
        var results = new List<Vector<T>>();

        foreach (var text in texts)
        {
            var encoded = _tokenizer.Encode(text);
            var inputIds = encoded.TokenIds;

            var paddedIds = new List<int>();
            for (int i = 0; i < _maxSequenceLength; i++)
            {
                paddedIds.Add(i < inputIds.Count ? inputIds[i] : 0);
            }

            var embedded = EmbedTextTokens(paddedIds);
            var embedding = MeanPool(embedded);
            var normalized = Normalize(embedding);
            results.Add(normalized);
        }

        return results;
    }

    private Tensor<T> EmbedTextTokens(IReadOnlyList<int> tokenIds)
    {
        int seqLen = tokenIds.Count;
        var embeddings = Tensor<T>.CreateDefault([seqLen, _lmHiddenDim], NumOps.Zero);

        if (_textTokenEmbedding is null || _textPositionalEmbeddings is null)
        {
            return embeddings;
        }

        if (seqLen == 0 || _lmHiddenDim == 0)
        {
            return embeddings;
        }

        var ids = new Tensor<int>(tokenIds.ToArray(), [seqLen]).Cast<T>();
        var tokenRows = new Tensor<T>[seqLen];
        for (int i = 0; i < seqLen; i++)
        {
            var tokenEmb = _textTokenEmbedding.Forward(Engine.TensorSlice(ids, [i], [1]));
            tokenRows[i] = Engine.TensorSlice(tokenEmb, [0, 0], [1, _lmHiddenDim]);
        }

        // token row + position row (zero past the table), as fresh values off the tape
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            int withPosition = Math.Min(seqLen, _textPositionalEmbeddings.Shape[0]);
            var positions = withPosition == seqLen
                ? Engine.TensorSlice(_textPositionalEmbeddings, [0, 0], [seqLen, _lmHiddenDim])
                : Engine.TensorConcatenate(new[]
                {
                    Engine.TensorSlice(_textPositionalEmbeddings, [0, 0], [withPosition, _lmHiddenDim]),
                    new Tensor<T>([seqLen - withPosition, _lmHiddenDim])
                }, 0);
            Engine.TensorAdd(Engine.TensorConcatenate(tokenRows, 0), positions).CopyTo(embeddings.AsWritableSpan());
        }

        return embeddings;
    }

    private Vector<T> MeanPool(Tensor<T> features)
    {
        int seqLen = features.Shape[0];
        int hiddenDim = features.Shape[1];
        var result = new Vector<T>(hiddenDim);

        if (hiddenDim == 0) return result;

        // Column sums over the sequence, divided by its length.
        var sums = seqLen > 0
            ? Engine.ReduceSum(Engine.TensorSlice(features, [0, 0], [seqLen, hiddenDim]), new[] { 0 }, keepDims: false)
            : new Tensor<T>([hiddenDim]);
        Engine.TensorDivideScalar(sums, NumOps.FromDouble(seqLen)).CopyTo(result.AsWritableSpan());

        return result;
    }

    private Vector<T> Normalize(Vector<T> vec)
    {
        return VectorHelper.Normalize(vec);
    }

    /// <inheritdoc/>
    public T ComputeSimilarity(Vector<T> embedding1, Vector<T> embedding2)
    {
        T dotProduct = NumOps.Zero;
        T norm1 = NumOps.Zero;
        T norm2 = NumOps.Zero;

        // Over the common prefix of the two embeddings.
        int length = Math.Min(embedding1.Length, embedding2.Length);
        if (length > 0)
        {
            var first = embedding1.GetSubVector(0, length);
            var second = embedding2.GetSubVector(0, length);
            dotProduct = Engine.DotProduct(first, second);
            norm1 = Engine.DotProduct(first, first);
            norm2 = Engine.DotProduct(second, second);
        }

        T normProduct = NumOps.Multiply(NumOps.Sqrt(norm1), NumOps.Sqrt(norm2));
        if (NumOps.LessThan(normProduct, NumOps.FromDouble(1e-10)))
            return NumOps.Zero;

        return NumOps.Divide(dotProduct, normProduct);
    }

    /// <inheritdoc/>
    public T ComputeImageTextSimilarity(Tensor<T> image, string text)
    {
        var imageEmbedding = GetImageEmbedding(image);
        var textEmbedding = GetTextEmbedding(text);
        return ComputeSimilarity(imageEmbedding, textEmbedding);
    }

    /// <inheritdoc/>
    public Dictionary<string, T> ZeroShotClassify(Tensor<T> image, IEnumerable<string> classLabels)
    {
        var imageEmbedding = GetImageEmbedding(image);
        var result = new Dictionary<string, T>();
        var scores = new List<(string label, T score)>();

        foreach (var label in classLabels)
        {
            var textEmbedding = GetTextEmbedding(label);
            var similarity = ComputeSimilarity(imageEmbedding, textEmbedding);
            scores.Add((label, similarity));
        }

        var expScores = scores.Select(s => (s.label, exp: Math.Exp(NumOps.ToDouble(s.score)))).ToList();
        double sumExp = expScores.Sum(s => s.exp);

        foreach (var (label, exp) in expScores)
        {
            result[label] = NumOps.FromDouble(exp / sumExp);
        }

        return result;
    }

    /// <inheritdoc/>
    public IEnumerable<(int Index, T Score)> RetrieveImages(
        string query,
        IEnumerable<Vector<T>> imageEmbeddings,
        int topK = 10)
    {
        var queryEmbedding = GetTextEmbedding(query);
        var embeddingsList = imageEmbeddings.ToList();
        var scores = new List<(int Index, T Score)>();

        for (int i = 0; i < embeddingsList.Count; i++)
        {
            var similarity = ComputeSimilarity(queryEmbedding, embeddingsList[i]);
            scores.Add((i, similarity));
        }

        return scores.OrderByDescending(s => NumOps.ToDouble(s.Score)).Take(topK);
    }

    /// <inheritdoc/>
    public IEnumerable<(int Index, T Score)> RetrieveTexts(
        Tensor<T> image,
        IEnumerable<string> texts,
        int topK = 10)
    {
        var imageEmbedding = GetImageEmbedding(image);
        var textsList = texts.ToList();
        var scores = new List<(int Index, T Score)>();

        for (int i = 0; i < textsList.Count; i++)
        {
            var textEmbedding = GetTextEmbedding(textsList[i]);
            var similarity = ComputeSimilarity(imageEmbedding, textEmbedding);
            scores.Add((i, similarity));
        }

        return scores.OrderByDescending(s => NumOps.ToDouble(s.Score)).Take(topK);
    }

    /// <inheritdoc/>
    public string GenerateCaption(Tensor<T> image, int maxLength = 77)
    {
        return FewShotGenerate([], image, "Describe this image:", maxLength);
    }

    /// <inheritdoc/>
    public string AnswerQuestion(Tensor<T> image, string question, int maxLength = 64)
    {
        return FewShotVQA([], image, question);
    }

    #endregion

    #region IFlamingoModel Implementation

    /// <inheritdoc/>
    public string FewShotGenerate(
        IEnumerable<(Tensor<T> Image, string Text)> examples,
        Tensor<T> queryImage,
        string? queryPrompt = null,
        int maxLength = 256)
    {
        var examplesList = examples.ToList();
        if (examplesList.Count > _maxImagesInContext)
        {
            throw new ArgumentException(
                $"Too many examples. Maximum allowed: {_maxImagesInContext}",
                nameof(examples));
        }

        var allImageFeatures = new List<Tensor<T>>();
        foreach (var (image, _) in examplesList)
        {
            allImageFeatures.Add(ExtractPerceiverFeatures(image));
        }
        allImageFeatures.Add(ExtractPerceiverFeatures(queryImage));

        var contextBuilder = new List<string>();
        for (int i = 0; i < examplesList.Count; i++)
        {
            contextBuilder.Add($"<image>{examplesList[i].Text}");
        }
        contextBuilder.Add($"<image>{queryPrompt ?? ""}");

        string context = string.Join(" ", contextBuilder);
        var encoded = _tokenizer.Encode(context);
        var inputIds = encoded.TokenIds;

        return GenerateWithVisualContext(inputIds, allImageFeatures, maxLength);
    }

    /// <inheritdoc/>
    public string GenerateWithMultipleImages(
        IEnumerable<Tensor<T>> images,
        string prompt,
        int maxLength = 512)
    {
        var imagesList = images.ToList();
        if (imagesList.Count > _maxImagesInContext)
        {
            throw new ArgumentException(
                $"Too many images. Maximum allowed: {_maxImagesInContext}",
                nameof(images));
        }

        var allImageFeatures = imagesList.Select(img => ExtractPerceiverFeatures(img)).ToList();
        var encoded = _tokenizer.Encode(prompt);
        var inputIds = encoded.TokenIds;

        return GenerateWithVisualContext(inputIds, allImageFeatures, maxLength);
    }

    /// <inheritdoc/>
    public Dictionary<string, T> InContextClassify(
        IEnumerable<(Tensor<T> Image, string Label)> labeledExamples,
        Tensor<T> queryImage)
    {
        var examplesList = labeledExamples.ToList();
        var labels = examplesList.Select(e => e.Label).Distinct().ToList();

        var examples = examplesList.Select(e => (e.Image, $"This is: {e.Label}")).ToList();
        var generated = FewShotGenerate(examples, queryImage, "This is:", maxLength: 50);

        var result = new Dictionary<string, T>();
        double totalScore = 0;

        foreach (var label in labels)
        {
            double score = generated.ToLowerInvariant().Contains(label.ToLowerInvariant()) ? 1.0 : 0.1;
            totalScore += score;
            result[label] = NumOps.FromDouble(score);
        }

        foreach (var label in labels)
        {
            result[label] = NumOps.Divide(result[label], NumOps.FromDouble(totalScore));
        }

        return result;
    }

    /// <inheritdoc/>
    public string FewShotVQA(
        IEnumerable<(Tensor<T> Image, string Question, string Answer)> examples,
        Tensor<T> queryImage,
        string question)
    {
        var examplesList = examples.ToList();
        var fewShotExamples = examplesList.Select(e =>
            (e.Image, $"Question: {e.Question}\nAnswer: {e.Answer}")).ToList();

        return FewShotGenerate(
            fewShotExamples,
            queryImage,
            $"Question: {question}\nAnswer:",
            maxLength: 128);
    }

    /// <inheritdoc/>
    public Tensor<T> ExtractPerceiverFeatures(Tensor<T> image)
    {
        var visionFeatures = ExtractVisionFeatures(image);
        return ExtractPerceiverFeaturesNative(visionFeatures);
    }

    private Tensor<T> ExtractVisionFeatures(Tensor<T> image) => ExtractVisionFeaturesNative(image);

    private Tensor<T> ExtractVisionFeaturesNative(Tensor<T> image)
    {
        if (_patchEmbedding is null || _visionPositionalEmbeddings is null)
        {
            throw new InvalidOperationException("Vision layers not initialized.");
        }

        var patchFeatures = _patchEmbedding.Forward(image);
        int numPatches = patchFeatures.Shape[0];
        int hiddenDim = patchFeatures.Shape.Length > 1 ? patchFeatures.Shape[1] : _visionHiddenDim;

        // patch + position over the block both cover, zero elsewhere, as fresh values off the tape
        var features = Tensor<T>.CreateDefault([numPatches, hiddenDim], NumOps.Zero);
        int blockRows = Math.Min(numPatches, _visionPositionalEmbeddings.Shape[0]);
        int blockCols = Math.Min(hiddenDim, _visionPositionalEmbeddings.Shape[1]);
        if (blockRows > 0 && blockCols > 0)
        {
            T[] block;
            using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
            {
                block = Engine.TensorAdd(
                    Engine.TensorSlice(patchFeatures, [0, 0], [blockRows, blockCols]),
                    Engine.TensorSlice(_visionPositionalEmbeddings, [0, 0], [blockRows, blockCols])).ToArray();
            }

            var featureSpan = features.AsWritableSpan();
            for (int i = 0; i < blockRows; i++)
            {
                block.AsSpan(i * blockCols, blockCols).CopyTo(featureSpan.Slice(i * hiddenDim, blockCols));
            }
        }

        var current = features;
        foreach (var layer in _visionEncoderLayers)
        {
            current = layer.Forward(current);
        }

        return current;
    }

    private Tensor<T> ExtractPerceiverFeaturesNative(Tensor<T> visionFeatures)
    {
        if (_perceiverQueries is null)
        {
            throw new InvalidOperationException("Perceiver queries not initialized.");
        }

        // A fresh copy of the leading [tokens, hidden] block of the learned queries
        var current = Tensor<T>.CreateDefault([_numPerceiverTokens, _lmHiddenDim], NumOps.Zero);
        if (_numPerceiverTokens > 0 && _lmHiddenDim > 0)
        {
            using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
            {
                Engine.TensorSlice(_perceiverQueries, [0, 0], [_numPerceiverTokens, _lmHiddenDim])
                    .CopyTo(current.AsWritableSpan());
            }
        }

        for (int i = 0; i < _perceiverLayers.Count; i += 3)
        {
            if (_perceiverLayers[i] is CrossAttentionLayer<T> crossAttn)
            {
                var attnOut = crossAttn.Forward(current, visionFeatures);
                current = AddTensors(current, attnOut);
            }

            if (i + 1 < _perceiverLayers.Count && i + 2 < _perceiverLayers.Count)
            {
                var ffn1 = _perceiverLayers[i + 1].Forward(current);
                var ffn2 = _perceiverLayers[i + 2].Forward(ffn1);
                current = AddTensors(current, ffn2);
            }
        }

        return current;
    }

    /// <inheritdoc/>
    public string DescribeVideo(
        IEnumerable<Tensor<T>> frames,
        string? prompt = null,
        int maxLength = 256)
    {
        var framesList = frames.ToList();
        var sampledFrames = new List<Tensor<T>>();
        int step = Math.Max(1, framesList.Count / _maxImagesInContext);
        for (int i = 0; i < framesList.Count && sampledFrames.Count < _maxImagesInContext; i += step)
        {
            sampledFrames.Add(framesList[i]);
        }

        string videoPrompt = prompt ?? "Describe what is happening in this video:";
        return GenerateWithMultipleImages(sampledFrames, videoPrompt, maxLength);
    }

    /// <inheritdoc/>
    public T ScoreImageText(Tensor<T> image, string text)
    {
        var imageEmbedding = GetImageEmbedding(image);
        var textEmbedding = GetTextEmbedding(text);
        var similarity = ComputeSimilarity(imageEmbedding, textEmbedding);

        double logProb = Math.Log(Math.Max((NumOps.ToDouble(similarity) + 1) / 2, 1e-10));
        var encoded = _tokenizer.Encode(text);
        return NumOps.FromDouble(logProb * encoded.TokenIds.Count);
    }

    /// <inheritdoc/>
    public IEnumerable<(int Index, T Score)> FewShotImageRetrieval(
        IEnumerable<Tensor<T>> queryExamples,
        string? queryDescription,
        IEnumerable<Tensor<T>> candidateImages,
        int topK = 10)
    {
        var queryList = queryExamples.ToList();
        var candidateList = candidateImages.ToList();

        var queryEmbeddings = queryList.Select(img => GetImageEmbedding(img)).ToList();
        var avgQueryEmbedding = new Vector<T>(_embeddingDimension);

        if (_embeddingDimension > 0)
        {
            // Each query adds over its common prefix, in order; the sums are divided by the query count
            var sum = new Tensor<T>([_embeddingDimension]);
            foreach (var emb in queryEmbeddings)
            {
                sum = AddToPrefix(sum, emb);
            }
            Engine.TensorDivideScalar(sum, NumOps.FromDouble(queryEmbeddings.Count)).CopyTo(avgQueryEmbedding.AsWritableSpan());
        }

        if (queryDescription is not null && queryDescription.Length > 0)
        {
            var textEmbedding = GetTextEmbedding(queryDescription);
            int common = Math.Min(_embeddingDimension, textEmbedding.Length);
            if (common > 0)
            {
                // (average + text) / 2 over the common prefix
                var blended = Engine.TensorDivideScalar(
                    Engine.TensorAdd(
                        Tensor<T>.FromVector(avgQueryEmbedding.GetSubVector(0, common)),
                        Tensor<T>.FromVector(textEmbedding.GetSubVector(0, common))),
                    NumOps.FromDouble(2.0));
                blended.CopyTo(avgQueryEmbedding.AsWritableSpan().Slice(0, common));
            }
        }

        avgQueryEmbedding = Normalize(avgQueryEmbedding);

        var scores = new List<(int Index, T Score)>();
        for (int i = 0; i < candidateList.Count; i++)
        {
            var candidateEmbedding = GetImageEmbedding(candidateList[i]);
            var similarity = ComputeSimilarity(avgQueryEmbedding, candidateEmbedding);
            scores.Add((i, similarity));
        }

        return scores.OrderByDescending(s => NumOps.ToDouble(s.Score)).Take(topK);
    }

    #endregion

    #region Helper Methods

    private string GenerateWithVisualContext(
        IReadOnlyList<int> inputIds,
        List<Tensor<T>> imageFeatures,
        int maxLength)
    {
        var generatedIds = new List<int>(inputIds);
        var specialTokens = _tokenizer.SpecialTokens;
        var eosTokenStr = specialTokens?.EosToken ?? "[SEP]";
        var eosEncoded = _tokenizer.Encode(eosTokenStr);
        int eosTokenId = eosEncoded.TokenIds.Count > 0 ? eosEncoded.TokenIds[0] : 0;

        var combinedImageFeatures = CombineImageFeatures(imageFeatures);

        if (_outputProjection is null)
        {
            // Fail fast: without the output projection there are no logits to decode, so
            // returning the (empty) suffix here would masquerade an unsupported state as a
            // successful empty generation.
            throw new InvalidOperationException(
                "Output projection must be initialized before Flamingo text generation can produce logits.");
        }

        // The shared AutoregressiveDecoder owns the loop + greedy argmax + EOS stop. The per-step
        // forward (text embed -> gated cross-attention every 4th LM layer -> output projection ->
        // last-position logits) is supplied via the closure. Flamingo decodes greedily, matching the
        // previous hand-rolled argmax (SampleFromLogits).
        var newTokens = Generation.AutoregressiveDecoder<T>.Decode(
            stepLogits: prev =>
            {
                if (prev.HasValue) generatedIds.Add(prev.Value);
                // Slide the context to the most recent _maxSequenceLength tokens — Take(seqLen)
                // would keep the FIRST tokens and silently drop every newly-generated token once
                // the sequence exceeds the window, so later forward passes never see the latest token.
                var contextIds = generatedIds
                    .Skip(Math.Max(0, generatedIds.Count - _maxSequenceLength))
                    .ToList();
                if (contextIds.Count == 0)
                {
                    throw new InvalidOperationException("Generation requires at least one prompt token.");
                }
                int seqLen = contextIds.Count;
                var embeddings = EmbedTextTokens(contextIds);

                var current = embeddings;
                int gatedAttnIdx = 0;
                for (int layer = 0; layer < _numLmLayers && layer < _languageModelLayers.Count; layer++)
                {
                    if (layer % 4 == 0 && gatedAttnIdx < _gatedCrossAttentionLayers.Count)
                    {
                        if (_gatedCrossAttentionLayers[gatedAttnIdx] is CrossAttentionLayer<T> crossAttn)
                        {
                            var attnOut = crossAttn.Forward(current, combinedImageFeatures);
                            current = AddTensors(current, attnOut);
                        }
                        gatedAttnIdx++;
                    }

                    current = _languageModelLayers[layer].Forward(current);
                }

                // A fresh copy of the last position's first _lmHiddenDim features
                int currentColumns = current.Shape[1];
                var lastPosition = new Tensor<T>(
                    current.ToArray().AsSpan((seqLen - 1) * currentColumns, _lmHiddenDim).ToArray(), [1, _lmHiddenDim]);

                var logits = _outputProjection.Forward(lastPosition);
                int vocab = logits.Shape[1];
                var v = new Vector<T>(vocab);
                logits.ToArray().AsSpan(0, vocab).CopyTo(v.AsWritableSpan());
                return v;
            },
            maxNewTokens: maxLength,
            options: Generation.SamplingOptions.Greedy,
            isEndToken: t => t == eosTokenId);

        return _tokenizer.Decode(new List<int>(newTokens));
    }

    private Tensor<T> CombineImageFeatures(List<Tensor<T>> imageFeatures)
    {
        if (imageFeatures.Count == 0)
        {
            return Tensor<T>.CreateDefault([1, _lmHiddenDim], NumOps.Zero);
        }

        int totalTokens = imageFeatures.Sum(f => f.Shape[0]);
        int hiddenDim = imageFeatures[0].Shape[1];

        var combined = Tensor<T>.CreateDefault([totalTokens, hiddenDim], NumOps.Zero);
        var combinedSpan = combined.AsWritableSpan();
        int offset = 0;

        foreach (var features in imageFeatures)
        {
            // Each image's rows, first hiddenDim columns, stacked in order
            int rows = features.Shape[0];
            int columns = features.Shape[1];
            var values = features.ToArray();
            for (int i = 0; i < rows; i++)
            {
                values.AsSpan(i * columns, hiddenDim).CopyTo(combinedSpan.Slice((offset + i) * hiddenDim, hiddenDim));
            }
            offset += rows;
        }

        return combined;
    }

    /// <summary>
    /// Adds <paramref name="vector"/> to the leading entries of <paramref name="sum"/> they share; the
    /// rest of the sum keeps its exact value.
    /// </summary>
    private Tensor<T> AddToPrefix(Tensor<T> sum, Vector<T> vector)
    {
        int length = sum.Length;
        int common = Math.Min(length, vector.Length);
        if (common == 0) return sum;
        var part = Tensor<T>.FromVector(vector.Length == common ? vector : vector.GetSubVector(0, common));
        var added = Engine.TensorAdd(common == length ? sum : Engine.TensorSlice(sum, [0], [common]), part);
        return common == length
            ? added
            : Engine.TensorConcatenate(new[] { added, Engine.TensorSlice(sum, [common], [length - common]) }, 0);
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorAdd(a, b);
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {

        SetTrainingMode(false);
        return Accelerate(input, () =>
        {
            var features = ExtractPerceiverFeatures(input);
            return features;
        });
    }

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        // The base implementation walks Layers as one flat sequential graph. Flamingo's
        // layer list is structural storage for separate vision, perceiver, gated-attention,
        // and language branches, so that walk does not represent PredictCore. Training the
        // flat list therefore optimized a different output than Predict measured and could
        // increase prediction loss even after a successful optimizer step. Run the same
        // vision/perceiver graph used by PredictCore while the caller-owned tape is active.
        EnsureLayerRandomSeedsWired();
        return ExtractPerceiverFeatures(input);
    }

    /// <inheritdoc/>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));

        // Layers is structural storage for four separate towers, not a sequential network. The
        // base implementation fed continuous image pixels through the text-token embedding and
        // reported an index-domain error instead of any activations. Capture the real image path
        // used by PredictCore and expose both architecturally meaningful stages.
        var visionFeatures = ExtractVisionFeatures(input);
        var perceiverFeatures = ExtractPerceiverFeaturesNative(visionFeatures);

        return new Dictionary<string, Tensor<T>>
        {
            ["vision_features"] = visionFeatures.Clone(),
            ["perceiver_features"] = perceiverFeatures.Clone(),
        };
    }

    // The layer streams this model holds outside Layers are discovered by ModelParameterGenerator and surfaced automatically; the hand-written hook that used to sit here was an override wearing a different name.

    // UpdateParameters restated a fold the base now derives from generated component registration.
    // Removed under AIDN082.
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ImageSize", _imageSize },
                { "EmbeddingDimension", _embeddingDimension },
                { "MaxSequenceLength", _maxSequenceLength },
                { "NumPerceiverTokens", _numPerceiverTokens },
                { "MaxImagesInContext", _maxImagesInContext },
                { "VisionHiddenDim", _visionHiddenDim },
                { "LmHiddenDim", _lmHiddenDim },
                { "NumVisionLayers", _numVisionLayers },
                { "NumLmLayers", _numLmLayers },
                { "NumPerceiverLayers", _numPerceiverLayers },
                { "VocabularySize", _vocabularySize },
                { "LanguageModelBackbone", _languageModelBackbone.ToString() },
                { "UseNativeMode", true },
                { "ParameterCount", ParameterCount },
                { "TaskType", Architecture.TaskType.ToString() }
            },
            ModelDataProvider = () => SerializeForMetadata()
        };
    }

    #endregion

    #region IMultimodalEmbedding Interface (Standard API)

    /// <inheritdoc/>
    public Vector<T> EncodeText(string text)
    {
        return GetTextEmbedding(text);
    }

    /// <inheritdoc/>
    public Matrix<T> EncodeTextBatch(IEnumerable<string> texts)
    {
        var embeddings = GetTextEmbeddings(texts).ToList();
        if (embeddings.Count == 0)
        {
            return new Matrix<T>(0, EmbeddingDimension);
        }

        var matrix = new Matrix<T>(embeddings.Count, embeddings[0].Length);
        int width = embeddings[0].Length;
        var destination = matrix.AsWritableSpan();
        for (int i = 0; i < embeddings.Count; i++)
        {
            // Row i holds embedding i; a shorter embedding leaves the rest of its row zero.
            var source = embeddings[i].AsSpan();
            if (source.Length > width)
                throw new ArgumentOutOfRangeException(nameof(embeddings),
                    $"Embedding {i} has {source.Length} values but the batch width is {width}.");
            source.CopyTo(destination.Slice(i * width, width));
        }
        return matrix;
    }

    /// <inheritdoc/>
    public Vector<T> EncodeImage(double[] imageData)
    {
        // Convert double[] to Tensor<T> in CHW format
        var tensor = ConvertToTensor(imageData);
        return GetImageEmbedding(tensor);
    }

    /// <inheritdoc/>
    public Matrix<T> EncodeImageBatch(IEnumerable<double[]> imageDataBatch)
    {
        var tensors = imageDataBatch.Select(ConvertToTensor);
        var embeddings = GetImageEmbeddings(tensors).ToList();
        if (embeddings.Count == 0)
        {
            return new Matrix<T>(0, EmbeddingDimension);
        }

        var matrix = new Matrix<T>(embeddings.Count, embeddings[0].Length);
        int width = embeddings[0].Length;
        var destination = matrix.AsWritableSpan();
        for (int i = 0; i < embeddings.Count; i++)
        {
            // Row i holds embedding i; a shorter embedding leaves the rest of its row zero.
            var source = embeddings[i].AsSpan();
            if (source.Length > width)
                throw new ArgumentOutOfRangeException(nameof(embeddings),
                    $"Embedding {i} has {source.Length} values but the batch width is {width}.");
            source.CopyTo(destination.Slice(i * width, width));
        }
        return matrix;
    }

    /// <inheritdoc/>
    public Dictionary<string, T> ZeroShotClassify(double[] imageData, IEnumerable<string> labels)
    {
        var tensor = ConvertToTensor(imageData);
        return ZeroShotClassify(tensor, labels);
    }

    /// <summary>
    /// Converts a double[] image to Tensor format.
    /// </summary>
    private Tensor<T> ConvertToTensor(double[] imageData)
    {
        if (imageData == null || imageData.Length == 0)
        {
            throw new ArgumentException("Image data cannot be null or empty.", nameof(imageData));
        }

        int channels = _channels;
        if (imageData.Length % channels != 0)
        {
            throw new ArgumentException($"Image data length ({imageData.Length}) must be divisible by {channels} channels.", nameof(imageData));
        }

        int pixels = imageData.Length / channels;
        int size = (int)Math.Sqrt(pixels);
        if (size * size != pixels)
        {
            throw new ArgumentException($"Image must be square. Got {pixels} pixels which is not a perfect square.", nameof(imageData));
        }

        var tensor = new Tensor<T>(new[] { channels, size, size });
        int idx = 0;
        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < size; h++)
            {
                for (int w = 0; w < size; w++)
                {
                    tensor[c, h, w] = NumOps.FromDouble(imageData[idx++]);
                }
            }
        }
        return tensor;
    }

    #endregion

}
