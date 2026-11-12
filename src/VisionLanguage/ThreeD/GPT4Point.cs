using AiDotNet.LearningRateSchedulers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.ThreeD;

/// <summary>
/// GPT4Point: unified point-language understanding and generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// GPT4Point (2024) is a unified framework for point cloud-language understanding and generation.
/// It bridges 3D point cloud representations with language models through a point cloud encoder
/// that projects geometric features into the LLM's embedding space, enabling both point cloud
/// captioning and text-conditioned 3D point cloud generation in a single model.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "GPT4Point: A Unified Framework for Point-Language Understanding and Generation (Various, 2024)"</item></list></para>
/// <para><b>For Beginners:</b> GPT4Point is a vision-language model for understanding and
/// generating 3D point clouds using natural language. Default values follow the original
/// paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a GPT4Point model for unified point cloud-language understanding
/// // and 3D point cloud generation from text descriptions
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new GPT4Point&lt;double&gt;(architecture, "gpt4point.onnx");
///
/// // Training mode with native layers
/// var trainModel = new GPT4Point&lt;double&gt;(architecture, new GPT4PointOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.ThreeD)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "GPT4Point: A Unified Framework for Point-Language Understanding and Generation",
    "https://arxiv.org/abs/2312.02980",
    Year = 2024,
    Authors = "Qi et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, WeightDecay = 0.05,
                ReferenceBatchSize = 32,
                Source = "Qi et al. 2024, Sec. 4: an initial learning rate of 1e-4, a weight decay of "
                        + "0.05, a batch size of 32 and the AdamW optimizer.")]
public partial class GPT4Point<T> : VisionLanguageModelBase<T>, IThreeDVisionLanguageModel<T>
{
    private readonly GPT4PointOptions _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public GPT4Point(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        GPT4PointOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new GPT4PointOptions();
        Options = _options;
        _useNativeMode = false;
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    public GPT4Point(
        NeuralNetworkArchitecture<T> architecture,
        GPT4PointOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new GPT4PointOptions();
        Options = _options;
        _useNativeMode = true;
        _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    /// <inheritdoc />
    public int EmbeddingDimension => _options.DecoderDim;
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;
    int IVisualEncoder<T>.ImageChannels => 3;
    /// <inheritdoc />
    public int MaxGenerationLength => _options.MaxGenerationLength;
    /// <inheritdoc />
    public int DecoderEmbeddingDim => _options.DecoderDim;
    /// <inheritdoc />
    public int MaxPoints => _options.MaxPoints;
    /// <inheritdoc />
    public int PointChannels => _options.PointChannels;

    /// <inheritdoc />
    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return L2Normalize(OnnxModel.Run(p));
        var c = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            c = Layers[i].Forward(c);
        return L2Normalize(c);
    }

    /// <summary>
    /// Generates from 2D image using GPT4Point's Point-QFormer alignment.
    /// Per the paper (Qi et al., 2024), GPT4Point bridges visual features with
    /// LLM tokens via a Q-Former architecture. For a single 2D image, the vision
    /// encoder extracts features which are fused with text instruction tokens via
    /// cross-attention before the LLM decoder generates the response.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);
        var encoderOut = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            encoderOut = Layers[i].Forward(encoderOut);

        // Fuse visual features with prompt tokens via ConcatenateTensors
        Tensor<T> fusedInput;
        if (prompt is not null)
        {
            var promptTokens = TokenizeText(prompt);
            fusedInput = encoderOut.ConcatenateTensors(promptTokens);
        }
        else
        {
            fusedInput = encoderOut;
        }

        var output = fusedInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);
        return output;
    }

    /// <summary>
    /// Processes 3D point cloud using GPT4Point's Point-QFormer alignment approach.
    /// Per the paper (Qi et al., 2024), GPT4Point uses a Point-QFormer architecture
    /// (analogous to BLIP-2's Q-Former) to bridge point cloud features with LLM
    /// tokens. Learned query tokens attend to point cloud features via cross-attention,
    /// producing a fixed-size set of point tokens aligned with the LLM embedding
    /// space. The model also supports point cloud generation (when
    /// SupportsPointGeneration is true) via a controlled generation module.
    /// </summary>
    public Tensor<T> GenerateFrom3D(Tensor<T> pointCloud, string prompt)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(pointCloud);

        int totalValues = pointCloud.Length;
        int channels = _options.PointChannels;
        int numPoints = Math.Min(totalValues / Math.Max(1, channels), _options.MaxPoints);
        if (numPoints == 0)
            numPoints = Math.Min(totalValues, _options.MaxPoints);
        int encoderDim = _options.PointEncoderDim;

        // Step 1: Point cloud feature extraction (Point-BERT backbone)
        // Downsample to patch centers and extract local features
        int numPatches = Math.Min(256, numPoints / 4);
        if (numPatches < 1)
            numPatches = 1;

        var engine = Engine;
        int pointsPerPatch = Math.Max(1, numPoints / numPatches);
        int pointChannels = Math.Max(0, channels);
        int patchValues = pointsPerPatch * pointChannels;
        // Patch p max-pools points [p * pointsPerPatch, (p + 1) * pointsPerPatch) below numPoints,
        // value (pt, c) into feature (pt * channels + c) % encoderDim.
        var patchFeatures = PatchMaxPool(
            pointCloud, numPatches, patchValues, patchValues, Math.Min(numPoints * pointChannels, totalValues), encoderDim);

        // Step 2: Point-QFormer with learned queries
        int numQueries = 32; // Fixed number of query tokens
        Tensor<T> pointTokens;
        if (encoderDim == 0)
        {
            pointTokens = new Tensor<T>([0]);
        }
        else
        {
            // Cross-attention: each query attends to all patch features. The query-key score of query q
            // and patch p is the mean of the patch's features [q * stride, min(q * stride + stride, dim))
            // (0 for an empty block) divided by sqrt(dim).
            int stride = Math.Max(1, encoderDim / numQueries);
            var blockPositions = engine.TensorAdd(
                engine.TensorBroadcastTo(
                    engine.TensorMultiplyScalar(engine.TensorArange<int>(0, numQueries, 1), stride).Reshape([numQueries, 1]),
                    [numQueries, stride]),
                engine.TensorBroadcastTo(engine.TensorArange<int>(0, stride, 1).Reshape([1, stride]), [numQueries, stride]));
            var inBlock = engine.TensorLessThan(blockPositions, encoderDim).Cast<T>();
            var blockValues = engine.TensorIndexSelect(
                patchFeatures, engine.TensorMin(blockPositions, encoderDim - 1).Reshape([numQueries * stride]), 1)
                .Reshape([numPatches, numQueries, stride]);
            blockValues = engine.TensorWhere(
                engine.TensorBroadcastTo(inBlock.Reshape([1, numQueries, stride]), [numPatches, numQueries, stride]),
                blockValues,
                new Tensor<T>([numPatches, numQueries, stride]));
            var blockCount = engine.TensorMax(engine.ReduceSum(inBlock, [1], false), NumOps.One);
            var score = engine.TensorDivideScalar(
                engine.TensorDivide(
                    engine.ReduceSum(blockValues, [2], false),
                    engine.TensorBroadcastTo(blockCount.Reshape([1, numQueries]), [numPatches, numQueries])),
                NumOps.FromDouble(Math.Sqrt(encoderDim)));
            var attnWeights = engine.Reshape(engine.TensorPermute(engine.TensorExp(score), [1, 0]), [numQueries, numPatches]);
            var attnSum = engine.TensorBroadcastTo(engine.ReduceSum(attnWeights, [1], true), [numQueries, numPatches]);

            // Aggregate values weighted by attention (uniform weights when the attention mass is not above 1e-8)
            var weights = engine.TensorWhere(
                engine.TensorGreaterThan(attnSum, NumOps.FromDouble(1e-8)),
                engine.TensorDivide(attnWeights, attnSum),
                Tensor<T>.CreateDefault([numQueries, numPatches], NumOps.FromDouble(1.0 / numPatches)));
            var queryOutputs = engine.TensorMatMul(weights, patchFeatures);

            // Step 3: Self-attention among query tokens for inter-query reasoning. Queries update in
            // order, each attending over the already-updated earlier queries.
            var queryRows = new List<Tensor<T>>(numQueries);
            for (int q = 0; q < numQueries; q++)
                queryRows.Add(engine.TensorSlice(queryOutputs, [q, 0], [1, encoderDim]).Reshape([encoderDim]));
            T sqrtDim = NumOps.FromDouble(Math.Sqrt(encoderDim));
            T epsilon = NumOps.FromDouble(1e-8);
            for (int q = 0; q < numQueries; q++)
            {
                var queryMatrix = StackLeading(queryRows, numQueries, encoderDim);
                var selfAttn = engine.TensorExp(engine.TensorDivideScalar(
                    engine.TensorMatMul(queryMatrix, queryRows[q].Reshape([encoderDim, 1])), sqrtDim));
                var selfAttnSum = engine.TensorBroadcastTo(engine.ReduceSum(selfAttn, [0], false), [encoderDim]);
                var updated = engine.TensorMatMul(selfAttn.Reshape([1, numQueries]), queryMatrix).Reshape([encoderDim]);
                queryRows[q] = engine.TensorWhere(
                    engine.TensorGreaterThan(selfAttnSum, epsilon),
                    engine.TensorDivide(updated, selfAttnSum),
                    queryRows[q]);
            }

            // Step 4: Pool queries into LLM-aligned point tokens
            pointTokens = engine.ReduceSum(
                engine.TensorDivideScalar(StackLeading(queryRows, numQueries, encoderDim), NumOps.FromDouble(numQueries)),
                [0], false);
        }

        // Step 5: Encode and fuse with prompt
        var encoded = pointTokens;
        for (int i = 0; i < _encoderLayerEnd && i < Layers.Count; i++)
            encoded = Layers[i].Forward(encoded);

        // Prompt gate 0.5 + sigmoid(prompt[d % promptLen] / 100) on every encoded value.
        var promptTokens = TokenizeText(prompt);
        int promptLen = promptTokens.Length;
        if (promptLen > 0)
        {
            var gate = engine.TensorAddScalar(
                engine.TensorSigmoid(engine.TensorDivideScalar(CyclicTake(promptTokens, encoded.Length), NumOps.FromDouble(100.0))),
                NumOps.FromDouble(0.5));
            encoded = engine.TensorMultiply(encoded, gate.Reshape(encoded.Shape.ToArray()));
        }

        var output = encoded;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _encoderLayerEnd = Layers.Count / 2;
        }
        else
        {
            Layers.AddRange(
                LayerHelper<T>.CreateDefaultPointCloudVLMLayers(
                    512,
                    _options.DecoderDim,
                    _options.NumVisionLayers,
                    _options.NumDecoderLayers,
                    _options.NumHeads,
                    _options.DropoutRate
                )
            );
            ComputeEncoderDecoderBoundary();
        }
    }

    private void ComputeEncoderDecoderBoundary()
    {
        int lpb = _options.DropoutRate > 0 ? 6 : 5;
        _encoderLayerEnd = _options.NumVisionLayers * lpb + 4;
    }

    private Tensor<T> TokenizeText(string text)
    {
        if (_tokenizer is null)
            throw new InvalidOperationException("Tokenizer not initialized.");
        var encoding = _tokenizer.Encode(text);
        int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength);
        return TokenIdsToTensor(encoding.TokenIds, seqLen);
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        var c = input;
        foreach (var l in Layers)
            c = l.Forward(c);
        return c;
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <inheritdoc />
    protected override Tensor<T> PreprocessImage(Tensor<T> image) =>
        NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "GPT4Point-Native" : "GPT4Point-ONNX",
            Description = "GPT4Point: unified point-language understanding and generation.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "GPT4Point";
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(GPT4Point<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
