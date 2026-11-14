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
/// 3DGraphLLM: 3D scene graph as LLM input for spatial reasoning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// 3DGraphLLM (CogAI, 2025) uses 3D scene graphs as structured input for large language models.
/// It constructs semantic graphs from 3D scenes where nodes represent objects with their spatial
/// properties and edges encode spatial relationships, providing the LLM with a structured
/// representation for precise spatial reasoning, object counting, and relationship understanding.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "3DGraphLLM: 3D Scene Graph as Input for Large Language Models (CogAI, 2025)"</item></list></para>
/// <para><b>For Beginners:</b> 3DGraphLLM is a vision-language model that uses scene graphs
/// for structured 3D spatial reasoning. Default values follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a 3DGraphLLM model for 3D scene graph reasoning
/// // using semantic graphs as structured input for spatial understanding
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new ThreeDGraphLLM&lt;double&gt;(architecture, "threedgraphllm.onnx");
///
/// // Training mode with native layers
/// var trainModel = new ThreeDGraphLLM&lt;double&gt;(architecture, new ThreeDGraphLLMOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.ThreeD)]
[ModelDomain(ModelDomain.GraphAnalysis)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "3DGraphLLM: Combining Semantic Graphs and Large Language Models for 3D Scene Understanding",
    "https://arxiv.org/abs/2412.18450",
    Year = 2025,
    Authors = "Rygalev et al."
)]
[PaperOptimizer(OptimizerKind.Unspecified, LearningRate = 5e-6, ReferenceBatchSize = 8,
                Schedule = LearningRateSchedulerType.CosineAnnealing,
                Source = "Zemskova and Yudin 2024, Sec. 4: a batch size of 8 trained for 3 epochs at an "
                        + "initial learning rate of 5e-6 following a cosine annealing schedule. The "
                        + "optimizer is left unspecified because the paper names none.")]
public partial class ThreeDGraphLLM<T> : VisionLanguageModelBase<T>, IThreeDVisionLanguageModel<T>
{
    private readonly ThreeDGraphLLMOptions _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public ThreeDGraphLLM(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        ThreeDGraphLLMOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new ThreeDGraphLLMOptions();
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

    public ThreeDGraphLLM(
        NeuralNetworkArchitecture<T> architecture,
        ThreeDGraphLLMOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new ThreeDGraphLLMOptions();
        Options = _options;
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
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
    /// Generates from 2D image using 3DGraphLLM's scene graph approach.
    /// Per the paper (2024), the system constructs a 3D scene graph from
    /// visual input. For a single 2D image, the vision encoder extracts
    /// spatial features which are fused with text instruction tokens via
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
    /// Processes 3D point cloud using 3DGraphLLM's scene graph approach. Per the
    /// paper (2024), the system constructs a 3D scene graph from point clouds:
    /// (1) segment the scene into objects (nodes), (2) compute spatial relationships
    /// between objects (edges: left-of, above, near, etc.), (3) encode the graph
    /// using a graph neural network with message passing between connected nodes,
    /// (4) serialize the graph tokens for the LLM. This graph-structured approach
    /// enables explicit spatial reasoning that free-form point features lack.
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
        int maxNodes = _options.MaxGraphNodes;
        int encoderDim = _options.PointEncoderDim;

        var engine = Engine;
        if (maxNodes < 0)
            throw new OverflowException("The maximum number of graph nodes cannot be negative.");
        // Points p < numPoints with p * channels + 2 < totalValues are read (the scan stops at the
        // first point that is not).
        int readPoints = ReadablePointCount(numPoints, channels, totalValues);

        Tensor<T> graphTokens;
        if (readPoints == 0)
        {
            graphTokens = new Tensor<T>([encoderDim]);
        }
        else
        {
            // Step 1: Build scene graph nodes via spatial clustering:
            // node = |(int)(3x) * 7919 ^ (int)(3y) * 7907 ^ (int)(3z) * 7901| % maxNodes.
            var coords = PointCoordinates(pointCloud, readPoints, channels);
            var nodeIds = HashPointCells(coords, 3.0, 7919, 7907, 7901, maxNodes);
            var pointFeatures = FoldedPointChannels(pointCloud, readPoints, channels, encoderDim);

            if (encoderDim == 0)
            {
                graphTokens = new Tensor<T>([0]);
            }
            else
            {
                var nodeSums = engine.ScatterAdd(
                    pointFeatures,
                    engine.TensorBroadcastTo(nodeIds.Reshape([readPoints, 1]), [readPoints, encoderDim]),
                    0,
                    maxNodes);

                // Normalize nodes
                var nodeCounts = engine.TensorBinCount(nodeIds, maxNodes).Cast<T>().Reshape([maxNodes, 1]);
                var safeCounts = engine.TensorMax(nodeCounts, NumOps.One);
                var active = engine.TensorGreaterThan(nodeCounts, NumOps.Zero);
                var nodePositions = engine.TensorDivide(
                    engine.ScatterAdd(coords, engine.TensorBroadcastTo(nodeIds.Reshape([readPoints, 1]), [readPoints, 3]), 0, maxNodes),
                    engine.TensorBroadcastTo(safeCounts, [maxNodes, 3]));
                var nodeFeatures = engine.TensorDivide(nodeSums, engine.TensorBroadcastTo(safeCounts, [maxNodes, encoderDim]));

                // Step 2: Build graph edges with spatial relationship labels.
                // Active nodes i != j within distance 3 are connected (dx = pos_j - pos_i).
                var nodeIndex = engine.TensorArange<T>(0, maxNodes, 1);
                var isSelf = engine.TensorEquals(
                    engine.TensorBroadcastTo(nodeIndex.Reshape([maxNodes, 1]), [maxNodes, maxNodes]),
                    engine.TensorBroadcastTo(nodeIndex.Reshape([1, maxNodes]), [maxNodes, maxNodes]));
                Tensor<T> Delta(int axis)
                {
                    var column = engine.TensorSlice(nodePositions, [0, axis], [maxNodes, 1]);
                    return engine.TensorSubtract(
                        engine.TensorBroadcastTo(column.Reshape([1, maxNodes]), [maxNodes, maxNodes]),
                        engine.TensorBroadcastTo(column, [maxNodes, maxNodes]));
                }
                var dx = Delta(0);
                var dy = Delta(1);
                var dz = Delta(2);
                var dist = engine.TensorSqrt(engine.TensorAdd(
                    engine.TensorAdd(engine.TensorMultiply(dx, dx), engine.TensorMultiply(dy, dy)),
                    engine.TensorMultiply(dz, dz)));
                // Only connect nearby nodes (spatial neighborhood); a NaN distance is not "> 3" and stays connected.
                var edgeMask = engine.TensorMultiply(
                    engine.TensorMultiply(
                        engine.TensorMultiply(active, active.Reshape([1, maxNodes])),
                        engine.ScalarMinusTensor(NumOps.One, isSelf)),
                    engine.ScalarMinusTensor(NumOps.One, engine.TensorGreaterThan(dist, NumOps.FromDouble(3.0))));
                // Edge features encode spatial relationship type
                var edgeWeight = engine.TensorExp(engine.TensorNegate(dist));
                // Directional encoding (above/below, left/right, front/back)
                var paddedDist = engine.TensorAddScalar(dist, NumOps.FromDouble(1e-8));
                var verticalRel = engine.TensorDivide(dy, paddedDist); // +1 = above, -1 = below
                var horizontalRel = engine.TensorDivide(dx, paddedDist);

                // Relationship bias sin(vertical (d + 1)) 0.05 + cos(horizontal (d + 1)) 0.05 summed over each
                // node's edges (the same every round).
                var frequency = engine.TensorArange<T>(1, encoderDim + 1, 1).Reshape([1, encoderDim]);
                T biasScale = NumOps.FromDouble(0.05);
                var noTerms = new Tensor<T>([maxNodes, encoderDim]);
                var relBiasRows = new Tensor<T>[maxNodes];
                Tensor<T> EdgeRow(int node) => engine.TensorBroadcastTo(
                    engine.TensorSlice(edgeMask, [node, 0], [1, maxNodes]).Reshape([maxNodes, 1]), [maxNodes, encoderDim]);
                for (int i = 0; i < maxNodes; i++)
                {
                    var vertical = engine.TensorSlice(verticalRel, [i, 0], [1, maxNodes]).Reshape([maxNodes, 1]);
                    var horizontal = engine.TensorSlice(horizontalRel, [i, 0], [1, maxNodes]).Reshape([maxNodes, 1]);
                    var relBias = engine.TensorAdd(
                        engine.TensorMultiplyScalar(engine.TensorSin(engine.TensorMultiply(vertical, frequency)), biasScale),
                        engine.TensorMultiplyScalar(engine.TensorCos(engine.TensorMultiply(horizontal, frequency)), biasScale));
                    relBiasRows[i] = engine.ReduceSum(engine.TensorWhere(EdgeRow(i), relBias, noTerms), [0], false);
                }

                // Message passing: each node aggregates its own feature plus edgeWeight * neighbor features,
                // then is L2-normalized (norm + 1e-8); inactive nodes are 0.
                int numRounds = 3; // GNN message passing rounds
                T epsilon = NumOps.FromDouble(1e-8);
                var inactiveRow = new Tensor<T>([encoderDim]);
                for (int round = 0; round < numRounds; round++)
                {
                    var newRows = new Tensor<T>[maxNodes];
                    for (int i = 0; i < maxNodes; i++)
                    {
                        // Masked per-pair products (not a matmul): a non-finite feature of an unconnected
                        // node must not leak into node i as 0 * inf = NaN.
                        var edgeWeightRow = engine.TensorBroadcastTo(
                            engine.TensorSlice(edgeWeight, [i, 0], [1, maxNodes]).Reshape([maxNodes, 1]), [maxNodes, encoderDim]);
                        var messages = engine.TensorWhere(
                            EdgeRow(i), engine.TensorMultiply(nodeFeatures, edgeWeightRow), noTerms);
                        var aggregated = engine.TensorAdd(
                            engine.TensorAdd(
                                engine.TensorSlice(nodeFeatures, [i, 0], [1, encoderDim]).Reshape([encoderDim]),
                                engine.ReduceSum(messages, [0], false)),
                            relBiasRows[i]);
                        // Normalize aggregated features
                        var norm = engine.TensorAddScalar(
                            engine.TensorSqrt(engine.ReduceSum(engine.TensorMultiply(aggregated, aggregated), [0], true)), epsilon);
                        newRows[i] = engine.TensorWhere(
                            engine.TensorBroadcastTo(engine.TensorSlice(active, [i, 0], [1, 1]).Reshape([1]), [encoderDim]),
                            engine.TensorDivide(aggregated, engine.TensorBroadcastTo(norm, [encoderDim])),
                            inactiveRow).Reshape([1, encoderDim]);
                    }
                    nodeFeatures = engine.TensorConcatenate(newRows, 0);
                }

                // Step 3: Serialize graph tokens for LLM: the node-count weighted mean of the active nodes
                // (the total weight is the number of read points).
                var countsWide = engine.TensorBroadcastTo(nodeCounts, [maxNodes, encoderDim]);
                var weighted = engine.TensorWhere(
                    engine.TensorBroadcastTo(active, [maxNodes, encoderDim]),
                    engine.TensorMultiply(nodeFeatures, countsWide),
                    noTerms);
                graphTokens = engine.TensorDivideScalar(engine.ReduceSum(weighted, [0], false), NumOps.FromDouble(readPoints));
            }
        }

        // Step 4: Encode and decode
        var encoded = graphTokens;
        for (int i = 0; i < _encoderLayerEnd && i < Layers.Count; i++)
            encoded = Layers[i].Forward(encoded);

        encoded = PromptGated(encoded, TokenizeText(prompt));

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
                    // Was hard-coded 512; drive the encoder's vision-token dim from the configured
                    // VisionDim so the encoder, the declared option, and the post-embedding token
                    // input all agree (mirrors the LEOVL fix; see AiDotNet.Tensors#775 context).
                    _options.VisionDim,
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
            Name = _useNativeMode ? "3DGraphLLM-Native" : "3DGraphLLM-ONNX",
            Description = "3DGraphLLM: 3D scene graph as LLM input for spatial reasoning.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "3DGraphLLM";
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(ThreeDGraphLLM<T>));
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
