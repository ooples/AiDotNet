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
/// LEO-VL: efficient 3D scene representation from multi-view RGB-D.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// LEO-VL (2025) provides efficient 3D scene understanding from multi-view RGB-D inputs.
/// It constructs 3D scene representations from multiple depth-camera views without requiring
/// explicit 3D reconstruction, using a view aggregation module to fuse multi-view features
/// into a coherent spatial representation for language-guided scene understanding and navigation.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "LEO-VL: Efficient 3D Scene Understanding via Multi-View RGB-D (Various, 2025)"</item></list></para>
/// <para><b>For Beginners:</b> LEO-VL is a vision-language model for 3D scene understanding
/// using multi-view RGB-D camera inputs. Default values follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a LEO-VL model for 3D scene understanding
/// // with efficient multi-view RGB-D representation learning
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new LEOVL&lt;double&gt;(architecture, "leovl.onnx");
///
/// // Training mode with native layers
/// var trainModel = new LEOVL&lt;double&gt;(architecture, new LEOVLOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.ThreeD)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "LEO: An Embodied Generalist Agent in 3D World",
    "https://arxiv.org/abs/2311.12871",
    Year = 2024,
    Authors = "Huang et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 3e-4, Beta1 = 0.9, Beta2 = 0.999,
                WeightDecay = 0.05, ReferenceBatchSize = 80, WarmupSteps = 400,
                MaxGradientNorm = 5.0, GradientAccumulationSteps = 4,
                Phase = TrainingPhase.Alignment,
                Source = "Huang et al. 2024, Table A.12: the alignment stage uses AdamW with a weight "
                        + "decay of 0.05, betas 0.9 and 0.999, a learning rate of 3e-4, 400 warmup "
                        + "steps, 4 accumulated gradient batches, a total batch size of 80 and a "
                        + "gradient norm of 5.0, over 5 epochs.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 3e-5, Beta1 = 0.9, Beta2 = 0.999,
                WeightDecay = 0.05, ReferenceBatchSize = 80, WarmupSteps = 400,
                MaxGradientNorm = 5.0, GradientAccumulationSteps = 4,
                Phase = TrainingPhase.FineTuning,
                Source = "Huang et al. 2024, Table A.13: the instruction-tuning stage repeats those "
                        + "settings at a learning rate of 3e-5 over 10 epochs.")]
public partial class LEOVL<T> : VisionLanguageModelBase<T>, IThreeDVisionLanguageModel<T>
{
    private readonly LEOVLOptions _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public LEOVL(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        LEOVLOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new LEOVLOptions();
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

    public LEOVL(
        NeuralNetworkArchitecture<T> architecture,
        LEOVLOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new LEOVLOptions();
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
    /// Generates from 2D image using LEO-VL's embodied multi-modal approach.
    /// Per the paper (Huang et al., 2024), LEO processes visual observations
    /// through object-centric 3D tokens. For a single 2D image, the vision
    /// encoder extracts spatial features which are fused with text instruction
    /// tokens via cross-attention before the LLM decoder generates the response.
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
    /// Processes 3D point cloud using LEO-VL's embodied spatial reasoning approach.
    /// Per the paper (Huang et al., 2024), LEO is an embodied multi-modal generalist
    /// that represents 3D scenes through: (1) object-centric 3D tokens extracted by
    /// segmenting the point cloud into objects, (2) spatial relationship encoding via
    /// relative 3D position embeddings between objects, (3) ego-centric representation
    /// where object features are encoded relative to the agent's viewpoint. The system
    /// processes XYZ+RGB+normal (9-channel) point clouds and uses NumViews RGB-D
    /// observations for multi-view scene understanding.
    /// </summary>
    public Tensor<T> GenerateFrom3D(Tensor<T> pointCloud, string prompt)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(pointCloud);

        int totalValues = pointCloud.Length;
        int channels = _options.PointChannels; // 9: XYZ+RGB+normals
        int numPoints = Math.Min(totalValues / Math.Max(1, channels), _options.MaxPoints);
        if (numPoints == 0)
            numPoints = Math.Min(totalValues, _options.MaxPoints);
        int encoderDim = _options.PointEncoderDim;

        // Step 1: Object-centric segmentation via spatial clustering
        // Group nearby points into object clusters using grid-based voxelization
        var engine = Engine;
        int maxObjects = 32;
        double voxelSize = 0.5;
        int visionDim = _options.VisionDim;

        // Points p < numPoints with p * channels + 2 < totalValues are read (the scan stops at the
        // first point that is not).
        int readPoints = ReadablePointCount(numPoints, channels, totalValues);

        Tensor<T> sceneTokens;
        if (readPoints == 0)
        {
            sceneTokens = new Tensor<T>([visionDim]);
        }
        else if (encoderDim == 0)
        {
            if (channels > 0 || visionDim > 0)
                throw new DivideByZeroException("Object features cannot be built with a zero point-encoder width.");
            sceneTokens = new Tensor<T>([0]);
        }
        else
        {
            var coords = PointCoordinates(pointCloud, readPoints, channels);

            // Hash-based voxel assignment → object cluster (integer arithmetic, as master):
            // |trunc(x / s) * 73856093 ^ trunc(y / s) * 19349669 ^ trunc(z / s) * 83492791| % maxObjects.
            var cells = engine.TensorTrunc(engine.TensorDivideScalar(coords, NumOps.FromDouble(voxelSize))).Cast<int>();
            var hashX = engine.TensorMultiplyScalar(engine.TensorSlice(cells, [0, 0], [readPoints, 1]), 73856093);
            var hashY = engine.TensorMultiplyScalar(engine.TensorSlice(cells, [0, 1], [readPoints, 1]), 19349669);
            var hashZ = engine.TensorMultiplyScalar(engine.TensorSlice(cells, [0, 2], [readPoints, 1]), 83492791);
            var mixed = engine.TensorBitwiseXor(engine.TensorBitwiseXor(hashX, hashY), hashZ).Reshape([readPoints]);
            if (engine.TensorSum(engine.TensorEquals(mixed, int.MinValue)) > 0)
                throw new OverflowException("Negating the minimum value of a twos complement number is invalid.");
            var objectIds = engine.TensorRemainder(
                engine.TensorAbs(mixed), Tensor<int>.CreateDefault([readPoints], maxObjects));

            // Accumulate point features (color + normals): value c of a point goes to feature c % encoderDim.
            var objectSums = engine.ScatterAdd(
                FoldedPointChannels(pointCloud, readPoints, channels, encoderDim),
                engine.TensorBroadcastTo(objectIds.Reshape([readPoints, 1]), [readPoints, encoderDim]),
                0,
                maxObjects);

            // Normalize centroids and features by the object point counts.
            var objectCounts = engine.TensorBinCount(objectIds, maxObjects).Cast<T>().Reshape([maxObjects, 1]);
            var safeCounts = engine.TensorMax(objectCounts, NumOps.One);
            var active = engine.TensorGreaterThan(objectCounts, NumOps.Zero);
            var objectCentroids = engine.TensorDivide(
                engine.ScatterAdd(
                    coords, engine.TensorBroadcastTo(objectIds.Reshape([readPoints, 1]), [readPoints, 3]), 0, maxObjects),
                engine.TensorBroadcastTo(safeCounts, [maxObjects, 3]));
            var objectFeatures = engine.TensorDivide(objectSums, engine.TensorBroadcastTo(safeCounts, [maxObjects, encoderDim]));

            // Step 2: Spatial relationship encoding via relative 3D position embeddings
            // Encode pairwise spatial relationships between active objects i != j:
            // feature_i += sum_j exp(-dist_ij) * 0.1 * feature_j + sin(dx (d + 1)) cos(dy (d + 1)) * 0.01,
            // with dx = centroid_j - centroid_i. Objects update in order, each reading the already
            // updated earlier objects.
            var objectIndex = engine.TensorArange<T>(0, maxObjects, 1);
            var isSelf = engine.TensorEquals(
                engine.TensorBroadcastTo(objectIndex.Reshape([maxObjects, 1]), [maxObjects, maxObjects]),
                engine.TensorBroadcastTo(objectIndex.Reshape([1, maxObjects]), [maxObjects, maxObjects]));
            var pairMask = engine.TensorMultiply(
                engine.TensorMultiply(active, active.Reshape([1, maxObjects])),
                engine.ScalarMinusTensor(NumOps.One, isSelf));
            Tensor<T> Delta(int axis)
            {
                var column = engine.TensorSlice(objectCentroids, [0, axis], [maxObjects, 1]);
                return engine.TensorSubtract(
                    engine.TensorBroadcastTo(column.Reshape([1, maxObjects]), [maxObjects, maxObjects]),
                    engine.TensorBroadcastTo(column, [maxObjects, maxObjects]));
            }
            var dx = Delta(0);
            var dy = Delta(1);
            var dz = Delta(2);
            var dist = engine.TensorAddScalar(
                engine.TensorSqrt(engine.TensorAdd(
                    engine.TensorAdd(engine.TensorMultiply(dx, dx), engine.TensorMultiply(dy, dy)),
                    engine.TensorMultiply(dz, dz))),
                NumOps.FromDouble(1e-8));
            // Spatial attention: closer objects have stronger influence
            var spatialAttn = engine.TensorMultiplyScalar(engine.TensorExp(engine.TensorNegate(dist)), NumOps.FromDouble(0.1));
            var frequency = engine.TensorArange<T>(1, encoderDim + 1, 1).Reshape([1, 1, encoderDim]);
            var pairShape = new[] { maxObjects, maxObjects, encoderDim };
            var relEmbed = engine.TensorMultiplyScalar(
                engine.TensorMultiply(
                    engine.TensorSin(engine.TensorMultiply(dx.Reshape([maxObjects, maxObjects, 1]), frequency)),
                    engine.TensorCos(engine.TensorMultiply(dy.Reshape([maxObjects, maxObjects, 1]), frequency))),
                NumOps.FromDouble(0.01));
            var relSum = engine.ReduceSum(
                engine.TensorWhere(
                    engine.TensorBroadcastTo(pairMask.Reshape([maxObjects, maxObjects, 1]), pairShape),
                    relEmbed,
                    new Tensor<T>(pairShape)),
                [1], false);

            var objectRows = new List<Tensor<T>>(maxObjects);
            for (int obj = 0; obj < maxObjects; obj++)
                objectRows.Add(engine.TensorSlice(objectFeatures, [obj, 0], [1, encoderDim]).Reshape([encoderDim]));
            var noTerms = new Tensor<T>([maxObjects, encoderDim]);
            for (int i = 0; i < maxObjects; i++)
            {
                // Modulate features with spatial relationships
                var current = StackLeading(objectRows, maxObjects, encoderDim);
                var rowMask = engine.TensorBroadcastTo(
                    engine.TensorSlice(pairMask, [i, 0], [1, maxObjects]).Reshape([maxObjects, 1]), [maxObjects, encoderDim]);
                var rowAttn = engine.TensorBroadcastTo(
                    engine.TensorSlice(spatialAttn, [i, 0], [1, maxObjects]).Reshape([maxObjects, 1]), [maxObjects, encoderDim]);
                var attended = engine.ReduceSum(
                    engine.TensorWhere(rowMask, engine.TensorMultiply(current, rowAttn), noTerms), [0], false);
                objectRows[i] = engine.TensorAdd(
                    objectRows[i],
                    engine.TensorAdd(attended, engine.TensorSlice(relSum, [i, 0], [1, encoderDim]).Reshape([encoderDim])));
            }

            // Step 3: Aggregate object tokens into scene representation. The scene tokens must match the
            // encoder input width (_options.VisionDim, wired in InitializeLayers), not the point-encoder
            // width — so tile the aggregated PointEncoderDim object features into VisionDim (identity when
            // the two dims match). Forwarding PointEncoderDim-wide tokens into the VisionDim encoder threw
            // "embedding dimension does not match weight dimension" whenever the dims differed.
            // Larger objects = more important: the scene is the point-count weighted mean of the active objects
            // (the total weight is the number of read points).
            var weighted = engine.TensorWhere(
                engine.TensorBroadcastTo(active, [maxObjects, encoderDim]),
                engine.TensorMultiply(
                    StackLeading(objectRows, maxObjects, encoderDim), engine.TensorBroadcastTo(objectCounts, [maxObjects, encoderDim])),
                noTerms);
            var sceneMean = engine.TensorDivideScalar(engine.ReduceSum(weighted, [0], false), NumOps.FromDouble(readPoints));
            sceneTokens = CyclicTake(sceneMean, visionDim);
        }

        // Step 4: Process through encoder and fuse with prompt
        var encoded = sceneTokens;
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
                    // Vision/point-token embedding dim the encoder's first attention consumes.
                    // Was hard-coded 512, which silently disagreed with VisionDim (1024 by paper
                    // default) — the encoder built [512,512] attention while callers/tests fed
                    // VisionDim-wide tokens, throwing "embedding dimension does not match weight
                    // dimension". Drive it from the configured VisionDim so the encoder, the
                    // declared option, and the post-patch-embedding token input all agree.
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
            Name = _useNativeMode ? "LEO-VL-Native" : "LEO-VL-ONNX",
            Description = "LEO-VL: efficient 3D scene representation from multi-view RGB-D.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "LEO-VL";
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(LEOVL<T>));
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
