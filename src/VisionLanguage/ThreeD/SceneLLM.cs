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
/// Scene-LLM: voxel-based 3D scene understanding with language models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Scene-LLM (2024) extends language models for 3D visual understanding and reasoning using
/// voxel-based scene representations. It discretizes 3D environments into voxel grids with
/// semantic features, enabling language-guided spatial reasoning about object locations,
/// room layouts, and navigation instructions in indoor environments.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Scene-LLM: Extending Language Model for 3D Visual Understanding and Reasoning (Various, 2024)"</item></list></para>
/// <para><b>For Beginners:</b> Scene-LLM is a vision-language model for voxel-based 3D scene
/// understanding with language-guided spatial reasoning. Default values follow the original
/// paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Scene-LLM model for voxel-based 3D scene understanding
/// // with language-guided spatial reasoning over indoor environments
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new SceneLLM&lt;double&gt;(architecture, "scenellm.onnx");
///
/// // Training mode with native layers
/// var trainModel = new SceneLLM&lt;double&gt;(architecture, new SceneLLMOptions());
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
    "Scene-LLM: Extending Language Model for 3D Visual Understanding and Reasoning",
    "https://arxiv.org/abs/2403.11401",
    Year = 2024,
    Authors = "Fu et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-5, ReferenceBatchSize = 64,
                WarmupSteps = 1000, Phase = TrainingPhase.PreTraining,
                Source = "Fu et al. 2024, Sec. 4: AdamW at a total batch size of 64, with a learning "
                        + "rate of 1e-5 to train the projection layer, starting with 1000 warmup steps "
                        + "at a warmup learning rate of 1e-6.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2e-5, ReferenceBatchSize = 64,
                WarmupSteps = 2000, Phase = TrainingPhase.FineTuning,
                Source = "Fu et al. 2024, Sec. 4: the stage that finetunes the projection layer "
                        + "together with the LLM uses a learning rate of 2e-5 and 2000 warmup steps, at "
                        + "the same total batch size of 64.")]
public partial class SceneLLM<T> : VisionLanguageModelBase<T>, IThreeDVisionLanguageModel<T>
{
    private readonly SceneLLMOptions _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public SceneLLM(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        SceneLLMOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new SceneLLMOptions();
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

    public SceneLLM(
        NeuralNetworkArchitecture<T> architecture,
        SceneLLMOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new SceneLLMOptions();
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
    /// Generates from 2D image using SceneLLM's hybrid scene representation.
    /// Per the paper (2024), SceneLLM uses coarse voxelization and fine-grained
    /// octree encoding for scene understanding. For a single 2D image, the vision
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
    /// Processes 3D point cloud using SceneLLM's hybrid scene representation.
    /// Per the paper (2024), SceneLLM handles large indoor scenes (up to 32K
    /// points) using a hybrid approach: (1) coarse voxelization at VoxelResolution
    /// to capture global scene structure, (2) fine-grained ego-centric octree
    /// encoding for regions near the query focus, (3) hierarchical aggregation
    /// where coarse features provide context and fine features provide detail.
    /// This enables efficient processing of room-scale and building-scale 3D
    /// scenes without losing detail in areas relevant to the query.
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
        int voxelRes = _options.VoxelResolution;
        int encoderDim = _options.PointEncoderDim;

        var engine = Engine;
        // Points p < numPoints with p * channels + 2 < totalValues are read (the scan stops at the
        // first point that is not).
        int readPoints = ReadablePointCount(numPoints, channels, totalValues);
        int totalVoxels = voxelRes * voxelRes * voxelRes;
        int maxTrackedVoxels = Math.Min(totalVoxels, 4096);
        if (maxTrackedVoxels < 0)
            throw new OverflowException("The voxel resolution gives a negative number of tracked voxels.");

        Tensor<T> sceneTokens;
        if (readPoints == 0)
        {
            sceneTokens = new Tensor<T>([encoderDim]);
        }
        else
        {
            if (maxTrackedVoxels == 0)
                throw new DivideByZeroException("Points cannot be assigned to zero tracked voxels.");
            var pointFeatures = FoldedPointChannels(pointCloud, readPoints, channels, encoderDim);
            if (encoderDim == 0)
            {
                sceneTokens = new Tensor<T>([0]);
            }
            else
            {
                // Step 1: Compute scene bounding box for voxelization (NaN coordinates never move a bound;
                // the bounds start at double.MaxValue / double.MinValue).
                var coords = PointCoordinates(pointCloud, readPoints, channels);
                var isNan = engine.TensorIsNan(coords);
                T highestStart = NumOps.FromDouble(double.MinValue);
                T lowestStart = NumOps.FromDouble(double.MaxValue);
                var lowest = engine.TensorMin(
                    engine.TensorNegate(engine.ReduceMax(
                        engine.TensorNegate(engine.TensorWhere(isNan, Tensor<T>.CreateDefault([readPoints, 3], lowestStart), coords)),
                        [0], true)),
                    lowestStart);
                var highest = engine.TensorMax(
                    engine.ReduceMax(engine.TensorWhere(isNan, Tensor<T>.CreateDefault([readPoints, 3], highestStart), coords), [0], true),
                    highestStart);
                var range = engine.TensorAddScalar(engine.TensorSubtract(highest, lowest), NumOps.FromDouble(1e-8));

                // Step 2: Coarse voxelization - global scene structure:
                // v = min(res - 1, (int)((x - min) / range * res)) per axis,
                // voxel = (vx res^2 + vy res + vz) % maxTrackedVoxels.
                var cells = engine.TensorMin(
                    engine.TensorTrunc(engine.TensorMultiplyScalar(
                        engine.TensorDivide(
                            engine.TensorSubtract(coords, engine.TensorBroadcastTo(lowest, [readPoints, 3])),
                            engine.TensorBroadcastTo(range, [readPoints, 3])),
                        NumOps.FromDouble(voxelRes))).Cast<int>(),
                    voxelRes - 1);
                var linear = engine.TensorAdd(
                    engine.TensorAdd(
                        engine.TensorMultiplyScalar(engine.TensorSlice(cells, [0, 0], [readPoints, 1]), voxelRes * voxelRes),
                        engine.TensorMultiplyScalar(engine.TensorSlice(cells, [0, 1], [readPoints, 1]), voxelRes)),
                    engine.TensorSlice(cells, [0, 2], [readPoints, 1])).Reshape([readPoints]);
                var voxelIds = engine.TensorRemainder(linear, Tensor<int>.CreateDefault([readPoints], maxTrackedVoxels));

                // Normalize voxel features
                var voxelCounts = engine.TensorBinCount(voxelIds, maxTrackedVoxels).Cast<T>().Reshape([maxTrackedVoxels, 1]);
                var countsWide = engine.TensorBroadcastTo(voxelCounts, [maxTrackedVoxels, encoderDim]);
                var voxelSums = engine.ScatterAdd(
                    pointFeatures,
                    engine.TensorBroadcastTo(voxelIds.Reshape([readPoints, 1]), [readPoints, encoderDim]),
                    0,
                    maxTrackedVoxels);
                var voxelFeatures = engine.TensorDivide(voxelSums, engine.TensorMax(countsWide, NumOps.One));

                // Step 3: Fine-grained octree encoding for ego-centric region
                // Focus on the scene centroid as the ego-center
                var center = engine.TensorDivideScalar(engine.TensorAdd(lowest, highest), NumOps.FromDouble(2.0));
                var offset = engine.TensorSubtract(coords, engine.TensorBroadcastTo(center, [readPoints, 3]));
                // Distance from ego-center → octree level weight
                var dist = engine.TensorSqrt(engine.ReduceSum(engine.TensorMultiply(offset, offset), [1], false));
                var weight = engine.TensorExp(engine.TensorMultiplyScalar(dist, NumOps.FromDouble(-2.0))); // Exponential falloff
                var fineWeight = engine.TensorBroadcastTo(engine.ReduceSum(weight, [0], false), [encoderDim]);
                var fineSum = engine.TensorMatMul(weight.Reshape([1, readPoints]), pointFeatures).Reshape([encoderDim]);
                var fineFeatures = engine.TensorWhere(
                    engine.TensorGreaterThan(fineWeight, NumOps.FromDouble(1e-8)),
                    engine.TensorDivide(fineSum, fineWeight),
                    fineSum);

                // Step 4: Hierarchical aggregation (coarse + fine): the point-count weighted mean of the
                // occupied voxels (the total weight is the number of read points).
                var weighted = engine.TensorWhere(
                    engine.TensorGreaterThan(countsWide, NumOps.Zero),
                    engine.TensorMultiply(voxelFeatures, countsWide),
                    new Tensor<T>([maxTrackedVoxels, encoderDim]));
                var coarseAgg = engine.TensorDivideScalar(engine.ReduceSum(weighted, [0], false), NumOps.FromDouble(readPoints));

                // Combine: coarse (global context) + fine (local detail)
                sceneTokens = engine.TensorAdd(
                    engine.TensorMultiplyScalar(coarseAgg, NumOps.FromDouble(0.4)),
                    engine.TensorMultiplyScalar(fineFeatures, NumOps.FromDouble(0.6)));
            }
        }

        // Step 5: Process through encoder and LLM decoder
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
            Name = _useNativeMode ? "Scene-LLM-Native" : "Scene-LLM-ONNX",
            Description = "Scene-LLM: voxel-based 3D scene understanding with language models.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "Scene-LLM";
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(SceneLLM<T>));
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
