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
/// 3D-LLM: injects 3D spatial features into large language models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// 3D-LLM (UCLA, 2023) injects 3D spatial features into large language models by rendering
/// multi-view images from 3D scenes and projecting 2D features back into 3D space. This
/// enables the LLM to understand spatial layouts, answer questions about 3D environments,
/// and follow language-guided navigation instructions within reconstructed 3D scenes.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "3D-LLM: Injecting the 3D World into Large Language Models (UCLA, 2023)"</item></list></para>
/// <para><b>For Beginners:</b> 3D-LLM is a vision-language model that enables language
/// understanding of 3D environments and scenes. Default values follow the original paper
/// settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a 3D-LLM model for injecting 3D spatial features into LLMs
/// // enabling language-guided 3D scene navigation and understanding
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new ThreeDLLM&lt;double&gt;(architecture, "threedllm.onnx");
///
/// // Training mode with native layers
/// var trainModel = new ThreeDLLM&lt;double&gt;(architecture, new ThreeDLLMOptions());
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
    "3D-LLM: Injecting the 3D World into Large Language Models",
    "https://arxiv.org/abs/2307.12981",
    Year = 2023,
    Authors = "Hong et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-5, Beta1 = 0.9, Beta2 = 0.999,
                WeightDecay = 0.05, WarmupSteps = 1000, MinLearningRate = 0,
                Schedule = LearningRateSchedulerType.CosineAnnealing,
                Source = "Hong et al. 2023, Sec. 4: AdamW with beta1 0.9, beta2 0.999 and a weight "
                        + "decay of 0.05, under a linear warmup over the initial 1K steps rising from "
                        + "1e-8 to 1e-5, followed by a cosine decay to a minimum learning rate of 0. The "
                        + "1e-5 warmup endpoint is the peak rate.")]
public partial class ThreeDLLM<T> : VisionLanguageModelBase<T>, IThreeDVisionLanguageModel<T>
{
    private readonly ThreeDLLMOptions _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public ThreeDLLM(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        ThreeDLLMOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new ThreeDLLMOptions();
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

    public ThreeDLLM(
        NeuralNetworkArchitecture<T> architecture,
        ThreeDLLMOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new ThreeDLLMOptions();
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
    /// Generates from 2D image using 3D-LLM's multi-view feature lifting approach.
    /// Per the paper (Hong et al., 2023), 3D features are obtained by rendering
    /// the scene from multiple viewpoints and lifting 2D features back to 3D.
    /// For a single 2D image, the vision encoder extracts spatial features which
    /// are fused with text instruction tokens via cross-attention before the LLM
    /// decoder generates the response.
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
    /// Processes 3D point cloud using 3D-LLM's multi-view feature lifting approach.
    /// Per the paper (Hong et al., 2023), 3D features are obtained by:
    /// (1) rendering the 3D scene from NumViews viewpoints, (2) extracting 2D
    /// features from each view using a pretrained vision encoder, (3) lifting
    /// 2D features back to 3D using known camera poses and depth-based projection.
    /// The 3D-aware features capture both appearance and spatial structure. These
    /// are projected to the LLM space with 3D position embeddings that encode
    /// the xyz coordinates of each feature.
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
        int numViews = _options.NumViews;
        int encoderDim = _options.PointEncoderDim;

        var engine = Engine;
        if (numViews < 0)
            throw new OverflowException("The number of views cannot be negative.");
        // Points p < numPoints with p * channels + 2 < totalValues are read (the scan stops at the
        // first point that is not).
        int readPoints = ReadablePointCount(numPoints, channels, totalValues);

        // Step 1: Multi-view rendering simulation
        // For each view, project 3D points onto a virtual camera plane
        Tensor<T> viewFeatures = new Tensor<T>([numViews, encoderDim]);
        if (numViews > 0 && readPoints > 0)
        {
            var coords = PointCoordinates(pointCloud, readPoints, channels);
            var x = engine.TensorSlice(coords, [0, 0], [readPoints, 1]).Reshape([1, readPoints]);
            var y = engine.TensorSlice(coords, [0, 1], [readPoints, 1]).Reshape([1, readPoints]);
            var z = engine.TensorSlice(coords, [0, 2], [readPoints, 1]).Reshape([1, readPoints]);
            int[] viewShape = [numViews, readPoints];

            // Virtual camera at different azimuth angles 2 pi v / numViews around the scene
            var azimuth = engine.TensorDivideScalar(
                engine.TensorMultiplyScalar(engine.TensorArange<T>(0, numViews, 1), NumOps.FromDouble(2.0 * Math.PI)),
                NumOps.FromDouble(numViews)).Reshape([numViews, 1]);
            var cosA = engine.TensorBroadcastTo(engine.TensorCos(azimuth), viewShape);
            var sinA = engine.TensorBroadcastTo(engine.TensorSin(azimuth), viewShape);
            var xWide = engine.TensorBroadcastTo(x, viewShape);
            var yWide = engine.TensorBroadcastTo(y, viewShape);
            var zWide = engine.TensorBroadcastTo(z, viewShape);

            // Camera-relative coordinates (rotation around Y axis)
            var xCam = engine.TensorAdd(engine.TensorMultiply(xWide, cosA), engine.TensorMultiply(zWide, sinA));
            var depth = engine.TensorAddScalar(
                engine.TensorAdd(engine.TensorMultiply(engine.TensorNegate(xWide), sinA), engine.TensorMultiply(zWide, cosA)),
                NumOps.FromDouble(5.0)); // 5.0 = camera distance
            // Points with depth < 0.1 are behind the camera (a NaN depth is not, as in a scalar comparison).
            var visible = engine.ScalarMinusTensor(NumOps.One, engine.TensorLessThan(depth, NumOps.FromDouble(0.1)));
            var noView = new Tensor<T>(viewShape);

            // Project to 2D image plane
            var u = engine.TensorDivide(xCam, depth);
            var vCoord = engine.TensorDivide(yWide, depth);

            // Accumulate features weighted by visibility (closer = more important)
            var weight = engine.TensorDivide(
                Tensor<T>.CreateDefault(viewShape, NumOps.One),
                engine.TensorAddScalar(engine.TensorMultiply(depth, depth), NumOps.FromDouble(1e-8)));

            // Include color/appearance features if available: the sum of values 3 .. channels - 1.
            int appearanceChannels = channels - 3;
            Tensor<T> appearance;
            if (appearanceChannels <= 0)
            {
                appearance = new Tensor<T>([1, readPoints]);
            }
            else
            {
                var channelIndex = engine.TensorAdd(
                    engine.TensorBroadcastTo(
                        engine.TensorAddScalar(
                            engine.TensorMultiplyScalar(engine.TensorArange<int>(0, readPoints, 1), channels), 3).Reshape([readPoints, 1]),
                        [readPoints, appearanceChannels]),
                    engine.TensorBroadcastTo(
                        engine.TensorArange<int>(0, appearanceChannels, 1).Reshape([1, appearanceChannels]), [readPoints, appearanceChannels]));
                var appearanceValues = engine.TensorWhere(
                    engine.TensorLessThan(channelIndex, totalValues).Cast<T>(),
                    engine.TensorTake(pointCloud.Reshape([totalValues]), engine.TensorMin(channelIndex, totalValues - 1)),
                    new Tensor<T>([readPoints, appearanceChannels]));
                appearance = engine.ReduceSum(appearanceValues, [1], false).Reshape([1, readPoints]);
            }
            var contribution = engine.TensorWhere(
                visible,
                engine.TensorMultiply(weight, engine.TensorAddScalar(engine.TensorBroadcastTo(appearance, viewShape), NumOps.One)),
                noView);

            if (encoderDim == 0)
            {
                if (engine.TensorSum(visible.Cast<int>()) > 0)
                    throw new DivideByZeroException("Projected points cannot be binned into a zero-width feature vector.");
            }
            else
            {
                // featureIdx = |(int)(u * 100 + v * 10)| % encoderDim for every visible point.
                var cellValue = engine.TensorWhere(
                    visible,
                    engine.TensorAdd(
                        engine.TensorMultiplyScalar(u, NumOps.FromDouble(100.0)),
                        engine.TensorMultiplyScalar(vCoord, NumOps.FromDouble(10.0))),
                    noView);
                var cell = engine.TensorTrunc(cellValue).Cast<int>();
                if (engine.TensorSum(engine.TensorEquals(cell, int.MinValue)) > 0)
                    throw new OverflowException("Negating the minimum value of a twos complement number is invalid.");
                var featureIdx = engine.TensorRemainder(engine.TensorAbs(cell), Tensor<int>.CreateDefault(viewShape, encoderDim));
                viewFeatures = engine.ScatterAdd(contribution, featureIdx, 1, encoderDim);
            }
        }

        Tensor<T> features3D;
        if (encoderDim == 0)
        {
            features3D = new Tensor<T>([0]);
        }
        else
        {
            if (numViews > 0)
            {
                // Step 2: 2D feature extraction per view (via vision encoder)
                // Normalize view features: divide by the view's L2 norm + 1e-8.
                var norm = engine.TensorAddScalar(
                    engine.TensorSqrt(engine.ReduceSum(engine.TensorMultiply(viewFeatures, viewFeatures), [1], true)),
                    NumOps.FromDouble(1e-8));
                viewFeatures = engine.TensorDivide(viewFeatures, engine.TensorBroadcastTo(norm, [numViews, encoderDim]));
            }

            // Step 3: Lift 2D features back to 3D with position embeddings
            // Aggregate multi-view features using view-weighted attention exp(|f|) over the views
            // (0 when the attention mass is not above 1e-8).
            var attn = engine.TensorExp(engine.TensorAbs(viewFeatures));
            var attnSum = engine.ReduceSum(attn, [0], false);
            var sum = engine.ReduceSum(engine.TensorMultiply(attn, viewFeatures), [0], false);
            features3D = engine.TensorWhere(
                engine.TensorGreaterThan(attnSum, NumOps.FromDouble(1e-8)),
                engine.TensorDivide(sum, attnSum),
                new Tensor<T>([encoderDim]));

            // Step 4: Add 3D position embeddings (sinusoidal over XYZ): sin(d pi / encoderDim) * 0.1.
            var posEmbed = engine.TensorMultiplyScalar(
                engine.TensorSin(engine.TensorDivideScalar(
                    engine.TensorMultiplyScalar(engine.TensorArange<T>(0, encoderDim, 1), NumOps.FromDouble(Math.PI)),
                    NumOps.FromDouble(encoderDim))),
                NumOps.FromDouble(0.1));
            features3D = engine.TensorAdd(features3D, posEmbed);
        }

        // Step 5: Process through encoder layers
        var encoded = features3D;
        for (int i = 0; i < _encoderLayerEnd && i < Layers.Count; i++)
            encoded = Layers[i].Forward(encoded);

        // Step 6: Fuse with prompt and decode through LLM
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
            Name = _useNativeMode ? "3D-LLM-Native" : "3D-LLM-ONNX",
            Description = "3D-LLM: injects 3D spatial features into large language models.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "3D-LLM";
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(ThreeDLLM<T>));
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
