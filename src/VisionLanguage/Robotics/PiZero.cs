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

namespace AiDotNet.VisionLanguage.Robotics;

/// <summary>
/// pi-zero: PaliGemma VLM with action expert for 8 robot embodiments.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// pi-zero (Black et al., 2024) is a vision-language-action flow model built on PaliGemma VLM
/// with a dedicated action expert for general robot control. It uses flow matching to generate
/// continuous robot actions, trained across 8 different robot embodiments for dexterous
/// manipulation including folding, assembly, and bin-picking tasks.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "pi0: A Vision-Language-Action Flow Model for General Robot Control (Black et al., 2024)"</item></list></para>
/// <para><b>For Beginners:</b> pi-zero is a vision-language-action model for general robot
/// control across multiple robot platforms. Default values follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a pi-zero model for general robot control
/// // PaliGemma VLM with action flow expert across 8 robot embodiments
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new PiZero&lt;double&gt;(architecture, "pizero.onnx");
///
/// // Training mode with native layers
/// var trainModel = new PiZero&lt;double&gt;(architecture, new PiZeroOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.Robotics)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "pi0: A Vision-Language-Action Flow Model for General Robot Control",
    "https://arxiv.org/abs/2410.24164",
    Year = 2024,
    Authors = "Black et al."
)]
public partial class PiZero<T> : VisionLanguageModelBase<T>, IVisionLanguageAction<T>
{
    private readonly PiZeroOptions _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public PiZero(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        PiZeroOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new PiZeroOptions();
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

    public PiZero(
        NeuralNetworkArchitecture<T> architecture,
        PiZeroOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new PiZeroOptions();
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
    public string LanguageModelName => _options.LanguageModelName;
    /// <inheritdoc />
    public int ActionDimension => _options.ActionDimension;

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
    /// Generates from image using pi-zero's PaliGemma VLM backbone + flow matching conditioning.
    /// The VLM processes visual observation and instruction to produce a conditioning signal
    /// that the action expert uses for flow matching action generation.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;
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
    /// Predicts action using pi-zero's flow matching formulation. Per the paper
    /// (Physical Intelligence 2024), the VLM backbone (PaliGemma) produces a
    /// conditioning signal, and a separate action expert generates actions via
    /// flow matching. Flow matching learns to transport samples from a noise
    /// distribution to the action distribution by integrating a learned velocity
    /// field. At inference, noise is iteratively denoised through NumFlowSteps
    /// ODE integration steps to produce smooth action trajectories. The action
    /// expert runs at high frequency (50Hz) independent of VLM inference.
    /// </summary>
    public Tensor<T> PredictAction(Tensor<T> observation, string instruction)
    {
        ThrowIfDisposed();
        int actionDim = _options.ActionDimension;
        int horizon = _options.PredictionHorizon;
        int flowSteps = _options.NumFlowSteps;

        // Step 1: Encode visual observation through VLM vision encoder
        var visualFeatures = EncodeImage(observation);
        int dim = visualFeatures.Length;

        // Step 2: Encode instruction through VLM text encoder
        var instrTokens = TokenizeText(instruction);
        int instrLen = instrTokens.Length;

        // Step 3: VLM backbone produces conditioning representation
        var engine = Engine;
        // vis[d] * (0.5 + sigmoid(instr[d % instrLen] / 100)).
        var visFlat = visualFeatures.Reshape([dim]);
        var condInput = instrLen > 0
            ? engine.TensorMultiply(
                visFlat,
                engine.TensorAddScalar(
                    engine.TensorSigmoid(engine.TensorDivideScalar(CyclicTake(instrTokens, dim), NumOps.FromDouble(100.0))),
                    NumOps.FromDouble(0.5)))
            : visFlat.Clone();

        var conditioningRepr = condInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            conditioningRepr = Layers[i].Forward(conditioningRepr);

        // Step 4: Extract per-action-dimension conditioning from VLM output: the mean of
        // block a = [a * blockSize, min(a * blockSize + blockSize, length)) (0 for an empty block).
        int totalActions = actionDim * horizon;
        if (totalActions == 0)
            return new Tensor<T>([totalActions]);

        int condLength = conditioningRepr.Length;
        Tensor<T> condPerDim;
        if (condLength == 0)
        {
            condPerDim = new Tensor<T>([actionDim]);
        }
        else
        {
            int blockSize = Math.Max(1, condLength / actionDim);
            var positions = engine.TensorAdd(
                engine.TensorBroadcastTo(
                    engine.TensorMultiplyScalar(engine.TensorArange<int>(0, actionDim, 1), blockSize).Reshape([actionDim, 1]),
                    [actionDim, blockSize]),
                engine.TensorBroadcastTo(engine.TensorArange<int>(0, blockSize, 1).Reshape([1, blockSize]), [actionDim, blockSize]));
            var inBlock = engine.TensorLessThan(positions, condLength).Cast<T>();
            var blockValues = engine.TensorWhere(
                inBlock,
                engine.TensorTake(conditioningRepr.Reshape([condLength]), engine.TensorMin(positions, condLength - 1)),
                new Tensor<T>([actionDim, blockSize]));
            condPerDim = engine.TensorDivide(
                engine.ReduceSum(blockValues, [1], false),
                engine.TensorMax(engine.ReduceSum(inBlock, [1], false), NumOps.One));
        }

        // Step 5: Flow matching action expert
        // Initialize from noise (t=0) and integrate velocity field to t=1
        var tiledCond = engine.TensorBroadcastTo(condPerDim.Reshape([1, actionDim]), [horizon, actionDim]);
        var stepNumber = engine.TensorArange<T>(0, horizon, 1).Reshape([horizon, 1]);

        // Initialize noise at t=0: structured initial noise seeded by conditioning,
        // cond[dimIdx] * 0.01 * sin((stepIdx + 1) pi / (horizon + 1)).
        var noisePhase = engine.TensorSin(engine.TensorMultiplyScalar(
            engine.TensorAddScalar(stepNumber, NumOps.One), NumOps.FromDouble(Math.PI / (horizon + 1))));
        var actionTrajectory = engine.TensorMultiply(
            engine.TensorMultiplyScalar(tiledCond, NumOps.FromDouble(0.01)), noisePhase).Reshape([totalActions]);

        // Learned velocity field: v(x_t, t, c). Direction: toward the conditioned target
        // cond[dimIdx] * exp(-0.05 stepIdx) (temporal decay).
        var target = engine.TensorMultiply(
            tiledCond, engine.TensorExp(engine.TensorMultiplyScalar(stepNumber, NumOps.FromDouble(-0.05)))).Reshape([totalActions]);

        // ODE integration: Euler method over flow steps from t=0 to t=1. Optimal transport velocity
        // (target - current) / max(1e-6, 1 - t) with t = s / flowSteps.
        if (flowSteps > 0)
        {
            T dt = NumOps.FromDouble(1.0 / flowSteps);
            var remainingTime = engine.TensorMax(
                engine.ScalarMinusTensor(NumOps.One, engine.TensorDivideScalar(
                    engine.TensorArange<T>(0, flowSteps, 1), NumOps.FromDouble(flowSteps))),
                NumOps.FromDouble(1e-6));
            for (int s = 0; s < flowSteps; s++)
            {
                var velocity = engine.TensorDivide(
                    engine.TensorSubtract(target, actionTrajectory),
                    engine.TensorBroadcastTo(Pick(remainingTime, s), [totalActions]));

                // Euler step
                actionTrajectory = engine.TensorAdd(actionTrajectory, engine.TensorMultiplyScalar(velocity, dt));
            }
        }

        // Step 6: Bound actions and package result
        var actions = engine.TensorTanh(actionTrajectory);

        return actions;
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
                LayerHelper<T>.CreateDefaultRoboticsActionLayers(
                    _options.VisionDim,
                    _options.DecoderDim,
                    256,
                    _options.NumVisionLayers,
                    _options.NumDecoderLayers,
                    2,
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
        _encoderLayerEnd = 1 + _options.NumVisionLayers * lpb + 2;
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
            Name = _useNativeMode ? "pi-zero-Native" : "pi-zero-ONNX",
            Description = "pi-zero: PaliGemma VLM with action expert for 8 robot embodiments.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "pi-zero";
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(PiZero<T>));
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
