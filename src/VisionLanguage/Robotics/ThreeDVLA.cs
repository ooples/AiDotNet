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

namespace AiDotNet.VisionLanguage.Robotics;

/// <summary>
/// 3D-VLA: connects vision-language-action to 3D world via generative world model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// 3D-VLA (UMass, 2024) connects vision-language-action models to the 3D world via a generative
/// world model. It processes 3D point cloud observations alongside language instructions and
/// generates both predicted future states (as a world model) and robot actions, enabling
/// planning through imagination in 3D space.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "3D-VLA: A 3D Vision-Language-Action Generative World Model (UMass, 2024)"</item></list></para>
/// <para><b>For Beginners:</b> 3D-VLA is a vision-language-action model that uses 3D world
/// modeling for robot planning and action generation. Default values follow the original
/// paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a 3D-VLA model for 3D vision-language-action reasoning
/// // connecting VLA to 3D world via a generative world model
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new ThreeDVLA&lt;double&gt;(architecture, "threedvla.onnx");
///
/// // Training mode with native layers
/// var trainModel = new ThreeDVLA&lt;double&gt;(architecture, new ThreeDVLAOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.Robotics)]
[ModelDomain(ModelDomain.ThreeD)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "3D-VLA: A 3D Vision-Language-Action Generative World Model",
    "https://arxiv.org/abs/2403.09631",
    Year = 2024,
    Authors = "Zhen et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-5, Beta1 = 0.9, Beta2 = 0.999,
                WeightDecay = 0.05, WarmupSteps = 1000, MinLearningRate = 1e-6,
                Schedule = LearningRateSchedulerType.CosineAnnealing,
                Source = "Zhen et al. 2024, Sec. 4: AdamW with beta1 0.9, beta2 0.999 and a weight "
                        + "decay of 0.05, under a linear warmup over the initial 1K steps rising from "
                        + "1e-8 to 1e-5, followed by a cosine decay to a minimum learning rate of 1e-6. "
                        + "The 1e-5 warmup endpoint is the peak rate.")]
public partial class ThreeDVLA<T> : VisionLanguageModelBase<T>, IVisionLanguageAction<T>
{
    private readonly ThreeDVLAOptions _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public ThreeDVLA(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        ThreeDVLAOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new ThreeDVLAOptions();
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

    public ThreeDVLA(
        NeuralNetworkArchitecture<T> architecture,
        ThreeDVLAOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new ThreeDVLAOptions();
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
    /// Generates from image using 3D-VLA's generative world model approach.
    /// The VLA processes visual observation and language instruction, projecting into a 3D-aware
    /// latent space. The world model predicts next state given action candidates, selecting actions
    /// that move toward the goal described by the instruction.
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
    /// Predicts action using 3D-VLA's generative world model approach. Per the
    /// paper (UMass 2024), 3D-VLA connects a VLA model to a 3D generative world
    /// model. The approach: (1) encode the visual observation, (2) project into
    /// a 3D-aware latent space (WorldModelDim), (3) use the world model to
    /// predict the next state given an action candidate, (4) select actions
    /// that move the predicted state toward the goal described by the instruction.
    /// The 3D world model operates in a learned latent space where spatial
    /// transformations (translation, rotation) correspond to robot actions.
    /// </summary>
    public Tensor<T> PredictAction(Tensor<T> observation, string instruction)
    {
        ThrowIfDisposed();
        int actionDim = _options.ActionDimension;
        int horizon = _options.PredictionHorizon;
        int worldDim = _options.WorldModelDim;

        // Step 1: Encode visual observation
        var visualFeatures = EncodeImage(observation);
        int dim = visualFeatures.Length;

        // Step 2: Encode instruction (goal specification)
        var instrTokens = TokenizeText(instruction);
        int instrLen = instrTokens.Length;

        // Step 3: Project to 3D world model latent space
        var engine = Engine;
        int blockSize = Math.Max(1, dim / worldDim);
        var worldLatent = BlockMeans(visualFeatures, worldDim, blockSize);

        // Step 4: Compute goal representation from instruction: tanh(instr[w % instrLen] / 100).
        var goalLatent = instrLen > 0
            ? engine.TensorTanh(engine.TensorDivideScalar(CyclicTake(instrTokens, worldDim), NumOps.FromDouble(100.0)))
            : new Tensor<T>([worldDim]);

        // Step 5: Process through VLM decoder for action reasoning.
        // Condition on world model state and goal difference: vis[d] + 0.2 (goal - world)[d % worldDim].
        var goalDiff = engine.TensorSubtract(goalLatent, worldLatent);
        var condInput = engine.TensorAdd(
            visualFeatures.Reshape([dim]),
            engine.TensorMultiplyScalar(CyclicTake(goalDiff, dim), NumOps.FromDouble(0.2)));

        var decoderOut = condInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            decoderOut = Layers[i].Forward(decoderOut);

        // Step 6: World model-guided action generation
        // Generate actions that move the predicted world state toward the goal
        int totalActions = actionDim * horizon;
        if (actionDim == 0 && horizon > 0)
            throw new DivideByZeroException("The world model update maps state dimensions onto zero action dimensions.");
        if (totalActions == 0)
            return new Tensor<T>([totalActions]);

        // Decoder output conditioning 0.1 decoderOut[min(actionIdx, length - 1)] of every action.
        int decoderLength = decoderOut.Length;
        if (decoderLength == 0)
            throw new IndexOutOfRangeException("The decoder output has no features to condition the actions on.");
        var decoderTerm = engine.TensorMultiplyScalar(
            engine.TensorTake(
                decoderOut.Reshape([decoderLength]),
                engine.TensorMin(engine.TensorArange<int>(0, totalActions, 1), decoderLength - 1)),
            NumOps.FromDouble(0.1));

        // Scale by remaining distance (larger steps when far from goal): 1 - step / horizon.
        var remainingFraction = engine.ScalarMinusTensor(
            NumOps.One,
            engine.TensorDivideScalar(engine.TensorArange<T>(0, horizon, 1), NumOps.FromDouble(horizon)));
        int wBlockSize = Math.Max(1, worldDim / actionDim);
        T stateRate = NumOps.FromDouble(0.1);
        T epsilon = NumOps.FromDouble(1e-8);

        // Current world state (evolves as we predict actions)
        var currentState = worldLatent;
        var stepActions = new List<Tensor<T>>(horizon);

        for (int step = 0; step < horizon; step++)
        {
            // Compute state-to-goal direction in world model latent space and normalize it
            // by its L2 norm + 1e-8.
            var direction = engine.TensorSubtract(goalLatent, currentState);
            var dirNorm = engine.TensorAddScalar(
                engine.TensorSqrt(engine.ReduceSum(engine.TensorMultiply(direction, direction), [0], true)), epsilon);
            direction = engine.TensorDivide(direction, engine.TensorBroadcastTo(dirNorm, [worldDim]));

            // Map world model direction to robot action space
            // The 7 DOF action maps to: 3 translation + 3 rotation + 1 gripper.
            // Each action is the mean of its block of world model dimensions, scaled by the
            // remaining distance, plus the decoder output conditioning.
            var actionVal = engine.TensorAdd(
                engine.TensorMultiply(BlockMeans(direction, actionDim, wBlockSize), Pick(remainingFraction, step)),
                engine.TensorSlice(decoderTerm, [step * actionDim], [actionDim]));
            stepActions.Add(actionVal);

            // Step 7: World model forward prediction - update current state
            // Apply predicted action's effect on world state: state[w] += 0.1 action[w % actionDim].
            currentState = engine.TensorAdd(currentState, engine.TensorMultiplyScalar(CyclicTake(actionVal, worldDim), stateRate));
        }

        // Step 8: Package result with tanh bounding
        var result = engine.TensorTanh(engine.TensorConcatenate(stepActions.ToArray(), 0));

        return result;
    }

    /// <summary>
    /// The mean of each block b = [b * blockSize, min(b * blockSize + blockSize, n)) of a flat
    /// sequence, or 0 for a block that starts past the end.
    /// </summary>
    /// <param name="values">The sequence (read flattened).</param>
    /// <param name="blocks">Number of blocks.</param>
    /// <param name="blockSize">Block width (at least 1).</param>
    /// <returns>A [blocks] tensor of block means.</returns>
    private Tensor<T> BlockMeans(Tensor<T> values, int blocks, int blockSize)
    {
        var engine = Engine;
        int n = values.Length;
        if (n == 0 || blocks == 0)
            return new Tensor<T>([blocks]);

        var positions = engine.TensorAdd(
            engine.TensorBroadcastTo(
                engine.TensorMultiplyScalar(engine.TensorArange<int>(0, blocks, 1), blockSize).Reshape([blocks, 1]),
                [blocks, blockSize]),
            engine.TensorBroadcastTo(engine.TensorArange<int>(0, blockSize, 1).Reshape([1, blockSize]), [blocks, blockSize]));
        var inBlock = engine.TensorLessThan(positions, n).Cast<T>();
        var blockValues = engine.TensorWhere(
            inBlock,
            engine.TensorTake(values.Reshape([n]), engine.TensorMin(positions, n - 1)),
            new Tensor<T>([blocks, blockSize]));
        return engine.TensorDivide(
            engine.ReduceSum(blockValues, [1], false),
            engine.TensorMax(engine.ReduceSum(inBlock, [1], false), NumOps.One));
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
            Name = _useNativeMode ? "3D-VLA-Native" : "3D-VLA-ONNX",
            Description =
                "3D-VLA: connects vision-language-action to 3D world via generative world model.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "3D-VLA";
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(ThreeDVLA<T>));
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
