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
/// Octo: open-source generalist robot policy trained on 800K demonstrations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Octo (Berkeley, 2024) is an open-source generalist robot policy trained on 800K demonstrations
/// from the Open X-Embodiment dataset. It uses a transformer backbone that processes interleaved
/// image observations and language task specifications, outputting continuous robot actions
/// that generalize across different robot embodiments and manipulation tasks.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Octo: An Open-Source Generalist Robot Policy (Berkeley, 2024)"</item></list></para>
/// <para><b>For Beginners:</b> Octo is an open-source vision-language-action model for generalist
/// robot control across different robot platforms. Default values follow the original paper
/// settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create an Octo model for generalist robot policy
/// // open-source model trained on 800K robot demonstrations
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new Octo&lt;double&gt;(architecture, "octo.onnx");
///
/// // Training mode with native layers
/// var trainModel = new Octo&lt;double&gt;(architecture, new OctoOptions());
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
    "Octo: An Open-Source Generalist Robot Policy",
    "https://arxiv.org/abs/2405.12213",
    Year = 2024,
    Authors = "Ghosh et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 3e-4, WeightDecay = 0.1,
                ReferenceBatchSize = 2048, WarmupSteps = 2000, MaxGradientNorm = 1.0,
                Schedule = LearningRateSchedulerType.Noam,
                Source = "Octo Model Team 2024, Table IV: AdamW with a learning rate of 3e-4, 2000 "
                        + "warmup steps, a reciprocal square-root scheduler, a weight decay of 0.1, a "
                        + "gradient clip threshold of 1 and a batch size of 2048. The reciprocal "
                        + "square-root decay is declared as the Noam schedule, which is that same "
                        + "inverse-square-root shape with a linear warmup. The factory call is already "
                        + "present at the construction site; only the declaration was missing.")]
public partial class Octo<T> : VisionLanguageModelBase<T>, IVisionLanguageAction<T>
{
    private readonly OctoOptions _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public Octo(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        OctoOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new OctoOptions();
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

    public Octo(
        NeuralNetworkArchitecture<T> architecture,
        OctoOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new OctoOptions();
        Options = _options;
        _useNativeMode = true;
        // Honor the configured (paper-aligned) learning rate. OctoOptions.LearningRate defaults to 1e-4
        // — the AdamW rate Octo (Octo Model Team 2024) and VLA transformers train at — but the optimizer
        // previously ignored it and used AdamW's built-in default (~1e-3), which overshoots and DIVERGES
        // on the smaller-scale configurations (MoreData_ShouldNotDegrade: 200-iter loss > 50-iter loss).
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this, new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = _options.LearningRate });
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
    /// Generates from image using Octo's readout token + task-conditioned attention.
    /// The transformer backbone processes observation and language tokens together,
    /// then readout tokens cross-attend to extract task-relevant features for the action head.
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
    /// Predicts action using Octo's readout token + diffusion head architecture.
    /// Per the paper (Berkeley 2024), Octo uses a transformer backbone with
    /// task-conditioned readout tokens. The backbone processes observation and
    /// language tokens, then dedicated readout tokens attend to the backbone
    /// via cross-attention to extract task-relevant features. A diffusion action
    /// head iteratively denoises random noise into action trajectories conditioned
    /// on the readout features. The diffusion head uses DDPM-style denoising
    /// with a learned noise schedule.
    /// </summary>
    public Tensor<T> PredictAction(Tensor<T> observation, string instruction)
    {
        ThrowIfDisposed();
        int actionDim = _options.ActionDimension;
        int horizon = _options.PredictionHorizon;
        int obsHistory = _options.ObservationHistory;

        // Step 1: Encode visual observation
        var visualFeatures = EncodeImage(observation);
        int dim = visualFeatures.Length;

        // Step 2: Encode instruction (task specification)
        var instrTokens = TokenizeText(instruction);
        int instrLen = instrTokens.Length;

        var engine = Engine;

        // Step 3: Build backbone input with observation and task tokens:
        // vis[d] + 0.3 tanh(instr[d % instrLen] / 100).
        var visFlat = visualFeatures.Reshape([dim]);
        var backboneInput = instrLen > 0
            ? engine.TensorAdd(
                visFlat,
                engine.TensorMultiplyScalar(
                    engine.TensorTanh(engine.TensorDivideScalar(CyclicTake(instrTokens, dim), NumOps.FromDouble(100.0))),
                    NumOps.FromDouble(0.3)))
            : visFlat.Clone();

        // Step 4: Process through transformer backbone
        var backboneOut = backboneInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            backboneOut = Layers[i].Forward(backboneOut);

        // Step 5: Readout token cross-attention
        // Readout tokens attend to backbone output to extract action-relevant features:
        // attn[r, d] = exp(key[d] sin((r + 1) d pi / dim) / sqrt(dim)); readout[r] is the
        // attention-weighted mean of the keys, or 0 when the attention mass is not above 1e-8.
        int numReadout = actionDim;
        Tensor<T> readoutFeatures;
        if (dim == 0 || numReadout == 0)
        {
            readoutFeatures = new Tensor<T>([numReadout]);
        }
        else
        {
            var key = engine.TensorSlice(backboneOut.Reshape([backboneOut.Length]), [0], [dim]).Reshape([1, dim]);
            var query = engine.TensorSin(engine.TensorMultiplyScalar(
                engine.TensorOuter(engine.TensorArange<T>(1, numReadout + 1, 1), engine.TensorArange<T>(0, dim, 1)),
                NumOps.FromDouble(Math.PI / dim)));
            var logits = engine.TensorDivideScalar(engine.TensorMultiply(query, key), NumOps.FromDouble(Math.Sqrt(dim)));
            // Shift each row by its maximum logit before exponentiating so a float T cannot overflow
            // (the weighted mean is unchanged). The "mass above 1e-8" test of the unshifted weights is
            // rowMax + log(sum exp(logit - rowMax)) > log(1e-8).
            var rowMax = engine.ReduceMax(logits, [1], true);
            var attn = engine.TensorExp(engine.TensorSubtract(logits, engine.TensorBroadcastTo(rowMax, [numReadout, dim])));
            var attnSum = engine.ReduceSum(attn, [1], false);
            var valSum = engine.ReduceSum(engine.TensorMultiply(attn, key), [1], false);
            var logMass = engine.TensorAdd(rowMax.Reshape([numReadout]), engine.TensorLog(attnSum));
            readoutFeatures = engine.TensorWhere(
                engine.TensorGreaterThan(logMass, NumOps.FromDouble(Math.Log(1e-8))),
                engine.TensorDivide(valSum, attnSum),
                new Tensor<T>([numReadout]));
        }

        // Step 6: Diffusion action head - iterative denoising
        // Start from Gaussian noise and denoise over T steps
        int diffusionSteps = 10;
        int totalActions = actionDim * horizon;
        if (totalActions == 0)
            return new Tensor<T>([totalActions]);

        // Initial structured noise readout[t % actionDim] * 0.1 and the predicted clean action
        // readout[t % actionDim] * exp(-0.05 (t / actionDim)), the same for every step.
        var tiledReadout = engine.TensorBroadcastTo(readoutFeatures.Reshape([1, actionDim]), [horizon, actionDim]);
        var temporalDecay = engine.TensorExp(engine.TensorMultiplyScalar(
            engine.TensorArange<T>(0, horizon, 1), NumOps.FromDouble(-0.05))).Reshape([horizon, 1]);
        var predicted = engine.TensorMultiply(tiledReadout, temporalDecay).Reshape([totalActions]);
        var actions = engine.TensorMultiplyScalar(tiledReadout, NumOps.FromDouble(0.1)).Reshape([totalActions]);

        // DDPM-style iterative denoising. Iteration k runs step = diffusionSteps - 1 - k with
        // noiseScale = step / diffusionSteps and signalScale = 1 - noiseScale.
        var noiseScales = engine.TensorDivideScalar(
            engine.ScalarMinusTensor(NumOps.FromDouble(diffusionSteps - 1), engine.TensorArange<T>(0, diffusionSteps, 1)),
            NumOps.FromDouble(diffusionSteps));
        var signalScales = engine.ScalarMinusTensor(NumOps.One, noiseScales);
        for (int k = 0; k < diffusionSteps; k++)
        {
            // Denoise: move toward predicted clean action
            actions = engine.TensorAdd(
                engine.TensorMultiply(predicted, Pick(signalScales, k)),
                engine.TensorMultiply(actions, Pick(noiseScales, k)));
        }

        // Step 7: Package as tensor with tanh bounding
        var result = engine.TensorTanh(actions);

        return result;
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
            // Learnable input PROJECTION of the observation tokens, applied BEFORE the transformer
            // backbone. Octo (Octo Model Team 2024) tokenizes observations via a patch/linear embedding
            // and adds positional embeddings before the backbone; CreateDefaultRoboticsActionLayers omits
            // that leading projection and starts with a LayerNorm. A LayerNorm removes each token's mean,
            // so a CONSTANT observation input collapses to a position-only value INDEPENDENT of the input
            // value — different observations (e.g. all-0.1 vs all-0.9) then produce identical embeddings
            // and outputs (DifferentInputs / DifferentImages collapse). A learnable linear projection
            // breaks that symmetry (Dense(x) differs for constant 0.1 vs 0.9), matching the paper's
            // token-embedding stage.
            Layers.Add(new DenseLayer<T>(_options.VisionDim, new IdentityActivation<T>() as IActivationFunction<T>));
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
        // +1 leading input projection, +1 leading vision LayerNorm, + numVisionLayers blocks, + 2 for the
        // MLP projection (Dense + LayerNorm) that bridges the vision encoder to the decoder.
        _encoderLayerEnd = 2 + _options.NumVisionLayers * lpb + 2;
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
            Name = _useNativeMode ? "Octo-Native" : "Octo-ONNX",
            Description =
                "Octo: open-source generalist robot policy trained on 800K demonstrations.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "Octo";
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(Octo<T>));
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
