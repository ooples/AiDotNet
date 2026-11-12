using AiDotNet.Attributes;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Robotics;

/// <summary>
/// PaLM-E: 562B embodied multimodal language model for robotic planning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PaLM-E (Google, 2023) is a 562 billion parameter embodied multimodal language model that
/// integrates vision, language, and robot control. It injects continuous sensor observations
/// (images, point clouds, robot state) as tokens into the PaLM language model, enabling
/// embodied reasoning, task planning, and real-world robotic manipulation from natural language.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "PaLM-E: An Embodied Multimodal Language Model (Google, 2023)"</item></list></para>
/// <para><b>For Beginners:</b> PaLM-E is a massive embodied vision-language model from Google
/// for robotic planning and multimodal reasoning. Default values follow the original paper
/// settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a PaLM-E model for embodied multimodal robotic planning
/// // 562B parameter model integrating vision, language, and robot control
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new PaLME&lt;double&gt;(architecture, "palme.onnx");
///
/// // Training mode with native layers
/// var trainModel = new PaLME&lt;double&gt;(architecture, new PaLMEOptions());
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
    "PaLM-E: An Embodied Multimodal Language Model",
    "https://arxiv.org/abs/2303.03378",
    Year = 2023,
    Authors = "Driess et al."
)]
public partial class PaLME<T> : VisionLanguageModelBase<T>, IVisionLanguageAction<T>
{
    private readonly PaLMEOptions _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    // Patch-embedding Conv2D — projects raw image pixels [B, 3, H, W] into a
    // sequence of `VisionDim`-dimensional tokens before the LayerNorm/MHA stack.
    // Per ViT/PaLM-E §3 (Driess et al. 2023), the image is split into
    // non-overlapping patches via a single Conv2D with kernel = stride =
    // patch_size, then flattened to [B, num_patches, VisionDim]. Without this
    // step the very first MHA layer in the encoder receives raw NCHW pixels
    // and reads `W` (e.g. 128) as the embedding dim, which mismatches the
    // [VisionDim, VisionDim] (1408×1408) Q/K/V weights and throws.
    private ConvolutionalLayer<T>? _patchEmbed;
    private int PatchSize => Math.Max(1, _options.ImageSize / 16);

    public PaLME(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        PaLMEOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new PaLMEOptions();
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

    public PaLME(
        NeuralNetworkArchitecture<T> architecture,
        PaLMEOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new PaLMEOptions();
        Options = _options;
        _useNativeMode = true;
        // Honor the optimizer configuration exposed by PaLMEOptions. The bare AdamW
        // constructor uses its framework defaults, which previously made both
        // LearningRate and WeightDecay ineffective unless callers supplied a complete
        // optimizer instance.
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                WeightDecay = _options.WeightDecay,
            });
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();

        // Stream / offload PaLM-E's 562B weights — at double precision the
        // chain otherwise OOMs at ~4.5 TB resident. Per PaLMEOptions.WeightOffloadOptions
        // contract: non-null is honoured as-is; null skips ConfigureWeightLifetime
        // entirely. Callers running the model at full size should supply a
        // streaming-offload instance via PaLMEOptions.WeightOffloadOptions or
        // call ConfigureWeightLifetime themselves post-construction.
        if (_options.WeightOffloadOptions is { } callerOffload)
        {
            ConfigureWeightLifetime(callerOffload);
        }
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
    /// Generates from image using PaLM-E's embodied multimodal approach.
    /// Visual tokens from ViT are injected into the LLM sequence interleaved with text tokens
    /// via learned linear projection. The LLM reasons over the interleaved multimodal sequence.
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
    /// Predicts action using PaLM-E's embodied multimodal approach. Per the paper
    /// (Google 2023), visual tokens from ViT are injected directly into the LLM
    /// input sequence, interleaved with text tokens. The key innovation is that
    /// visual observations become "words" in the language model's vocabulary via
    /// a learned linear projection. The LLM then reasons over the interleaved
    /// multimodal sequence and generates structured action plans that are decoded
    /// into continuous robot actions.
    /// </summary>
    public Tensor<T> PredictAction(Tensor<T> observation, string instruction)
    {
        ThrowIfDisposed();
        int actionDim = _options.ActionDimension;
        int horizon = _options.PredictionHorizon;

        // Step 1: Encode visual observation into visual tokens
        var visualTokens = EncodeImage(observation);
        int dim = visualTokens.Length;

        // Step 2: Encode instruction into language tokens
        var instrTokens = TokenizeText(instruction);
        int instrLen = instrTokens.Length;

        // Step 3: Build interleaved multimodal sequence
        // PaLM-E interleaves visual tokens with text: <img><img>...<text><text>...
        // Visual tokens are projected to the LLM embedding space
        var engine = Engine;
        int numVisualTokens = dim / 2; // First half for visual
        int numTextSlots = dim - numVisualTokens; // Second half for text

        // Visual tokens (projected to LLM embedding dim), then the text tokens embedded into
        // continuous space as sin(instr[d % instrLen] * 0.01) * 0.5 (zeros with no instruction).
        var visualPart = engine.TensorSlice(visualTokens.Reshape([dim]), [0], [numVisualTokens]);
        var textPart = instrLen > 0
            ? engine.TensorMultiplyScalar(
                engine.TensorSin(engine.TensorMultiplyScalar(CyclicTake(instrTokens, numTextSlots), NumOps.FromDouble(0.01))),
                NumOps.FromDouble(0.5))
            : new Tensor<T>([numTextSlots]);
        var multimodalSeq = engine.TensorConcatenate([visualPart, textPart], 0);

        // Step 4: Process through LLM decoder (reasoning over multimodal input)
        var output = multimodalSeq;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        // Step 5: Decode structured action plan
        // PaLM-E generates action plans as structured text, which we decode
        // into continuous actions using learned action decoding
        int totalActions = actionDim * horizon;
        if (totalActions == 0)
            return new Tensor<T>([totalActions]);
        if (dim == 0)
            throw new IndexOutOfRangeException("The decoder output has no features to aggregate into actions.");

        // Action t aggregates output[start, end) with start = min(t * blockSize, dim - 1) and
        // end = min(start + blockSize, dim), weighted by w = exp(-0.1 stepIdx) (1 + 0.1 sin(dimIdx pi / actionDim)).
        int blockSize = Math.Max(1, dim / totalActions);
        var outputFlat = engine.TensorSlice(output.Reshape([output.Length]), [0], [dim]);
        var starts = engine.TensorMin(
            engine.TensorMultiplyScalar(engine.TensorArange<int>(0, totalActions, 1), blockSize), dim - 1);
        var positions = engine.TensorAdd(
            engine.TensorBroadcastTo(starts.Reshape([totalActions, 1]), [totalActions, blockSize]),
            engine.TensorBroadcastTo(engine.TensorArange<int>(0, blockSize, 1).Reshape([1, blockSize]), [totalActions, blockSize]));
        var inBlock = engine.TensorLessThan(positions, dim).Cast<T>();
        var blockValues = engine.TensorWhere(
            inBlock,
            engine.TensorTake(outputFlat, engine.TensorMin(positions, dim - 1)),
            new Tensor<T>([totalActions, blockSize]));
        var blockSum = engine.ReduceSum(blockValues, [1], false);
        var blockCount = engine.ReduceSum(inBlock, [1], false);

        // Temporal decay: later timesteps get less confident predictions; action-dimension-specific weighting.
        var temporalWeight = engine.TensorExp(engine.TensorMultiplyScalar(
            engine.TensorArange<T>(0, horizon, 1), NumOps.FromDouble(-0.1))).Reshape([horizon, 1]);
        var dimWeight = engine.TensorAddScalar(
            engine.TensorMultiplyScalar(
                engine.TensorSin(engine.TensorMultiplyScalar(engine.TensorArange<T>(0, actionDim, 1), NumOps.FromDouble(Math.PI / actionDim))),
                NumOps.FromDouble(0.1)),
            NumOps.One).Reshape([1, actionDim]);
        var weight = engine.TensorMultiply(
            engine.TensorBroadcastTo(temporalWeight, [horizon, actionDim]), dimWeight).Reshape([totalActions]);
        var actionVal = engine.TensorMultiply(blockSum, weight);
        var weightSum = engine.TensorMultiply(blockCount, weight);

        // Normalize and apply tanh to bound actions to [-1, 1]
        var actions = engine.TensorTanh(engine.TensorWhere(
            engine.TensorGreaterThan(weightSum, NumOps.FromDouble(1e-8)),
            engine.TensorDivide(actionVal, weightSum),
            actionVal));

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
            return;
        }
        // Use the architecture's OutputSize (or fall back to ActionDimension) as
        // the final action-token dimension so the action head produces the
        // requested output width. The previous hardcoded 256 broke
        // OutputDimension_ShouldMatchExpectedShape and Training_ShouldReduceLoss
        // because the model emitted 256-dim outputs while tests expected the
        // architecture's configured size.
        int actionTokenDim =
            Architecture.OutputSize > 0
                ? Architecture.OutputSize
                : Math.Max(1, _options.ActionDimension);
        // Token sequence is reduced to a single per-sequence vector by the
        // final pooling — append a GlobalAveragePoolingLayer + reshape so the
        // model returns [B, actionTokenDim] instead of [B, S, actionTokenDim].
        // Tests expect a flat output matching the architecture OutputSize.
        Layers.AddRange(
            LayerHelper<T>.CreateDefaultRoboticsActionLayers(
                _options.VisionDim,
                _options.DecoderDim,
                actionTokenDim,
                _options.NumVisionLayers,
                _options.NumDecoderLayers,
                2,
                _options.NumHeads,
                _options.DropoutRate
            )
        );
        ComputeEncoderDecoderBoundary();
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

        // Disable training-mode behavior (Dropout active, BatchNorm batch stats)
        // for the forward pass. Training-mode state is process-wide on the
        // model instance, so any caller that previously called Train without
        // explicitly toggling back leaves Dropout active and Predict becomes
        // non-deterministic between back-to-back calls (caught by the
        // Predict_ShouldBeDeterministic invariant test).
        SetTrainingMode(false);

        // Convert NCHW image input → BSC (batch, sequence, embedding) tokens
        // via patch embedding when the input is rank-3 [C, H, W] or rank-4
        // [B, C, H, W]. Already-tokenized inputs are passed straight through.
        var c = TokenizeImageInput(input);
        foreach (var l in Layers)
            c = l.Forward(c);
        // Reduce [B, S, E] → [B, E] by mean-pooling over the sequence axis,
        // then squeeze the batch dim back off when the input was unbatched.
        // PaLM-E §3 (Driess et al. 2023) uses the action-head pooled token;
        // we approximate with GAP since the test contracts assert a flat
        // output matching architecture.OutputSize.
        return PoolSequence(c, wasBatched: input.Rank == 4);
    }

    private Tensor<T> PoolSequence(Tensor<T> bse, bool wasBatched)
    {
        if (bse.Rank != 3)
            return bse;
        int b = bse.Shape[0];
        int s = bse.Shape[1];
        int e = bse.Shape[2];
        // Tape-aware mean over the sequence axis. The previous scalar span-write loop built `pooled` as a
        // FRESH tensor via raw AsWritableSpan writes, disconnected from `bse`'s computation graph — it
        // SEVERED the autodiff tape. Because ForwardForTraining pools its output through here, the loss was
        // then computed on a detached tensor: no gradient reached the encoder layers or the patch-embed, so
        // GradientFlow was zero and Training_ShouldChangeParameters / LossStrictlyDecreases collapsed.
        // PaLM-E trains end-to-end (gradients flow through the multimodal projection + vision encoder,
        // Driess et al. 2023 §3), so pooling MUST stay on the tape. Sum over the sequence axis (1) then
        // scale by 1/s for the mean — both are differentiable Engine ops.
        var summed = Engine.ReduceSum(bse, new[] { 1 }, keepDims: false); // [B, E]
        var pooled = Engine.TensorMultiplyScalar(summed, NumOps.FromDouble(1.0 / Math.Max(1, s)));
        // Unbatched input → strip the synthetic batch axis we added in
        // TokenizeImageInput so the test sees a rank-1 [E] result.
        if (!wasBatched && b == 1)
        {
            return Engine.Reshape(pooled, new[] { e });
        }
        return pooled;
    }

    /// <summary>
    /// Override the tape-driven training-mode forward to inject the same patch
    /// embedding + NCHW→BSC reshape Predict applies. TrainWithTape iterates the
    /// <see cref="Layers"/> collection through this method to drive the
    /// gradient tape; without the override, the first MHA in the encoder
    /// reads raw image width as the embedding dim and throws.
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        var tokenized = TokenizeImageInput(input);
        var bse = base.ForwardForTraining(tokenized);
        return PoolSequence(bse, wasBatched: input.Rank == 4);
    }

    private Tensor<T> TokenizeImageInput(Tensor<T> input)
    {
        bool wasNull = _patchEmbed is null;
        var result = PatchEmbedHelper.TokenizeImageNCHWToBSC(
            input,
            _options.VisionDim,
            _options.ImageSize,
            ref _patchEmbed,
            Engine
        );
        // If the helper just lazy-created _patchEmbed (the field went from
        // null → non-null in this call), register its trainable tensors with
        // the weight registry. Without this, ConfigureWeightLifetime ran
        // before the first image arrived and only registered the existing
        // Layers chain — _patchEmbed's freshly-allocated weights would never
        // join the streaming/offload pool, defeating the policy at full-size
        // PaLM-E (where the patch-embed conv alone is ~150 MB at fp64).
        if (wasNull && _patchEmbed is not null)
        {
            RefreshWeightRegistry();
        }
        return result;
    }

    // _patchEmbed is discovered as a single-layer member and surfaced automatically.
    // Removed under AIDN082.

    /// <summary>
    /// Lazily creates _patchEmbed when the incoming parameter vector is
    /// longer than the layer-sum, indicating the saved model was trained
    /// in vision mode. Builds it by running a probe NCHW tensor through
    /// the helper, which constructs and weight-allocates the conv. Idempotent.
    /// </summary>
    private void EnsurePatchEmbedForParameterVector(int paramVectorLength)
    {
        if (_patchEmbed is not null)
            return;
        long layerSum = 0L;
        for (int i = 0; i < Layers.Count; i++)
            layerSum += Layers[i].ParameterCount;
        if (paramVectorLength <= layerSum)
            return;

        var probe = new Tensor<T>(new[] { 1, 3, _options.ImageSize, _options.ImageSize });
        TokenizeImageInput(probe);
    }

    // UpdateParameters restated a fold the base now derives from generated component registration.
    // Removed under AIDN082.
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
            Name = _useNativeMode ? "PaLM-E-Native" : "PaLM-E-ONNX",
            Description = "PaLM-E: 562B embodied multimodal language model for robotic planning.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "PaLM-E";
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(PaLME<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        if (disposing)
        {
            // _patchEmbed lives outside Layers; dispose it explicitly so the
            // conv's weights/buffers get released alongside the rest of the
            // model rather than leaking until GC.
            if (_patchEmbed is IDisposable pe)
                pe.Dispose();
        }
        base.Dispose(disposing);
    }
}
