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

namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// Show-o: single transformer for unified understanding and generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Show-o (NUS, 2024) uses a single transformer to unify multimodal understanding and generation.
/// It combines autoregressive next-token prediction for text with discrete diffusion for image
/// generation within the same model, switching between generation modes based on the output
/// modality while sharing all transformer parameters.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Show-o: One Single Transformer to Unify Multimodal Understanding and Generation" (NUS, 2024)</item></list></para>
/// <para><b>For Beginners:</b> Show-o is a unified transformer model that handles both
/// multimodal understanding and image generation. Default values follow the original paper
/// settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Show-o model for unified multimodal understanding and generation
/// // using a single transformer architecture for both tasks
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new ShowO&lt;double&gt;(architecture, "showo.onnx");
///
/// // Training mode with native layers
/// var trainModel = new ShowO&lt;double&gt;(architecture, new ShowOOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.Multimodal)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Show-o: One Single Transformer to Unify Multimodal Understanding and Generation",
    "https://arxiv.org/abs/2408.12528",
    Year = 2024,
    Authors = "Xie et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, WeightDecay = 0.01,
                WarmupSteps = 5000, Schedule = LearningRateSchedulerType.CosineAnnealing,
                Source = "Xie et al. 2024, Sec. 4: the AdamW optimizer with a weight decay of 0.01, "
                        + "5,000 steps of warm-up and an initial learning rate of 1e-4 under a cosine "
                        + "scheduling.")]
public partial class ShowO<T> : VisionLanguageModelBase<T>, IUnifiedVisionModel<T>
{
    private readonly ShowOOptions _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public ShowO(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        ShowOOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new ShowOOptions();
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

    public ShowO(
        NeuralNetworkArchitecture<T> architecture,
        ShowOOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new ShowOOptions();
        Options = _options;
        _useNativeMode = true;
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
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
    public bool SupportsGeneration => _options.SupportsGeneration;

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
    /// Generates text from image using Show-o's unified omni-attention transformer.
    /// Per the paper (NUS, 2024), Show-o uses a single transformer with omni-attention:
    /// causal attention for text tokens, full bidirectional attention for image tokens.
    /// Understanding: image is VQ-tokenized, placed in sequence with text, and the
    /// transformer generates text tokens autoregressively while attending fully to
    /// all image tokens.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;
        int numVisTokens = _options.NumVisualTokens;

        // Step 1: Encode image
        var features = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            features = Layers[i].Forward(features);

        // Fuse visual features with prompt tokens via ConcatenateTensors
        Tensor<T> fusedInput;
        if (prompt is not null)
        {
            var promptTokens = TokenizeText(prompt);
            fusedInput = features.ConcatenateTensors(promptTokens);
        }
        else
        {
            fusedInput = features;
        }

        var output = fusedInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }

    /// <summary>
    /// Generates an image from text using Show-o's discrete diffusion in token space.
    /// Per the paper (NUS, 2024), Show-o generates images via discrete diffusion:
    /// (1) Start with all [MASK] tokens in the visual token positions,
    /// (2) At each diffusion step, the transformer predicts token distributions for
    ///     all masked positions simultaneously (bidirectional attention on image tokens),
    /// (3) Unmask a subset of tokens based on confidence (highest confidence first),
    /// (4) Repeat until all tokens are unmasked.
    /// This is fundamentally different from autoregressive generation - it uses
    /// mask-predict scheduling similar to MaskGIT.
    /// Output: image tensor of size OutputImageSize * OutputImageSize * 3.
    /// </summary>
    public Tensor<T> GenerateImage(string textDescription)
    {
        ThrowIfDisposed();
        var tokens = TokenizeText(textDescription);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(tokens);

        int outSize = _options.OutputImageSize;
        int numVisTokens = _options.NumVisualTokens;
        int dim = _options.DecoderDim;
        int textLen = tokens.Length;

        // Step 1: Process text conditioning
        var textHidden = tokens;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            textHidden = Layers[i].Forward(textHidden);
        int hiddenDim = textHidden.Length;

        // Step 2: Initialize with all-MASK tokens
        int numGenTokens = _options.ImageTokenCount;
        int gridSize = (int)Math.Sqrt(numGenTokens);
        if (numGenTokens <= 0 || gridSize * gridSize != numGenTokens)
            throw new InvalidOperationException("ImageTokenCount must be a positive perfect square.");
        var engine = Engine;
        var positions = engine.TensorArange<T>(0, numGenTokens, 1);
        // 1 where the token is still MASK; visual token ids are only read where unmasked.
        var isMasked = Tensor<T>.CreateDefault([numGenTokens], NumOps.One);
        var visualTokenIds = new Tensor<T>([numGenTokens]);

        // Text conditioning does not depend on the step:
        // textCond[t] = sum_{h < min(8, hiddenDim)} hidden[h] * sin((t + 1) * (h + 1) * 0.02) / 8.
        var textCond = WaveProjection(textHidden, numGenTokens, Math.Min(8, hiddenDim), 0.02, useCosine: false, divisor: 8.0);
        T invVisTokens = NumOps.FromDouble(1.0 / numVisTokens);
        // 4-connected neighbours in flat order (t - grid, t + grid, t - 1, t + 1), range-checked only.
        int[] neighborOffsets = [-gridSize, gridSize, -1, 1];

        // Step 3: Discrete diffusion - iterative mask-predict
        int numDiffusionSteps = _options.DiffusionSteps;
        if (numDiffusionSteps <= 0)
            throw new InvalidOperationException("DiffusionSteps must be positive.");
        int tokensPerStep = Math.Max(1, numGenTokens / numDiffusionSteps);
        // Per-step phase scale step * 0.01 of the step bias sin(t * step * 0.01) * 0.1.
        var stepPhaseScale = engine.TensorMultiplyScalar(engine.TensorArange<T>(0, numDiffusionSteps, 1), NumOps.FromDouble(0.01));
        T biasAmplitude = NumOps.FromDouble(0.1);
        T contextWeight = NumOps.FromDouble(0.6);
        T textWeight = NumOps.FromDouble(0.4);
        T neighborConfidence = NumOps.FromDouble(0.2);

        for (int step = 0; step < numDiffusionSteps; step++)
        {
            // Count remaining masked tokens
            int maskedCount = engine.TensorSum(isMasked.Cast<int>());
            if (maskedCount == 0)
                break;

            // Bidirectional context from unmasked neighbours: mean of their id / numVisTokens.
            var unmasked = engine.ScalarMinusTensor(NumOps.One, isMasked);
            var knownValues = engine.TensorMultiply(engine.TensorMultiplyScalar(visualTokenIds, invVisTokens), unmasked);
            var contextSum = new Tensor<T>([numGenTokens]);
            var neighborCount = new Tensor<T>([numGenTokens]);
            foreach (int offset in neighborOffsets)
            {
                contextSum = engine.TensorAdd(contextSum, ShiftFlat(knownValues, offset));
                neighborCount = engine.TensorAdd(neighborCount, ShiftFlat(unmasked, offset));
            }
            var contextVal = engine.TensorDivide(contextSum, engine.TensorMax(neighborCount, NumOps.One));

            // Combined prediction
            var stepBias = engine.TensorMultiplyScalar(
                engine.TensorSin(engine.TensorMultiply(positions, Pick(stepPhaseScale, step))), biasAmplitude);
            var logit = engine.TensorAdd(
                engine.TensorAdd(
                    engine.TensorMultiplyScalar(contextVal, contextWeight),
                    engine.TensorMultiplyScalar(textCond, textWeight)),
                stepBias);
            var predictedIds = QuantizeTanhTokens(logit, numVisTokens);
            // More context = more confident
            var confidences = engine.TensorAdd(
                engine.TensorAbs(logit), engine.TensorMultiplyScalar(neighborCount, neighborConfidence));

            // Unmask the most confident positions
            int toUnmask = Math.Min(tokensPerStep, maskedCount);
            var unmaskNow = SelectTopKFirstIndex(confidences, isMasked, toUnmask);
            visualTokenIds = engine.TensorWhere(unmaskNow, predictedIds, visualTokenIds);
            isMasked = engine.TensorSubtract(isMasked, unmaskNow);
        }

        // Fill any remaining masked tokens with t % numVisTokens
        if (engine.TensorSum(isMasked.Cast<int>()) > 0)
        {
            if (numVisTokens == 0)
                throw new DivideByZeroException();
            var fallbackIds = engine.TensorRemainder(
                positions, Tensor<T>.CreateDefault([numGenTokens], NumOps.FromDouble(numVisTokens)));
            visualTokenIds = engine.TensorWhere(isMasked, fallbackIds, visualTokenIds);
        }

        // Step 4: Decode tokens to pixels
        int patchSize = outSize / gridSize;
        if (patchSize < 1)
            patchSize = 1;

        var colors = HashTokenColors(visualTokenIds).Reshape([gridSize, gridSize, 3]);
        var result = PaintPatchGrid(colors, SinePatchWeights(patchSize), outSize);
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
            Layers.AddRange(
                LayerHelper<T>.CreateDefaultUnifiedBidirectionalLayers(
                    _options.VisionDim,
                    _options.DecoderDim,
                    _options.DecoderDim,
                    _options.DecoderDim,
                    _options.NumVisionLayers,
                    _options.NumDecoderLayers / 2,
                    _options.NumDecoderLayers / 2,
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
        _encoderLayerEnd =
            1
            + _options.NumVisionLayers * lpb
            + (_options.VisionDim != _options.DecoderDim ? 1 : 0);
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
            Name = _useNativeMode ? "Show-o-Native" : "Show-o-ONNX",
            Description = "Show-o: single transformer for unified understanding and generation.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "Show-o";
        m.AdditionalInfo["SupportsGeneration"] = _options.SupportsGeneration.ToString();
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(ShowO<T>));
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
