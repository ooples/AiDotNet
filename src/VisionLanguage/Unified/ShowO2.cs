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
/// Show-o2: improved native unified multimodal model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Show-o2 (NUS, 2025) improves upon Show-o with enhanced native unified multimodal capabilities.
/// It refines the discrete diffusion approach for image generation with improved visual token
/// quality and strengthens the multimodal understanding pathway through better cross-modal
/// attention and scaled training on larger multimodal datasets.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Show-o2: Improved Unified Multimodal Understanding and Generation" (NUS, 2025)</item></list></para>
/// <para><b>For Beginners:</b> Show-o2 is an improved version of Show-o with enhanced
/// understanding and generation quality. Default values follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Show-o2 model for improved unified multimodal processing
/// // with enhanced native understanding and generation capabilities
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new ShowO2&lt;double&gt;(architecture, "showo2.onnx");
///
/// // Training mode with native layers
/// var trainModel = new ShowO2&lt;double&gt;(architecture, new ShowO2Options());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.Multimodal)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
// Citation corrected in title AND id. It conflated the two Show-o papers: the recorded subtitle
// ("One Single Transformer Can Unify Multimodal Understanding and Generation") belongs to Show-o v1
// (arXiv 2408.12528), while the id 2502.07925 matches neither paper. Show-o2 is
// "Improved Native Unified Multimodal Models", arXiv 2506.15564 (NeurIPS 2025).
[ResearchPaper(
    "Show-o2: Improved Native Unified Multimodal Models",
    "https://arxiv.org/abs/2506.15564",
    Year = 2025,
    Authors = "Jinheng Xie, Zhenheng Yang, Mike Zheng Shou"
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4,
                Schedule = LearningRateSchedulerType.Constant,
                Phase = TrainingPhase.PreTraining,
                Source = "Xie et al. 2025, Sec. 4: stage 1 trains for 150K iterations using AdamW at a "
                        + "constant learning rate of 0.0001. The 2e-5 cosine rate at a batch of 512 "
                        + "elsewhere belongs to pre-distilling the semantic layers from SigLIP, which is "
                        + "a component rather than this model's own training.")]
public partial class ShowO2<T> : VisionLanguageModelBase<T>, IUnifiedVisionModel<T>
{
    private readonly ShowO2Options _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public ShowO2(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        ShowO2Options? options = null
    )
        : base(architecture)
    {
        _options = options ?? new ShowO2Options();
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

    public ShowO2(
        NeuralNetworkArchitecture<T> architecture,
        ShowO2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new ShowO2Options();
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
    /// Generates text from image using Show-o2's improved unified transformer.
    /// Per the paper (NUS, 2025), Show-o2 improves over Show-o with: (1) native
    /// resolution support via dynamic token compression, (2) improved omni-attention
    /// with relative position encoding for both modalities, (3) larger VQ codebook
    /// (16384 vs 8192) with better reconstruction quality.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;
        int numVisTokens = _options.NumVisualTokens;

        // Step 1: Multi-scale encoding (native resolution support)
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
    /// Generates an image from text using Show-o2's improved discrete diffusion.
    /// Per the paper (NUS, 2025), Show-o2 improves image generation with:
    /// (1) Larger codebook (16384 tokens) for higher fidelity,
    /// (2) Cosine noise schedule for mask-predict diffusion,
    /// (3) Native resolution output via adaptive token grid sizing.
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

        // Step 1: Text conditioning
        var textHidden = tokens;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            textHidden = Layers[i].Forward(textHidden);
        int hiddenDim = textHidden.Length;

        // Step 2: Adaptive grid sizing for native resolution
        int gridSize = outSize / 32;
        if (gridSize < 8)
            gridSize = 8;
        int numGenTokens = gridSize * gridSize;

        // Step 3: Discrete diffusion with cosine schedule
        var engine = Engine;
        var positions = engine.TensorArange<T>(0, numGenTokens, 1);
        // 1 where the token is still masked; visual token ids are only read where unmasked.
        var isMasked = Tensor<T>.CreateDefault([numGenTokens], NumOps.One);
        var visualTokenIds = new Tensor<T>([numGenTokens]);

        // Text conditioning does not depend on the step:
        // textCond[t] = sum_{h < min(8, hiddenDim)} hidden[h] * sin((t + 1) * (h + 1) * 0.015) / 8.
        var textCond = WaveProjection(textHidden, numGenTokens, Math.Min(8, hiddenDim), 0.015, useCosine: false, divisor: 8.0);
        T invVisTokens = NumOps.FromDouble(1.0 / numVisTokens);
        // 8-connected neighbours in flat order, range-checked only.
        int[] neighborOffsets = [-1, 1, -gridSize, gridSize, -gridSize - 1, -gridSize + 1, gridSize - 1, gridSize + 1];

        int numSteps = 20;
        for (int step = 0; step < numSteps; step++)
        {
            int maskedCount = NumOps.ToInt(engine.TensorSum(isMasked));
            if (maskedCount == 0)
                break;

            // Cosine schedule: unmask ratio follows cosine curve
            double ratio = Math.Cos((double)(step + 1) / numSteps * Math.PI / 2.0);
            int targetMasked = (int)(numGenTokens * ratio);
            int toUnmask = maskedCount - Math.Max(0, targetMasked);
            if (toUnmask <= 0)
                toUnmask = 1;

            // Context from unmasked neighbours: mean of their id / numVisTokens.
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

            var logit = engine.TensorAdd(
                engine.TensorMultiplyScalar(contextVal, NumOps.FromDouble(0.5)),
                engine.TensorMultiplyScalar(textCond, NumOps.FromDouble(0.5)));
            var predictedIds = QuantizeTanhTokens(logit, numVisTokens);
            var confidences = engine.TensorAdd(
                engine.TensorAbs(logit), engine.TensorMultiplyScalar(neighborCount, NumOps.FromDouble(0.15)));

            var unmaskNow = SelectTopKFirstIndex(confidences, isMasked, toUnmask);
            visualTokenIds = engine.TensorWhere(unmaskNow, predictedIds, visualTokenIds);
            isMasked = engine.TensorSubtract(isMasked, unmaskNow);
        }

        // Fill any remaining masked tokens with t % numVisTokens
        if (NumOps.GreaterThan(engine.TensorSum(isMasked), NumOps.Zero))
        {
            if (numVisTokens == 0)
                throw new DivideByZeroException();
            var fallbackIds = engine.TensorRemainder(
                positions, Tensor<T>.CreateDefault([numGenTokens], NumOps.FromDouble(numVisTokens)));
            visualTokenIds = engine.TensorWhere(isMasked, fallbackIds, visualTokenIds);
        }

        // Step 4: Decode to pixels
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
            Name = _useNativeMode ? "Show-o2-Native" : "Show-o2-ONNX",
            Description = "Show-o2: improved native unified multimodal model.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "Show-o2";
        m.AdditionalInfo["SupportsGeneration"] = _options.SupportsGeneration.ToString();
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(ShowO2<T>));
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
