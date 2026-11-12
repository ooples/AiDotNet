using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.VideoLanguage;

/// <summary>
/// LLaVA-Video: synthetic dataset-trained video instruction model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// LLaVA-Video (ByteDance, 2024) is a video instruction-tuned model trained with synthetic
/// video-text datasets. It uses GPT-4V-generated video descriptions as training data to build
/// robust video understanding capabilities, handling video question answering, temporal
/// reasoning, and video captioning with high-quality instruction following.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Video Instruction Tuning With Synthetic Data" (ByteDance, 2024)</item></list></para>
/// <para><b>For Beginners:</b> LLaVA-Video is a video-language model trained with synthetic
/// data for video instruction following. Default values follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a LLaVA-Video model for video instruction following
/// // trained with synthetic video-text datasets for robust understanding
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new LLaVAVideo&lt;double&gt;(architecture, "llavavideo.onnx");
///
/// // Training mode with native layers
/// var trainModel = new LLaVAVideo&lt;double&gt;(architecture, new LLaVAVideoOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.Video)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Video Instruction Tuning With Synthetic Data",
    "https://arxiv.org/abs/2410.02713",
    Year = 2024,
    Authors = "Zhang et al."
)]
public partial class LLaVAVideo<T> : VisionLanguageModelBase<T>, IVideoLanguageModel<T>
{
    private readonly LLaVAVideoOptions _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public LLaVAVideo(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        LLaVAVideoOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new LLaVAVideoOptions();
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

    public LLaVAVideo(
        NeuralNetworkArchitecture<T> architecture,
        LLaVAVideoOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new LLaVAVideoOptions();
        Options = _options;
        if (architecture.InputType == InputType.FourDimensional && architecture.InputHeight > 0)
        {
            _options = new LLaVAVideoOptions(_options) { ImageSize = architecture.InputHeight };
            Options = _options;
        }
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
    public int MaxFrames => _options.MaxFrames;

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
    /// Generates from a single image using LLaVA-Video's instruction-tuned visual-text fusion.
    /// Trained on LLaVA-Video-178K synthetic data with temporal reasoning examples, the model
    /// processes visual features with instruction-conditioned cross-attention. For single images,
    /// the temporal reasoning adapts to spatial reasoning over image content.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;

        // Step 1: Vision encoder
        var encoderOut = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            encoderOut = Layers[i].Forward(encoderOut);
        int visLen = encoderOut.Length;

        // Step 2: Tokenize instruction prompt
        Tensor<T>? promptTokens = null;
        int promptLen = 0;
        if (prompt is not null)
        {
            promptTokens = TokenizeText(prompt);
            promptLen = promptTokens.Length;
        }

        // Step 3: Instruction-conditioned visual-text cross-attention
        // 2-layer MLP cross-modal connector (Linear -> GELU -> Linear)
        var decoderInput = GatedConnectorInput(encoderOut, promptTokens, dim, _options.VocabSize);

        // Step 4: LLM decoder
        var output = decoderInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);
        return output;
    }

    /// <summary>
    /// Generates output from video frames using LLaVA-Video's temporal token pooling with
    /// cross-frame attention weighting. Per the paper (ByteDance 2024), frame features are
    /// pooled using a learned importance weighting based on inter-frame similarity: frames
    /// that are more distinct from their neighbors receive higher weight, emphasizing scene
    /// transitions and action boundaries. The model is trained on LLaVA-Video-178K synthetic
    /// data which specifically includes temporal reasoning examples (action ordering, duration
    /// estimation, causal reasoning).
    /// </summary>
    public Tensor<T> GenerateFromVideo(IReadOnlyList<Tensor<T>> frames, string? prompt = null)
    {
        ThrowIfDisposed();
        int count = Math.Min(frames.Count, _options.MaxFrames);
        if (count == 0)
            throw new ArgumentException("At least one frame is required.", nameof(frames));

        // Step 1: Encode all frames
        var frameFeatures = new Tensor<T>[count];
        for (int f = 0; f < count; f++)
            frameFeatures[f] = EncodeImage(frames[f]);

        int dim = frameFeatures[0].Length;

        // Step 2: Compute inter-frame similarity for importance weighting
        // Frames that differ from their neighbors are more informative (scene changes, actions)
        var engine = Engine;
        var stacked = StackLeading(frameFeatures, count, dim);
        Tensor<T> frameWeights;
        if (count == 1)
        {
            frameWeights = Tensor<T>.CreateDefault([1], NumOps.One);
        }
        else
        {
            // rms[f] = sqrt(mean_d (x[f] - x[f + 1])^2) between consecutive frames.
            var diff = engine.TensorSubtract(
                engine.TensorSlice(stacked, [0, 0], [count - 1, dim]), engine.TensorSlice(stacked, [1, 0], [count - 1, dim]));
            var rms = engine.TensorSqrt(engine.TensorMultiplyScalar(
                engine.ReduceSum(engine.TensorMultiply(diff, diff), [1], false), NumOps.FromDouble(1.0 / dim)));
            // Each frame averages its distance to the previous and the next frame.
            var zero = new Tensor<T>([1]);
            var one = Tensor<T>.CreateDefault([1], NumOps.One);
            var ones = Tensor<T>.CreateDefault([count - 1], NumOps.One);
            var dissimilarity = engine.TensorAdd(
                engine.TensorConcatenate([zero, rms], 0), engine.TensorConcatenate([rms, zero], 0));
            var neighborCount = engine.TensorAdd(
                engine.TensorConcatenate([zero, ones], 0), engine.TensorConcatenate([ones, zero], 0));
            frameWeights = engine.TensorDivide(dissimilarity, neighborCount);
        }

        // Softmax normalization of weights (NaN weights are skipped when finding the max).
        var negInf = Tensor<T>.CreateDefault([count], NumOps.FromDouble(double.NegativeInfinity));
        T maxWeight = engine.TensorMaxValue(engine.TensorWhere(engine.TensorIsNan(frameWeights), negInf, frameWeights));
        frameWeights = engine.TensorExp(engine.TensorSubtractScalar(frameWeights, maxWeight));
        frameWeights = engine.TensorDivideScalar(frameWeights, engine.TensorSum(frameWeights));

        // Step 3: Weighted temporal pooling
        var pooled = engine.TensorMatMul(frameWeights.Reshape([1, count]), stacked).Reshape([dim]);

        // Step 4: Decode through LLM layers
        var output = pooled;
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
                LayerHelper<T>.CreateDefaultVideoTemporalVLMLayers(
                    _options.VisionDim,
                    _options.VisionDim,
                    _options.DecoderDim,
                    _options.NumVisionLayers,
                    2,
                    _options.NumDecoderLayers,
                    _options.NumHeads,
                    _options.DropoutRate,
                    imageHeight: _options.ImageSize,
                    imageWidth: _options.ImageSize,
                    imageChannels: _options.ImageChannels,
                    patchSize: _options.PatchSize
                )
            );
            ComputeEncoderDecoderBoundary();
        }
    }

    private void ComputeEncoderDecoderBoundary()
    {
        // Encoder/decoder boundary mirrors the layer layout produced by
        // LayerHelper<T>.CreateDefaultVideoTemporalVLMLayers:
        //   PatchEmbedding(1) + InitialNorm(1)
        //   + (NumVisionLayers blocks × lpb layers/block)   ← vision encoder
        //   + (2 temporal blocks × lpb layers/block)        ← temporal encoder
        //   + ProjectionMLP(2)                              ← end of encoder
        //   + (NumDecoderLayers blocks × lpb layers/block)  ← LLM decoder
        // lpb (layers-per-block) is 5 (Norm+Attn+Norm+FFN1+FFN2) or 6 with dropout.
        const int patchEmbedLayers = 1;
        const int initialNormLayer = 1;
        const int temporalBlocks = 2;
        const int projectionLayers = 2;
        int lpb = _options.DropoutRate > 0 ? 6 : 5;
        _encoderLayerEnd =
            patchEmbedLayers
            + initialNormLayer
            + (_options.NumVisionLayers * lpb)
            + (temporalBlocks * lpb)
            + projectionLayers;
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
            Name = _useNativeMode ? "LLaVA-Video-Native" : "LLaVA-Video-ONNX",
            Description = "LLaVA-Video: synthetic dataset-trained video instruction model.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "LLaVA-Video";
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(LLaVAVideo<T>));
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
