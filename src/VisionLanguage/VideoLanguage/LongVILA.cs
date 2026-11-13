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
/// LongVILA: long-context visual language model for 1hr+ videos.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// LongVILA (NVIDIA, 2024) scales long-context visual language models for processing videos
/// exceeding one hour in duration. It uses multi-modal sequence parallelism to distribute
/// long video frame sequences across multiple GPUs and extends the context window to handle
/// thousands of frames for long-form video understanding and temporal reasoning.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "LongVILA: Scaling Long-Context Visual Language Models for Long Videos" (NVIDIA, 2024)</item></list></para>
/// <para><b>For Beginners:</b> LongVILA is a video-language model from NVIDIA for understanding
/// long videos of 1 hour or more. Default values follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a LongVILA model for long-context video understanding
/// // capable of processing 1hr+ videos with long-context attention
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new LongVILA&lt;double&gt;(architecture, "longvila.onnx");
///
/// // Training mode with native layers
/// var trainModel = new LongVILA&lt;double&gt;(architecture, new LongVILAOptions());
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
    "LongVILA: Scaling Long-Context Visual Language Models for Long Videos",
    "https://arxiv.org/abs/2408.10188",
    Year = 2024,
    Authors = "Xue et al."
)]
public partial class LongVILA<T> : VisionLanguageModelBase<T>, IVideoLanguageModel<T>
{
    private readonly LongVILAOptions _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public LongVILA(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        LongVILAOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new LongVILAOptions();
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

    public LongVILA(
        NeuralNetworkArchitecture<T> architecture,
        LongVILAOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new LongVILAOptions();
        Options = _options;
        if (architecture.InputType == InputType.FourDimensional && architecture.InputHeight > 0)
        {
            if (architecture.InputWidth > 0 && architecture.InputWidth != architecture.InputHeight)
                throw new ArgumentException(
                    "LongVILA native mode requires square frames (InputWidth must equal InputHeight).",
                    nameof(architecture)
                );
            if (architecture.InputDepth > 0 && architecture.InputDepth != 3)
                throw new ArgumentException(
                    "LongVILA native mode requires 3-channel RGB frames (InputDepth must equal 3).",
                    nameof(architecture)
                );
            _options = new LongVILAOptions(_options) { ImageSize = architecture.InputHeight };
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
    /// Generates from a single image using LongVILA's MM-SP single-chunk processing.
    /// For a single image, the Multi-Modal Sequence Parallelism reduces to a single chunk
    /// with local self-attention over visual features. Text tokens are fused via cross-attention
    /// to condition the generation. No hierarchical temporal attention needed for single frames.
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

        // Step 2: Tokenize prompt
        Tensor<T>? promptTokens = null;
        int promptLen = 0;
        if (prompt is not null)
        {
            promptTokens = TokenizeText(prompt);
            promptLen = promptTokens.Length;
        }

        // Step 3: Single-chunk local self-attention + text cross-attention
        // 2-layer MLP cross-modal connector (Linear -> GELU -> Linear)
        var decoderInput = GatedConnectorInput(encoderOut, promptTokens, dim, _options.VocabSize);

        // Step 4: LLM decoder
        var output = decoderInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);
        return output;
    }

    /// <summary>
    /// Generates output from video frames using LongVILA's chunked multi-modal sequence
    /// processing for long videos (1hr+). Per the paper (NVIDIA 2024), the system uses
    /// Multi-Modal Sequence Parallelism (MM-SP): video frames are divided into temporal
    /// chunks, each chunk is encoded independently, then chunk representations are aggregated
    /// using hierarchical temporal attention. Within each chunk, a local context window
    /// maintains fine-grained temporal details. Across chunks, a global summary captures
    /// long-range temporal dependencies. This enables processing of 1000+ frames.
    /// </summary>
    public Tensor<T> GenerateFromVideo(IReadOnlyList<Tensor<T>> frames, string? prompt = null)
    {
        ThrowIfDisposed();
        int count = Math.Min(frames.Count, _options.MaxFrames);
        if (count == 0)
            throw new ArgumentException("At least one frame is required.", nameof(frames));

        int dim = 0;

        // Step 1: Divide video into temporal chunks for MM-SP
        int chunkSize = 16; // Frames per chunk (local context window)
        int numChunks = (count + chunkSize - 1) / chunkSize;

        // Step 2: Process each chunk independently (local temporal processing)
        var engine = Engine;
        T attnFloor = NumOps.FromDouble(1e-8);
        var unitSum = Tensor<T>.CreateDefault([1], NumOps.One);
        var chunkSummaries = new Tensor<T>[numChunks];
        for (int c = 0; c < numChunks; c++)
        {
            int chunkStart = c * chunkSize;
            int chunkEnd = Math.Min(chunkStart + chunkSize, count);
            int chunkFrames = chunkEnd - chunkStart;

            // Local temporal attention within chunk: weight = exp(rms magnitude), normalized when the
            // weights sum above 1e-8, otherwise uniform.
            // Encode frames in this chunk; the feature width comes from the first chunk whose first
            // frame is non-empty.
            var chunkFeatures = new Tensor<T>[chunkFrames];
            for (int f = 0; f < chunkFrames; f++)
                chunkFeatures[f] = EncodeImage(frames[chunkStart + f]);
            if (dim == 0)
                dim = chunkFeatures[0].Length;

            var stacked = StackLeading(chunkFeatures, chunkFrames, dim);
            // magnitude / dim, with dim as a [1] tensor.
            var chunkDim = engine.TensorBroadcastTo(engine.TensorArange<T>(dim, dim + 1, 1), [chunkFrames]);
            var attnWeights = engine.TensorExp(engine.TensorSqrt(engine.TensorDivide(
                engine.ReduceSum(engine.TensorMultiply(stacked, stacked), [1], false), chunkDim)));
            var attnSum = engine.ReduceSum(attnWeights, [0], true);
            var normalize = engine.TensorBroadcastTo(engine.TensorGreaterThan(attnSum, attnFloor), [chunkFrames]);
            var normalized = engine.TensorDivide(
                attnWeights, engine.TensorBroadcastTo(engine.TensorWhere(engine.TensorGreaterThan(attnSum, attnFloor), attnSum, unitSum), [chunkFrames]));
            var ones = Tensor<T>.CreateDefault([chunkFrames], NumOps.One);
            var uniform = engine.TensorDivideScalar(ones, engine.TensorSum(ones));
            var weights = engine.TensorWhere(normalize, normalized, uniform);

            // Normalize and aggregate
            chunkSummaries[c] = engine.TensorMatMul(weights.Reshape([1, chunkFrames]), stacked);
        }

        T invDim = NumOps.FromDouble(1.0 / dim);

        // Step 3: Hierarchical temporal attention across chunks (global aggregation)
        // Each chunk summary attends to all other chunks based on temporal distance
        var summaries = engine.TensorConcatenate(chunkSummaries, 0);
        // V-shaped attention: attend more to beginning and end of video:
        // bias[c] = 1 + 0.5 * (cos(2 pi c / max(1, numChunks - 1)) + 1) / 2
        var temporalPos = engine.TensorMultiplyScalar(
            engine.TensorArange<T>(0, numChunks, 1), NumOps.FromDouble(Math.PI * 2.0 / Math.Max(1, numChunks - 1)));
        var positionalBias = engine.TensorAddScalar(
            engine.TensorMultiplyScalar(engine.TensorAddScalar(engine.TensorCos(temporalPos), NumOps.One), NumOps.FromDouble(0.25)),
            NumOps.One);
        var chunkMagnitude = engine.TensorSqrt(engine.TensorMultiplyScalar(
            engine.ReduceSum(engine.TensorMultiply(summaries, summaries), [1], false), invDim));
        var chunkAttnWeights = engine.TensorExp(engine.TensorMultiply(chunkMagnitude, positionalBias));
        T chunkAttnSum = engine.TensorSum(chunkAttnWeights);
        var chunkWeights = NumOps.GreaterThan(chunkAttnSum, NumOps.FromDouble(1e-8))
            ? engine.TensorDivideScalar(chunkAttnWeights, chunkAttnSum)
            : Tensor<T>.CreateDefault([numChunks], NumOps.FromDouble(1.0 / numChunks));
        var globalOutput = engine.TensorMatMul(chunkWeights.Reshape([1, numChunks]), summaries).Reshape([dim]);

        // Step 4: Decode through LLM layers
        var output = globalOutput;
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
            // LongVILA / VILA (Xue et al. 2024, arXiv:2408.10188) = image-video joint encoding with a
            // plain LINEAR projector to the LLM (no dedicated temporal transformer; long-context is a
            // training/systems feature, not an architectural block). Residual ViT + LLM blocks (the old
            // shared builder's non-residual stack collapsed to an input-independent output after training).
            Layers.AddRange(
                LayerHelper<T>.CreateDefaultVideoLinearProjectorVLMLayers(
                    _options.VisionDim,
                    _options.DecoderDim,
                    _options.NumVisionLayers,
                    _options.NumDecoderLayers,
                    _options.NumHeads,
                    _options.DropoutRate,
                    imageHeight: _options.ImageSize,
                    imageWidth: _options.ImageSize,
                    imageChannels: 3,
                    patchSize: 16
                )
            );
            ComputeEncoderDecoderBoundary();
        }
    }

    private void ComputeEncoderDecoderBoundary()
    {
        // CreateDefaultVideoLinearProjectorVLMLayers layout: PatchEmbedding(1) + LayerNorm(1) +
        // numVisionLayers·(TransformerEncoderLayer(1) + optional Dropout) before the projection +
        // LLM decoder. The vision-encoder segment ends after the vision blocks.
        int perBlock = _options.DropoutRate > 0 ? 2 : 1;
        _encoderLayerEnd = 2 + _options.NumVisionLayers * perBlock;
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
            Name = _useNativeMode ? "LongVILA-Native" : "LongVILA-ONNX",
            Description = "LongVILA: long-context visual language model for 1hr+ videos.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "LongVILA";
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        m.AdditionalInfo["MaxVideoMinutes"] = _options.MaxVideoMinutes.ToString();
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(LongVILA<T>));
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
