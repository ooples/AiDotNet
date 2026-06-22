using AiDotNet.Attributes;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>
/// PaLI-3: efficient PaLI with SigLIP ViT encoder, smaller and better.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PaLI-3 (Chen et al., 2023) achieves strong performance with a smaller model by replacing
/// the contrastive ViT with a SigLIP ViT encoder. The architecture retains the encoder-decoder
/// design but benefits from the improved vision representations of SigLIP pretraining.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "PaLI-3 Vision Language Models: Smaller, Faster, Stronger" (Chen et al., 2023)</item></list></para>
/// <para><b>For Beginners:</b> PaLI-3 achieves strong vision-language performance with a much
/// smaller model than PaLI-X by replacing the contrastive ViT encoder with a SigLIP ViT that
/// uses sigmoid-based contrastive loss instead of softmax. This produces better vision
/// representations while being significantly more efficient, demonstrating that smarter
/// pretraining can compensate for smaller model size. Default values follow the original
/// paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a PaLI-3 model for efficient vision-language understanding
/// // with SigLIP ViT encoder for smaller, faster, stronger performance
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new PaLI3&lt;double&gt;(architecture, "pali3.onnx");
///
/// // Training mode with native layers
/// var trainModel = new PaLI3&lt;double&gt;(architecture, new PaLI3Options());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "PaLI-3 Vision Language Models: Smaller, Faster, Stronger",
    "https://arxiv.org/abs/2310.09199",
    Year = 2023,
    Authors = "Chen et al."
)]
public class PaLI3<T> : VisionLanguageModelBase<T>, IGenerativeVisionLanguageModel<T>
{
    private readonly PaLI3Options _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public PaLI3(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        PaLI3Options? options = null
    )
        : base(architecture)
    {
        _options = options ?? new PaLI3Options();
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

    public PaLI3(
        NeuralNetworkArchitecture<T> architecture,
        PaLI3Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new PaLI3Options();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    public int EmbeddingDimension => _options.DecoderDim;
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;
    int IVisualEncoder<T>.ImageChannels => 3;
    public int MaxGenerationLength => _options.MaxGenerationLength;
    public int DecoderEmbeddingDim => _options.DecoderDim;

    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return L2Normalize(OnnxModel.Run(p));
        var c = TokenizeIfNCHW(p);
        for (int i = 0; i < _encoderLayerEnd; i++)
            c = Layers[i].Forward(c);
        return L2Normalize(c);
    }

    /// <summary>
    /// Generates text using PaLI-3's efficient SigLIP-based architecture.
    /// PaLI-3 (Chen et al., 2023) achieves strong results with a smaller model:
    /// (1) SigLIP ViT encoder: uses sigmoid loss instead of softmax contrastive
    ///     loss, providing more discriminative visual representations without
    ///     requiring a global softmax normalization across the batch,
    /// (2) Linear projection maps SigLIP features to decoder embedding space,
    /// (3) Visual tokens prepended to text for encoder-decoder processing,
    /// (4) Dual training objective: SigLIP contrastive alignment ensures strong
    ///     visual representations, while prefix LM generates text output,
    /// (5) Classification token pooling: SigLIP uses sigmoid per-pair scoring
    ///     rather than softmax over full batch, enabling efficient scaling.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        // Step 1: SigLIP ViT encoder (sigmoid-based contrastive pretraining)
        var encoderOut = TokenizeIfNCHW(p);
        for (int i = 0; i < _encoderLayerEnd; i++)
            encoderOut = Layers[i].Forward(encoderOut);

        // Step 2: Tokenize task-prefixed prompt
        Tensor<T>? promptTokens = null;
        if (prompt is not null)
            promptTokens = TokenizeText(prompt);

        // Step 3: Build prepended sequence [visual_tokens | task_prefix | text_tokens]
        // PaLI-3 prepends SigLIP visual tokens to text and feeds through decoder
        var decoderInput = encoderOut;
        if (promptTokens is not null)
            decoderInput = encoderOut.ConcatenateTensors(promptTokens);

        // Step 4: Text decoder (prefix LM) generates output
        var output = decoderInput;
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
                LayerHelper<T>.CreateDefaultEncoderDecoderVLMLayers(
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
        var tokens = new Tensor<T>([seqLen]);
        for (int i = 0; i < seqLen; i++)
            tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]);
        return tokens;
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        SetTrainingMode(false);
        var c = TokenizeIfNCHW(input);
        foreach (var l in Layers)
            c = l.Forward(c);
        return c;
    }

    private ConvolutionalLayer<T>? _patchEmbed;

    private Tensor<T> TokenizeIfNCHW(Tensor<T> input) =>
        PatchEmbedHelper.TokenizeImageNCHWToBSC(
            input,
            _options.VisionDim,
            _options.ImageSize,
            ref _patchEmbed,
            Engine
        );

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            // Tokenize NCHW image inputs the same way Predict does so the
            // tape-based training loop sees the patch-embedded BSC sequence
            // the encoder layer stack expects.
            TrainWithTape(TokenizeIfNCHW(input), expected);
        }
        finally
        {
            // try/finally so a TrainWithTape throw doesn't leave the model
            // stuck in training mode (dropout / norm-running-stats remain on
            // for every subsequent Predict call until something else flips
            // it back).
            SetTrainingMode(false);
        }
    }

    /// <summary>
    /// Surfaces _patchEmbed (which lives outside Layers) to the base
    /// weight-registry walker so its trainable tensors land in the
    /// streaming pool when ConfigureWeightLifetime is called.
    /// </summary>
    protected override IEnumerable<LayerBase<T>?> GetExtraTrainableLayers()
    {
        yield return _patchEmbed;
    }

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
        int layerSum = 0;
        foreach (var l in Layers)
            layerSum += (int)l.ParameterCount;
        if (paramVectorLength <= layerSum)
            return;

        var probe = new Tensor<T>(new[] { 1, 3, _options.ImageSize, _options.ImageSize });
        TokenizeIfNCHW(probe);
    }

    // _patchEmbed lives outside Layers but is trainable in native mode. Override
    // ParameterCount / GetParameters / SetParameters / UpdateParameters together
    // so all four agree on the layout: _patchEmbed slice first, then Layers in
    // order. Without this, the optimizer reads N params via GetParameters and
    // hands back N to UpdateParameters — but UpdateParameters consumes
    // _patchEmbed.ParameterCount extra slots from the front, shifting every
    // Layer's slice and corrupting weights.
    public override long ParameterCount =>
        (_patchEmbed?.ParameterCount ?? 0) + (int)Layers.Sum(l => l.ParameterCount);

    public override Vector<T> GetParameters()
    {
        var perLayer = Layers.Select(l => l.GetParameters()).ToList();
        int patchLen = (int)(_patchEmbed?.ParameterCount ?? 0);
        int total = patchLen + perLayer.Sum(p => p.Length);
        var result = new Vector<T>(total);
        int idx = 0;
        if (patchLen > 0)
        {
            var patchParams = _patchEmbed!.GetParameters();
            for (int i = 0; i < patchParams.Length; i++)
                result[idx++] = patchParams[i];
        }
        foreach (var p in perLayer)
        {
            for (int i = 0; i < p.Length; i++)
                result[idx++] = p[i];
        }
        return result;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        // If the saved parameter vector includes patch-embed weights but
        // this instance hasn't seen an image yet, lazy-create _patchEmbed
        // so the slice layout matches the saved vector. Otherwise the
        // patch-embed slice silently drops.
        EnsurePatchEmbedForParameterVector(parameters.Length);

        int idx = 0;
        if (_patchEmbed is not null)
        {
            int pc = checked((int)_patchEmbed.ParameterCount);
            if (pc > 0)
            {
                _patchEmbed.SetParameters(parameters.Slice(idx, pc));
                idx += pc;
            }
        }
        foreach (var l in Layers)
        {
            int c = checked((int)l.ParameterCount);
            l.SetParameters(parameters.Slice(idx, c));
            idx += c;
        }
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Cannot update parameters in ONNX mode.");
        EnsurePatchEmbedForParameterVector(parameters.Length);
        int idx = 0;
        if (_patchEmbed is not null)
        {
            int pc = checked((int)_patchEmbed.ParameterCount);
            if (pc > 0)
            {
                _patchEmbed.UpdateParameters(parameters.Slice(idx, pc));
                idx += pc;
            }
        }
        foreach (var l in Layers)
        {
            int c = (int)l.ParameterCount;
            l.UpdateParameters(parameters.Slice(idx, c));
            idx += c;
        }
    }

    protected override Tensor<T> PreprocessImage(Tensor<T> image) =>
        NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "PaLI-3-Native" : "PaLI-3-ONNX",
            Description =
                "PaLI-3 Vision Language Models: Smaller, Faster, Stronger (Chen et al., 2023)",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "PaLI-3";
        m.AdditionalInfo["GenerativeType"] = _options.ArchitectureType.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.ImageSize);
        writer.Write(_options.VisionDim);
        writer.Write(_options.DecoderDim);
        writer.Write(_options.NumVisionLayers);
        writer.Write(_options.NumDecoderLayers);
        writer.Write(_options.NumHeads);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp))
            _options.ModelPath = mp;
        _options.ImageSize = reader.ReadInt32();
        _options.VisionDim = reader.ReadInt32();
        _options.DecoderDim = reader.ReadInt32();
        _options.NumVisionLayers = reader.ReadInt32();
        _options.NumDecoderLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new PaLI3<T>(Architecture, mp, _options);
        return new PaLI3<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(PaLI3<T>));
    }

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
