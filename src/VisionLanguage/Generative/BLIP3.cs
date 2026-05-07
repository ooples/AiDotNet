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
/// BLIP-3 (xGen-MM): scaled vision-language model with interleaved data and any-to-any generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// BLIP-3/xGen-MM (Salesforce, 2024) scales the BLIP-2 architecture with interleaved image-text
/// data (OBELICS), larger Q-Former capacity, and any-to-any generation capabilities.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "xGen-MM (BLIP-3): A Family of Open Large Multimodal Models" (Salesforce, 2024)</item></list></para>
/// <para><b>For Beginners:</b> BLIP-3 (also called xGen-MM) is Salesforce's scaled multimodal model
/// that can understand images, answer questions about them, and generate both text and images from
/// interleaved image-text inputs. It builds on BLIP-2's Q-Former design with larger capacity and
/// training on web-scraped interleaved data. Default values follow the original paper settings.</para>
/// <para><b>Architecture layout:</b> Triple-stream — vision encoder lives in
/// <see cref="NeuralNetworkBase{T}.Layers"/> (so default Predict / TrainWithTape walk only it),
/// Q-Former lives in a private auxiliary stream, decoder lives in another. Image+text generation
/// goes through <see cref="GenerateFromImage"/> which walks all three streams; raw
/// <see cref="Predict"/> returns the vision-only forward (matching the IVisualEncoder contract).</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
/// var trainModel = new BLIP3&lt;double&gt;(architecture, new BLIP3Options());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("xGen-MM (BLIP-3): A Family of Open Large Multimodal Models", "https://arxiv.org/abs/2408.08872", Year = 2024, Authors = "Xue et al.")]
public class BLIP3<T> : VisionLanguageModelBase<T>, IGenerativeVisionLanguageModel<T>
{
    private readonly BLIP3Options _options;
    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;

    // Triple-stream: vision (Layers) -> Q-Former (_qFormerLayers) -> decoder (_decoderLayers).
    // Vision lives in Layers so the inherited Predict / TrainWithTape walk it; the other two
    // are auxiliary streams registered with the base for weight-registry surfacing.
    private readonly List<ILayer<T>> _qFormerLayers = new List<ILayer<T>>();
    private readonly List<ILayer<T>> _decoderLayers = new List<ILayer<T>>();

    public BLIP3(NeuralNetworkArchitecture<T> architecture, string modelPath, BLIP3Options? options = null) : base(architecture)
    {
        _options = options ?? new BLIP3Options();
        SyncImageSizeWithArchitecture();
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

    public BLIP3(NeuralNetworkArchitecture<T> architecture, BLIP3Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new BLIP3Options();
        SyncImageSizeWithArchitecture();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    private void SyncImageSizeWithArchitecture()
    {
        int h = Architecture.InputHeight;
        int w = Architecture.InputWidth;
        if (h > 0 && w > 0 && h == w) _options.ImageSize = h;
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
        if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p));
        var c = p;
        foreach (var l in Layers) c = l.Forward(c);
        return L2Normalize(c);
    }

    /// <summary>
    /// Generates text using BLIP-3/xGen-MM's scaled Q-Former architecture.
    /// BLIP-3 (Salesforce, 2024) extends BLIP-2 with:
    /// (1) Scaled Q-Former with increased capacity for richer visual queries,
    /// (2) Interleaved image-text training on OBELICS data for in-context learning,
    /// (3) Any-to-any generation: visual tokens from Q-Former are projected and
    ///     interleaved with text tokens for flexible multimodal generation,
    /// (4) LLM decoder (Phi-3) generates text conditioned on interleaved visual+text.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        // Step 1: Vision encoder
        var visionOut = p;
        foreach (var l in Layers) visionOut = l.Forward(visionOut);

        // Step 2: Scaled Q-Former cross-attention
        var qFormerOut = visionOut;
        foreach (var l in _qFormerLayers) qFormerOut = l.Forward(qFormerOut);

        // Step 3: Tokenize prompt for interleaved sequence
        Tensor<T>? promptTokens = null;
        if (prompt is not null) promptTokens = TokenizeText(prompt);

        // Step 4: Concatenate Q-Former output with prompt tokens for LLM decoder
        var decoderInput = qFormerOut;
        if (promptTokens is not null) decoderInput = qFormerOut.ConcatenateTensors(promptTokens);

        // Step 5: LLM decoder generates output
        var output = decoderInput;
        foreach (var l in _decoderLayers) output = l.Forward(output);

        return output;
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            // Caller-supplied layer graph: keep historic single-list behaviour.
            Layers.AddRange(Architecture.Layers);
            return;
        }

        // CreateDefaultQFormerGenerativeLayers emits:
        //   [pre-norm + N×vision-block, optional projection, M×qformer-block, K×decoder-block]
        //
        // Block sizes (no dropout): vision = 5, qformer = 7, decoder ~ same as transformer = 5.
        // With dropout add 1 per block. Compute boundaries first, then split.
        int blockSize = _options.DropoutRate > 0 ? 6 : 5;
        int qfBlockSize = _options.DropoutRate > 0 ? 8 : 7;
        int visionLayerEnd = 1 + _options.NumVisionLayers * blockSize;
        int qfProj = _options.VisionDim != _options.QFormerDim ? 1 : 0;
        int qFormerLayerEnd = visionLayerEnd + qfProj + _options.NumQFormerLayers * qfBlockSize;

        var allLayers = LayerHelper<T>.CreateDefaultQFormerGenerativeLayers(
            _options.VisionDim, _options.QFormerDim, _options.DecoderDim,
            _options.NumVisionLayers, _options.NumQFormerLayers, _options.NumDecoderLayers,
            _options.NumQueryTokens, _options.NumHeads, _options.NumQFormerHeads,
            _options.DropoutRate);

        int idx = 0;
        foreach (var layer in allLayers)
        {
            if (idx < visionLayerEnd) Layers.Add(layer);
            else if (idx < qFormerLayerEnd) _qFormerLayers.Add(layer);
            else _decoderLayers.Add(layer);
            idx++;
        }

        RegisterAuxiliaryEncoderStream(_qFormerLayers);
        RegisterAuxiliaryEncoderStream(_decoderLayers);
    }

    private Tensor<T> TokenizeText(string text)
    {
        if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized.");
        var encoding = _tokenizer.Encode(text);
        int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength);
        var tokens = new Tensor<T>([seqLen]);
        for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]);
        return tokens;
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        SetTrainingMode(false);
        var c = PreprocessImage(input);
        foreach (var l in Layers) c = l.Forward(c);
        return c;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        // Pass _optimizer through to TrainWithTape so the configured
        // (or defaulted) AdamW is used instead of the base class's
        // GetOrCreateBaseOptimizer default. See BridgeTower.Train for
        // full rationale.
        try { TrainWithTape(PreprocessImage(input), expected, _optimizer); }
        finally { SetTrainingMode(false); }
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode.");
        int idx = 0;
        foreach (var l in Layers) { int c = (int)l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
    }

    /// <inheritdoc />
    protected override IEnumerable<LayerBase<T>?> GetExtraTrainableLayers()
        => EnumerateAuxiliaryStreamTrainableLayers();

    protected override Tensor<T> PreprocessImage(Tensor<T> image) =>
        NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "BLIP3-Native" : "BLIP3-ONNX",
            Description = "BLIP-3/xGen-MM: A Family of Open Large Multimodal Models (Salesforce, 2024)",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumQFormerLayers + _options.NumDecoderLayers
        };
        m.AdditionalInfo["Architecture"] = "BLIP3";
        m.AdditionalInfo["GenerativeType"] = _options.ArchitectureType.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.ImageSize);
        writer.Write(_options.VisionDim);
        writer.Write(_options.QFormerDim);
        writer.Write(_options.DecoderDim);
        writer.Write(_options.NumVisionLayers);
        writer.Write(_options.NumQFormerLayers);
        writer.Write(_options.NumDecoderLayers);
        writer.Write(_options.NumHeads);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.ImageSize = reader.ReadInt32();
        _options.VisionDim = reader.ReadInt32();
        _options.QFormerDim = reader.ReadInt32();
        _options.DecoderDim = reader.ReadInt32();
        _options.NumVisionLayers = reader.ReadInt32();
        _options.NumQFormerLayers = reader.ReadInt32();
        _options.NumDecoderLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new BLIP3<T>(Architecture, mp, _options);
        return new BLIP3<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(BLIP3<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing)
        {
            // OnnxModel is allocated by the ONNX-mode constructor and
            // wraps a native ONNX Runtime session — without disposing it
            // here, repeated create/dispose cycles leak unmanaged session
            // memory.
            OnnxModel?.Dispose();
        }
        base.Dispose(disposing);
    }
}
