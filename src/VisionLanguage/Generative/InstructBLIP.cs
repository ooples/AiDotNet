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
/// InstructBLIP: instruction-tuned BLIP-2 for zero-shot generalization across vision-language tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// InstructBLIP (Dai et al., NeurIPS 2023) instruction-tunes the Q-Former component of BLIP-2
/// to extract instruction-aware visual features. The instruction is fed to both the Q-Former
/// (to guide visual feature extraction) and the LLM (to guide text generation).
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning" (Dai et al., NeurIPS 2023)</item></list></para>
/// <para><b>For Beginners:</b> InstructBLIP adds instruction-following capability to the
/// BLIP-2 model by instruction-tuning the Q-Former to extract visual features that are
/// relevant to the given instruction. The instruction is fed to both the Q-Former (to guide
/// what visual information to extract) and the LLM (to guide text generation), enabling
/// zero-shot generalization across diverse vision-language tasks. Default values follow
/// the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
/// var trainModel = new InstructBLIP&lt;double&gt;(architecture, new InstructBLIPOptions());
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
[ResearchPaper(
    "InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning",
    "https://arxiv.org/abs/2305.06500",
    Year = 2023,
    Authors = "Dai et al."
)]
public class InstructBLIP<T> : VisionLanguageModelBase<T>, IGenerativeVisionLanguageModel<T>
{
    private readonly InstructBLIPOptions _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;

    private readonly List<ILayer<T>> _qFormerLayers = new List<ILayer<T>>();
    private readonly List<ILayer<T>> _decoderLayers = new List<ILayer<T>>();

    public InstructBLIP(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        InstructBLIPOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new InstructBLIPOptions();
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

    public InstructBLIP(
        NeuralNetworkArchitecture<T> architecture,
        InstructBLIPOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new InstructBLIPOptions();
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
        if (h > 0 && w > 0 && h == w)
            _options.ImageSize = h;
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
        var c = p;
        foreach (var l in Layers)
            c = l.Forward(c);
        return L2Normalize(c);
    }

    /// <summary>
    /// Generates text using InstructBLIP's instruction-aware Q-Former architecture.
    /// InstructBLIP (Dai et al., NeurIPS 2023) extends BLIP-2 with:
    /// (1) Instruction-aware Q-Former: the instruction text is fed into the Q-Former
    ///     alongside learnable queries, so visual feature extraction is guided by the
    ///     specific task instruction (not just generic visual encoding),
    /// (2) Dual instruction routing: instruction goes to both Q-Former (visual extraction)
    ///     and the LLM decoder (text generation), creating instruction-conditioned features,
    /// (3) Instruction-tuned on 26 datasets with task-diverse instructions for zero-shot
    ///     generalization to unseen vision-language tasks.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        // Step 1: Frozen ViT vision encoder
        var visionOut = p;
        foreach (var l in Layers)
            visionOut = l.Forward(visionOut);

        // Step 2: Instruction-aware Q-Former
        var qFormerOut = visionOut;
        foreach (var l in _qFormerLayers)
            qFormerOut = l.Forward(qFormerOut);

        // Step 3: Tokenize instruction for dual routing
        Tensor<T>? promptTokens = null;
        if (prompt is not null)
            promptTokens = TokenizeText(prompt);

        // Step 4: Concatenate Q-Former output with instruction tokens for LLM input
        // InstructBLIP routes instruction to both Q-Former and LLM decoder
        var decoderInput = qFormerOut;
        if (promptTokens is not null)
            decoderInput = qFormerOut.ConcatenateTensors(promptTokens);

        // Step 5: LLM decoder generates text conditioned on visual + instruction
        var output = decoderInput;
        foreach (var l in _decoderLayers)
            output = l.Forward(output);

        return output;
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            return;
        }

        int blockSize = _options.DropoutRate > 0 ? 6 : 5;
        int qfBlockSize = _options.DropoutRate > 0 ? 8 : 7;
        // 2 + ...: PatchEmbeddingLayer + pre-norm, then N×vision-block.
        int visionLayerEnd = 2 + _options.NumVisionLayers * blockSize;
        int qfProj = _options.VisionDim != _options.QFormerDim ? 1 : 0;
        int qFormerLayerEnd = visionLayerEnd + qfProj + _options.NumQFormerLayers * qfBlockSize;

        var allLayers = LayerHelper<T>.CreateDefaultQFormerGenerativeLayers(
            _options.VisionDim,
            _options.QFormerDim,
            _options.DecoderDim,
            _options.NumVisionLayers,
            _options.NumQFormerLayers,
            _options.NumDecoderLayers,
            _options.NumQueryTokens,
            _options.NumHeads,
            _options.NumQFormerHeads,
            _options.DropoutRate
        );

        int idx = 0;
        foreach (var layer in allLayers)
        {
            if (idx < visionLayerEnd)
                Layers.Add(layer);
            else if (idx < qFormerLayerEnd)
                _qFormerLayers.Add(layer);
            else
                _decoderLayers.Add(layer);
            idx++;
        }

        RegisterAuxiliaryEncoderStream(_qFormerLayers);
        RegisterAuxiliaryEncoderStream(_decoderLayers);
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

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        SetTrainingMode(false);
        var c = PreprocessImage(input);
        foreach (var l in Layers)
            c = l.Forward(c);
        return c;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            TrainWithTape(PreprocessImage(input), expected);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Cannot update parameters in ONNX mode.");
        int idx = 0;
        foreach (var l in Layers)
        {
            int c = (int)l.ParameterCount;
            l.UpdateParameters(parameters.Slice(idx, c));
            idx += c;
        }
        // Sync the auxiliary streams (Q-Former / perceiver / decoder /
        // regression head, depending on model) — see OpenFlamingo.UpdateParameters
        // for full rationale (dual-stream split, GetExtraTrainableLayers
        // widens the flat parameter vector to include them, so a writeback
        // that only walks Layers leaves auxiliary streams on stale weights
        // and the model state silently de-syncs across streams).
        foreach (var l in EnumerateAuxiliaryStreamTrainableLayers())
        {
            if (l is null)
                continue;
            int c = (int)l.ParameterCount;
            l.UpdateParameters(parameters.Slice(idx, c));
            idx += c;
        }
    }

    /// <inheritdoc />
    protected override IEnumerable<LayerBase<T>?> GetExtraTrainableLayers() =>
        EnumerateAuxiliaryStreamTrainableLayers();

    protected override Tensor<T> PreprocessImage(Tensor<T> image) =>
        NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "InstructBLIP-Native" : "InstructBLIP-ONNX",
            Description =
                "InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning (Dai et al., NeurIPS 2023)",
            FeatureCount = _options.DecoderDim,
            Complexity =
                _options.NumVisionLayers + _options.NumQFormerLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "InstructBLIP";
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
        if (!string.IsNullOrEmpty(mp))
            _options.ModelPath = mp;
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
            return new InstructBLIP<T>(Architecture, mp, _options);
        return new InstructBLIP<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(InstructBLIP<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        if (disposing)
        {
            OnnxModel?.Dispose();
        }
        base.Dispose(disposing);
    }
}
