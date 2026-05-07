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

namespace AiDotNet.VisionLanguage.Foundational;

/// <summary>
/// LXMERT (Learning Cross-Modality Encoder Representations from Transformers) with three-encoder architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// LXMERT (Tan and Bansal, EMNLP 2019) uses three transformer encoders: an object relationship
/// encoder for visual features, a language encoder for text, and a cross-modality encoder that
/// performs cross-attention between the two modalities.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "LXMERT: Learning Cross-Modality Encoder Representations from Transformers" (Tan and Bansal, EMNLP 2019)</item></list></para>
/// <para><b>For Beginners:</b> LXMERT is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
/// var trainModel = new LXMERT&lt;double&gt;(architecture, new LXMERTOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("LXMERT: Learning Cross-Modality Encoder Representations from Transformers", "https://arxiv.org/abs/1908.07490", Year = 2019, Authors = "Tan and Bansal")]
public class LXMERT<T> : VisionLanguageModelBase<T>, IVisionLanguageFusionModel<T>
{
    private readonly LXMERTOptions _options;
    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;

    // Three-encoder split: object-relationship encoder (vision) in Layers,
    // text encoder + cross-modality encoder as auxiliary streams.
    private readonly List<ILayer<T>> _textCrossModalLayers = new List<ILayer<T>>();

    public LXMERT(NeuralNetworkArchitecture<T> architecture, string modelPath, LXMERTOptions? options = null) : base(architecture)
    {
        _options = options ?? new LXMERTOptions();
        SyncImageSizeWithArchitecture();
        _useNativeMode = false;
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.FusionDim;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    public LXMERT(NeuralNetworkArchitecture<T> architecture, LXMERTOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new LXMERTOptions();
        SyncImageSizeWithArchitecture();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.FusionDim;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    private void SyncImageSizeWithArchitecture()
    {
        int h = Architecture.InputHeight;
        int w = Architecture.InputWidth;
        if (h > 0 && w > 0 && h == w) _options.ImageSize = h;
    }

    public int EmbeddingDimension => _options.FusionDim;
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;
    int IVisualEncoder<T>.ImageChannels => 3;
    public int FusionEmbeddingDim => _options.FusionDim;
    public int MaxSequenceLength => _options.MaxSequenceLength;

    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p));
        var c = p;
        foreach (var l in Layers) c = l.Forward(c);
        return L2Normalize(c);
    }

    public Tensor<T> FuseImageText(Tensor<T> image, string text)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        // Encoder 1: Object relationship encoder (vision features) — Layers
        var visionOut = p;
        foreach (var l in Layers) visionOut = l.Forward(visionOut);

        // Encoder 2 + 3: Language + cross-modality encoders — auxiliary stream.
        // The previous implementation walked _textCrossModalLayers with single-
        // input Forward(current) only, which made every CrossAttentionLayer
        // fall through to its self-attention fallback (Forward(input) routes
        // to ForwardCrossAttention(input, input)). The cross-modality stack
        // therefore never saw the vision features at all — it ran as
        // text-only self-attention, then visionOut was concatenated to the
        // result. That defeated the entire LXMERT three-encoder design.
        //
        // Run the language portion text-only, then dispatch each cross-modal
        // block with both modalities so language→vision attends to vision
        // and vision→language attends to language. Layer order inside the
        // auxiliary stream matches CreateDefaultCrossModalFusionLayers in
        // LayerHelper:
        //   * 0..textProjEnd-1   : text projection (0 or 2 layers)
        //   * textProjEnd..textEnd-1 : N_text × blockSize  language self-attn blocks
        //   * textEnd..end       : N_cross × crossBlockSize bidirectional blocks
        // where blockSize = 5 (or 6 with dropout) and crossBlockSize = 7
        // (or 8 with dropout).
        var textState = TokenizeText(text);
        int blockSize = _options.DropoutRate > 0 ? 6 : 5;
        int crossBlockSize = _options.DropoutRate > 0 ? 8 : 7;
        int textProjEnd = (_options.TextDim != _options.FusionDim) ? 2 : 0;
        int textEnd = textProjEnd + _options.NumTextLayers * blockSize;

        // Language encoder + projection: text-only self-attention.
        for (int i = 0; i < textEnd; i++) textState = _textCrossModalLayers[i].Forward(textState);

        // Cross-modality encoder: bidirectional cross-attention blocks.
        // Each block layout matches the factory exactly:
        //   [0] CrossAttention (lang → vision): Forward(textState, visionOut)
        //   [1] LayerNorm                       on textState
        //   [2] CrossAttention (vision → lang): Forward(visionOut, textState)
        //   [3] LayerNorm                       on visionOut
        //   [4] DenseLayer FFN1                 on textState
        //   [5] DenseLayer FFN2                 on textState
        //   [6] LayerNorm                       on textState
        //   [7] (DropoutLayer)                  on textState (optional)
        for (int blockStart = textEnd; blockStart < _textCrossModalLayers.Count; blockStart += crossBlockSize)
        {
            // Capture the pre-cross-attn streams so the second cross-attn
            // (vision → lang) gets the original textState, not the post-
            // language-update one — bidirectional cross-attn in the LXMERT
            // paper updates both streams from a shared input snapshot.
            var preText = textState;
            var preVision = visionOut;
            // CrossAttentionLayer overrides LayerBase.Forward(params Tensor<T>[])
            // with the cross-attn-aware [query, context] dispatch — but that
            // overload lives on LayerBase / CrossAttentionLayer, not on
            // ILayer<T>. Cast through LayerBase<T> so the params overload
            // resolves; single-input fallback would re-trigger the self-
            // attention bug we are fixing.
            var langToVision = (LayerBase<T>)_textCrossModalLayers[blockStart];
            var visionToLang = (LayerBase<T>)_textCrossModalLayers[blockStart + 2];
            textState = langToVision.Forward(preText, preVision);
            textState = _textCrossModalLayers[blockStart + 1].Forward(textState);
            visionOut = visionToLang.Forward(preVision, preText);
            visionOut = _textCrossModalLayers[blockStart + 3].Forward(visionOut);
            textState = _textCrossModalLayers[blockStart + 4].Forward(textState);
            textState = _textCrossModalLayers[blockStart + 5].Forward(textState);
            textState = _textCrossModalLayers[blockStart + 6].Forward(textState);
            if (crossBlockSize == 8) textState = _textCrossModalLayers[blockStart + 7].Forward(textState);
        }

        return visionOut.ConcatenateTensors(textState);
    }

    public T ComputeMatchingScore(Tensor<T> image, string text)
    {
        // Run the full LXMERT three-encoder fusion so the matching score
        // reflects bidirectional cross-attention. The previous implementation
        // walked _textCrossModalLayers text-only and then computed cosine
        // similarity between the vision projection and a text-self-attention
        // output — neither stream had ever attended to the other, so the
        // similarity was driven entirely by chance overlap of the random
        // initial projections. FuseImageText now does the right thing
        // (cross-attn dispatched with both streams), so reuse it.
        if (IsOnnxMode && OnnxModel is not null)
        {
            var imageEmb = EncodeImage(image);
            var textTokens = TokenizeText(text);
            var textEmb = L2Normalize(OnnxModel.Run(textTokens));
            return CosineSimilarity(imageEmb, textEmb);
        }
        var fused = FuseImageText(image, text);
        // FuseImageText concatenates [vision, text] along the last dim;
        // matching score is the cosine similarity between the two halves.
        // Split via Tensor<T>.Slice(axis, start, end?) on the trailing axis.
        int lastDim = fused.Shape[^1];
        int half = lastDim / 2;
        int axis = fused.Rank - 1;
        var visionHalf = fused.Slice(axis, 0, half);
        var textHalf = fused.Slice(axis, half, lastDim);
        return CosineSimilarity(L2Normalize(visionHalf), L2Normalize(textHalf));
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture is DualStreamArchitecture<T> dual)
        {
            Layers.AddRange(dual.VisionLayers);
            _textCrossModalLayers.AddRange(dual.TextLayers);
            RegisterAuxiliaryEncoderStream(_textCrossModalLayers);
            return;
        }

        int blockSize = _options.DropoutRate > 0 ? 6 : 5;
        int visionLayerEnd = (_options.VisionDim != _options.FusionDim ? 1 : 0)
            + 1 + _options.NumRelationshipLayers * blockSize;

        var allLayers = LayerHelper<T>.CreateDefaultCrossModalFusionLayers(
            _options.VisionDim, _options.TextDim, _options.FusionDim,
            _options.NumRelationshipLayers, _options.NumTextLayers, _options.NumCrossModalityLayers,
            _options.NumHeads, _options.DropoutRate);

        int idx = 0;
        foreach (var layer in allLayers)
        {
            if (idx < visionLayerEnd) Layers.Add(layer);
            else _textCrossModalLayers.Add(layer);
            idx++;
        }

        RegisterAuxiliaryEncoderStream(_textCrossModalLayers);
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
        // Pass _optimizer through to TrainWithTape so the configured (or
        // defaulted) AdamW is used instead of the base class's
        // GetOrCreateBaseOptimizer default Adam at lr=1e-3. See
        // BridgeTower.Train for full rationale.
        try { TrainWithTape(PreprocessImage(input), expected, _optimizer); }
        finally { SetTrainingMode(false); }
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode.");
        int idx = 0;
        foreach (var l in Layers) { int c = (int)l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
        // Cross-modality stream is part of the trainable graph (registered
        // via RegisterAuxiliaryEncoderStream and surfaced through
        // GetExtraTrainableLayers), so its parameter slices live alongside
        // the vision encoder's in the flat parameter vector.
        foreach (var l in _textCrossModalLayers) { int c = (int)l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
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
            Name = _useNativeMode ? "LXMERT-Native" : "LXMERT-ONNX",
            Description = "LXMERT: Learning Cross-Modality Encoder Representations from Transformers (Tan and Bansal, EMNLP 2019)",
            FeatureCount = _options.FusionDim,
            Complexity = _options.NumRelationshipLayers + _options.NumTextLayers + _options.NumCrossModalityLayers
        };
        m.AdditionalInfo["Architecture"] = "LXMERT";
        m.AdditionalInfo["FusionType"] = _options.FusionType.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.ImageSize);
        writer.Write(_options.VisionDim);
        writer.Write(_options.TextDim);
        writer.Write(_options.FusionDim);
        writer.Write(_options.NumRelationshipLayers);
        writer.Write(_options.NumTextLayers);
        writer.Write(_options.NumCrossModalityLayers);
        writer.Write(_options.NumHeads);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.ImageSize = reader.ReadInt32();
        _options.VisionDim = reader.ReadInt32();
        _options.TextDim = reader.ReadInt32();
        _options.FusionDim = reader.ReadInt32();
        _options.NumRelationshipLayers = reader.ReadInt32();
        _options.NumTextLayers = reader.ReadInt32();
        _options.NumCrossModalityLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new LXMERT<T>(Architecture, mp, _options);
        return new LXMERT<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(LXMERT<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing) { OnnxModel?.Dispose(); }
        base.Dispose(disposing);
    }
}
