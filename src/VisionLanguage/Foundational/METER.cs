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
/// METER (Multimodal End-to-end TransformER) with systematic VLP component study.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// METER (Dou et al., CVPR 2022) is a systematic study of vision-language pre-training components.
/// It uses a CLIP ViT vision encoder and RoBERTa text encoder connected by co-attention transformer
/// fusion layers, providing an optimized combination of architecture choices for VLP.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "An Empirical Study of Training End-to-End Vision-and-Language Transformers" (Dou et al., CVPR 2022)</item></list></para>
/// <para><b>For Beginners:</b> METER is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
/// var trainModel = new METER&lt;double&gt;(architecture, new METEROptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("An Empirical Study of Training End-to-End Vision-and-Language Transformers", "https://arxiv.org/abs/2111.02387", Year = 2022, Authors = "Dou et al.")]
public class METER<T> : VisionLanguageModelBase<T>, IVisionLanguageFusionModel<T>
{
    private readonly METEROptions _options;
    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;

    private readonly List<ILayer<T>> _textCoAttnLayers = new List<ILayer<T>>();

    public METER(NeuralNetworkArchitecture<T> architecture, string modelPath, METEROptions? options = null) : base(architecture)
    {
        _options = options ?? new METEROptions();
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

    public METER(NeuralNetworkArchitecture<T> architecture, METEROptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new METEROptions();
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

        var visionOut = p;
        foreach (var l in Layers) visionOut = l.Forward(visionOut);

        // Run the text stream first, then dispatch each co-attention block
        // with both modalities so vision→text and text→vision actually
        // attend to each other. The previous implementation walked
        // _textCoAttnLayers single-input only, which routed every
        // CrossAttentionLayer through its self-attention fallback
        // (Forward(input) = ForwardCrossAttention(input, input)) — the
        // co-attention encoder degenerated into text self-attention and
        // visionOut was concatenated to the result without ever entering
        // the fusion stack. Layer order inside _textCoAttnLayers matches
        // CreateDefaultDualStreamFusionLayers in LayerHelper:
        //   * 0..textProjEnd-1   : text pre-norm (1) + optional projection (1)
        //   * textProjEnd..textEnd-1 : N_text × blockSize text self-attn blocks
        //   * textEnd..end       : N_fusion × coAttnBlockSize co-attention blocks
        // where blockSize = 5 (or 6 with dropout) and coAttnBlockSize = 7
        // (or 8 with dropout).
        var textState = TokenizeText(text);
        int blockSize = _options.DropoutRate > 0 ? 6 : 5;
        int coAttnBlockSize = _options.DropoutRate > 0 ? 8 : 7;
        int textProjEnd = 1 + (_options.TextDim != _options.FusionDim ? 1 : 0);
        int textEnd = textProjEnd + _options.NumTextLayers * blockSize;

        // Text encoder (pre-norm + projection + N self-attn blocks).
        for (int i = 0; i < textEnd; i++) textState = _textCoAttnLayers[i].Forward(textState);

        // Co-attention fusion blocks: bidirectional cross-attention between
        // the two modality streams. Each block layout matches the factory:
        //   [0] CrossAttention (vision → text): query=visionOut, context=textState
        //   [1] LayerNorm                       on visionOut
        //   [2] CrossAttention (text → vision): query=textState, context=visionOut
        //   [3] LayerNorm                       on textState
        //   [4] DenseLayer FFN1                 on textState
        //   [5] DenseLayer FFN2                 on textState
        //   [6] LayerNorm                       on textState
        //   [7] (DropoutLayer)                  on textState (optional)
        for (int blockStart = textEnd; blockStart < _textCoAttnLayers.Count; blockStart += coAttnBlockSize)
        {
            var preText = textState;
            var preVision = visionOut;
            var visionToText = (LayerBase<T>)_textCoAttnLayers[blockStart];
            var textToVision = (LayerBase<T>)_textCoAttnLayers[blockStart + 2];
            visionOut = visionToText.Forward(preVision, preText);
            visionOut = _textCoAttnLayers[blockStart + 1].Forward(visionOut);
            textState = textToVision.Forward(preText, preVision);
            textState = _textCoAttnLayers[blockStart + 3].Forward(textState);
            textState = _textCoAttnLayers[blockStart + 4].Forward(textState);
            textState = _textCoAttnLayers[blockStart + 5].Forward(textState);
            textState = _textCoAttnLayers[blockStart + 6].Forward(textState);
            if (coAttnBlockSize == 8) textState = _textCoAttnLayers[blockStart + 7].Forward(textState);
        }

        return visionOut.ConcatenateTensors(textState);
    }

    public T ComputeMatchingScore(Tensor<T> image, string text)
    {
        // Run the full dual-stream + co-attention fusion so the matching
        // score reflects bidirectional attention. The previous text-only
        // walk over _textCoAttnLayers compared the vision projection
        // against text-self-attention output without either stream
        // having attended to the other — similarity was driven entirely
        // by chance overlap of the random initial projections.
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
            _textCoAttnLayers.AddRange(dual.TextLayers);
            RegisterAuxiliaryEncoderStream(_textCoAttnLayers);
            return;
        }

        int blockSize = _options.DropoutRate > 0 ? 6 : 5;
        int visionLayerEnd = 1 + _options.NumVisionLayers * blockSize
            + (_options.VisionDim != _options.FusionDim ? 1 : 0);

        var allLayers = LayerHelper<T>.CreateDefaultDualStreamFusionLayers(
            _options.VisionDim, _options.TextDim, _options.FusionDim,
            _options.NumVisionLayers, _options.NumTextLayers, _options.NumCrossAttentionLayers,
            _options.NumHeads, _options.DropoutRate);

        int idx = 0;
        foreach (var layer in allLayers)
        {
            if (idx < visionLayerEnd) Layers.Add(layer);
            else _textCoAttnLayers.Add(layer);
            idx++;
        }

        RegisterAuxiliaryEncoderStream(_textCoAttnLayers);
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
        // Co-attention stream is part of the trainable graph (registered
        // via RegisterAuxiliaryEncoderStream and surfaced through
        // GetExtraTrainableLayers), so its parameter slices live alongside
        // the vision encoder's in the flat parameter vector.
        foreach (var l in _textCoAttnLayers) { int c = (int)l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
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
            Name = _useNativeMode ? "METER-Native" : "METER-ONNX",
            Description = "METER: An Empirical Study of Training End-to-End Vision-and-Language Transformers (Dou et al., CVPR 2022)",
            FeatureCount = _options.FusionDim,
            Complexity = _options.NumVisionLayers + _options.NumTextLayers + _options.NumCrossAttentionLayers
        };
        m.AdditionalInfo["Architecture"] = "METER";
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
        writer.Write(_options.NumVisionLayers);
        writer.Write(_options.NumTextLayers);
        writer.Write(_options.NumCrossAttentionLayers);
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
        _options.NumVisionLayers = reader.ReadInt32();
        _options.NumTextLayers = reader.ReadInt32();
        _options.NumCrossAttentionLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new METER<T>(Architecture, mp, _options);
        return new METER<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(METER<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing) { OnnxModel?.Dispose(); }
        base.Dispose(disposing);
    }
}
