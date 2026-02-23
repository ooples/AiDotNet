using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// LiT (Locked-image Tuning) model that freezes a pre-trained image encoder and only
/// trains the text encoder for efficient contrastive image-text alignment.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// LiT (Zhai et al., CVPR 2022) demonstrates that contrastive image-text training can be made
/// dramatically more efficient by freezing a high-quality pre-trained vision encoder (e.g., from
/// ImageNet-21k) and only training the text encoder to align with it. This "locked-image tuning"
/// achieves 85.2% zero-shot accuracy on ImageNet with significantly reduced training cost.
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "LiT: Zero-Shot Transfer with Locked-image text Tuning" (Zhai et al., CVPR 2022)</item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> LiT is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
public class LiT<T> : VisionLanguageModelBase<T>, IContrastiveVisionLanguageModel<T>
{
    private readonly LiTOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed; private int _visionLayerEnd;

    public LiT(NeuralNetworkArchitecture<T> architecture, string imageEncoderModelPath, LiTOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new LiTOptions();
        _useNativeMode = false;
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.VisionEmbeddingDim;

        if (string.IsNullOrWhiteSpace(imageEncoderModelPath))
            throw new ArgumentException("Image encoder model path cannot be null or empty.", nameof(imageEncoderModelPath));
        if (!File.Exists(imageEncoderModelPath))
            throw new FileNotFoundException($"ONNX model not found: {imageEncoderModelPath}", imageEncoderModelPath);

        _options.ImageEncoderModelPath = imageEncoderModelPath;
        OnnxImageEncoder = new OnnxModel<T>(imageEncoderModelPath, _options.OnnxOptions);

        if (_options.TextEncoderModelPath is { } tp && !string.IsNullOrEmpty(tp))
        {
            if (!File.Exists(tp)) throw new FileNotFoundException($"Text encoder ONNX model not found: {tp}", tp);
            OnnxTextEncoder = new OnnxModel<T>(tp, _options.OnnxOptions);
        }

        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    public LiT(NeuralNetworkArchitecture<T> architecture, LiTOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new LiTOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.VisionEmbeddingDim;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    public int EmbeddingDimension => _options.VisionEmbeddingDim;
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;
    int IVisualEncoder<T>.ImageChannels => 3;
    public int MaxSequenceLength => _options.MaxSequenceLength;
    public int TextEmbeddingDimension => _options.TextEmbeddingDim;
    public int ProjectionDimension => _options.ProjectionDim;
    public T Temperature => NumOps.FromDouble(_options.Temperature);

    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var preprocessed = PreprocessImage(image); if (IsOnnxMode && OnnxImageEncoder is not null) return L2Normalize(OnnxImageEncoder.Run(preprocessed)); return L2Normalize(ForwardVisionEncoder(preprocessed)); }
    public Tensor<T> EncodeText(string text) { ThrowIfDisposed(); var tokenized = TokenizeText(text); if (IsOnnxMode && OnnxTextEncoder is not null) return L2Normalize(OnnxTextEncoder.Run(tokenized)); return L2Normalize(ForwardTextEncoder(tokenized)); }
    public Tensor<T>[] EncodeTexts(string[] texts) { var e = new Tensor<T>[texts.Length]; for (int i = 0; i < texts.Length; i++) e[i] = EncodeText(texts[i]); return e; }
    public T ComputeSimilarity(Tensor<T> image, string text) { return CosineSimilarity(EncodeImage(image), EncodeText(text)); }

    public Dictionary<string, T> ZeroShotClassify(Tensor<T> image, string[] labels)
    {
        var imageEmb = EncodeImage(image);
        var textEmbs = EncodeTexts(labels);
        var logits = new Tensor<T>([labels.Length]);
        double temp = _options.Temperature;
        for (int i = 0; i < labels.Length; i++)
            logits[i] = NumOps.FromDouble(NumOps.ToDouble(CosineSimilarity(imageEmb, textEmbs[i])) / temp);
        var probs = Softmax(logits);
        var result = new Dictionary<string, T>();
        for (int i = 0; i < labels.Length; i++) result[labels[i]] = probs[i];
        return result;
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        { Layers.AddRange(Architecture.Layers); _visionLayerEnd = Layers.Count / 2; }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultOpenCLIPLayers(
                visionEmbeddingDim: _options.VisionEmbeddingDim,
                textEmbeddingDim: _options.TextEmbeddingDim,
                projectionDim: _options.ProjectionDim,
                numVisionLayers: _options.NumVisionLayers,
                numTextLayers: _options.NumTextLayers,
                numVisionHeads: _options.NumVisionHeads,
                numTextHeads: _options.NumTextHeads,
                dropoutRate: _options.DropoutRate));
            int lpb = _options.DropoutRate > 0 ? 6 : 5; _visionLayerEnd = 2 + _options.NumVisionLayers * lpb;
        }
    }

    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxImageEncoder is not null) return OnnxImageEncoder.Run(input); var current = input; foreach (var layer in Layers) current = layer.Forward(current); return current; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var output = Predict(input); var grad = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(grad); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var layer in Layers) { int count = layer.ParameterCount; layer.UpdateParameters(parameters.Slice(idx, count)); idx += count; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var meta = new ModelMetadata<T> { Name = _useNativeMode ? "LiT-Native" : "LiT-ONNX", Description = "LiT: Zero-Shot Transfer with Locked-image text Tuning (Zhai et al., CVPR 2022)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.ProjectionDim, Complexity = _options.NumVisionLayers + _options.NumTextLayers };
        meta.AdditionalInfo["Architecture"] = "LiT";
        meta.AdditionalInfo["FreezeVisionEncoder"] = _options.FreezeVisionEncoder.ToString();
        meta.AdditionalInfo["ProjectionDim"] = _options.ProjectionDim.ToString();
        return meta;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ImageEncoderModelPath ?? string.Empty); writer.Write(_options.TextEncoderModelPath ?? string.Empty); writer.Write(_options.ImageSize); writer.Write(_options.VisionEmbeddingDim); writer.Write(_options.TextEmbeddingDim); writer.Write(_options.ProjectionDim); writer.Write(_options.Temperature); writer.Write(_options.FreezeVisionEncoder); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string imgPath = reader.ReadString(); if (!string.IsNullOrEmpty(imgPath)) _options.ImageEncoderModelPath = imgPath; string txtPath = reader.ReadString(); if (!string.IsNullOrEmpty(txtPath)) _options.TextEncoderModelPath = txtPath; _options.ImageSize = reader.ReadInt32(); _options.VisionEmbeddingDim = reader.ReadInt32(); _options.TextEmbeddingDim = reader.ReadInt32(); _options.ProjectionDim = reader.ReadInt32(); _options.Temperature = reader.ReadDouble(); _options.FreezeVisionEncoder = reader.ReadBoolean(); if (!_useNativeMode && _options.ImageEncoderModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxImageEncoder = new OnnxModel<T>(p, _options.OnnxOptions); if (_options.TextEncoderModelPath is { } tp2 && !string.IsNullOrEmpty(tp2)) OnnxTextEncoder = new OnnxModel<T>(tp2, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ImageEncoderModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new LiT<T>(Architecture, mp, _options); return new LiT<T>(Architecture, _options); }

    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }
    private Tensor<T> ForwardVisionEncoder(Tensor<T> input) { var current = input; for (int i = 0; i < _visionLayerEnd; i++) current = Layers[i].Forward(current); return current; }
    private Tensor<T> ForwardTextEncoder(Tensor<T> tokens) { var current = tokens; for (int i = _visionLayerEnd; i < Layers.Count; i++) current = Layers[i].Forward(current); return current; }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(LiT<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; if (disposing) { OnnxImageEncoder?.Dispose(); OnnxTextEncoder?.Dispose(); } base.Dispose(disposing); }
}
