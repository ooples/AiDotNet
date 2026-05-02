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

namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// BiomedCLIP model fine-tuned on 15M biomedical image-text pairs from PubMed Central.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// BiomedCLIP (Zhang et al., 2023) adapts CLIP for the biomedical domain using PMC-15M, achieving
/// state-of-the-art zero-shot biomedical image classification with ViT-B/16 + PubMedBERT.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "BiomedCLIP: A Multimodal Biomedical Foundation Model" (Zhang et al., 2023)</item></list></para>
/// <para><b>For Beginners:</b> BiomedCLIP adapts CLIP for the biomedical domain by fine-tuning
/// on 15 million image-text pairs from PubMed Central articles. It uses ViT-B/16 for images
/// and PubMedBERT for text, achieving state-of-the-art zero-shot biomedical image classification
/// for tasks like pathology slide classification and radiology report matching. Default values
/// follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a BiomedCLIP model for biomedical image-text alignment
/// // fine-tuned on 15M PubMed Central image-text pairs
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new BiomedCLIP&lt;double&gt;(architecture, "biomedclip.onnx");
///
/// // Training mode with native layers
/// var trainModel = new BiomedCLIP&lt;double&gt;(architecture, new BiomedCLIPOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.Healthcare)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("BiomedCLIP: A Multimodal Biomedical Foundation Model Pretrained from Fifteen Million Scientific Image-Text Pairs", "https://arxiv.org/abs/2303.00915", Year = 2023, Authors = "Zhang et al.")]
public class BiomedCLIP<T> : VisionLanguageModelBase<T>, IContrastiveVisionLanguageModel<T>
{
    private readonly BiomedCLIPOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed; private int _visionLayerEnd;

    public BiomedCLIP(NeuralNetworkArchitecture<T> architecture, string imageEncoderModelPath, BiomedCLIPOptions? options = null) : base(architecture) { _options = options ?? new BiomedCLIPOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.VisionEmbeddingDim; if (string.IsNullOrWhiteSpace(imageEncoderModelPath)) throw new ArgumentException("Image encoder model path cannot be null or empty.", nameof(imageEncoderModelPath)); if (!File.Exists(imageEncoderModelPath)) throw new FileNotFoundException($"ONNX model not found: {imageEncoderModelPath}", imageEncoderModelPath); _options.ImageEncoderModelPath = imageEncoderModelPath; OnnxImageEncoder = new OnnxModel<T>(imageEncoderModelPath, _options.OnnxOptions); if (_options.TextEncoderModelPath is { } tp && !string.IsNullOrEmpty(tp)) { if (!File.Exists(tp)) throw new FileNotFoundException($"Text ONNX not found: {tp}", tp); OnnxTextEncoder = new OnnxModel<T>(tp, _options.OnnxOptions); } _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public BiomedCLIP(NeuralNetworkArchitecture<T> architecture, BiomedCLIPOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new BiomedCLIPOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.VisionEmbeddingDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.VisionEmbeddingDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxSequenceLength => _options.MaxSequenceLength; public int TextEmbeddingDimension => _options.TextEmbeddingDim; public int ProjectionDimension => _options.ProjectionDim; public T Temperature => NumOps.FromDouble(_options.Temperature);
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxImageEncoder is not null) return L2Normalize(OnnxImageEncoder.Run(p)); return L2Normalize(ForwardVisionEncoder(p)); }
    public Tensor<T> EncodeText(string text) { ThrowIfDisposed(); var t = TokenizeText(text); if (IsOnnxMode && OnnxTextEncoder is not null) return L2Normalize(OnnxTextEncoder.Run(t)); return L2Normalize(ForwardTextEncoder(t)); }
    public Tensor<T>[] EncodeTexts(string[] texts) { var e = new Tensor<T>[texts.Length]; for (int i = 0; i < texts.Length; i++) e[i] = EncodeText(texts[i]); return e; }
    public T ComputeSimilarity(Tensor<T> image, string text) => CosineSimilarity(EncodeImage(image), EncodeText(text));
    public Dictionary<string, T> ZeroShotClassify(Tensor<T> image, string[] labels) { var ie = EncodeImage(image); var te = EncodeTexts(labels); var logits = new Tensor<T>([labels.Length]); double temp = _options.Temperature; for (int i = 0; i < labels.Length; i++) logits[i] = NumOps.FromDouble(NumOps.ToDouble(CosineSimilarity(ie, te[i])) / temp); var probs = Softmax(logits); var r = new Dictionary<string, T>(); for (int i = 0; i < labels.Length; i++) r[labels[i]] = probs[i]; return r; }
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _visionLayerEnd = Layers.Count / 2; } else { Layers.AddRange(LayerHelper<T>.CreateDefaultOpenCLIPLayers(_options.VisionEmbeddingDim, _options.TextEmbeddingDim, _options.ProjectionDim, _options.NumVisionLayers, _options.NumTextLayers, _options.NumVisionHeads, _options.NumTextHeads, _options.DropoutRate)); int lpb = _options.DropoutRate > 0 ? 6 : 5; _visionLayerEnd = 2 + _options.NumVisionLayers * lpb; } }
    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxImageEncoder is not null) return OnnxImageEncoder.Run(input);
        SetTrainingMode(false);
        var c = TokenizeIfNCHW(input);
        foreach (var l in Layers) c = l.Forward(c);
        return c;
    }

    private ConvolutionalLayer<T>? _patchEmbed;

    private Tensor<T> TokenizeIfNCHW(Tensor<T> input) =>
        PatchEmbedHelper.TokenizeImageNCHWToBSC(
            input, _options.VisionEmbeddingDim, _options.ImageSize, ref _patchEmbed, Engine);

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        // Tokenize NCHW image inputs the same way Predict does so the
        // tape-based training loop sees the patch-embedded BSC sequence
        // the encoder layer stack actually expects. Without this, callers
        // (and the model-family invariant tests) that pass an NCHW image
        // would either crash on the first encoder layer's shape check or
        // train against zero gradients flowing through a wrong-shape path.
        TrainWithTape(TokenizeIfNCHW(input), expected);
        SetTrainingMode(false);
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

    // _patchEmbed lives outside Layers but is trainable in native mode. Override
    // ParameterCount / GetParameters / SetParameters / UpdateParameters together
    // so all four agree on the layout: _patchEmbed slice first, then Layers in
    // order. Without this, the optimizer reads N params via GetParameters and
    // hands back N to UpdateParameters — but UpdateParameters consumes
    // _patchEmbed.ParameterCount extra slots from the front, shifting every
    // Layer's slice and corrupting weights.
    public override int ParameterCount =>
        (_patchEmbed?.ParameterCount ?? 0) +
        Layers.Sum(l => l.ParameterCount);

    public override Vector<T> GetParameters()
    {
        var perLayer = Layers.Select(l => l.GetParameters()).ToList();
        int patchLen = _patchEmbed?.ParameterCount ?? 0;
        int total = patchLen + perLayer.Sum(p => p.Length);
        var result = new Vector<T>(total);
        int idx = 0;
        if (patchLen > 0)
        {
            var patchParams = _patchEmbed!.GetParameters();
            for (int i = 0; i < patchParams.Length; i++) result[idx++] = patchParams[i];
        }
        foreach (var p in perLayer)
        {
            for (int i = 0; i < p.Length; i++) result[idx++] = p[i];
        }
        return result;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        // If the saved parameter vector includes patch-embed weights but
        // this instance hasn't seen an image input yet (so _patchEmbed is
        // still null), construct it on-demand so the slice layout matches
        // the saved vector. Without this, deserialize / Clone-from-saved
        // after a vision-mode train silently drops the patch-embed slice
        // and leaves _patchEmbed null — the next image forward would then
        // re-create it with random weights.
        EnsurePatchEmbedForParameterVector(parameters.Length);

        int idx = 0;
        if (_patchEmbed is not null)
        {
            int pc = _patchEmbed.ParameterCount;
            if (pc > 0)
            {
                _patchEmbed.SetParameters(parameters.Slice(idx, pc));
                idx += pc;
            }
        }
        foreach (var l in Layers)
        {
            int c = l.ParameterCount;
            l.SetParameters(parameters.Slice(idx, c));
            idx += c;
        }
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode.");
        EnsurePatchEmbedForParameterVector(parameters.Length);
        int idx = 0;
        if (_patchEmbed is not null)
        {
            int pc = _patchEmbed.ParameterCount;
            if (pc > 0)
            {
                _patchEmbed.UpdateParameters(parameters.Slice(idx, pc));
                idx += pc;
            }
        }
        foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
    }

    /// <summary>
    /// Lazily creates _patchEmbed when the incoming parameter vector is
    /// longer than the layer-sum, indicating the saved model was trained
    /// in vision mode. Builds it by running a probe NCHW tensor through
    /// the helper, which constructs and weight-allocates the conv. Idempotent.
    /// </summary>
    private void EnsurePatchEmbedForParameterVector(int paramVectorLength)
    {
        if (_patchEmbed is not null) return;
        int layerSum = 0;
        foreach (var l in Layers) layerSum += l.ParameterCount;
        if (paramVectorLength <= layerSum) return;

        var probe = new Tensor<T>(new[] { 1, 3, _options.ImageSize, _options.ImageSize });
        TokenizeIfNCHW(probe);
    }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "BiomedCLIP-Native" : "BiomedCLIP-ONNX", Description = "BiomedCLIP: A Multimodal Biomedical Foundation Model (Zhang et al., 2023)", FeatureCount = _options.ProjectionDim, Complexity = _options.NumVisionLayers + _options.NumTextLayers }; m.AdditionalInfo["Architecture"] = "BiomedCLIP"; m.AdditionalInfo["Domain"] = _options.Domain.ToString(); m.AdditionalInfo["Dataset"] = _options.Dataset.ToString(); m.AdditionalInfo["MedicalTextEncoder"] = _options.MedicalTextEncoder; return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ImageEncoderModelPath ?? string.Empty); writer.Write(_options.TextEncoderModelPath ?? string.Empty); writer.Write(_options.ImageSize); writer.Write(_options.VisionEmbeddingDim); writer.Write(_options.TextEmbeddingDim); writer.Write(_options.ProjectionDim); writer.Write(_options.Temperature); writer.Write((int)_options.Domain); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string ip = reader.ReadString(); if (!string.IsNullOrEmpty(ip)) _options.ImageEncoderModelPath = ip; string tp = reader.ReadString(); if (!string.IsNullOrEmpty(tp)) _options.TextEncoderModelPath = tp; _options.ImageSize = reader.ReadInt32(); _options.VisionEmbeddingDim = reader.ReadInt32(); _options.TextEmbeddingDim = reader.ReadInt32(); _options.ProjectionDim = reader.ReadInt32(); _options.Temperature = reader.ReadDouble(); _options.Domain = (DomainSpecialization)reader.ReadInt32(); if (!_useNativeMode && _options.ImageEncoderModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxImageEncoder = new OnnxModel<T>(p, _options.OnnxOptions); if (_options.TextEncoderModelPath is { } t2 && !string.IsNullOrEmpty(t2)) OnnxTextEncoder = new OnnxModel<T>(t2, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ImageEncoderModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new BiomedCLIP<T>(Architecture, mp, _options); return new BiomedCLIP<T>(Architecture, _options); }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var enc = _tokenizer.Encode(text); int sl = Math.Min(enc.TokenIds.Count, _options.MaxSequenceLength); var tk = new Tensor<T>([sl]); for (int i = 0; i < sl; i++) tk[i] = NumOps.FromDouble(enc.TokenIds[i]); return tk; }
    private Tensor<T> ForwardVisionEncoder(Tensor<T> input) { var c = TokenizeIfNCHW(input); for (int i = 0; i < _visionLayerEnd; i++) c = Layers[i].Forward(c); return c; }
    private Tensor<T> ForwardTextEncoder(Tensor<T> tokens) { var c = tokens; for (int i = _visionLayerEnd; i < Layers.Count; i++) c = Layers[i].Forward(c); return c; }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(BiomedCLIP<T>)); }
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing)
        {
            OnnxImageEncoder?.Dispose();
            OnnxTextEncoder?.Dispose();
            // _patchEmbed lives outside Layers (so it doesn't get disposed by the
            // base class's Layers walker). Dispose it here so the conv weights
            // and any registered tensors are released alongside the rest of the
            // encoder when the model is disposed.
            if (_patchEmbed is IDisposable pe) pe.Dispose();
        }
        base.Dispose(disposing);
    }
}
