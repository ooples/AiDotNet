using AiDotNet.Attributes;
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

namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// DFN-CLIP (Data Filtering Networks for CLIP) model using filtered high-quality training data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DFN-CLIP (Fang et al., 2023) uses a small CLIP model to score and filter image-text pairs
/// from a large noisy pool, then trains a larger model on only high-quality data. Achieves 83.0%
/// zero-shot on ImageNet with ViT-H/14.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Data Filtering Networks" (Fang et al., 2023)</item></list></para>
/// <para><b>For Beginners:</b> DFN-CLIP improves CLIP by using a small trained CLIP model to
/// score and filter image-text pairs from a massive noisy dataset, keeping only high-quality
/// pairs. The larger model trained on this filtered data achieves 83.0% zero-shot ImageNet
/// accuracy with ViT-H/14, demonstrating that data quality matters more than quantity.
/// Default values follow the original paper settings.</para>
/// <para><b>Architecture layout:</b> Mirrors the PyTorch / HuggingFace CLIP layout —
/// vision encoder (patch embedding + transformer + projection) lives in
/// <see cref="NeuralNetworkBase{T}.Layers"/> so the default forward and tape-based
/// training paths walk it correctly. The text encoder lives in a separate field
/// surfaced through <c>GetExtraTrainableLayers</c> so it participates in
/// streaming offload / weight-registry hooks but isn't on the vision-only forward
/// path. Real contrastive training is done through paired-data APIs
/// (<see cref="ComputeSimilarity"/> / <see cref="ZeroShotClassify"/>); the
/// inherited <see cref="Predict"/> + <see cref="Train(Tensor{T},Tensor{T})"/>
/// surface is the vision-encoder-only path that lets CLIP plug into
/// generic NN consumers without crashing on cross-stack layer walks.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a DFN-CLIP model trained on filtered high-quality data
/// // using data filtering networks for improved CLIP training
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new DFNCLIP&lt;double&gt;(architecture, "dfnclip.onnx");
///
/// // Training mode with native layers
/// var trainModel = new DFNCLIP&lt;double&gt;(architecture, new DFNCLIPOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Data Filtering Networks", "https://arxiv.org/abs/2309.17425", Year = 2023, Authors = "Fang et al.")]
public class DFNCLIP<T> : VisionLanguageModelBase<T>, IContrastiveVisionLanguageModel<T>
{
    private readonly DFNCLIPOptions _options;
    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;

    // Text encoder lives outside `Layers` so the inherited forward / tape paths
    // (Predict, TrainWithTape, GetParameters, UpdateParameters) operate on the
    // vision-only stack — feeding vision features through text-encoder layers
    // reliably mismatches dims (visionEmbeddingDim != textEmbeddingDim in OpenCLIP)
    // and was the root cause of the family-wide invariant-test failures.
    // Surfaced via `GetExtraTrainableLayers` so streaming offload and the weight
    // registry still see these weights.
    private readonly List<ILayer<T>> _textEncoderLayers = new List<ILayer<T>>();

    public DFNCLIP(
        NeuralNetworkArchitecture<T> architecture,
        string imageEncoderModelPath,
        DFNCLIPOptions? options = null) : base(architecture)
    {
        _options = options ?? new DFNCLIPOptions();
        SyncImageSizeWithArchitecture();
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
            if (!File.Exists(tp))
                throw new FileNotFoundException($"Text ONNX not found: {tp}", tp);
            OnnxTextEncoder = new OnnxModel<T>(tp, _options.OnnxOptions);
        }
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    public DFNCLIP(
        NeuralNetworkArchitecture<T> architecture,
        DFNCLIPOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new DFNCLIPOptions();
        SyncImageSizeWithArchitecture();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.VisionEmbeddingDim;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    /// <summary>
    /// Override the layer-stack starting shape so lazy
    /// <see cref="LayerNormalizationLayer{T}"/> / <see cref="DenseLayer{T}"/>
    /// resolve to the post-patch-embed channel dim instead of the raw NCHW
    /// spatial dim. See <see cref="BiomedCLIP{T}"/> override for full
    /// rationale — same patch-grid math
    /// (<c>patchSize = max(1, imageSize / 16)</c>,
    /// <c>tokens = (imageSize / patchSize)²</c>) and same OpenCLIP layer
    /// stack downstream of <see cref="PatchEmbedHelper.TokenizeImageNCHWToBSC"/>.
    /// </summary>
    protected override int[]? TryGetArchitectureInputShape()
    {
        int imageSize = _options.ImageSize;
        if (imageSize <= 0) return base.TryGetArchitectureInputShape();
        int patchSize = System.Math.Max(1, imageSize / 16);
        int tokens = (imageSize / patchSize) * (imageSize / patchSize);
        return new[] { 1, tokens, _options.VisionEmbeddingDim };
    }

    /// <summary>
    /// Aligns <c>_options.ImageSize</c> with <c>Architecture.InputHeight</c> when
    /// the architecture declares an explicit square spatial extent. The paper-
    /// faithful default (224) is fine when the architecture leaves H/W unset
    /// (= 0); a CI-fast architecture at 128×128 must drive imageSize=128 or the
    /// patch grid (imageSize / 16) mismatches the actual input and PatchEmbedding
    /// resolves the wrong number of patches.
    /// </summary>
    private void SyncImageSizeWithArchitecture()
    {
        int h = Architecture.InputHeight;
        int w = Architecture.InputWidth;
        if (h > 0 && w > 0 && h == w) _options.ImageSize = h;
    }

    public int EmbeddingDimension => _options.VisionEmbeddingDim;
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;
    int IVisualEncoder<T>.ImageChannels => 3;
    public int MaxSequenceLength => _options.MaxSequenceLength;
    public int TextEmbeddingDimension => _options.TextEmbeddingDim;
    public int ProjectionDimension => _options.ProjectionDim;
    public T Temperature => NumOps.FromDouble(_options.Temperature);

    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxImageEncoder is not null)
            return L2Normalize(OnnxImageEncoder.Run(p));
        var current = p;
        foreach (var layer in Layers)
            current = layer.Forward(current);
        return L2Normalize(current);
    }

    public Tensor<T> EncodeText(string text)
    {
        ThrowIfDisposed();
        var t = TokenizeText(text);
        if (IsOnnxMode && OnnxTextEncoder is not null)
            return L2Normalize(OnnxTextEncoder.Run(t));
        var current = t;
        foreach (var layer in _textEncoderLayers)
            current = layer.Forward(current);
        return L2Normalize(current);
    }

    public Tensor<T>[] EncodeTexts(string[] texts)
    {
        var e = new Tensor<T>[texts.Length];
        for (int i = 0; i < texts.Length; i++) e[i] = EncodeText(texts[i]);
        return e;
    }

    public T ComputeSimilarity(Tensor<T> image, string text) =>
        CosineSimilarity(EncodeImage(image), EncodeText(text));

    public Dictionary<string, T> ZeroShotClassify(Tensor<T> image, string[] labels)
    {
        var ie = EncodeImage(image);
        var te = EncodeTexts(labels);
        var logits = new Tensor<T>([labels.Length]);
        double temp = _options.Temperature;
        for (int i = 0; i < labels.Length; i++)
            logits[i] = NumOps.FromDouble(NumOps.ToDouble(CosineSimilarity(ie, te[i])) / temp);
        var probs = Softmax(logits);
        var r = new Dictionary<string, T>();
        for (int i = 0; i < labels.Length; i++) r[labels[i]] = probs[i];
        return r;
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            // Caller-supplied layer graph: keep historic single-list behaviour.
            // Caller is responsible for a self-consistent topology.
            Layers.AddRange(Architecture.Layers);
            return;
        }

        // ViT patch embedding (Dosovitskiy et al. 2021 §3.1): NCHW image →
        // [B, num_patches, visionEmbeddingDim] tokens. Driven by ImageSize so a
        // CI-fast 128² architecture and a paper-faithful 224² production
        // architecture produce the same downstream token rank.
        int patchSize = Math.Max(1, _options.ImageSize / 16);
        Layers.Add(new PatchEmbeddingLayer<T>(
            patchSize: patchSize,
            embeddingDim: _options.VisionEmbeddingDim,
            expectedInputChannels: 3));

        // OpenCLIP factory emits [vision: pre-norm, N×block, projection,
        //                         text:  pre-norm, N×block, projection].
        // Block size = 5 layers (or 6 with dropout). Split index = 2 + N×blockSize.
        int blockSize = _options.DropoutRate > 0 ? 6 : 5;
        int visionLayerCount = 2 + _options.NumVisionLayers * blockSize;
        var openClip = LayerHelper<T>.CreateDefaultOpenCLIPLayers(
            visionEmbeddingDim: _options.VisionEmbeddingDim,
            textEmbeddingDim: _options.TextEmbeddingDim,
            projectionDim: _options.ProjectionDim,
            numVisionLayers: _options.NumVisionLayers,
            numTextLayers: _options.NumTextLayers,
            numVisionHeads: _options.NumVisionHeads,
            numTextHeads: _options.NumTextHeads,
            dropoutRate: _options.DropoutRate);
        int idx = 0;
        foreach (var layer in openClip)
        {
            if (idx < visionLayerCount) Layers.Add(layer);
            else _textEncoderLayers.Add(layer);
            idx++;
        }
    }

    /// <summary>
    /// Vision-only forward: walks <see cref="NeuralNetworkBase{T}.Layers"/>
    /// (patch embedding → vision transformer → vision projection) on the
    /// preprocessed image. The text-encoder stack is reachable through
    /// <see cref="EncodeText"/>; the contrastive image-text path goes
    /// through <see cref="ComputeSimilarity"/>.
    /// </summary>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxImageEncoder is not null) return OnnxImageEncoder.Run(input);
        SetTrainingMode(false);
        var current = PreprocessImage(input);
        foreach (var layer in Layers)
            current = layer.Forward(current);
        return current;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        // Forward-path entrypoint matches Predict (PreprocessImage → Layers).
        // TrainWithTape walks Layers, so the vision-only graph is what gets
        // gradients + parameter updates here. The contrastive paired-data
        // training path that updates both encoders is a separate API
        // (TrainContrastive(images, texts) — to be added) and is not what
        // generic-NN consumers expect from Train(input, target).
        TrainWithTape(PreprocessImage(input), expected);
        SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode.");
        int idx = 0;
        foreach (var l in Layers)
        {
            int c = (int)l.ParameterCount;
            l.UpdateParameters(parameters.Slice(idx, c));
            idx += c;
        }
    }

    /// <summary>
    /// Surface the text-encoder stack to streaming-offload / weight-registry
    /// hooks (NeuralNetworkBase.GetExtraTrainableLayers contract) without
    /// extending the flat parameter APIs (GetParameters / ParameterCount /
    /// SetParameters) — those keep the SCOPE CONTRACT (= Layers only) so
    /// flat-vector consumers don't accidentally double-count.
    /// </summary>
    protected override IEnumerable<LayerBase<T>?> GetExtraTrainableLayers()
    {
        foreach (var layer in _textEncoderLayers)
            if (layer is LayerBase<T> lb) yield return lb;
    }

    protected override Tensor<T> PreprocessImage(Tensor<T> image) =>
        NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "DFN-CLIP-Native" : "DFN-CLIP-ONNX",
            Description = "DFN-CLIP: Data Filtering Networks (Fang et al., 2023)",
            FeatureCount = _options.ProjectionDim,
            Complexity = _options.NumVisionLayers + _options.NumTextLayers
        };
        m.AdditionalInfo["Architecture"] = "DFN-CLIP";
        m.AdditionalInfo["FilteringThreshold"] = _options.FilteringThreshold.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.ImageEncoderModelPath ?? string.Empty);
        writer.Write(_options.TextEncoderModelPath ?? string.Empty);
        writer.Write(_options.ImageSize);
        writer.Write(_options.VisionEmbeddingDim);
        writer.Write(_options.TextEmbeddingDim);
        writer.Write(_options.ProjectionDim);
        writer.Write(_options.Temperature);
        writer.Write(_options.FilteringThreshold);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string ip = reader.ReadString();
        if (!string.IsNullOrEmpty(ip)) _options.ImageEncoderModelPath = ip;
        string tp = reader.ReadString();
        if (!string.IsNullOrEmpty(tp)) _options.TextEncoderModelPath = tp;
        _options.ImageSize = reader.ReadInt32();
        _options.VisionEmbeddingDim = reader.ReadInt32();
        _options.TextEmbeddingDim = reader.ReadInt32();
        _options.ProjectionDim = reader.ReadInt32();
        _options.Temperature = reader.ReadDouble();
        _options.FilteringThreshold = reader.ReadDouble();
        if (!_useNativeMode && _options.ImageEncoderModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxImageEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
        if (_options.TextEncoderModelPath is { } t2 && !string.IsNullOrEmpty(t2))
            OnnxTextEncoder = new OnnxModel<T>(t2, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ImageEncoderModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new DFNCLIP<T>(Architecture, mp, _options);
        return new DFNCLIP<T>(Architecture, _options);
    }

    private Tensor<T> TokenizeText(string text)
    {
        if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized.");
        var enc = _tokenizer.Encode(text);
        int sl = Math.Min(enc.TokenIds.Count, _options.MaxSequenceLength);
        var tk = new Tensor<T>([sl]);
        for (int i = 0; i < sl; i++) tk[i] = NumOps.FromDouble(enc.TokenIds[i]);
        return tk;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(DFNCLIP<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing)
        {
            OnnxImageEncoder?.Dispose();
            OnnxTextEncoder?.Dispose();
        }
        base.Dispose(disposing);
    }
}
