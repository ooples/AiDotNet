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
/// <para><b>Architecture layout:</b> Mirrors PyTorch / HuggingFace CLIP — vision encoder (patch
/// embedding + transformer + projection) lives in <see cref="NeuralNetworkBase{T}.Layers"/>;
/// text encoder lives in a separate field surfaced through <c>GetExtraTrainableLayers</c>.
/// The default <see cref="Predict"/> + <see cref="Train(Tensor{T},Tensor{T})"/> path is
/// vision-only — it patch-embeds the input and walks <c>Layers</c>. Text-encoder access goes
/// through <see cref="EncodeText"/>; contrastive image-text similarity through
/// <see cref="ComputeSimilarity"/> / <see cref="ZeroShotClassify"/>.</para>
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
[ResearchPaper(
    "BiomedCLIP: A Multimodal Biomedical Foundation Model Pretrained from Fifteen Million Scientific Image-Text Pairs",
    "https://arxiv.org/abs/2303.00915",
    Year = 2023,
    Authors = "Zhang et al."
)]
public class BiomedCLIP<T> : VisionLanguageModelBase<T>, IContrastiveVisionLanguageModel<T>
{
    private readonly BiomedCLIPOptions _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;

    // Text encoder lives outside `Layers`. See DFNCLIP for the rationale —
    // visionEmbeddingDim and textEmbeddingDim need not match (and don't, in
    // OpenCLIP-defaults), so feeding vision-projection output through text-
    // encoder layers walks into a hard dim mismatch.
    private readonly List<ILayer<T>> _textEncoderLayers = new List<ILayer<T>>();

    public BiomedCLIP(
        NeuralNetworkArchitecture<T> architecture,
        string imageEncoderModelPath,
        BiomedCLIPOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new BiomedCLIPOptions();
        SyncImageSizeWithArchitecture();
        _useNativeMode = false;
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.VisionEmbeddingDim;
        if (string.IsNullOrWhiteSpace(imageEncoderModelPath))
            throw new ArgumentException(
                "Image encoder model path cannot be null or empty.",
                nameof(imageEncoderModelPath)
            );
        if (!File.Exists(imageEncoderModelPath))
            throw new FileNotFoundException(
                $"ONNX model not found: {imageEncoderModelPath}",
                imageEncoderModelPath
            );
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

    public BiomedCLIP(
        NeuralNetworkArchitecture<T> architecture,
        BiomedCLIPOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new BiomedCLIPOptions();
        SyncImageSizeWithArchitecture();
        _useNativeMode = true;
        // Paper-faithful CLIP training (Radford et al. 2021 §5.1, §A.1):
        // AdamW with β₁=0.9, β₂=0.98, ε=1e-6, weight_decay=0.2,
        // base learning rate = 5e-4. Default Adam at lr=1e-3 without
        // warmup oscillates on paper-scale ViT-B/16 — Training_ShouldReduceLoss
        // saw loss diverge from 1.249 → 1.801 over 30 unscheduled steps
        // because the first-step Adam update overshoots before the moments
        // stabilize. Paper-faithful lr=5e-4 with AdamW's decoupled weight
        // decay keeps the trajectory monotonically decreasing in MSE under
        // the test framework's default fixed-target regression contract.
        _optimizer =
            optimizer
            ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
                this,
                new Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = 5e-4,
                    Beta1 = 0.9,
                    Beta2 = 0.98,
                    Epsilon = 1e-6,
                    WeightDecay = 0.2,
                }
            );
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.VisionEmbeddingDim;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    /// <summary>
    /// Aligns <c>_options.ImageSize</c> with <c>Architecture.InputHeight</c> when
    /// the architecture declares an explicit square spatial extent. Same
    /// rationale as DFNCLIP — a CI-fast architecture at 128×128 must drive
    /// imageSize=128 or the patch grid mismatches the runtime input.
    /// </summary>
    private void SyncImageSizeWithArchitecture()
    {
        int h = Architecture.InputHeight;
        int w = Architecture.InputWidth;
        if (h > 0 && w > 0 && h == w)
            _options.ImageSize = h;
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
        if (IsOnnxMode)
        {
            if (OnnxImageEncoder is null)
                throw new InvalidOperationException(
                    "Image encoding in ONNX mode requires a configured image encoder model path. "
                        + "Provide one via the ONNX-mode constructor or BiomedCLIPOptions.ImageEncoderModelPath."
                );
            return L2Normalize(OnnxImageEncoder.Run(p));
        }
        return L2Normalize(ForwardVisionEncoder(p));
    }

    public Tensor<T> EncodeText(string text)
    {
        ThrowIfDisposed();
        var t = TokenizeText(text);
        // Fail fast in ONNX mode when no text encoder is configured. The
        // ONNX-mode constructor never populates _textEncoderLayers, so the
        // previous fallback to ForwardTextEncoder ran the un-trained
        // (empty) text stack and L2-normalized normalized token IDs —
        // ComputeSimilarity / ZeroShotClassify silently returned garbage
        // instead of surfacing the configuration error.
        if (IsOnnxMode)
        {
            if (OnnxTextEncoder is null)
                throw new InvalidOperationException(
                    "Text encoding in ONNX mode requires a configured text encoder model path. "
                        + "Set BiomedCLIPOptions.TextEncoderModelPath before constructing the model."
                );
            return L2Normalize(OnnxTextEncoder.Run(t));
        }
        return L2Normalize(ForwardTextEncoder(t));
    }

    public Tensor<T>[] EncodeTexts(string[] texts)
    {
        var e = new Tensor<T>[texts.Length];
        for (int i = 0; i < texts.Length; i++)
            e[i] = EncodeText(texts[i]);
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
        for (int i = 0; i < labels.Length; i++)
            r[labels[i]] = probs[i];
        return r;
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            // Caller-supplied layer graph: keep historic single-list behaviour.
            Layers.AddRange(Architecture.Layers);
            return;
        }

        // ViT patch embedding (Dosovitskiy et al. 2021 §3.1) as Layers[0]:
        // turns NCHW image input into [B, num_patches, visionEmbeddingDim].
        // Putting the patch-embed inside Layers (matching DFNCLIP and the
        // rest of the OpenCLIP-style encoders) means the base class's
        // Predict / Train / GetNamedLayerActivations / ResolveLazyLayerShapes
        // / Serialize / Clone all walk the full vision graph by default —
        // no per-method override needed to teach them about the
        // tokenization step. patchSize = max(1, imageSize / 16) matches
        // PatchEmbedHelper.TokenizeImageNCHWToBSC so a CI-fast 128² and a
        // paper-faithful 224² architecture both produce the same
        // downstream token rank.
        int patchSize = System.Math.Max(1, _options.ImageSize / 16);
        Layers.Add(
            new PatchEmbeddingLayer<T>(
                patchSize: patchSize,
                embeddingDim: _options.VisionEmbeddingDim,
                expectedInputChannels: 3
            )
        );

        // Split OpenCLIP factory output: vision portion → Layers, text → _textEncoderLayers.
        // Block size = 5 layers (or 6 with dropout). Vision block count = 2 + N×blockSize
        // (pre-norm + N transformer blocks + projection); text the same.
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
            dropoutRate: _options.DropoutRate
        );
        int idx = 0;
        foreach (var layer in openClip)
        {
            if (idx < visionLayerCount)
                Layers.Add(layer);
            else
                _textEncoderLayers.Add(layer);
            idx++;
        }
    }

    /// <summary>
    /// Vision-only forward: walks <see cref="NeuralNetworkBase{T}.Layers"/>
    /// (patch embedding → vision transformer → vision projection) on the
    /// preprocessed image. The text-encoder stack is reachable through
    /// <see cref="EncodeText"/>; the contrastive image-text path goes
    /// through <see cref="ComputeSimilarity"/> / <see cref="ZeroShotClassify"/>.
    /// </summary>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        // Both paths must see the same preprocessed input: native path forwards
        // through Layers below, ONNX path runs the same encoder under
        // OnnxImageEncoder. Skipping PreprocessImage on the ONNX branch makes
        // the two paths produce different mean/std-offset inputs to the model.
        var current = PreprocessImage(input);
        if (IsOnnxMode && OnnxImageEncoder is not null)
            return OnnxImageEncoder.Run(current);
        SetTrainingMode(false);
        foreach (var l in Layers)
            current = l.Forward(current);
        return current;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        ThrowIfDisposed();
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            // PatchEmbeddingLayer is now Layers[0]; the tape-based training
            // loop walks the full vision graph (NCHW image → patch-embed →
            // transformer → projection) end-to-end so gradients flow back
            // through the patch-embed conv too. Pass our paper-faithful
            // AdamW optimizer (β₂=0.98, weight_decay=0.2, lr=5e-4) instead
            // of the base class's default Adam (lr=1e-3) — see ctor for why.
            TrainWithTape(input, expected, _optimizer);
        }
        finally
        {
            // try/finally so a TrainWithTape throw doesn't leave the model
            // stuck in training mode.
            SetTrainingMode(false);
        }
    }

    /// <summary>
    /// Surfaces the text-encoder stack to the base weight-registry walker
    /// (streaming-offload / weight-pool hooks) without extending the flat
    /// parameter APIs (GetParameters / ParameterCount / SetParameters) —
    /// those keep the SCOPE CONTRACT (= Layers only, which now includes
    /// the vision patch-embed too) so flat-vector consumers don't
    /// accidentally double-count.
    /// </summary>
    protected override IEnumerable<LayerBase<T>?> GetExtraTrainableLayers()
    {
        foreach (var layer in _textEncoderLayers)
            if (layer is LayerBase<T> lb)
                yield return lb;
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
    }

    protected override Tensor<T> PreprocessImage(Tensor<T> image) =>
        NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "BiomedCLIP-Native" : "BiomedCLIP-ONNX",
            Description =
                "BiomedCLIP: A Multimodal Biomedical Foundation Model (Zhang et al., 2023)",
            FeatureCount = _options.ProjectionDim,
            Complexity = _options.NumVisionLayers + _options.NumTextLayers,
        };
        m.AdditionalInfo["Architecture"] = "BiomedCLIP";
        m.AdditionalInfo["Domain"] = _options.Domain.ToString();
        m.AdditionalInfo["Dataset"] = _options.Dataset.ToString();
        m.AdditionalInfo["MedicalTextEncoder"] = _options.MedicalTextEncoder;
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
        writer.Write((int)_options.Domain);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string ip = reader.ReadString();
        if (!string.IsNullOrEmpty(ip))
            _options.ImageEncoderModelPath = ip;
        string tp = reader.ReadString();
        if (!string.IsNullOrEmpty(tp))
            _options.TextEncoderModelPath = tp;
        _options.ImageSize = reader.ReadInt32();
        _options.VisionEmbeddingDim = reader.ReadInt32();
        _options.TextEmbeddingDim = reader.ReadInt32();
        _options.ProjectionDim = reader.ReadInt32();
        _options.Temperature = reader.ReadDouble();
        _options.Domain = (DomainSpecialization)reader.ReadInt32();
        if (!_useNativeMode && _options.ImageEncoderModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxImageEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
        if (_options.TextEncoderModelPath is { } t2 && !string.IsNullOrEmpty(t2))
            OnnxTextEncoder = new OnnxModel<T>(t2, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (
            !_useNativeMode
            && _options.ImageEncoderModelPath is { } mp
            && !string.IsNullOrEmpty(mp)
        )
            return new BiomedCLIP<T>(Architecture, mp, _options);
        return new BiomedCLIP<T>(Architecture, _options);
    }

    private Tensor<T> TokenizeText(string text)
    {
        if (_tokenizer is null)
            throw new InvalidOperationException("Tokenizer not initialized.");
        var enc = _tokenizer.Encode(text);
        int sl = Math.Min(enc.TokenIds.Count, _options.MaxSequenceLength);
        var tk = new Tensor<T>([sl]);
        for (int i = 0; i < sl; i++)
            tk[i] = NumOps.FromDouble(enc.TokenIds[i]);
        return tk;
    }

    /// <summary>
    /// Vision encoder forward (patch embedding + transformer + projection).
    /// Walks <see cref="NeuralNetworkBase{T}.Layers"/> end-to-end on NCHW
    /// input — Layers[0] is now the <see cref="PatchEmbeddingLayer{T}"/>.
    /// Used by <see cref="EncodeImage"/>.
    /// </summary>
    private Tensor<T> ForwardVisionEncoder(Tensor<T> input)
    {
        var c = input;
        foreach (var layer in Layers)
            c = layer.Forward(c);
        return c;
    }

    /// <summary>
    /// Text encoder forward (transformer + projection). Walks
    /// <see cref="_textEncoderLayers"/>. Used by <see cref="EncodeText"/>.
    /// </summary>
    private Tensor<T> ForwardTextEncoder(Tensor<T> tokens)
    {
        var c = tokens;
        foreach (var layer in _textEncoderLayers)
            c = layer.Forward(c);
        return c;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(BiomedCLIP<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        if (disposing)
        {
            OnnxImageEncoder?.Dispose();
            OnnxTextEncoder?.Dispose();
        }
        base.Dispose(disposing);
    }
}
