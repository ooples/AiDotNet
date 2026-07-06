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
/// BASIC (Combined Scaling for Zero-shot Transfer Learning) model using CoAtNet vision encoder.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// BASIC (Pham et al., 2022) combines CoAtNet (Convolution + Attention) vision encoder with
/// large-scale contrastive image-text pre-training, achieving 85.7% zero-shot ImageNet via
/// architecture+data+batch-size combined scaling.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Combined Scaling for Zero-shot Transfer Learning" (Pham et al., 2022)</item></list></para>
/// <para><b>For Beginners:</b> BASIC pushes CLIP performance higher by combining three things at
/// scale: a stronger vision architecture (CoAtNet, which mixes convolutions and attention),
/// a larger dataset, and bigger batch sizes during training. The result is 85.7% zero-shot
/// ImageNet accuracy. Default values follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
/// var trainModel = new BASIC&lt;double&gt;(architecture, new BASICOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Combined Scaling for Zero-shot Transfer Learning",
    "https://arxiv.org/abs/2111.10050",
    Year = 2022,
    Authors = "Pham et al."
)]
public class BASIC<T> : VisionLanguageModelBase<T>, IContrastiveVisionLanguageModel<T>
{
    private readonly BASICOptions _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;

    public BASIC(
        NeuralNetworkArchitecture<T> architecture,
        string imageEncoderModelPath,
        BASICOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new BASICOptions();
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
                throw new FileNotFoundException($"Text encoder ONNX model not found: {tp}", tp);
            OnnxTextEncoder = new OnnxModel<T>(tp, _options.OnnxOptions);
        }

        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    public BASIC(
        NeuralNetworkArchitecture<T> architecture,
        BASICOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new BASICOptions();
        SyncImageSizeWithArchitecture();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.VisionEmbeddingDim;
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
        var preprocessed = PreprocessImage(image);
        if (IsOnnxMode && OnnxImageEncoder is not null)
            return L2Normalize(OnnxImageEncoder.Run(preprocessed));
        var c = preprocessed;
        foreach (var l in Layers)
            c = l.Forward(c);
        return L2Normalize(c);
    }

    public Tensor<T> EncodeText(string text)
    {
        ThrowIfDisposed();
        var tokenized = TokenizeText(text);
        if (IsOnnxMode && OnnxTextEncoder is not null)
            return L2Normalize(OnnxTextEncoder.Run(tokenized));
        var c = tokenized;
        foreach (var l in TextEncoderLayers)
            c = l.Forward(c);
        return L2Normalize(c);
    }

    public Tensor<T>[] EncodeTexts(string[] texts)
    {
        var embeddings = new Tensor<T>[texts.Length];
        for (int i = 0; i < texts.Length; i++)
            embeddings[i] = EncodeText(texts[i]);
        return embeddings;
    }

    public T ComputeSimilarity(Tensor<T> image, string text) =>
        CosineSimilarity(EncodeImage(image), EncodeText(text));

    public Dictionary<string, T> ZeroShotClassify(Tensor<T> image, string[] labels)
    {
        var imageEmb = EncodeImage(image);
        var textEmbs = EncodeTexts(labels);
        var logits = new Tensor<T>([labels.Length]);
        double temp = _options.Temperature;
        for (int i = 0; i < labels.Length; i++)
            logits[i] = NumOps.FromDouble(
                NumOps.ToDouble(CosineSimilarity(imageEmb, textEmbs[i])) / temp
            );
        var probs = Softmax(logits);
        var result = new Dictionary<string, T>();
        for (int i = 0; i < labels.Length; i++)
            result[labels[i]] = probs[i];
        return result;
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture is DualStreamArchitecture<T> dual)
        {
            Layers.AddRange(dual.VisionLayers);
            TextEncoderLayers.AddRange(dual.TextLayers);
            return;
        }

        // BASIC vision uses CoAtNet (CNN stages → Transformer stages):
        //   CNN stage  : Dense + LN + Dense + LN + [Dropout]   (4 or 5 layers)
        //   Trans stage: MHA + LN + Dense + Dense + LN + [Dropout]   (5 or 6 layers)
        // First half of vision layers is CNN, second half is Transformer.
        // Vision encoder lives in Layers, text encoder in TextEncoderLayers.
        int cnnStages = _options.NumVisionLayers / 2;
        int transformerStages = _options.NumVisionLayers - cnnStages;
        int cnnLpb = _options.DropoutRate > 0 ? 5 : 4;
        int transformerLpb = _options.DropoutRate > 0 ? 6 : 5;
        int visionLayerCount = 1 + cnnStages * cnnLpb + transformerStages * transformerLpb + 1;
        SplitDualStreamLayers(
            LayerHelper<T>.CreateDefaultBASICLayers(
                visionEmbeddingDim: _options.VisionEmbeddingDim,
                textEmbeddingDim: _options.TextEmbeddingDim,
                projectionDim: _options.ProjectionDim,
                numVisionLayers: _options.NumVisionLayers,
                numTextLayers: _options.NumTextLayers,
                numVisionHeads: _options.NumVisionHeads,
                numTextHeads: _options.NumTextHeads,
                dropoutRate: _options.DropoutRate
            ),
            visionLayerCount
        );
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        // Normalize ONNX inputs the same way the native path and
        // EncodeImage do — both call PreprocessImage. Without this the
        // ONNX fast path would diverge silently from native (different
        // mean / std offsets reach the model), which is hard to debug.
        var current = PreprocessImage(input);
        if (IsOnnxMode && OnnxImageEncoder is not null)
            return OnnxImageEncoder.Run(current);
        SetTrainingMode(false);
        foreach (var layer in Layers)
            current = layer.Forward(current);
        return current;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            TrainWithTape(PreprocessImage(input), expected, _optimizer);
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
        foreach (var layer in Layers)
        {
            int count = (int)layer.ParameterCount;
            layer.UpdateParameters(parameters.Slice(idx, count));
            idx += count;
        }
        // After the dual-stream split (vision in Layers, text in
        // TextEncoderLayers via VisionLanguageModelBase), text encoder
        // weights live outside Layers but ParameterCount / GetParameters
        // include them. Apply the trailing parameter slice to the text
        // stream too — without this, SetParameters-style flows leave the
        // text encoder on its old weights and the model state silently
        // de-syncs across the two streams.
        foreach (var layer in TextEncoderLayers)
        {
            int count = (int)layer.ParameterCount;
            layer.UpdateParameters(parameters.Slice(idx, count));
            idx += count;
        }
    }

    /// <inheritdoc />
    protected override IEnumerable<LayerBase<T>?> GetExtraTrainableLayers() =>
        EnumerateTextEncoderTrainableLayers();

    protected override Tensor<T> PreprocessImage(Tensor<T> image) =>
        NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var meta = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "BASIC-Native" : "BASIC-ONNX",
            Description =
                "BASIC: Combined Scaling for Zero-shot Transfer Learning (Pham et al., 2022)",
            FeatureCount = _options.ProjectionDim,
            Complexity = _options.NumVisionLayers + _options.NumTextLayers,
        };
        meta.AdditionalInfo["Architecture"] = "BASIC";
        meta.AdditionalInfo["VisionEncoder"] = "CoAtNet";
        meta.AdditionalInfo["ProjectionDim"] = _options.ProjectionDim.ToString();
        return meta;
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
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string imgPath = reader.ReadString();
        if (!string.IsNullOrEmpty(imgPath))
            _options.ImageEncoderModelPath = imgPath;
        string txtPath = reader.ReadString();
        if (!string.IsNullOrEmpty(txtPath))
            _options.TextEncoderModelPath = txtPath;
        _options.ImageSize = reader.ReadInt32();
        _options.VisionEmbeddingDim = reader.ReadInt32();
        _options.TextEmbeddingDim = reader.ReadInt32();
        _options.ProjectionDim = reader.ReadInt32();
        _options.Temperature = reader.ReadDouble();
        if (!_useNativeMode && _options.ImageEncoderModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxImageEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
        if (_options.TextEncoderModelPath is { } tp2 && !string.IsNullOrEmpty(tp2))
            OnnxTextEncoder = new OnnxModel<T>(tp2, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (
            !_useNativeMode
            && _options.ImageEncoderModelPath is { } mp
            && !string.IsNullOrEmpty(mp)
        )
            return new BASIC<T>(Architecture, mp, _options);
        return new BASIC<T>(Architecture, _options);
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

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(BASIC<T>));
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
