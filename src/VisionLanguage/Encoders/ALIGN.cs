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
/// ALIGN (A Large-scale ImaGe and Noisy-text embedding) model for zero-shot classification
/// and cross-modal retrieval using EfficientNet as the vision encoder.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ALIGN (Jia et al., ICML 2021) demonstrates that simple dual-encoder contrastive learning
/// achieves strong image-text alignment when trained at massive scale. Unlike CLIP which uses
/// a Vision Transformer, ALIGN uses an EfficientNet-B7 as its vision encoder, showing that
/// the contrastive learning recipe is architecture-agnostic.
/// </para>
/// <para>
/// <b>Key Innovation:</b> ALIGN trains on 1.8 billion noisy alt-text image-text pairs without
/// expensive filtering, showing that scale compensates for data noise. The EfficientNet backbone
/// provides a CNN-based alternative to ViT for vision encoding.
/// </para>
/// <para>
/// <b>For Beginners:</b> ALIGN is similar to CLIP but uses a different type of image model
/// (EfficientNet, which is a convolutional neural network) instead of a Vision Transformer.
/// It was trained on a very large but noisy dataset, proving that more data beats cleaner data.
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision" (Jia et al., ICML 2021)</item>
/// </list>
/// </para>
/// </remarks>
public class ALIGN<T> : VisionLanguageModelBase<T>, IContrastiveVisionLanguageModel<T>
{
    private readonly ALIGNOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _visionLayerEnd;

    /// <summary>
    /// Creates an ALIGN model in ONNX inference mode.
    /// </summary>
    public ALIGN(NeuralNetworkArchitecture<T> architecture, string imageEncoderModelPath, ALIGNOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new ALIGNOptions();
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
                throw new FileNotFoundException($"Text encoder ONNX model not found: {tp}", tp);
            OnnxTextEncoder = new OnnxModel<T>(tp, _options.OnnxOptions);
        }

        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    /// <summary>
    /// Creates an ALIGN model in native training mode.
    /// </summary>
    public ALIGN(NeuralNetworkArchitecture<T> architecture, ALIGNOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new ALIGNOptions();
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

    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessImage(image);
        if (IsOnnxMode && OnnxImageEncoder is not null)
            return L2Normalize(OnnxImageEncoder.Run(preprocessed));
        return L2Normalize(ForwardVisionEncoder(preprocessed));
    }

    public Tensor<T> EncodeText(string text)
    {
        ThrowIfDisposed();
        var tokenized = TokenizeText(text);
        if (IsOnnxMode && OnnxTextEncoder is not null)
            return L2Normalize(OnnxTextEncoder.Run(tokenized));
        return L2Normalize(ForwardTextEncoder(tokenized));
    }

    public Tensor<T>[] EncodeTexts(string[] texts)
    {
        var embeddings = new Tensor<T>[texts.Length];
        for (int i = 0; i < texts.Length; i++)
            embeddings[i] = EncodeText(texts[i]);
        return embeddings;
    }

    public T ComputeSimilarity(Tensor<T> image, string text)
    {
        var imageEmb = EncodeImage(image);
        var textEmb = EncodeText(text);
        return CosineSimilarity(imageEmb, textEmb);
    }

    public Dictionary<string, T> ZeroShotClassify(Tensor<T> image, string[] labels)
    {
        var imageEmb = EncodeImage(image);
        var textEmbs = EncodeTexts(labels);
        var logits = new Tensor<T>([labels.Length]);
        double temp = _options.Temperature;
        for (int i = 0; i < labels.Length; i++)
        {
            double sim = NumOps.ToDouble(CosineSimilarity(imageEmb, textEmbs[i]));
            logits[i] = NumOps.FromDouble(sim / temp);
        }
        var probs = Softmax(logits);
        var result = new Dictionary<string, T>();
        for (int i = 0; i < labels.Length; i++)
            result[labels[i]] = probs[i];
        return result;
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _visionLayerEnd = Layers.Count / 2;
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultALIGNLayers(
                visionEmbeddingDim: _options.VisionEmbeddingDim,
                textEmbeddingDim: _options.TextEmbeddingDim,
                projectionDim: _options.ProjectionDim,
                numVisionLayers: _options.NumVisionLayers,
                numTextLayers: _options.NumTextLayers,
                numTextHeads: _options.NumTextHeads,
                dropoutRate: _options.DropoutRate));
            // ALIGN vision uses MBConv blocks: Dense+LN+Dense+LN+[Dropout] per layer (no MHA)
            int visionLpb = _options.DropoutRate > 0 ? 5 : 4;
            _visionLayerEnd = 2 + _options.NumVisionLayers * visionLpb;
        }
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxImageEncoder is not null)
            return OnnxImageEncoder.Run(input);
        var current = input;
        foreach (var layer in Layers)
            current = layer.Forward(current);
        return current;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        var output = Predict(input);
        var grad = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());
        var gt = Tensor<T>.FromVector(grad);
        for (int i = Layers.Count - 1; i >= 0; i--)
            gt = Layers[i].Backward(gt);
        _optimizer?.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode.");
        int idx = 0;
        foreach (var layer in Layers)
        {
            int count = layer.ParameterCount;
            layer.UpdateParameters(parameters.Slice(idx, count));
            idx += count;
        }
    }

    protected override Tensor<T> PreprocessImage(Tensor<T> image)
        => NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var meta = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "ALIGN-Native" : "ALIGN-ONNX",
            Description = "ALIGN: A Large-scale ImaGe and Noisy-text embedding (Jia et al., ICML 2021)",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = _options.ProjectionDim,
            Complexity = _options.NumVisionLayers + _options.NumTextLayers
        };
        meta.AdditionalInfo["Architecture"] = "ALIGN";
        meta.AdditionalInfo["VisionEncoder"] = "EfficientNet-B7";
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
        if (!string.IsNullOrEmpty(imgPath)) _options.ImageEncoderModelPath = imgPath;
        string txtPath = reader.ReadString();
        if (!string.IsNullOrEmpty(txtPath)) _options.TextEncoderModelPath = txtPath;
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
        if (!_useNativeMode && _options.ImageEncoderModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new ALIGN<T>(Architecture, mp, _options);
        return new ALIGN<T>(Architecture, _options);
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

    private Tensor<T> ForwardVisionEncoder(Tensor<T> input)
    {
        var current = input;
        for (int i = 0; i < _visionLayerEnd; i++)
            current = Layers[i].Forward(current);
        return current;
    }

    private Tensor<T> ForwardTextEncoder(Tensor<T> tokens)
    {
        var current = tokens;
        for (int i = _visionLayerEnd; i < Layers.Count; i++)
            current = Layers[i].Forward(current);
        return current;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(ALIGN<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing) { OnnxImageEncoder?.Dispose(); OnnxTextEncoder?.Dispose(); }
        base.Dispose(disposing);
    }
}
