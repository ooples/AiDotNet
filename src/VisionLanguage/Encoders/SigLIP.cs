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
/// SigLIP (Sigmoid Loss for Language-Image Pre-training) model for zero-shot classification
/// and cross-modal retrieval with improved batch scaling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SigLIP (Zhai et al., ICCV 2023) replaces CLIP's softmax-based InfoNCE loss with a sigmoid loss
/// that operates on individual image-text pairs independently. This removes the need for global
/// batch normalization, enabling efficient training with very large batch sizes (up to 1M pairs).
/// The model achieves 84.5% zero-shot on ImageNet with ViT-L/16@384.
/// </para>
/// <para>
/// <b>Architecture:</b>
/// <list type="number">
/// <item><b>Vision encoder</b>: ViT with patch sizes 16 or 14, supporting multiple image resolutions</item>
/// <item><b>Text encoder</b>: Transformer with causal attention, shared vocabulary</item>
/// <item><b>Sigmoid contrastive loss</b>: Per-pair binary classification: positive = 1, negative = -1</item>
/// <item><b>Learnable temperature + bias</b>: sigmoid(z * (sim/t + b)) where t and b are learned</item>
/// </list>
/// </para>
/// <para>
/// <b>Key Innovation:</b> The sigmoid loss removes the softmax normalization across the full batch,
/// which means each image-text pair is treated independently. This allows efficient distributed training
/// without needing to gather all embeddings globally, and empirically gives better performance at scale.
/// </para>
/// <para>
/// <b>For Beginners:</b> SigLIP is an improved version of CLIP. The main difference is in how it
/// learns: instead of comparing every image with every text in a batch simultaneously (which requires
/// lots of memory), SigLIP looks at each image-text pair one at a time and asks "do these match?".
/// This simple change makes training faster and results better.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 768, outputSize: 768);
/// var siglip = new SigLIP&lt;float&gt;(arch, "siglip_vit_b16.onnx");
/// var probs = siglip.ZeroShotClassify(imageTensor, new[] { "a dog", "a cat", "a bird" });
/// </code>
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "Sigmoid Loss for Language Image Pre-Training" (Zhai et al., ICCV 2023)</item>
/// <item>Paper: "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding" (2025)</item>
/// <item>Repository: https://github.com/google-research/big_vision (SigLIP components)</item>
/// </list>
/// </para>
/// </remarks>
public class SigLIP<T> : VisionLanguageModelBase<T>, IContrastiveVisionLanguageModel<T>
{
    #region Fields

    private readonly SigLIPOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _visionLayerEnd;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a SigLIP model in ONNX inference mode from a pre-trained model file.
    /// </summary>
    public SigLIP(NeuralNetworkArchitecture<T> architecture, string imageEncoderModelPath, SigLIPOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new SigLIPOptions();
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
    /// Creates a SigLIP model in native training mode.
    /// </summary>
    public SigLIP(NeuralNetworkArchitecture<T> architecture, SigLIPOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new SigLIPOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.VisionEmbeddingDim;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    #endregion

    #region IContrastiveVisionLanguageModel

    /// <inheritdoc />
    public int EmbeddingDimension => _options.VisionEmbeddingDim;

    /// <inheritdoc />
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;

    /// <inheritdoc />
    int IVisualEncoder<T>.ImageChannels => 3;

    /// <inheritdoc />
    public int MaxSequenceLength => _options.MaxSequenceLength;

    /// <inheritdoc />
    public int TextEmbeddingDimension => _options.TextEmbeddingDim;

    /// <inheritdoc />
    public int ProjectionDimension => _options.ProjectionDim;

    /// <inheritdoc />
    public T Temperature => NumOps.FromDouble(_options.Temperature);

    /// <inheritdoc />
    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessImage(image);

        if (IsOnnxMode && OnnxImageEncoder is not null)
            return L2Normalize(OnnxImageEncoder.Run(preprocessed));

        var output = ForwardVisionEncoder(preprocessed);
        return L2Normalize(output);
    }

    /// <inheritdoc />
    public Tensor<T> EncodeText(string text)
    {
        ThrowIfDisposed();
        var tokenized = TokenizeText(text);

        if (IsOnnxMode && OnnxTextEncoder is not null)
            return L2Normalize(OnnxTextEncoder.Run(tokenized));

        var output = ForwardTextEncoder(tokenized);
        return L2Normalize(output);
    }

    /// <inheritdoc />
    public Tensor<T>[] EncodeTexts(string[] texts)
    {
        var embeddings = new Tensor<T>[texts.Length];
        for (int i = 0; i < texts.Length; i++)
            embeddings[i] = EncodeText(texts[i]);
        return embeddings;
    }

    /// <inheritdoc />
    public T ComputeSimilarity(Tensor<T> image, string text)
    {
        var imageEmb = EncodeImage(image);
        var textEmb = EncodeText(text);
        return CosineSimilarity(imageEmb, textEmb);
    }

    /// <inheritdoc />
    public Dictionary<string, T> ZeroShotClassify(Tensor<T> image, string[] labels)
    {
        var imageEmb = EncodeImage(image);
        var textEmbs = EncodeTexts(labels);

        // SigLIP uses sigmoid scoring: sigmoid(sim/t + b) per pair
        var scores = new Tensor<T>([labels.Length]);
        double temp = _options.Temperature;
        double bias = _options.SigmoidBias;

        for (int i = 0; i < labels.Length; i++)
        {
            double sim = NumOps.ToDouble(CosineSimilarity(imageEmb, textEmbs[i]));
            double logit = sim / temp + bias;
            scores[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-logit)));
        }

        // Normalize to probabilities
        double total = 0;
        for (int i = 0; i < scores.Length; i++)
            total += NumOps.ToDouble(scores[i]);

        var result = new Dictionary<string, T>();
        for (int i = 0; i < labels.Length; i++)
        {
            double prob = total > 1e-8 ? NumOps.ToDouble(scores[i]) / total : 1.0 / labels.Length;
            result[labels[i]] = NumOps.FromDouble(prob);
        }

        return result;
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc />
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultSigLIPLayers(
                visionEmbeddingDim: _options.VisionEmbeddingDim,
                textEmbeddingDim: _options.TextEmbeddingDim,
                projectionDim: _options.ProjectionDim,
                numVisionLayers: _options.NumVisionLayers,
                numTextLayers: _options.NumTextLayers,
                numVisionHeads: _options.NumVisionHeads,
                numTextHeads: _options.NumTextHeads,
                dropoutRate: _options.DropoutRate));
            int lpb = _options.DropoutRate > 0 ? 6 : 5;
            _visionLayerEnd = 2 + _options.NumVisionLayers * lpb;
        }
    }

    /// <inheritdoc />
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

    /// <inheritdoc />
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

    /// <inheritdoc />
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

    /// <inheritdoc />
    protected override Tensor<T> PreprocessImage(Tensor<T> image)
        => NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        string name = _options.UseSigLIP2 ? "SigLIP2" : "SigLIP";
        var meta = new ModelMetadata<T>
        {
            Name = _useNativeMode ? $"{name}-Native" : $"{name}-ONNX",
            Description = _options.UseSigLIP2
                ? "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding (2025)"
                : "SigLIP: Sigmoid Loss for Language Image Pre-Training (Zhai et al., ICCV 2023)",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = _options.ProjectionDim,
            Complexity = _options.NumVisionLayers + _options.NumTextLayers
        };
        meta.AdditionalInfo["Architecture"] = name;
        meta.AdditionalInfo["VisionEncoder"] = _options.VisionEncoderVariant.ToString();
        meta.AdditionalInfo["LossType"] = _options.LossType.ToString();
        meta.AdditionalInfo["ProjectionDim"] = _options.ProjectionDim.ToString();
        meta.AdditionalInfo["SigmoidBias"] = _options.SigmoidBias.ToString();
        meta.AdditionalInfo["Multilingual"] = _options.Multilingual.ToString();
        return meta;
    }

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.ImageEncoderModelPath ?? string.Empty);
        writer.Write(_options.TextEncoderModelPath ?? string.Empty);
        writer.Write(_options.ImageSize);
        writer.Write(_options.VisionEmbeddingDim);
        writer.Write(_options.TextEmbeddingDim);
        writer.Write(_options.ProjectionDim);
        writer.Write(_options.NumVisionLayers);
        writer.Write(_options.NumTextLayers);
        writer.Write(_options.NumVisionHeads);
        writer.Write(_options.NumTextHeads);
        writer.Write(_options.Temperature);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.SigmoidBias);
        writer.Write(_options.UseSigLIP2);
        writer.Write(_options.Multilingual);
    }

    /// <inheritdoc />
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
        _options.NumVisionLayers = reader.ReadInt32();
        _options.NumTextLayers = reader.ReadInt32();
        _options.NumVisionHeads = reader.ReadInt32();
        _options.NumTextHeads = reader.ReadInt32();
        _options.Temperature = reader.ReadDouble();
        _options.DropoutRate = reader.ReadDouble();
        _options.SigmoidBias = reader.ReadDouble();
        _options.UseSigLIP2 = reader.ReadBoolean();
        _options.Multilingual = reader.ReadBoolean();

        if (!_useNativeMode && _options.ImageEncoderModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxImageEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
        if (_options.TextEncoderModelPath is { } tp2 && !string.IsNullOrEmpty(tp2))
            OnnxTextEncoder = new OnnxModel<T>(tp2, _options.OnnxOptions);
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ImageEncoderModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new SigLIP<T>(Architecture, mp, _options);
        return new SigLIP<T>(Architecture, _options);
    }

    #endregion

    #region Private Helpers

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
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SigLIP<T>));
    }

    #endregion

    #region Disposal

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing) { OnnxImageEncoder?.Dispose(); OnnxTextEncoder?.Dispose(); }
        base.Dispose(disposing);
    }

    #endregion
}
