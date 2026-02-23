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
/// OpenCLIP (Open Contrastive Language-Image Pre-training) model for zero-shot classification
/// and cross-modal retrieval using open-source training data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// OpenCLIP (Ilharco et al., 2021) is an open-source reproduction and extension of OpenAI's CLIP,
/// trained on publicly available datasets like LAION-2B (2 billion image-text pairs) and LAION-5B
/// (5 billion pairs). It reproduces CLIP's dual-encoder architecture with a Vision Transformer (ViT)
/// for images and a text transformer for text, both projecting into a shared embedding space.
/// </para>
/// <para>
/// <b>Architecture:</b>
/// <list type="number">
/// <item><b>Vision encoder</b>: ViT-B/32 through ViT-bigG/14, processing images as patch sequences</item>
/// <item><b>Text encoder</b>: GPT-2-style transformer with causal attention mask</item>
/// <item><b>Projection heads</b>: Linear projections mapping both modalities to shared 512/768/1024-dim space</item>
/// <item><b>Contrastive loss</b>: InfoNCE with learnable temperature (symmetric cross-entropy)</item>
/// </list>
/// </para>
/// <para>
/// <b>Key Innovation:</b> OpenCLIP demonstrates that CLIP's training recipe works effectively with
/// public data. Models trained on LAION-2B match or exceed OpenAI CLIP performance, with the largest
/// ViT-bigG/14 variant achieving 80.1% zero-shot accuracy on ImageNet.
/// </para>
/// <para>
/// <b>For Beginners:</b> OpenCLIP works exactly like CLIP - it understands both images and text.
/// You give it an image and some text labels, and it tells you which label best matches the image,
/// without any additional training. The "Open" in OpenCLIP means it was trained on publicly available
/// data, so anyone can reproduce and verify the results.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 768, outputSize: 512);
/// var openclip = new OpenCLIP&lt;float&gt;(arch, "openclip_vit_b32.onnx");
/// var probs = openclip.ZeroShotClassify(imageTensor, new[] { "a dog", "a cat", "a bird" });
/// </code>
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "Reproducible Scaling Laws for Contrastive Language-Image Learning" (Cherti et al., CVPR 2023)</item>
/// <item>Repository: https://github.com/mlfoundations/open_clip</item>
/// </list>
/// </para>
/// </remarks>
public class OpenCLIP<T> : VisionLanguageModelBase<T>, IContrastiveVisionLanguageModel<T>
{
    #region Fields

    private readonly OpenCLIPOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _visionLayerEnd;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates an OpenCLIP model in ONNX inference mode from a pre-trained model file.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions.</param>
    /// <param name="imageEncoderModelPath">Path to the pre-trained ONNX image encoder file.</param>
    /// <param name="options">Optional configuration. Defaults are used if null.</param>
    public OpenCLIP(NeuralNetworkArchitecture<T> architecture, string imageEncoderModelPath, OpenCLIPOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new OpenCLIPOptions();
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
    /// Creates an OpenCLIP model in native training mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions.</param>
    /// <param name="options">Optional configuration. Defaults are used if null.</param>
    /// <param name="optimizer">Optional gradient-based optimizer. AdamW is used if null.</param>
    public OpenCLIP(NeuralNetworkArchitecture<T> architecture, OpenCLIPOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new OpenCLIPOptions();
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

        var output = Forward(preprocessed);
        return L2Normalize(output);
    }

    /// <inheritdoc />
    public Tensor<T> EncodeText(string text)
    {
        ThrowIfDisposed();
        var tokenized = TokenizeText(text);

        if (IsOnnxMode && OnnxTextEncoder is not null)
            return L2Normalize(OnnxTextEncoder.Run(tokenized));

        // In native mode, run through text layers (second half of layers list)
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

        // Compute logits: similarity / temperature
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultOpenCLIPLayers(
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
        var meta = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "OpenCLIP-Native" : "OpenCLIP-ONNX",
            Description = "OpenCLIP: Open-source Contrastive Language-Image Pre-training (Cherti et al., CVPR 2023)",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = _options.ProjectionDim,
            Complexity = _options.NumVisionLayers + _options.NumTextLayers
        };
        meta.AdditionalInfo["Architecture"] = "OpenCLIP";
        meta.AdditionalInfo["VisionEncoder"] = _options.VisionEncoderVariant.ToString();
        meta.AdditionalInfo["TextEncoder"] = _options.TextEncoderVariant.ToString();
        meta.AdditionalInfo["ProjectionDim"] = _options.ProjectionDim.ToString();
        meta.AdditionalInfo["PretrainingDataset"] = _options.Dataset.ToString();
        meta.AdditionalInfo["CoCaVariant"] = _options.UseCoCaVariant.ToString();
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
        writer.Write((int)_options.Dataset);
        writer.Write(_options.UseCoCaVariant);
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
        _options.Dataset = (PretrainingDataset)reader.ReadInt32();
        _options.UseCoCaVariant = reader.ReadBoolean();

        if (!_useNativeMode && _options.ImageEncoderModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxImageEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
        if (_options.TextEncoderModelPath is { } tp2 && !string.IsNullOrEmpty(tp2))
            OnnxTextEncoder = new OnnxModel<T>(tp2, _options.OnnxOptions);
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ImageEncoderModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new OpenCLIP<T>(Architecture, mp, _options);
        return new OpenCLIP<T>(Architecture, _options);
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

    private Tensor<T> ForwardTextEncoder(Tensor<T> tokens)
    {
        var current = tokens;
        for (int i = _visionLayerEnd; i < Layers.Count; i++)
            current = Layers[i].Forward(current);
        return current;
    }

    private Tensor<T> Forward(Tensor<T> input)
    {
        var current = input;
        for (int i = 0; i < _visionLayerEnd; i++)
            current = Layers[i].Forward(current);
        return current;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(OpenCLIP<T>));
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
