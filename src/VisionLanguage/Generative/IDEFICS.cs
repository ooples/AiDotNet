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

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>
/// IDEFICS: 80B open reproduction of Flamingo for interleaved image-text generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// IDEFICS (Laurencon et al., 2023) is an 80B parameter open-source model that reproduces
/// the Flamingo architecture. It uses an OpenCLIP ViT-H vision encoder, a perceiver resampler
/// for visual token compression, and gated cross-attention layers interleaved within a
/// LLaMA-based decoder for multimodal text generation.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents" (Laurencon et al., NeurIPS 2023)</item></list></para>
/// <para><b>For Beginners:</b> IDEFICS is an 80 billion parameter open-source model that
/// replicates DeepMind's Flamingo architecture. It processes interleaved sequences of images
/// and text using gated cross-attention layers inserted into a LLaMA decoder, with a perceiver
/// resampler compressing visual features. Trained on the OBELICS web-scraped dataset, it
/// excels at few-shot multimodal tasks. Default values follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
/// var trainModel = new IDEFICS&lt;double&gt;(architecture, new IDEFICSOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents",
    "https://arxiv.org/abs/2306.16527",
    Year = 2023,
    Authors = "Laurencon et al."
)]
public partial class IDEFICS<T> : VisionLanguageModelBase<T>, IGenerativeVisionLanguageModel<T>
{
    private readonly IDEFICSOptions _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;

    private readonly List<ILayer<T>> _perceiverLayers = new List<ILayer<T>>();
    private readonly List<ILayer<T>> _decoderLayers = new List<ILayer<T>>();

    public IDEFICS(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        IDEFICSOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new IDEFICSOptions();
        SyncImageSizeWithArchitecture();
        _useNativeMode = false;
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    public IDEFICS(
        NeuralNetworkArchitecture<T> architecture,
        IDEFICSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new IDEFICSOptions();
        SyncImageSizeWithArchitecture();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
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

    public int EmbeddingDimension => _options.DecoderDim;
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;
    int IVisualEncoder<T>.ImageChannels => 3;
    public int MaxGenerationLength => _options.MaxGenerationLength;
    public int DecoderEmbeddingDim => _options.DecoderDim;

    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return L2Normalize(OnnxModel.Run(p));
        var c = p;
        foreach (var l in Layers)
            c = l.Forward(c);
        return L2Normalize(c);
    }

    /// <summary>
    /// Generates text using IDEFICS's 80B Flamingo-style architecture.
    /// IDEFICS (Laurencon et al., NeurIPS 2023) replicates Flamingo at 80B with:
    /// (1) OpenCLIP ViT-H/14 vision encoder for visual feature extraction,
    /// (2) Perceiver resampler compresses variable-length visual features into fixed
    ///     64 latent tokens via cross-attention with learnable queries,
    /// (3) Gated cross-attention layers interleaved every 4th layer in LLaMA decoder:
    ///     each has a learned tanh gate initialized near zero for stable training,
    /// (4) Trained on interleaved image-text data (OBELICS) for in-context learning.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        // Step 1: OpenCLIP ViT-H vision encoder
        var visionOut = p;
        foreach (var l in Layers)
            visionOut = l.Forward(visionOut);

        // Step 2: Perceiver resampler compresses visual features into fixed latent tokens
        var perceiverOut = visionOut;
        foreach (var l in _perceiverLayers)
            perceiverOut = l.Forward(perceiverOut);

        // Step 3: Tokenize prompt
        Tensor<T>? promptTokens = null;
        if (prompt is not null)
            promptTokens = TokenizeText(prompt);

        // Step 4: Concatenate perceiver output with prompt tokens
        var decoderInput = perceiverOut;
        if (promptTokens is not null)
            decoderInput = perceiverOut.ConcatenateTensors(promptTokens);

        // Step 5: LLaMA decoder
        var output = decoderInput;
        foreach (var l in _decoderLayers)
            output = l.Forward(output);

        return output;
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture is TripleStreamArchitecture<T> triple)
        {
            Layers.AddRange(triple.VisionLayers);
            _perceiverLayers.AddRange(triple.AuxiliaryLayers);
            _decoderLayers.AddRange(triple.TextOrDecoderLayers);
            RegisterAuxiliaryEncoderStream(_perceiverLayers);
            RegisterAuxiliaryEncoderStream(_decoderLayers);
            return;
        }

        int blockSize = _options.DropoutRate > 0 ? 6 : 5;
        int pBlockSize = _options.DropoutRate > 0 ? 8 : 7;
        int visionLayerEnd = 1 + _options.NumVisionLayers * blockSize;
        int pProj = _options.VisionDim != _options.PerceiverDim ? 1 : 0;
        int perceiverLayerEnd = visionLayerEnd + pProj + _options.NumPerceiverLayers * pBlockSize;

        var allLayers = LayerHelper<T>.CreateDefaultPerceiverResamplerLayers(
            _options.VisionDim,
            _options.PerceiverDim,
            _options.DecoderDim,
            _options.NumVisionLayers,
            _options.NumPerceiverLayers,
            _options.NumDecoderLayers,
            _options.NumLatents,
            _options.NumHeads,
            _options.NumPerceiverHeads,
            _options.DropoutRate
        );

        int idx = 0;
        foreach (var layer in allLayers)
        {
            if (idx < visionLayerEnd)
                Layers.Add(layer);
            else if (idx < perceiverLayerEnd)
                _perceiverLayers.Add(layer);
            else
                _decoderLayers.Add(layer);
            idx++;
        }

        RegisterAuxiliaryEncoderStream(_perceiverLayers);
        RegisterAuxiliaryEncoderStream(_decoderLayers);
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

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        // Both paths must see the same preprocessed input.
        var c = PreprocessImage(input);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(c);
        SetTrainingMode(false);
        foreach (var l in Layers)
            c = l.Forward(c);
        return c;
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

    // This forwarded to a helper the base now calls from its own
    // GetExtraTrainableLayers, so the override restated it. Removed under AIDN082.

    protected override Tensor<T> PreprocessImage(Tensor<T> image) =>
        NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "IDEFICS-Native" : "IDEFICS-ONNX",
            Description =
                "IDEFICS: Open-Source 80B Reproduction of Flamingo (Laurencon et al., NeurIPS 2023)",
            FeatureCount = _options.DecoderDim,
            Complexity =
                _options.NumVisionLayers + _options.NumPerceiverLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "IDEFICS";
        m.AdditionalInfo["GenerativeType"] = _options.ArchitectureType.ToString();
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(IDEFICS<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
