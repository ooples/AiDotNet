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
/// IDEFICS3: state-of-the-art 8B VLM trained with Docmatix for document understanding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// IDEFICS3 (Laurencon et al., 2024) is an 8B parameter model that builds upon IDEFICS2 with
/// improved training data including the Docmatix dataset for document understanding. It retains
/// the SigLIP + perceiver + Llama 3.1 architecture but achieves state-of-the-art results on
/// document QA benchmarks while being trained exclusively on open datasets.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Building and better understanding vision-language models: insights and future directions" (Laurencon et al., 2024)</item></list></para>
/// <para><b>For Beginners:</b> IDEFICS3 builds on IDEFICS2 with improved training data,
/// including the Docmatix dataset for document understanding, and upgrades the language backbone
/// to Llama 3.1. It achieves state-of-the-art results on document QA benchmarks while being
/// trained exclusively on open datasets, making it one of the strongest open-source VLMs at
/// the 8B parameter scale. Default values follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
/// var trainModel = new IDEFICS3&lt;double&gt;(architecture, new IDEFICS3Options());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Building and better understanding vision-language models: insights and future directions",
    "https://arxiv.org/abs/2408.12637",
    Year = 2024,
    Authors = "Laurencon et al."
)]
public class IDEFICS3<T> : VisionLanguageModelBase<T>, IGenerativeVisionLanguageModel<T>
{
    private readonly IDEFICS3Options _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;

    private readonly List<ILayer<T>> _perceiverLayers = new List<ILayer<T>>();
    private readonly List<ILayer<T>> _decoderLayers = new List<ILayer<T>>();

    public IDEFICS3(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        IDEFICS3Options? options = null
    )
        : base(architecture)
    {
        _options = options ?? new IDEFICS3Options();
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

    public IDEFICS3(
        NeuralNetworkArchitecture<T> architecture,
        IDEFICS3Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new IDEFICS3Options();
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
    /// Generates text using IDEFICS3's SigLIP+Llama-3.1 architecture trained on Docmatix.
    /// IDEFICS3 (Laurencon et al., 2024) iterates on IDEFICS2 with:
    /// (1) Llama 3.1 decoder backbone (replaces Mistral-7B) for stronger language modeling,
    /// (2) Trained on Docmatix dataset for document QA — state-of-the-art document VLM,
    /// (3) Same SigLIP-SO400M + perceiver pooling pipeline as IDEFICS2 for efficiency,
    /// (4) Native resolution sub-image splitting preserved for high-res documents.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        var visionOut = p;
        foreach (var l in Layers)
            visionOut = l.Forward(visionOut);

        var perceiverOut = visionOut;
        foreach (var l in _perceiverLayers)
            perceiverOut = l.Forward(perceiverOut);

        Tensor<T>? promptTokens = null;
        if (prompt is not null)
            promptTokens = TokenizeText(prompt);

        var decoderInput = perceiverOut;
        if (promptTokens is not null)
            decoderInput = perceiverOut.ConcatenateTensors(promptTokens);

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
            // Vision encoder, perceiver, and decoder supplied as named streams.
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
        // Sync the auxiliary streams (Q-Former / perceiver / decoder /
        // regression head, depending on model) — see OpenFlamingo.UpdateParameters
        // for full rationale (dual-stream split, GetExtraTrainableLayers
        // widens the flat parameter vector to include them, so a writeback
        // that only walks Layers leaves auxiliary streams on stale weights
        // and the model state silently de-syncs across streams).
        foreach (var l in EnumerateAuxiliaryStreamTrainableLayers())
        {
            if (l is null)
                continue;
            int c = (int)l.ParameterCount;
            l.UpdateParameters(parameters.Slice(idx, c));
            idx += c;
        }
    }

    /// <inheritdoc />
    protected override IEnumerable<LayerBase<T>?> GetExtraTrainableLayers() =>
        EnumerateAuxiliaryStreamTrainableLayers();

    protected override Tensor<T> PreprocessImage(Tensor<T> image) =>
        NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "IDEFICS3-Native" : "IDEFICS3-ONNX",
            Description =
                "IDEFICS3: Building and better understanding vision-language models (Laurencon et al., 2024)",
            FeatureCount = _options.DecoderDim,
            Complexity =
                _options.NumVisionLayers + _options.NumPerceiverLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "IDEFICS3";
        m.AdditionalInfo["GenerativeType"] = _options.ArchitectureType.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.ImageSize);
        writer.Write(_options.VisionDim);
        writer.Write(_options.PerceiverDim);
        writer.Write(_options.DecoderDim);
        writer.Write(_options.NumVisionLayers);
        writer.Write(_options.NumPerceiverLayers);
        writer.Write(_options.NumDecoderLayers);
        writer.Write(_options.NumHeads);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp))
            _options.ModelPath = mp;
        _options.ImageSize = reader.ReadInt32();
        _options.VisionDim = reader.ReadInt32();
        _options.PerceiverDim = reader.ReadInt32();
        _options.DecoderDim = reader.ReadInt32();
        _options.NumVisionLayers = reader.ReadInt32();
        _options.NumPerceiverLayers = reader.ReadInt32();
        _options.NumDecoderLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new IDEFICS3<T>(Architecture, mp, _options);
        return new IDEFICS3<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(IDEFICS3<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
