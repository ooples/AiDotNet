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
/// IDEFICS2: 8B efficient VLM with SigLIP encoder and Mistral-7B decoder.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// IDEFICS2 (Laurencon et al., 2024) is an 8B parameter efficient VLM that replaces the
/// OpenCLIP encoder with SigLIP and uses Mistral-7B as the language backbone. It introduces
/// a learned perceiver pooling strategy and native resolution image processing via sub-image
/// splitting for improved document understanding.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "What matters when building vision-language models?" (Laurencon et al., 2024)</item></list></para>
/// <para><b>For Beginners:</b> IDEFICS2 is a much more efficient 8 billion parameter successor
/// to the original 80B IDEFICS. It replaces the OpenCLIP encoder with SigLIP and uses Mistral-7B
/// as the language backbone, introducing learned perceiver pooling and native resolution image
/// processing that splits images into sub-images for better document understanding. Default
/// values follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
/// var trainModel = new IDEFICS2&lt;double&gt;(architecture, new IDEFICS2Options());
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
    "What matters when building vision-language models?",
    "https://arxiv.org/abs/2405.02246",
    Year = 2024,
    Authors = "Laurencon et al."
)]
public class IDEFICS2<T> : VisionLanguageModelBase<T>, IGenerativeVisionLanguageModel<T>
{
    private readonly IDEFICS2Options _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;

    private readonly List<ILayer<T>> _perceiverLayers = new List<ILayer<T>>();
    private readonly List<ILayer<T>> _decoderLayers = new List<ILayer<T>>();

    public IDEFICS2(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        IDEFICS2Options? options = null
    )
        : base(architecture)
    {
        _options = options ?? new IDEFICS2Options();
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

    public IDEFICS2(
        NeuralNetworkArchitecture<T> architecture,
        IDEFICS2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new IDEFICS2Options();
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
    /// Generates text using IDEFICS2's efficient 8B SigLIP+Mistral architecture.
    /// IDEFICS2 (Laurencon et al., 2024) replaces components for efficiency:
    /// (1) SigLIP-SO400M-14 vision encoder (replaces OpenCLIP for better efficiency),
    /// (2) Native resolution processing: splits high-res images into sub-images,
    /// (3) Learned perceiver pooling: 64 latent queries with learned attention,
    /// (4) Mistral-7B decoder (replaces LLaMA-7B) for stronger language modeling.
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

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        SetTrainingMode(false);
        var c = PreprocessImage(input);
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
            Name = _useNativeMode ? "IDEFICS2-Native" : "IDEFICS2-ONNX",
            Description =
                "IDEFICS2: What matters when building vision-language models? (Laurencon et al., 2024)",
            FeatureCount = _options.DecoderDim,
            Complexity =
                _options.NumVisionLayers + _options.NumPerceiverLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "IDEFICS2";
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
            return new IDEFICS2<T>(Architecture, mp, _options);
        return new IDEFICS2<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(IDEFICS2<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        if (disposing)
        {
            OnnxModel?.Dispose();
        }
        base.Dispose(disposing);
    }
}
