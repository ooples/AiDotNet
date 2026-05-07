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

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>
/// Qwen3-VL: latest series with 2B/4B/8B/32B variants and cross-attention resampler.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Qwen3-VL is the latest generation of the Qwen vision-language model series, available in
/// multiple sizes (2B, 4B, 8B, 32B). It inherits the cross-attention resampler architecture
/// from Qwen2-VL with improvements in training data and model optimization, using Qwen3 as
/// the language backbone.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Qwen3-VL Technical Report" (2025)</item></list></para>
/// <para><b>For Beginners:</b> Qwen3-VL is the latest generation of Alibaba's Qwen vision-language
/// models, available in 2B, 4B, 8B, and 32B sizes to fit different compute budgets. It builds
/// on the innovations from Qwen2-VL (dynamic resolution, M-RoPE) with improved training data
/// and model optimization using the Qwen3 language backbone. The cross-attention resampler
/// efficiently compresses visual tokens before feeding them to the language model, keeping
/// inference fast even for high-resolution images. The range of sizes means you can pick the
/// right trade-off between performance and cost for your application. Default values follow
/// the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
/// var trainModel = new Qwen3VL&lt;double&gt;(architecture, new Qwen3VLOptions());
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
[ResearchPaper("Qwen3-VL Technical Report", "https://qwenlm.github.io/blog/qwen3-vl/", Year = 2025, Authors = "Qwen Team")]
public class Qwen3VL<T> : VisionLanguageModelBase<T>, IInstructionTunedVLM<T>
{
    private readonly Qwen3VLOptions _options;
    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;

    private readonly List<ILayer<T>> _resamplerLayers = new List<ILayer<T>>();
    private readonly List<ILayer<T>> _decoderLayers = new List<ILayer<T>>();

    public Qwen3VL(NeuralNetworkArchitecture<T> architecture, string modelPath, Qwen3VLOptions? options = null) : base(architecture)
    {
        _options = options ?? new Qwen3VLOptions();
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

    public Qwen3VL(NeuralNetworkArchitecture<T> architecture, Qwen3VLOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new Qwen3VLOptions();
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
        if (h > 0 && w > 0 && h == w) _options.ImageSize = h;
    }

    public int EmbeddingDimension => _options.DecoderDim;
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;
    int IVisualEncoder<T>.ImageChannels => 3;
    public int MaxGenerationLength => _options.MaxGenerationLength;
    public int DecoderEmbeddingDim => _options.DecoderDim;
    public string LanguageModelName => _options.LanguageModelName;

    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p));
        var c = p;
        foreach (var l in Layers) c = l.Forward(c);
        return L2Normalize(c);
    }

    /// <summary>
    /// Generates text using Qwen3-VL's cross-attention resampler architecture.
    /// Qwen3-VL (2025) features:
    /// (1) Multi-size variants (2B / 4B / 8B / 32B) for different compute budgets,
    /// (2) Inherits Qwen2-VL's dynamic resolution + M-RoPE,
    /// (3) Cross-attention resampler with learnable query tokens,
    /// (4) Qwen3 language backbone for stronger reasoning.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        var visionOut = p;
        foreach (var l in Layers) visionOut = l.Forward(visionOut);

        var resamplerOut = visionOut;
        foreach (var l in _resamplerLayers) resamplerOut = l.Forward(resamplerOut);

        Tensor<T>? promptTokens = null;
        if (prompt is not null) promptTokens = TokenizeText(prompt);

        var decoderInput = resamplerOut;
        if (promptTokens is not null) decoderInput = resamplerOut.ConcatenateTensors(promptTokens);

        var output = decoderInput;
        foreach (var l in _decoderLayers) output = l.Forward(output);

        return output;
    }

    public Tensor<T> Chat(Tensor<T> image, IEnumerable<(string Role, string Content)> conversationHistory, string userMessage)
    {
        ThrowIfDisposed();
        var sb = new System.Text.StringBuilder();
        sb.Append(_options.SystemPrompt);
        foreach (var (role, content) in conversationHistory) sb.Append($"\n{role}: {content}");
        sb.Append($"\nUser: {userMessage}\nAssistant:");
        return GenerateFromImage(image, sb.ToString());
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            return;
        }

        int blockSize = _options.DropoutRate > 0 ? 6 : 5;
        int rBlockSize = _options.DropoutRate > 0 ? 8 : 7;
        int visionLayerEnd = 1 + _options.NumVisionLayers * blockSize;
        int rProj = _options.VisionDim != _options.ResamplerDim ? 1 : 0;
        int resamplerLayerEnd = visionLayerEnd + rProj + _options.NumResamplerLayers * rBlockSize;

        var allLayers = LayerHelper<T>.CreateDefaultPerceiverResamplerLayers(
            _options.VisionDim, _options.ResamplerDim, _options.DecoderDim,
            _options.NumVisionLayers, _options.NumResamplerLayers, _options.NumDecoderLayers,
            _options.MaxVisualTokens, _options.NumHeads, _options.NumResamplerHeads,
            _options.DropoutRate);

        int idx = 0;
        foreach (var layer in allLayers)
        {
            if (idx < visionLayerEnd) Layers.Add(layer);
            else if (idx < resamplerLayerEnd) _resamplerLayers.Add(layer);
            else _decoderLayers.Add(layer);
            idx++;
        }

        RegisterAuxiliaryEncoderStream(_resamplerLayers);
        RegisterAuxiliaryEncoderStream(_decoderLayers);
    }

    private Tensor<T> TokenizeText(string text)
    {
        if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized.");
        var encoding = _tokenizer.Encode(text);
        int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength);
        var tokens = new Tensor<T>([seqLen]);
        for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]);
        return tokens;
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        SetTrainingMode(false);
        var c = PreprocessImage(input);
        foreach (var l in Layers) c = l.Forward(c);
        return c;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        try { TrainWithTape(PreprocessImage(input), expected); }
        finally { SetTrainingMode(false); }
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode.");
        int idx = 0;
        foreach (var l in Layers) { int c = (int)l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
        // Sync the auxiliary streams (resampler / abstractor / decoder /
        // visual decoder, depending on model) — see OpenFlamingo.UpdateParameters
        // for full rationale (dual-stream split, GetExtraTrainableLayers
        // widens the flat parameter vector to include them, so a writeback
        // that only walks Layers leaves auxiliary streams on stale weights
        // and the model state silently de-syncs across streams).
        foreach (var l in EnumerateAuxiliaryStreamTrainableLayers())
        {
            if (l is null) continue;
            int c = (int)l.ParameterCount;
            l.UpdateParameters(parameters.Slice(idx, c));
            idx += c;
        }
    }

    /// <inheritdoc />
    protected override IEnumerable<LayerBase<T>?> GetExtraTrainableLayers()
        => EnumerateAuxiliaryStreamTrainableLayers();

    protected override Tensor<T> PreprocessImage(Tensor<T> image) =>
        NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "Qwen3-VL-Native" : "Qwen3-VL-ONNX",
            Description = "Qwen3-VL Technical Report (Qwen Team, 2025)",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumResamplerLayers + _options.NumDecoderLayers
        };
        m.AdditionalInfo["Architecture"] = "Qwen3-VL";
        m.AdditionalInfo["InstructionType"] = _options.InstructionArchitectureType.ToString();
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.ImageSize);
        writer.Write(_options.VisionDim);
        writer.Write(_options.ResamplerDim);
        writer.Write(_options.DecoderDim);
        writer.Write(_options.NumVisionLayers);
        writer.Write(_options.NumResamplerLayers);
        writer.Write(_options.NumDecoderLayers);
        writer.Write(_options.NumHeads);
        writer.Write(_options.NumResamplerHeads);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.ImageSize = reader.ReadInt32();
        _options.VisionDim = reader.ReadInt32();
        _options.ResamplerDim = reader.ReadInt32();
        _options.DecoderDim = reader.ReadInt32();
        _options.NumVisionLayers = reader.ReadInt32();
        _options.NumResamplerLayers = reader.ReadInt32();
        _options.NumDecoderLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        _options.NumResamplerHeads = reader.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new Qwen3VL<T>(Architecture, mp, _options);
        return new Qwen3VL<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Qwen3VL<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
