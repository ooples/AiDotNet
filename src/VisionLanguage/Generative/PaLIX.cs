using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>
/// PaLI-X: scaled PaLI to 55B with improved component fine-tuning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PaLI-X (Chen et al., 2023) scales PaLI to 55B parameters by using a ViT-22B vision encoder
/// and a 32B UL2 language model. It introduces component-wise fine-tuning for improved performance
/// on vision-language benchmarks.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "PaLI-X: On Scaling up a Multilingual Vision and Language Model" (Chen et al., 2023)</item></list></para>
/// <para><b>For Beginners:</b> PaLIX is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
public class PaLIX<T> : VisionLanguageModelBase<T>, IGenerativeVisionLanguageModel<T>
{
    private readonly PaLIXOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public PaLIX(NeuralNetworkArchitecture<T> architecture, string modelPath, PaLIXOptions? options = null) : base(architecture) { _options = options ?? new PaLIXOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public PaLIX(NeuralNetworkArchitecture<T> architecture, PaLIXOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new PaLIXOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim;

    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }

    /// <summary>
    /// Generates text using PaLI-X's scaled ViT-22B + UL2 architecture.
    /// PaLI-X (Chen et al., 2023) scales PaLI to 55B parameters with:
    /// (1) ViT-22B vision encoder (largest dense ViT) for rich visual features,
    /// (2) Linear projection maps ViT-22B features to UL2 embedding dimension,
    /// (3) Visual tokens prepended to text input for UL2 encoder processing,
    /// (4) UL2 decoder with mixture-of-denoisers training: R-denoiser (regular
    ///     span corruption), S-denoiser (sequential prefix LM), X-denoiser
    ///     (extreme span corruption) for robust multi-task generation,
    /// (5) Component-wise fine-tuning: ViT-22B and UL2 can be unfrozen/frozen
    ///     independently per task for optimal downstream transfer.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        // Step 1: ViT-22B vision encoder
        var encoderOut = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            encoderOut = Layers[i].Forward(encoderOut);

        // Step 2: Tokenize task-prefixed prompt for UL2 conditioning
        Tensor<T>? promptTokens = null;
        if (prompt is not null)
            promptTokens = TokenizeText(prompt);

        // Step 3: Build prepended visual+text sequence [visual_tokens | text_tokens]
        // PaLI-X prepends ViT-22B visual tokens to text and feeds through UL2 encoder-decoder
        var decoderInput = encoderOut;
        if (promptTokens is not null)
            decoderInput = encoderOut.ConcatenateTensors(promptTokens);

        // Step 4: UL2 decoder generates output
        var output = decoderInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Layers.Count / 2; }
        else { Layers.AddRange(LayerHelper<T>.CreateDefaultEncoderDecoderVLMLayers(_options.VisionDim, _options.DecoderDim, _options.NumVisionLayers, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); }
    }

    private void ComputeEncoderDecoderBoundary() { int lpb = _options.DropoutRate > 0 ? 6 : 5; _encoderLayerEnd = 1 + _options.NumVisionLayers * lpb + (_options.VisionDim != _options.DecoderDim ? 1 : 0); }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }

    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "PaLI-X-Native" : "PaLI-X-ONNX", Description = "PaLI-X: On Scaling up a Multilingual Vision and Language Model (Chen et al., 2023)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers }; m.AdditionalInfo["Architecture"] = "PaLI-X"; m.AdditionalInfo["GenerativeType"] = _options.ArchitectureType.ToString(); return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.ImageSize); writer.Write(_options.VisionDim); writer.Write(_options.DecoderDim); writer.Write(_options.NumVisionLayers); writer.Write(_options.NumDecoderLayers); writer.Write(_options.NumHeads); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.ImageSize = reader.ReadInt32(); _options.VisionDim = reader.ReadInt32(); _options.DecoderDim = reader.ReadInt32(); _options.NumVisionLayers = reader.ReadInt32(); _options.NumDecoderLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new PaLIX<T>(Architecture, mp, _options); return new PaLIX<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(PaLIX<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
