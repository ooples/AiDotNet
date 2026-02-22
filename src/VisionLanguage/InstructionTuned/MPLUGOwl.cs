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

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>
/// mPLUG-Owl: modular VLM with visual abstractor for vision-language alignment.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// mPLUG-Owl (Alibaba, 2023) uses a modular architecture with a visual abstractor module that
/// learns to compress and align visual features from the ViT encoder before feeding them into
/// the LLaMA language model. The abstractor uses learnable queries with cross-attention.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "mPLUG-Owl: Modularization Empowers Large Language Models with Multimodality" (Ye et al., 2023)</item></list></para>
/// <para><b>For Beginners:</b> MPLUGOwl is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
public class MPLUGOwl<T> : VisionLanguageModelBase<T>, IInstructionTunedVLM<T>
{
    private readonly MPLUGOwlOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _visionLayerEnd; private int _abstractorLayerEnd;

    public MPLUGOwl(NeuralNetworkArchitecture<T> architecture, string modelPath, MPLUGOwlOptions? options = null) : base(architecture) { _options = options ?? new MPLUGOwlOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public MPLUGOwl(NeuralNetworkArchitecture<T> architecture, MPLUGOwlOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new MPLUGOwlOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public string LanguageModelName => _options.LanguageModelName;

    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _visionLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }

    /// <summary>
    /// Generates text using mPLUG-Owl's visual abstractor architecture.
    /// mPLUG-Owl (Ye et al., 2023) uses:
    /// (1) ViT-L/14 vision encoder for visual feature extraction,
    /// (2) Visual abstractor module with 65 learnable query tokens that cross-attend
    ///     to all visual features, compressing them into a fixed-length representation,
    /// (3) Cross-attention in abstractor: queries attend to visual keys/values,
    ///     producing aligned visual tokens for the language model,
    /// (4) LLaMA decoder backbone with abstracted visual token prefix.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        int abstractorDim = _options.AbstractorDim;
        int numQueries = _options.MaxVisualTokens;

        // Step 1: ViT-L/14 vision encoder
        var visionOut = p;
        for (int i = 0; i < _visionLayerEnd; i++)
            visionOut = Layers[i].Forward(visionOut);

        // Step 2: Visual abstractor layers
        var abstractorOut = visionOut;
        for (int i = _visionLayerEnd; i < _abstractorLayerEnd; i++)
            abstractorOut = Layers[i].Forward(abstractorOut);

        // Fuse features with prompt tokens via ConcatenateTensors
        Tensor<T> fusedInput;
        if (prompt is not null)
        {
            var promptTokens = TokenizeText(prompt);
            fusedInput = abstractorOut.ConcatenateTensors(promptTokens);
        }
        else
        {
            fusedInput = abstractorOut;
        }

        var output = fusedInput;
        for (int i = _abstractorLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }

    public Tensor<T> Chat(Tensor<T> image, IEnumerable<(string Role, string Content)> conversationHistory, string userMessage) { ThrowIfDisposed(); var sb = new System.Text.StringBuilder(); sb.Append(_options.SystemPrompt); foreach (var (role, content) in conversationHistory) sb.Append($"\n{role}: {content}"); sb.Append($"\nUser: {userMessage}\nAssistant:"); return GenerateFromImage(image, sb.ToString()); }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _visionLayerEnd = Layers.Count / 3; _abstractorLayerEnd = Layers.Count * 2 / 3; }
        else { Layers.AddRange(LayerHelper<T>.CreateDefaultPerceiverResamplerLayers(_options.VisionDim, _options.AbstractorDim, _options.DecoderDim, _options.NumVisionLayers, _options.NumAbstractorLayers, _options.NumDecoderLayers, _options.MaxVisualTokens, _options.NumHeads, _options.NumAbstractorHeads, _options.DropoutRate)); ComputeAbstractorBoundaries(); }
    }

    private void ComputeAbstractorBoundaries()
    {
        int lpb = _options.DropoutRate > 0 ? 6 : 5;
        _visionLayerEnd = 1 + _options.NumVisionLayers * lpb;
        int abstractorProj = _options.VisionDim != _options.AbstractorDim ? 1 : 0;
        int rLpb = _options.DropoutRate > 0 ? 8 : 7;
        _abstractorLayerEnd = _visionLayerEnd + abstractorProj + _options.NumAbstractorLayers * rLpb;
    }

    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }

    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "mPLUG-Owl-Native" : "mPLUG-Owl-ONNX", Description = "mPLUG-Owl: Modular VLM with Visual Abstractor (Ye et al., 2023)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumAbstractorLayers + _options.NumDecoderLayers }; m.AdditionalInfo["Architecture"] = "mPLUG-Owl"; m.AdditionalInfo["InstructionType"] = _options.InstructionArchitectureType.ToString(); m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName; return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.ImageSize); writer.Write(_options.VisionDim); writer.Write(_options.AbstractorDim); writer.Write(_options.DecoderDim); writer.Write(_options.NumVisionLayers); writer.Write(_options.NumAbstractorLayers); writer.Write(_options.NumDecoderLayers); writer.Write(_options.NumHeads); writer.Write(_options.NumAbstractorHeads); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.ImageSize = reader.ReadInt32(); _options.VisionDim = reader.ReadInt32(); _options.AbstractorDim = reader.ReadInt32(); _options.DecoderDim = reader.ReadInt32(); _options.NumVisionLayers = reader.ReadInt32(); _options.NumAbstractorLayers = reader.ReadInt32(); _options.NumDecoderLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); _options.NumAbstractorHeads = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new MPLUGOwl<T>(Architecture, mp, _options); return new MPLUGOwl<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(MPLUGOwl<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
