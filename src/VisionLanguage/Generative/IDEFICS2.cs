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
/// <para><b>For Beginners:</b> IDEFICS2 is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
public class IDEFICS2<T> : VisionLanguageModelBase<T>, IGenerativeVisionLanguageModel<T>
{
    private readonly IDEFICS2Options _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _visionLayerEnd; private int _perceiverLayerEnd;

    public IDEFICS2(NeuralNetworkArchitecture<T> architecture, string modelPath, IDEFICS2Options? options = null) : base(architecture) { _options = options ?? new IDEFICS2Options(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public IDEFICS2(NeuralNetworkArchitecture<T> architecture, IDEFICS2Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new IDEFICS2Options(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim;

    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _visionLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }

    /// <summary>
    /// Generates text using IDEFICS2's efficient SigLIP + perceiver pooling architecture.
    /// IDEFICS2 (Laurencon et al., 2024) improves over IDEFICS with:
    /// (1) SigLIP vision encoder (sigmoid contrastive) instead of OpenCLIP,
    /// (2) Learned perceiver pooling: reduces visual tokens via cross-attention with
    ///     learnable queries, more parameter-efficient than full perceiver resampler,
    /// (3) Native resolution processing via sub-image splitting: high-resolution images
    ///     are split into tiles, each processed by SigLIP, then pooled together,
    /// (4) Mistral-7B decoder with cross-attention to pooled visual features,
    /// (5) Only 8B parameters total while achieving strong document understanding.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        // Step 1: SigLIP vision encoder
        var visionOut = p;
        for (int i = 0; i < _visionLayerEnd; i++)
            visionOut = Layers[i].Forward(visionOut);

        // Step 2: Learned perceiver pooling
        var perceiverOut = visionOut;
        for (int i = _visionLayerEnd; i < _perceiverLayerEnd; i++)
            perceiverOut = Layers[i].Forward(perceiverOut);

        // Step 3: Tokenize prompt
        Tensor<T>? promptTokens = null;
        if (prompt is not null)
            promptTokens = TokenizeText(prompt);

        // Step 4: Concatenate perceiver output with prompt tokens
        // IDEFICS2 uses cross-attention in Mistral decoder layers
        var decoderInput = perceiverOut;
        if (promptTokens is not null)
            decoderInput = perceiverOut.ConcatenateTensors(promptTokens);

        // Step 5: Mistral-7B decoder
        var output = decoderInput;
        for (int i = _perceiverLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _visionLayerEnd = Layers.Count / 3; _perceiverLayerEnd = Layers.Count * 2 / 3; }
        else { Layers.AddRange(LayerHelper<T>.CreateDefaultPerceiverResamplerLayers(_options.VisionDim, _options.PerceiverDim, _options.DecoderDim, _options.NumVisionLayers, _options.NumPerceiverLayers, _options.NumDecoderLayers, _options.NumLatents, _options.NumHeads, _options.NumPerceiverHeads, _options.DropoutRate)); ComputePerceiverBoundaries(); }
    }

    private void ComputePerceiverBoundaries()
    {
        int lpb = _options.DropoutRate > 0 ? 6 : 5;
        _visionLayerEnd = 1 + _options.NumVisionLayers * lpb;
        int perceiverProj = _options.VisionDim != _options.PerceiverDim ? 1 : 0;
        int pLpb = _options.DropoutRate > 0 ? 8 : 7;
        _perceiverLayerEnd = _visionLayerEnd + perceiverProj + _options.NumPerceiverLayers * pLpb;
    }

    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }

    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "IDEFICS2-Native" : "IDEFICS2-ONNX", Description = "IDEFICS2: What matters when building vision-language models? (Laurencon et al., 2024)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumPerceiverLayers + _options.NumDecoderLayers }; m.AdditionalInfo["Architecture"] = "IDEFICS2"; m.AdditionalInfo["GenerativeType"] = _options.ArchitectureType.ToString(); return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.ImageSize); writer.Write(_options.VisionDim); writer.Write(_options.PerceiverDim); writer.Write(_options.DecoderDim); writer.Write(_options.NumVisionLayers); writer.Write(_options.NumPerceiverLayers); writer.Write(_options.NumDecoderLayers); writer.Write(_options.NumHeads); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.ImageSize = reader.ReadInt32(); _options.VisionDim = reader.ReadInt32(); _options.PerceiverDim = reader.ReadInt32(); _options.DecoderDim = reader.ReadInt32(); _options.NumVisionLayers = reader.ReadInt32(); _options.NumPerceiverLayers = reader.ReadInt32(); _options.NumDecoderLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new IDEFICS2<T>(Architecture, mp, _options); return new IDEFICS2<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(IDEFICS2<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
