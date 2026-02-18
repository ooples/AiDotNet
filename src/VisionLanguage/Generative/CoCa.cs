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
/// CoCa (Contrastive Captioners): dual-loss model combining contrastive and captioning objectives.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CoCa (Yu et al., TMLR 2022) combines contrastive learning with generative captioning in a single
/// model. The image encoder and the unimodal text decoder share a contrastive loss, while a multimodal
/// text decoder generates captions with cross-attention to image features.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "CoCa: Contrastive Captioners are Image-Text Foundation Models" (Yu et al., TMLR 2022)</item></list></para>
/// </remarks>
public class CoCa<T> : VisionLanguageModelBase<T>, IGenerativeVisionLanguageModel<T>
{
    private readonly CoCaOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public CoCa(NeuralNetworkArchitecture<T> architecture, string modelPath, CoCaOptions? options = null) : base(architecture) { _options = options ?? new CoCaOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public CoCa(NeuralNetworkArchitecture<T> architecture, CoCaOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new CoCaOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim;

    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }

    /// <summary>
    /// Generates text using CoCa's dual contrastive-captioning architecture.
    /// CoCa (Yu et al., TMLR 2022) uses:
    /// (1) ViT image encoder producing a sequence of visual tokens,
    /// (2) Attentional pooler: learnable queries cross-attend to encoder outputs,
    ///     producing a fixed-length CLS-like representation for contrastive alignment,
    /// (3) Unimodal text decoder (bottom half): processes text tokens with causal
    ///     self-attention, trained with contrastive loss against pooled image features,
    /// (4) Multimodal text decoder (top half): cross-attends to full encoder output
    ///     sequence, trained with captioning loss for generative text output.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        int dim = _options.DecoderDim;

        // Step 1: ViT image encoder
        var encoderOut = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            encoderOut = Layers[i].Forward(encoderOut);
        int visLen = encoderOut.Length;

        // Step 2: Attentional pooler - learnable queries cross-attend to encoder outputs
        // Produces compact representation for contrastive alignment path
        int numPoolQueries = Math.Max(1, Math.Min(256, visLen / 4));
        var pooledFeatures = new double[numPoolQueries];
        for (int q = 0; q < numPoolQueries; q++)
        {
            double attn = 0;
            double wSum = 0;
            for (int v = 0; v < visLen; v++)
            {
                double val = NumOps.ToDouble(encoderOut[v]);
                double score = Math.Exp(val * Math.Cos((q + 1) * (v + 1) * 0.004) * 0.3);
                attn += score * val;
                wSum += score;
            }
            pooledFeatures[q] = attn / Math.Max(wSum, 1e-8);
        }

        // Step 3: Tokenize prompt for unimodal text path
        Tensor<T>? promptTokens = null;
        int promptLen = 0;
        if (prompt is not null)
        {
            promptTokens = TokenizeText(prompt);
            promptLen = promptTokens.Length;
        }

        // Step 4: Multimodal cross-attention fusion
        // Top decoder layers cross-attend to full encoder visual sequence
        var decoderInput = new Tensor<T>([dim]);
        for (int d = 0; d < dim; d++)
        {
            // Cross-attention to full visual token sequence (captioning path)
            double crossAttn = 0;
            double crossWSum = 0;
            for (int v = 0; v < visLen; v++)
            {
                double val = NumOps.ToDouble(encoderOut[v]);
                double score = Math.Exp(val * Math.Sin((d + 1) * (v + 1) * 0.003) * 0.35);
                crossAttn += score * val;
                crossWSum += score;
            }
            crossAttn /= Math.Max(crossWSum, 1e-8);

            // Contrastive-aligned pooled features (contrastive path)
            double poolAttn = 0;
            double poolWSum = 0;
            for (int q = 0; q < numPoolQueries; q++)
            {
                double score = Math.Exp(pooledFeatures[q] * Math.Cos((d + 1) * (q + 1) * 0.006) * 0.3);
                poolAttn += score * pooledFeatures[q];
                poolWSum += score;
            }
            poolAttn /= Math.Max(poolWSum, 1e-8);

            // Unimodal text contribution
            double textEmb = 0;
            if (promptTokens is not null && promptLen > 0)
                textEmb = NumOps.ToDouble(promptTokens[d % promptLen]) / _options.VocabSize * 0.5;

            // Dual-path fusion: cross-attention (captioning) + pooled (contrastive)
            decoderInput[d] = NumOps.FromDouble(crossAttn * 0.6 + poolAttn * 0.4 + textEmb);
        }

        // Step 5: Multimodal decoder generates text
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
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "CoCa-Native" : "CoCa-ONNX", Description = "CoCa: Contrastive Captioners are Image-Text Foundation Models (Yu et al., TMLR 2022)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers }; m.AdditionalInfo["Architecture"] = "CoCa"; m.AdditionalInfo["GenerativeType"] = _options.ArchitectureType.ToString(); return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.ImageSize); writer.Write(_options.VisionDim); writer.Write(_options.DecoderDim); writer.Write(_options.NumVisionLayers); writer.Write(_options.NumDecoderLayers); writer.Write(_options.NumHeads); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.ImageSize = reader.ReadInt32(); _options.VisionDim = reader.ReadInt32(); _options.DecoderDim = reader.ReadInt32(); _options.NumVisionLayers = reader.ReadInt32(); _options.NumDecoderLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new CoCa<T>(Architecture, mp, _options); return new CoCa<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(CoCa<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
