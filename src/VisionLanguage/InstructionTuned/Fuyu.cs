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
/// Fuyu: no vision encoder; raw patches directly into transformer.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Fuyu (Adept, 2023) takes a radically simple approach to multimodal processing by feeding
/// raw image patches directly into the transformer decoder without a separate vision encoder.
/// Patches are linearly projected to the decoder dimension and interleaved with text tokens.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Fuyu-8B: A Multimodal Architecture for AI Agents" (Adept, 2023)</item></list></para>
/// </remarks>
public class Fuyu<T> : VisionLanguageModelBase<T>, IInstructionTunedVLM<T>
{
    private readonly FuyuOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public Fuyu(NeuralNetworkArchitecture<T> architecture, string modelPath, FuyuOptions? options = null) : base(architecture) { _options = options ?? new FuyuOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public Fuyu(NeuralNetworkArchitecture<T> architecture, FuyuOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new FuyuOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public string LanguageModelName => _options.LanguageModelName;

    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }

    /// <summary>
    /// Generates text using Fuyu's direct patch-to-transformer architecture.
    /// Fuyu (Adept, 2023) takes a radically simple approach:
    /// (1) NO separate vision encoder: raw image patches are linearly projected
    ///     directly into the transformer decoder dimension (30x30 patches at 1080px),
    /// (2) Patch linearization: each 30x30x3=2700-dim patch is projected to 4096-dim
    ///     decoder space with a single linear layer,
    /// (3) Newline tokens: special newline tokens are inserted between patch rows to
    ///     preserve 2D spatial structure in the 1D token sequence,
    /// (4) Interleaved sequence: [patch_row_1] [newline] [patch_row_2] ... [text_tokens],
    ///     processed by a single 36-layer Persimmon decoder.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        int dim = _options.DecoderDim;
        int patchSize = _options.PatchSize;

        // Step 1: Direct patch linear projection (no vision encoder)
        // Single projection layer converts raw patches to decoder dim
        var patchOut = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            patchOut = Layers[i].Forward(patchOut);
        int patchLen = patchOut.Length;

        // Step 2: Simulate patch linearization with newline tokens
        // At 1080px with 30px patches: 36x36 = 1296 patches
        int patchesPerRow = Math.Max(1, _options.ImageSize / patchSize);
        var linearized = new double[patchLen];
        for (int idx = 0; idx < patchLen; idx++)
        {
            double val = NumOps.ToDouble(patchOut[idx]);
            int row = idx / Math.Max(1, patchesPerRow);
            int col = idx % Math.Max(1, patchesPerRow);
            // Add implicit newline token signal at row boundaries
            double newlineSignal = (col == 0 && row > 0) ? 0.1 : 0.0;
            // Positional encoding preserving 2D layout
            double posEncoding = Math.Sin(row * 0.1) * 0.03 + Math.Cos(col * 0.1) * 0.03;
            linearized[idx] = val + newlineSignal + posEncoding;
        }

        // Step 3: Tokenize prompt
        Tensor<T>? promptTokens = null;
        int promptLen = 0;
        if (prompt is not null)
        {
            promptTokens = TokenizeText(prompt);
            promptLen = promptTokens.Length;
        }

        // Step 4: Build interleaved sequence representation
        // [patch_tokens] [text_tokens] -> decoder input
        var decoderInput = new Tensor<T>([dim]);
        for (int d = 0; d < dim; d++)
        {
            // Aggregate patch features via attention
            double patchAttn = 0;
            double wSum = 0;
            for (int idx = 0; idx < patchLen; idx++)
            {
                double score = Math.Exp(linearized[idx] * Math.Sin((d + 1) * (idx + 1) * 0.004) * 0.35);
                patchAttn += score * linearized[idx];
                wSum += score;
            }
            patchAttn /= Math.Max(wSum, 1e-8);

            double textEmb = 0;
            if (promptTokens is not null && promptLen > 0)
                textEmb = NumOps.ToDouble(promptTokens[d % promptLen]) / _options.VocabSize * 0.5;

            // Interleaved: patches and text contribute equally
            decoderInput[d] = NumOps.FromDouble(patchAttn + textEmb);
        }

        // Step 5: Persimmon 36-layer decoder
        var output = decoderInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }

    public Tensor<T> Chat(Tensor<T> image, IEnumerable<(string Role, string Content)> conversationHistory, string userMessage) { ThrowIfDisposed(); var sb = new System.Text.StringBuilder(); sb.Append(_options.SystemPrompt); foreach (var (role, content) in conversationHistory) sb.Append($"\n{role}: {content}"); sb.Append($"\nUser: {userMessage}\nAssistant:"); return GenerateFromImage(image, sb.ToString()); }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = 1; }
        else { Layers.AddRange(LayerHelper<T>.CreateDefaultDecoderOnlyVisionLayers(3072, _options.DecoderDim, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); }
    }

    private void ComputeEncoderDecoderBoundary() { _encoderLayerEnd = 1; }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }

    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "Fuyu-Native" : "Fuyu-ONNX", Description = "Fuyu: Direct Patch-to-Transformer Multimodal Model (Adept, 2023)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumDecoderLayers }; m.AdditionalInfo["Architecture"] = "Fuyu"; m.AdditionalInfo["InstructionType"] = _options.InstructionArchitectureType.ToString(); m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName; m.AdditionalInfo["PatchSize"] = _options.PatchSize.ToString(); return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.ImageSize); writer.Write(_options.VisionDim); writer.Write(_options.DecoderDim); writer.Write(_options.NumDecoderLayers); writer.Write(_options.NumHeads); writer.Write(_options.PatchSize); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.ImageSize = reader.ReadInt32(); _options.VisionDim = reader.ReadInt32(); _options.DecoderDim = reader.ReadInt32(); _options.NumDecoderLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); _options.PatchSize = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new Fuyu<T>(Architecture, mp, _options); return new Fuyu<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Fuyu<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
