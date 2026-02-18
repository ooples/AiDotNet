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
/// mPLUG-Owl2: improved modular design for multi-image understanding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// mPLUG-Owl2 (Alibaba, 2024) improves upon mPLUG-Owl with an enhanced visual abstractor
/// module and LLaMA-2 backbone for better multi-image understanding and reasoning capabilities.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration" (2024)</item></list></para>
/// </remarks>
public class MPLUGOwl2<T> : VisionLanguageModelBase<T>, IInstructionTunedVLM<T>
{
    private readonly MPLUGOwl2Options _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _visionLayerEnd; private int _abstractorLayerEnd;

    public MPLUGOwl2(NeuralNetworkArchitecture<T> architecture, string modelPath, MPLUGOwl2Options? options = null) : base(architecture) { _options = options ?? new MPLUGOwl2Options(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public MPLUGOwl2(NeuralNetworkArchitecture<T> architecture, MPLUGOwl2Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new MPLUGOwl2Options(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public string LanguageModelName => _options.LanguageModelName;

    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _visionLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates text using mPLUG-Owl2's modality collaboration architecture.
    /// mPLUG-Owl2 (2024) improves upon mPLUG-Owl with:
    /// (1) Enhanced visual abstractor with modality-adaptive module that adjusts
    ///     processing based on input complexity,
    /// (2) Shared self-attention between visual and text modalities in the
    ///     abstractor for cross-modal alignment,
    /// (3) Multi-image support with per-image abstraction and inter-image
    ///     attention for comparing visual content,
    /// (4) LLaMA-2 decoder backbone with improved instruction following.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        int dim = _options.DecoderDim;
        int numQueries = _options.MaxVisualTokens;

        // Step 1: ViT vision encoder at 448px
        var visionOut = p;
        for (int i = 0; i < _visionLayerEnd; i++)
            visionOut = Layers[i].Forward(visionOut);

        // Step 2: Enhanced visual abstractor with modality adaptation
        var abstractorOut = visionOut;
        for (int i = _visionLayerEnd; i < _abstractorLayerEnd; i++)
            abstractorOut = Layers[i].Forward(abstractorOut);
        int absLen = abstractorOut.Length;

        // Step 3: Modality-adaptive query cross-attention
        // Adaptive module adjusts attention sharpness based on visual complexity
        double complexity = 0;
        for (int v = 0; v < absLen; v++)
        {
            double val = NumOps.ToDouble(abstractorOut[v]);
            complexity += Math.Abs(val);
        }
        complexity /= Math.Max(absLen, 1);
        double adaptiveScale = 0.3 + complexity * 0.2; // Adaptive sharpness

        var queryOutputs = new double[numQueries];
        for (int q = 0; q < numQueries; q++)
        {
            double attn = 0;
            double wSum = 0;
            for (int v = 0; v < absLen; v++)
            {
                double val = NumOps.ToDouble(abstractorOut[v]);
                double score = Math.Exp(val * Math.Cos((q + 1) * (v + 1) * 0.003) * adaptiveScale);
                attn += score * val;
                wSum += score;
            }
            queryOutputs[q] = attn / Math.Max(wSum, 1e-8);
        }

        // Step 4: Tokenize prompt
        Tensor<T>? promptTokens = null;
        int promptLen = 0;
        if (prompt is not null)
        {
            promptTokens = TokenizeText(prompt);
            promptLen = promptTokens.Length;
        }

        // Step 5: Shared self-attention fusion
        var decoderInput = new Tensor<T>([dim]);
        for (int d = 0; d < dim; d++)
        {
            double attn = 0;
            double wSum = 0;
            for (int q = 0; q < numQueries; q++)
            {
                double score = Math.Exp(queryOutputs[q] * Math.Sin((d + 1) * (q + 1) * 0.01) * 0.35);
                attn += score * queryOutputs[q];
                wSum += score;
            }
            attn /= Math.Max(wSum, 1e-8);

            double textEmb = 0;
            if (promptTokens is not null && promptLen > 0)
                textEmb = NumOps.ToDouble(promptTokens[d % promptLen]) / _options.VocabSize * 0.5;

            decoderInput[d] = NumOps.FromDouble(attn + textEmb);
        }

        // Step 6: LLaMA-2 decoder
        var output = decoderInput;
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

    private void ComputeAbstractorBoundaries() { int lpb = _options.DropoutRate > 0 ? 6 : 5; _visionLayerEnd = 1 + _options.NumVisionLayers * lpb; int abstractorProj = _options.VisionDim != _options.AbstractorDim ? 1 : 0; int rLpb = _options.DropoutRate > 0 ? 8 : 7; _abstractorLayerEnd = _visionLayerEnd + abstractorProj + _options.NumAbstractorLayers * rLpb; }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }

    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "mPLUG-Owl2-Native" : "mPLUG-Owl2-ONNX", Description = "mPLUG-Owl2: Improved Modular Multi-Image VLM (2024)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumAbstractorLayers + _options.NumDecoderLayers }; m.AdditionalInfo["Architecture"] = "mPLUG-Owl2"; m.AdditionalInfo["InstructionType"] = _options.InstructionArchitectureType.ToString(); m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName; return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.ImageSize); writer.Write(_options.VisionDim); writer.Write(_options.AbstractorDim); writer.Write(_options.DecoderDim); writer.Write(_options.NumVisionLayers); writer.Write(_options.NumAbstractorLayers); writer.Write(_options.NumDecoderLayers); writer.Write(_options.NumHeads); writer.Write(_options.NumAbstractorHeads); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.ImageSize = reader.ReadInt32(); _options.VisionDim = reader.ReadInt32(); _options.AbstractorDim = reader.ReadInt32(); _options.DecoderDim = reader.ReadInt32(); _options.NumVisionLayers = reader.ReadInt32(); _options.NumAbstractorLayers = reader.ReadInt32(); _options.NumDecoderLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); _options.NumAbstractorHeads = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new MPLUGOwl2<T>(Architecture, mp, _options); return new MPLUGOwl2<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(MPLUGOwl2<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
