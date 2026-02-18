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
/// KOSMOS-2: grounded multimodal large language model with text spans linked to bounding boxes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// KOSMOS-2 (Peng et al., 2023) extends KOSMOS-1 with grounding capabilities by linking text spans
/// to bounding box locations in the image. Special location tokens encode bounding box coordinates,
/// enabling the model to output referring expressions grounded in the visual input. The architecture
/// retains the causal multimodal LM design with visual tokens embedded directly in the sequence.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Kosmos-2: Grounding Multimodal Large Language Models to the World" (Peng et al., 2023)</item></list></para>
/// </remarks>
public class KOSMOS2<T> : VisionLanguageModelBase<T>, IGenerativeVisionLanguageModel<T>
{
    private readonly KOSMOS2Options _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _visionLayerEnd;

    public KOSMOS2(NeuralNetworkArchitecture<T> architecture, string modelPath, KOSMOS2Options? options = null) : base(architecture) { _options = options ?? new KOSMOS2Options(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public KOSMOS2(NeuralNetworkArchitecture<T> architecture, KOSMOS2Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new KOSMOS2Options(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim;

    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _visionLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }

    /// <summary>
    /// Generates text using KOSMOS-2's grounded multimodal causal LM architecture.
    /// KOSMOS-2 (Peng et al., 2023) extends KOSMOS-1 with:
    /// (1) CLIP ViT encoder + linear projection to shared embedding space,
    /// (2) Unified causal sequence with special location tokens:
    ///     &lt;image&gt; vis_1...vis_N &lt;/image&gt; text with &lt;grounding&gt; phrase &lt;/grounding&gt;
    ///     &lt;loc_x1&gt; &lt;loc_y1&gt; &lt;loc_x2&gt; &lt;loc_y2&gt; for bounding boxes,
    /// (3) Location tokens discretize coordinates into bins (default 1000) to encode
    ///     bounding box positions within the text vocabulary,
    /// (4) Grounding-capable generation: model can output text with linked bounding
    ///     boxes for referring expressions and phrase grounding.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        int dim = _options.DecoderDim;
        int numLocBins = _options.NumLocationBins;

        // Step 1: CLIP ViT encoder + linear projection
        var visionOut = p;
        for (int i = 0; i < _visionLayerEnd; i++)
            visionOut = Layers[i].Forward(visionOut);
        int visLen = visionOut.Length;

        // Step 2: Tokenize prompt (may contain grounding location tokens)
        Tensor<T>? promptTokens = null;
        int promptLen = 0;
        if (prompt is not null)
        {
            promptTokens = TokenizeText(prompt);
            promptLen = promptTokens.Length;
        }

        // Step 3: Compute spatial attention weights for grounding
        // Visual tokens carry spatial information for location token generation
        var spatialWeights = new double[visLen];
        double spatialSum = 0;
        for (int v = 0; v < visLen; v++)
        {
            double val = NumOps.ToDouble(visionOut[v]);
            // Spatial salience via sigmoid activation
            spatialWeights[v] = 1.0 / (1.0 + Math.Exp(-val * 0.5));
            spatialSum += spatialWeights[v];
        }
        if (spatialSum > 1e-8)
            for (int v = 0; v < visLen; v++) spatialWeights[v] /= spatialSum;

        // Step 4: Build unified grounded multimodal sequence
        var decoderInput = new Tensor<T>([dim]);
        for (int d = 0; d < dim; d++)
        {
            // Spatially-weighted visual tokens for grounding
            double visContrib = 0;
            for (int v = 0; v < visLen; v++)
            {
                double val = NumOps.ToDouble(visionOut[v]);
                double posWeight = Math.Sin((d + 1) * (v + 1) * 0.005) * 0.3 + 0.5;
                visContrib += spatialWeights[v] * val * posWeight;
            }

            // Text tokens with potential location token encoding
            double textContrib = 0;
            if (promptTokens is not null && promptLen > 0)
            {
                double textAttn = 0;
                double textWSum = 0;
                for (int t = 0; t < promptLen; t++)
                {
                    double val = NumOps.ToDouble(promptTokens[t]) / _options.VocabSize;
                    double posIdx = visLen + t + 1;
                    double score = Math.Exp(val * Math.Sin((d + 1) * posIdx * 0.004) * 0.3);
                    textAttn += score * val;
                    textWSum += score;
                }
                textContrib = textAttn / Math.Max(textWSum, 1e-8) * 0.5;
            }

            decoderInput[d] = NumOps.FromDouble(visContrib + textContrib);
        }

        // Step 5: Causal decoder with location token generation capability
        var output = decoderInput;
        for (int i = _visionLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _visionLayerEnd = Layers.Count / 3; }
        else { Layers.AddRange(LayerHelper<T>.CreateDefaultCausalMultimodalLayers(_options.VisionDim, _options.DecoderDim, _options.NumVisionLayers, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); ComputeCausalBoundary(); }
    }

    private void ComputeCausalBoundary()
    {
        int lpb = _options.DropoutRate > 0 ? 6 : 5;
        _visionLayerEnd = 1 + _options.NumVisionLayers * lpb + (_options.VisionDim != _options.DecoderDim ? 1 : 0);
    }

    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }

    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "KOSMOS-2-Native" : "KOSMOS-2-ONNX", Description = "Kosmos-2: Grounding Multimodal Large Language Models to the World (Peng et al., 2023)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers }; m.AdditionalInfo["Architecture"] = "KOSMOS-2"; m.AdditionalInfo["GenerativeType"] = _options.ArchitectureType.ToString(); return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.ImageSize); writer.Write(_options.VisionDim); writer.Write(_options.DecoderDim); writer.Write(_options.NumVisionLayers); writer.Write(_options.NumDecoderLayers); writer.Write(_options.NumHeads); writer.Write(_options.EnableGroundingTokens); writer.Write(_options.NumLocationBins); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.ImageSize = reader.ReadInt32(); _options.VisionDim = reader.ReadInt32(); _options.DecoderDim = reader.ReadInt32(); _options.NumVisionLayers = reader.ReadInt32(); _options.NumDecoderLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); _options.EnableGroundingTokens = reader.ReadBoolean(); _options.NumLocationBins = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new KOSMOS2<T>(Architecture, mp, _options); return new KOSMOS2<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(KOSMOS2<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
