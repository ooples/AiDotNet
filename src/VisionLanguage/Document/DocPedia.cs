using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// DocPedia: frequency-domain document understanding model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "DocPedia: Unleashing the Power of Large Multimodal Model in the Frequency Domain" (2024)</item></list></para>
/// </remarks>
public class DocPedia<T> : VisionLanguageModelBase<T>, IDocumentUnderstandingModel<T>
{
    private readonly DocPediaOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public DocPedia(NeuralNetworkArchitecture<T> architecture, string modelPath, DocPediaOptions? options = null) : base(architecture) { _options = options ?? new DocPediaOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public DocPedia(NeuralNetworkArchitecture<T> architecture, DocPediaOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new DocPediaOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public bool IsOcrFree => _options.IsOcrFree;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates text from a document image using DocPedia's frequency-domain pipeline.
    /// Per the paper (2024), DocPedia processes high-resolution documents in the frequency domain:
    /// (1) DCT (Discrete Cosine Transform) converts spatial image patches to frequency coefficients,
    ///     enabling efficient high-res processing without the quadratic cost of spatial attention,
    /// (2) Frequency-aware visual encoding: low-frequency components capture document layout/structure,
    ///     mid-frequency captures text body, high-frequency captures fine details (serifs, subscripts),
    /// (3) Multi-frequency fusion: different frequency bands are weighted by content type before
    ///     projection to the LLM space,
    /// (4) The LLM receives frequency-domain visual tokens for text generation.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;

        // Step 1: Vision encoder
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);
        int visDim = visualFeatures.Length;

        // Step 2: DCT-based frequency decomposition of visual features
        // Decompose into low, mid, and high frequency bands
        int blockSize = Math.Min(visDim, 256);
        var lowFreq = new double[blockSize];
        var midFreq = new double[blockSize];
        var highFreq = new double[blockSize];

        for (int k = 0; k < blockSize; k++)
        {
            // Type-II DCT: X(k) = sum_n x(n) * cos(pi*(2n+1)*k / (2*N))
            double dctCoeff = 0;
            int N = Math.Min(visDim, blockSize);
            for (int n = 0; n < N; n++)
            {
                double xn = NumOps.ToDouble(visualFeatures[n % visDim]);
                dctCoeff += xn * Math.Cos(Math.PI * (2 * n + 1) * k / (2.0 * N));
            }
            dctCoeff *= Math.Sqrt(2.0 / N);
            if (k == 0) dctCoeff *= 1.0 / Math.Sqrt(2.0); // DC component normalization

            // Frequency band assignment
            double freqRatio = (double)k / blockSize;
            if (freqRatio < 0.15)
                lowFreq[k] = dctCoeff;   // Layout/structure
            else if (freqRatio < 0.55)
                midFreq[k] = dctCoeff;   // Text body
            else
                highFreq[k] = dctCoeff;  // Fine details
        }

        // Step 3: Multi-frequency weighted fusion
        // Weight each band based on content: text-heavy docs need more mid-frequency
        double lowEnergy = 0, midEnergy = 0, highEnergy = 0;
        for (int k = 0; k < blockSize; k++)
        {
            lowEnergy += lowFreq[k] * lowFreq[k];
            midEnergy += midFreq[k] * midFreq[k];
            highEnergy += highFreq[k] * highFreq[k];
        }
        double totalEnergy = lowEnergy + midEnergy + highEnergy + 1e-8;
        double lowWeight = 0.3 + 0.2 * (lowEnergy / totalEnergy);   // Layout is always important
        double midWeight = 0.4 + 0.3 * (midEnergy / totalEnergy);   // Text body emphasis
        double highWeight = 0.3 + 0.2 * (highEnergy / totalEnergy); // Fine detail

        // Step 4: Build decoder input from frequency-domain features
        Tensor<T>? promptTokens = null;
        int promptLen = 0;
        if (prompt is not null)
        {
            promptTokens = TokenizeText(prompt);
            promptLen = promptTokens.Length;
        }

        var decoderInput = new Tensor<T>([dim]);
        for (int d = 0; d < dim; d++)
        {
            int freqIdx = d % blockSize;
            // Weighted multi-frequency representation
            double freqEmb = lowFreq[freqIdx] * lowWeight
                           + midFreq[freqIdx] * midWeight
                           + highFreq[freqIdx] * highWeight;

            // Inverse DCT-like reconstruction for position-dependent features
            double spatialRecon = 0;
            int numBasis = Math.Min(32, blockSize);
            for (int k = 0; k < numBasis; k++)
            {
                double coeff = lowFreq[k] * lowWeight + midFreq[k] * midWeight + highFreq[k] * highWeight;
                spatialRecon += coeff * Math.Cos(Math.PI * (2 * d + 1) * k / (2.0 * dim));
            }
            spatialRecon /= numBasis;

            double promptCond = 0;
            if (promptTokens is not null && promptLen > 0)
                promptCond = NumOps.ToDouble(promptTokens[d % promptLen]) / _options.VocabSize * 0.5;

            decoderInput[d] = NumOps.FromDouble(freqEmb * 0.4 + spatialRecon * 0.6 + promptCond);
        }

        // Step 5: LLM decoder
        var output = decoderInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }
    public Tensor<T> ExtractText(Tensor<T> documentImage) { ThrowIfDisposed(); return GenerateFromImage(documentImage, "Extract all text from this document."); }
    public Tensor<T> AnswerDocumentQuestion(Tensor<T> documentImage, string question) { ThrowIfDisposed(); return GenerateFromImage(documentImage, question); }
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Layers.Count / 2; } else { Layers.AddRange(LayerHelper<T>.CreateDefaultEncoderDecoderVLMLayers(_options.VisionDim, _options.DecoderDim, _options.NumVisionLayers, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); } }
    private void ComputeEncoderDecoderBoundary() { int lpb = _options.DropoutRate > 0 ? 6 : 5; _encoderLayerEnd = 1 + _options.NumVisionLayers * lpb + (_options.VisionDim != _options.DecoderDim ? 1 : 0); }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "DocPedia-Native" : "DocPedia-ONNX", Description = "DocPedia: frequency-domain document understanding model.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "DocPedia";
        m.AdditionalInfo["OcrFree"] = _options.IsOcrFree.ToString();
        m.AdditionalInfo["FrequencyDomain"] = _options.EnableFrequencyDomain.ToString();
        return m;
    }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.ImageSize);
        writer.Write(_options.VisionDim);
        writer.Write(_options.DecoderDim);
        writer.Write(_options.NumVisionLayers);
        writer.Write(_options.NumDecoderLayers);
        writer.Write(_options.NumHeads);
        writer.Write(_options.IsOcrFree);
        writer.Write(_options.MaxOutputTokens);
        writer.Write(_options.EnableFrequencyDomain);
    }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.ImageSize = reader.ReadInt32();
        _options.VisionDim = reader.ReadInt32();
        _options.DecoderDim = reader.ReadInt32();
        _options.NumVisionLayers = reader.ReadInt32();
        _options.NumDecoderLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        _options.IsOcrFree = reader.ReadBoolean();
        _options.MaxOutputTokens = reader.ReadInt32();
        _options.EnableFrequencyDomain = reader.ReadBoolean();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new DocPedia<T>(Architecture, mp, _options); return new DocPedia<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(DocPedia<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
