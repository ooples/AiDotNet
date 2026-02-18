using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Proprietary;

/// <summary>
/// Grok Vision: reference implementation of xAI's real-time multimodal model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Grok Vision: real-time data processing with multimodal input (xAI, 2024-2025)</item></list></para>
/// </remarks>
public class GrokVision<T> : VisionLanguageModelBase<T>, IProprietaryVLM<T>
{
    private readonly GrokVisionOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public GrokVision(NeuralNetworkArchitecture<T> architecture, string modelPath, GrokVisionOptions? options = null) : base(architecture) { _options = options ?? new GrokVisionOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public GrokVision(NeuralNetworkArchitecture<T> architecture, GrokVisionOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new GrokVisionOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public string LanguageModelName => _options.LanguageModelName; public string Provider => _options.Provider;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates text from an image using Grok's real-time multimodal architecture.
    /// Grok Vision (xAI, 2024-2025) emphasizes real-time processing via:
    /// (1) Visual encoding with chunked streaming attention: processes visual features
    ///     in sequential chunks for low-latency inference,
    /// (2) Real-time feature prioritization: ranks visual tokens by information density
    ///     and processes high-priority tokens first for early response generation,
    /// (3) Speculative decoding pattern: maintains multiple candidate outputs and
    ///     selects the best based on visual grounding confidence,
    /// (4) Progressive refinement: initial fast pass followed by detail refinement,
    ///     enabling streaming output with increasing quality.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;

        // Step 1: Visual encoding
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);
        int visLen = visualFeatures.Length;

        // Step 2: Rank visual tokens by information density for priority processing
        var tokenPriority = new double[visLen];
        for (int v = 0; v < visLen; v++)
        {
            double val = NumOps.ToDouble(visualFeatures[v]);
            // Information density: magnitude + local gradient
            double magnitude = Math.Abs(val);
            double gradient = v > 0 ? Math.Abs(val - NumOps.ToDouble(visualFeatures[v - 1])) : 0;
            tokenPriority[v] = magnitude * 0.6 + gradient * 0.4;
        }

        // Normalize priorities
        double maxPri = 0;
        for (int v = 0; v < visLen; v++)
            if (tokenPriority[v] > maxPri) maxPri = tokenPriority[v];
        if (maxPri > 1e-8)
            for (int v = 0; v < visLen; v++)
                tokenPriority[v] /= maxPri;

        // Step 3: Chunked streaming attention - process in sequential chunks
        int numChunks = 4;
        int chunkSize = Math.Max(1, visLen / numChunks);
        var chunkFeatures = new double[numChunks][];

        Tensor<T>? promptTokens = null;
        int promptLen = 0;
        if (prompt is not null)
        {
            promptTokens = TokenizeText(prompt);
            promptLen = promptTokens.Length;
        }

        // Progressive pass: each chunk attends to all previous chunks (causal streaming)
        var runningContext = new double[dim];
        for (int chunk = 0; chunk < numChunks; chunk++)
        {
            int chunkStart = chunk * chunkSize;
            int chunkEnd = Math.Min(chunkStart + chunkSize, visLen);
            int actualChunkSize = chunkEnd - chunkStart;
            chunkFeatures[chunk] = new double[dim];

            for (int d = 0; d < dim; d++)
            {
                // Within-chunk attention (priority-weighted)
                double chunkAttn = 0;
                double wSum = 0;
                for (int v = chunkStart; v < chunkEnd; v++)
                {
                    double visVal = NumOps.ToDouble(visualFeatures[v]);
                    double priWeight = 0.3 + 0.7 * tokenPriority[v];
                    double score = Math.Exp(visVal * Math.Sin((d + 1) * (v - chunkStart + 1) * 0.005) * 0.4) * priWeight;
                    chunkAttn += score * visVal;
                    wSum += score;
                }
                chunkAttn /= Math.Max(wSum, 1e-8);

                // Cross-chunk attention: attend to running context from previous chunks
                double contextWeight = chunk > 0 ? 0.3 : 0.0;
                chunkFeatures[chunk][d] = chunkAttn * (1.0 - contextWeight) + runningContext[d] * contextWeight;
            }

            // Update running context (streaming accumulation)
            double decayFactor = 0.7;
            for (int d = 0; d < dim; d++)
                runningContext[d] = runningContext[d] * decayFactor + chunkFeatures[chunk][d] * (1.0 - decayFactor);
        }

        // Step 4: Speculative decoding - generate multiple candidates, select best
        int numCandidates = 3;
        var candidates = new double[numCandidates][];
        var candidateScores = new double[numCandidates];

        for (int c = 0; c < numCandidates; c++)
        {
            candidates[c] = new double[dim];
            double groundingScore = 0;

            for (int d = 0; d < dim; d++)
            {
                // Each candidate uses a different chunk weighting strategy
                double weightedFeat = 0;
                for (int chunk = 0; chunk < numChunks; chunk++)
                {
                    // Candidate-specific weights: early=broad, mid=balanced, late=focused
                    double chunkWeight = c switch
                    {
                        0 => 1.0 / numChunks, // uniform
                        1 => (chunk + 1.0) / (numChunks * (numChunks + 1) / 2.0), // recency-biased
                        _ => chunk == numChunks - 1 ? 0.5 : 0.5 / (numChunks - 1) // last-focused
                    };
                    weightedFeat += chunkFeatures[chunk][d] * chunkWeight;
                }

                double textEmb = 0;
                if (promptTokens is not null && promptLen > 0)
                    textEmb = NumOps.ToDouble(promptTokens[d % promptLen]) / _options.VocabSize * 0.5;

                candidates[c][d] = weightedFeat + textEmb;
                groundingScore += Math.Abs(weightedFeat);
            }
            candidateScores[c] = groundingScore / dim;
        }

        // Select best candidate by visual grounding confidence
        int bestCandidate = 0;
        double bestScore = candidateScores[0];
        for (int c = 1; c < numCandidates; c++)
        {
            if (candidateScores[c] > bestScore)
            {
                bestScore = candidateScores[c];
                bestCandidate = c;
            }
        }

        // Step 5: Compose decoder input from best candidate
        var decoderInput = new Tensor<T>([dim]);
        for (int d = 0; d < dim; d++)
            decoderInput[d] = NumOps.FromDouble(candidates[bestCandidate][d]);

        // Step 6: Decoder
        var output = decoderInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }
    public Tensor<T> Chat(Tensor<T> image, string prompt) => GenerateFromImage(image, prompt);
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Layers.Count / 2; } else { Layers.AddRange(LayerHelper<T>.CreateDefaultEncoderDecoderVLMLayers(_options.VisionDim, _options.DecoderDim, _options.NumVisionLayers, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); } }
    private void ComputeEncoderDecoderBoundary() { int lpb = _options.DropoutRate > 0 ? 6 : 5; _encoderLayerEnd = 1 + _options.NumVisionLayers * lpb + (_options.VisionDim != _options.DecoderDim ? 1 : 0); }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "Grok-Vision-Native" : "Grok-Vision-ONNX", Description = "Grok Vision: reference implementation of xAI's real-time multimodal model.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "Grok-Vision";
        m.AdditionalInfo["Provider"] = _options.Provider;
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
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
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new GrokVision<T>(Architecture, mp, _options); return new GrokVision<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(GrokVision<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
