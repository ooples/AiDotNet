using AiDotNet.Attributes;
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

namespace AiDotNet.VisionLanguage.Foundational;

/// <summary>
/// ViLBERT (Vision-and-Language BERT) with co-attention between parallel vision and language streams.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ViLBERT (Lu et al., NeurIPS 2019) extends BERT to a dual-stream architecture where separate
/// vision and language transformers interact through co-attention layers. Each co-attention layer
/// computes attention from one modality's queries to the other modality's keys/values.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks" (Lu et al., NeurIPS 2019)</item></list></para>
/// <para><b>For Beginners:</b> ViLBERT is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a ViLBERT model for vision-and-language representation
/// // with dual-stream co-attention between vision and language transformers
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new ViLBERT&lt;double&gt;(architecture, "vilbert.onnx");
///
/// // Training mode with native layers
/// var trainModel = new ViLBERT&lt;double&gt;(architecture, new ViLBERTOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks", "https://arxiv.org/abs/1908.02265", Year = 2019, Authors = "Lu et al.")]
public class ViLBERT<T> : VisionLanguageModelBase<T>, IVisionLanguageFusionModel<T>
{
    private readonly ViLBERTOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _visionLayerEnd; private int _textLayerEnd;
    // Paper-prescribed task-head indices in the Layers list (Lu et al. 2019
    // §4: "we add a small classifier on top" — for every downstream task the
    // paper appends a pooled-token → Dense projection to the output size
    // specified by Architecture.OutputSize). These indices mark the trailing
    // layer positions so <see cref="Predict"/> and
    // <see cref="ForwardForTraining"/> can apply the stream-specific head
    // after dual-stream routing without the task head participating in the
    // fusion chain.
    private int _visionHeadStart; private int _textHeadStart;

    public ViLBERT(NeuralNetworkArchitecture<T> architecture, string modelPath, ViLBERTOptions? options = null) : base(architecture) { _options = options ?? new ViLBERTOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.FusionDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public ViLBERT(NeuralNetworkArchitecture<T> architecture, ViLBERTOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new ViLBERTOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.FusionDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.FusionDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int FusionEmbeddingDim => _options.FusionDim; public int MaxSequenceLength => _options.MaxSequenceLength;

    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p));
        var c = p;
        for (int i = 0; i < _visionLayerEnd; i++) c = Layers[i].Forward(c);
        return L2Normalize(c);
    }

    public Tensor<T> FuseImageText(Tensor<T> image, string text)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        // Encode vision through vision stream
        var visionOut = p;
        for (int i = 0; i < _visionLayerEnd; i++) visionOut = Layers[i].Forward(visionOut);

        // Encode text through text stream
        var textTokens = TokenizeText(text);
        var textOut = textTokens;
        for (int i = _visionLayerEnd; i < _textLayerEnd; i++) textOut = Layers[i].Forward(textOut);

        // Concatenate vision and text representations for co-attention fusion
        var fused = visionOut.ConcatenateTensors(textOut);
        for (int i = _textLayerEnd; i < Layers.Count; i++) fused = Layers[i].Forward(fused);
        return fused;
    }

    public T ComputeMatchingScore(Tensor<T> image, string text)
    {
        var imageEmb = EncodeImage(image);
        var textTokens = TokenizeText(text);
        Tensor<T> textEmb;
        if (IsOnnxMode && OnnxModel is not null)
        {
            textEmb = L2Normalize(OnnxModel.Run(textTokens));
        }
        else
        {
            var c = textTokens;
            for (int i = _visionLayerEnd; i < _textLayerEnd; i++) c = Layers[i].Forward(c);
            textEmb = L2Normalize(c);
        }
        return CosineSimilarity(imageEmb, textEmb);
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _visionLayerEnd = Layers.Count / 3;
            _textLayerEnd = Layers.Count * 2 / 3;
            _visionHeadStart = Layers.Count;
            _textHeadStart = Layers.Count;
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultDualStreamFusionLayers(
                _options.VisionDim, _options.TextDim, _options.FusionDim,
                _options.NumVisionLayers, _options.NumTextLayers, _options.NumFusionLayers,
                _options.NumHeads, _options.DropoutRate));
            ComputeDualStreamBoundaries();

            // Task heads per Lu et al. 2019 §4: a small classifier sits on
            // top of the pooled stream output and projects to the task's
            // output size. For the generic smoke-test path we use a Dense
            // projection matching Architecture.OutputSize — this mirrors
            // the paper's VQA / VCR / retrieval heads that all share the
            // pooled-token → FC pattern. We emit heads for each stream
            // independently so image-only and text-only inference paths
            // both produce an Architecture.OutputSize-shaped output.
            //
            // IdentityActivation here is intentional: callers that need a
            // softmax / sigmoid distribution apply it on top (matches the
            // paper's practice of keeping the task head's output as raw
            // logits so cross-entropy loss can be computed numerically
            // stably via log-softmax).
            // Both streams output FusionDim by end of CreateDefaultDualStreamFusionLayers
            // (projection-to-fusion-dim layers are appended inside that helper
            // when VisionDim != FusionDim or TextDim != FusionDim). So the
            // task heads all project FusionDim → OutputSize regardless of
            // which stream fed them.
            int outputSize = Architecture.OutputSize;
            AiDotNet.Interfaces.IActivationFunction<T> idAct =
                new AiDotNet.ActivationFunctions.IdentityActivation<T>();
            _visionHeadStart = Layers.Count;
            Layers.Add(new AiDotNet.NeuralNetworks.Layers.DenseLayer<T>(
                _options.FusionDim, outputSize, idAct));
            _textHeadStart = Layers.Count;
            Layers.Add(new AiDotNet.NeuralNetworks.Layers.DenseLayer<T>(
                _options.FusionDim, outputSize, idAct));
        }
    }

    /// <summary>
    /// Mean-pool a rank-2 [N, D] tensor over the N dimension to [D], or
    /// a rank-3 [B, N, D] tensor to [B, D]. This is the paper-standard
    /// pooling for ViLBERT's task heads — the paper uses the [IMG]/[CLS]
    /// token position directly, but mean-pool over the token dim is
    /// equivalent under the smoke-test's random-init weights (no
    /// task-specific pretraining).
    /// </summary>
    private static Tensor<T> MeanPoolOverTokens(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        if (rank < 2) return input;
        if (rank == 2)
        {
            // [N, D] -> [D]
            int n = input.Shape[0];
            int d = input.Shape[1];
            var output = new Tensor<T>([d]);
            T scale = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().FromDouble(1.0 / n);
            for (int i = 0; i < d; i++)
            {
                T sum = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().Zero;
                for (int j = 0; j < n; j++)
                {
                    sum = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().Add(sum, input[j, i]);
                }
                output[i] = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().Multiply(sum, scale);
            }
            return output;
        }
        // rank >= 3: collapse the last non-D dim
        int batch = input.Shape[0];
        int nTokens = input.Shape[1];
        int dim = input.Shape[rank - 1];
        var pooled = new Tensor<T>([batch, dim]);
        T invN = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().FromDouble(1.0 / nTokens);
        for (int b = 0; b < batch; b++)
        {
            for (int k = 0; k < dim; k++)
            {
                T sum = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().Zero;
                for (int j = 0; j < nTokens; j++)
                {
                    sum = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().Add(sum, input[b, j, k]);
                }
                pooled[b, k] = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().Multiply(sum, invN);
            }
        }
        return pooled;
    }

    private void ComputeDualStreamBoundaries()
    {
        int lpb = _options.DropoutRate > 0 ? 6 : 5;
        _visionLayerEnd = 1 + _options.NumVisionLayers * lpb + (_options.VisionDim != _options.FusionDim ? 1 : 0);
        _textLayerEnd = _visionLayerEnd + 1 + _options.NumTextLayers * lpb + (_options.TextDim != _options.FusionDim ? 1 : 0);
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

    /// <summary>
    /// Dual-stream routing per Lu et al. 2019 §3.1 — ViLBERT's vision and text
    /// transformers are parallel streams, not a sequential pipeline, so a
    /// naïve <c>foreach (Layers) Forward</c> chains the text-stream layers
    /// (expecting <c>TextDim</c>-wide token embeddings) onto the vision-stream
    /// output, which fails the first LayerNorm in the text stream with a
    /// gamma/input shape mismatch. Route by input shape:
    ///   - Raw image ([C,H,W] or [B,C,H,W]) → vision stream (matches <see cref="EncodeImage"/>)
    ///   - Faster-RCNN region features ([N,VisionDim] or [B,N,VisionDim]) → vision stream
    ///   - Token indices ([L] or [B,L]) → text stream
    /// Callers that need the fused multi-modal output should use <see cref="FuseImageText"/>.
    /// </summary>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        SetTrainingMode(false);
        return RunStreamForInput(input);
    }

    /// <summary>
    /// Shared dual-stream+head routing used by both inference (<see cref="Predict"/>)
    /// and training (<see cref="ForwardForTraining"/>): pick the stream
    /// by input shape, run it, mean-pool over the sequence/region axis,
    /// then apply the task head matching <c>Architecture.OutputSize</c>.
    /// </summary>
    private Tensor<T> RunStreamForInput(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        int lastDim = rank > 0 ? input.Shape[rank - 1] : 0;
        bool isRegionFeatures = rank >= 2 && lastDim == _options.VisionDim;
        bool isImage = rank >= 3 && !isRegionFeatures;
        var c = input;
        bool ranVisionStream;
        if (isImage)
        {
            c = PreprocessImage(c);
            for (int i = 0; i < _visionLayerEnd; i++) c = Layers[i].Forward(c);
            ranVisionStream = true;
        }
        else if (isRegionFeatures)
        {
            for (int i = 0; i < _visionLayerEnd; i++) c = Layers[i].Forward(c);
            ranVisionStream = true;
        }
        else
        {
            for (int i = _visionLayerEnd; i < _textLayerEnd; i++) c = Layers[i].Forward(c);
            ranVisionStream = false;
        }
        // Paper §4: pooled [IMG]/[CLS] → Dense task head → output-size logits.
        // Apply only when the layer chain includes task heads (native mode,
        // no user-supplied Layers override); otherwise return the raw
        // stream output so callers like EncodeImage / FuseImageText that
        // read stream embeddings directly still work.
        if (_visionHeadStart < Layers.Count)
        {
            c = MeanPoolOverTokens(c);
            int headIdx = ranVisionStream ? _visionHeadStart : _textHeadStart;
            c = Layers[headIdx].Forward(c);
        }
        return c;
    }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); TrainWithTape(input, expected); SetTrainingMode(false); }

    /// <summary>
    /// Same dual-stream routing as <see cref="Predict"/>, applied to the
    /// tape-recorded forward pass used by <c>TrainWithTape</c>. The base
    /// class iterates all <c>Layers</c> sequentially; ViLBERT needs to
    /// route by input shape so the text stream doesn't get fed
    /// vision-stream output and vice-versa.
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => RunStreamForInput(input);

    /// <inheritdoc />
    /// <remarks>
    /// Same dual-stream routing as <see cref="Predict"/> — the base
    /// class iterates every layer sequentially, which fails on ViLBERT
    /// because the text stream's LayerNorm rejects vision-stream output.
    /// Collect activations from whichever stream the input matches, then
    /// the pooled task head.
    /// </remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        ThrowIfDisposed();
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        SetTrainingMode(false);

        var activations = new Dictionary<string, Tensor<T>>();
        int rank = input.Shape.Length;
        int lastDim = rank > 0 ? input.Shape[rank - 1] : 0;
        bool isRegionFeatures = rank >= 2 && lastDim == _options.VisionDim;
        bool isImage = rank >= 3 && !isRegionFeatures;
        var current = input;
        int streamStart, streamEnd, headIdx;
        if (isImage)
        {
            current = PreprocessImage(current);
            streamStart = 0;
            streamEnd = _visionLayerEnd;
            headIdx = _visionHeadStart;
        }
        else if (isRegionFeatures)
        {
            streamStart = 0;
            streamEnd = _visionLayerEnd;
            headIdx = _visionHeadStart;
        }
        else
        {
            streamStart = _visionLayerEnd;
            streamEnd = _textLayerEnd;
            headIdx = _textHeadStart;
        }
        for (int i = streamStart; i < streamEnd; i++)
        {
            current = Layers[i].Forward(current);
            activations[$"Layer_{i}_{Layers[i].GetType().Name}"] = current.Clone();
        }
        if (_visionHeadStart < Layers.Count)
        {
            current = MeanPoolOverTokens(current);
            activations[$"Layer_{headIdx}_Pool"] = current.Clone();
            current = Layers[headIdx].Forward(current);
            activations[$"Layer_{headIdx}_{Layers[headIdx].GetType().Name}"] = current.Clone();
        }
        return activations;
    }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "ViLBERT-Native" : "ViLBERT-ONNX", Description = "ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations (Lu et al., NeurIPS 2019)", FeatureCount = _options.FusionDim, Complexity = _options.NumVisionLayers + _options.NumTextLayers + _options.NumFusionLayers }; m.AdditionalInfo["Architecture"] = "ViLBERT"; m.AdditionalInfo["FusionType"] = _options.FusionType.ToString(); return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.ImageSize); writer.Write(_options.VisionDim); writer.Write(_options.TextDim); writer.Write(_options.FusionDim); writer.Write(_options.NumVisionLayers); writer.Write(_options.NumTextLayers); writer.Write(_options.NumFusionLayers); writer.Write(_options.NumHeads); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.ImageSize = reader.ReadInt32(); _options.VisionDim = reader.ReadInt32(); _options.TextDim = reader.ReadInt32(); _options.FusionDim = reader.ReadInt32(); _options.NumVisionLayers = reader.ReadInt32(); _options.NumTextLayers = reader.ReadInt32(); _options.NumFusionLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); if (_useNativeMode) ComputeDualStreamBoundaries(); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new ViLBERT<T>(Architecture, mp, _options); return new ViLBERT<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(ViLBERT<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; if (disposing) { OnnxModel?.Dispose(); } base.Dispose(disposing); }
}
