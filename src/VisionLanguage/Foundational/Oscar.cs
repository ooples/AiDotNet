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
/// Oscar (Object-Semantics Aligned pre-training) using object tags as anchor points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Oscar (Li et al., ECCV 2020) uses detected object tags as "anchor points" to ease cross-modal
/// alignment. The input to the transformer is a triple of (word tokens, object tags, region features),
/// processed in a single BERT stream. Pre-training uses masked token loss and contrastive loss.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks" (Li et al., ECCV 2020)</item></list></para>
/// <para><b>For Beginners:</b> Oscar is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create an Oscar model for vision-language tasks
/// // using object tags as anchor points for cross-modal alignment
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new Oscar&lt;double&gt;(architecture, "oscar.onnx");
///
/// // Training mode with native layers
/// var trainModel = new Oscar&lt;double&gt;(architecture, new OscarOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Embedding)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks", "https://arxiv.org/abs/2004.06165", Year = 2020, Authors = "Li et al.")]
public class Oscar<T> : VisionLanguageModelBase<T>, IVisionLanguageFusionModel<T>
{
    private readonly OscarOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _projectionLayerEnd;
    // Task head index — Li et al. 2020 Oscar appends a task-specific
    // head (Dense projection) on top of the pooled output for image
    // captioning, VQA, GQA, NLVR2, text-image retrieval (paper §4).
    private int _taskHeadIdx;

    public Oscar(NeuralNetworkArchitecture<T> architecture, string modelPath, OscarOptions? options = null) : base(architecture) { _options = options ?? new OscarOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.FusionDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public Oscar(NeuralNetworkArchitecture<T> architecture, OscarOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new OscarOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.FusionDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.FusionDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int FusionEmbeddingDim => _options.FusionDim; public int MaxSequenceLength => _options.MaxSequenceLength;

    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p));
        // Single-stream: run through all layers
        var c = p;
        for (int i = 0; i < Layers.Count; i++) c = Layers[i].Forward(c);
        return L2Normalize(c);
    }

    public Tensor<T> FuseImageText(Tensor<T> image, string text)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        // Project image features to fusion dim
        var imageProj = p;
        for (int i = 0; i < _projectionLayerEnd; i++) imageProj = Layers[i].Forward(imageProj);

        // Oscar uses object tags as anchor points between word tokens and region features.
        // In single-stream, image and text are concatenated and processed jointly.
        var textTokens = TokenizeText(text);
        var combined = imageProj.ConcatenateTensors(textTokens);
        var c = combined;
        for (int i = _projectionLayerEnd; i < Layers.Count; i++) c = Layers[i].Forward(c);
        return c;
    }

    public T ComputeMatchingScore(Tensor<T> image, string text)
    {
        var imageEmb = EncodeImage(image);
        var textTokens = TokenizeText(text);
        Tensor<T> textEmb;
        if (IsOnnxMode && OnnxModel is not null) { textEmb = L2Normalize(OnnxModel.Run(textTokens)); }
        else { var c = textTokens; for (int i = 0; i < Layers.Count; i++) c = Layers[i].Forward(c); textEmb = L2Normalize(c); }
        return CosineSimilarity(imageEmb, textEmb);
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _projectionLayerEnd = 0;
            _taskHeadIdx = Layers.Count;
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultSingleStreamFusionLayers(
                _options.VisionDim, _options.TextDim, _options.FusionDim,
                _options.NumFusionLayers, _options.NumHeads, _options.DropoutRate));
            _projectionLayerEnd = (_options.VisionDim != _options.FusionDim ? 2 : 0)
                                 + (_options.TextDim != _options.FusionDim ? 2 : 0);
            _taskHeadIdx = Layers.Count;
            AiDotNet.Interfaces.IActivationFunction<T> idAct =
                new AiDotNet.ActivationFunctions.IdentityActivation<T>();
            Layers.Add(new AiDotNet.NeuralNetworks.Layers.DenseLayer<T>(
                _options.FusionDim, Architecture.OutputSize, idAct));
        }
    }

    private static Tensor<T> MeanPoolOverTokens(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        if (rank != 2) return input;
        int n = input.Shape[0]; int d = input.Shape[1];
        var output = new Tensor<T>([d]);
        T invN = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().FromDouble(1.0 / n);
        for (int i = 0; i < d; i++)
        {
            T sum = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().Zero;
            for (int j = 0; j < n; j++)
                sum = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().Add(sum, input[j, i]);
            output[i] = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().Multiply(sum, invN);
        }
        return output;
    }

    private Tensor<T> RunStream(Tensor<T> input)
    {
        var c = input;
        int end = _taskHeadIdx;
        for (int i = 0; i < end; i++) c = Layers[i].Forward(c);
        if (_taskHeadIdx < Layers.Count)
        {
            c = MeanPoolOverTokens(c);
            c = Layers[_taskHeadIdx].Forward(c);
        }
        return c;
    }

    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>(); SetTrainingMode(false); return RunStream(input); }
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => RunStream(input);
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); TrainWithTape(input, expected); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "Oscar-Native" : "Oscar-ONNX", Description = "Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks (Li et al., ECCV 2020)", FeatureCount = _options.FusionDim, Complexity = _options.NumFusionLayers }; m.AdditionalInfo["Architecture"] = "Oscar"; m.AdditionalInfo["FusionType"] = _options.FusionType.ToString(); return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.ImageSize); writer.Write(_options.VisionDim); writer.Write(_options.TextDim); writer.Write(_options.FusionDim); writer.Write(_options.NumFusionLayers); writer.Write(_options.NumHeads); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.ImageSize = reader.ReadInt32(); _options.VisionDim = reader.ReadInt32(); _options.TextDim = reader.ReadInt32(); _options.FusionDim = reader.ReadInt32(); _options.NumFusionLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); if (_useNativeMode) _projectionLayerEnd = (_options.VisionDim != _options.FusionDim ? 2 : 0) + (_options.TextDim != _options.FusionDim ? 2 : 0); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new Oscar<T>(Architecture, mp, _options); return new Oscar<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Oscar<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; if (disposing) { OnnxModel?.Dispose(); } base.Dispose(disposing); }
}
