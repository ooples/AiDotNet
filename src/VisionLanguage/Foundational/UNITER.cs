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
/// UNITER (Universal Image-TExt Representation) with conditional masking pre-training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// UNITER (Chen et al., ECCV 2020) uses a single-stream transformer with conditional masking where
/// either image regions or text tokens are masked during pre-training, forcing the model to learn
/// cross-modal alignment. Four pre-training tasks: MLM, MRM (with KL divergence), ITM, and WRA.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "UNITER: UNiversal Image-TExt Representation Learning" (Chen et al., ECCV 2020)</item></list></para>
/// <para><b>For Beginners:</b> UNITER is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a UNITER model for universal image-text representation
/// // with conditional masking pre-training for cross-modal alignment
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new UNITER&lt;double&gt;(architecture, "uniter.onnx");
///
/// // Training mode with native layers
/// var trainModel = new UNITER&lt;double&gt;(architecture, new UNITEROptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "UNITER: UNiversal Image-TExt Representation Learning",
    "https://arxiv.org/abs/1909.11740",
    Year = 2020,
    Authors = "Chen et al."
)]
public partial class UNITER<T> : VisionLanguageModelBase<T>, IVisionLanguageFusionModel<T>
{
    private readonly UNITEROptions _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _projectionLayerEnd;

    // Paper §3 (Chen et al. 2020) appends a task-specific classifier on
    // top of the pooled transformer output for every downstream task
    // (VQA, NLVR2, VCR, VE, image-text retrieval, RefCOCO). Index of the
    // task head layer (a Dense(FusionDim, Architecture.OutputSize))
    // emitted at the tail of <see cref="Layers"/> so the stream-aware
    // Predict / ForwardForTraining can apply it after the shared
    // transformer.
    private int _taskHeadIdx;

    public UNITER(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        UNITEROptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new UNITEROptions();
        _useNativeMode = false;
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.FusionDim;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    public UNITER(
        NeuralNetworkArchitecture<T> architecture,
        UNITEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new UNITEROptions();
        _options.Validate();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                // The official UNITER recipe uses AdamW with transformer-scale
                // learning rates, beta2=0.98, epsilon=1e-6, weight decay, and
                // global-norm clipping. Previously this configured optimizer was
                // never passed to TrainWithTape, which silently substituted the
                // framework's generic Adam at 1e-3.
                InitialLearningRate = _options.LearningRate,
                WeightDecay = _options.WeightDecay,
                Beta1 = 0.9,
                Beta2 = 0.98,
                Epsilon = 1e-6,
                EnableGradientClipping = true,
                MaxGradientNorm = _options.MaxGradientNorm,
                SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
                LearningRateScheduler = new AiDotNet.LearningRateSchedulers.LinearWarmupScheduler(
                    baseLearningRate: _options.LearningRate,
                    warmupSteps: _options.WarmupSteps,
                    totalSteps: _options.TotalTrainingSteps,
                    warmupInitLr: _options.WarmupInitialLearningRate
                        ?? _options.LearningRate / System.Math.Max(1, _options.WarmupSteps),
                    decayMode: AiDotNet.LearningRateSchedulers.LinearWarmupScheduler.DecayMode.Linear,
                    endLr: _options.EndLearningRate),
            });
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.FusionDim;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    public int EmbeddingDimension => _options.FusionDim;
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;
    int IVisualEncoder<T>.ImageChannels => 3;
    public int FusionEmbeddingDim => _options.FusionDim;
    public int MaxSequenceLength => _options.MaxSequenceLength;

    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return L2Normalize(OnnxModel.Run(p));
        // Single-stream: run through all layers
        var c = p;
        for (int i = 0; i < Layers.Count; i++)
            c = Layers[i].Forward(c);
        return L2Normalize(c);
    }

    public Tensor<T> FuseImageText(Tensor<T> image, string text)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        // Project image features to fusion dim
        var imageProj = p;
        for (int i = 0; i < _projectionLayerEnd; i++)
            imageProj = Layers[i].Forward(imageProj);

        // UNITER uses conditional masking: single-stream processes concatenated image+text jointly.
        var textTokens = TokenizeText(text);
        var combined = imageProj.ConcatenateTensors(textTokens);
        var c = combined;
        for (int i = _projectionLayerEnd; i < Layers.Count; i++)
            c = Layers[i].Forward(c);
        return c;
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
            for (int i = 0; i < Layers.Count; i++)
                c = Layers[i].Forward(c);
            textEmb = L2Normalize(c);
        }
        return CosineSimilarity(imageEmb, textEmb);
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _projectionLayerEnd = 0;
            _taskHeadIdx = Layers.Count;
        }
        else
        {
            Layers.AddRange(
                LayerHelper<T>.CreateDefaultSingleStreamFusionLayers(
                    _options.VisionDim,
                    _options.TextDim,
                    _options.FusionDim,
                    _options.NumFusionLayers,
                    _options.NumHeads,
                    _options.DropoutRate
                )
            );
            _projectionLayerEnd =
                (_options.VisionDim != _options.FusionDim ? 2 : 0)
                + (_options.TextDim != _options.FusionDim ? 2 : 0);

            // Paper §3 task head: pooled transformer output → Dense → OutputSize.
            _taskHeadIdx = Layers.Count;
            AiDotNet.Interfaces.IActivationFunction<T> idAct =
                new AiDotNet.ActivationFunctions.IdentityActivation<T>();
            Layers.Add(
                new AiDotNet.NeuralNetworks.Layers.DenseLayer<T>(Architecture.OutputSize, idAct)
            );
        }
    }

    /// <summary>Mean-pool a rank-2 [N, D] tensor over the N axis to [D]; pass through otherwise.</summary>
    /// <summary>
    /// Mean-pools token embeddings [tokens, dim] down to a single [dim] vector for the task head.
    /// </summary>
    /// <remarks>
    /// This must be a RECORDED reduction. Accumulating the mean element by element produces a tensor
    /// with no history on the autodiff tape, and because this sits between the encoder stack and the
    /// task head it cuts the backward pass in half: only the head receives gradients while every
    /// encoder layer stays frozen, and the head then optimizes against features that can never adapt.
    /// The identical helper in VinVL made that model's training loss RISE in proportion to the
    /// learning rate until it was switched to Engine.ReduceMean.
    /// </remarks>
    private Tensor<T> MeanPoolOverTokens(Tensor<T> input)
    {
        if (input.Shape.Length != 2)
            return input;

        return Engine.ReduceMean(input, new[] { 0 }, keepDims: false);
    }

    /// <summary>
    /// Shared forward that runs the single-stream transformer and task head.
    /// Chen et al. 2020 feeds region features directly into the projection
    /// layer (Dense(VisionDim, FusionDim)) — raw pixels are not a valid
    /// input because Faster-RCNN is a separate upstream stage.
    /// </summary>
    private Tensor<T> RunStream(Tensor<T> input)
    {
        // All non-head layers form the projection + transformer stack.
        var c = input;
        int end = _taskHeadIdx;
        for (int i = 0; i < end; i++)
            c = Layers[i].Forward(c);
        // Task head: pooled → Dense(FusionDim, OutputSize).
        if (_taskHeadIdx < Layers.Count)
        {
            c = MeanPoolOverTokens(c);
            c = Layers[_taskHeadIdx].Forward(c);
        }
        return c;
    }

    private Tensor<T> TokenizeText(string text)
    {
        if (_tokenizer is null)
            throw new InvalidOperationException("Tokenizer not initialized.");
        var encoding = _tokenizer.Encode(text);
        int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength);
        var tokens = new Tensor<T>([seqLen]);
        for (int i = 0; i < seqLen; i++)
            tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]);
        return tokens;
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        SetTrainingMode(false);
        return RunStream(input);
    }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => RunStream(input);

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        TrainWithTape(input, expected, _optimizer);
        SetTrainingMode(false);
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PreprocessImage(Tensor<T> image) =>
        NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "UNITER-Native" : "UNITER-ONNX",
            Description =
                "UNITER: UNiversal Image-TExt Representation Learning (Chen et al., ECCV 2020)",
            FeatureCount = _options.FusionDim,
            Complexity = _options.NumFusionLayers,
        };
        m.AdditionalInfo["Architecture"] = "UNITER";
        m.AdditionalInfo["FusionType"] = _options.FusionType.ToString();
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(UNITER<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        if (disposing)
        {
            OnnxModel?.Dispose();
        }
        base.Dispose(disposing);
    }
}
