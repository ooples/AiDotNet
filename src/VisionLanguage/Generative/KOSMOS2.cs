using AiDotNet.LearningRateSchedulers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
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
/// <para><b>For Beginners:</b> KOSMOS-2 extends KOSMOS-1 with visual grounding — the ability
/// to link words in generated text to specific bounding box locations in the image. It uses
/// special location tokens to encode bounding box coordinates, enabling the model to output
/// phrases like "the dog &lt;box&gt;x1,y1,x2,y2&lt;/box&gt;" that point to objects in the image.
/// Default values follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
/// var trainModel = new KOSMOS2&lt;double&gt;(architecture, new KOSMOS2Options());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.Detection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Kosmos-2: Grounding Multimodal Large Language Models to the World",
    "https://arxiv.org/abs/2306.14824",
    Year = 2023,
    Authors = "Peng et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2e-4, Beta1 = 0.9, Beta2 = 0.98,
                WarmupSteps = 375, MinLearningRate = 0,
                Schedule = LearningRateSchedulerType.LinearWarmup,
                PostWarmupDecay = LinearWarmupScheduler.DecayMode.Linear,
                Source = "Peng et al. 2023, Sec. 3: the AdamW optimizer with betas (0.9, 0.98), the "
                        + "learning rate increasing to 2e-4 over the first 375 warm-up steps and then "
                        + "decaying linearly to zero.")]
public partial class KOSMOS2<T> : VisionLanguageModelBase<T>, IGenerativeVisionLanguageModel<T>
{
    private readonly KOSMOS2Options _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private readonly bool _useNativeMode;
    private KosmosModelCore<T>? _core;
    private bool _disposed;

    /// <summary>Creates the model backed by an ONNX export.</summary>
    public KOSMOS2(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        KOSMOS2Options? options = null
    )
        : base(architecture)
    {
        _options = options ?? new KOSMOS2Options();
        SyncImageSizeWithArchitecture();
        _useNativeMode = false;
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    /// <summary>Creates a trainable native model.</summary>
    public KOSMOS2(
        NeuralNetworkArchitecture<T> architecture,
        KOSMOS2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new KOSMOS2Options();
        SyncImageSizeWithArchitecture();
        _useNativeMode = true;
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    private void SyncImageSizeWithArchitecture()
    {
        int h = Architecture.InputHeight;
        int w = Architecture.InputWidth;
        if (h > 0 && w > 0 && h == w)
            _options.ImageSize = h;
    }

    /// <inheritdoc/>
    public int EmbeddingDimension => _options.DecoderDim;
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;
    int IVisualEncoder<T>.ImageChannels => 3;

    /// <inheritdoc/>
    public int MaxGenerationLength => _options.MaxGenerationLength;

    /// <inheritdoc/>
    public int DecoderEmbeddingDim => _options.DecoderDim;

    private KosmosModelCore<T> Core => _core ?? throw new NotSupportedException("KOSMOS-2 is in ONNX mode; the native model is not built.");

    /// <summary>The <c>NumImageTokens</c> image embeddings the decoder reads, <c>[NumImageTokens, DecoderDim]</c>.</summary>
    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);
        SetTrainingMode(false);
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return Core.ImageEmbeddings(p);
    }

    /// <summary>
    /// Greedy generation after <c>&lt;s&gt; &lt;image&gt; [image] &lt;/image&gt;</c> and the prompt; returns the generated
    /// token ids (ending at EOS or after <c>MaxGenerationLength</c> tokens).
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);
        SetTrainingMode(false);
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return Core.Generate(p, prompt is null ? Array.Empty<int>() : TokenizeText(prompt));
    }

    /// <summary>Next-token logits <c>[NumImageTokens + 3 + prompt, VocabSize]</c> for an image and prompt token ids.</summary>
    public Tensor<T> PredictTokens(Tensor<T> image, IReadOnlyList<int> promptIds)
    {
        ThrowIfDisposed();
        if (promptIds is null) throw new ArgumentNullException(nameof(promptIds));
        SetTrainingMode(false);
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return Core.Logits(PreprocessImage(image), promptIds);
    }

    /// <summary>Trains on one image and its caption token ids (next-token cross-entropy on the caption).</summary>
    public void TrainCaption(Tensor<T> image, IReadOnlyList<int> captionIds)
    {
        if (captionIds is null) throw new ArgumentNullException(nameof(captionIds));
        var target = new Tensor<T>(new[] { captionIds.Count });
        for (int i = 0; i < captionIds.Count; i++) target[i] = NumOps.FromDouble(captionIds[i]);
        Train(image, target);
    }

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
            throw new NotSupportedException(
                "Custom Architecture.Layers is not supported for KOSMOS-2: its vision encoder, resampler and decoder " +
                "exchange image embeddings through the token sequence, which a flat layer list cannot express.");
        _core = new KosmosModelCore<T>(_options, false, 1, KosmosPositionEncoding.Sinusoidal);
        Layers.AddRange(_core.Layers());
    }

    private int[] TokenizeText(string text)
    {
        if (_tokenizer is null)
            throw new InvalidOperationException("Tokenizer not initialized.");
        return _tokenizer.Encode(text).TokenIds.Take(_options.MaxSequenceLength).ToArray();
    }

    /// <inheritdoc/>
    /// <remarks>
    /// An image in; next-token logits for <c>&lt;s&gt; &lt;image&gt; [image] &lt;/image&gt;</c> out,
    /// <c>[NumImageTokens + 3, VocabSize]</c>. The last row is the distribution of the caption's first token.
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        SetTrainingMode(false);
        return Core.Logits(PreprocessImage(input), Array.Empty<int>());
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <paramref name="expected"/> is either caption token ids <c>[T]</c> (next-token cross-entropy on the caption,
    /// the pre-training objective) or a <c>[NumImageTokens + 3, VocabSize]</c> soft target over
    /// <see cref="PredictCore"/>'s positions.
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (expected is null) throw new ArgumentNullException(nameof(expected));
        var core = Core;
        TrainWithCustomObjective(PreprocessImage(input), expected, (image, target) => core.Loss(image, target), _optimizer);
    }

    /// <summary>
    /// Parameters cannot be written while the model is backed by a loaded ONNX graph.
    /// </summary>
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessImage(Tensor<T> image) =>
        NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "KOSMOS-2-Native" : "KOSMOS-2-ONNX",
            Description = "Kosmos-2: Grounding Multimodal Large Language Models to the World (Peng et al., 2023)",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "KOSMOS-2";
        m.AdditionalInfo["GenerativeType"] = _options.ArchitectureType.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(KOSMOS2<T>));
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        if (disposing)
        {
            // OnnxModel wraps a native ONNX Runtime session; dispose it so create/dispose cycles do not leak.
            OnnxModel?.Dispose();
        }
        base.Dispose(disposing);
    }
}
