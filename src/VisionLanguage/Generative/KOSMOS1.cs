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
/// KOSMOS-1: multimodal large language model with visual tokens embedded in causal LM.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// KOSMOS-1 (Huang et al., 2023) is a multimodal large language model that embeds visual tokens
/// directly into a causal language model. Image features from a CLIP ViT are linearly projected
/// into the same embedding space as text tokens, then the combined sequence is processed by a
/// causal transformer decoder for unified multimodal understanding and generation.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Language Is Not All You Need: Aligning Perception with Language Models" (Huang et al., 2023)</item></list></para>
/// <para><b>For Beginners:</b> KOSMOS-1 from Microsoft embeds image features directly into
/// a causal language model as if they were text tokens. Image patches from a CLIP ViT are
/// linearly projected into the same embedding space as text, then the combined image-text
/// sequence is processed by a standard causal transformer for unified multimodal understanding
/// and generation. Default values follow the original paper settings.</para>
/// <para><b>Architecture layout:</b> Vision encoder + projection live in
/// <see cref="NeuralNetworkBase{T}.Layers"/>; the causal transformer decoder lives in a private
/// auxiliary stream. <see cref="Predict"/> returns the vision-only embedding;
/// <see cref="GenerateFromImage"/> walks both streams to generate the multimodal output.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
/// var trainModel = new KOSMOS1&lt;double&gt;(architecture, new KOSMOS1Options());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Language Is Not All You Need: Aligning Perception with Language Models",
    "https://arxiv.org/abs/2302.14045",
    Year = 2023,
    Authors = "Huang et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2e-4, Beta1 = 0.9, Beta2 = 0.98,
                WarmupSteps = 375, MinLearningRate = 0,
                Schedule = LearningRateSchedulerType.LinearWarmup,
                PostWarmupDecay = LinearWarmupScheduler.DecayMode.Linear,
                Source = "Huang et al. 2023, Sec. 3: AdamW with betas (0.9, 0.98), the learning rate "
                        + "increasing to 2e-4 over the first 375 warming-up steps and then decaying "
                        + "linearly to 0.")]
public partial class KOSMOS1<T> : VisionLanguageModelBase<T>, IGenerativeVisionLanguageModel<T>
{
    private readonly KOSMOS1Options _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private readonly bool _useNativeMode;
    private KosmosModelCore<T>? _core;
    private bool _disposed;

    /// <summary>Creates the model backed by an ONNX export.</summary>
    public KOSMOS1(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        KOSMOS1Options? options = null
    )
        : base(architecture)
    {
        _options = options ?? new KOSMOS1Options();
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
    public KOSMOS1(
        NeuralNetworkArchitecture<T> architecture,
        KOSMOS1Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new KOSMOS1Options();
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

    private KosmosModelCore<T> Core => _core ?? throw new NotSupportedException("KOSMOS-1 is in ONNX mode; the native model is not built.");

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
                "Custom Architecture.Layers is not supported for KOSMOS-1: its vision encoder, resampler and decoder " +
                "exchange image embeddings through the token sequence, which a flat layer list cannot express.");
        _core = new KosmosModelCore<T>(_options, true, _options.ResamplerDepth, KosmosPositionEncoding.XPos);
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
            Name = _useNativeMode ? "KOSMOS-1-Native" : "KOSMOS-1-ONNX",
            Description = "KOSMOS-1: Language Is Not All You Need: Aligning Perception with Language Models (Huang et al., 2023)",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "KOSMOS-1";
        m.AdditionalInfo["GenerativeType"] = _options.ArchitectureType.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(KOSMOS1<T>));
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
