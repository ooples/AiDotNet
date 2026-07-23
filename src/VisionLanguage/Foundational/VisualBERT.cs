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
/// VisualBERT: single-stream transformer that concatenates visual and text tokens.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// VisualBERT (Li et al., 2019) concatenates Faster R-CNN region features with text token embeddings
/// and processes them in a single BERT-style transformer, enabling implicit cross-modal alignment
/// through self-attention over the combined sequence.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "VisualBERT: A Simple and Performant Baseline for Vision and Language" (Li et al., 2019)</item></list></para>
/// <para><b>For Beginners:</b> VisualBERT is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a VisualBERT model for vision-language understanding
/// // with concatenated visual and text tokens in a single BERT stream
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new VisualBERT&lt;double&gt;(architecture, "visualbert.onnx");
///
/// // Training mode with native layers
/// var trainModel = new VisualBERT&lt;double&gt;(architecture, new VisualBERTOptions());
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
    "VisualBERT: A Simple and Performant Baseline for Vision and Language",
    "https://arxiv.org/abs/1908.03557",
    Year = 2019,
    Authors = "Li et al."
)]
public class VisualBERT<T> : VisionLanguageModelBase<T>, IVisionLanguageFusionModel<T>
{
    private readonly VisualBERTOptions _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _projectionLayerEnd;

    // Task head index — Li et al. 2019 VisualBERT uses a pooled-token →
    // Dense projection for every downstream task (VQA, VCR, NLVR2,
    // Flickr30k). Stored at the tail of Layers so RunStream can apply
    // it after the shared transformer.
    private int _taskHeadIdx;

    public VisualBERT(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        VisualBERTOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new VisualBERTOptions();
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

    public VisualBERT(
        NeuralNetworkArchitecture<T> architecture,
        VisualBERTOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new VisualBERTOptions();
        _useNativeMode = true;
        // AdamW's default InitialLearningRate is 0.01, which is too aggressive here: on the memorization
        // task the model reaches loss ~0.006 by 50 iterations and then OSCILLATES around the minimum with
        // Adam momentum, so 200-iteration loss came out slightly HIGHER than 50-iteration
        // (MoreData_ShouldNotDegrade). The oscillation amplitude scales with the LR, so a small 2e-4 LR
        // keeps the descent monotonic through 200 iterations while still converging.
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = 0.0002 });
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
        // Project image then run through joint transformer
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

        // VisualBERT concatenates visual region features with text tokens in a single BERT stream.
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
            _taskHeadIdx = Layers.Count;
            AiDotNet.Interfaces.IActivationFunction<T> idAct =
                new AiDotNet.ActivationFunctions.IdentityActivation<T>();
            Layers.Add(
                new AiDotNet.NeuralNetworks.Layers.DenseLayer<T>(Architecture.OutputSize, idAct)
            );
        }
    }

    private Tensor<T> MeanPoolOverTokens(Tensor<T> input)
    {
        if (input.Shape.Length != 2)
            return input;
        // Tape-aware mean over the token axis (axis 0): [n, d] -> [d]. The previous scalar nested loop
        // built the pooled tensor element-by-element, which SEVERS the gradient tape — so during training
        // no gradient flowed back through the pool into the transformer body; only the task head learned,
        // and LossStrictlyDecreases / MoreData failed. Engine.ReduceMean keeps the op on the tape.
        return Engine.ReduceMean(input, [0], keepDims: false);
    }

    private Tensor<T> RunStream(Tensor<T> input)
    {
        var c = input;
        int end = _taskHeadIdx;
        for (int i = 0; i < end; i++)
            c = Layers[i].Forward(c);
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
        // Pass the model's own optimizer (AdamW @ LR 0.001) — the 2-arg TrainWithTape falls back to the
        // base default optimizer (LR 0.01), so the model's configured, more-stable LR was being ignored
        // (MoreData 200-iter overshoot).
        TrainWithTape(input, expected, _optimizer);
        SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Cannot update parameters in ONNX mode.");
        int idx = 0;
        foreach (var l in Layers)
        {
            int c = (int)l.ParameterCount;
            l.UpdateParameters(parameters.Slice(idx, c));
            idx += c;
        }
    }

    protected override Tensor<T> PreprocessImage(Tensor<T> image) =>
        NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "VisualBERT-Native" : "VisualBERT-ONNX",
            Description =
                "VisualBERT: A Simple and Performant Baseline for Vision and Language (Li et al., 2019)",
            FeatureCount = _options.FusionDim,
            Complexity = _options.NumFusionLayers,
        };
        m.AdditionalInfo["Architecture"] = "VisualBERT";
        m.AdditionalInfo["FusionType"] = _options.FusionType.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.ImageSize);
        writer.Write(_options.VisionDim);
        writer.Write(_options.TextDim);
        writer.Write(_options.FusionDim);
        writer.Write(_options.NumFusionLayers);
        writer.Write(_options.NumHeads);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp))
            _options.ModelPath = mp;
        _options.ImageSize = reader.ReadInt32();
        _options.VisionDim = reader.ReadInt32();
        _options.TextDim = reader.ReadInt32();
        _options.FusionDim = reader.ReadInt32();
        _options.NumFusionLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
        if (_useNativeMode)
            _projectionLayerEnd =
                (_options.VisionDim != _options.FusionDim ? 2 : 0)
                + (_options.TextDim != _options.FusionDim ? 2 : 0);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new VisualBERT<T>(Architecture, mp, _options);
        return new VisualBERT<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(VisualBERT<T>));
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
