using AiDotNet.ActivationFunctions;
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

namespace AiDotNet.VisionLanguage.Robotics;

/// <summary>
/// RT-2: vision-language-action model that transfers web knowledge to robotic control
/// (Brohan et al., Google DeepMind, 2023, arXiv:2307.15818).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// RT-2 fine-tunes a large vision-language model (PaLI-X or PaLM-E) on robot demonstration data
/// by representing continuous robot actions as text tokens. Each action dimension is uniformly
/// discretized into 256 bins (paper §3.2) and each bin is mapped to one of the 256 least
/// frequently used vocabulary tokens. The VLM then emits these tokens autoregressively just as
/// it would emit ordinary text, so all of the model's web-scale knowledge transfers directly to
/// robot control. This class composes a generic ViT-style vision encoder, MLP projection, LLM
/// decoder, action-bin head and full-vocabulary projection so the same forward path can serve
/// both vision-language reasoning and robotic action generation, mirroring the encoder-decoder
/// PaLI/PaLM-E backbone used in the paper.
/// </para>
/// <para><b>Paper-faithful pieces implemented here:</b></para>
/// <list type="bullet">
///   <item>256-bin per-dimension uniform action discretization mapped to vocabulary tokens via <see cref="RT2ActionTokenizer{T}"/> (paper §3.2).</item>
///   <item>Multimodal fusion via concatenated [visual_tokens, text_tokens] sequence fed into the decoder, matching PaLI/PaLM-E encoder-decoder context layout (paper §3.1).</item>
///   <item>Autoregressive greedy decoding of <c>ActionDimension × PredictionHorizon</c> action tokens, identical to the inference path used in the paper.</item>
///   <item>Full-vocabulary projection head so the same forward pass can serve VQA / co-fine-tuning batches (paper §4.2) and robot-action batches without architectural changes.</item>
/// </list>
/// <para><b>What is NOT verified in-session:</b></para>
/// <list type="bullet">
///   <item>Bit-exact numerical parity against a public PaLI-X / PaLM-E reference checkpoint (would require loading the original weights, which are not publicly released).</item>
///   <item>Behaviour on the Open-X / RT-2 evaluation suite (requires a physical robot or sim environment).</item>
/// </list>
/// <para><b>For Beginners:</b> RT-2 is the model that proved a generic Internet-trained vision-language
/// model can drive a real robot just by treating "move arm forward 3cm" as a sentence to generate,
/// rather than by training a separate policy network. Default values follow the published 55B-parameter
/// PaLI-X configuration but can be scaled down via <see cref="RT2Options"/>.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new RT2&lt;double&gt;(architecture, "rt2.onnx");
///
/// // Training mode with native layers
/// var trainModel = new RT2&lt;double&gt;(architecture, new RT2Options());
/// var action = trainModel.PredictAction(image, "pick up the red cup");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.Robotics)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control",
    "https://arxiv.org/abs/2307.15818",
    Year = 2023,
    Authors = "Brohan et al."
)]
public class RT2<T> : VisionLanguageModelBase<T>, IVisionLanguageAction<T>
{
    private readonly RT2Options _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer _tokenizer;
    private readonly RT2ActionTokenizer<T> _actionTokenizer;

    // Learned token embedding table — replaces the previous deterministic
    // sinusoidal placeholder in EmbedInstructionTokens /
    // AppendActionTokenEmbedding. Per Brohan et al. 2023 §3.1 RT-2's
    // text + action tokens flow through the same learned PaLI/PaLM-X
    // embedding table; the synthetic sin/cos vectors weren't model-
    // faithful and decoupled from the vocab head's training signal.
    // Sized for VocabSize (covers both natural language and the
    // action-bin tokens which live in a reserved window of the same
    // vocab — paper §3.2). Embedding dim = decoder dim so the
    // embedding table can be tied with the LM head's pre-projection
    // weights in a future weight-tying pass.
    private readonly EmbeddingLayer<T> _tokenEmbedding;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public override ModelOptions GetOptions() => _options;

    public RT2(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        RT2Options? options = null
    )
        : base(architecture)
    {
        _options = options ?? new RT2Options();
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
        _actionTokenizer = CreateActionTokenizer(_options);
        _tokenEmbedding = new EmbeddingLayer<T>(_options.VocabSize, _options.DecoderDim);
        InitializeLayers();
    }

    public RT2(
        NeuralNetworkArchitecture<T> architecture,
        RT2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new RT2Options();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        _actionTokenizer = CreateActionTokenizer(_options);
        _tokenEmbedding = new EmbeddingLayer<T>(_options.VocabSize, _options.DecoderDim);
        InitializeLayers();
    }

    public int EmbeddingDimension => _options.DecoderDim;
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;
    int IVisualEncoder<T>.ImageChannels => 3;
    public int MaxGenerationLength => _options.MaxGenerationLength;
    public int DecoderEmbeddingDim => _options.DecoderDim;
    public string LanguageModelName => _options.LanguageModelName;
    public int ActionDimension => _options.ActionDimension;

    /// <summary>Action tokenizer used during decode. Internal because
    /// it's a plumbing/helper type — the facade API
    /// (<see cref="PredictAction"/>, <see cref="GenerateFromImage"/>)
    /// is the supported way to interact with RT-2. Test/training-data
    /// preparation that needs to encode demo actions for cross-entropy
    /// loss can access this via InternalsVisibleTo from the test
    /// assembly.</summary>
    internal RT2ActionTokenizer<T> ActionTokenizer => _actionTokenizer;

    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return L2Normalize(OnnxModel.Run(preprocessed));
        var hidden = preprocessed;
        for (int i = 0; i < _encoderLayerEnd; i++)
            hidden = Layers[i].Forward(hidden);
        return L2Normalize(hidden);
    }

    /// <summary>
    /// Runs the full encoder + decoder pipeline once and returns vocabulary-sized logits at the final
    /// position. The default <see cref="PredictAction"/> wraps this with an autoregressive loop and
    /// greedy decoding over the action-bin token range.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
        {
            // Fail fast for prompted ONNX generation rather than silently
            // dropping the prompt and producing an image-only response.
            // RT-2's ONNX export currently bundles the vision encoder only;
            // language-conditioned decoding requires the full multimodal
            // graph (PaLI-X / PaLM-E decoder) which isn't represented in
            // the single-input ONNX path. Wiring it would require either
            // (a) exporting the decoder as a separate ONNX model and
            // running a two-stage pipeline, or (b) bundling both into one
            // multimodal ONNX graph — either path is significant work
            // tracked alongside the parallel ONNX-multimodal feature.
            if (!string.IsNullOrWhiteSpace(prompt))
            {
                throw new NotSupportedException(
                    "Prompted generation is not implemented for ONNX mode. "
                        + "Use the native (non-ONNX) constructor for RT-2 if you need "
                        + "language-conditioned action prediction, or call this method "
                        + "without a prompt to run vision-only ONNX inference."
                );
            }
            return OnnxModel.Run(preprocessed);
        }

        var encoderHidden = preprocessed;
        for (int i = 0; i < _encoderLayerEnd; i++)
            encoderHidden = Layers[i].Forward(encoderHidden);

        var fused = prompt is null
            ? encoderHidden
            : FuseVisualAndTextEmbeddings(encoderHidden, TokenizeText(prompt));

        var output = fused;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);
        return output;
    }

    /// <summary>
    /// Predicts a continuous robot action using RT-2's action-as-text formulation
    /// (Brohan et al. 2023, §3.2). Visual observation and language instruction are
    /// fused at the encoder boundary; the decoder + LM head then autoregressively
    /// generates <see cref="ActionDimension"/> × <see cref="PredictionHorizon"/>
    /// action-bin tokens by greedy selection inside the tokenizer's reserved vocabulary
    /// window. <see cref="RT2ActionTokenizer{T}"/> decodes the emitted tokens back to
    /// continuous actions in each dimension's range.
    /// </summary>
    public Tensor<T> PredictAction(Tensor<T> observation, string instruction)
    {
        ThrowIfDisposed();
        // ONNX-mode action prediction requires the full vision+language+
        // decoder pipeline (paper §3.1) which the current single-input
        // ONNX export doesn't represent. Fail fast rather than running
        // the native decode path while pretending the user-loaded ONNX
        // weights matter — that would silently bypass the loaded
        // checkpoint and produce actions from randomly-initialized
        // native layers instead.
        if (IsOnnxMode)
        {
            throw new NotSupportedException(
                "PredictAction is not implemented for ONNX mode — the loaded "
                    + "ONNX graph is vision-only. Use the native (non-ONNX) "
                    + "constructor to enable action prediction."
            );
        }
        int actionDim = _options.ActionDimension;
        int horizon = Math.Max(1, _options.PredictionHorizon);

        var preprocessed = PreprocessImage(observation);

        // Vision encoder: image patches → visual feature sequence.
        var encoderHidden = preprocessed;
        for (int i = 0; i < _encoderLayerEnd; i++)
            encoderHidden = Layers[i].Forward(encoderHidden);

        // Multimodal fusion: prepend visual tokens to instruction tokens (paper §3.1, PaLI-style context).
        var instrTokens = TokenizeText(instruction);
        var fused = FuseVisualAndTextEmbeddings(encoderHidden, instrTokens);

        int totalActionTokens = actionDim * horizon;
        var generatedTokens = new int[totalActionTokens];

        // Autoregressive decode (greedy argmax over the tokenizer's bin window).
        var decoderState = fused;
        for (int step = 0; step < totalActionTokens; step++)
        {
            var hidden = decoderState;
            for (int i = _encoderLayerEnd; i < Layers.Count; i++)
                hidden = Layers[i].Forward(hidden);

            int nextToken = _actionTokenizer.GreedyActionToken(hidden);
            generatedTokens[step] = nextToken;

            decoderState = AppendActionTokenEmbedding(decoderState, nextToken);
        }

        if (horizon == 1)
            return _actionTokenizer.DecodeAction(generatedTokens);
        return _actionTokenizer.DecodeHorizon(generatedTokens, horizon);
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            // Custom architectures need an EXPLICIT encoder/decoder
            // boundary. The previous "Layers.Count / 2" heuristic
            // silently routed half the layers into the encoder stack
            // and half into the decoder, which only happened to be
            // correct for the default RT-2 topology. A user-supplied
            // architecture with, say, 3 vision layers + 6 decoder
            // layers would put one decoder layer into the encoder
            // stack and one vision layer into the decoder stack —
            // producing valid-looking shapes but completely wrong
            // semantics. Fail fast and require the metadata until
            // RT2Options exposes an explicit EncoderLayerCount /
            // DecoderLayerCount pair for custom architectures.
            throw new NotSupportedException(
                "Custom RT2 architectures (Architecture.Layers populated) "
                    + "must declare an explicit encoder/decoder boundary. The "
                    + "previous 'half-and-half' split was a heuristic that "
                    + "silently misrouted layers when vision/decoder counts "
                    + "weren't equal. Either use the default-topology "
                    + "constructor (Architecture.Layers null), or extend "
                    + "RT2Options with an EncoderLayerCount property and pin "
                    + "_encoderLayerEnd to that value here."
            );
        }

        Layers.AddRange(
            LayerHelper<T>.CreateDefaultRoboticsActionLayers(
                visionDim: _options.VisionDim,
                decoderDim: _options.DecoderDim,
                actionDim: _actionTokenizer.NumBins,
                numVisionLayers: _options.NumVisionLayers,
                numDecoderLayers: _options.NumDecoderLayers,
                numActionLayers: 2,
                numHeads: _options.NumHeads,
                dropoutRate: _options.DropoutRate
            )
        );

        // Vocabulary-projection head (paper §3.1): RT-2 emits ordinary text tokens (for the
        // VQA co-fine-tuning batches) and action-bin tokens (for the robot-trajectory batches)
        // through the SAME projection so that web-scale knowledge transfers. We replace the
        // action-only head with an LN + VocabSize-dimensional dense layer; the tokenizer slices
        // the reserved 256-bin window at decode time.
        IActivationFunction<T> vocabHeadActivation = new IdentityActivation<T>();
        Layers.Add(new LayerNormalizationLayer<T>());
        Layers.Add(new DenseLayer<T>(_options.VocabSize, vocabHeadActivation));

        ComputeEncoderDecoderBoundary();
        ValidateEncoderDecoderBoundary(_encoderLayerEnd);
    }

    private void ComputeEncoderDecoderBoundary()
    {
        int layersPerBlock = TransformerBlockLayerCount(_options.DropoutRate);
        // Vision stack: 1 leading LN + numVisionLayers × block + 2 projection layers (Dense + LN).
        _encoderLayerEnd = 1 + _options.NumVisionLayers * layersPerBlock + 2;
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

    /// <summary>
    /// Concatenates visual encoder output with instruction-token embeddings along the sequence
    /// dimension to form the PaLI-style joint context vector per Brohan et al. 2023 §3.1.
    /// Text-token embedding goes through the learned <see cref="_tokenEmbedding"/> table so
    /// it shares the same representation the LM head is trained against.
    /// </summary>
    private Tensor<T> FuseVisualAndTextEmbeddings(
        Tensor<T> visualFeatures,
        Tensor<T> instructionTokens
    )
    {
        if (instructionTokens.Length == 0)
            return visualFeatures;
        var textEmbed = EmbedTokenIds(instructionTokens);
        return visualFeatures.ConcatenateTensors(textEmbed);
    }

    /// <summary>
    /// Looks up token embeddings through the shared trainable
    /// <see cref="_tokenEmbedding"/> layer. Replaces the previous
    /// deterministic sinusoidal placeholder that fabricated sin/cos
    /// vectors from token IDs — those weren't model-faithful and
    /// decoupled the encoder/decoder input representation from the
    /// vocab-head training signal. Per Brohan et al. 2023 §3.1, text
    /// tokens AND action-bin tokens use the same learned embedding
    /// table; this method is the shared lookup path.
    /// </summary>
    private Tensor<T> EmbedTokenIds(Tensor<T> tokenIds)
    {
        return _tokenEmbedding.Forward(tokenIds);
    }

    /// <summary>
    /// Appends the learned embedding of the most-recently-generated
    /// action-bin token to the running decoder context. Goes through
    /// the same <see cref="_tokenEmbedding"/> table as text tokens
    /// (action bins live in a reserved window of the vocab per paper
    /// §3.2), so autoregressive decoding stays consistent with the
    /// representation the LM head was trained against.
    /// </summary>
    private Tensor<T> AppendActionTokenEmbedding(Tensor<T> decoderState, int tokenId)
    {
        var singleTokenInput = new Tensor<T>([1]);
        singleTokenInput[0] = NumOps.FromDouble(tokenId);
        var tokenEmbed = _tokenEmbedding.Forward(singleTokenInput);
        return decoderState.ConcatenateTensors(tokenEmbed);
    }

    private static RT2ActionTokenizer<T> CreateActionTokenizer(RT2Options options)
    {
        return new RT2ActionTokenizer<T>(
            actionDim: options.ActionDimension,
            numBins: 256,
            vocabSize: options.VocabSize
        );
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        var hidden = input;
        foreach (var layer in Layers)
            hidden = layer.Forward(hidden);
        return hidden;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expected);
        }
        finally
        {
            // If TrainWithTape throws (NaN gradient, optimizer state
            // corruption, layer-side numerical issue, etc.) we still
            // need to flip training mode back off — otherwise the next
            // Predict() on this instance would run dropout / batchnorm
            // in training mode and produce silently-incorrect outputs.
            SetTrainingMode(false);
        }
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Cannot update parameters in ONNX mode.");
        int idx = 0;
        foreach (var layer in Layers)
        {
            int count = (int)layer.ParameterCount;
            layer.UpdateParameters(parameters.Slice(idx, count));
            idx += count;
        }
    }

    protected override Tensor<T> PreprocessImage(Tensor<T> image) =>
        NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var meta = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "RT-2-Native" : "RT-2-ONNX",
            Description =
                "RT-2: vision-language-action model that transfers web knowledge to robotic control (Brohan et al., 2023, arXiv:2307.15818).",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        meta.AdditionalInfo["Architecture"] = "RT-2";
        meta.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        meta.AdditionalInfo["ActionBins"] = _actionTokenizer.NumBins.ToString();
        meta.AdditionalInfo["ActionDimension"] = _options.ActionDimension.ToString();
        meta.AdditionalInfo["VocabularySize"] = _options.VocabSize.ToString();
        meta.AdditionalInfo["ActionTokenWindow"] =
            $"[{_actionTokenizer.TokenIdOffset}, {_actionTokenizer.TokenIdEndExclusive})";
        return meta;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.ImageSize);
        writer.Write(_options.VisionDim);
        writer.Write(_options.DecoderDim);
        writer.Write(_options.NumVisionLayers);
        writer.Write(_options.NumDecoderLayers);
        writer.Write(_options.NumHeads);
        writer.Write(_options.ActionDimension);
        writer.Write(_options.VocabSize);
        writer.Write(_options.PredictionHorizon);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp))
            _options.ModelPath = mp;
        _options.ImageSize = reader.ReadInt32();
        _options.VisionDim = reader.ReadInt32();
        _options.DecoderDim = reader.ReadInt32();
        _options.NumVisionLayers = reader.ReadInt32();
        _options.NumDecoderLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        _options.ActionDimension = reader.ReadInt32();
        _options.VocabSize = reader.ReadInt32();
        _options.PredictionHorizon = reader.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new RT2<T>(Architecture, mp, _options);
        return new RT2<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(RT2<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
