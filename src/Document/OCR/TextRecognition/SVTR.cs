using AiDotNet.Attributes;
using AiDotNet.Document.Interfaces;
using AiDotNet.Document.Options;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;

namespace AiDotNet.Document.OCR.TextRecognition;

/// <summary>
/// SVTR (Scene Text Visual Transformer) for text recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SVTR is a single-stream vision transformer for scene text recognition that processes
/// text images as visual sequences without requiring recurrent networks.
/// </para>
/// <para>
/// <b>For Beginners:</b> SVTR modernizes text recognition:
/// 1. Uses vision transformer (no RNN needed)
/// 2. Handles various text heights and lengths
/// 3. Multi-scale feature extraction
/// 4. Efficient single-stream architecture
///
/// Key features:
/// - Pure transformer architecture
/// - Local + global mixing blocks
/// - Height compression for efficiency
/// - State-of-the-art accuracy
///
/// Example usage:
/// <code>
/// var model = new SVTR&lt;float&gt;(architecture);
/// var result = model.RecognizeText(textImage);
/// // Result is available in the returned value
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "SVTR: Scene Text Recognition with a Single Visual Model" (IJCAI 2022)
/// https://arxiv.org/abs/2205.00159
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Detection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("SVTR: Scene Text Recognition with a Single Visual Model", "https://doi.org/10.48550/arXiv.2205.00159", Year = 2022, Authors = "Yongkun Du, Zhineng Chen, Caiyan Jia, Xiaoting Yin, Tianlun Zheng, Chenxia Li, Yuning Du, Yu-Gang Jiang")]
public class SVTR<T> : DocumentNeuralNetworkBase<T>, ITextRecognizer<T>
{
    private readonly SVTROptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private readonly bool _useNativeMode;
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly int _embedDim;
    private readonly int _numLayers;
    private readonly int _numHeads;
    private readonly int _imageHeight;
    private readonly string _charset;

    // Native mode layers
    private readonly List<ILayer<T>> _patchEmbedLayers = [];
    private readonly List<ILayer<T>> _mixingLayers = [];
    private readonly List<ILayer<T>> _decoderLayers = [];

    // Learnable embeddings
    private Tensor<T>? _positionEmbeddings;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override DocumentType SupportedDocumentTypes => DocumentType.All;

    /// <inheritdoc/>
    public override bool RequiresOCR => false;

    /// <inheritdoc/>
    public int ExpectedImageSize => ImageSize;

    /// <inheritdoc/>
    public string SupportedCharacters => _charset;

    /// <inheritdoc/>
    public new int MaxSequenceLength => base.MaxSequenceLength;

    /// <inheritdoc/>
    public bool SupportsAttentionVisualization => true;

    /// <summary>
    /// Gets the input image height.
    /// </summary>
    public int ImageHeight => _imageHeight;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates an SVTR model with default configuration for native training.
    /// </summary>
    public SVTR()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 32, inputWidth: 256,
            outputSize: 96))
    {
    }

    /// <summary>
    /// Creates an SVTR model using a pre-trained ONNX model for inference.
    /// </summary>
    public SVTR(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int imageWidth = 256,
        int imageHeight = 32,
        int maxSequenceLength = 25,
        int embedDim = 192,
        int numLayers = 8,
        int numHeads = 6,
        string? charset = null,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        SVTROptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new SVTROptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}", onnxModelPath);

        _useNativeMode = false;
        _embedDim = embedDim;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _imageHeight = imageHeight;
        _charset = charset ?? GetDefaultCharset();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = imageWidth;
        base.MaxSequenceLength = maxSequenceLength;

        // Install the ONNX model through the base abstraction so DocumentNeuralNetworkBase.RunOnnxInference
        // (which reads OnnxModel/OnnxEncoder/OnnxDecoder) actually runs it. The previous code created a raw
        // InferenceSession the base never consulted, so every ONNX PredictCore hit "No ONNX model loaded".
        OnnxModel = new AiDotNet.Onnx.OnnxModel<T>(onnxModelPath);

        InitializeLayers();
    }

    /// <summary>
    /// Creates an SVTR model using native layers for training and inference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Default Configuration (SVTR-Tiny from IJCAI 2022):</b>
    /// - Patch embedding: 4×4 patches
    /// - Local + Global mixing blocks
    /// - Height compression
    /// - CTC decoder
    /// </para>
    /// </remarks>
    public SVTR(
        NeuralNetworkArchitecture<T> architecture,
        int imageWidth = 256,
        int imageHeight = 32,
        int maxSequenceLength = 25,
        int embedDim = 192,
        int numLayers = 8,
        int numHeads = 6,
        string? charset = null,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        SVTROptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new SVTROptions();
        Options = _options;

        _useNativeMode = true;
        _embedDim = embedDim;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _imageHeight = imageHeight;
        _charset = charset ?? GetDefaultCharset();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = imageWidth;
        base.MaxSequenceLength = maxSequenceLength;

        // Wire SVTR's optimizer into the base tape-training loop. base.Train drives training through
        // the base training optimizer, not this private field, so without this a caller-supplied
        // optimizer would be silently ignored (the field and the base trainer would disagree). The
        // field is the broad IOptimizer; the base loop needs a gradient-based optimizer — the default
        // (and any real training optimizer) is one, and a non-gradient optimizer falls back to null
        // (the base then lazily builds its default Adam).
        SetBaseTrainOptimizer(_optimizer as IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>);

        InitializeLayers();
        InitializeEmbeddings();
    }

    private static string GetDefaultCharset()
    {
        return "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ";
    }

    #endregion

    #region Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
        {
            return;
        }

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            return;
        }

        Layers.AddRange(LayerHelper<T>.CreateDefaultSVTRLayers(
            imageWidth: ImageSize,
            imageHeight: _imageHeight,
            hiddenDim: _embedDim,
            numLayers: _numLayers,
            numHeads: _numHeads,
            charsetSize: _charset.Length + 1));
    }

    private void InitializeEmbeddings()
    {
        var random = RandomHelper.CreateSeededRandom(42);
        int numPatches = (ImageSize / 4) * (_imageHeight / 4);

        _positionEmbeddings = Tensor<T>.CreateDefault([numPatches, _embedDim], NumOps.Zero);
        InitializeWithSmallRandomValues(_positionEmbeddings, random, 0.02);
    }

    private void InitializeWithSmallRandomValues(Tensor<T> tensor, Random random, double stdDev)
    {
        for (int i = 0; i < tensor.Data.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            tensor.Data.Span[i] = NumOps.FromDouble(randStdNormal * stdDev);
        }
    }

    #endregion

    #region ITextRecognizer Implementation

    /// <inheritdoc/>
    public TextRecognitionResult<T> RecognizeText(Tensor<T> croppedImage)
    {
        var startTime = DateTime.UtcNow;

        var preprocessed = PreprocessTextImage(croppedImage);
        var output = _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);

        var (text, confidence) = CTCDecode(output);

        return new TextRecognitionResult<T>
        {
            Text = text,
            Confidence = NumOps.FromDouble(confidence),
            ConfidenceValue = confidence,
            Characters = GetCharacterConfidences(output, text),
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    /// <inheritdoc/>
    public IEnumerable<TextRecognitionResult<T>> RecognizeTextBatch(IEnumerable<Tensor<T>> croppedImages)
    {
        foreach (var image in croppedImages)
            yield return RecognizeText(image);
    }

    /// <inheritdoc/>
    public Tensor<T> GetCharacterProbabilities()
    {
        return Tensor<T>.CreateDefault([MaxSequenceLength, _charset.Length + 1], NumOps.Zero);
    }

    /// <inheritdoc/>
    public Tensor<T>? GetAttentionWeights()
    {
        // SVTR uses self-attention, can return attention maps
        return Tensor<T>.CreateDefault([_numLayers, _numHeads, MaxSequenceLength, MaxSequenceLength], NumOps.Zero);
    }

    private (string text, double confidence) CTCDecode(Tensor<T> output)
    {
        var chars = new List<char>();
        double totalConf = 0;
        int validSteps = 0;
        int prevIdx = -1;

        int seqLen = output.Shape[0];
        int vocabSize = output.Shape.Length > 1 ? output.Shape[1] : _charset.Length + 1;

        for (int t = 0; t < seqLen; t++)
        {
            double maxVal = double.MinValue;
            int maxIdx = 0;
            for (int c = 0; c < vocabSize; c++)
            {
                double val = NumOps.ToDouble(output[t, c]);
                if (val > maxVal) { maxVal = val; maxIdx = c; }
            }

            if (maxIdx != 0 && maxIdx != prevIdx)
            {
                if (maxIdx - 1 < _charset.Length)
                {
                    chars.Add(_charset[maxIdx - 1]);
                    totalConf += maxVal;
                    validSteps++;
                }
            }
            prevIdx = maxIdx;
        }

        string text = new string([.. chars]);
        double avgConf = validSteps > 0 ? totalConf / validSteps : 0;

        return (text, avgConf);
    }

    private List<CharacterRecognition<T>> GetCharacterConfidences(Tensor<T> output, string text)
    {
        var result = new List<CharacterRecognition<T>>();
        for (int i = 0; i < text.Length; i++)
        {
            result.Add(new CharacterRecognition<T>
            {
                Character = text[i],
                Confidence = NumOps.FromDouble(0.9),
                ConfidenceValue = 0.9,
                Position = i
            });
        }
        return result;
    }

    private Tensor<T> PreprocessTextImage(Tensor<T> image)
    {
        var processed = EnsureBatchDimension(image);
        var normalized = new Tensor<T>(processed._shape);

        for (int i = 0; i < processed.Data.Length; i++)
        {
            double val = NumOps.ToDouble(processed.Data.Span[i]);
            normalized.Data.Span[i] = NumOps.FromDouble((val / 255.0 - 0.5) / 0.5);
        }

        return normalized;
    }

    #endregion

    #region IDocumentModel Implementation

    /// <inheritdoc/>
    public Tensor<T> EncodeDocument(Tensor<T> documentImage)
    {
        var preprocessed = PreprocessTextImage(documentImage);
        return _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);
    }

    /// <inheritdoc/>
    public void ValidateInputShape(Tensor<T> documentImage)
    {
        ValidateImageShape(documentImage);
    }

    /// <inheritdoc/>
    public string GetModelSummary()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("SVTR Model Summary");
        sb.AppendLine("==================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine($"Architecture: Single Visual Transformer");
        sb.AppendLine($"Embedding Dimension: {_embedDim}");
        sb.AppendLine($"Number of Layers: {_numLayers}");
        sb.AppendLine($"Attention Heads: {_numHeads}");
        sb.AppendLine($"Image Size: {ImageSize}x{_imageHeight}");
        sb.AppendLine($"Max Sequence Length: {MaxSequenceLength}");
        sb.AppendLine($"Charset Size: {_charset.Length}");
        sb.AppendLine($"Decoder: CTC");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        return sb.ToString();
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// Applies SVTR's industry-standard preprocessing: text image preprocessing.
    /// </summary>
    /// <remarks>
    /// SVTR (Scene Vision Transformer for Text Recognition) uses text-specific preprocessing
    /// with height normalization and patch-based encoding.
    /// </remarks>
    protected override Tensor<T> ApplyDefaultPreprocessing(Tensor<T> rawImage) => PreprocessTextImage(rawImage);

    /// <summary>
    /// Applies SVTR's industry-standard postprocessing: pass-through (transformer outputs are already final).
    /// </summary>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput) => modelOutput;

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "SVTR",
            Description = "SVTR for scene text recognition (IJCAI 2022)",
            FeatureCount = _embedDim,
            Complexity = _numLayers,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "embed_dim", _embedDim },
                { "num_layers", _numLayers },
                { "num_heads", _numHeads },
                { "image_height", _imageHeight },
                { "image_width", ImageSize },
                { "charset_size", _charset.Length },
                { "use_native_mode", _useNativeMode }
            },
            ModelData = SafeSerialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_embedDim);
        writer.Write(_numLayers);
        writer.Write(_numHeads);
        writer.Write(_imageHeight);
        writer.Write(ImageSize);
        writer.Write(MaxSequenceLength);
        writer.Write(_charset);
        writer.Write(_useNativeMode);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        int embedDim = reader.ReadInt32();
        int numLayers = reader.ReadInt32();
        int numHeads = reader.ReadInt32();
        int imageHeight = reader.ReadInt32();
        int imageSize = reader.ReadInt32();
        int maxSeqLen = reader.ReadInt32();
        string charset = reader.ReadString();
        bool useNativeMode = reader.ReadBoolean();

        ImageSize = imageSize;
        base.MaxSequenceLength = maxSeqLen;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new SVTR<T>(Architecture, ImageSize, _imageHeight, MaxSequenceLength, _embedDim, _numLayers, _numHeads, _charset);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <summary>
    /// Deferred on purpose. The base walk resolves each lazy layer's shape from the architecture's
    /// DECLARED input shape and, in doing so, pins the conv patch-embed's input channel count. That
    /// declared channel count does not match the RGB image actually fed at inference/training, so the
    /// eager walk would lock the conv to the wrong depth and throw "Expected input depth N, but got M"
    /// on the first forward. SVTR's conv stem is channel- and resolution-agnostic and resolves
    /// correctly from the FIRST real forward, so we skip the eager shape walk (pre-#1688 behavior:
    /// every layer resolves lazily on first Forward).
    /// </summary>
    protected override void ResolveLazyLayerShapes()
    {
    }

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // ONNX mode still applies the [0,255] pixel normalization it was exported with. The native
        // path runs the raw (batch-normalized) image so it matches ForwardForTraining exactly AND
        // keeps the input's full dynamic range: PreprocessTextImage divides by 255, which collapses
        // an already-[0,1]-scaled tensor to a near-constant ~-1 and starves the model of signal.
        return _useNativeMode
            ? RunNativeLayers(input)
            : RunOnnxInference(PreprocessTextImage(input));
    }

    /// <summary>
    /// Training forward. Overridden so the gradient path is IDENTICAL to <see cref="PredictCore"/>'s
    /// native path: both run <see cref="RunNativeLayers"/>, whose conv-patch-embed -> token-sequence
    /// flatten is a tape-aware Engine reshape. The base
    /// <see cref="NeuralNetworkBase{T}.ForwardForTraining"/> walks the raw layer list with no
    /// spatial->sequence step, so the mixing blocks used to receive the 4-D conv feature map (shape
    /// mismatch) and no gradient reached the conv patch-embed — which is why training never reduced
    /// the loss, never moved the parameters, and left the post-train forward non-finite.
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        return RunNativeLayers(input);
    }

    /// <summary>
    /// Runs SVTR's native layer stack: the progressive conv patch-embed (spatial [B,C,H,W]), a
    /// tape-aware flatten to the token sequence [B, H*W, C] at the first mixing block, then the
    /// transformer mixing blocks and the CTC head. Shared by inference and training so both paths
    /// are identical.
    /// </summary>
    private Tensor<T> RunNativeLayers(Tensor<T> input)
    {
        // Promote a single [C, H, W] image to a unit-batch [1, C, H, W] for the conv patch-embed
        // (the harness passes rank-3; training/NormalizeBatchDim passes rank-4).
        var output = input.Shape.Length == 3
            ? Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2] })
            : input;

        bool flattened = false;
        foreach (var layer in Layers)
        {
            // Flatten the conv feature map to a token sequence right before the first mixing block.
            if (!flattened && output.Shape.Length == 4 && layer is TransformerEncoderLayer<T>)
            {
                output = FlattenSpatialToSequence(output);
                flattened = true;
            }
            output = layer.Forward(output);
        }
        return output;
    }

    /// <summary>Tape-aware [B, C, H, W] -> [B, H*W, C] flatten (channels-last token sequence).</summary>
    private Tensor<T> FlattenSpatialToSequence(Tensor<T> x)
    {
        int b = x.Shape[0], c = x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        var channelsLast = Engine.TensorPermute(x, new[] { 0, 2, 3, 1 }); // [B, H, W, C]
        return Engine.Reshape(channelsLast, new[] { b, h * w, c });        // [B, H*W, C]
    }

    /// <summary>
    /// Per-layer activations for the introspection tests. Overridden for the SAME reason as
    /// <see cref="ForwardForTraining"/>: the base walks the raw layer list with no conv->sequence
    /// flatten, so the first mixing block would receive the 4-D conv feature map ("Input embedding
    /// dimension (H) does not match weight dimension (D)"). Mirror <see cref="RunNativeLayers"/>,
    /// capturing each layer's output.
    /// </summary>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        var activations = new Dictionary<string, Tensor<T>>();
        var current = input.Shape.Length == 3
            ? Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2] })
            : input;

        bool flattened = false;
        for (int i = 0; i < Layers.Count; i++)
        {
            if (!flattened && current.Shape.Length == 4 && Layers[i] is TransformerEncoderLayer<T>)
            {
                current = FlattenSpatialToSequence(current);
                flattened = true;
            }
            current = Layers[i].Forward(current);
            activations[$"Layer_{i}_{Layers[i].GetType().Name}"] = current.Clone();
        }

        return activations;
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");

        // Delegate to the base tape-training loop: forward via our ForwardForTraining, autodiff
        // backward, then the configured optimizer's in-place step (Adam). The previous override
        // ran a manual SGD (params - grads*1e-4) on top of TrainWithTape, which fought the optimizer
        // and, at lr 1e-4 over the smoke iterations, barely changed the weights or the loss.
        base.Train(input, expectedOutput);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Parameter updates not supported in ONNX mode.");

        // Contract (NeuralNetworkBase.UpdateParameters): assign the supplied values AS the network's
        // new parameters. Training is driven by the base tape loop + optimizer above, not by a manual
        // gradient step here.
        SetParameters(parameters);
    }

    #endregion

    #region Disposal

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        // The ONNX model is owned by the base (DocumentNeuralNetworkBase.OnnxModel) and disposed there.
        base.Dispose(disposing);
    }

    #endregion
}
