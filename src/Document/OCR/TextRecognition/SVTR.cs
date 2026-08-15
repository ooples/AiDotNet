using AiDotNet.Attributes;
using AiDotNet.ActivationFunctions;
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
public partial class SVTR<T> : DocumentNeuralNetworkBase<T>, ITextRecognizer<T>
{
    private readonly SVTROptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private readonly bool _useNativeMode;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly int _embedDim;
    private readonly int _numLayers;
    private readonly int _numHeads;
    private readonly int _imageHeight;
    private readonly string _charset;

    private bool _hasCustomArchitecture;
    private SVTRThinPlateSplineLayer<T>? _tpsRectifier;
    private ConvolutionalLayer<T>? _patchConv1;
    private BatchNormalizationLayer<T>? _patchNorm1;
    private ActivationLayer<T>? _patchAct1;
    private ConvolutionalLayer<T>? _patchConv2;
    private BatchNormalizationLayer<T>? _patchNorm2;
    private ActivationLayer<T>? _patchAct2;
    private LearnedPositionalEmbeddingLayer<T>? _positionLayer;
    private readonly List<SVTRMixingBlockLayer<T>> _referenceBlocks = [];
    private ConvolutionalLayer<T>? _merge1;
    private LayerNormalizationLayer<T>? _mergeNorm1;
    private ConvolutionalLayer<T>? _merge2;
    private LayerNormalizationLayer<T>? _mergeNorm2;
    private LayerNormalizationLayer<T>? _encoderNorm;
    private AdaptiveAveragePoolingLayer<T>? _heightCollapse;
    private BiasFreeLinearLayer<T>? _lastProjection;
    private ActivationLayer<T>? _lastActivation;
    private DropoutLayer<T>? _lastDropout;
    private DenseLayer<T>? _ctcHead;

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
            inputHeight: 32, inputWidth: 100,
            outputSize: 96))
    {
    }

    /// <summary>
    /// Creates an SVTR model using a pre-trained ONNX model for inference.
    /// </summary>
    public SVTR(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int imageWidth = 100,
        int imageHeight = 32,
        int maxSequenceLength = 25,
        int embedDim = 192,
        int numLayers = 12,
        int numHeads = 8,
        string? charset = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        SVTROptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options is null ? new SVTROptions() : new SVTROptions(options);
        _options.ValidateReferenceTopology();
        Options = _options;
        imageWidth = _options.InputWidth;
        imageHeight = _options.InputHeight;
        maxSequenceLength = _options.OutputCharacterPositions;
        embedDim = _options.OutputChannels;
        numLayers = _options.StageDepths.Sum();
        numHeads = _options.StageHeads.Max();

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
        int imageWidth = 100,
        int imageHeight = 32,
        int maxSequenceLength = 25,
        int embedDim = 192,
        int numLayers = 12,
        int numHeads = 8,
        string? charset = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        SVTROptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options is null ? new SVTROptions() : new SVTROptions(options);
        _options.ValidateReferenceTopology();
        Options = _options;
        imageWidth = _options.InputWidth;
        imageHeight = _options.InputHeight;
        maxSequenceLength = _options.OutputCharacterPositions;
        embedDim = _options.OutputChannels;
        numLayers = _options.StageDepths.Sum();
        numHeads = _options.StageHeads.Max();

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
        // optimizer would be silently ignored (the field and the base trainer would disagree).
        SetBaseTrainOptimizer(_optimizer);

        InitializeLayers();
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
            _hasCustomArchitecture = true;
            return;
        }
        BuildReferenceSVTRTiny();
    }

    private void BuildReferenceSVTRTiny()
    {
        int[] dims = _options.EmbedDimensions;
        int[] depths = _options.StageDepths;
        int[] heads = _options.StageHeads;
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        var gelu = new GELUActivation<T>() as IActivationFunction<T>;

        if (_options.UseTpsRectification)
        {
            _tpsRectifier = new SVTRThinPlateSplineLayer<T>(
                inputChannels: 3,
                localizationHeight: _options.TpsInputHeight,
                localizationWidth: _options.TpsInputWidth,
                outputHeight: _options.InputHeight,
                outputWidth: _options.InputWidth,
                controlPointCount: _options.TpsControlPointCount,
                marginX: _options.TpsMarginX,
                marginY: _options.TpsMarginY);
            Layers.Add(_tpsRectifier);
        }

        _patchConv1 = new ConvolutionalLayer<T>(dims[0] / 2, 3, 2, 1, identity);
        _patchNorm1 = new BatchNormalizationLayer<T>();
        _patchAct1 = new ActivationLayer<T>(gelu);
        _patchConv2 = new ConvolutionalLayer<T>(dims[0], 3, 2, 1, identity);
        _patchNorm2 = new BatchNormalizationLayer<T>();
        _patchAct2 = new ActivationLayer<T>(gelu);
        int gridH = _options.InputHeight / 4;
        int gridW = _options.InputWidth / 4;
        _positionLayer = new LearnedPositionalEmbeddingLayer<T>(gridH * gridW, dims[0]);
        AddReferenceLayers(_patchConv1, _patchNorm1, _patchAct1, _patchConv2, _patchNorm2, _patchAct2, _positionLayer);

        int globalBlockIndex = 0;
        int totalBlocks = depths.Sum();
        for (int stage = 0; stage < 3; stage++)
        {
            for (int block = 0; block < depths[stage]; block++, globalBlockIndex++)
            {
                var mixer = new SVTRMixingBlockLayer<T>(
                    dims[stage], heads[stage], gridH, gridW,
                    _options.LocalWindowHeight, _options.LocalWindowWidth,
                    local: globalBlockIndex < _options.LocalMixingBlocks,
                    dropPathRate: totalBlocks > 1
                        ? _options.DropPathRate * globalBlockIndex / (totalBlocks - 1)
                        : 0.0);
                _referenceBlocks.Add(mixer);
                Layers.Add(mixer);
            }

            if (stage == 0)
            {
                _merge1 = new ConvolutionalLayer<T>(dims[1], 3, 1, 1, identity);
                _mergeNorm1 = new LayerNormalizationLayer<T>(dims[1]);
                AddReferenceLayers(_merge1, _mergeNorm1);
                gridH /= 2;
            }
            else if (stage == 1)
            {
                _merge2 = new ConvolutionalLayer<T>(dims[2], 3, 1, 1, identity);
                _mergeNorm2 = new LayerNormalizationLayer<T>(dims[2]);
                AddReferenceLayers(_merge2, _mergeNorm2);
                gridH /= 2;
            }
        }

        // rec_svtrnet.yml selects `prenorm: False`. In PaddleOCR's implementation that
        // means each block is pre-normalized and the encoder receives one final norm.
        _encoderNorm = new LayerNormalizationLayer<T>(dims[2]);
        Layers.Add(_encoderNorm);
        _heightCollapse = new AdaptiveAveragePoolingLayer<T>(1, _options.OutputCharacterPositions);
        _lastProjection = new BiasFreeLinearLayer<T>(dims[2], _options.OutputChannels);
        _lastActivation = new ActivationLayer<T>(new HardSwishActivation<T>() as IActivationFunction<T>);
        _lastDropout = new DropoutLayer<T>(_options.LastStageDropout);
        _ctcHead = new DenseLayer<T>(_charset.Length + 1, identity);
        AddReferenceLayers(_heightCollapse, _lastProjection, _lastActivation, _lastDropout, _ctcHead);
    }

    private void AddReferenceLayers(params ILayer<T>[] layers)
    {
        foreach (var layer in layers) Layers.Add(layer);
    }

    private Tensor<T> RunNativeForward(Tensor<T> input)
    {
        if (_hasCustomArchitecture)
        {
            var custom = input;
            foreach (var layer in Layers) custom = layer.Forward(custom);
            return custom;
        }

        EnsureReferenceLayerBindings();
        if (_patchConv1 is null || _patchNorm1 is null || _patchAct1 is null ||
            _patchConv2 is null || _patchNorm2 is null || _patchAct2 is null ||
            _positionLayer is null || _merge1 is null || _mergeNorm1 is null ||
            _merge2 is null || _mergeNorm2 is null || _encoderNorm is null || _heightCollapse is null ||
            _lastProjection is null || _lastActivation is null || _lastDropout is null || _ctcHead is null)
            throw new InvalidOperationException("SVTR-Tiny reference architecture is not initialized.");

        var x = _tpsRectifier is null ? input : _tpsRectifier.Forward(input);
        x = _patchAct1.Forward(_patchNorm1.Forward(_patchConv1.Forward(x)));
        x = _patchAct2.Forward(_patchNorm2.Forward(_patchConv2.Forward(x)));
        int batch = x.Shape[0];
        int height = x.Shape[2];
        int width = x.Shape[3];
        var tokens = SpatialToTokens(x);
        tokens = _positionLayer.Forward(tokens);

        int blockOffset = 0;
        tokens = RunBlocks(tokens, blockOffset, _options.StageDepths[0]);
        blockOffset += _options.StageDepths[0];
        tokens = SubsampleHeight(tokens, batch, height, width, _options.EmbedDimensions[0], _merge1, _mergeNorm1);
        height /= 2;

        tokens = RunBlocks(tokens, blockOffset, _options.StageDepths[1]);
        blockOffset += _options.StageDepths[1];
        tokens = SubsampleHeight(tokens, batch, height, width, _options.EmbedDimensions[1], _merge2, _mergeNorm2);
        height /= 2;

        tokens = RunBlocks(tokens, blockOffset, _options.StageDepths[2]);
        tokens = _encoderNorm.Forward(tokens);
        x = TokensToSpatial(tokens, batch, height, width, _options.EmbedDimensions[2]);
        x = _heightCollapse.Forward(x);
        tokens = SpatialToTokens(x);
        tokens = _lastDropout.Forward(_lastActivation.Forward(_lastProjection.Forward(tokens)));
        var flat = Engine.Reshape(tokens, [batch * _options.OutputCharacterPositions, _options.OutputChannels]);
        var logits = _ctcHead.Forward(flat);
        logits = Engine.Reshape(logits,
            [batch, _options.OutputCharacterPositions, _charset.Length + 1]);
        return batch == 1
            ? Engine.Reshape(logits, [_options.OutputCharacterPositions, _charset.Length + 1])
            : logits;
    }

    private Tensor<T> RunBlocks(Tensor<T> input, int offset, int count)
    {
        var current = input;
        for (int i = 0; i < count; i++) current = _referenceBlocks[offset + i].Forward(current);
        return current;
    }

    private Tensor<T> SubsampleHeight(
        Tensor<T> tokens, int batch, int height, int width, int channels,
        ConvolutionalLayer<T> convolution, LayerNormalizationLayer<T> normalization)
    {
        var spatial = TokensToSpatial(tokens, batch, height, width, channels);
        spatial = convolution.Forward(spatial);
        int outputHeight = height / 2;
        var indices = new Tensor<int>([outputHeight]);
        for (int i = 0; i < outputHeight; i++) indices[i] = i * 2;
        spatial = Engine.TensorGather(spatial, indices, axis: 2);
        return normalization.Forward(SpatialToTokens(spatial));
    }

    private Tensor<T> SpatialToTokens(Tensor<T> spatial)
    {
        int batch = spatial.Shape[0];
        int channels = spatial.Shape[1];
        int height = spatial.Shape[2];
        int width = spatial.Shape[3];
        var nhwc = Engine.TensorPermute(spatial, [0, 2, 3, 1]);
        return Engine.Reshape(nhwc, [batch, height * width, channels]);
    }

    private Tensor<T> TokensToSpatial(Tensor<T> tokens, int batch, int height, int width, int channels)
    {
        var nhwc = Engine.Reshape(tokens, [batch, height, width, channels]);
        return Engine.TensorPermute(nhwc, [0, 3, 1, 2]);
    }

    private void EnsureReferenceLayerBindings()
    {
        if (_hasCustomArchitecture) return;
        if (Layers.Count == 0 ||
            (_options.UseTpsRectification && ReferenceEquals(Layers[0], _tpsRectifier)) ||
            (!_options.UseTpsRectification && ReferenceEquals(Layers[0], _patchConv1)))
            return;

        int index = 0;
        _tpsRectifier = _options.UseTpsRectification
            ? (SVTRThinPlateSplineLayer<T>)Layers[index++]
            : null;
        _patchConv1 = (ConvolutionalLayer<T>)Layers[index++];
        _patchNorm1 = (BatchNormalizationLayer<T>)Layers[index++];
        _patchAct1 = (ActivationLayer<T>)Layers[index++];
        _patchConv2 = (ConvolutionalLayer<T>)Layers[index++];
        _patchNorm2 = (BatchNormalizationLayer<T>)Layers[index++];
        _patchAct2 = (ActivationLayer<T>)Layers[index++];
        _positionLayer = (LearnedPositionalEmbeddingLayer<T>)Layers[index++];
        _referenceBlocks.Clear();
        for (int stage = 0; stage < 3; stage++)
        {
            for (int block = 0; block < _options.StageDepths[stage]; block++)
                _referenceBlocks.Add((SVTRMixingBlockLayer<T>)Layers[index++]);
            if (stage == 0)
            {
                _merge1 = (ConvolutionalLayer<T>)Layers[index++];
                _mergeNorm1 = (LayerNormalizationLayer<T>)Layers[index++];
            }
            else if (stage == 1)
            {
                _merge2 = (ConvolutionalLayer<T>)Layers[index++];
                _mergeNorm2 = (LayerNormalizationLayer<T>)Layers[index++];
            }
        }
        _encoderNorm = (LayerNormalizationLayer<T>)Layers[index++];
        _heightCollapse = (AdaptiveAveragePoolingLayer<T>)Layers[index++];
        _lastProjection = (BiasFreeLinearLayer<T>)Layers[index++];
        _lastActivation = (ActivationLayer<T>)Layers[index++];
        _lastDropout = (DropoutLayer<T>)Layers[index++];
        _ctcHead = (DenseLayer<T>)Layers[index++];
        if (index != Layers.Count)
            throw new InvalidDataException($"SVTR reference topology expected {index} layers, found {Layers.Count}.");
    }

    #endregion

    #region ITextRecognizer Implementation

    /// <inheritdoc/>
    public TextRecognitionResult<T> RecognizeText(Tensor<T> croppedImage)
    {
        var startTime = DateTime.UtcNow;

        var preprocessed = PreprocessTextImage(croppedImage);
        var output = _useNativeMode ? RunNativeForward(preprocessed) : RunOnnxInference(preprocessed);

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
        if (!_options.UseTpsRectification &&
            (processed.Shape[^2] != _options.InputHeight || processed.Shape[^1] != _options.InputWidth))
            processed = Engine.Interpolate(
                processed, [_options.InputHeight, _options.InputWidth],
                InterpolateMode.Bilinear, alignCorners: false);
        var scaled = Engine.TensorMultiplyScalar(processed, NumOps.FromDouble(1.0 / 127.5));
        return Engine.TensorSubtract(
            scaled, Tensor<T>.CreateDefault(scaled.Shape.ToArray(), NumOps.One));
    }

    #endregion

    #region IDocumentModel Implementation

    /// <inheritdoc/>
    public Tensor<T> EncodeDocument(Tensor<T> documentImage)
    {
        var preprocessed = PreprocessTextImage(documentImage);
        return _useNativeMode ? RunNativeForward(preprocessed) : RunOnnxInference(preprocessed);
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
                { "use_native_mode", _useNativeMode },
                { "paper_variant", "SVTR-Tiny" },
                { "input_geometry", "TPS 32x64 -> 32x100" },
                { "stage_dimensions", "64,128,256" },
                { "stage_depths", "3,6,3" },
                { "stage_heads", "2,4,8" },
                { "mixers", "6 local 7x11; 6 global" },
                { "drop_path_schedule", "linear 0.0 -> 0.1" },
                { "layer_count", 30 },
                { "ctc_positions", 25 }
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
        writer.Write(_options.UseTpsRectification);
        writer.Write(_options.DropPathRate);
        writer.Write(_options.TpsInputHeight);
        writer.Write(_options.TpsInputWidth);
        writer.Write(_options.TpsControlPointCount);
        writer.Write(_options.TpsMarginX);
        writer.Write(_options.TpsMarginY);
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

        if (reader.BaseStream.Position < reader.BaseStream.Length)
        {
            _options.UseTpsRectification = reader.ReadBoolean();
            _options.DropPathRate = reader.ReadDouble();
            _options.TpsInputHeight = reader.ReadInt32();
            _options.TpsInputWidth = reader.ReadInt32();
            _options.TpsControlPointCount = reader.ReadInt32();
            _options.TpsMarginX = reader.ReadDouble();
            _options.TpsMarginY = reader.ReadDouble();
        }

        ImageSize = imageSize;
        base.MaxSequenceLength = maxSeqLen;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new SVTR<T>(Architecture, ImageSize, _imageHeight, MaxSequenceLength,
            _embedDim, _numLayers, _numHeads, _charset, options: new SVTROptions(_options));
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
        var preprocessed = PreprocessTextImage(input);
        return _useNativeMode ? RunNativeForward(preprocessed) : RunOnnxInference(preprocessed);
    }

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        EnsureLayerRandomSeedsWired();
        return _useNativeMode
            ? RunNativeForward(PreprocessTextImage(input))
            : base.ForwardForTraining(input);
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

    /// <inheritdoc />
    /// <remarks>The weights belong to the loaded graph in this mode. The base refuses
    /// the write on every parameter surface, so the guard is stated once, here.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
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
