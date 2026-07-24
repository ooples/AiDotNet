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
using Microsoft.ML.OnnxRuntime;

namespace AiDotNet.Document.OCR.TextRecognition;

/// <summary>
/// CRNN (Convolutional Recurrent Neural Network) for sequence-based text recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CRNN combines CNN for image feature extraction with RNN (BiLSTM) for sequence modeling,
/// trained with CTC loss for variable-length text recognition without explicit character
/// segmentation.
/// </para>
/// <para>
/// <b>For Beginners:</b> CRNN works by:
/// 1. CNN extracts visual features from the text image
/// 2. BiLSTM models the sequence of features
/// 3. CTC decoding converts outputs to text
///
/// Key advantages:
/// - No need to segment individual characters
/// - Handles variable-length text
/// - End-to-end trainable
/// - Works with horizontal text lines
///
/// Example usage:
/// <code>
/// var model = new CRNN&lt;float&gt;(architecture);
/// var result = model.RecognizeText(croppedTextImage);
/// // Result is available in the returned value
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "An End-to-End Trainable Neural Network for Image-based Sequence Recognition" (TPAMI 2017)
/// https://arxiv.org/abs/1507.05717
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Detection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition", "https://doi.org/10.48550/arXiv.1507.05717", Year = 2017, Authors = "Baoguang Shi, Xiang Bai, Cong Yao")]
public class CRNN<T> : DocumentNeuralNetworkBase<T>, ITextRecognizer<T>
{
    private readonly CRNNOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private string? _onnxModelPath;
    private IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private int _cnnChannels;
    private int _rnnHiddenSize;
    private int _rnnLayers;
    private string _charset;

    private Tensor<T>? _lastCharacterProbs;

    // Native mode layers
    private readonly List<ILayer<T>> _cnnLayersList = [];
    private readonly List<ILayer<T>> _rnnLayersList = [];
    private readonly List<ILayer<T>> _outputLayersList = [];

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override DocumentType SupportedDocumentTypes => DocumentType.All;

    /// <inheritdoc/>
    public override bool RequiresOCR => false; // CRNN is the OCR recognizer     

    /// <inheritdoc/>
    public int ExpectedImageSize => ImageSize;

    /// <inheritdoc/>
    public string SupportedCharacters => _charset;

    /// <inheritdoc/>
    public new int MaxSequenceLength => base.MaxSequenceLength;

    /// <inheritdoc/>
    public bool SupportsAttentionVisualization => false;

    /// <summary>
    /// Gets the input image height expected by the model.
    /// </summary>
    public int ImageHeight => 32;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a CRNN model using a pre-trained ONNX model for inference.
    /// </summary>
    public CRNN(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int imageWidth = 128,
        int maxSequenceLength = 32,
        int cnnChannels = 512,
        int rnnHiddenSize = 256,
        int rnnLayers = 2,
        string? charset = null,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        CRNNOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new CRNNOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}", onnxModelPath);

        _useNativeMode = false;
        _cnnChannels = cnnChannels;
        _rnnHiddenSize = rnnHiddenSize;
        _rnnLayers = rnnLayers;
        _charset = charset ?? GetDefaultCharset();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _onnxModelPath = onnxModelPath;

        ImageSize = imageWidth;
        base.MaxSequenceLength = maxSequenceLength;

        _onnxSession = new InferenceSession(onnxModelPath);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a CRNN model using native layers for training and inference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Default Configuration (CRNN from TPAMI 2017):</b>
    /// - 7-layer CNN with batch normalization
    /// - 2-layer BiLSTM with 256 hidden units
    /// - CTC loss for sequence training
    /// - Input: 32×W×1 (grayscale) or 32×W×3 (RGB)
    /// </para>
    /// </remarks>
    public CRNN(
        NeuralNetworkArchitecture<T> architecture,
        int imageWidth = 128,
        int maxSequenceLength = 32,
        int cnnChannels = 512,
        int rnnHiddenSize = 256,
        int rnnLayers = 2,
        string? charset = null,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        CRNNOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new CRNNOptions();
        Options = _options;

        _useNativeMode = true;
        _cnnChannels = cnnChannels;
        _rnnHiddenSize = rnnHiddenSize;
        _rnnLayers = rnnLayers;
        _charset = charset ?? GetDefaultCharset();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _onnxModelPath = null;

        ImageSize = imageWidth;
        base.MaxSequenceLength = maxSequenceLength;

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
            return;
        }

        Layers.AddRange(LayerHelper<T>.CreateDefaultCRNNLayers(
            imageWidth: ImageSize,
            imageHeight: ImageHeight,
            cnnChannels: _cnnChannels,
            rnnHiddenSize: _rnnHiddenSize,
            rnnLayers: _rnnLayers,
            charsetSize: _charset.Length + 1, // +1 for CTC blank
            inputDepth: Architecture.InputDepth));
    }

    #endregion

    #region ITextRecognizer Implementation

    /// <inheritdoc/>
    public TextRecognitionResult<T> RecognizeText(Tensor<T> croppedImage)
    {
        var startTime = DateTime.UtcNow;

        var preprocessed = PreprocessTextImage(croppedImage);
        var output = _useNativeMode
            ? CanonicalizeCtcLogits(Forward(preprocessed))
            : CanonicalizeCtcLogits(RunOnnxInference(preprocessed));

        _lastCharacterProbs = output;

        // CTC decoding
        var (text, confidence, characters) = CTCDecode(output);

        return new TextRecognitionResult<T>
        {
            Text = text,
            Confidence = NumOps.FromDouble(confidence),
            ConfidenceValue = confidence,
            Characters = characters,
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
        return _lastCharacterProbs ?? Tensor<T>.CreateDefault([MaxSequenceLength, _charset.Length + 1], NumOps.Zero);
    }

    /// <inheritdoc/>
    public Tensor<T>? GetAttentionWeights()
    {
        return null; // CRNN doesn't use attention
    }

    private (string text, double confidence, List<CharacterRecognition<T>> characters) CTCDecode(Tensor<T> output)
    {
        var chars = new List<char>();
        var characterResults = new List<CharacterRecognition<T>>();
        double totalConf = 0;
        int validSteps = 0;
        int prevIdx = -1;

        var (seqLen, vocabSize, logitAt) = ResolveCtcOutput(output);

        for (int t = 0; t < seqLen; t++)
        {
            double maxLogit = double.MinValue;
            int maxIdx = 0;
            for (int c = 0; c < vocabSize; c++)
            {
                double logit = logitAt(t, c);
                if (logit > maxLogit) { maxLogit = logit; maxIdx = c; }
            }

            double sumExp = 0;
            for (int c = 0; c < vocabSize; c++)
            {
                double logit = logitAt(t, c);
                sumExp += Math.Exp(logit - maxLogit);
            }

            double prob = sumExp > 0 ? 1.0 / sumExp : 0.0;

            // Skip blank (index 0) and repeated characters
            if (maxIdx != 0 && maxIdx != prevIdx)
            {
                if (maxIdx - 1 < _charset.Length)
                {
                    char ch = _charset[maxIdx - 1];
                    chars.Add(ch);
                    characterResults.Add(new CharacterRecognition<T>
                    {
                        Character = ch,
                        Confidence = NumOps.FromDouble(prob),
                        ConfidenceValue = prob,
                        Position = chars.Count - 1
                    });
                    totalConf += prob;
                    validSteps++;
                }
            }
            prevIdx = maxIdx;
        }

        string text = new string([.. chars]);
        double avgConf = validSteps > 0 ? totalConf / validSteps : 0;

        return (text, avgConf, characterResults);
    }

    private Tensor<T> PreprocessTextImage(Tensor<T> image)
    {
        var processed = EnsureBatchDimension(image);

        // CRNN has a fixed-height image contract: every crop is resized to
        // [ImageHeight, ImageSize] before the CNN (Shi et al., 2017). The old
        // implementation only normalized, allowing the source instance's lazy
        // convolution geometry to adapt to an arbitrary caller size while a
        // fresh clone rebuilt from the configured dimensions. That changed the
        // spatial token count across Clone (e.g. 18,816 vs 1,536 outputs).
        // Nearest-neighbor sampling is deterministic and sufficient here; a
        // caller that wants higher-quality interpolation can install the public
        // preprocessing transformer.
        int batchSize = processed.Shape[0];
        int channels = processed.Shape[1];
        int sourceHeight = processed.Shape[2];
        int sourceWidth = processed.Shape[3];
        int targetHeight = ImageHeight;
        int targetWidth = ImageSize;

        var normalized = new Tensor<T>([batchSize, channels, targetHeight, targetWidth]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int y = 0; y < targetHeight; y++)
                {
                    int sourceY = Math.Min(sourceHeight - 1, y * sourceHeight / targetHeight);
                    for (int x = 0; x < targetWidth; x++)
                    {
                        int sourceX = Math.Min(sourceWidth - 1, x * sourceWidth / targetWidth);
                        double val = NumOps.ToDouble(processed[b, c, sourceY, sourceX]);
                        normalized[b, c, y, x] = NumOps.FromDouble((val / 255.0 - 0.5) * 2.0);
                    }
                }
            }
        }

        return normalized;
    }

    #endregion

    #region IDocumentModel Implementation

    /// <inheritdoc/>
    public Tensor<T> EncodeDocument(Tensor<T> documentImage)
    {
        var preprocessed = PreprocessTextImage(documentImage);
        var logits = _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);
        return CanonicalizeCtcLogits(logits);
    }

    /// <inheritdoc/>
    public void ValidateInputShape(Tensor<T> documentImage)
    {
        ValidateImageShape(documentImage);

        int height = documentImage.Shape[^2];
        if (height != ImageHeight)
        {
            throw new ArgumentException(
                $"CRNN expects image height {ImageHeight} but got {height}.",
                nameof(documentImage));
        }

        int width = documentImage.Shape[^1];
        if (width <= 0)
        {
            throw new ArgumentException("CRNN expects a positive image width.", nameof(documentImage));
        }
    }

    private (int timeSteps, int classCount, Func<int, int, double> logitAt) ResolveCtcOutput(Tensor<T> output)
    {
        int expectedClasses = _charset.Length + 1;

        if (output.Rank == 2)
        {
            int classDim;
            int timeDim;

            if (output.Shape[1] == expectedClasses)
            {
                classDim = 1;
                timeDim = 0;
            }
            else if (output.Shape[0] == expectedClasses)
            {
                classDim = 0;
                timeDim = 1;
            }
            else
            {
                classDim = 1;
                timeDim = 0;
            }

            int timeSteps = output.Shape[timeDim];
            int classCount = output.Shape[classDim];
            if (classDim == 1)
            {
                return (timeSteps, classCount, (t, c) => NumOps.ToDouble(output[t, c]));
            }

            return (timeSteps, classCount, (t, c) => NumOps.ToDouble(output[c, t]));
        }

        if (output.Rank == 3)
        {
            int classDim = Array.IndexOf(output._shape, expectedClasses);
            if (classDim < 0)
            {
                classDim = 2;
            }

            int dimA;
            int dimB;
            switch (classDim)
            {
                case 0:
                    dimA = 1;
                    dimB = 2;
                    break;
                case 1:
                    dimA = 0;
                    dimB = 2;
                    break;
                default:
                    dimA = 0;
                    dimB = 1;
                    break;
            }
            int batchDim = output.Shape[dimA] == 1 ? dimA : output.Shape[dimB] == 1 ? dimB : dimA;
            int timeDim = batchDim == dimA ? dimB : dimA;

            int timeSteps = output.Shape[timeDim];
            int classCount = output.Shape[classDim];
            int[] indices = new int[3];
            indices[batchDim] = 0;

            return (timeSteps, classCount, (t, c) =>
            {
                indices[timeDim] = t;
                indices[classDim] = c;
                return NumOps.ToDouble(output[indices]);
            }
            );
        }

        throw new ArgumentException("CTC output must be a 2D or 3D tensor.", nameof(output));
    }

    /// <inheritdoc/>
    public string GetModelSummary()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("CRNN Model Summary");
        sb.AppendLine("==================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine($"Architecture: VGG-style CNN + BiLSTM");
        sb.AppendLine($"CNN Channels: {_cnnChannels}");
        sb.AppendLine($"RNN Hidden Size: {_rnnHiddenSize}");
        sb.AppendLine($"RNN Layers: {_rnnLayers}");
        sb.AppendLine($"Image Height: {ImageHeight}");
        sb.AppendLine($"Image Width: {ImageSize}");
        sb.AppendLine($"Max Sequence Length: {MaxSequenceLength}");
        sb.AppendLine($"Charset Size: {_charset.Length}");
        sb.AppendLine($"Decoder: CTC (Greedy)");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        return sb.ToString();
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// Applies CRNN's industry-standard preprocessing: text image preprocessing.
    /// </summary>
    /// <remarks>
    /// CRNN (Convolutional Recurrent Neural Network) uses text-specific preprocessing
    /// with grayscale conversion and height normalization to 32px.
    /// </remarks>
    protected override Tensor<T> ApplyDefaultPreprocessing(Tensor<T> rawImage)
    {
        return PreprocessTextImage(rawImage);
    }

    /// <summary>
    /// Applies CRNN's industry-standard postprocessing: pass-through (CTC outputs are already final).
    /// </summary>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput) => modelOutput;

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "CRNN",
            Description = "CRNN for sequence text recognition (TPAMI 2017)",
            FeatureCount = _cnnChannels,
            Complexity = _rnnLayers,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "cnn_channels", _cnnChannels },
                { "rnn_hidden_size", _rnnHiddenSize },
                { "rnn_layers", _rnnLayers },
                { "charset_size", _charset.Length },
                { "image_height", ImageHeight },
                { "image_width", ImageSize },
                { "use_native_mode", _useNativeMode }
            },
            ModelData = SafeSerialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_cnnChannels);
        writer.Write(_rnnHiddenSize);
        writer.Write(_rnnLayers);
        writer.Write(ImageSize);
        writer.Write(MaxSequenceLength);
        writer.Write(_charset);
        writer.Write(_useNativeMode);
        writer.Write(_onnxModelPath ?? string.Empty);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        int cnnChannels = reader.ReadInt32();
        int rnnHiddenSize = reader.ReadInt32();
        int rnnLayers = reader.ReadInt32();
        int imageSize = reader.ReadInt32();
        int maxSeqLen = reader.ReadInt32();
        string charset = reader.ReadString();
        bool useNativeMode = reader.ReadBoolean();
        string? onnxModelPath = null;
        if (reader.BaseStream.Position < reader.BaseStream.Length)
        {
            onnxModelPath = reader.ReadString();
        }

        _cnnChannels = cnnChannels;
        _rnnHiddenSize = rnnHiddenSize;
        _rnnLayers = rnnLayers;
        _charset = charset;
        _useNativeMode = useNativeMode;
        _onnxModelPath = string.IsNullOrWhiteSpace(onnxModelPath) ? null : onnxModelPath;

        ImageSize = imageSize;
        base.MaxSequenceLength = maxSeqLen;

        // Native-mode layers (with their trained weights) are already reconstructed by
        // the base DeserializeInternalUnchecked before this override runs, so do NOT
        // clear + re-initialize them here — that would discard the deserialized weights
        // and leave the model randomly initialized. (In ONNX mode InitializeLayers is a
        // no-op, so dropping the call changes nothing there.)
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode)
        {
            string onnxModelPath = _onnxModelPath ?? throw new InvalidOperationException(
                "Missing ONNX model path required to clone CRNN instance.");
            if (string.IsNullOrWhiteSpace(onnxModelPath))
            {
                throw new InvalidOperationException(
                    "Missing ONNX model path required to clone CRNN instance.");
            }

            return new CRNN<T>(
                Architecture,
                onnxModelPath,
                ImageSize,
                MaxSequenceLength,
                _cnnChannels,
                _rnnHiddenSize,
                _rnnLayers,
                _charset,
                optimizer: null,
                lossFunction: LossFunction);
        }

        return new CRNN<T>(
            Architecture,
            ImageSize,
            MaxSequenceLength,
            _cnnChannels,
            _rnnHiddenSize,
            _rnnLayers,
            _charset,
            optimizer: null,
            lossFunction: LossFunction);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// CRNN's convolutional and dense layers resolve their parameter shapes on
    /// the first image forward. Recreate that resolved state before copying so
    /// a clone cannot silently retain freshly initialized lazy weights.
    /// </remarks>
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        var copy = (CRNN<T>)CreateNewInstance();
        if (copy.Layers.Count != Layers.Count)
            throw new InvalidOperationException("CRNN clone layer topology does not match the source model.");

        for (int i = 0; i < Layers.Count; i++)
        {
            var source = Layers[i];
            var destination = copy.Layers[i];
            int[] inputShape = source.GetInputShape();
            if (destination is LayerBase<T> destinationBase &&
                !destinationBase.IsShapeResolved &&
                inputShape.Length > 0 &&
                Array.TrueForAll(inputShape, dimension => dimension > 0))
            {
                destinationBase.ResolveFromShape(inputShape);
            }

            destination.SetParameters(source.GetParameters());
            if (source is ILayerSerializationExtras<T> sourceExtras &&
                destination is ILayerSerializationExtras<T> destinationExtras)
            {
                destinationExtras.SetExtraParameters(sourceExtras.GetExtraParameters());
            }
        }

        copy.SetTrainingMode(false);
        return copy;
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        var preprocessed = PreprocessTextImage(input);
        var logits = _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);
        return CanonicalizeCtcLogits(logits);
    }

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) =>
        CanonicalizeCtcLogits(Forward(input));

    /// <summary>
    /// Converts the CNN/recurrent head's spatial logits to the public CTC contract
    /// [batch, time, classes], pooling deterministically to MaxSequenceLength.
    /// </summary>
    private Tensor<T> CanonicalizeCtcLogits(Tensor<T> logits)
    {
        int classes = _charset.Length + 1;
        if (logits.Rank == 3 && logits.Shape[^1] == classes &&
            logits.Shape[1] == MaxSequenceLength)
            return logits;

        if (logits.Shape[^1] != classes)
            throw new InvalidOperationException(
                $"CRNN output must have {classes} classes in its final dimension, but got shape [{string.Join(", ", logits.Shape)}].");

        int batch = logits.Rank >= 3 ? logits.Shape[0] : 1;
        int positions = logits.Length / checked(batch * classes);
        var flattened = Engine.Reshape(logits, [batch, positions, classes]);
        int timeSteps = MaxSequenceLength;
        if (positions == timeSteps)
            return flattened;

        if (positions < timeSteps)
        {
            int repeats = (timeSteps + positions - 1) / positions;
            var expanded = Engine.TensorRepeatElements(flattened, repeats, axis: 1);
            return expanded.Shape[1] == timeSteps
                ? expanded
                : Engine.TensorSlice(expanded, [0, 0, 0], [batch, timeSteps, classes]);
        }

        if (positions % timeSteps != 0)
            return Engine.TensorSlice(flattened, [0, 0, 0], [batch, timeSteps, classes]);

        int positionsPerStep = positions / timeSteps;
        var grouped = Engine.Reshape(flattened, [batch, timeSteps, positionsPerStep, classes]);
        return Engine.ReduceMean(grouped, [2], keepDims: false);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");

        SetTrainingMode(true);
        try
        {
            var preprocessedInput = PreprocessTextImage(input);
            if (_optimizer is IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> gradientOptimizer)
                TrainWithTape(preprocessedInput, expectedOutput, gradientOptimizer);
            else
                TrainWithTape(preprocessedInput, expectedOutput);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> gradients)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Parameter updates not supported in ONNX mode.");

        var currentParams = GetParameters();
        T lr = NumOps.FromDouble(0.0001);
        
        currentParams = Engine.Subtract(currentParams, Engine.Multiply(gradients, lr));
        SetParameters(currentParams);
    }

    #endregion

    #region Disposal

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
            _onnxSession?.Dispose();
        base.Dispose(disposing);
    }

    #endregion
}
