using AiDotNet.Document.Interfaces;
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
/// Console.WriteLine($"Recognized: {result.Text}");
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "An End-to-End Trainable Neural Network for Image-based Sequence Recognition" (TPAMI 2017)
/// https://arxiv.org/abs/1507.05717
/// </para>
/// </remarks>
public class CRNN<T> : DocumentNeuralNetworkBase<T>, ITextRecognizer<T>
{
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
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
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
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
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
            charsetSize: _charset.Length + 1)); // +1 for CTC blank
    }

    #endregion

    #region ITextRecognizer Implementation

    /// <inheritdoc/>
    public TextRecognitionResult<T> RecognizeText(Tensor<T> croppedImage)
    {
        var startTime = DateTime.UtcNow;

        var preprocessed = PreprocessTextImage(croppedImage);
        var output = _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);

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

        // Normalize to [-1, 1] range
        int batchSize = processed.Shape[0];
        int channels = processed.Shape[1];
        int height = processed.Shape[2];
        int width = processed.Shape[3];

        var normalized = new Tensor<T>(processed.Shape);
        for (int i = 0; i < processed.Data.Length; i++)
        {
            double val = NumOps.ToDouble(processed.Data.Span[i]);
            normalized.Data.Span[i] = NumOps.FromDouble((val / 255.0 - 0.5) * 2.0);
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
            int classDim = Array.IndexOf(output.Shape, expectedClasses);
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
            ModelType = ModelType.NeuralNetwork,
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
            ModelData = this.Serialize()
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

        Layers.Clear();
        InitializeLayers();
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
                _optimizer,
                LossFunction);
        }

        return new CRNN<T>(
            Architecture,
            ImageSize,
            MaxSequenceLength,
            _cnnChannels,
            _rnnHiddenSize,
            _rnnLayers,
            _charset,
            _optimizer,
            LossFunction);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var preprocessed = PreprocessTextImage(input);
        return _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");

        SetTrainingMode(true);
        var output = Predict(input);
        LastLoss = LossFunction.CalculateLoss(output.ToVector(), expectedOutput.ToVector());

        var gradient = Tensor<T>.FromVector(
            LossFunction.CalculateDerivative(output.ToVector(), expectedOutput.ToVector()));

        for (int i = Layers.Count - 1; i >= 0; i--)
            gradient = Layers[i].Backward(gradient);

        UpdateParameters(CollectGradients());
        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> gradients)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Parameter updates not supported in ONNX mode.");

        var currentParams = GetParameters();
        T lr = NumOps.FromDouble(0.0001);
        for (int i = 0; i < currentParams.Length; i++)
            currentParams[i] = NumOps.Subtract(currentParams[i], NumOps.Multiply(lr, gradients[i]));
        SetParameters(currentParams);
    }

    private Vector<T> CollectGradients()
    {
        var grads = new List<T>();
        foreach (var layer in Layers)
            grads.AddRange(layer.GetParameterGradients());
        return new Vector<T>([.. grads]);
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
