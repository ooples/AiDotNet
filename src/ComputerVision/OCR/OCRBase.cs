using AiDotNet.Augmentation.Image;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.OCR;

/// <summary>
/// Result of OCR processing.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class OCRResult<T>
{
    /// <summary>
    /// List of recognized text regions.
    /// </summary>
    public List<RecognizedText<T>> TextRegions { get; set; } = new();

    /// <summary>
    /// Full text concatenated from all regions.
    /// </summary>
    public string FullText { get; set; } = string.Empty;

    /// <summary>
    /// Time taken for inference.
    /// </summary>
    public TimeSpan InferenceTime { get; set; }

    /// <summary>
    /// Width of the input image.
    /// </summary>
    public int ImageWidth { get; set; }

    /// <summary>
    /// Height of the input image.
    /// </summary>
    public int ImageHeight { get; set; }
}

/// <summary>
/// Represents recognized text in an image region.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class RecognizedText<T>
{
    /// <summary>
    /// The recognized text string.
    /// </summary>
    public string Text { get; set; } = string.Empty;

    /// <summary>
    /// Bounding box around the text.
    /// </summary>
    public BoundingBox<T>? Box { get; set; }

    /// <summary>
    /// Polygon points for rotated/curved text.
    /// </summary>
    public List<(T X, T Y)> Polygon { get; set; } = new();

    /// <summary>
    /// Overall confidence of the recognition.
    /// </summary>
    public T Confidence { get; set; }

    /// <summary>
    /// Per-character confidences (if available).
    /// </summary>
    public List<T> CharacterConfidences { get; set; } = new();

    /// <summary>
    /// Language detected for this text.
    /// </summary>
    public string? Language { get; set; }

    /// <summary>
    /// Creates a new recognized text.
    /// </summary>
    public RecognizedText(string text, T confidence)
    {
        Text = text;
        Confidence = confidence;
    }
}

/// <summary>
/// Options for OCR models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class OCROptions<T>
{
    private static readonly INumericOperations<T> NumOps =
        Tensors.Helpers.MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// OCR mode (scene text, document, or both).
    /// </summary>
    public OCRMode Mode { get; set; } = OCRMode.SceneText;

    /// <summary>
    /// Text detection model to use.
    /// </summary>
    public TextDetectionModel DetectionModel { get; set; } = TextDetectionModel.DBNet;

    /// <summary>
    /// Text recognition model to use.
    /// </summary>
    public TextRecognitionModel RecognitionModel { get; set; } = TextRecognitionModel.CRNN;

    /// <summary>
    /// Character set/vocabulary for recognition.
    /// </summary>
    public string? CharacterSet { get; set; }

    /// <summary>
    /// Supported languages.
    /// </summary>
    public string[] SupportedLanguages { get; set; } = new[] { "en" };

    /// <summary>
    /// Whether to detect text orientation.
    /// </summary>
    public bool DetectOrientation { get; set; } = true;

    /// <summary>
    /// Whether to correct skewed text.
    /// </summary>
    public bool CorrectSkew { get; set; } = true;

    /// <summary>
    /// Minimum confidence threshold.
    /// </summary>
    public T ConfidenceThreshold { get; set; } = NumOps.FromDouble(0.5);

    /// <summary>
    /// Whether to group text into lines/paragraphs.
    /// </summary>
    public bool GroupTextLines { get; set; } = true;

    /// <summary>
    /// Input image height for the recognition model.
    /// </summary>
    public int RecognitionHeight { get; set; } = 32;

    /// <summary>
    /// Maximum input width for the recognition model.
    /// </summary>
    public int MaxRecognitionWidth { get; set; } = 320;

    /// <summary>
    /// Maximum sequence length for recognition.
    /// </summary>
    public int MaxSequenceLength { get; set; } = 100;

    /// <summary>
    /// Whether to use pretrained weights.
    /// </summary>
    public bool UsePretrained { get; set; } = true;
}

/// <summary>
/// OCR processing modes.
/// </summary>
public enum OCRMode
{
    /// <summary>Scene text recognition (signs, billboards, etc.).</summary>
    SceneText,
    /// <summary>Document text recognition (scanned documents, etc.).</summary>
    Document,
    /// <summary>Both scene and document text.</summary>
    Both
}

/// <summary>
/// Text detection model types.
/// </summary>
public enum TextDetectionModel
{
    /// <summary>CRAFT detector.</summary>
    CRAFT,
    /// <summary>EAST detector.</summary>
    EAST,
    /// <summary>DBNet detector.</summary>
    DBNet
}

/// <summary>
/// Text recognition model types.
/// </summary>
public enum TextRecognitionModel
{
    /// <summary>CRNN (Convolutional Recurrent Neural Network).</summary>
    CRNN,
    /// <summary>TrOCR (Transformer-based OCR).</summary>
    TrOCR
}

/// <summary>
/// Base class for OCR models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class OCRBase<T> : ModelBase<T, Tensor<T>, Tensor<T>>
{
    // Engine and NumOps inherited from ModelBase
    protected readonly OCROptions<T> Options;

    /// <summary>
    /// Default character set for recognition.
    /// </summary>
    protected static readonly string DefaultCharacterSet =
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ";

    /// <summary>
    /// Character to index mapping.
    /// </summary>
    protected readonly Dictionary<char, int> CharToIndex;

    /// <summary>
    /// Gets the set of characters this model can emit.
    /// </summary>
    /// <remarks>
    /// Recognition decodes class indices into characters from this set, so a caller needs it to
    /// know what the model is capable of reading -- and every character in the recognised text
    /// must come from it.
    /// </remarks>
    public string CharacterSet => Options.CharacterSet ?? DefaultCharacterSet;

    /// <summary>
    /// Gets the maximum number of characters the decoder will emit for one text region.
    /// </summary>
    public int MaxSequenceLength => Options.MaxSequenceLength;

    /// <summary>
    /// Index to character mapping.
    /// </summary>
    protected readonly Dictionary<int, char> IndexToChar;

    /// <summary>
    /// Name of this OCR model.
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Creates a new OCR model.
    /// </summary>
    protected OCRBase(OCROptions<T> options)
    {
        Options = options;

        string charset = options.CharacterSet ?? DefaultCharacterSet;

        CharToIndex = new Dictionary<char, int>();
        IndexToChar = new Dictionary<int, char>();

        // Index 0 is reserved for blank/CTC token
        IndexToChar[0] = '\0';

        for (int i = 0; i < charset.Length; i++)
        {
            CharToIndex[charset[i]] = i + 1;
            IndexToChar[i + 1] = charset[i];
        }
    }

    /// <summary>
    /// Recognizes text in an image.
    /// </summary>
    /// <param name="image">Input image tensor [batch, channels, height, width].</param>
    /// <returns>OCR result with recognized text.</returns>
    public abstract OCRResult<T> Recognize(Tensor<T> image);

    /// <summary>
    /// Recognizes text in a cropped text region.
    /// </summary>
    /// <param name="croppedImage">Cropped text region tensor.</param>
    /// <returns>Recognized text and confidence.</returns>
    public abstract (string text, T confidence) RecognizeText(Tensor<T> croppedImage);

    /// <summary>
    /// Gets the vocabulary size (number of classes).
    /// </summary>
    public int VocabularySize => IndexToChar.Count;

    /// <summary>
    /// Preprocesses a text crop for recognition.
    /// </summary>
    protected virtual Tensor<T> PreprocessCrop(Tensor<T> crop)
    {
        var prepared = PreprocessCropCore(crop);
        NoteResolvedInput(prepared);
        return prepared;
    }

    private Tensor<T> PreprocessCropCore(Tensor<T> crop)
    {
        int targetH = Options.RecognitionHeight;
        int srcH = crop.Shape[2];
        int srcW = crop.Shape[3];

        // Maintain aspect ratio
        int targetW = (int)Math.Round((double)srcW / srcH * targetH);
        targetW = Math.Min(targetW, Options.MaxRecognitionWidth);

        // Resize
        var resized = ResizeBilinear(crop, targetH, targetW);

        // Normalize to [-1, 1]
        for (int i = 0; i < resized.Length; i++)
        {
            double val = NumOps.ToDouble(resized[i]) / 255.0;
            resized[i] = NumOps.FromDouble(val * 2.0 - 1.0);
        }

        return resized;
    }

    /// <summary>
    /// Decodes CTC output to text.
    /// </summary>
    protected string DecodeCTC(Tensor<T> logits)
    {
        int seqLen = logits.Shape[1];
        int numClasses = logits.Shape[2];

        var result = new List<char>();
        int prevIndex = 0;

        for (int t = 0; t < seqLen; t++)
        {
            // Find argmax
            int maxIdx = 0;
            double maxVal = double.NegativeInfinity;

            for (int c = 0; c < numClasses; c++)
            {
                double val = NumOps.ToDouble(logits[0, t, c]);
                if (val > maxVal)
                {
                    maxVal = val;
                    maxIdx = c;
                }
            }

            // Skip blanks and repeated characters
            if (maxIdx != 0 && maxIdx != prevIndex)
            {
                if (IndexToChar.TryGetValue(maxIdx, out char ch))
                {
                    result.Add(ch);
                }
            }

            prevIndex = maxIdx;
        }

        return new string(result.ToArray());
    }

    /// <summary>
    /// Decodes attention-based output to text.
    /// </summary>
    protected string DecodeAttention(Tensor<T> logits, int endTokenId)
    {
        int seqLen = logits.Shape[1];
        int numClasses = logits.Shape[2];

        var result = new List<char>();

        for (int t = 0; t < seqLen; t++)
        {
            int maxIdx = 0;
            double maxVal = double.NegativeInfinity;

            for (int c = 0; c < numClasses; c++)
            {
                double val = NumOps.ToDouble(logits[0, t, c]);
                if (val > maxVal)
                {
                    maxVal = val;
                    maxIdx = c;
                }
            }

            if (maxIdx == endTokenId)
                break;

            if (IndexToChar.TryGetValue(maxIdx, out char ch))
            {
                result.Add(ch);
            }
        }

        return new string(result.ToArray());
    }

    /// <summary>
    /// Computes confidence from logits.
    /// </summary>
    protected T ComputeConfidence(Tensor<T> logits, string decodedText)
    {
        if (string.IsNullOrEmpty(decodedText))
            return NumOps.FromDouble(0);

        int seqLen = logits.Shape[1];
        int numClasses = logits.Shape[2];

        double totalConf = 0;
        int count = 0;

        for (int t = 0; t < seqLen && count < decodedText.Length; t++)
        {
            // Apply softmax and get max probability
            double maxLogit = double.NegativeInfinity;
            for (int c = 0; c < numClasses; c++)
            {
                maxLogit = Math.Max(maxLogit, NumOps.ToDouble(logits[0, t, c]));
            }

            double sumExp = 0;
            for (int c = 0; c < numClasses; c++)
            {
                sumExp += Math.Exp(NumOps.ToDouble(logits[0, t, c]) - maxLogit);
            }

            // Get probability of predicted character
            int maxIdx = 0;
            double maxVal = double.NegativeInfinity;
            for (int c = 0; c < numClasses; c++)
            {
                double val = NumOps.ToDouble(logits[0, t, c]);
                if (val > maxVal)
                {
                    maxVal = val;
                    maxIdx = c;
                }
            }

            if (maxIdx != 0) // Not blank
            {
                double prob = Math.Exp(maxVal - maxLogit) / sumExp;
                totalConf += prob;
                count++;
            }
        }

        return NumOps.FromDouble(count > 0 ? totalConf / count : 0);
    }

    /// <summary>
    /// Resizes tensor using bilinear interpolation.
    /// </summary>
    protected Tensor<T> ResizeBilinear(Tensor<T> input, int targetH, int targetW)
    {
        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int srcH = input.Shape[2];
        int srcW = input.Shape[3];

        var output = new Tensor<T>(new[] { batch, channels, targetH, targetW });

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < targetH; h++)
                {
                    for (int w = 0; w < targetW; w++)
                    {
                        double srcY = (double)h / targetH * srcH;
                        double srcX = (double)w / targetW * srcW;

                        int y0 = (int)Math.Floor(srcY);
                        int x0 = (int)Math.Floor(srcX);
                        int y1 = Math.Min(y0 + 1, srcH - 1);
                        int x1 = Math.Min(x0 + 1, srcW - 1);

                        double wy1 = srcY - y0;
                        double wy0 = 1.0 - wy1;
                        double wx1 = srcX - x0;
                        double wx0 = 1.0 - wx1;

                        double v00 = NumOps.ToDouble(input[b, c, y0, x0]);
                        double v01 = NumOps.ToDouble(input[b, c, y0, x1]);
                        double v10 = NumOps.ToDouble(input[b, c, y1, x0]);
                        double v11 = NumOps.ToDouble(input[b, c, y1, x1]);

                        double val = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);
                        output[b, c, h, w] = NumOps.FromDouble(val);
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Gets the total parameter count.
    /// </summary>
    public abstract long GetParameterCount();

    /// <summary>
    /// Loads pretrained weights.
    /// </summary>
    public abstract Task LoadWeightsAsync(string pathOrUrl, CancellationToken cancellationToken = default);

    /// <summary>
    /// Saves model weights.
    /// </summary>
    public abstract void SaveWeights(string path);

    #region ModelBase Overrides

    /// <summary>
    /// Runs OCR and returns region info as a tensor [numRegions, 6].
    /// Columns: confidence, textLength, x1, y1, x2, y2.
    /// </summary>
    /// <summary>
    /// Returns the model's raw, differentiable recognition output (see <see cref="ForwardLogits"/>).
    /// </summary>
    /// <remarks>
    /// Use <see cref="Recognize"/> to read text. <see cref="Predict"/> used to run
    /// <see cref="Recognize"/> and pack its decoded regions into a <c>[regions, 6]</c> tensor of
    /// confidence, text length and box - a decoded summary that has no gradient, so nothing trained
    /// against it could ever learn. It now returns the network output that
    /// <see cref="Train"/> fits, matching the detection bases.
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        NoteResolvedInput(input);
        return ForwardLogits(input);
    }

    /// <summary>
    /// Runs the differentiable recognition forward pass on an image and returns its raw output:
    /// per-timestep character logits for a CTC recognizer, the encoder output and first decoding step
    /// for an encoder-decoder recognizer. Every trainable weight must be reachable from it.
    /// </summary>
    /// <param name="image">The image or cropped text line, NCHW.</param>
    /// <returns>The raw recognition output that <see cref="Train"/> fits.</returns>
    protected abstract Tensor<T> ForwardLogits(Tensor<T> image);

    /// <summary>
    /// Gets the step size used by <see cref="Train"/>. Override it to match a paper recipe.
    /// </summary>
    protected virtual double TrainingLearningRate => 0.001;

    /// <inheritdoc />
    /// <summary>
    /// Runs one training step against the model's raw recognition output.
    /// </summary>
    /// <param name="input">The training image.</param>
    /// <param name="expectedOutput">The desired output, shaped like <see cref="Predict"/>.</param>
    /// <remarks>
    /// This was an empty method, so CRNN and TrOCR ignored training entirely. The step records
    /// <see cref="ForwardLogits"/> on a gradient tape, takes mean squared error against
    /// <paramref name="expectedOutput"/> and updates every live trainable weight. A recognition loss
    /// (CTC, or teacher-forced cross-entropy on target text) is the right objective for a full
    /// training recipe and belongs in an override; this base step is what makes the models trainable.
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (input is null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (expectedOutput is null)
        {
            throw new ArgumentNullException(nameof(expectedOutput));
        }

        TensorModelTrainer<T>.Step(this, input, expectedOutput, NumOps.FromDouble(TrainingLearningRate), ForwardLogits);
    }

    /// <inheritdoc />
    public override ILossFunction<T> DefaultLossFunction => new MeanSquaredErrorLoss<T>();

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        var copy = DeepCopy();
        InterfaceGuard.Parameterizable(copy).SetParameters(parameters);
        return copy;
    }

    /// <inheritdoc />
    // See the note on ObjectDetectorBase: MemberwiseClone gave a shallow copy that shared
    // weights with the original. ModelBase's rebuild-and-reload DeepCopy is correct here.

    #endregion

    /// <summary>
    /// The shape of the first input this model's forward pass ran on. Its lazily-shaped layers sized
    /// their weights from it, so replaying it on a rebuilt copy reproduces the same parameter
    /// topology. Scratch: never persisted, and rebuilt copies record their own.
    /// </summary>
    [AiDotNet.Attributes.Scratch]
    private int[]? _resolvedInputShape;

    /// <summary>Records the input shape on the first forward pass.</summary>
    private void NoteResolvedInput(Tensor<T> input)
    {
        if (_resolvedInputShape is not null || input is null)
        {
            return;
        }

        var shape = new int[input.Shape.Length];
        for (int i = 0; i < shape.Length; i++)
        {
            shape[i] = input.Shape[i];
        }

        _resolvedInputShape = shape;
    }

    /// <inheritdoc />
    /// <remarks>
    /// Runs the copy once on a zero input of the shape this model has already processed, so its
    /// lazily-shaped layers (the convolutions behind the Conv2D adapter, the backbone's lazy layers)
    /// size their weights exactly as this model's did before its state is loaded into them.
    /// </remarks>
    protected override void PrepareCopyForStateRestore(ModelBase<T, Tensor<T>, Tensor<T>> copy)
    {
        if (_resolvedInputShape is not null && copy is OCRBase<T> rebuilt)
        {
            rebuilt.Predict(new Tensor<T>(_resolvedInputShape));
        }
    }

    /// <summary>
    /// Gets the number of channels in the images this model reads.
    /// </summary>
    /// <remarks>RGB unless a model overrides it; every backbone here is built for three channels.</remarks>
    protected virtual int InputChannels => 3;

    /// <summary>
    /// Gives a model that has never run a concrete parameter topology, so its state can be captured.
    /// </summary>
    /// <remarks>
    /// Several layers size their weights on their first forward pass. Until then the model reports
    /// its parameters as shape-deferred, which is correct for a parameter query but made
    /// <see cref="Serialize"/> - and therefore <c>Clone</c> - throw on a freshly constructed model.
    /// Running the network once on a zero image of the configured recognition height and maximum width resolves exactly the
    /// shapes the first real image would, because every image is resized to that size first.
    /// </remarks>
    private void ResolveDeferredParameters()
    {
        if (_resolvedInputShape is not null)
        {
            return;
        }

        Predict(new Tensor<T>(new[] { 1, InputChannels, Options.RecognitionHeight, Options.MaxRecognitionWidth }));
    }

    /// <inheritdoc />
    /// <remarks>Resolves shape-deferred layers first; see <see cref="ResolveDeferredParameters"/>.</remarks>
    public override byte[] Serialize()
    {
        ResolveDeferredParameters();
        return base.Serialize();
    }
}
