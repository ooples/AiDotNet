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

    /// <summary>
    /// Seed for weight initialization, so two models built from equal options start from equal weights.
    /// Default: 42, matching <see cref="AiDotNet.Models.Options.ObjectDetectionOptions{T}.Seed"/>.
    /// Null leaves initialization unseeded.
    /// </summary>
    public int? RandomSeed { get; set; } = 42;
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
public abstract class OCRBase<T> : VisionTaskModelBase<T>, AiDotNet.Interfaces.IRecognitionTrainingModel<T>
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
        // Arm the per-layer initialization seed scope before the derived constructor builds any layer
        // (the same root-model contract DiffusionModelBase follows). Every layer, the BackboneLayerShims
        // adapters' inner layers, and the necks and query tables that draw from the scope then take a
        // deterministic seed, so two models built from equal options start from equal weights (#2201).
        // A null seed leaves the scope unarmed and initialization stays unseeded.
        AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.ResetForModelConstruction(options.RandomSeed);
        // No backbone offer: a recognizer builds no DetectionBackboneArchitecture, so an offer made here was never
        // consumed and the next standalone backbone on this thread inherited this model's seed. A detector built inside
        // a reader (DocumentReader, SceneTextReader) makes its own offer from its own base.

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
    public (string text, T confidence) RecognizeText(Tensor<T> croppedImage)
    {
        if (croppedImage is null) throw new ArgumentNullException(nameof(croppedImage));
        // The same preparation training applies (ForwardLogits, teacher forcing). The end-to-end readers pass raw
        // crops here, and a recogniser that skipped it read images that training had never seen.
        return RecognizePreparedText(PreprocessCrop(croppedImage));
    }

    /// <summary>Recognises one crop that <see cref="PreprocessCrop"/> has already prepared.</summary>
    /// <param name="preparedCrop">The resized, normalised crop.</param>
    protected abstract (string text, T confidence) RecognizePreparedText(Tensor<T> preparedCrop);

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

        // The budget is emitted characters, not timesteps: CTC blanks and repeats consume
        // sequence positions without consuming the caller's character budget.
        for (int t = 0; t < seqLen && result.Count < Options.MaxSequenceLength; t++)
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

        for (int t = 0; t < seqLen && result.Count < Options.MaxSequenceLength; t++)
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
    /// One training step on text-line images and their transcriptions, using the model's own recognition
    /// objective (CTC for CRNN, teacher-forced cross-entropy for TrOCR).
    /// </summary>
    /// <param name="images">The text-line images, <c>[batch, channels, height, width]</c> or one <c>[channels, height, width]</c>.</param>
    /// <param name="transcriptions">One transcription per image. Every character must be in the character set.</param>
    public void TrainRecognition(Tensor<T> images, IReadOnlyList<string> transcriptions)
    {
        if (images is null) throw new ArgumentNullException(nameof(images));
        if (transcriptions is null) throw new ArgumentNullException(nameof(transcriptions));
        int batch = images.Rank == 4 ? images.Shape[0] : 1;
        if (transcriptions.Count != batch)
            throw new ArgumentException($"{transcriptions.Count} transcriptions for a batch of {batch} images.", nameof(transcriptions));
        Train(images, EncodeTranscriptions(transcriptions));
    }

    /// <summary>
    /// Transcriptions as label ids <c>[batch, longest]</c> through the character set. Trailing zeros (the
    /// blank / padding id) pad the shorter rows, which both recognition objectives ignore.
    /// </summary>
    /// <exception cref="ArgumentException">A transcription contains a character outside the character set.</exception>
    protected Tensor<T> EncodeTranscriptions(IReadOnlyList<string> transcriptions)
    {
        if (transcriptions is null) throw new ArgumentNullException(nameof(transcriptions));
        int width = Math.Max(1, transcriptions.Max(text => (text ?? throw new ArgumentException("Transcriptions cannot be null.", nameof(transcriptions))).Length));
        var labels = new Tensor<T>(new[] { transcriptions.Count, width });
        for (int b = 0; b < transcriptions.Count; b++)
            for (int i = 0; i < transcriptions[b].Length; i++)
            {
                char ch = transcriptions[b][i];
                if (!CharToIndex.TryGetValue(ch, out int id))
                    throw new ArgumentException($"Character '{ch}' of transcription {b} is not in the character set.", nameof(transcriptions));
                labels[b, i] = NumOps.FromDouble(id);
            }
        return labels;
    }
    /// <inheritdoc />
    /// <remarks>Recognisers regress their raw logits, not the decoded prediction.</remarks>
    protected override Tensor<T> TrainingForward(Tensor<T> input) => ForwardLogits(input);

    /// <inheritdoc />
    protected override int[] DeferredParameterProbeShape =>
        new[] { 1, InputChannels, Options.RecognitionHeight, Options.MaxRecognitionWidth };

    #endregion

}
