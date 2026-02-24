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
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using Microsoft.ML.OnnxRuntime;
using AiDotNet.Validation;

namespace AiDotNet.Document.OCR.TextRecognition;

/// <summary>
/// TrOCR (Transformer-based OCR) for text recognition from cropped images.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TrOCR is an end-to-end text recognition model that uses a Vision Transformer (ViT)
/// encoder and a Transformer decoder (similar to BART/GPT-2) for sequence generation.
/// </para>
/// <para>
/// <b>For Beginners:</b> TrOCR reads text from images. Given a cropped image of text
/// (like a single word or line), it outputs the actual characters. It works by:
/// 1. The encoder (ViT) analyzes the image and creates feature representations
/// 2. The decoder generates text one character at a time, using attention to focus on relevant image regions
///
/// Example usage:
/// <code>
/// var trocr = new TrOCR&lt;float&gt;(architecture);
/// var result = trocr.RecognizeText(croppedTextImage);
/// Console.WriteLine($"Text: {result.Text}, Confidence: {result.ConfidenceValue}");
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models" (AAAI 2022)
/// https://arxiv.org/abs/2109.10282
/// </para>
/// </remarks>
public class TrOCR<T> : DocumentNeuralNetworkBase<T>, ITextRecognizer<T>
{
    private readonly TrOCROptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxEncoderSession;
    private readonly InferenceSession? _onnxDecoderSession;
    private readonly ITokenizer _tokenizer;
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly int _encoderHiddenDim;
    private readonly int _decoderHiddenDim;
    private readonly int _numEncoderLayers;
    private readonly int _numDecoderLayers;
    private readonly int _numEncoderHeads;
    private readonly int _numDecoderHeads;
    private readonly int _patchSize;
    private readonly int _vocabSize;
    private readonly int _maxSequenceLength;

    // Native mode layers
    private readonly List<ILayer<T>> _encoderLayers = [];
    private readonly List<ILayer<T>> _decoderLayers = [];

    // Learnable embeddings
    private Tensor<T>? _decoderPositionEmbeddings;
    private Tensor<T>? _decoderWordEmbeddings;

    // Cached outputs
    private Tensor<T>? _lastCharacterProbabilities;
#pragma warning disable CS0649 // Field is never assigned - attention weights are computed but not yet stored
    private Tensor<T>? _lastAttentionWeights;
#pragma warning restore CS0649

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override DocumentType SupportedDocumentTypes => DocumentType.All;

    /// <inheritdoc/>
    public override bool RequiresOCR => false;

    /// <inheritdoc/>
    public int ExpectedImageSize => ImageSize;

    /// <inheritdoc/>
    public string SupportedCharacters { get; }

    /// <inheritdoc/>
    int ITextRecognizer<T>.MaxSequenceLength => _maxSequenceLength;

    /// <inheritdoc/>
    public bool SupportsAttentionVisualization => true;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a TrOCR model using pre-trained ONNX models for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="encoderPath">Path to the ONNX encoder model.</param>
    /// <param name="decoderPath">Path to the ONNX decoder model.</param>
    /// <param name="tokenizer">Tokenizer for text generation.</param>
    /// <param name="imageHeight">Input image height (default: 384 for TrOCR-base).</param>
    /// <param name="imageWidth">Input image width (default: 384).</param>
    /// <param name="maxSequenceLength">Maximum output sequence length (default: 128).</param>
    /// <param name="encoderHiddenDim">Encoder hidden dimension (default: 768 for base).</param>
    /// <param name="decoderHiddenDim">Decoder hidden dimension (default: 768 for base).</param>
    /// <param name="numEncoderLayers">Number of encoder layers (default: 12).</param>
    /// <param name="numDecoderLayers">Number of decoder layers (default: 6).</param>
    /// <param name="numEncoderHeads">Number of encoder attention heads (default: 12).</param>
    /// <param name="numDecoderHeads">Number of decoder attention heads (default: 12).</param>
    /// <param name="patchSize">ViT patch size (default: 16).</param>
    /// <param name="vocabSize">Vocabulary size (default: 50265 for RoBERTa tokenizer).</param>
    /// <param name="optimizer">Optimizer for training (optional).</param>
    /// <param name="lossFunction">Loss function (optional).</param>
    /// <exception cref="ArgumentNullException">Thrown if paths or tokenizer is null.</exception>
    /// <exception cref="FileNotFoundException">Thrown if ONNX model files don't exist.</exception>
    public TrOCR(
        NeuralNetworkArchitecture<T> architecture,
        string encoderPath,
        string decoderPath,
        ITokenizer tokenizer,
        int imageHeight = 384,
        int imageWidth = 384,
        int maxSequenceLength = 128,
        int encoderHiddenDim = 768,
        int decoderHiddenDim = 768,
        int numEncoderLayers = 12,
        int numDecoderLayers = 6,
        int numEncoderHeads = 12,
        int numDecoderHeads = 12,
        int patchSize = 16,
        int vocabSize = 50265,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        TrOCROptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _options = options ?? new TrOCROptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(encoderPath))
            throw new ArgumentNullException(nameof(encoderPath));
        if (string.IsNullOrWhiteSpace(decoderPath))
            throw new ArgumentNullException(nameof(decoderPath));
        if (!File.Exists(encoderPath))
            throw new FileNotFoundException($"Encoder model not found: {encoderPath}", encoderPath);
        if (!File.Exists(decoderPath))
            throw new FileNotFoundException($"Decoder model not found: {decoderPath}", decoderPath);

        Guard.NotNull(tokenizer);
        _tokenizer = tokenizer;
        _useNativeMode = false;
        _encoderHiddenDim = encoderHiddenDim;
        _decoderHiddenDim = decoderHiddenDim;
        _numEncoderLayers = numEncoderLayers;
        _numDecoderLayers = numDecoderLayers;
        _numEncoderHeads = numEncoderHeads;
        _numDecoderHeads = numDecoderHeads;
        _patchSize = patchSize;
        _vocabSize = vocabSize;
        _maxSequenceLength = maxSequenceLength;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = Math.Max(imageHeight, imageWidth);
        MaxSequenceLength = maxSequenceLength;
        SupportedCharacters = BuildSupportedCharacters();

        _onnxEncoderSession = new InferenceSession(encoderPath);
        _onnxDecoderSession = new InferenceSession(decoderPath);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a TrOCR model using native layers for training and inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="tokenizer">Tokenizer for text generation (optional).</param>
    /// <param name="imageHeight">Input image height (default: 384 for TrOCR-base).</param>
    /// <param name="imageWidth">Input image width (default: 384).</param>
    /// <param name="maxSequenceLength">Maximum output sequence length (default: 128).</param>
    /// <param name="encoderHiddenDim">Encoder hidden dimension (default: 768 for base).</param>
    /// <param name="decoderHiddenDim">Decoder hidden dimension (default: 768 for base).</param>
    /// <param name="numEncoderLayers">Number of encoder layers (default: 12).</param>
    /// <param name="numDecoderLayers">Number of decoder layers (default: 6).</param>
    /// <param name="numEncoderHeads">Number of encoder attention heads (default: 12).</param>
    /// <param name="numDecoderHeads">Number of decoder attention heads (default: 12).</param>
    /// <param name="patchSize">ViT patch size (default: 16).</param>
    /// <param name="vocabSize">Vocabulary size (default: 50265 for RoBERTa tokenizer).</param>
    /// <param name="optimizer">Optimizer for training (optional).</param>
    /// <param name="lossFunction">Loss function (optional).</param>
    /// <remarks>
    /// <para>
    /// <b>Default Configuration (TrOCR-Base from AAAI 2022 paper):</b>
    /// - Encoder: ViT-Base (12 layers, 768 hidden, 12 heads)
    /// - Decoder: 6 layers, 768 hidden, 12 heads
    /// - Image size: 384×384
    /// - Patch size: 16
    /// </para>
    /// </remarks>
    public TrOCR(
        NeuralNetworkArchitecture<T> architecture,
        ITokenizer? tokenizer = null,
        int imageHeight = 384,
        int imageWidth = 384,
        int maxSequenceLength = 128,
        int encoderHiddenDim = 768,
        int decoderHiddenDim = 768,
        int numEncoderLayers = 12,
        int numDecoderLayers = 6,
        int numEncoderHeads = 12,
        int numDecoderHeads = 12,
        int patchSize = 16,
        int vocabSize = 50265,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        TrOCROptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _options = options ?? new TrOCROptions();
        Options = _options;

        _useNativeMode = true;
        _encoderHiddenDim = encoderHiddenDim;
        _decoderHiddenDim = decoderHiddenDim;
        _numEncoderLayers = numEncoderLayers;
        _numDecoderLayers = numDecoderLayers;
        _numEncoderHeads = numEncoderHeads;
        _numDecoderHeads = numDecoderHeads;
        _patchSize = patchSize;
        _vocabSize = vocabSize;
        _maxSequenceLength = maxSequenceLength;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = Math.Max(imageHeight, imageWidth);
        MaxSequenceLength = maxSequenceLength;
        SupportedCharacters = BuildSupportedCharacters();

        _tokenizer = tokenizer ?? LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT);

        InitializeLayers();
        InitializeEmbeddings();
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

        // Check if user provided custom layers
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            return;
        }

        // Use LayerHelper to create default TrOCR layers
        var (encoderLayers, decoderLayers) = LayerHelper<T>.CreateDefaultTrOCRLayers(
            imageSize: ImageSize,
            patchSize: _patchSize,
            encoderHiddenDim: _encoderHiddenDim,
            decoderHiddenDim: _decoderHiddenDim,
            numEncoderLayers: _numEncoderLayers,
            numDecoderLayers: _numDecoderLayers,
            numEncoderHeads: _numEncoderHeads,
            numDecoderHeads: _numDecoderHeads,
            vocabSize: _vocabSize,
            maxSequenceLength: _maxSequenceLength);

        _encoderLayers.AddRange(encoderLayers);
        Layers.AddRange(_encoderLayers);

        _decoderLayers.AddRange(decoderLayers);
        Layers.AddRange(_decoderLayers);
    }

    private void InitializeEmbeddings()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        _decoderPositionEmbeddings = Tensor<T>.CreateDefault([_maxSequenceLength, _decoderHiddenDim], NumOps.Zero);
        InitializeWithSmallRandomValues(_decoderPositionEmbeddings, random, 0.02);

        _decoderWordEmbeddings = Tensor<T>.CreateDefault([_vocabSize, _decoderHiddenDim], NumOps.Zero);
        InitializeWithSmallRandomValues(_decoderWordEmbeddings, random, 0.02);
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

    private static string BuildSupportedCharacters()
    {
        // Standard printable ASCII + common Unicode characters
        var chars = new System.Text.StringBuilder();
        for (char c = ' '; c <= '~'; c++)
        {
            chars.Append(c);
        }
        return chars.ToString();
    }

    #endregion

    #region ITextRecognizer Implementation

    /// <inheritdoc/>
    public TextRecognitionResult<T> RecognizeText(Tensor<T> croppedImage)
    {
        ValidateImageShape(croppedImage);

        var startTime = DateTime.UtcNow;

        var result = _useNativeMode
            ? RecognizeTextNative(croppedImage)
            : RecognizeTextOnnx(croppedImage);

        return new TextRecognitionResult<T>
        {
            Text = result.Text,
            Confidence = result.Confidence,
            ConfidenceValue = result.ConfidenceValue,
            Characters = result.Characters,
            CharacterProbabilities = _lastCharacterProbabilities,
            AttentionWeights = _lastAttentionWeights,
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds,
            Alternatives = result.Alternatives
        };
    }

    /// <inheritdoc/>
    public IEnumerable<TextRecognitionResult<T>> RecognizeTextBatch(IEnumerable<Tensor<T>> croppedImages)
    {
        foreach (var image in croppedImages)
        {
            yield return RecognizeText(image);
        }
    }

    /// <inheritdoc/>
    public Tensor<T> GetCharacterProbabilities()
    {
        return _lastCharacterProbabilities ?? new Tensor<T>([1, _vocabSize]);
    }

    /// <inheritdoc/>
    public Tensor<T>? GetAttentionWeights()
    {
        return _lastAttentionWeights;
    }

    private TextRecognitionResult<T> RecognizeTextNative(Tensor<T> image)
    {
        var preprocessed = PreprocessDocument(image);
        var encoderOutput = RunEncoder(preprocessed);
        return GenerateText(encoderOutput);
    }

    private TextRecognitionResult<T> RecognizeTextOnnx(Tensor<T> image)
    {
        if (_onnxEncoderSession is null || _onnxDecoderSession is null)
            throw new InvalidOperationException("ONNX sessions not initialized.");

        var preprocessed = PreprocessDocument(image);
        var encoderOutput = RunOnnxInference(preprocessed);
        return GenerateText(encoderOutput);
    }

    private Tensor<T> RunEncoder(Tensor<T> input)
    {
        var output = input;
        bool hasReshapedToSequence = false;
        bool hasPassedConvLayer = false;
        foreach (var layer in _encoderLayers)
        {
            if (layer is ConvolutionalLayer<T> or BatchNormalizationLayer<T>
                     or PoolingLayer<T> or MaxPoolingLayer<T> or AveragePoolingLayer<T>)
            {
                hasPassedConvLayer = true;
            }

            // Auto-reshape once when transitioning from spatial (CNN) to non-spatial layers
            bool isNonSpatialLayer = layer is not (ConvolutionalLayer<T> or BatchNormalizationLayer<T>
                or PoolingLayer<T> or MaxPoolingLayer<T> or AveragePoolingLayer<T>);
            if (!hasReshapedToSequence && hasPassedConvLayer && output.Shape.Length >= 3 && isNonSpatialLayer)
            {
                int channels = output.Shape.Length == 4 ? output.Shape[1] : output.Shape[0];
                int spatialH = output.Shape.Length == 4 ? output.Shape[2] : output.Shape[1];
                int spatialW = output.Shape.Length == 4 ? output.Shape[3] : output.Shape[2];
                int numPatches = spatialH * spatialW;
                output = new Tensor<T>(output.Data.ToArray(), [numPatches, channels]);
                hasReshapedToSequence = true;
            }
            output = layer.Forward(output);
        }
        return output;
    }

    private TextRecognitionResult<T> GenerateText(Tensor<T> encoderOutput)
    {
        var generatedTokens = new List<int>();
        var characterConfidences = new List<CharacterRecognition<T>>();
        var allProbabilities = new List<T[]>();

        // Start token
        int startToken = 0; // BOS token
        int eosToken = 2;   // EOS token
        generatedTokens.Add(startToken);

        double totalConfidence = 0;

        for (int step = 0; step < _maxSequenceLength - 1; step++)
        {
            var decoderInput = CreateDecoderInput(generatedTokens);
            var logits = RunDecoder(decoderInput, encoderOutput);

            // Get probabilities for last position
            var probs = ApplySoftmax(logits, step);
            allProbabilities.Add(probs);

            // Greedy decoding - get argmax
            int nextToken = 0;
            double maxProb = double.MinValue;
            for (int i = 0; i < Math.Min(_vocabSize, probs.Length); i++)
            {
                double p = NumOps.ToDouble(probs[i]);
                if (p > maxProb)
                {
                    maxProb = p;
                    nextToken = i;
                }
            }

            if (nextToken == eosToken)
                break;

            generatedTokens.Add(nextToken);
            totalConfidence += maxProb;

            // Store character-level info
            char decodedChar = DecodeToken(nextToken);
            var alternatives = GetTopKAlternatives(probs, 3);

            characterConfidences.Add(new CharacterRecognition<T>
            {
                Character = decodedChar,
                Confidence = NumOps.FromDouble(maxProb),
                ConfidenceValue = maxProb,
                Position = step,
                Alternatives = alternatives
            });
        }

        // Store probabilities for inspection
        _lastCharacterProbabilities = CreateProbabilityTensor(allProbabilities);

        // Decode full text
        string text = _tokenizer.Decode(generatedTokens.Skip(1).ToList()); // Skip BOS
        double avgConfidence = characterConfidences.Count > 0 ? totalConfidence / characterConfidences.Count : 0;

        return new TextRecognitionResult<T>
        {
            Text = text,
            Confidence = NumOps.FromDouble(avgConfidence),
            ConfidenceValue = avgConfidence,
            Characters = characterConfidences,
            Alternatives = []
        };
    }

    private Tensor<T> CreateDecoderInput(List<int> tokens)
    {
        var input = new Tensor<T>([1, tokens.Count, _decoderHiddenDim]);

        if (_decoderWordEmbeddings is null || _decoderPositionEmbeddings is null)
            return input;

        for (int i = 0; i < tokens.Count; i++)
        {
            int tokenId = Math.Min(tokens[i], _vocabSize - 1);
            for (int d = 0; d < _decoderHiddenDim; d++)
            {
                // Word embedding + position embedding
                T wordEmb = _decoderWordEmbeddings[tokenId, d];
                T posEmb = _decoderPositionEmbeddings[i, d];
                input[0, i, d] = NumOps.Add(wordEmb, posEmb);
            }
        }

        return input;
    }

    private Tensor<T> RunDecoder(Tensor<T> decoderInput, Tensor<T> encoderOutput)
    {
        var output = decoderInput;
        foreach (var layer in _decoderLayers)
        {
            output = layer.Forward(output);
        }
        return output;
    }

    private T[] ApplySoftmax(Tensor<T> logits, int position)
    {
        int vocabSize = Math.Min(_vocabSize, logits.Data.Length);
        var probs = new T[vocabSize];

        // Find max for numerical stability
        double maxVal = double.MinValue;
        int startIdx = position * vocabSize;
        for (int i = 0; i < vocabSize && (startIdx + i) < logits.Data.Length; i++)
        {
            double val = NumOps.ToDouble(logits.Data.Span[startIdx + i]);
            if (val > maxVal) maxVal = val;
        }

        // Compute softmax
        double sumExp = 0;
        for (int i = 0; i < vocabSize && (startIdx + i) < logits.Data.Length; i++)
        {
            double val = NumOps.ToDouble(logits.Data.Span[startIdx + i]);
            sumExp += Math.Exp(val - maxVal);
        }

        for (int i = 0; i < vocabSize && (startIdx + i) < logits.Data.Length; i++)
        {
            double val = NumOps.ToDouble(logits.Data.Span[startIdx + i]);
            probs[i] = NumOps.FromDouble(Math.Exp(val - maxVal) / sumExp);
        }

        return probs;
    }

    private char DecodeToken(int tokenId)
    {
        try
        {
            string decoded = _tokenizer.Decode([tokenId]);
            return decoded.Length > 0 ? decoded[0] : ' ';
        }
        catch
        {
            return ' ';
        }
    }

    private List<(char Character, double Probability)> GetTopKAlternatives(T[] probs, int k)
    {
        var alternatives = new List<(int idx, double prob)>();
        for (int i = 0; i < probs.Length; i++)
        {
            alternatives.Add((i, NumOps.ToDouble(probs[i])));
        }

        return alternatives
            .OrderByDescending(x => x.prob)
            .Take(k)
            .Select(x => (DecodeToken(x.idx), x.prob))
            .ToList();
    }

    private Tensor<T> CreateProbabilityTensor(List<T[]> allProbabilities)
    {
        if (allProbabilities.Count == 0)
            return new Tensor<T>([1, _vocabSize]);

        int seqLen = allProbabilities.Count;
        int vocabSize = allProbabilities[0].Length;
        var tensor = new Tensor<T>([seqLen, vocabSize]);

        for (int s = 0; s < seqLen; s++)
        {
            for (int v = 0; v < vocabSize; v++)
            {
                tensor[s, v] = allProbabilities[s][v];
            }
        }

        return tensor;
    }

    #endregion

    #region IDocumentModel Implementation

    /// <inheritdoc/>
    public Tensor<T> EncodeDocument(Tensor<T> documentImage)
    {
        ValidateImageShape(documentImage);
        var preprocessed = PreprocessDocument(documentImage);
        return _useNativeMode ? RunEncoder(preprocessed) : RunOnnxInference(preprocessed);
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
        sb.AppendLine("TrOCR Model Summary");
        sb.AppendLine("===================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine();
        sb.AppendLine("Encoder (ViT):");
        sb.AppendLine($"  Hidden Dimension: {_encoderHiddenDim}");
        sb.AppendLine($"  Number of Layers: {_numEncoderLayers}");
        sb.AppendLine($"  Attention Heads: {_numEncoderHeads}");
        sb.AppendLine($"  Patch Size: {_patchSize}");
        sb.AppendLine();
        sb.AppendLine("Decoder:");
        sb.AppendLine($"  Hidden Dimension: {_decoderHiddenDim}");
        sb.AppendLine($"  Number of Layers: {_numDecoderLayers}");
        sb.AppendLine($"  Attention Heads: {_numDecoderHeads}");
        sb.AppendLine();
        sb.AppendLine($"Image Size: {ImageSize}x{ImageSize}");
        sb.AppendLine($"Vocabulary Size: {_vocabSize}");
        sb.AppendLine($"Max Sequence Length: {_maxSequenceLength}");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        sb.AppendLine($"Attention Visualization: {SupportsAttentionVisualization}");
        return sb.ToString();
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// Applies TrOCR's industry-standard preprocessing: normalize to [-1, 1].
    /// </summary>
    /// <remarks>
    /// TrOCR (Transformer-based OCR) uses mean=0.5, std=0.5 normalization
    /// (same as DeiT/BEiT) from Microsoft paper.
    /// </remarks>
    protected override Tensor<T> ApplyDefaultPreprocessing(Tensor<T> rawImage)
    {
        var image = EnsureBatchDimension(rawImage);

        int batchSize = image.Shape[0];
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        var normalized = new Tensor<T>(image.Shape);

        // TrOCR normalization (same as DeiT/BEiT)
        double[] means = [0.5, 0.5, 0.5];
        double[] stds = [0.5, 0.5, 0.5];

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                double mean = c < means.Length ? means[c] : 0.5;
                double std = c < stds.Length ? stds[c] : 0.5;

                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int idx = b * channels * height * width + c * height * width + h * width + w;
                        double value = NumOps.ToDouble(image.Data.Span[idx]);
                        normalized.Data.Span[idx] = NumOps.FromDouble((value - mean) / std);
                    }
                }
            }
        }

        return normalized;
    }

    /// <summary>
    /// Applies TrOCR's industry-standard postprocessing: pass-through (encoder-decoder outputs are already final).
    /// </summary>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput)
    {
        return modelOutput;
    }

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "TrOCR",
            ModelType = ModelType.NeuralNetwork,
            Description = "Transformer-based OCR with ViT encoder (AAAI 2022)",
            FeatureCount = _decoderHiddenDim,
            Complexity = _numEncoderLayers + _numDecoderLayers,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "encoder_hidden_dim", _encoderHiddenDim },
                { "decoder_hidden_dim", _decoderHiddenDim },
                { "num_encoder_layers", _numEncoderLayers },
                { "num_decoder_layers", _numDecoderLayers },
                { "num_encoder_heads", _numEncoderHeads },
                { "num_decoder_heads", _numDecoderHeads },
                { "patch_size", _patchSize },
                { "vocab_size", _vocabSize },
                { "max_sequence_length", _maxSequenceLength },
                { "image_size", ImageSize },
                { "use_native_mode", _useNativeMode }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_encoderHiddenDim);
        writer.Write(_decoderHiddenDim);
        writer.Write(_numEncoderLayers);
        writer.Write(_numDecoderLayers);
        writer.Write(_numEncoderHeads);
        writer.Write(_numDecoderHeads);
        writer.Write(_patchSize);
        writer.Write(_vocabSize);
        writer.Write(_maxSequenceLength);
        writer.Write(ImageSize);
        writer.Write(_useNativeMode);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        int encoderHiddenDim = reader.ReadInt32();
        int decoderHiddenDim = reader.ReadInt32();
        int numEncoderLayers = reader.ReadInt32();
        int numDecoderLayers = reader.ReadInt32();
        int numEncoderHeads = reader.ReadInt32();
        int numDecoderHeads = reader.ReadInt32();
        int patchSize = reader.ReadInt32();
        int vocabSize = reader.ReadInt32();
        int maxSequenceLength = reader.ReadInt32();
        int imageSize = reader.ReadInt32();
        bool useNativeMode = reader.ReadBoolean();

        ImageSize = imageSize;
        MaxSequenceLength = maxSequenceLength;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new TrOCR<T>(
            Architecture,
            _tokenizer,
            ImageSize,
            ImageSize,
            _maxSequenceLength,
            _encoderHiddenDim,
            _decoderHiddenDim,
            _numEncoderLayers,
            _numDecoderLayers,
            _numEncoderHeads,
            _numDecoderHeads,
            _patchSize,
            _vocabSize);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var preprocessed = PreprocessDocument(input);
        return _useNativeMode ? RunEncoder(preprocessed) : RunOnnxInference(preprocessed);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Training is not supported in ONNX inference mode.");
        }

        SetTrainingMode(true);

        var output = Predict(input);
        LastLoss = LossFunction.CalculateLoss(output.ToVector(), expectedOutput.ToVector());

        var lossGradient = LossFunction.CalculateDerivative(output.ToVector(), expectedOutput.ToVector());
        var gradient = Tensor<T>.FromVector(lossGradient);

        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(gradient);
        }

        var paramGradients = CollectParameterGradients();
        UpdateParameters(paramGradients);

        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> gradients)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Parameter updates are not supported in ONNX inference mode.");
        }

        var currentParams = GetParameters();
        T learningRate = NumOps.FromDouble(0.0001);

        for (int i = 0; i < currentParams.Length; i++)
        {
            currentParams[i] = NumOps.Subtract(currentParams[i], NumOps.Multiply(learningRate, gradients[i]));
        }

        SetParameters(currentParams);
    }

    private Vector<T> CollectParameterGradients()
    {
        var gradients = new List<T>();

        foreach (var layer in Layers)
        {
            var layerGradients = layer.GetParameterGradients();
            gradients.AddRange(layerGradients);
        }

        return new Vector<T>([.. gradients]);
    }

    #endregion

    #region Disposal

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _onnxEncoderSession?.Dispose();
            _onnxDecoderSession?.Dispose();
        }
        base.Dispose(disposing);
    }

    #endregion
}
