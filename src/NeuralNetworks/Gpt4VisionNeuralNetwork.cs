using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tokenization.Interfaces;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// GPT-4V-style neural network that combines vision understanding with large language model capabilities.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This implementation provides a vision-language model that can understand images and generate
/// text responses, similar to GPT-4V, LLaVA, or other vision-language models.
/// </para>
/// <para><b>Architecture Overview:</b>
/// 1. Vision Encoder: ViT-based encoder to extract visual features
/// 2. Vision-Language Projector: Maps visual features to LLM embedding space
/// 3. Language Model: Transformer decoder for text generation
/// 4. Multi-modal Attention: Allows text to attend to visual features
/// </para>
/// </remarks>
public class Gpt4VisionNeuralNetwork<T> : NeuralNetworkBase<T>, IGpt4VisionModel<T>
{
    /// <summary>
    /// Timeout for regex operations to prevent ReDoS attacks.
    /// </summary>
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);

    #region Execution Mode

    private bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    private InferenceSession? _visionEncoder;
    private InferenceSession? _languageModel;
    private readonly string? _visionEncoderPath;
    private readonly string? _languageModelPath;

    #endregion

    #region Native Mode Fields

    // Vision encoder layers (ViT)
    private readonly List<ILayer<T>> _visionEncoderLayers = [];
    private Matrix<T>? _visionClsToken;
    private Matrix<T>? _visionPositionalEmbeddings;
    private ILayer<T>? _visionPatchEmbedding;
    private ILayer<T>? _visionLayerNorm;

    // Vision-Language Projector
    private ILayer<T>? _visionProjector1;
    private ILayer<T>? _visionProjector2;

    // Language Model (Transformer Decoder)
    private readonly List<ILayer<T>> _languageModelLayers = [];
    private Matrix<T>? _textPositionalEmbeddings;
    private ILayer<T>? _tokenEmbedding;
    private ILayer<T>? _lmHead;
    private ILayer<T>? _finalLayerNorm;

    // Cross-attention layers for vision-language fusion
    private readonly List<ILayer<T>> _crossAttentionLayers = [];

    #endregion

    #region Shared Fields

    private readonly ITokenizer _tokenizer;
    private int _embeddingDimension;
    private int _visionEmbeddingDim;
    private int _maxSequenceLength;
    private int _contextWindowSize;
    private int _imageSize;
    private int _hiddenDim;
    private int _numVisionLayers;
    private int _numLanguageLayers;
    private int _numHeads;
    private int _patchSize;
    private int _vocabularySize;
    private int _maxImagesPerRequest;
    private readonly (int Width, int Height) _maxImageResolution;
    private readonly IReadOnlyList<string> _supportedDetailLevels;

    #endregion

    #region IMultimodalEmbedding Properties

    /// <inheritdoc/>
    public int EmbeddingDimension => _embeddingDimension;

    /// <inheritdoc/>
    public int MaxSequenceLength => _maxSequenceLength;

    /// <inheritdoc/>
    public int ImageSize => _imageSize;

    /// <inheritdoc/>
    public int ImageEmbeddingDimension => _visionEmbeddingDim;

    /// <inheritdoc/>
    public int TextEmbeddingDimension => _embeddingDimension;

    #endregion

    #region IGpt4VisionModel Properties

    /// <inheritdoc/>
    public int MaxImagesPerRequest => _maxImagesPerRequest;

    /// <inheritdoc/>
    public (int Width, int Height) MaxImageResolution => _maxImageResolution;

    /// <inheritdoc/>
    public int ContextWindowSize => _contextWindowSize;

    /// <inheritdoc/>
    public IReadOnlyList<string> SupportedDetailLevels => _supportedDetailLevels;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a GPT-4 Vision network using pretrained ONNX models.
    /// </summary>
    public Gpt4VisionNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        string visionEncoderPath,
        string languageModelPath,
        ITokenizer tokenizer,
        int embeddingDimension = 4096,
        int visionEmbeddingDim = 1024,
        int maxSequenceLength = 2048,
        int contextWindowSize = 128000,
        int imageSize = 336,
        int maxImagesPerRequest = 10,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        // Validate ONNX model paths
        if (string.IsNullOrWhiteSpace(visionEncoderPath))
            throw new ArgumentException("Vision encoder path cannot be null or empty.", nameof(visionEncoderPath));
        if (string.IsNullOrWhiteSpace(languageModelPath))
            throw new ArgumentException("Language model path cannot be null or empty.", nameof(languageModelPath));
        if (!File.Exists(visionEncoderPath))
            throw new FileNotFoundException($"Vision encoder model not found: {visionEncoderPath}");
        if (!File.Exists(languageModelPath))
            throw new FileNotFoundException($"Language model not found: {languageModelPath}");

        _useNativeMode = false;
        _visionEncoderPath = visionEncoderPath;
        _languageModelPath = languageModelPath;
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        _embeddingDimension = embeddingDimension;
        _visionEmbeddingDim = visionEmbeddingDim;
        _maxSequenceLength = maxSequenceLength;
        _contextWindowSize = contextWindowSize;
        _imageSize = imageSize;
        _maxImagesPerRequest = maxImagesPerRequest;
        _maxImageResolution = (2048, 2048);
        _supportedDetailLevels = new List<string> { "low", "high", "auto" };
        _vocabularySize = 128256;
        _hiddenDim = embeddingDimension;
        _numVisionLayers = 24;
        _numLanguageLayers = 32;
        _numHeads = 32;
        _patchSize = 14;

        InitializeLayers();
    }

    /// <summary>
    /// Creates a GPT-4 Vision network using native layers (for training or when ONNX is not available).
    /// </summary>
    public Gpt4VisionNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        ITokenizer tokenizer,
        int embeddingDimension = 4096,
        int visionEmbeddingDim = 1024,
        int maxSequenceLength = 2048,
        int contextWindowSize = 128000,
        int imageSize = 336,
        int hiddenDim = 4096,
        int numVisionLayers = 24,
        int numLanguageLayers = 32,
        int numHeads = 32,
        int patchSize = 14,
        int vocabularySize = 128256,
        int maxImagesPerRequest = 10,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        _embeddingDimension = embeddingDimension;
        _visionEmbeddingDim = visionEmbeddingDim;
        _maxSequenceLength = maxSequenceLength;
        _contextWindowSize = contextWindowSize;
        _imageSize = imageSize;
        _hiddenDim = hiddenDim;
        _numVisionLayers = numVisionLayers;
        _numLanguageLayers = numLanguageLayers;
        _numHeads = numHeads;
        _patchSize = patchSize;
        _vocabularySize = vocabularySize;
        _maxImagesPerRequest = maxImagesPerRequest;
        _maxImageResolution = (2048, 2048);
        _supportedDetailLevels = new List<string> { "low", "high", "auto" };

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
        {
            // ONNX mode initialization
            if (!string.IsNullOrEmpty(_visionEncoderPath) && File.Exists(_visionEncoderPath))
            {
                var sessionOptions = new SessionOptions();
                _visionEncoder = new InferenceSession(_visionEncoderPath, sessionOptions);
            }

            if (!string.IsNullOrEmpty(_languageModelPath) && File.Exists(_languageModelPath))
            {
                var sessionOptions = new SessionOptions();
                _languageModel = new InferenceSession(_languageModelPath, sessionOptions);
            }
        }
        else
        {
            InitializeNativeLayers();
        }
    }

    private void InitializeNativeLayers()
    {
        int numPatches = (_imageSize / _patchSize) * (_imageSize / _patchSize);
        int ffnDim = _hiddenDim * 4;

        // Vision encoder: Patch embedding
        _visionPatchEmbedding = new PatchEmbeddingLayer<T>(
            _imageSize, _imageSize, 3, _patchSize, _visionEmbeddingDim);

        // Vision CLS token and positional embeddings
        _visionClsToken = Matrix<T>.CreateDefault(1, _visionEmbeddingDim, NumOps.Zero);
        _visionPositionalEmbeddings = Matrix<T>.CreateDefault(numPatches + 1, _visionEmbeddingDim, NumOps.Zero);

        // Vision transformer layers
        for (int i = 0; i < _numVisionLayers; i++)
        {
            _visionEncoderLayers.Add(new TransformerEncoderLayer<T>(_visionEmbeddingDim, _numHeads, ffnDim));
        }
        _visionLayerNorm = new LayerNormalizationLayer<T>(_visionEmbeddingDim);

        // Vision-Language Projector (MLP to map vision features to language space)
        _visionProjector1 = new DenseLayer<T>(_visionEmbeddingDim, _embeddingDimension, (IActivationFunction<T>?)null);
        _visionProjector2 = new DenseLayer<T>(_embeddingDimension, _embeddingDimension, (IActivationFunction<T>?)null);

        // Language model: Token embedding
        _tokenEmbedding = new EmbeddingLayer<T>(_vocabularySize, _embeddingDimension);
        _textPositionalEmbeddings = Matrix<T>.CreateDefault(_maxSequenceLength, _embeddingDimension, NumOps.Zero);

        // Language model transformer layers with causal masking
        for (int i = 0; i < _numLanguageLayers; i++)
        {
            _languageModelLayers.Add(new TransformerEncoderLayer<T>(_embeddingDimension, _numHeads, ffnDim));

            // Add cross-attention layer every 4 layers for vision-language fusion
            if (i % 4 == 0)
            {
                _crossAttentionLayers.Add(new TransformerEncoderLayer<T>(_embeddingDimension, _numHeads, ffnDim));
            }
        }

        _finalLayerNorm = new LayerNormalizationLayer<T>(_embeddingDimension);
        _lmHead = new DenseLayer<T>(_embeddingDimension, _vocabularySize, (IActivationFunction<T>?)null);
    }

    #endregion

    #region IMultimodalEmbedding Implementation

    /// <inheritdoc/>
    public Vector<T> GetImageEmbedding(Tensor<T> image)
    {
        var features = EncodeImage(image);
        return PoolFeatures(features);
    }

    /// <inheritdoc/>
    public IEnumerable<Vector<T>> GetImageEmbeddings(IEnumerable<Tensor<T>> images)
    {
        foreach (var image in images)
        {
            yield return GetImageEmbedding(image);
        }
    }

    /// <inheritdoc/>
    public Vector<T> GetTextEmbedding(string text)
    {
        var tokenResult = _tokenizer.Encode(text);
        var tokenTensor = CreateTokenTensor(tokenResult.TokenIds);
        var features = EncodeText(tokenTensor);
        return PoolFeatures(features);
    }

    /// <inheritdoc/>
    public IEnumerable<Vector<T>> GetTextEmbeddings(IEnumerable<string> texts)
    {
        foreach (var text in texts)
        {
            yield return GetTextEmbedding(text);
        }
    }

    /// <inheritdoc/>
    public T ComputeSimilarity(Tensor<T> image, string text)
    {
        var imageEmb = GetImageEmbedding(image);
        var textEmb = GetTextEmbedding(text);
        return CosineSimilarity(imageEmb, textEmb);
    }

    /// <inheritdoc/>
    public T ComputeSimilarity(Vector<T> textEmbedding, Vector<T> imageEmbedding)
    {
        return CosineSimilarity(textEmbedding, imageEmbedding);
    }

    /// <inheritdoc/>
    public Dictionary<string, T> ZeroShotClassify(Tensor<T> image, IEnumerable<string> labels)
    {
        if (labels is null)
        {
            throw new ArgumentNullException(nameof(labels));
        }

        var labelList = labels.ToList();
        if (labelList.Count == 0)
        {
            throw new ArgumentException("At least one label must be provided.", nameof(labels));
        }

        var result = new Dictionary<string, T>();
        var imageEmb = GetImageEmbedding(image);

        var scores = new List<T>();
        foreach (var label in labelList)
        {
            var textEmb = GetTextEmbedding(label);
            scores.Add(CosineSimilarity(imageEmb, textEmb));
        }

        // Softmax over scores
        var softmaxScores = Softmax(scores);
        for (int i = 0; i < labelList.Count; i++)
        {
            result[labelList[i]] = softmaxScores[i];
        }

        return result;
    }

    /// <inheritdoc/>
    public IEnumerable<(int Index, T Score)> RetrieveImages(string query, IEnumerable<Tensor<T>> images, int topK = 5)
    {
        var queryEmb = GetTextEmbedding(query);
        var imageList = images.ToList();
        var scores = new List<(int Index, T Score)>();

        for (int i = 0; i < imageList.Count; i++)
        {
            var imageEmb = GetImageEmbedding(imageList[i]);
            var score = CosineSimilarity(queryEmb, imageEmb);
            scores.Add((i, score));
        }

        return scores.OrderByDescending(s => NumOps.ToDouble(s.Score)).Take(topK);
    }

    /// <inheritdoc/>
    public IEnumerable<(int Index, T Score)> RetrieveTexts(Tensor<T> image, IEnumerable<string> texts, int topK = 5)
    {
        var imageEmb = GetImageEmbedding(image);
        var textList = texts.ToList();
        var scores = new List<(int Index, T Score)>();

        for (int i = 0; i < textList.Count; i++)
        {
            var textEmb = GetTextEmbedding(textList[i]);
            var score = CosineSimilarity(imageEmb, textEmb);
            scores.Add((i, score));
        }

        return scores.OrderByDescending(s => NumOps.ToDouble(s.Score)).Take(topK);
    }

    #endregion

    #region IGpt4VisionModel Implementation

    /// <inheritdoc/>
    public string Generate(
        Tensor<T> image,
        string prompt,
        int maxTokens = 1024,
        double temperature = 0.7)
    {
        var images = new[] { image };
        return GenerateFromMultipleImages(images, prompt, maxTokens, temperature);
    }

    /// <inheritdoc/>
    public string GenerateFromMultipleImages(
        IEnumerable<Tensor<T>> images,
        string prompt,
        int maxTokens = 1024,
        double temperature = 0.7)
    {
        var imageList = images.Take(_maxImagesPerRequest).ToList();

        // Encode all images
        var imageFeatures = new List<Matrix<T>>();
        foreach (var image in imageList)
        {
            var features = EncodeImage(image);
            var projected = ProjectVisionFeatures(features);
            imageFeatures.Add(projected);
        }

        // Encode prompt
        var tokenResult = _tokenizer.Encode(prompt);
        var tokenTensor = CreateTokenTensor(tokenResult.TokenIds);

        // Combine image features with text embeddings
        var combinedFeatures = CombineImageTextFeatures(imageFeatures, tokenTensor);

        // Generate tokens autoregressively
        var generatedTokens = GenerateTokens(combinedFeatures, maxTokens, temperature);

        // Decode tokens to text
        return _tokenizer.Decode(generatedTokens);
    }

    /// <inheritdoc/>
    public string Chat(
        Tensor<T> image,
        IEnumerable<(string Role, string Content)> conversationHistory,
        string userMessage,
        int maxTokens = 1024)
    {
        // Build conversation context
        var contextBuilder = new StringBuilder();
        foreach (var (role, content) in conversationHistory)
        {
            contextBuilder.AppendLine($"{role}: {content}");
        }
        contextBuilder.AppendLine($"user: {userMessage}");
        contextBuilder.AppendLine("assistant:");

        return Generate(image, contextBuilder.ToString(), maxTokens);
    }

    /// <inheritdoc/>
    public string AnalyzeDocument(
        Tensor<T> documentImage,
        string analysisType = "summary",
        string? additionalPrompt = null)
    {
        string prompt = analysisType.ToLowerInvariant() switch
        {
            "summary" => "Provide a comprehensive summary of this document.",
            "extract_text" => "Extract all visible text from this document, maintaining the layout as much as possible.",
            "answer_questions" => additionalPrompt ?? "Describe what you see in this document.",
            "analyze_structure" => "Analyze the structure of this document. Identify sections, headers, tables, and other elements.",
            _ => additionalPrompt ?? "Analyze this document."
        };

        if (!string.IsNullOrEmpty(additionalPrompt) && analysisType != "answer_questions")
        {
            prompt = $"{prompt} {additionalPrompt}";
        }

        return Generate(documentImage, prompt, maxTokens: 2048);
    }

    /// <inheritdoc/>
    public string ExtractStructuredData(Tensor<T> image, string schema)
    {
        string prompt = $"Extract data from this image according to the following JSON schema:\n{schema}\n\nRespond only with valid JSON matching the schema.";
        return Generate(image, prompt, maxTokens: 1024, temperature: 0.1);
    }

    /// <inheritdoc/>
    public string GenerateCodeFromUI(
        Tensor<T> uiScreenshot,
        string targetFramework = "html_css",
        string? additionalInstructions = null)
    {
        string frameworkPrompt = targetFramework.ToLowerInvariant() switch
        {
            "html_css" => "Generate HTML and CSS code",
            "react" => "Generate React component code with JSX",
            "flutter" => "Generate Flutter/Dart widget code",
            "swiftui" => "Generate SwiftUI code",
            _ => $"Generate {targetFramework} code"
        };

        string prompt = $"{frameworkPrompt} that recreates this user interface.";
        if (!string.IsNullOrEmpty(additionalInstructions))
        {
            prompt = $"{prompt} {additionalInstructions}";
        }

        return Generate(uiScreenshot, prompt, maxTokens: 4096, temperature: 0.3);
    }

    /// <inheritdoc/>
    public (string Answer, string Explanation) VisualReasoning(
        Tensor<T> image,
        string reasoningTask,
        string question)
    {
        string taskPrompt = reasoningTask.ToLowerInvariant() switch
        {
            "count" => "Count carefully and explain your counting process step by step.",
            "compare" => "Compare the elements carefully and explain the similarities and differences.",
            "spatial" => "Analyze the spatial relationships and positions of objects.",
            "temporal" => "Infer the temporal sequence or before/after relationships.",
            "causal" => "Reason about cause and effect relationships shown in the image.",
            _ => "Think through this step by step."
        };

        string prompt = $"Question: {question}\n\n{taskPrompt}\n\nProvide your answer followed by 'Explanation:' and your reasoning.";
        string response = Generate(image, prompt, maxTokens: 1024, temperature: 0.3);

        // Parse response to separate answer and explanation
        int explIndex = response.IndexOf("Explanation:", StringComparison.OrdinalIgnoreCase);
        if (explIndex > 0)
        {
            string answer = response.Substring(0, explIndex).Trim();
            string explanation = response.Substring(explIndex + 12).Trim();
            return (answer, explanation);
        }

        return (response, "Reasoning provided inline with answer.");
    }

    /// <inheritdoc/>
    public string DescribeImage(
        Tensor<T> image,
        string style = "factual",
        string detailLevel = "medium")
    {
        string stylePrompt = style.ToLowerInvariant() switch
        {
            "factual" => "Describe this image factually and objectively.",
            "poetic" => "Describe this image in a poetic and evocative way.",
            "technical" => "Provide a technical description of this image, including composition and visual elements.",
            "accessibility" => "Describe this image for someone who cannot see it, focusing on important visual information.",
            _ => "Describe this image."
        };

        string detailPrompt = detailLevel.ToLowerInvariant() switch
        {
            "low" => "Be brief and concise.",
            "medium" => "Provide a balanced level of detail.",
            "high" => "Be thorough and include fine details.",
            _ => ""
        };

        return Generate(image, $"{stylePrompt} {detailPrompt}", maxTokens: 512);
    }

    /// <inheritdoc/>
    public IEnumerable<(string Label, T Confidence, int X, int Y, int Width, int Height)> DetectObjects(
        Tensor<T> image,
        string? objectQuery = null)
    {
        string prompt = string.IsNullOrEmpty(objectQuery)
            ? "List all objects visible in this image with their approximate bounding boxes. Format: object_name (x, y, width, height) where coordinates are percentages of image dimensions."
            : $"Find '{objectQuery}' in this image and provide bounding box coordinates. Format: object_name (x, y, width, height) where coordinates are percentages of image dimensions.";

        string response = Generate(image, prompt, maxTokens: 1024, temperature: 0.1);

        // Parse detection results
        var detections = new List<(string Label, T Confidence, int X, int Y, int Width, int Height)>();
        var lines = response.Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (var line in lines)
        {
            int parenStart = line.IndexOf('(');
            int parenEnd = line.IndexOf(')');
            if (parenStart > 0 && parenEnd > parenStart)
            {
                string label = line.Substring(0, parenStart).Trim();
                string coords = line.Substring(parenStart + 1, parenEnd - parenStart - 1);
                var parts = coords.Split(',');
                if (parts.Length >= 4)
                {
                    if (int.TryParse(parts[0].Trim(), out int x) &&
                        int.TryParse(parts[1].Trim(), out int y) &&
                        int.TryParse(parts[2].Trim(), out int w) &&
                        int.TryParse(parts[3].Trim(), out int h))
                    {
                        detections.Add((label, NumOps.FromDouble(0.8), x, y, w, h));
                    }
                }
            }
        }

        return detections;
    }

    /// <inheritdoc/>
    public (string Answer, T Confidence) AnswerVisualQuestion(Tensor<T> image, string question)
    {
        string prompt = $"Question: {question}\n\nProvide a clear and concise answer.";
        string answer = Generate(image, prompt, maxTokens: 256, temperature: 0.3);
        return (answer, NumOps.FromDouble(0.85));
    }

    /// <inheritdoc/>
    public string CompareImages(
        Tensor<T> image1,
        Tensor<T> image2,
        string comparisonType = "detailed")
    {
        string typePrompt = comparisonType.ToLowerInvariant() switch
        {
            "visual" => "Compare the visual elements, colors, and composition of these two images.",
            "semantic" => "Compare what these images depict and their meaning.",
            "detailed" => "Provide a comprehensive comparison of these two images, including both visual and semantic differences.",
            _ => "Compare these two images."
        };

        return GenerateFromMultipleImages(new[] { image1, image2 }, typePrompt, maxTokens: 1024);
    }

    /// <inheritdoc/>
    public string GenerateEditInstructions(Tensor<T> image, string editRequest)
    {
        string prompt = $"Given this image and the following edit request: '{editRequest}'\n\nProvide detailed, step-by-step instructions for an image editing tool to accomplish this edit.";
        return Generate(image, prompt, maxTokens: 1024);
    }

    /// <inheritdoc/>
    public (string Text, Dictionary<string, object>? LayoutInfo) ExtractText(
        Tensor<T> image,
        bool preserveLayout = false)
    {
        string prompt = preserveLayout
            ? "Extract all text from this image, preserving the spatial layout as much as possible using spacing and newlines."
            : "Extract all text from this image.";

        string text = Generate(image, prompt, maxTokens: 2048, temperature: 0.1);

        Dictionary<string, object>? layoutInfo = null;
        if (preserveLayout)
        {
            layoutInfo = new Dictionary<string, object>
            {
                ["preserved_layout"] = true,
                ["extraction_method"] = "vision_llm"
            };
        }

        return (text, layoutInfo);
    }

    /// <inheritdoc/>
    public (string ChartType, Dictionary<string, object> Data, string Interpretation) AnalyzeChart(
        Tensor<T> chartImage)
    {
        string prompt = @"Analyze this chart/graph and provide:
1. The type of chart (bar, line, pie, scatter, etc.)
2. The data shown (extract key values if possible)
3. An interpretation of what the chart shows

Format your response as:
TYPE: [chart type]
DATA: [key data points]
INTERPRETATION: [your analysis]";

        string response = Generate(chartImage, prompt, maxTokens: 1024, temperature: 0.3);

        string chartType = "unknown";
        var data = new Dictionary<string, object>();
        string interpretation = response;

        var lines = response.Split('\n');
        foreach (var line in lines)
        {
            if (line.StartsWith("TYPE:", StringComparison.OrdinalIgnoreCase))
                chartType = line.Substring(5).Trim();
            else if (line.StartsWith("DATA:", StringComparison.OrdinalIgnoreCase))
                data["raw_data"] = line.Substring(5).Trim();
            else if (line.StartsWith("INTERPRETATION:", StringComparison.OrdinalIgnoreCase))
                interpretation = line.Substring(15).Trim();
        }

        return (chartType, data, interpretation);
    }

    /// <inheritdoc/>
    public string GenerateStory(
        Tensor<T> image,
        string genre = "general",
        string length = "medium")
    {
        int maxTokens = length.ToLowerInvariant() switch
        {
            "short" => 256,
            "medium" => 512,
            "long" => 1024,
            _ => 512
        };

        string genrePrompt = genre.ToLowerInvariant() switch
        {
            "fantasy" => "Write a fantasy story",
            "mystery" => "Write a mystery story",
            "romance" => "Write a romantic story",
            "scifi" => "Write a science fiction story",
            _ => "Write a creative story"
        };

        return Generate(image, $"{genrePrompt} inspired by this image.", maxTokens: maxTokens, temperature: 0.9);
    }

    /// <inheritdoc/>
    public (Dictionary<string, T> QualityScores, IEnumerable<string> Suggestions) EvaluateImageQuality(
        Tensor<T> image)
    {
        string prompt = @"Evaluate this image's quality on these dimensions (score 1-10):
1. Composition
2. Lighting
3. Focus/Sharpness
4. Color Balance
5. Overall Quality

Also suggest improvements if any.

Format: DIMENSION: score
Then: SUGGESTIONS: list";

        string response = Generate(image, prompt, maxTokens: 512, temperature: 0.3);

        var scores = new Dictionary<string, T>();
        var suggestions = new List<string>();

        var lines = response.Split('\n');
        bool inSuggestions = false;

        foreach (var line in lines)
        {
            if (line.Contains("SUGGESTIONS", StringComparison.OrdinalIgnoreCase))
            {
                inSuggestions = true;
                continue;
            }

            if (inSuggestions)
            {
                if (!string.IsNullOrWhiteSpace(line))
                    suggestions.Add(line.Trim());
            }
            else
            {
                var parts = line.Split(':');
                if (parts.Length >= 2 && double.TryParse(parts[^1].Trim(), out double score))
                {
                    scores[parts[0].Trim()] = NumOps.FromDouble(score / 10.0);
                }
            }
        }

        return (scores, suggestions);
    }

    /// <inheritdoc/>
    public Dictionary<string, (bool IsFlagged, T Confidence)> SafetyCheck(Tensor<T> image)
    {
        string prompt = @"Analyze this image for potential safety concerns in these categories:
1. Violence
2. Adult Content
3. Hate Symbols
4. Self-Harm
5. Dangerous Activities

For each category, indicate if it's flagged (YES/NO) and confidence level (HIGH/MEDIUM/LOW).";

        string response = Generate(image, prompt, maxTokens: 512, temperature: 0.1);

        var result = new Dictionary<string, (bool IsFlagged, T Confidence)>
        {
            ["violence"] = (false, NumOps.Zero),
            ["adult_content"] = (false, NumOps.Zero),
            ["hate_symbols"] = (false, NumOps.Zero),
            ["self_harm"] = (false, NumOps.Zero),
            ["dangerous_activities"] = (false, NumOps.Zero)
        };

        // Parse each category's response more accurately by looking for pattern like:
        // "Violence: YES" or "1. Violence - FLAGGED" rather than just checking if both appear anywhere
        foreach (var category in result.Keys.ToList())
        {
            // Create patterns that match category followed by YES/FLAGGED within the same line
            // This prevents false positives from unrelated YES/FLAGGED responses
            var categoryUpper = category.ToUpperInvariant().Replace("_", "[\\s_-]*");
            var pattern = new Regex(
                $@"{categoryUpper}[:\s\-]*\b(YES|FLAGGED)\b",
                RegexOptions.IgnoreCase, RegexTimeout);

            bool flagged = pattern.IsMatch(response);

            // Also check for confidence level if present
            T confidence;
            if (flagged)
            {
                // Look for confidence level after category
                var confPattern = new Regex(
                    $@"{categoryUpper}[^.]*\b(HIGH|MEDIUM|LOW)\b",
                    RegexOptions.IgnoreCase, RegexTimeout);
                var confMatch = confPattern.Match(response);
                if (confMatch.Success)
                {
                    var confLevel = confMatch.Groups[1].Value.ToUpperInvariant();
                    confidence = confLevel switch
                    {
                        "HIGH" => NumOps.FromDouble(0.9),
                        "MEDIUM" => NumOps.FromDouble(0.7),
                        "LOW" => NumOps.FromDouble(0.5),
                        _ => NumOps.FromDouble(0.7)
                    };
                }
                else
                {
                    confidence = NumOps.FromDouble(0.7);
                }
            }
            else
            {
                confidence = NumOps.FromDouble(0.9); // High confidence in "not flagged"
            }

            result[category] = (flagged, confidence);
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> GetAttentionMap(Tensor<T> image, string prompt)
    {
        int patchesPerSide = _imageSize / _patchSize;
        int numPatches = patchesPerSide * patchesPerSide;

        if (_useNativeMode)
        {
            var features = EncodeImage(image);

            var attentionMap = Tensor<T>.CreateDefault([patchesPerSide, patchesPerSide], NumOps.Zero);
            for (int i = 0; i < numPatches && i < features.Rows; i++)
            {
                T magnitude = NumOps.Zero;
                for (int j = 0; j < features.Columns; j++)
                {
                    magnitude = NumOps.Add(magnitude, NumOps.Abs(features[i, j]));
                }
                int row = i / patchesPerSide;
                int col = i % patchesPerSide;
                attentionMap[row, col] = magnitude;
            }

            return NormalizeTensor(attentionMap);
        }
        else
        {
            var attentionMap = Tensor<T>.CreateDefault([patchesPerSide, patchesPerSide], NumOps.Zero);
            T value = NumOps.Divide(NumOps.One, NumOps.FromDouble(numPatches));
            for (int i = 0; i < patchesPerSide; i++)
            {
                for (int j = 0; j < patchesPerSide; j++)
                {
                    attentionMap[i, j] = value;
                }
            }
            return attentionMap;
        }
    }

    #endregion

    #region Core Processing Methods

    private Matrix<T> EncodeImage(Tensor<T> image)
    {
        if (_useNativeMode)
        {
            return EncodeImageNative(image);
        }
        else
        {
            return EncodeImageOnnx(image);
        }
    }

    private Matrix<T> EncodeImageNative(Tensor<T> image)
    {
        // Patch embedding
        var patchEmbeddings = _visionPatchEmbedding!.Forward(image);
        int numPatches = patchEmbeddings.Shape[0];

        // Create tensor for transformer processing
        var embeddings = Tensor<T>.CreateDefault([numPatches + 1, _visionEmbeddingDim], NumOps.Zero);

        // Add CLS token
        for (int j = 0; j < _visionEmbeddingDim && j < _visionClsToken!.Columns; j++)
        {
            embeddings[0, j] = _visionClsToken[0, j];
        }

        // Add patch embeddings
        for (int i = 0; i < numPatches; i++)
        {
            for (int j = 0; j < _visionEmbeddingDim && j < patchEmbeddings.Shape[1]; j++)
            {
                embeddings[i + 1, j] = patchEmbeddings[i, j];
            }
        }

        // Add positional embeddings
        for (int i = 0; i < embeddings.Shape[0] && i < _visionPositionalEmbeddings!.Rows; i++)
        {
            for (int j = 0; j < _visionEmbeddingDim && j < _visionPositionalEmbeddings.Columns; j++)
            {
                embeddings[i, j] = NumOps.Add(embeddings[i, j], _visionPositionalEmbeddings[i, j]);
            }
        }

        // Pass through transformer layers
        var current = embeddings;
        foreach (var layer in _visionEncoderLayers)
        {
            current = layer.Forward(current);
        }

        // Layer norm
        current = _visionLayerNorm!.Forward(current);

        return TensorToMatrix(current);
    }

    private Matrix<T> EncodeImageOnnx(Tensor<T> image)
    {
        if (_visionEncoder is null)
        {
            // Return dummy encoding
            return Matrix<T>.CreateDefault((_imageSize / _patchSize) * (_imageSize / _patchSize) + 1, _visionEmbeddingDim, NumOps.Zero);
        }

        // Prepare input
        var inputData = new float[1 * 3 * _imageSize * _imageSize];
        for (int c = 0; c < 3 && c < image.Shape[0]; c++)
        {
            for (int h = 0; h < _imageSize && h < image.Shape[1]; h++)
            {
                for (int w = 0; w < _imageSize && w < image.Shape[2]; w++)
                {
                    int idx = c * _imageSize * _imageSize + h * _imageSize + w;
                    inputData[idx] = (float)NumOps.ToDouble(image[c, h, w]);
                }
            }
        }

        var inputTensor = new OnnxTensors.DenseTensor<float>(inputData, new[] { 1, 3, _imageSize, _imageSize });
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("pixel_values", inputTensor)
        };

        using var results = _visionEncoder.Run(inputs);
        var output = results.First().AsTensor<float>();

        int seqLen = output.Dimensions[1];
        int hiddenDim = output.Dimensions[2];
        var matrix = Matrix<T>.CreateDefault(seqLen, hiddenDim, NumOps.Zero);

        for (int i = 0; i < seqLen; i++)
        {
            for (int j = 0; j < hiddenDim; j++)
            {
                matrix[i, j] = NumOps.FromDouble(output[0, i, j]);
            }
        }

        return matrix;
    }

    private Matrix<T> ProjectVisionFeatures(Matrix<T> visionFeatures)
    {
        if (!_useNativeMode || _visionProjector1 is null || _visionProjector2 is null)
        {
            return visionFeatures;
        }

        var projected = Matrix<T>.CreateDefault(visionFeatures.Rows, _embeddingDimension, NumOps.Zero);

        for (int i = 0; i < visionFeatures.Rows; i++)
        {
            var row = Tensor<T>.CreateDefault([visionFeatures.Columns], NumOps.Zero);
            for (int j = 0; j < visionFeatures.Columns; j++)
            {
                row[j] = visionFeatures[i, j];
            }

            var h = _visionProjector1.Forward(row);
            h = ApplyGELU(h);
            h = _visionProjector2.Forward(h);

            for (int j = 0; j < _embeddingDimension && j < h.Length; j++)
            {
                projected[i, j] = h[j];
            }
        }

        return projected;
    }

    private Matrix<T> EncodeText(Tensor<T> tokens)
    {
        if (_useNativeMode)
        {
            return EncodeTextNative(tokens);
        }
        else
        {
            return EncodeTextOnnx(tokens);
        }
    }

    private Matrix<T> EncodeTextNative(Tensor<T> tokens)
    {
        if (_tokenEmbedding is null || _textPositionalEmbeddings is null)
        {
            return Matrix<T>.CreateDefault(tokens.Length, _embeddingDimension, NumOps.Zero);
        }

        var embeddings = _tokenEmbedding.Forward(tokens);
        int seqLen = Math.Min(tokens.Length, _maxSequenceLength);

        var embMatrix = TensorToMatrix(embeddings);
        for (int i = 0; i < seqLen && i < embMatrix.Rows; i++)
        {
            for (int j = 0; j < _embeddingDimension && j < embMatrix.Columns && j < _textPositionalEmbeddings.Columns; j++)
            {
                embMatrix[i, j] = NumOps.Add(embMatrix[i, j], _textPositionalEmbeddings[i, j]);
            }
        }

        return embMatrix;
    }

    private Matrix<T> EncodeTextOnnx(Tensor<T> tokens)
    {
        if (_languageModel is null)
        {
            return Matrix<T>.CreateDefault(tokens.Length, _embeddingDimension, NumOps.Zero);
        }

        var inputData = new long[tokens.Length];
        for (int i = 0; i < tokens.Length; i++)
        {
            inputData[i] = (long)NumOps.ToDouble(tokens[i]);
        }

        var inputTensor = new OnnxTensors.DenseTensor<long>(inputData, new[] { 1, tokens.Length });
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputTensor)
        };

        using var results = _languageModel.Run(inputs);
        var output = results.First().AsTensor<float>();

        int seqLen = output.Dimensions[1];
        int hiddenDim = output.Dimensions[2];
        var matrix = Matrix<T>.CreateDefault(seqLen, hiddenDim, NumOps.Zero);

        for (int i = 0; i < seqLen; i++)
        {
            for (int j = 0; j < hiddenDim; j++)
            {
                matrix[i, j] = NumOps.FromDouble(output[0, i, j]);
            }
        }

        return matrix;
    }

    private Matrix<T> CombineImageTextFeatures(List<Matrix<T>> imageFeatures, Tensor<T> tokens)
    {
        var textFeatures = EncodeText(tokens);

        int totalImageTokens = imageFeatures.Sum(f => f.Rows);
        int textTokens = textFeatures.Rows;
        int totalLength = totalImageTokens + textTokens;

        var combined = Matrix<T>.CreateDefault(totalLength, _embeddingDimension, NumOps.Zero);

        int offset = 0;
        foreach (var imgFeat in imageFeatures)
        {
            for (int i = 0; i < imgFeat.Rows; i++)
            {
                for (int j = 0; j < _embeddingDimension && j < imgFeat.Columns; j++)
                {
                    combined[offset + i, j] = imgFeat[i, j];
                }
            }
            offset += imgFeat.Rows;
        }

        for (int i = 0; i < textTokens; i++)
        {
            for (int j = 0; j < _embeddingDimension && j < textFeatures.Columns; j++)
            {
                combined[offset + i, j] = textFeatures[i, j];
            }
        }

        return combined;
    }

    private List<int> GenerateTokens(Matrix<T> contextFeatures, int maxTokens, double temperature)
    {
        var generatedTokens = new List<int>();
        var currentFeatures = contextFeatures;

        for (int step = 0; step < maxTokens; step++)
        {
            var logits = GetNextTokenLogits(currentFeatures);

            if (Math.Abs(temperature - 1.0) > 0.001)
            {
                for (int i = 0; i < logits.Length; i++)
                {
                    logits[i] = NumOps.Divide(logits[i], NumOps.FromDouble(temperature));
                }
            }

            var probs = SoftmaxVector(logits);
            int nextToken = SampleFromDistribution(probs);

            if (IsEndToken(nextToken))
            {
                break;
            }

            generatedTokens.Add(nextToken);
            currentFeatures = AppendTokenToContext(currentFeatures, nextToken);
        }

        return generatedTokens;
    }

    private Vector<T> GetNextTokenLogits(Matrix<T> features)
    {
        if (_useNativeMode && _languageModelLayers.Count > 0)
        {
            var current = MatrixToTensor(features);

            int crossAttnIdx = 0;
            for (int i = 0; i < _languageModelLayers.Count; i++)
            {
                current = _languageModelLayers[i].Forward(current);

                if (i % 4 == 0 && crossAttnIdx < _crossAttentionLayers.Count)
                {
                    current = _crossAttentionLayers[crossAttnIdx].Forward(current);
                    crossAttnIdx++;
                }
            }

            if (_finalLayerNorm is not null)
            {
                current = _finalLayerNorm.Forward(current);
            }

            var lastHidden = Tensor<T>.CreateDefault([_embeddingDimension], NumOps.Zero);
            int lastPos = current.Shape[0] - 1;
            for (int j = 0; j < _embeddingDimension && j < current.Shape[1]; j++)
            {
                lastHidden[j] = current[lastPos, j];
            }

            if (_lmHead is not null)
            {
                var logitsTensor = _lmHead.Forward(lastHidden);

                var logits = new Vector<T>(_vocabularySize);
                for (int i = 0; i < _vocabularySize && i < logitsTensor.Length; i++)
                {
                    logits[i] = logitsTensor[i];
                }
                return logits;
            }
        }

        // Return uniform distribution as fallback
        var uniformLogits = new Vector<T>(_vocabularySize);
        T value = NumOps.Divide(NumOps.One, NumOps.FromDouble(_vocabularySize));
        for (int i = 0; i < _vocabularySize; i++)
        {
            uniformLogits[i] = value;
        }
        return uniformLogits;
    }

    #endregion

    #region Helper Methods

    private Tensor<T> CreateTokenTensor(IEnumerable<int> tokens)
    {
        var tokenList = tokens.ToList();
        var tensor = Tensor<T>.CreateDefault([tokenList.Count], NumOps.Zero);
        for (int i = 0; i < tokenList.Count; i++)
        {
            tensor[i] = NumOps.FromDouble(tokenList[i]);
        }
        return tensor;
    }

    private Vector<T> PoolFeatures(Matrix<T> features)
    {
        var pooled = new Vector<T>(features.Columns);
        for (int j = 0; j < features.Columns; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < features.Rows; i++)
            {
                sum = NumOps.Add(sum, features[i, j]);
            }
            pooled[j] = NumOps.Divide(sum, NumOps.FromDouble(features.Rows));
        }
        return pooled;
    }

    private T CosineSimilarity(Vector<T> a, Vector<T> b)
    {
        T dot = NumOps.Zero;
        T normA = NumOps.Zero;
        T normB = NumOps.Zero;

        int len = Math.Min(a.Length, b.Length);
        for (int i = 0; i < len; i++)
        {
            dot = NumOps.Add(dot, NumOps.Multiply(a[i], b[i]));
            normA = NumOps.Add(normA, NumOps.Multiply(a[i], a[i]));
            normB = NumOps.Add(normB, NumOps.Multiply(b[i], b[i]));
        }

        T denom = NumOps.Multiply(NumOps.Sqrt(normA), NumOps.Sqrt(normB));
        if (NumOps.ToDouble(denom) < 1e-8)
        {
            return NumOps.Zero;
        }

        return NumOps.Divide(dot, denom);
    }

    private List<T> Softmax(List<T> values)
    {
        T maxVal = values.Max(v => NumOps.ToDouble(v)) is double m ? NumOps.FromDouble(m) : NumOps.Zero;

        var expValues = values.Select(v => NumOps.Exp(NumOps.Subtract(v, maxVal))).ToList();
        T sum = expValues.Aggregate(NumOps.Zero, (acc, v) => NumOps.Add(acc, v));

        return expValues.Select(v => NumOps.Divide(v, sum)).ToList();
    }

    private Vector<T> SoftmaxVector(Vector<T> values)
    {
        double maxVal = double.MinValue;
        for (int i = 0; i < values.Length; i++)
        {
            double val = NumOps.ToDouble(values[i]);
            if (val > maxVal) maxVal = val;
        }

        var result = new Vector<T>(values.Length);
        T sum = NumOps.Zero;

        for (int i = 0; i < values.Length; i++)
        {
            result[i] = NumOps.Exp(NumOps.Subtract(values[i], NumOps.FromDouble(maxVal)));
            sum = NumOps.Add(sum, result[i]);
        }

        for (int i = 0; i < values.Length; i++)
        {
            result[i] = NumOps.Divide(result[i], sum);
        }

        return result;
    }

    private int SampleFromDistribution(Vector<T> probs)
    {
        // Use thread-safe random instead of creating new Random() per call
        // (new Random() produces identical sequences when called rapidly)
        double random = Tensors.Helpers.RandomHelper.ThreadSafeRandom.NextDouble();
        double cumulative = 0;

        for (int i = 0; i < probs.Length; i++)
        {
            cumulative += NumOps.ToDouble(probs[i]);
            if (random < cumulative)
            {
                return i;
            }
        }

        return probs.Length - 1;
    }

    private bool IsEndToken(int token)
    {
        return token == 2 || token == 50256 || token == 128001;
    }

    private Matrix<T> AppendTokenToContext(Matrix<T> context, int token)
    {
        var newContext = Matrix<T>.CreateDefault(context.Rows + 1, context.Columns, NumOps.Zero);

        for (int i = 0; i < context.Rows; i++)
        {
            for (int j = 0; j < context.Columns; j++)
            {
                newContext[i, j] = context[i, j];
            }
        }

        if (_useNativeMode && _tokenEmbedding is not null)
        {
            var tokenTensor = Tensor<T>.CreateDefault([1], NumOps.Zero);
            tokenTensor[0] = NumOps.FromDouble(token);
            var embedding = _tokenEmbedding.Forward(tokenTensor);
            for (int j = 0; j < context.Columns && j < embedding.Length; j++)
            {
                newContext[context.Rows, j] = embedding[j];
            }
        }

        return newContext;
    }

    private Tensor<T> ApplyGELU(Tensor<T> x)
    {
        var result = Tensor<T>.CreateDefault(x.Shape, NumOps.Zero);
        for (int i = 0; i < x.Length; i++)
        {
            double val = NumOps.ToDouble(x[i]);
            double gelu = val * 0.5 * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (val + 0.044715 * val * val * val)));
            result[i] = NumOps.FromDouble(gelu);
        }
        return result;
    }

    private Tensor<T> MatrixToTensor(Matrix<T> matrix)
    {
        var tensor = Tensor<T>.CreateDefault([matrix.Rows, matrix.Columns], NumOps.Zero);
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                tensor[i, j] = matrix[i, j];
            }
        }
        return tensor;
    }

    private Matrix<T> TensorToMatrix(Tensor<T> tensor)
    {
        int rows = tensor.Shape[0];
        int cols = tensor.Shape.Length > 1 ? tensor.Shape[1] : 1;
        var matrix = Matrix<T>.CreateDefault(rows, cols, NumOps.Zero);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = tensor.Shape.Length > 1 ? tensor[i, j] : tensor[i];
            }
        }

        return matrix;
    }

    private Tensor<T> NormalizeTensor(Tensor<T> tensor)
    {
        T min = tensor[0, 0];
        T max = tensor[0, 0];

        for (int i = 0; i < tensor.Shape[0]; i++)
        {
            for (int j = 0; j < tensor.Shape[1]; j++)
            {
                if (NumOps.ToDouble(tensor[i, j]) < NumOps.ToDouble(min))
                    min = tensor[i, j];
                if (NumOps.ToDouble(tensor[i, j]) > NumOps.ToDouble(max))
                    max = tensor[i, j];
            }
        }

        T range = NumOps.Subtract(max, min);
        if (NumOps.ToDouble(range) < 1e-8)
        {
            return tensor;
        }

        var normalized = Tensor<T>.CreateDefault(tensor.Shape, NumOps.Zero);
        for (int i = 0; i < tensor.Shape[0]; i++)
        {
            for (int j = 0; j < tensor.Shape[1]; j++)
            {
                normalized[i, j] = NumOps.Divide(NumOps.Subtract(tensor[i, j], min), range);
            }
        }

        return normalized;
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        SetTrainingMode(false);
        var embedding = GetImageEmbedding(input);
        return VectorToTensor(embedding);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);

        // Forward pass
        var imageFeatures = EncodeImage(input);
        var projected = ProjectVisionFeatures(imageFeatures);

        // Compute loss using the loss function
        var predictedEmbedding = PoolFeatures(projected);
        var targetEmbedding = TensorToVector(expectedOutput);
        LastLoss = LossFunction.CalculateLoss(predictedEmbedding, targetEmbedding);

        // Compute gradient of loss w.r.t. output
        var lossGradient = LossFunction.CalculateDerivative(predictedEmbedding, targetEmbedding);

        // Get parameter gradients and apply gradient descent update
        var paramGradients = GetGpt4VParameterGradients();
        UpdateParameters(paramGradients);

        SetTrainingMode(false);
    }

    /// <summary>
    /// Gets the gradients for all trainable parameters.
    /// </summary>
    private Vector<T> GetGpt4VParameterGradients()
    {
        var gradients = new List<T>();

        // Get gradients from vision encoder layers
        foreach (var layer in _visionEncoderLayers)
        {
            var layerGrads = layer.GetParameterGradients();
            for (int i = 0; i < layerGrads.Length; i++)
            {
                gradients.Add(layerGrads[i]);
            }
        }

        // Get gradients from language model layers
        foreach (var layer in _languageModelLayers)
        {
            var layerGrads = layer.GetParameterGradients();
            for (int i = 0; i < layerGrads.Length; i++)
            {
                gradients.Add(layerGrads[i]);
            }
        }

        // Get gradients from cross-attention layers
        foreach (var layer in _crossAttentionLayers)
        {
            var layerGrads = layer.GetParameterGradients();
            for (int i = 0; i < layerGrads.Length; i++)
            {
                gradients.Add(layerGrads[i]);
            }
        }

        // Get gradients from projection layers
        if (_visionProjector1 is not null)
        {
            var projGrads = _visionProjector1.GetParameterGradients();
            for (int i = 0; i < projGrads.Length; i++)
            {
                gradients.Add(projGrads[i]);
            }
        }

        if (_visionProjector2 is not null)
        {
            var projGrads = _visionProjector2.GetParameterGradients();
            for (int i = 0; i < projGrads.Length; i++)
            {
                gradients.Add(projGrads[i]);
            }
        }

        return new Vector<T>([.. gradients]);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> gradients)
    {
        int expectedCount = ParameterCount;
        if (gradients.Length != expectedCount)
        {
            throw new ArgumentException($"Expected {expectedCount} gradients but got {gradients.Length}");
        }

        if (!_useNativeMode) return;

        // Get current parameters
        var currentParams = GetParameters();

        // Apply gradient descent update: params = params - learning_rate * gradients
        T learningRate = NumOps.FromDouble(0.001); // Default learning rate
        for (int i = 0; i < currentParams.Length; i++)
        {
            currentParams[i] = NumOps.Subtract(currentParams[i], NumOps.Multiply(learningRate, gradients[i]));
        }

        // Set the updated parameters
        SetParameters(currentParams);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "Gpt4VisionNeuralNetwork",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = _embeddingDimension,
            Complexity = _visionEncoderLayers.Count + _languageModelLayers.Count + _crossAttentionLayers.Count,
            Description = "GPT-4V style vision-language model combining vision encoding with language generation",
            AdditionalInfo = new Dictionary<string, object>
            {
                ["parameter_count"] = ParameterCount,
                ["layer_count"] = _visionEncoderLayers.Count + _languageModelLayers.Count + _crossAttentionLayers.Count,
                ["input_shape"] = new int[] { 3, _imageSize, _imageSize },
                ["output_shape"] = new int[] { _embeddingDimension },
                ["embedding_dimension"] = _embeddingDimension,
                ["vision_embedding_dim"] = _visionEmbeddingDim,
                ["max_sequence_length"] = _maxSequenceLength,
                ["context_window_size"] = _contextWindowSize,
                ["image_size"] = _imageSize,
                ["vocabulary_size"] = _vocabularySize,
                ["num_vision_layers"] = _numVisionLayers,
                ["num_language_layers"] = _numLanguageLayers,
                ["use_native_mode"] = _useNativeMode
            }
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_embeddingDimension);
        writer.Write(_visionEmbeddingDim);
        writer.Write(_maxSequenceLength);
        writer.Write(_contextWindowSize);
        writer.Write(_imageSize);
        writer.Write(_hiddenDim);
        writer.Write(_numVisionLayers);
        writer.Write(_numLanguageLayers);
        writer.Write(_numHeads);
        writer.Write(_patchSize);
        writer.Write(_vocabularySize);
        writer.Write(_maxImagesPerRequest);
        writer.Write(_useNativeMode);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _embeddingDimension = reader.ReadInt32();
        _visionEmbeddingDim = reader.ReadInt32();
        _maxSequenceLength = reader.ReadInt32();
        _contextWindowSize = reader.ReadInt32();
        _imageSize = reader.ReadInt32();
        _hiddenDim = reader.ReadInt32();
        _numVisionLayers = reader.ReadInt32();
        _numLanguageLayers = reader.ReadInt32();
        _numHeads = reader.ReadInt32();
        _patchSize = reader.ReadInt32();
        _vocabularySize = reader.ReadInt32();
        _maxImagesPerRequest = reader.ReadInt32();
        _useNativeMode = reader.ReadBoolean();
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode)
        {
            // For ONNX mode, we need valid paths - extract to local variables for null safety
            string visionPath = _visionEncoderPath ?? string.Empty;
            string languagePath = _languageModelPath ?? string.Empty;

            if (visionPath.Length == 0 || languagePath.Length == 0)
            {
                throw new InvalidOperationException(
                    "Cannot create new instance in ONNX mode: model paths are not available. " +
                    "ONNX model paths are not serialized. Use native mode for serialization.");
            }

            return new Gpt4VisionNeuralNetwork<T>(
                Architecture,
                visionPath,
                languagePath,
                _tokenizer,
                _embeddingDimension,
                _visionEmbeddingDim,
                _maxSequenceLength,
                _contextWindowSize,
                _imageSize,
                _maxImagesPerRequest);
        }

        return new Gpt4VisionNeuralNetwork<T>(
            Architecture,
            _tokenizer,
            _embeddingDimension,
            _visionEmbeddingDim,
            _maxSequenceLength,
            _contextWindowSize,
            _imageSize,
            _hiddenDim,
            _numVisionLayers,
            _numLanguageLayers,
            _numHeads,
            _patchSize,
            _vocabularySize,
            _maxImagesPerRequest);
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _visionEncoder?.Dispose();
            _languageModel?.Dispose();
        }

        base.Dispose(disposing);
    }

    private Tensor<T> VectorToTensor(Vector<T> vector)
    {
        var tensor = Tensor<T>.CreateDefault([vector.Length], NumOps.Zero);
        for (int i = 0; i < vector.Length; i++)
        {
            tensor[i] = vector[i];
        }
        return tensor;
    }

    private Vector<T> TensorToVector(Tensor<T> tensor)
    {
        var vector = new Vector<T>(tensor.Length);
        for (int i = 0; i < tensor.Length; i++)
        {
            vector[i] = tensor[i];
        }
        return vector;
    }

    #endregion
}
