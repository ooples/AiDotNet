using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for GPT-4V-style models that combine vision understanding with large language model capabilities.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// GPT-4V represents the integration of vision capabilities into large language models,
/// enabling sophisticated visual reasoning, multi-turn conversations about images,
/// and complex visual-linguistic tasks.
/// </para>
/// <para><b>For Beginners:</b> GPT-4V is like giving ChatGPT the ability to see!
///
/// Key capabilities:
/// - Visual reasoning: Understanding relationships, counting, spatial awareness
/// - Multi-turn dialogue: Context-aware conversations about images
/// - Document understanding: Reading and analyzing documents, charts, diagrams
/// - Code generation from screenshots: Understanding UI and generating code
/// - Creative tasks: Describing images poetically, writing stories from images
///
/// Architecture concepts:
/// 1. Vision Encoder: Processes images into visual tokens
/// 2. Visual-Language Alignment: Maps visual features to LLM embedding space
/// 3. Large Language Model: Generates text responses conditioned on visual input
/// 4. Multi-modal Attention: Allows text to attend to relevant image regions
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("Gpt4VisionModel")]
public interface IGpt4VisionModel<T> : IMultimodalEmbedding<T>
{
    /// <summary>
    /// Gets the maximum number of images that can be processed in a single request.
    /// </summary>
    int MaxImagesPerRequest { get; }

    /// <summary>
    /// Gets the maximum resolution supported for input images.
    /// </summary>
    (int Width, int Height) MaxImageResolution { get; }

    /// <summary>
    /// Gets the context window size in tokens.
    /// </summary>
    int ContextWindowSize { get; }

    /// <summary>
    /// Gets the supported image detail levels.
    /// </summary>
    IReadOnlyList<string> SupportedDetailLevels { get; }

    /// <summary>
    /// Generates a response based on an image and text prompt.
    /// </summary>
    /// <param name="image">The input image tensor [channels, height, width].</param>
    /// <param name="prompt">The text prompt or question about the image.</param>
    /// <param name="maxTokens">Maximum tokens to generate.</param>
    /// <param name="temperature">Sampling temperature (0-2).</param>
    /// <returns>Generated text response.</returns>
    string Generate(
        Tensor<T> image,
        string prompt,
        int maxTokens = 1024,
        double temperature = 0.7);

    /// <summary>
    /// Generates a response based on multiple images and text prompt.
    /// </summary>
    /// <param name="images">Multiple input images.</param>
    /// <param name="prompt">The text prompt referencing the images.</param>
    /// <param name="maxTokens">Maximum tokens to generate.</param>
    /// <param name="temperature">Sampling temperature.</param>
    /// <returns>Generated text response.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Compare and analyze multiple images!
    ///
    /// Examples:
    /// - "What are the differences between these two images?"
    /// - "Which of these products looks more appealing?"
    /// - "Describe how these images are related."
    /// </para>
    /// </remarks>
    string GenerateFromMultipleImages(
        IEnumerable<Tensor<T>> images,
        string prompt,
        int maxTokens = 1024,
        double temperature = 0.7);

    /// <summary>
    /// Conducts a multi-turn conversation about an image.
    /// </summary>
    /// <param name="image">The image being discussed.</param>
    /// <param name="conversationHistory">Previous turns as (role, content) pairs.</param>
    /// <param name="userMessage">The new user message.</param>
    /// <param name="maxTokens">Maximum tokens to generate.</param>
    /// <returns>Generated assistant response.</returns>
    string Chat(
        Tensor<T> image,
        IEnumerable<(string Role, string Content)> conversationHistory,
        string userMessage,
        int maxTokens = 1024);

    /// <summary>
    /// Analyzes a document image (PDF page, screenshot, etc.).
    /// </summary>
    /// <param name="documentImage">The document image.</param>
    /// <param name="analysisType">Type: "summary", "extract_text", "answer_questions", "analyze_structure".</param>
    /// <param name="additionalPrompt">Optional additional instructions.</param>
    /// <returns>Analysis result.</returns>
    /// <remarks>
    /// <para>
    /// Specialized for understanding structured documents like:
    /// - PDF pages and scanned documents
    /// - Charts and graphs
    /// - Tables and spreadsheets
    /// - Forms and invoices
    /// </para>
    /// </remarks>
    string AnalyzeDocument(
        Tensor<T> documentImage,
        string analysisType = "summary",
        string? additionalPrompt = null);

    /// <summary>
    /// Extracts structured data from an image.
    /// </summary>
    /// <param name="image">The input image.</param>
    /// <param name="schema">JSON schema describing expected output structure.</param>
    /// <returns>Extracted data as JSON string.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Get structured data from images!
    ///
    /// Example schema: {"name": "string", "price": "number", "in_stock": "boolean"}
    /// From a product image, extracts: {"name": "Widget", "price": 29.99, "in_stock": true}
    /// </para>
    /// </remarks>
    string ExtractStructuredData(
        Tensor<T> image,
        string schema);

    /// <summary>
    /// Generates code from a UI screenshot.
    /// </summary>
    /// <param name="uiScreenshot">Screenshot of a user interface.</param>
    /// <param name="targetFramework">Target framework: "html_css", "react", "flutter", "swiftui".</param>
    /// <param name="additionalInstructions">Optional styling or functionality instructions.</param>
    /// <returns>Generated code.</returns>
    string GenerateCodeFromUI(
        Tensor<T> uiScreenshot,
        string targetFramework = "html_css",
        string? additionalInstructions = null);

    /// <summary>
    /// Performs visual reasoning tasks.
    /// </summary>
    /// <param name="image">The input image.</param>
    /// <param name="reasoningTask">Task type: "count", "compare", "spatial", "temporal", "causal".</param>
    /// <param name="question">Specific question for the reasoning task.</param>
    /// <returns>Reasoning result with explanation.</returns>
    (string Answer, string Explanation) VisualReasoning(
        Tensor<T> image,
        string reasoningTask,
        string question);

    /// <summary>
    /// Describes an image with specified style and detail level.
    /// </summary>
    /// <param name="image">The input image.</param>
    /// <param name="style">Description style: "factual", "poetic", "technical", "accessibility".</param>
    /// <param name="detailLevel">Detail level: "low", "medium", "high".</param>
    /// <returns>Generated description.</returns>
    string DescribeImage(
        Tensor<T> image,
        string style = "factual",
        string detailLevel = "medium");

    /// <summary>
    /// Identifies and locates objects in an image with bounding boxes.
    /// </summary>
    /// <param name="image">The input image.</param>
    /// <param name="objectQuery">Optional specific objects to find, or null for all objects.</param>
    /// <returns>List of detected objects with bounding boxes and confidence scores.</returns>
    IEnumerable<(string Label, T Confidence, int X, int Y, int Width, int Height)> DetectObjects(
        Tensor<T> image,
        string? objectQuery = null);

    /// <summary>
    /// Answers a visual question with confidence score.
    /// </summary>
    /// <param name="image">The input image.</param>
    /// <param name="question">Question about the image.</param>
    /// <returns>Answer and confidence score.</returns>
    (string Answer, T Confidence) AnswerVisualQuestion(
        Tensor<T> image,
        string question);

    /// <summary>
    /// Compares two images and describes their differences.
    /// </summary>
    /// <param name="image1">First image.</param>
    /// <param name="image2">Second image.</param>
    /// <param name="comparisonType">Type: "visual", "semantic", "detailed".</param>
    /// <returns>Comparison description.</returns>
    string CompareImages(
        Tensor<T> image1,
        Tensor<T> image2,
        string comparisonType = "detailed");

    /// <summary>
    /// Generates image editing instructions based on a modification request.
    /// </summary>
    /// <param name="image">The original image.</param>
    /// <param name="editRequest">Description of desired edit.</param>
    /// <returns>Structured editing instructions.</returns>
    string GenerateEditInstructions(
        Tensor<T> image,
        string editRequest);

    /// <summary>
    /// Performs OCR with layout understanding.
    /// </summary>
    /// <param name="image">Image containing text.</param>
    /// <param name="preserveLayout">Whether to preserve spatial layout in output.</param>
    /// <returns>Extracted text with optional layout information.</returns>
    (string Text, Dictionary<string, object>? LayoutInfo) ExtractText(
        Tensor<T> image,
        bool preserveLayout = false);

    /// <summary>
    /// Analyzes a chart or graph and extracts data.
    /// </summary>
    /// <param name="chartImage">Image of a chart or graph.</param>
    /// <returns>Chart analysis including type, data points, and interpretation.</returns>
    (string ChartType, Dictionary<string, object> Data, string Interpretation) AnalyzeChart(
        Tensor<T> chartImage);

    /// <summary>
    /// Generates a creative story or narrative based on an image.
    /// </summary>
    /// <param name="image">The inspiring image.</param>
    /// <param name="genre">Story genre: "fantasy", "mystery", "romance", "scifi", "general".</param>
    /// <param name="length">Approximate length: "short", "medium", "long".</param>
    /// <returns>Generated story.</returns>
    string GenerateStory(
        Tensor<T> image,
        string genre = "general",
        string length = "medium");

    /// <summary>
    /// Evaluates image quality and provides improvement suggestions.
    /// </summary>
    /// <param name="image">The image to evaluate.</param>
    /// <returns>Quality assessment with scores and suggestions.</returns>
    (Dictionary<string, T> QualityScores, IEnumerable<string> Suggestions) EvaluateImageQuality(
        Tensor<T> image);

    /// <summary>
    /// Identifies potential safety concerns in an image.
    /// </summary>
    /// <param name="image">The image to analyze.</param>
    /// <returns>Safety assessment with categories and confidence levels.</returns>
    Dictionary<string, (bool IsFlagged, T Confidence)> SafetyCheck(
        Tensor<T> image);

    /// <summary>
    /// Gets attention weights showing which image regions influenced the response.
    /// </summary>
    /// <param name="image">The input image.</param>
    /// <param name="prompt">The prompt used.</param>
    /// <returns>Attention map tensor [height, width] showing importance weights.</returns>
    Tensor<T> GetAttentionMap(
        Tensor<T> image,
        string prompt);
}
