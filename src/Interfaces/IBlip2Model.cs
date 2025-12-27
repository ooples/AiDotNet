using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for BLIP-2 (Bootstrapped Language-Image Pre-training 2) models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// BLIP-2 is a more efficient and powerful successor to BLIP that uses a Q-Former
/// (Querying Transformer) to bridge frozen image encoders with frozen large language models.
/// This architecture enables better vision-language understanding with significantly
/// less training compute.
/// </para>
/// <para><b>For Beginners:</b> BLIP-2 is like having a smart translator between images and language!
///
/// Key innovation - the Q-Former:
/// - Uses special "query tokens" to ask questions about the image
/// - These queries learn to extract the most useful visual information
/// - The extracted features then connect to powerful language models (LLMs)
///
/// Why BLIP-2 is special:
/// - Uses frozen (pre-trained) image encoders like ViT-G
/// - Uses frozen LLMs like OPT or Flan-T5
/// - Only trains the small Q-Former bridge (much cheaper!)
/// - Gets state-of-the-art results with less compute
///
/// Use cases (same as BLIP but better):
/// - More accurate image captioning
/// - Better visual question answering
/// - More nuanced image-text understanding
/// - Can leverage larger LLMs for better generation
/// </para>
/// </remarks>
public interface IBlip2Model<T> : IMultimodalEmbedding<T>
{
    /// <summary>
    /// Gets the number of learnable query tokens used by the Q-Former.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The query tokens are learnable embeddings that interact with the frozen
    /// image encoder through cross-attention to extract visual features.
    /// Typically 32 queries are used.
    /// </para>
    /// </remarks>
    int NumQueryTokens { get; }

    /// <summary>
    /// Gets the type of language model backend used for generation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// BLIP-2 can use different LLM backends:
    /// - OPT (Open Pre-trained Transformer) - decoder-only
    /// - Flan-T5 - encoder-decoder
    /// The choice affects generation capabilities and quality.
    /// </para>
    /// </remarks>
    string LanguageModelType { get; }

    /// <summary>
    /// Extracts visual features using the Q-Former's learnable queries.
    /// </summary>
    /// <param name="image">The preprocessed image tensor with shape [channels, height, width].</param>
    /// <returns>Query output features with shape [numQueries, queryDim].</returns>
    /// <remarks>
    /// <para>
    /// The Q-Former uses cross-attention between learnable query tokens and
    /// the frozen image encoder output to extract query_num visual features.
    /// These features are then projected to match the LLM's input dimension.
    /// </para>
    /// <para><b>For Beginners:</b> Think of this as asking 32 questions about the image!
    ///
    /// Process:
    /// 1. Image goes through frozen ViT encoder -> patch features
    /// 2. Query tokens attend to patch features via cross-attention
    /// 3. Each query learns to focus on different aspects
    /// 4. Output: 32 feature vectors summarizing the image
    ///
    /// These 32 features are what gets sent to the language model.
    /// </para>
    /// </remarks>
    Tensor<T> ExtractQFormerFeatures(Tensor<T> image);

    /// <summary>
    /// Generates a caption for an image using the LLM backend.
    /// </summary>
    /// <param name="image">The preprocessed image tensor.</param>
    /// <param name="prompt">Optional prompt to guide generation (e.g., "a photo of").</param>
    /// <param name="maxLength">Maximum number of tokens to generate.</param>
    /// <param name="numBeams">Number of beams for beam search.</param>
    /// <param name="temperature">Sampling temperature (lower = more deterministic).</param>
    /// <returns>The generated caption.</returns>
    /// <remarks>
    /// <para>
    /// Uses the Q-Former to extract visual features, projects them to the LLM space,
    /// and then uses the LLM to generate text conditioned on these visual tokens.
    /// </para>
    /// <para><b>For Beginners:</b> This generates descriptions using a powerful language model!
    ///
    /// The prompt helps guide the style:
    /// - "a photo of" -> descriptive captions
    /// - "Question: What is this? Answer:" -> Q&amp;A style
    /// - No prompt -> model's default behavior
    ///
    /// Temperature controls randomness:
    /// - 0.0-0.3: Very focused, deterministic
    /// - 0.7-1.0: More creative, varied
    /// </para>
    /// </remarks>
    string GenerateCaption(
        Tensor<T> image,
        string? prompt = null,
        int maxLength = 30,
        int numBeams = 5,
        double temperature = 1.0);

    /// <summary>
    /// Generates multiple diverse captions for an image.
    /// </summary>
    /// <param name="image">The preprocessed image tensor.</param>
    /// <param name="numCaptions">Number of captions to generate.</param>
    /// <param name="prompt">Optional prompt to guide generation.</param>
    /// <param name="maxLength">Maximum length per caption.</param>
    /// <param name="temperature">Sampling temperature for diversity.</param>
    /// <param name="topP">Nucleus sampling probability threshold.</param>
    /// <returns>Collection of generated captions with their log probabilities.</returns>
    /// <remarks>
    /// <para>
    /// Uses nucleus (top-p) sampling with temperature to generate diverse captions.
    /// Returns captions with their generation scores for ranking.
    /// </para>
    /// </remarks>
    IEnumerable<(string Caption, T Score)> GenerateCaptions(
        Tensor<T> image,
        int numCaptions = 5,
        string? prompt = null,
        int maxLength = 30,
        double temperature = 0.9,
        double topP = 0.95);

    /// <summary>
    /// Answers a question about an image using the LLM backend.
    /// </summary>
    /// <param name="image">The preprocessed image tensor.</param>
    /// <param name="question">The question to answer about the image.</param>
    /// <param name="maxLength">Maximum answer length.</param>
    /// <returns>The generated answer.</returns>
    /// <remarks>
    /// <para>
    /// Formats the question appropriately for the LLM backend and generates
    /// an answer conditioned on both the visual features and the question.
    /// BLIP-2's LLM backend typically provides more detailed and accurate answers
    /// than BLIP's decoder.
    /// </para>
    /// <para><b>For Beginners:</b> Ask any question about an image!
    ///
    /// BLIP-2 is better at VQA because:
    /// - Uses a powerful LLM (OPT/Flan-T5) for generation
    /// - LLM has more world knowledge
    /// - Can give more detailed, reasoned answers
    ///
    /// Examples:
    /// - "What is the person doing?" -> "The person is riding a bicycle down a street"
    /// - "What color is the car?" -> "The car is red"
    /// - "Is it raining?" -> "No, it appears to be a sunny day"
    /// </para>
    /// </remarks>
    string AnswerQuestion(Tensor<T> image, string question, int maxLength = 30);

    /// <summary>
    /// Computes image-text matching score using the Q-Former's ITM head.
    /// </summary>
    /// <param name="image">The preprocessed image tensor.</param>
    /// <param name="text">The text to match against the image.</param>
    /// <returns>Matching probability between 0 and 1.</returns>
    /// <remarks>
    /// <para>
    /// Uses the Q-Former's image-text matching head which applies cross-attention
    /// between query features and text features to determine if they match.
    /// This is trained with hard negative mining for better discrimination.
    /// </para>
    /// </remarks>
    T ComputeImageTextMatch(Tensor<T> image, string text);

    /// <summary>
    /// Computes image-text contrastive similarity using Q-Former features.
    /// </summary>
    /// <param name="image">The preprocessed image tensor.</param>
    /// <param name="text">The text to compare.</param>
    /// <returns>Contrastive similarity score.</returns>
    /// <remarks>
    /// <para>
    /// Uses the Q-Former's image-text contrastive (ITC) learning objective.
    /// Computes similarity between the CLS token of query outputs and text features.
    /// Faster than ITM but less accurate for fine-grained matching.
    /// </para>
    /// <para><b>For Beginners:</b> Quick similarity check between image and text!
    ///
    /// Difference from ITM (Image-Text Matching):
    /// - ITC: Fast, uses embedding similarity (like CLIP)
    /// - ITM: Slower, uses cross-attention for deeper analysis
    ///
    /// Use ITC for:
    /// - Large-scale retrieval (searching millions of images)
    /// - Quick filtering before detailed matching
    ///
    /// Use ITM for:
    /// - Final ranking of candidates
    /// - When accuracy matters more than speed
    /// </para>
    /// </remarks>
    T ComputeContrastiveSimilarity(Tensor<T> image, string text);

    /// <summary>
    /// Performs visual grounding to locate objects described in text.
    /// </summary>
    /// <param name="image">The preprocessed image tensor.</param>
    /// <param name="description">Text description of the object to locate.</param>
    /// <returns>Bounding box coordinates [x1, y1, x2, y2] normalized to [0, 1].</returns>
    /// <remarks>
    /// <para>
    /// Uses the Q-Former's attention patterns to identify which image regions
    /// correspond to the text description. Returns a bounding box for the
    /// most likely region.
    /// </para>
    /// <para><b>For Beginners:</b> Find where something is in an image!
    ///
    /// Given text like "the red car on the left", this finds and returns
    /// the bounding box coordinates for that object.
    ///
    /// The output is normalized coordinates:
    /// - [0, 0, 1, 1] would be the entire image
    /// - [0.5, 0.5, 1, 1] would be the bottom-right quarter
    ///
    /// Use cases:
    /// - Object detection from natural language
    /// - Referring expression comprehension
    /// - Interactive image editing ("remove the person on the right")
    /// </para>
    /// </remarks>
    Vector<T> GroundText(Tensor<T> image, string description);

    /// <summary>
    /// Generates text conditioned on both image and text context (instructed generation).
    /// </summary>
    /// <param name="image">The preprocessed image tensor.</param>
    /// <param name="instruction">The instruction or context for generation.</param>
    /// <param name="maxLength">Maximum generation length.</param>
    /// <returns>The generated response.</returns>
    /// <remarks>
    /// <para>
    /// Enables instruction-following behavior where the model generates text
    /// based on both visual input and textual instructions. This is particularly
    /// powerful with instruction-tuned LLM backends like Flan-T5.
    /// </para>
    /// <para><b>For Beginners:</b> Give instructions about what to do with the image!
    ///
    /// Examples:
    /// - "Describe this image in detail" -> Detailed description
    /// - "List all the objects in this image" -> Bulleted list
    /// - "Write a story based on this image" -> Creative narrative
    /// - "Explain what is happening" -> Scene analysis
    ///
    /// This is more flexible than simple captioning because you can
    /// customize the output format and content through instructions.
    /// </para>
    /// </remarks>
    string GenerateWithInstruction(Tensor<T> image, string instruction, int maxLength = 100);

    /// <summary>
    /// Performs zero-shot image classification using text prompts.
    /// </summary>
    /// <param name="image">The preprocessed image tensor.</param>
    /// <param name="classLabels">The candidate class labels.</param>
    /// <param name="useItm">If true, use ITM for scoring; if false, use ITC.</param>
    /// <returns>Dictionary mapping class labels to probability scores.</returns>
    /// <remarks>
    /// <para>
    /// Classifies images into categories without any training on those specific categories.
    /// Can use either ITC (faster) or ITM (more accurate) for scoring.
    /// </para>
    /// </remarks>
    Dictionary<string, T> ZeroShotClassify(
        Tensor<T> image,
        IEnumerable<string> classLabels,
        bool useItm = false);

    /// <summary>
    /// Retrieves the most relevant images for a text query.
    /// </summary>
    /// <param name="query">The text query.</param>
    /// <param name="imageFeatures">Pre-computed Q-Former features for images.</param>
    /// <param name="topK">Number of results to return.</param>
    /// <param name="useItmReranking">Whether to rerank top results using ITM.</param>
    /// <param name="rerankTopN">Number of candidates to rerank with ITM.</param>
    /// <returns>Indices of top-K matching images with scores.</returns>
    /// <remarks>
    /// <para>
    /// Two-stage retrieval:
    /// 1. Fast ITC-based retrieval to get candidates
    /// 2. Optional ITM reranking for higher precision
    /// </para>
    /// </remarks>
    IEnumerable<(int Index, T Score)> RetrieveImages(
        string query,
        IEnumerable<Tensor<T>> imageFeatures,
        int topK = 10,
        bool useItmReranking = true,
        int rerankTopN = 100);
}
