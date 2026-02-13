using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Metrics;

/// <summary>
/// CLIPScore metric for evaluating text-image alignment and image quality.
/// </summary>
/// <remarks>
/// <para>
/// CLIPScore measures how well an image matches a text description (or reference image)
/// using CLIP embeddings. It's widely used for evaluating text-to-image generation models.
/// </para>
/// <para>
/// Two main variants:
/// - CLIPScore (text-image): Measures alignment between generated images and text prompts
/// - RefCLIPScore (image-image): Measures similarity between generated and reference images
/// </para>
/// <para>
/// Typical CLIPScore values (0-100 scale):
/// - &gt;30: Excellent alignment (image matches text well)
/// - 25-30: Good alignment
/// - 20-25: Moderate alignment
/// - &lt;20: Poor alignment
/// </para>
/// <para>
/// Based on "CLIPScore: A Reference-free Evaluation Metric for Image Captioning"
/// by Hessel et al. (2021)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for computations</typeparam>
public class CLIPScore<T> where T : struct
{
    private readonly INumericOperations<T> _numOps;
    private readonly IMultimodalEmbedding<T> _clipModel;

    /// <summary>
    /// Gets the CLIP model used for computing embeddings.
    /// </summary>
    public IMultimodalEmbedding<T> ClipModel => _clipModel;

    /// <summary>
    /// Initializes a new instance of CLIPScore calculator.
    /// </summary>
    /// <param name="clipModel">A CLIP model implementing IMultimodalEmbedding interface</param>
    public CLIPScore(IMultimodalEmbedding<T> clipModel)
    {
        _clipModel = clipModel ?? throw new ArgumentNullException(nameof(clipModel),
            "A CLIP model is required for CLIPScore computation");
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Computes CLIPScore between an image and a text description.
    /// </summary>
    /// <param name="imageData">Image data as a flattened array (CHW format, normalized to [0,1] or [-1,1])</param>
    /// <param name="text">The text description to compare against</param>
    /// <returns>CLIPScore on a 0-100 scale. Higher is better.</returns>
    public double ComputeTextImageScore(double[] imageData, string text)
    {
        if (imageData == null || imageData.Length == 0)
        {
            throw new ArgumentException("Image data cannot be empty", nameof(imageData));
        }
        if (string.IsNullOrWhiteSpace(text))
        {
            throw new ArgumentException("Text cannot be empty", nameof(text));
        }

        // Encode image and text
        var imageEmbedding = _clipModel.EncodeImage(imageData);
        var textEmbedding = _clipModel.EncodeText(text);

        // Compute cosine similarity
        T similarity = _clipModel.ComputeSimilarity(imageEmbedding, textEmbedding);

        // Scale to 0-100 range
        // CLIP similarity is typically in [-1, 1], with good matches being > 0.2
        double simDouble = _numOps.ToDouble(similarity);
        double score = Math.Max(0, simDouble) * 100.0;

        return score;
    }

    /// <summary>
    /// Computes CLIPScore for a batch of images and their corresponding text descriptions.
    /// </summary>
    /// <param name="images">List of image data arrays</param>
    /// <param name="texts">List of text descriptions (same length as images)</param>
    /// <returns>Average CLIPScore across all pairs.</returns>
    public double ComputeTextImageScoreBatch(IList<double[]> images, IList<string> texts)
    {
        if (images.Count != texts.Count)
        {
            throw new ArgumentException("Number of images must match number of texts");
        }

        if (images.Count == 0)
        {
            return 0.0;
        }

        double totalScore = 0.0;
        for (int i = 0; i < images.Count; i++)
        {
            totalScore += ComputeTextImageScore(images[i], texts[i]);
        }

        return totalScore / images.Count;
    }

    /// <summary>
    /// Computes RefCLIPScore between a generated image and a reference image.
    /// </summary>
    /// <param name="generatedImageData">Generated image data</param>
    /// <param name="referenceImageData">Reference image data</param>
    /// <returns>RefCLIPScore on a 0-100 scale. Higher is better.</returns>
    public double ComputeImageImageScore(double[] generatedImageData, double[] referenceImageData)
    {
        if (generatedImageData == null || generatedImageData.Length == 0)
        {
            throw new ArgumentException("Generated image data cannot be empty", nameof(generatedImageData));
        }
        if (referenceImageData == null || referenceImageData.Length == 0)
        {
            throw new ArgumentException("Reference image data cannot be empty", nameof(referenceImageData));
        }

        // Encode both images
        var genEmbedding = _clipModel.EncodeImage(generatedImageData);
        var refEmbedding = _clipModel.EncodeImage(referenceImageData);

        // Compute cosine similarity
        T similarity = _clipModel.ComputeSimilarity(genEmbedding, refEmbedding);

        // Scale to 0-100 range
        double simDouble = _numOps.ToDouble(similarity);
        double score = Math.Max(0, simDouble) * 100.0;

        return score;
    }

    /// <summary>
    /// Computes combined CLIPScore using both text-image and image-image similarity.
    /// </summary>
    /// <param name="generatedImageData">Generated image data</param>
    /// <param name="text">Text description used for generation</param>
    /// <param name="referenceImageData">Reference image (optional)</param>
    /// <param name="textWeight">Weight for text-image score (default 0.7)</param>
    /// <returns>Combined CLIPScore on a 0-100 scale.</returns>
    public double ComputeCombinedScore(
        double[] generatedImageData,
        string text,
        double[]? referenceImageData = null,
        double textWeight = 0.7)
    {
        double textScore = ComputeTextImageScore(generatedImageData, text);

        if (referenceImageData == null)
        {
            return textScore;
        }

        double imageScore = ComputeImageImageScore(generatedImageData, referenceImageData);

        // Weighted combination
        return textWeight * textScore + (1.0 - textWeight) * imageScore;
    }

    /// <summary>
    /// Computes CLIPScore for image captioning evaluation.
    /// </summary>
    /// <param name="imageData">Image data</param>
    /// <param name="candidateCaption">Generated caption to evaluate</param>
    /// <param name="referenceCaptions">Reference captions (optional, for RefCLIPScore)</param>
    /// <returns>CLIPScore for the caption.</returns>
    public double ComputeCaptionScore(
        double[] imageData,
        string candidateCaption,
        IEnumerable<string>? referenceCaptions = null)
    {
        // Compute image-caption CLIPScore
        double imageTextScore = ComputeTextImageScore(imageData, candidateCaption);

        if (referenceCaptions == null || !referenceCaptions.Any())
        {
            return imageTextScore;
        }

        // Compute text-text similarity with reference captions
        var candidateEmbedding = _clipModel.EncodeText(candidateCaption);
        double maxTextSim = 0.0;

        foreach (string refCaption in referenceCaptions)
        {
            var refEmbedding = _clipModel.EncodeText(refCaption);
            T similarity = _clipModel.ComputeSimilarity(candidateEmbedding, refEmbedding);
            double simDouble = _numOps.ToDouble(similarity);
            maxTextSim = Math.Max(maxTextSim, simDouble);
        }

        // Combine image-text score with text-text similarity
        // This helps ensure the caption is both relevant to the image AND similar to reference captions
        double textRefScore = maxTextSim * 100.0;

        // Harmonic mean of the two scores (rewards both aspects)
        if (imageTextScore <= 0 || textRefScore <= 0)
        {
            return 0.0;
        }

        return 2.0 * imageTextScore * textRefScore / (imageTextScore + textRefScore);
    }

    /// <summary>
    /// Computes directional similarity for image editing evaluation.
    /// </summary>
    /// <remarks>
    /// This measures whether the edit direction in image space matches the edit direction in text space.
    /// Used for evaluating text-guided image editing models.
    /// </remarks>
    /// <param name="sourceImageData">Original image data</param>
    /// <param name="editedImageData">Edited image data</param>
    /// <param name="sourceText">Source text description</param>
    /// <param name="editText">Target/edit text description</param>
    /// <returns>Directional similarity score on a 0-100 scale.</returns>
    public double ComputeDirectionalSimilarity(
        double[] sourceImageData,
        double[] editedImageData,
        string sourceText,
        string editText)
    {
        // Get embeddings
        var srcImgEmb = _clipModel.EncodeImage(sourceImageData);
        var editImgEmb = _clipModel.EncodeImage(editedImageData);
        var srcTextEmb = _clipModel.EncodeText(sourceText);
        var editTextEmb = _clipModel.EncodeText(editText);

        // Compute direction vectors
        var imgDirection = SubtractVectors(editImgEmb, srcImgEmb);
        var textDirection = SubtractVectors(editTextEmb, srcTextEmb);

        // Normalize direction vectors
        imgDirection = NormalizeVector(imgDirection);
        textDirection = NormalizeVector(textDirection);

        // Compute cosine similarity between directions
        T similarity = ComputeCosineSimilarity(imgDirection, textDirection);
        double simDouble = _numOps.ToDouble(similarity);

        // Scale to 0-100 range
        return (simDouble + 1.0) * 50.0; // Maps [-1, 1] to [0, 100]
    }

    /// <summary>
    /// Computes CLIPScore improvement between before and after images.
    /// </summary>
    /// <param name="beforeImageData">Image before processing</param>
    /// <param name="afterImageData">Image after processing</param>
    /// <param name="text">Text description to measure alignment against</param>
    /// <returns>Improvement in CLIPScore (positive = after is better aligned).</returns>
    public double ComputeScoreImprovement(
        double[] beforeImageData,
        double[] afterImageData,
        string text)
    {
        double beforeScore = ComputeTextImageScore(beforeImageData, text);
        double afterScore = ComputeTextImageScore(afterImageData, text);

        return afterScore - beforeScore;
    }

    /// <summary>
    /// Subtracts one vector from another.
    /// </summary>
    private Vector<T> SubtractVectors(Vector<T> a, Vector<T> b)
    {
        var result = new Vector<T>(a.Length);
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = _numOps.Subtract(a[i], b[i]);
        }
        return result;
    }

    /// <summary>
    /// Normalizes a vector to unit length.
    /// </summary>
    private Vector<T> NormalizeVector(Vector<T> v)
    {
        T normSq = _numOps.Zero;
        for (int i = 0; i < v.Length; i++)
        {
            normSq = _numOps.Add(normSq, _numOps.Multiply(v[i], v[i]));
        }

        double normDouble = Math.Sqrt(_numOps.ToDouble(normSq));
        if (normDouble < 1e-10)
        {
            return v;
        }

        T normInv = _numOps.FromDouble(1.0 / normDouble);
        var result = new Vector<T>(v.Length);
        for (int i = 0; i < v.Length; i++)
        {
            result[i] = _numOps.Multiply(v[i], normInv);
        }
        return result;
    }

    /// <summary>
    /// Computes cosine similarity between two vectors.
    /// </summary>
    private T ComputeCosineSimilarity(Vector<T> a, Vector<T> b)
    {
        T dot = _numOps.Zero;
        T normASq = _numOps.Zero;
        T normBSq = _numOps.Zero;

        for (int i = 0; i < a.Length; i++)
        {
            dot = _numOps.Add(dot, _numOps.Multiply(a[i], b[i]));
            normASq = _numOps.Add(normASq, _numOps.Multiply(a[i], a[i]));
            normBSq = _numOps.Add(normBSq, _numOps.Multiply(b[i], b[i]));
        }

        T denominator = _numOps.Sqrt(_numOps.Multiply(normASq, normBSq));

        double denomDouble = _numOps.ToDouble(denominator);
        if (denomDouble < 1e-10)
        {
            return _numOps.Zero;
        }

        return _numOps.Divide(dot, denominator);
    }
}

/// <summary>
/// Aesthetic Score metric using CLIP for evaluating image aesthetics.
/// </summary>
/// <remarks>
/// <para>
/// Aesthetic Score uses CLIP embeddings trained on aesthetic preference data to predict
/// how visually appealing an image is. This is commonly used in image generation to
/// filter or rank outputs by aesthetic quality.
/// </para>
/// <para>
/// Typical aesthetic scores (1-10 scale):
/// - &gt;7: High aesthetic quality
/// - 5-7: Average aesthetic quality
/// - &lt;5: Low aesthetic quality
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for computations</typeparam>
public class AestheticScore<T> where T : struct
{
    private readonly INumericOperations<T> _numOps;
    private readonly IMultimodalEmbedding<T> _clipModel;
    private readonly Vector<T>? _aestheticWeights;
    private readonly string[] _positivePrompts;
    private readonly string[] _negativePrompts;

    /// <summary>
    /// Default positive aesthetic prompts used for zero-shot aesthetic scoring.
    /// </summary>
    public static readonly string[] DefaultPositivePrompts = new[]
    {
        "a beautiful photograph",
        "an aesthetically pleasing image",
        "a high quality professional photo",
        "an artistic masterpiece",
        "a visually stunning image"
    };

    /// <summary>
    /// Default negative aesthetic prompts used for zero-shot aesthetic scoring.
    /// </summary>
    public static readonly string[] DefaultNegativePrompts = new[]
    {
        "a low quality photograph",
        "an ugly image",
        "a poorly composed photo",
        "an amateur snapshot",
        "a visually unappealing image"
    };

    /// <summary>
    /// Initializes a new instance of AestheticScore calculator.
    /// </summary>
    /// <param name="clipModel">A CLIP model for computing embeddings.</param>
    /// <param name="aestheticWeights">Pre-trained aesthetic prediction weights (optional).</param>
    /// <param name="positivePrompts">Custom positive aesthetic prompts for zero-shot scoring (optional, uses defaults if null).</param>
    /// <param name="negativePrompts">Custom negative aesthetic prompts for zero-shot scoring (optional, uses defaults if null).</param>
    public AestheticScore(
        IMultimodalEmbedding<T> clipModel,
        Vector<T>? aestheticWeights = null,
        string[]? positivePrompts = null,
        string[]? negativePrompts = null)
    {
        Guard.NotNull(clipModel);
        _clipModel = clipModel;
        _numOps = MathHelper.GetNumericOperations<T>();
        _aestheticWeights = aestheticWeights;
        _positivePrompts = positivePrompts ?? DefaultPositivePrompts;
        _negativePrompts = negativePrompts ?? DefaultNegativePrompts;

        // Validate prompt arrays are not empty to avoid divide-by-zero in ComputeZeroShot
        if (_positivePrompts.Length == 0)
        {
            throw new ArgumentException("Positive prompts array cannot be empty.", nameof(positivePrompts));
        }

        if (_negativePrompts.Length == 0)
        {
            throw new ArgumentException("Negative prompts array cannot be empty.", nameof(negativePrompts));
        }
    }

    /// <summary>
    /// Computes aesthetic score for an image.
    /// </summary>
    /// <param name="imageData">Image data as a flattened array</param>
    /// <returns>Aesthetic score on a 1-10 scale. Higher is better.</returns>
    public double Compute(double[] imageData)
    {
        if (_aestheticWeights != null)
        {
            // Use trained aesthetic predictor
            return ComputeWithTrainedWeights(imageData);
        }

        // Use zero-shot approach with positive/negative prompts
        return ComputeZeroShot(imageData);
    }

    /// <summary>
    /// Computes aesthetic score using trained weights.
    /// </summary>
    private double ComputeWithTrainedWeights(double[] imageData)
    {
        var imageEmbedding = _clipModel.EncodeImage(imageData);

        // Linear projection: score = weights^T * embedding
        T score = _numOps.Zero;
        for (int i = 0; i < Math.Min(_aestheticWeights!.Length, imageEmbedding.Length); i++)
        {
            score = _numOps.Add(score, _numOps.Multiply(_aestheticWeights[i], imageEmbedding[i]));
        }

        // Sigmoid to map to [0, 1] then scale to [1, 10]
        double scoreDouble = _numOps.ToDouble(score);
        double sigmoid = 1.0 / (1.0 + Math.Exp(-scoreDouble));
        return 1.0 + sigmoid * 9.0;
    }

    /// <summary>
    /// Computes aesthetic score using zero-shot comparison with aesthetic prompts.
    /// </summary>
    private double ComputeZeroShot(double[] imageData)
    {
        var imageEmbedding = _clipModel.EncodeImage(imageData);

        // Compute similarity with positive prompts
        double positiveScore = 0.0;
        foreach (string prompt in _positivePrompts)
        {
            var textEmbedding = _clipModel.EncodeText(prompt);
            T similarity = _clipModel.ComputeSimilarity(imageEmbedding, textEmbedding);
            positiveScore += _numOps.ToDouble(similarity);
        }
        positiveScore /= _positivePrompts.Length;

        // Compute similarity with negative prompts
        double negativeScore = 0.0;
        foreach (string prompt in _negativePrompts)
        {
            var textEmbedding = _clipModel.EncodeText(prompt);
            T similarity = _clipModel.ComputeSimilarity(imageEmbedding, textEmbedding);
            negativeScore += _numOps.ToDouble(similarity);
        }
        negativeScore /= _negativePrompts.Length;

        // Compute relative aesthetic score
        // Higher positive similarity + lower negative similarity = higher aesthetic score
        double aestheticDiff = positiveScore - negativeScore;

        // Map to 1-10 scale using sigmoid-like function
        // Typical values for aestheticDiff are in range [-0.3, 0.3]
        double normalized = 1.0 / (1.0 + Math.Exp(-aestheticDiff * 10));
        return 1.0 + normalized * 9.0;
    }

    /// <summary>
    /// Computes aesthetic scores for a batch of images.
    /// </summary>
    /// <param name="images">List of image data arrays</param>
    /// <returns>Array of aesthetic scores.</returns>
    public double[] ComputeBatch(IList<double[]> images)
    {
        var scores = new double[images.Count];
        for (int i = 0; i < images.Count; i++)
        {
            scores[i] = Compute(images[i]);
        }
        return scores;
    }

    /// <summary>
    /// Ranks images by aesthetic score.
    /// </summary>
    /// <param name="images">List of image data arrays</param>
    /// <returns>Indices of images sorted by aesthetic score (highest first).</returns>
    public int[] RankByAesthetic(IList<double[]> images)
    {
        var scores = ComputeBatch(images);

        return scores
            .Select((score, index) => (score, index))
            .OrderByDescending(x => x.score)
            .Select(x => x.index)
            .ToArray();
    }
}
