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
    protected static IEngine Engine => AiDotNetEngine.Current;
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
        Guard.NotNull(clipModel);
        _clipModel = clipModel;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Computes CLIPScore between an image and a text description.
    /// </summary>
    /// <param name="imageData">Image data as a flattened array (CHW format, normalized to [0,1] or [-1,1])</param>
    /// <param name="text">The text description to compare against</param>
    /// <returns>CLIPScore on a 0-100 scale. Higher is better.</returns>
    public T ComputeTextImageScore(double[] imageData, string text)
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
        return ScaleSimilarity(similarity);
    }

    /// <summary>
    /// Computes CLIPScore for a batch of images and their corresponding text descriptions.
    /// </summary>
    /// <param name="images">List of image data arrays</param>
    /// <param name="texts">List of text descriptions (same length as images)</param>
    /// <returns>Average CLIPScore across all pairs.</returns>
    public T ComputeTextImageScoreBatch(IList<double[]> images, IList<string> texts)
    {
        if (images.Count != texts.Count)
        {
            throw new ArgumentException("Number of images must match number of texts");
        }

        if (images.Count == 0)
        {
            return _numOps.Zero;
        }

        // One model call per pair (the model's own ComputeSimilarity), then one engine mean.
        var scores = images.Select((image, index) => ComputeTextImageScore(image, texts[index])).ToArray();
        return Engine.Mean(Vector<T>.Wrap(scores));
    }

    /// <summary>
    /// Computes RefCLIPScore between a generated image and a reference image.
    /// </summary>
    /// <param name="generatedImageData">Generated image data</param>
    /// <param name="referenceImageData">Reference image data</param>
    /// <returns>RefCLIPScore on a 0-100 scale. Higher is better.</returns>
    public T ComputeImageImageScore(double[] generatedImageData, double[] referenceImageData)
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
        return ScaleSimilarity(similarity);
    }

    /// <summary>
    /// Computes combined CLIPScore using both text-image and image-image similarity.
    /// </summary>
    /// <param name="generatedImageData">Generated image data</param>
    /// <param name="text">Text description used for generation</param>
    /// <param name="referenceImageData">Reference image (optional)</param>
    /// <param name="textWeight">Weight for text-image score (default 0.7)</param>
    /// <returns>Combined CLIPScore on a 0-100 scale.</returns>
    public T ComputeCombinedScore(
        double[] generatedImageData,
        string text,
        double[]? referenceImageData = null,
        double textWeight = 0.7)
    {
        T textScore = ComputeTextImageScore(generatedImageData, text);

        if (referenceImageData == null)
        {
            return textScore;
        }

        T imageScore = ComputeImageImageScore(generatedImageData, referenceImageData);

        // Weighted combination
        return _numOps.Add(
            _numOps.Multiply(_numOps.FromDouble(textWeight), textScore),
            _numOps.Multiply(_numOps.FromDouble(1.0 - textWeight), imageScore));
    }

    /// <summary>
    /// Computes CLIPScore for image captioning evaluation.
    /// </summary>
    /// <param name="imageData">Image data</param>
    /// <param name="candidateCaption">Generated caption to evaluate</param>
    /// <param name="referenceCaptions">Reference captions (optional, for RefCLIPScore)</param>
    /// <returns>CLIPScore for the caption.</returns>
    public T ComputeCaptionScore(
        double[] imageData,
        string candidateCaption,
        IEnumerable<string>? referenceCaptions = null)
    {
        // Compute image-caption CLIPScore
        T imageTextScore = ComputeTextImageScore(imageData, candidateCaption);

        if (referenceCaptions == null || !referenceCaptions.Any())
        {
            return imageTextScore;
        }

        // Compute text-text similarity with reference captions (one model call per reference), then
        // RefCLIPScore's max(max_r cos(c, r), 0) as one engine reduction.
        var candidateEmbedding = _clipModel.EncodeText(candidateCaption);
        var similarities = referenceCaptions
            .Select(refCaption => _clipModel.ComputeSimilarity(candidateEmbedding, _clipModel.EncodeText(refCaption)))
            .ToArray();
        T maxTextSim = MathHelper.Max(_numOps.Zero, Engine.TensorMaxValue(Vector<T>.Wrap(similarities).AsTensor()));

        // Combine image-text score with text-text similarity
        // This helps ensure the caption is both relevant to the image AND similar to reference captions
        T textRefScore = _numOps.Multiply(maxTextSim, _numOps.FromDouble(100.0));

        // Harmonic mean of the two scores (rewards both aspects)
        if (!_numOps.GreaterThan(imageTextScore, _numOps.Zero) || !_numOps.GreaterThan(textRefScore, _numOps.Zero))
        {
            return _numOps.Zero;
        }

        return _numOps.Divide(
            _numOps.Multiply(_numOps.FromDouble(2.0), _numOps.Multiply(imageTextScore, textRefScore)),
            _numOps.Add(imageTextScore, textRefScore));
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
    public T ComputeDirectionalSimilarity(
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
        imgDirection = NormalizeDirection(imgDirection);
        textDirection = NormalizeDirection(textDirection);

        // Compute cosine similarity between directions
        T similarity = CosineSimilarity(imgDirection, textDirection);

        // Scale to 0-100 range: maps [-1, 1] to [0, 100]
        return _numOps.Multiply(_numOps.Add(similarity, _numOps.One), _numOps.FromDouble(50.0));
    }

    /// <summary>
    /// Computes CLIPScore improvement between before and after images.
    /// </summary>
    /// <param name="beforeImageData">Image before processing</param>
    /// <param name="afterImageData">Image after processing</param>
    /// <param name="text">Text description to measure alignment against</param>
    /// <returns>Improvement in CLIPScore (positive = after is better aligned).</returns>
    public T ComputeScoreImprovement(
        double[] beforeImageData,
        double[] afterImageData,
        string text)
    {
        T beforeScore = ComputeTextImageScore(beforeImageData, text);
        T afterScore = ComputeTextImageScore(afterImageData, text);

        return _numOps.Subtract(afterScore, beforeScore);
    }

    /// <summary>
    /// Maps a cosine similarity to CLIPScore's 0-100 scale: 100 * max(cos, 0) (Hessel et al. 2021; torchmetrics).
    /// </summary>
    private T ScaleSimilarity(T similarity)
    {
        return _numOps.Multiply(MathHelper.Max(similarity, _numOps.Zero), _numOps.FromDouble(100.0));
    }

    /// <summary>
    /// Scales a direction to unit length; a direction shorter than 1e-10 is returned unchanged.
    /// </summary>
    private Vector<T> NormalizeDirection(Vector<T> direction)
    {
        T norm = _numOps.Sqrt(Engine.DotProduct(direction, direction));
        if (_numOps.LessThan(norm, _numOps.FromDouble(1e-10)))
        {
            return direction;
        }

        return Engine.Divide(direction, norm);
    }

    /// <summary>
    /// Cosine similarity clamped to [-1, 1]; 0 when the norm product is at most 1e-10.
    /// </summary>
    private T CosineSimilarity(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException(
                $"Vectors must have equal length for cosine similarity; got {a.Length} and {b.Length}.");
        }

        T dot = Engine.DotProduct(a, b);
        T denom = _numOps.Sqrt(_numOps.Multiply(Engine.DotProduct(a, a), Engine.DotProduct(b, b)));
        if (!_numOps.GreaterThan(denom, _numOps.FromDouble(1e-10)))
        {
            return _numOps.Zero;
        }

        return MathHelper.Clamp(_numOps.Divide(dot, denom), _numOps.Negate(_numOps.One), _numOps.One);
    }

    /// <summary>
    /// Subtracts one vector from another.
    /// </summary>
    private Vector<T> SubtractVectors(Vector<T> a, Vector<T> b)
    {
        return Engine.Subtract(a, b);
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
    private readonly T _aestheticBias;
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
    /// <param name="aestheticBias">Bias of the trained linear aesthetic head (optional, default 0).</param>
    public AestheticScore(
        IMultimodalEmbedding<T> clipModel,
        Vector<T>? aestheticWeights = null,
        string[]? positivePrompts = null,
        string[]? negativePrompts = null,
        T? aestheticBias = null)
    {
        Guard.NotNull(clipModel);
        _clipModel = clipModel;
        _numOps = MathHelper.GetNumericOperations<T>();
        _aestheticWeights = aestheticWeights;
        _aestheticBias = aestheticBias ?? _numOps.Zero;
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
    /// <returns>
    /// Aesthetic score. With trained weights this is the linear head's rating of the L2-normalised CLIP image
    /// embedding (the LAION aesthetic predictor's 1-10 scale); zero-shot it is mapped to [1, 10].
    /// </returns>
    public T Compute(double[] imageData)
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
    /// <remarks>
    /// Follows the LAION aesthetic predictor (Schuhmann et al., LAION-5B 2022): the CLIP image embedding is
    /// L2-normalised and passed through the linear head, score = w . (e / ||e||) + b, read directly as the rating.
    /// </remarks>
    private T ComputeWithTrainedWeights(double[] imageData)
    {
        var imageEmbedding = _clipModel.EncodeImage(imageData);

        // Linear projection: score = weights^T * normalized embedding + bias
        var aestheticWeights = _aestheticWeights ?? throw new InvalidOperationException(
            "Aesthetic weights have not been initialized.");

        if (aestheticWeights.Length != imageEmbedding.Length)
        {
            throw new InvalidOperationException(
                $"Embedding dimension mismatch: aesthetic weights have {aestheticWeights.Length} dimensions " +
                $"but image embedding has {imageEmbedding.Length} dimensions.");
        }

        var engine = AiDotNetEngine.Current;
        T norm = _numOps.Sqrt(engine.DotProduct(imageEmbedding, imageEmbedding));
        T dot = engine.DotProduct(aestheticWeights, imageEmbedding);
        T projection = _numOps.GreaterThan(norm, _numOps.Zero) ? _numOps.Divide(dot, norm) : dot;
        return _numOps.Add(projection, _aestheticBias);
    }

    /// <summary>
    /// Computes aesthetic score using zero-shot comparison with aesthetic prompts.
    /// </summary>
    private T ComputeZeroShot(double[] imageData)
    {
        var engine = AiDotNetEngine.Current;
        var imageEmbedding = _clipModel.EncodeImage(imageData);

        // Mean similarity with the positive and with the negative prompts (one model call per prompt).
        var positiveSimilarities = _positivePrompts
            .Select(prompt => _clipModel.ComputeSimilarity(imageEmbedding, _clipModel.EncodeText(prompt)))
            .ToArray();
        T positiveScore = engine.Mean(Vector<T>.Wrap(positiveSimilarities));

        var negativeSimilarities = _negativePrompts
            .Select(prompt => _clipModel.ComputeSimilarity(imageEmbedding, _clipModel.EncodeText(prompt)))
            .ToArray();
        T negativeScore = engine.Mean(Vector<T>.Wrap(negativeSimilarities));

        // Compute relative aesthetic score
        // Higher positive similarity + lower negative similarity = higher aesthetic score
        T aestheticDiff = _numOps.Subtract(positiveScore, negativeScore);

        // Map to 1-10 scale using sigmoid-like function
        // Typical values for aestheticDiff are in range [-0.3, 0.3]
        T normalized = MathHelper.Sigmoid(_numOps.Multiply(aestheticDiff, _numOps.FromDouble(10.0)));
        return _numOps.Add(_numOps.One, _numOps.Multiply(normalized, _numOps.FromDouble(9.0)));
    }

    /// <summary>
    /// Computes aesthetic scores for a batch of images.
    /// </summary>
    /// <param name="images">List of image data arrays</param>
    /// <returns>Vector of aesthetic scores, one per image.</returns>
    public Vector<T> ComputeBatch(IList<double[]> images)
    {
        return Vector<T>.Wrap(images.Select(Compute).ToArray());
    }

    /// <summary>
    /// Ranks images by aesthetic score.
    /// </summary>
    /// <param name="images">List of image data arrays</param>
    /// <returns>Indices of images sorted by aesthetic score (highest first; ties keep input order).</returns>
    public int[] RankByAesthetic(IList<double[]> images)
    {
        var scores = images.Select(Compute).ToArray();

        return scores
            .Select((score, index) => (score, index))
            .OrderByDescending(x => x.score)
            .Select(x => x.index)
            .ToArray();
    }
}
