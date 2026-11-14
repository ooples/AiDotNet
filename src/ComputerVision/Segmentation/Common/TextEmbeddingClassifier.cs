using AiDotNet.Helpers;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Classifies dense pixel embeddings against text embeddings from a CLIP-style text encoder, the way
/// open-vocabulary and referring segmentation models compare the two modalities.
/// </summary>
/// <remarks>
/// <para>
/// Both sides are L2-normalized and compared by dot product, so each logit is a cosine similarity
/// divided by a temperature: <c>logit[k, p] = cos(e_p, t_k) / tau</c>. CLIP (Radford et al. 2021)
/// learns <c>1 / tau</c> as its logit scale and clips it at 100; CAT-Seg, SAN, SED, Mask-Adapter and
/// Open-Vocabulary SAM all reuse that CLIP scale, so <c>tau = 0.01</c> is the default the options carry.
/// </para>
/// <para>
/// The norm is <c>sqrt(sum(x^2) + eps^2)</c> rather than <c>max(||x||, eps)</c>: the two agree whenever
/// <c>||x|| &gt;&gt; eps</c>, and the smooth form keeps the gradient finite for an all-zero embedding.
/// </para>
/// </remarks>
internal static class TextEmbeddingClassifier
{
    /// <summary>Squared epsilon added under the square root of every L2 norm.</summary>
    private const double NormEpsilonSquared = 1e-24;

    /// <summary>The epsilon itself: the floor on the per-slice scale, so an all-zero slice is not divided by zero.</summary>
    private const double NormEpsilon = 1e-12;

    /// <summary>Half's smallest positive normal value, the floor for an epsilon Half would flush to zero.</summary>
    private const double SmallestHalfNormal = 6.103515625e-05;

    /// <summary>
    /// The logit scale <c>1 / temperature</c> converted to T, rejected when it is not finite in T: the options
    /// validator is not generic, so a temperature like 1e-6 passes it yet its reciprocal overflows Half, and every
    /// logit would then be infinite or NaN.
    /// </summary>
    /// <exception cref="ArgumentException">The reciprocal of <paramref name="temperature"/> is not finite in T.</exception>
    internal static T LogitScale<T>(double temperature)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var scale = numOps.FromDouble(1.0 / temperature);
        if (numOps.IsNaN(scale) || numOps.IsInfinity(scale))
            throw new ArgumentException(
                $"The temperature {temperature} gives a logit scale that is not finite in {typeof(T).Name}.",
                nameof(temperature));
        return scale;
    }

    /// <summary>
    /// Encodes each text with <paramref name="encoder"/> and returns the L2-normalized embeddings as
    /// rows of a <c>[K, D]</c> tensor.
    /// </summary>
    /// <exception cref="ArgumentException">An embedding's length is not the encoder's declared dimension.</exception>
    internal static Tensor<T> EncodeNormalized<T>(ITextEncoder<T> encoder, IReadOnlyList<string> texts)
    {
        if (encoder is null) throw new ArgumentNullException(nameof(encoder));
        if (texts is null) throw new ArgumentNullException(nameof(texts));
        if (texts.Count == 0) throw new ArgumentException("At least one text is required.", nameof(texts));

        int dim = encoder.TextEmbeddingDimension;
        var encoded = encoder.EncodeTexts(texts.ToArray());
        if (encoded.Length != texts.Count)
            throw new InvalidOperationException(
                $"The text encoder returned {encoded.Length} embeddings for {texts.Count} texts.");

        var rows = new Tensor<T>[encoded.Length];
        for (int k = 0; k < encoded.Length; k++)
        {
            var embedding = encoded[k];
            if (embedding is null || embedding.Length != dim)
                throw new ArgumentException(
                    $"The text encoder returned an embedding of length {embedding?.Length ?? 0} for \"{texts[k]}\"; " +
                    $"its declared TextEmbeddingDimension is {dim}.", nameof(encoder));
            rows[k] = embedding.Reshape(new[] { 1, dim });
        }

        var engine = AiDotNetEngine.Current;
        return NormalizeRows(engine.TensorConcatenate(rows, 0));
    }

    /// <summary>
    /// Scores pixel embeddings <c>[D, H, W]</c> or <c>[B, D, H, W]</c> against normalized text
    /// embeddings <c>[K, D]</c>, returning <c>cos / tau</c> logits <c>[K, H, W]</c> or <c>[B, K, H, W]</c>.
    /// </summary>
    /// <remarks>Built from engine ops only, so a training forward stays on the gradient tape.</remarks>
    internal static Tensor<T> Classify<T>(Tensor<T> pixelEmbeddings, Tensor<T> normalizedText, T logitScale)
    {
        if (pixelEmbeddings is null) throw new ArgumentNullException(nameof(pixelEmbeddings));
        if (normalizedText is null) throw new ArgumentNullException(nameof(normalizedText));
        if (pixelEmbeddings.Rank != 3 && pixelEmbeddings.Rank != 4)
            throw new ArgumentException("Pixel embeddings must be [D, H, W] or [B, D, H, W].", nameof(pixelEmbeddings));
        if (normalizedText.Rank != 2)
            throw new ArgumentException("Text embeddings must be [K, D].", nameof(normalizedText));

        bool batched = pixelEmbeddings.Rank == 4;
        int b = batched ? pixelEmbeddings.Shape[0] : 1;
        int offset = batched ? 1 : 0;
        int d = pixelEmbeddings.Shape[offset], h = pixelEmbeddings.Shape[offset + 1], w = pixelEmbeddings.Shape[offset + 2];
        int k = normalizedText.Shape[0];
        if (normalizedText.Shape[1] != d)
            throw new ArgumentException(
                $"The network produces {d}-dimensional pixel embeddings but the text embeddings are " +
                $"{normalizedText.Shape[1]}-dimensional. The decoder's output width must equal the text " +
                "encoder's TextEmbeddingDimension.", nameof(pixelEmbeddings));

        var engine = AiDotNetEngine.Current;
        var flat = engine.Reshape(pixelEmbeddings, new[] { b, d, h * w });
        var normalized = NormalizeAxis(flat, 1);
        // [B, K, D] x [B, D, HW] -> [B, K, HW]. The text matrix is broadcast over B explicitly: TensorMatMul has
        // no 2-D x 3-D form (it supports 2Dx2D, NDx2D and same-rank NDxND) and throws for [K, D] x [B, D, HW].
        var text = engine.TensorBroadcastTo(engine.Reshape(normalizedText, new[] { 1, k, d }), new[] { b, k, d });
        var cosine = engine.TensorMatMul(text, normalized);
        var logits = engine.TensorMultiplyScalar(cosine, logitScale);
        return batched
            ? engine.Reshape(logits, new[] { b, k, h, w })
            : engine.Reshape(logits, new[] { k, h, w });
    }

    /// <summary>L2-normalizes each row of a <c>[K, D]</c> tensor.</summary>
    internal static Tensor<T> NormalizeRows<T>(Tensor<T> rows) => NormalizeAxis(rows, 1);

    /// <remarks>
    /// Each slice along <paramref name="axis"/> is first divided by its largest |entry| -- the normalized
    /// result does not depend on that factor, so it carries no gradient -- which bounds every square by 1:
    /// in Half an entry above 256 squared overflowed to infinity. The epsilon terms are raised to Half's
    /// smallest normal value where Half would flush them to zero (1e-24 is 0 in Half, making an all-zero
    /// embedding 0 / 0 = NaN).
    /// </remarks>
    private static Tensor<T> NormalizeAxis<T>(Tensor<T> x, int axis)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var largest = engine.StopGradient(engine.TensorClampMin(
            engine.ReduceMax(engine.TensorAbs(x), new[] { axis }, true), RepresentableEpsilon(numOps, NormEpsilon)));
        var scaled = engine.TensorDivide(x, engine.TensorBroadcastTo(largest, x._shape));
        var sumSquares = engine.ReduceSum(engine.TensorSquare(scaled), new[] { axis }, true);
        var norm = engine.TensorSqrt(engine.TensorAddScalar(sumSquares, RepresentableEpsilon(numOps, NormEpsilonSquared)));
        return engine.TensorDivide(scaled, engine.TensorBroadcastTo(norm, x._shape));
    }

    /// <summary>
    /// <paramref name="epsilon"/> as a <typeparamref name="T"/>, or Half's smallest normal value when
    /// <typeparamref name="T"/> flushes it to zero.
    /// </summary>
    private static T RepresentableEpsilon<T>(INumericOperations<T> numOps, double epsilon)
    {
        T value = numOps.FromDouble(epsilon);
        return numOps.GreaterThan(value, numOps.Zero) ? value : numOps.FromDouble(SmallestHalfNormal);
    }
}
