using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.SpeechRecognition;

/// <summary>
/// Frame-wise greedy CTC decoding with a per-token confidence, expressed as whole-tensor engine operations.
/// </summary>
/// <remarks>
/// <para>
/// For every frame the decoder picks the highest-scoring vocabulary entry, collapses consecutive repeats,
/// drops the blank (id 0) and stops after <c>maxTokens</c> emitted tokens. The confidence of an emitted
/// token is its frame's softmax probability, <c>1 / sum(exp(logit - max))</c>, and the returned confidence
/// is the mean over the emitted tokens (0 when nothing is emitted).
/// </para>
/// <para>
/// Ties and non-finite logits follow a left-to-right <c>value &gt; best</c> scan that starts at negative
/// infinity: the lowest index among equal maxima wins, NaN and negative infinity never win, and a frame
/// whose logits are all NaN or negative infinity decodes to the blank. The index comes from a composite key
/// (<c>vocab - column</c> on the frame's maxima) rather than from an argmax whose tie order is unspecified.
/// </para>
/// </remarks>
internal static class CtcGreedyDecoding
{
    /// <summary>
    /// Decodes <c>[frames, vocab]</c> logits, or a single <c>[vocab]</c> frame, read with the two-index
    /// element indexer (one index for a single frame).
    /// </summary>
    /// <exception cref="ArgumentException">The logits have rank above 2 and at least one element would be read.</exception>
    internal static (List<int> tokens, double confidence) Decode<T>(Tensor<T> logits, int maxTokens)
    {
        int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1;
        int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0];
        if (numFrames <= 0 || vocabSize <= 0 || maxTokens <= 0)
        {
            return (new List<int>(), 0.0);
        }

        if (logits.Rank > 2)
        {
            // The two-index element read of a frame-wise decode is only defined for rank-2 tensors.
            throw new ArgumentException("Number of indices must match the tensor's rank.");
        }

        return DecodeFrames(logits.Reshape(numFrames, vocabSize), maxTokens);
    }

    /// <summary>
    /// Decodes logits addressed by flat position: frame <c>t</c> is the flat elements
    /// <c>[t * vocab, (t + 1) * vocab)</c>, with <c>frames = Shape[0]</c> and <c>vocab = Shape[^1]</c>.
    /// </summary>
    internal static (List<int> tokens, double confidence) DecodeFlat<T>(Tensor<T> logits, int maxTokens)
    {
        int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1;
        int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0];
        if (numFrames <= 0 || vocabSize <= 0 || maxTokens <= 0)
        {
            return (new List<int>(), 0.0);
        }

        // frames * vocab never exceeds the element count: it is the product of a subset of the dimensions.
        var flat = logits.Reshape(logits.Length);
        var frames = AiDotNetEngine.Current.TensorNarrow(flat, 0, 0, numFrames * vocabSize);
        return DecodeFrames(frames.Reshape(numFrames, vocabSize), maxTokens);
    }

    private static (List<int> tokens, double confidence) DecodeFrames<T>(Tensor<T> scores, int maxTokens)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int numFrames = scores.Shape[0];
        int vocabSize = scores.Shape[1];
        var shape = new[] { numFrames, vocabSize };

        // Decoding only reads values; nothing here belongs on a gradient tape.
        using (new NoGradScope<T>())
        {
            var zeros = engine.TensorZerosLike(scores);
            var ones = engine.TensorOnesLike(scores);

            // Entries that can win a "value > best" scan started at negative infinity.
            var negativeInfinity = engine.TensorWhere(
                engine.TensorIsInf(scores), engine.TensorLessThan(scores, ops.Zero), zeros);
            var eligible = engine.TensorSubtract(
                engine.TensorWhere(engine.TensorIsNan(scores), zeros, ones), negativeInfinity);
            var clean = engine.TensorWhere(eligible, scores, engine.TensorFullLike(scores, ops.MinValue));

            var rowMax = engine.ReduceMax(clean, new[] { 1 }, true);
            var rowMaxWide = engine.TensorBroadcastTo(rowMax, shape);
            var isMax = engine.TensorMultiply(engine.TensorEquals(clean, rowMaxWide), eligible).Cast<int>();

            // key = vocab - column on a frame's maxima, so the largest key is the lowest maximal column; 0 = none.
            var reversedColumn = engine.TensorBroadcastTo(
                engine.TensorArange<int>(vocabSize, 0, -1).Reshape(1, vocabSize), shape);
            var key = engine.ReduceMax(engine.TensorMultiply(isMax, reversedColumn), new[] { 1 }, false);
            var bestIndex = engine.TensorMultiply(
                engine.ScalarMinusTensor(vocabSize, key), engine.TensorGreaterThan(key, 0));

            // Softmax probability of the winning entry: 1 / sum(exp(logit - max)).
            var sumExp = engine.ReduceSum(
                engine.TensorExp(engine.TensorSubtract(scores, rowMaxWide)), new[] { 1 }, false);
            var frameConfidence = engine.TensorReciprocal(sumExp);

            // Emit where the index differs from the previous frame's and is not the blank. The first frame is
            // compared against 0, which there is the same as "not blank".
            var startPad = new Tensor<int>(new[] { 1 });
            var previous = numFrames == 1
                ? startPad
                : engine.TensorConcatenate(new[] { startPad, engine.TensorNarrow(bestIndex, 0, 0, numFrames - 1) }, 0);
            var emitted = engine.TensorMultiply(
                engine.TensorNotEquals(bestIndex, previous), engine.TensorGreaterThan(bestIndex, 0));

            // Only the first maxTokens emissions count; decoding stops there.
            var overLimit = engine.TensorGreaterThan(engine.TensorCumSum(emitted, 0), maxTokens);
            var kept = engine.TensorMultiply(emitted, engine.ScalarMinusTensor(1, overLimit));

            var positions = engine.TensorNonzero(kept);
            int count = positions.Shape[0];
            if (count == 0)
            {
                return (new List<int>(), 0.0);
            }

            var keptFrames = positions.Reshape(count);
            var tokens = engine.TensorIndexSelect(bestIndex, keptFrames, 0).ToArray();
            T confidenceSum = engine.TensorSum(engine.TensorIndexSelect(frameConfidence, keptFrames, 0));
            return (new List<int>(tokens), ops.ToDouble(confidenceSum) / count);
        }
    }
}
