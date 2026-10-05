using AiDotNet.ComputerVision;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Token-level objectives and decoding that every generative sequence model shares: teacher-forced
/// cross-entropy over next-token logits, cross-entropy against a soft target, and greedy decoding.
/// </summary>
/// <remarks>
/// They take logits <c>[rows, vocab]</c> where row r predicts the token after position r. Encoder-decoders
/// (UDOP, Florence-2), decoder-only VLMs (KOSMOS, GOT-OCR2) and plain language models therefore share one
/// tape-connected implementation instead of each re-deriving it.
/// </remarks>
public abstract partial class NeuralNetworkBase<T>
{
    /// <summary>
    /// Mean negative log-likelihood of <paramref name="labels"/>: label t is read from row
    /// <c><paramref name="firstRow"/> + t</c> of <paramref name="logits"/>. This is the teacher-forced
    /// next-token cross-entropy and stays on the tape.
    /// </summary>
    protected internal static Tensor<T> TokenCrossEntropy(Tensor<T> logits, IReadOnlyList<int> labels, int firstRow = 0)
    {
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        if (labels is null) throw new ArgumentNullException(nameof(labels));
        if (logits.Rank != 2) throw new ArgumentException($"Logits must be [rows, vocab]; got rank {logits.Rank}.", nameof(logits));
        if (labels.Count == 0) throw new ArgumentException("At least one label is required.", nameof(labels));
        int rows = logits.Shape[0], vocab = logits.Shape[1];
        if (firstRow < 0 || firstRow + labels.Count > rows)
            throw new ArgumentException($"{labels.Count} labels from row {firstRow} do not fit {rows} logit rows.", nameof(firstRow));
        var engine = AiDotNetEngine.Current;
        var log = engine.TensorLogSoftmax(logits, axis: 1);
        var entries = new int[labels.Count];
        for (int t = 0; t < labels.Count; t++)
        {
            int label = labels[t];
            if (label < 0 || label >= vocab)
                throw new ArgumentOutOfRangeException(nameof(labels), $"Label {label} at {t} is outside the {vocab}-token vocabulary.");
            entries[t] = ((firstRow + t) * vocab) + label;
        }
        var picked = CvTensorOps<T>.Select(engine.Reshape(log, new[] { log.Length }), entries, 0);
        return engine.TensorMultiplyScalar(engine.ReduceSum(picked, null),
            MathHelper.GetNumericOperations<T>().FromDouble(-1.0 / labels.Count));
    }

    /// <summary>
    /// Cross-entropy of <paramref name="logits"/> against a soft <paramref name="target"/> of the same shape:
    /// <c>-scale * sum(target * log_softmax(logits))</c>. With one row and scale 1 this is the cross-entropy of
    /// the first generated token. A caller averaging over R rows passes <c>1/R</c>.
    /// </summary>
    protected internal static Tensor<T> SoftTargetCrossEntropy(Tensor<T> logits, Tensor<T> target, double scale = 1.0)
    {
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        if (target is null) throw new ArgumentNullException(nameof(target));
        if (!logits.Shape.ToArray().SequenceEqual(target.Shape.ToArray()))
            throw new ArgumentException(
                $"The soft target [{string.Join(", ", target.Shape.ToArray())}] must match the logits [{string.Join(", ", logits.Shape.ToArray())}].",
                nameof(target));
        var engine = AiDotNetEngine.Current;
        var log = engine.TensorLogSoftmax(logits, axis: logits.Rank - 1);
        return engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(target, log), null),
            MathHelper.GetNumericOperations<T>().FromDouble(-scale));
    }

    /// <summary>
    /// Token ids from a target tensor: each value is rounded, passed through <paramref name="clamp"/>, and the
    /// result is capped at <paramref name="maxCount"/> ids.
    /// </summary>
    protected internal static int[] TokenIds(Tensor<T> target, Func<int, int> clamp, int maxCount = int.MaxValue)
    {
        if (target is null) throw new ArgumentNullException(nameof(target));
        if (clamp is null) throw new ArgumentNullException(nameof(clamp));
        var ops = MathHelper.GetNumericOperations<T>();
        var ids = new int[Math.Min(target.Length, maxCount)];
        for (int i = 0; i < ids.Length; i++) ids[i] = clamp((int)Math.Round(ops.ToDouble(target.Data.Span[i])));
        return ids;
    }

    /// <summary>
    /// The teacher-forcing decoder input for <paramref name="labels"/>: <paramref name="startToken"/> followed
    /// by every label except the last. Row t of the decoder then predicts label t.
    /// </summary>
    protected internal static int[] ShiftRight(IReadOnlyList<int> labels, int startToken)
    {
        if (labels is null) throw new ArgumentNullException(nameof(labels));
        var input = new int[labels.Count];
        if (input.Length == 0) return input;
        input[0] = startToken;
        for (int t = 1; t < input.Length; t++) input[t] = labels[t - 1];
        return input;
    }

    /// <summary>Generated ids as a rank-1 tensor.</summary>
    protected internal static Tensor<T> TokenTensor(IReadOnlyList<int> ids)
    {
        if (ids is null) throw new ArgumentNullException(nameof(ids));
        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) result[i] = ops.FromDouble(ids[i]);
        return result;
    }
    /// <summary>The highest-scoring token of row <paramref name="row"/>. The first maximum wins.</summary>
    protected internal static int GreedyToken(Tensor<T> logits, int row)
    {
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        var ops = MathHelper.GetNumericOperations<T>();
        int best = 0;
        double bestValue = double.NegativeInfinity;
        for (int v = 0; v < logits.Shape[1]; v++)
        {
            double value = ops.ToDouble(logits[row, v]);
            if (value > bestValue) { bestValue = value; best = v; }
        }
        return best;
    }

    /// <summary>
    /// Greedy decoding. On each step it scores <paramref name="context"/> with <paramref name="nextLogits"/>,
    /// takes the best token of the last row, records it, and stops when <paramref name="stopsAfter"/>(token,
    /// step) is true. Otherwise the token is appended to <paramref name="context"/>. Returns the generated
    /// tokens, including the stopping one.
    /// </summary>
    protected internal static List<int> GreedyDecode(
        Func<List<int>, Tensor<T>> nextLogits, List<int> context, int maxSteps, Func<int, int, bool> stopsAfter)
    {
        if (nextLogits is null) throw new ArgumentNullException(nameof(nextLogits));
        if (context is null) throw new ArgumentNullException(nameof(context));
        if (stopsAfter is null) throw new ArgumentNullException(nameof(stopsAfter));
        var generated = new List<int>();
        for (int step = 0; step < maxSteps; step++)
        {
            var logits = nextLogits(context);
            int best = GreedyToken(logits, logits.Shape[0] - 1);
            generated.Add(best);
            if (stopsAfter(best, step)) break;
            context.Add(best);
        }
        return generated;
    }
}
