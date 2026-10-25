using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks;

/// <summary>The causal language-model loss as Hugging Face's <c>ForCausalLMLoss</c> computes it.</summary>
internal static class CausalLanguageModelLoss
{
    /// <summary>
    /// Each position predicts the next label; the cross-entropy is averaged over the labels that are not
    /// <paramref name="ignoreIndex"/> (zero, still on the tape, when there are none).
    /// </summary>
    /// <param name="logits">Logits <c>[B, L, V]</c>.</param>
    /// <param name="labels">Labels <c>[B, L]</c>, aligned with the inputs (position t's logits score label t + 1).</param>
    /// <param name="ignoreIndex">The label that is not scored (-100).</param>
    public static Tensor<T> Compute<T>(IEngine engine, Tensor<T> logits, int[,] labels, int ignoreIndex = -100)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int b = logits.Shape[0], l = logits.Shape[1], v = logits.Shape[2];
        var picks = new List<int>();
        for (int i = 0; i < b; i++)
            for (int t = 0; t + 1 < l; t++)
            {
                int label = labels[i, t + 1];
                if (label == ignoreIndex) continue;
                if (label < 0 || label >= v) throw new ArgumentOutOfRangeException(nameof(labels), $"Label {label} is outside [0, {v}).");
                picks.Add((i * l + t) * v + label);
            }
        if (picks.Count == 0) return engine.TensorMultiplyScalar(engine.ReduceSum(logits, null), ops.Zero);
        var logProbabilities = engine.Reshape(engine.TensorLogSoftmax(logits, axis: 2), new[] { b * l * v, 1 });
        var index = new Tensor<int>(new[] { picks.Count }, new Vector<int>(picks.ToArray()));
        var picked = engine.TensorIndexSelect(logProbabilities, index, 0);
        return engine.TensorMultiplyScalar(engine.ReduceSum(picked, null), ops.FromDouble(-1.0 / picks.Count));
    }
}
