using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// <b>Local Error Signals</b> (Nøkland &amp; Eidnes, 2019). A backprop-free, per-layer <i>supervised</i> credit
/// rule: every hidden layer carries its own learned linear classifier <c>W_i</c> (<c>[features_i, classes]</c>) to
/// the output labels, and the layer's teaching signal is the gradient of <b>its own</b> cross-entropy against the
/// true target — <c>(softmax(h_i·W_i) − y)·W_iᵀ</c>. There is no backward chain and no privileged intermediate
/// target: each layer is trained greedily to make its representation predict the label, and <c>W_i</c> is trained
/// locally each step.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <remarks>
/// <para>
/// <b>Why it exists.</b> DFA/Direct-Kolen-Pollack route the <i>top-layer</i> error to each hidden layer through a
/// feedback matrix; that signal is weak for deep routing layers (e.g. attention) far from the output, and empirically
/// fails to train them. Local Error Signals instead give every layer a genuine supervised gradient from the labels,
/// so a routing layer receives a real target regardless of its distance from the readout. Because the topology is
/// <i>direct</i> (each layer talks only to the labels, never to its neighbours) it also applies when the trainable
/// layers are <b>non-contiguous</b> — separated by frozen or non-parameterised layers — which sequential target
/// propagation cannot handle.
/// </para>
/// <para>
/// <b>For Beginners:</b> instead of one exact error chained back from the end (back-prop) or one random shortcut of
/// the final error (DFA), every layer here gets its own little "guess the answer from what I have so far" classifier
/// and learns from how wrong that guess is. Layers close to and far from the output all get a strong, direct signal.
/// </para>
/// </remarks>
internal sealed class LocalErrorSignalCreditRule<T> : CreditRuleBase<T>
{
    private readonly double _classifierLearningRate;
    private readonly double _weightDecay;

    /// <summary>Creates the rule.</summary>
    /// <param name="seed">Optional RNG seed for reproducible per-layer classifier initialisation.</param>
    /// <param name="classifierLearningRate">Step size for the local per-layer classifiers (default 0.05).</param>
    /// <param name="weightDecay">L2 decay on the classifiers (default 0).</param>
    public LocalErrorSignalCreditRule(int? seed = null, double classifierLearningRate = 0.05, double weightDecay = 0.0)
        : base(seed)
    {
        _classifierLearningRate = classifierLearningRate;
        _weightDecay = weightDecay;
    }

    /// <inheritdoc />
    public override string Name => "LocalErrorSignal";

    /// <inheritdoc />
    public override void ComputeTeachingSignals(ICreditAssignmentContext<T> context)
    {
        int classes = context.OutputError.Shape[1];
        // One learned classifier W_i [features_i, classes] per hidden layer (output layer trained exactly by the engine).
        var classifier = EnsureFeedback(context, (layers, i) =>
            layers[i].IsOutputLayer ? null : (layers[i].FlatFeatureSize, classes));

        var target = TargetMatrix(context); // [B, classes]
        var ops = context.NumOps;
        foreach (var layer in context.Layers)
        {
            if (layer.IsOutputLayer) continue;
            var h = FlatMatrix(layer.Output);                        // [B, M_i]
            var logits = ProjectThrough(h, classifier[layer.Index]!); // [B, classes]
            var error = SoftmaxRowsMinus(logits, target, ops);        // softmax(logits) − y   [B, classes]
            // teaching signal = dLocalCE/dh_i = error · W_iᵀ   -> [B, M_i]
            var teach = error.Multiply(Transpose(classifier[layer.Index]!, ops));
            layer.TeachingSignal = ToTeachingSignal(teach, layer.OutputShape);
        }
    }

    /// <inheritdoc />
    protected override void UpdateFeedback(ICreditAssignmentContext<T> context)
    {
        var classifier = Feedback;
        if (classifier.Count == 0) return;
        var target = TargetMatrix(context);
        var ops = context.NumOps;
        foreach (var layer in context.Layers)
        {
            if (layer.IsOutputLayer) continue;
            var h = FlatMatrix(layer.Output);                        // [B, M_i]
            var logits = ProjectThrough(h, classifier[layer.Index]!); // [B, classes]
            var error = SoftmaxRowsMinus(logits, target, ops);        // [B, classes]
            var grad = MeanOuter(h, error, ops);                     // hᵀ·error / batch  -> [M_i, classes]
            KpUpdate(classifier[layer.Index]!, grad, _classifierLearningRate, _weightDecay, ops);
        }
    }

    // ---- local helpers -----------------------------------------------------------------------

    /// <summary>Row-wise softmax of <paramref name="logits"/> minus <paramref name="target"/> (numerically stable).</summary>
    private static Matrix<T> SoftmaxRowsMinus(Matrix<T> logits, Matrix<T> target, INumericOperations<T> ops)
    {
        int rows = logits.Rows, cols = logits.Columns;
        var result = new Matrix<T>(rows, cols);
        for (int b = 0; b < rows; b++)
        {
            double max = double.NegativeInfinity;
            for (int j = 0; j < cols; j++) max = Math.Max(max, ops.ToDouble(logits[b, j]));
            double sum = 0;
            var exp = new double[cols];
            for (int j = 0; j < cols; j++) { exp[j] = Math.Exp(ops.ToDouble(logits[b, j]) - max); sum += exp[j]; }
            double inv = sum > 1e-30 ? 1.0 / sum : 0.0;
            for (int j = 0; j < cols; j++)
                result[b, j] = ops.FromDouble(exp[j] * inv - ops.ToDouble(target[b, j]));
        }
        return result;
    }

    /// <summary>Transpose of <paramref name="m"/>.</summary>
    private static Matrix<T> Transpose(Matrix<T> m, INumericOperations<T> ops)
    {
        var t = new Matrix<T>(m.Columns, m.Rows);
        for (int i = 0; i < m.Rows; i++)
            for (int j = 0; j < m.Columns; j++)
                t[j, i] = m[i, j];
        return t;
    }
}
