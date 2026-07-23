using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// <b>Difference Target Propagation</b> (Lee, Zhang, Fischer &amp; Bengio, 2015). A backprop-free credit rule that
/// propagates <i>targets</i> (not gradients) backward through <b>learned inverses</b>. Every layer <c>j</c> carries an
/// approximate inverse <c>g_j</c> of its own forward map, trained online to <b>reconstruct its input from its
/// output</b>. Targets thread from the output toward the input with the <i>difference correction</i> that gives the
/// method its name:
/// <code>
/// ĥ_L   = h_L − η·(∂Loss/∂h_L)                 // output target: a small step down the loss
/// ĥ_{j−1} = h_{j−1} − g_j(h_j) + g_j(ĥ_j)      // difference-corrected hidden target
/// </code>
/// and each hidden layer's teaching signal is <c>h_{j−1} − ĥ_{j−1} = g_j(h_j) − g_j(ĥ_j)</c> — the layer is then
/// trained locally to move its output toward its target. The <c>−g_j(h_j)+g_j(ĥ_j)</c> difference term cancels the
/// inverse's reconstruction error, which is what makes target propagation stable when the inverse is imperfect.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <remarks>
/// <para>
/// <b>How it differs from the rest of the zoo.</b> Feedback Alignment routes the error through a <i>fixed random</i>
/// matrix; Kolen-Pollack trains its feedback to track the forward weights' <i>transpose</i> (<c>Wᵀ</c>); Local Error
/// Signals gives every layer a <i>direct</i> supervised classifier to the labels. Difference Target Propagation
/// instead trains each feedback matrix to be the layer's <i>inverse</i> (≈ <c>W⁺</c>, the pseudo-inverse) by an
/// activation-reconstruction objective, and threads corrected targets — not the raw error — backward through them.
/// Because targets chain from one layer to the previous one, this rule requires the trainable layers to be
/// <b>contiguous</b> (each layer's input is the previous layer's output); for non-contiguous stacks (trainable layers
/// separated by frozen blocks) use <see cref="LocalErrorSignalCreditRule{T}"/> or Direct Feedback Alignment.
/// </para>
/// <para>
/// <b>For Beginners:</b> instead of sending each layer an error message, this rule sends each layer a <i>target</i> —
/// "here is roughly what your output should have been." It figures out those targets by learning a little "undo"
/// function for every layer (guess the input from the output), and it cleverly subtracts that function's own mistake
/// so the targets stay trustworthy even before the undo functions are any good.
/// </para>
/// </remarks>
internal sealed class DifferenceTargetPropagationCreditRule<T> : CreditRuleBase<T>
{
    private readonly double _inverseLearningRate;
    private readonly double _weightDecay;
    private readonly double _outputStepSize;

    /// <summary>Creates the rule.</summary>
    /// <param name="seed">Optional RNG seed for reproducible inverse initialisation.</param>
    /// <param name="inverseLearningRate">Step size for the per-layer reconstruction (inverse) learning (default 0.05).</param>
    /// <param name="weightDecay">L2 decay on the learned inverses (default 0).</param>
    /// <param name="outputStepSize">The step η taken from the output activation toward the loss-reducing target (default 1.0).</param>
    public DifferenceTargetPropagationCreditRule(
        int? seed = null,
        double inverseLearningRate = 0.05,
        double weightDecay = 0.0,
        double outputStepSize = 1.0)
        : base(seed)
    {
        _inverseLearningRate = inverseLearningRate;
        _weightDecay = weightDecay;
        _outputStepSize = outputStepSize;
    }

    /// <inheritdoc />
    public override string Name => "DifferenceTargetPropagation";

    /// <inheritdoc />
    public override void ComputeTeachingSignals(ICreditAssignmentContext<T> context)
    {
        var layers = context.Layers;
        int last = layers.Count - 1;
        if (last < 1) return; // a single trainable layer is trained exactly by the engine; nothing to propagate

        var ops = context.NumOps;

        // One learned inverse g_j per layer j>=1: maps layer j's OUTPUT space (M_j) back to its INPUT space (In_j).
        var inverse = EnsureFeedback(context, (ls, i) =>
            i >= 1 ? (ls[i].FlatFeatureSize, InFeatures(ls[i])) : ((int rows, int cols)?)null);

        // Difference Target Propagation needs contiguous layers: each layer's input must be the previous layer's
        // output, so a hidden target ĥ_{j-1} (in In_j space) is a valid target for layer j-1's output (M_{j-1} space).
        for (int j = 1; j <= last; j++)
        {
            if (InFeatures(layers[j]) != layers[j - 1].FlatFeatureSize)
            {
                throw new NotSupportedException(
                    "DifferenceTargetPropagation requires contiguous trainable layers (each layer's input must be the " +
                    "previous trainable layer's output); a shape-changing or frozen layer sits between trainable layers " +
                    $"{j - 1} and {j}. Use CreditRule.LocalErrorSignal or CreditRule.DirectFeedbackAlignment for " +
                    "non-contiguous stacks (e.g. attention separated by frozen blocks).");
            }
        }

        // Output target: a small step down the loss. OutputError = prediction − target = ∂Loss/∂logits for CE/MSE.
        var hLast = FlatMatrix(layers[last].Output);              // [B, M_last]
        var outputError = ErrorMatrix(context);                  // [B, M_last]
        var targetAbove = SubtractScaled(hLast, outputError, _outputStepSize, ops); // ĥ_L = h_L − η·err

        // Thread targets from the output toward the input through the learned inverses (difference-corrected).
        for (int j = last; j >= 1; j--)
        {
            var hj = FlatMatrix(layers[j].Output);               // [B, M_j]
            var gAtOutput = ProjectThrough(hj, inverse[j]!);     // g_j(h_j)   [B, In_j]
            var gAtTarget = ProjectThrough(targetAbove, inverse[j]!); // g_j(ĥ_j)   [B, In_j]

            // teaching for layer j-1 = h_{j-1} − ĥ_{j-1} = g_j(h_j) − g_j(ĥ_j)  (the difference correction).
            var delta = Subtract(gAtOutput, gAtTarget, ops);     // [B, In_j = M_{j-1}]
            layers[j - 1].TeachingSignal = ToTeachingSignal(delta, layers[j - 1].OutputShape);

            // ĥ_{j-1} = h_{j-1} − delta, carried up for the next (shallower) layer.
            if (j - 1 >= 1)
            {
                var hjm1 = FlatMatrix(layers[j - 1].Output);     // [B, M_{j-1}]
                targetAbove = Subtract(hjm1, delta, ops);
            }
        }
    }

    /// <inheritdoc />
    protected override void UpdateFeedback(ICreditAssignmentContext<T> context)
    {
        var inverse = Feedback;
        if (inverse.Count == 0) return;
        var ops = context.NumOps;
        var layers = context.Layers;

        // Train each inverse g_j to reconstruct layer j's INPUT from its OUTPUT: minimise ‖g_j(h_j) − input_j‖².
        for (int j = 1; j < layers.Count; j++)
        {
            if (inverse[j] is null) continue;
            var yOut = FlatMatrix(layers[j].Output);             // [B, M_j]
            var xIn = FlatMatrix(layers[j].Input);               // [B, In_j]
            var reconstruction = ProjectThrough(yOut, inverse[j]!); // [B, In_j]
            var residual = Subtract(reconstruction, xIn, ops);   // g_j(h_j) − input_j
            var grad = MeanOuter(yOut, residual, ops);           // yᵀ·residual / B  -> [M_j, In_j]
            KpUpdate(inverse[j]!, grad, _inverseLearningRate, _weightDecay, ops);
        }
    }

    // ---- local helpers -----------------------------------------------------------------------

    /// <summary>Element-wise <c>a − b</c>.</summary>
    private static Matrix<T> Subtract(Matrix<T> a, Matrix<T> b, INumericOperations<T> ops)
    {
        var r = new Matrix<T>(a.Rows, a.Columns);
        for (int i = 0; i < a.Rows; i++)
            for (int j = 0; j < a.Columns; j++)
                r[i, j] = ops.Subtract(a[i, j], b[i, j]);
        return r;
    }

    /// <summary>Element-wise <c>a − scale·b</c>.</summary>
    private static Matrix<T> SubtractScaled(Matrix<T> a, Matrix<T> b, double scale, INumericOperations<T> ops)
    {
        T s = ops.FromDouble(scale);
        var r = new Matrix<T>(a.Rows, a.Columns);
        for (int i = 0; i < a.Rows; i++)
            for (int j = 0; j < a.Columns; j++)
                r[i, j] = ops.Subtract(a[i, j], ops.Multiply(s, b[i, j]));
        return r;
    }
}
