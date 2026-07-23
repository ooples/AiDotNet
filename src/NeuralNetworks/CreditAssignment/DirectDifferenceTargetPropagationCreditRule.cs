using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// <b>Direct Difference Target Propagation (DDTP-linear)</b> — Meulemans, Carzaniga, Suykens, Sacramento &amp;
/// Grewe, 2020 ("A theoretical framework for target propagation"). A backprop-free rule that keeps Difference Target
/// Propagation's <b>learned inverses</b> but routes them <b>directly from the output</b> (like Direct Feedback
/// Alignment) instead of chaining layer-to-layer. Every hidden layer <c>i</c> owns a direct feedback map <c>q_i</c>
/// from the output space to its own, trained online to <b>reconstruct its activation from the network output</b>
/// (minimise ‖q_i(h_L) − h_i‖²). Its teaching signal is the difference-corrected
/// <c>q_i(h_L) − q_i(ĥ_L) = q_i(h_L − ĥ_L)</c> where <c>ĥ_L = h_L − η·(∂Loss/∂h_L)</c> is the output target — i.e.
/// the global output error routed through a <i>learned</i> direct inverse.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <remarks>
/// <para>
/// <b>Where it sits in the zoo.</b> It fills the missing corner of the design matrix:
/// <list type="bullet">
///   <item>Direct Feedback Alignment — direct topology, <i>fixed random</i> feedback.</item>
///   <item>Difference Target Propagation — <i>learned inverse</i> feedback, but <i>sequential</i> (contiguous layers only).</item>
///   <item><b>This rule</b> — <i>learned inverse</i> feedback <i>and</i> direct topology, so it trains deep routing
///   layers whose feedback is a real (trained) inverse ≈ <c>W⁺</c> <b>and</b> applies to <b>non-contiguous</b> stacks
///   (trainable layers separated by frozen blocks), which sequential Difference Target Propagation cannot.</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> like Direct Feedback Alignment it sends every layer a shortcut of the final error — but the
/// shortcut here is <i>learned</i>: each layer trains a little "guess my activation from the network's output"
/// function, and that trained guess (rather than a fixed random matrix) carries the error back. It keeps the
/// wide reach of the direct shortcut while making the shortcut meaningful.
/// </para>
/// </remarks>
internal sealed class DirectDifferenceTargetPropagationCreditRule<T> : CreditRuleBase<T>
{
    private readonly double _inverseLearningRate;
    private readonly double _weightDecay;

    /// <summary>Creates the rule.</summary>
    /// <param name="seed">Optional RNG seed for reproducible direct-inverse initialisation.</param>
    /// <param name="inverseLearningRate">Step size for the per-layer direct-reconstruction learning (default 0.05).</param>
    /// <param name="weightDecay">L2 decay on the learned direct inverses (default 0).</param>
    public DirectDifferenceTargetPropagationCreditRule(int? seed = null, double inverseLearningRate = 0.05, double weightDecay = 0.0)
        : base(seed)
    {
        _inverseLearningRate = inverseLearningRate;
        _weightDecay = weightDecay;
    }

    /// <inheritdoc />
    public override string Name => "DirectDifferenceTargetPropagation";

    /// <inheritdoc />
    public override void ComputeTeachingSignals(ICreditAssignmentContext<T> context)
    {
        int outputFeatures = context.OutputError.Shape[1];
        // One learned direct inverse q_i per hidden layer: maps the OUTPUT space (M_L) to layer i's output space (M_i).
        var directInverse = EnsureFeedback(context, (layers, i) =>
            layers[i].IsOutputLayer ? ((int rows, int cols)?)null : (outputFeatures, layers[i].FlatFeatureSize));

        // ĥ_L = h_L − η·(∂Loss/∂h_L). With η folded into the optimizer's learning rate the teaching signal is
        // q_i(h_L − ĥ_L) = q_i(outputError) = outputError · q_i — the global error through the learned direct inverse.
        var error = ErrorMatrix(context); // [B, C]
        foreach (var layer in context.Layers)
        {
            if (layer.IsOutputLayer) continue; // engine trains the output layer with the exact loss gradient
            var projected = ProjectThrough(error, directInverse[layer.Index]!); // [B, C] · [C, M_i] = [B, M_i]
            layer.TeachingSignal = ToTeachingSignal(projected, layer.OutputShape);
        }
    }

    /// <inheritdoc />
    protected override void UpdateFeedback(ICreditAssignmentContext<T> context)
    {
        var directInverse = Feedback;
        if (directInverse.Count == 0) return;
        var ops = context.NumOps;
        var layers = context.Layers;

        // The network output activation h_L (the output layer's forward output).
        ICreditLayer<T>? outputLayer = null;
        for (int i = 0; i < layers.Count; i++) if (layers[i].IsOutputLayer) { outputLayer = layers[i]; break; }
        if (outputLayer is null) return;
        var hL = FlatMatrix(outputLayer.Output); // [B, M_L]

        // Train each direct inverse q_i to reconstruct its hidden activation from the output: minimise ‖q_i(h_L) − h_i‖².
        foreach (var layer in layers)
        {
            if (layer.IsOutputLayer) continue;
            var qi = directInverse[layer.Index];
            if (qi is null) continue;
            var hi = FlatMatrix(layer.Output);              // [B, M_i]
            var reconstruction = ProjectThrough(hL, qi);    // h_L · q_i  -> [B, M_i]
            var residual = Subtract(reconstruction, hi, ops); // q_i(h_L) − h_i
            var grad = MeanOuter(hL, residual, ops);        // h_Lᵀ·residual / B  -> [M_L, M_i]
            KpUpdate(qi, grad, _inverseLearningRate, _weightDecay, ops);
        }
    }

    /// <summary>Element-wise <c>a − b</c>.</summary>
    private static Matrix<T> Subtract(Matrix<T> a, Matrix<T> b, INumericOperations<T> ops)
    {
        var r = new Matrix<T>(a.Rows, a.Columns);
        for (int i = 0; i < a.Rows; i++)
            for (int j = 0; j < a.Columns; j++)
                r[i, j] = ops.Subtract(a[i, j], b[i, j]);
        return r;
    }
}
