using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// <b>Direct Random Target Projection (DRTP)</b> — Frenkel, Lefebvre &amp; Bol, 2021 ("Learning without Feedback:
/// Fixed Random Learning Signals Allow for Feedforward Training of Deep Neural Networks"). Each hidden layer's
/// teaching signal is a <i>fixed random projection of the one-hot target</i> rather than of the output error — so
/// no backward error path is needed at all. Because the (negated) target supplies a valid descent direction on
/// average (the output error <c>prediction − target ≈ −target</c> early in training), projecting the target through
/// a fixed random matrix still trains the hidden layers.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> DRTP is the most hardware-friendly rule in this family — it never sends any error signal
/// backward. It only needs to know the correct answer (the target), which it projects through a fixed random matrix
/// to tell each layer which way to adjust. Surprisingly, that alone is enough for the network to learn.
/// </para>
/// </remarks>
internal sealed class DrtpCreditRule<T> : CreditRuleBase<T>
{
    public DrtpCreditRule(int? seed = null) : base(seed) { }

    public override string Name => "DRTP";

    public override void ComputeTeachingSignals(ICreditAssignmentContext<T> context)
    {
        int outputFeatures = context.OutputError.Shape[1];
        var feedback = EnsureFeedback(context, (layers, i) =>
            layers[i].IsOutputLayer ? null : (outputFeatures, layers[i].FlatFeatureSize));

        var target = TargetMatrix(context); // [B, C] one-hot
        var ops = context.NumOps;
        foreach (var layer in context.Layers)
        {
            if (layer.IsOutputLayer) continue;
            // Project the target, then negate: the descent direction matches DFA's error projection because
            // (prediction − target) ≈ −target while the prediction is still near-uniform.
            var projected = ProjectThrough(target, feedback[layer.Index]!); // [B, C] · [C, M_i] = [B, M_i]
            for (int b = 0; b < projected.Rows; b++)
                for (int j = 0; j < projected.Columns; j++)
                    projected[b, j] = ops.Negate(projected[b, j]);
            layer.TeachingSignal = ToTeachingSignal(projected, layer.OutputShape);
        }
    }
}

/// <summary>
/// <b>Normalized Direct Feedback Alignment</b> (Launay et al., 2020 style). Vanilla DFA projects the global output
/// error onto every hidden layer through a fixed random matrix; without care the projected signal's magnitude
/// drifts with layer width and depth, which destabilizes training of deep / Transformer networks. This variant
/// keeps the magnitude stable: the feedback matrices are built with <b>unit-norm columns</b>, and each layer's
/// projected teaching signal is rescaled per sample to match the output error's magnitude. Otherwise identical to
/// Direct Feedback Alignment.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> this is DFA with a volume knob. In a deep network the plain random-shortcut signal can
/// grow or shrink from layer to layer and derail training; the normalized version keeps every layer's message at a
/// steady, comparable strength, which is what lets DFA-style learning work on deep and Transformer models.
/// </para>
/// </remarks>
internal sealed class NormalizedDfaCreditRule<T> : CreditRuleBase<T>
{
    public NormalizedDfaCreditRule(int? seed = null) : base(seed) { }

    public override string Name => "DFANormalized";

    public override void ComputeTeachingSignals(ICreditAssignmentContext<T> context)
    {
        int outputFeatures = context.OutputError.Shape[1];
        var feedback = EnsureFeedback(context, (layers, i) =>
            layers[i].IsOutputLayer ? null : (outputFeatures, layers[i].FlatFeatureSize),
            normalizeColumns: true);

        var error = ErrorMatrix(context); // [B, C]
        var ops = context.NumOps;
        foreach (var layer in context.Layers)
        {
            if (layer.IsOutputLayer) continue;
            var projected = ProjectThrough(error, feedback[layer.Index]!); // [B, M_i]
            RescaleRowsToMatch(projected, error, ops);                      // depth-stable magnitude
            layer.TeachingSignal = ToTeachingSignal(projected, layer.OutputShape);
        }
    }
}
