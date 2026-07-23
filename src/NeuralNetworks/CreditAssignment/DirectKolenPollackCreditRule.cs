using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// <b>Direct Kolen-Pollack (DKP)</b>: Kolen-Pollack feedback learning applied in the Direct Feedback Alignment
/// topology. As in DFA the global output error is projected directly onto every hidden layer through a per-layer
/// feedback matrix <c>B_i</c> (shape <c>[outputFeatures, features_i]</c>), so the rule applies uniformly to dense,
/// attention, feed-forward, LayerNorm and embedding layers. Unlike DFA the matrices are <b>learned</b>: each step
/// <c>B_i</c> receives the same outer-product increment (plus weight decay) that a direct linear read-out from
/// layer <c>i</c>'s activation to the output error would receive, so <c>B_i</c> aligns to the effective forward
/// Jacobian from that layer to the output and the routed teaching signal approaches the true error.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> DKP starts like DFA (a random shortcut to each layer) but then <i>trains</i> those
/// shortcuts every step so they steadily become better estimates of the exact feedback — combining DFA's
/// attention-friendly wiring with back-propagation-like accuracy.
/// </para>
/// </remarks>
internal sealed class DirectKolenPollackCreditRule<T> : CreditRuleBase<T>
{
    private readonly double _feedbackLearningRate;
    private readonly double _weightDecay;

    public DirectKolenPollackCreditRule(int? seed = null, double feedbackLearningRate = 0.05, double weightDecay = 0.001)
        : base(seed)
    {
        _feedbackLearningRate = feedbackLearningRate;
        _weightDecay = weightDecay;
    }

    public override string Name => "DirectKolenPollack";

    public override void ComputeTeachingSignals(ICreditAssignmentContext<T> context)
    {
        int outputFeatures = context.OutputError.Shape[1];
        var feedback = EnsureFeedback(context, (layers, i) =>
            layers[i].IsOutputLayer ? null : (outputFeatures, layers[i].FlatFeatureSize));

        var error = ErrorMatrix(context); // [B, C]
        foreach (var layer in context.Layers)
        {
            if (layer.IsOutputLayer) continue;
            var projected = ProjectThrough(error, feedback[layer.Index]!); // [B, C] · [C, M_i] = [B, M_i]
            layer.TeachingSignal = ToTeachingSignal(projected, layer.OutputShape);
        }
    }

    protected override void UpdateFeedback(ICreditAssignmentContext<T> context)
    {
        var feedback = Feedback;
        if (feedback.Count == 0) return;
        var error = ErrorMatrix(context); // [B, C]
        var ops = context.NumOps;

        // Align B_i to the effective forward map from layer i to the output: give it the increment a direct linear
        // read-out (output ≈ activation_i · B_iᵀ) trained against the error would receive: ΔB_i ∝ errorᵀ · activation_i.
        foreach (var layer in context.Layers)
        {
            if (layer.IsOutputLayer) continue;
            Matrix<T> activation = FlatMatrix(layer.Output); // [B, M_i]
            Matrix<T> grad = MeanOuter(error, activation, ops); // [C, M_i]
            KpUpdate(feedback[layer.Index]!, grad, _feedbackLearningRate, _weightDecay, ops);
        }
    }
}
