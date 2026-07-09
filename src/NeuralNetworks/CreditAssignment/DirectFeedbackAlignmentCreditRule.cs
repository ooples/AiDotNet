using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// <b>Direct Feedback Alignment</b> (Nøkland, 2016; scaled to Transformers by Launay et al., 2020). Instead of
/// propagating the error sequentially through the network, DFA projects the <i>global</i> output error directly
/// onto every hidden layer through a per-layer fixed random matrix. Each layer's teaching signal depends only on
/// the output error — there is no backward chain — so the rule applies uniformly to dense, attention,
/// feed-forward, LayerNorm and embedding layers.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> DFA hands every hidden layer a <i>fixed random shortcut</i> of the final output error
/// instead of the exact back-propagated signal, and the network still learns because the forward weights rotate to
/// align with the random feedback. It is the simplest member of this family and the one that scales to Transformers.
/// </para>
/// </remarks>
internal sealed class DirectFeedbackAlignmentCreditRule<T> : CreditRuleBase<T>
{
    public DirectFeedbackAlignmentCreditRule(int? seed = null) : base(seed) { }

    public override string Name => "DirectFeedbackAlignment";

    public override void ComputeTeachingSignals(ICreditAssignmentContext<T> context)
    {
        int outputFeatures = context.OutputError.Shape[1];
        var feedback = EnsureFeedback(context, (layers, i) =>
            layers[i].IsOutputLayer ? null : (outputFeatures, layers[i].FlatFeatureSize));

        var error = ErrorMatrix(context); // [B, C]
        foreach (var layer in context.Layers)
        {
            if (layer.IsOutputLayer) continue; // engine uses the exact loss gradient here
            var projected = ProjectThrough(error, feedback[layer.Index]!); // [B, C] · [C, M_i] = [B, M_i]
            layer.TeachingSignal = ToTeachingSignal(projected, layer.OutputShape);
        }
    }
}
