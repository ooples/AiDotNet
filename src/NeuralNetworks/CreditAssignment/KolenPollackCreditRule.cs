using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// <b>Kolen-Pollack (KP)</b> feedback learning (Kolen &amp; Pollack, 1994; Akrout et al., 2019, "Deep Learning
/// without Weight Transport"). Like Feedback Alignment the error is routed sequentially through per-layer feedback
/// matrices instead of the transpose forward weights — but here the feedback matrices are <b>learned</b>. Each step
/// every feedback matrix <c>B_j</c> receives the <i>same</i> outer-product increment (plus weight decay) that its
/// forward weight <c>W_j</c> receives, so the difference <c>W_j − B_j</c> decays and <c>B_j</c> converges to
/// <c>W_j</c>. As alignment improves, the routed teaching signal approaches the exact back-propagated error, so KP
/// closes the credit-assignment gap that fixed-feedback FA/DFA leave open on deep networks.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> KP is Feedback Alignment that <i>learns</i>: every step it nudges each layer's random
/// feedback matrix to look a little more like the real forward weight, so after a while the shortcut messages it
/// sends each layer are almost as good as exact back-propagation — which is why it keeps working on deep networks.
/// </para>
/// <para>
/// <b>Topology.</b> <c>B_j</c> (shape <c>[outFeatures_j, inFeatures_j]</c>) plays the role of the forward weight
/// <c>W_j</c>: the teaching signal at layer <c>j</c>'s input is <c>δ_j · B_j</c>, which — when <c>B_j → W_j</c> —
/// equals the true back-propagated error <c>δ_j · W_j</c>. That input-side teaching signal becomes the trainable
/// layer below's output teaching signal (the layers must chain, i.e. each layer's input feature count equals the
/// previous trainable layer's output feature count — as in a dense stack). For attention/normalization use
/// <see cref="DirectKolenPollackCreditRule{T}"/>.
/// </para>
/// </remarks>
internal sealed class KolenPollackCreditRule<T> : CreditRuleBase<T>
{
    private readonly double _feedbackLearningRate;
    private readonly double _weightDecay;

    public KolenPollackCreditRule(int? seed = null, double feedbackLearningRate = 0.05, double weightDecay = 0.001)
        : base(seed)
    {
        _feedbackLearningRate = feedbackLearningRate;
        _weightDecay = weightDecay;
    }

    public override string Name => "KolenPollack";

    // B_j (j >= 1) maps layer j's output-teaching to its input-teaching; layer 0 needs none (nothing below consumes it).
    private Matrix<T>?[] EnsureFeedback(ICreditAssignmentContext<T> context) =>
        EnsureFeedback(context, (layers, j) => j == 0 ? null : (layers[j].FlatFeatureSize, InFeatures(layers[j])));

    public override void ComputeTeachingSignals(ICreditAssignmentContext<T> context)
    {
        var feedback = EnsureFeedback(context);
        var layers = context.Layers;
        int last = layers.Count - 1;

        // δ starts as the output error at the output layer's output space, then is transported down one trainable
        // layer at a time via that layer's learned feedback matrix.
        Matrix<T> delta = ErrorMatrix(context); // [B, outFeatures_last]
        for (int j = last; j >= 1; j--)
        {
            var b = feedback[j]!;                          // [outFeatures_j, inFeatures_j]
            Matrix<T> teachingBelow = delta.Multiply(b);   // [B, inFeatures_j] == below layer's output space

            var below = layers[j - 1];
            if (teachingBelow.Columns != below.FlatFeatureSize)
            {
                throw new NotSupportedException(
                    $"KolenPollack requires consecutive trainable layers to chain (layer {j}'s input feature count " +
                    $"{teachingBelow.Columns} must equal layer {j - 1}'s output feature count {below.FlatFeatureSize}). " +
                    "Use CreditRule.DirectKolenPollack for attention / non-chaining architectures.");
            }

            below.TeachingSignal = ToTeachingSignal(teachingBelow, below.OutputShape);
            delta = teachingBelow;
        }
    }

    protected override void UpdateFeedback(ICreditAssignmentContext<T> context)
    {
        var feedback = Feedback;
        if (feedback.Count == 0) return;
        var layers = context.Layers;
        int last = layers.Count - 1;
        var ops = context.NumOps;

        // Each feedback matrix receives the same mean outer-product increment its forward weight receives:
        // ΔB_j ∝ (δ_j^out)ᵀ · input_j, with weight decay — driving B_j → W_j (Kolen-Pollack).
        for (int j = 1; j <= last; j++)
        {
            Matrix<T> deltaOut = j == last ? ErrorMatrix(context) : FlatMatrix(layers[j].TeachingSignal!);
            Matrix<T> input = FlatMatrix(layers[j].Input); // [B, inFeatures_j]
            Matrix<T> grad = MeanOuter(deltaOut, input, ops); // [outFeatures_j, inFeatures_j]
            KpUpdate(feedback[j]!, grad, _feedbackLearningRate, _weightDecay, ops);
        }
    }
}
