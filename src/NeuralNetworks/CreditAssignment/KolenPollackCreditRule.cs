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
    private Matrix<T>?[]? _previousForwardWeights;

    public KolenPollackCreditRule(int? seed = null, double feedbackLearningRate = 1.0, double weightDecay = 0.001)
        : base(seed)
    {
        if (double.IsNaN(feedbackLearningRate) || double.IsInfinity(feedbackLearningRate) || feedbackLearningRate < 0)
            throw new ArgumentOutOfRangeException(nameof(feedbackLearningRate), "The feedback update scale must be finite and non-negative.");
        if (double.IsNaN(weightDecay) || double.IsInfinity(weightDecay) || weightDecay < 0 || weightDecay >= 1)
            throw new ArgumentOutOfRangeException(nameof(weightDecay), "Weight decay must be finite and in [0, 1).");

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

        // The gradient hook runs before the external optimizer mutates W. At the beginning of the next
        // step, observe the completed optimizer delta and give B that same increment. This preserves the
        // Kolen-Pollack symmetry rule for Adam and other adaptive optimizers; the old fixed-rate SGD
        // approximation updated B differently from W and therefore could not reliably align them.
        ApplyObservedForwardUpdates(layers, feedback, context.NumOps);

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
        // This hook runs before the caller's optimizer step, while W still has its old value.
        // ApplyObservedForwardUpdates consumes the completed delta at the next step.
    }

    private void ApplyObservedForwardUpdates(
        IReadOnlyList<ICreditLayer<T>> layers,
        IReadOnlyList<Matrix<T>?> feedback,
        INumericOperations<T> ops)
    {
        if (_previousForwardWeights is null || _previousForwardWeights.Length != layers.Count)
            _previousForwardWeights = new Matrix<T>?[layers.Count];

        T decay = ops.FromDouble(_weightDecay);
        T updateScale = ops.FromDouble(_feedbackLearningRate);

        for (int j = 1; j < layers.Count; j++)
        {
            Matrix<T> current = layers[j].Weights ?? throw new NotSupportedException(
                $"KolenPollack requires layer {j} to expose a single rank-2 forward weight matrix. " +
                "Use CreditRule.DirectKolenPollack for attention, normalization, or composite layers.");
            Matrix<T> b = feedback[j]!;
            Matrix<T>? previous = _previousForwardWeights[j];

            if (previous is not null &&
                previous.Rows == current.Rows && previous.Columns == current.Columns)
            {
                for (int row = 0; row < b.Rows; row++)
                {
                    for (int column = 0; column < b.Columns; column++)
                    {
                        T forwardIncrement = ops.Subtract(current[row, column], previous[row, column]);
                        T alignmentError = ops.Subtract(b[row, column], previous[row, column]);
                        b[row, column] = ops.Subtract(
                            ops.Add(b[row, column], ops.Multiply(updateScale, forwardIncrement)),
                            ops.Multiply(decay, alignmentError));
                    }
                }
            }

            _previousForwardWeights[j] = current.Clone();
        }
    }
}
