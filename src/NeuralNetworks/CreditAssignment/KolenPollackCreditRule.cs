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
/// <b>Topology.</b> <c>B_j</c> (shape <c>[outFeatures_j, inFeatures_j]</c>) plays the exact role of the forward
/// weight <c>W_j</c>: the teaching signal at layer <c>j</c>'s input is <c>δ_j · B_j</c>, which — when
/// <c>B_j → W_j</c> — equals the true back-propagated error <c>δ_j · W_j</c>. That input-side teaching signal
/// becomes the trainable layer below's output teaching signal (the layers must chain, i.e. each layer's input
/// feature count equals the previous trainable layer's output feature count — as in a dense stack). For the
/// attention/normalization case use <see cref="DirectKolenPollackCreditRule{T}"/>.
/// </para>
/// </remarks>
internal sealed class KolenPollackCreditRule<T> : CreditRuleBase<T>, IFeedbackLearningRule<T>
{
    // _feedback[j] : [outFeatures_j, inFeatures_j], learned toward W_j. _feedback[0] is unused (the first
    // trainable layer needs no downward feedback matrix — nothing below it consumes its input teaching).
    private Matrix<T>?[]? _feedback;
    private int[]? _outSignature;
    private int[]? _inSignature;

    private readonly double _feedbackLearningRate;
    private readonly double _weightDecay;

    public KolenPollackCreditRule(int? seed = null, double feedbackLearningRate = 0.05, double weightDecay = 0.001)
        : base(seed)
    {
        _feedbackLearningRate = feedbackLearningRate;
        _weightDecay = weightDecay;
    }

    public override string Name => "KolenPollack";

    public override void Initialize(ICreditAssignmentContext<T> context)
    {
        if (IsInitializedFor(context)) return;

        var layers = context.Layers;
        var random = ResolveRandom(context);

        _feedback = new Matrix<T>?[layers.Count];
        _outSignature = new int[layers.Count];
        _inSignature = new int[layers.Count];
        for (int j = 0; j < layers.Count; j++)
        {
            int outFeatures = layers[j].FlatFeatureSize;
            int inFeatures = InFeatures(layers[j]);
            _outSignature[j] = outFeatures;
            _inSignature[j] = inFeatures;
            // j == 0 has no downstream trainable layer that consumes its input teaching, so no feedback matrix.
            _feedback[j] = j == 0
                ? null
                : RandomGaussian(outFeatures, inFeatures, outFeatures, random, context.NumOps);
        }
    }

    public override void ComputeTeachingSignals(ICreditAssignmentContext<T> context)
    {
        if (!IsInitializedFor(context)) Initialize(context);
        var layers = context.Layers;
        int last = layers.Count - 1;

        // δ starts as the output error at the output layer's output space, then is transported down one trainable
        // layer at a time via that layer's learned feedback matrix.
        Matrix<T> delta = ErrorMatrix(context); // [B, outFeatures_last]
        for (int j = last; j >= 1; j--)
        {
            var b = _feedback![j]!;               // [outFeatures_j, inFeatures_j]
            Matrix<T> teachingBelow = delta.Multiply(b); // [B, inFeatures_j] == below layer's output space

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

    public void OnParametersUpdated(ICreditAssignmentContext<T> context)
    {
        if (_feedback is null) return;
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
            KpUpdate(_feedback[j]!, grad, _feedbackLearningRate, _weightDecay, ops);
        }
    }

    private static int InFeatures(ICreditLayer<T> layer)
    {
        var shape = layer.Input.Shape;
        int flat = 1;
        for (int i = 1; i < shape.Length; i++) flat *= shape[i];
        return flat;
    }

    private bool IsInitializedFor(ICreditAssignmentContext<T> context)
    {
        if (_feedback is null || _outSignature is null || _inSignature is null) return false;
        var layers = context.Layers;
        if (_feedback.Length != layers.Count) return false;
        for (int j = 0; j < layers.Count; j++)
            if (_outSignature[j] != layers[j].FlatFeatureSize || _inSignature[j] != InFeatures(layers[j]))
                return false;
        return true;
    }
}
