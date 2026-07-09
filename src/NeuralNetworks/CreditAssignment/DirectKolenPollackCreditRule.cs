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
internal sealed class DirectKolenPollackCreditRule<T> : CreditRuleBase<T>, IFeedbackLearningRule<T>
{
    // _feedback[i] : [outputFeatures, features_i], learned. The output layer has no feedback matrix.
    private Matrix<T>?[]? _feedback;
    private int[]? _shapeSignature;

    private readonly double _feedbackLearningRate;
    private readonly double _weightDecay;

    public DirectKolenPollackCreditRule(int? seed = null, double feedbackLearningRate = 0.05, double weightDecay = 0.001)
        : base(seed)
    {
        _feedbackLearningRate = feedbackLearningRate;
        _weightDecay = weightDecay;
    }

    public override string Name => "DirectKolenPollack";

    public override void Initialize(ICreditAssignmentContext<T> context)
    {
        if (IsInitializedFor(context)) return;

        var layers = context.Layers;
        int outputFeatures = layers[layers.Count - 1].FlatFeatureSize;
        var random = ResolveRandom(context);

        _feedback = new Matrix<T>?[layers.Count];
        _shapeSignature = new int[layers.Count + 1];
        _shapeSignature[layers.Count] = outputFeatures;
        for (int i = 0; i < layers.Count; i++)
        {
            _feedback[i] = layers[i].IsOutputLayer
                ? null
                : RandomGaussian(outputFeatures, layers[i].FlatFeatureSize, outputFeatures, random, context.NumOps);
            _shapeSignature[i] = layers[i].FlatFeatureSize;
        }
    }

    public override void ComputeTeachingSignals(ICreditAssignmentContext<T> context)
    {
        if (!IsInitializedFor(context)) Initialize(context);
        var error = ErrorMatrix(context); // [B, C]

        foreach (var layer in context.Layers)
        {
            if (layer.IsOutputLayer) continue;
            var projected = error.Multiply(_feedback![layer.Index]!); // [B, C] · [C, M_i] = [B, M_i]
            layer.TeachingSignal = ToTeachingSignal(projected, layer.OutputShape);
        }
    }

    public void OnParametersUpdated(ICreditAssignmentContext<T> context)
    {
        if (_feedback is null) return;
        var error = ErrorMatrix(context); // [B, C]
        var ops = context.NumOps;

        // Align B_i to the effective forward map from layer i to the output: give it the increment a direct linear
        // read-out (output ≈ activation_i · B_iᵀ) trained against the error would receive: ΔB_i ∝ errorᵀ · activation_i.
        foreach (var layer in context.Layers)
        {
            if (layer.IsOutputLayer) continue;
            Matrix<T> activation = FlatMatrix(layer.Output); // [B, M_i]
            Matrix<T> grad = MeanOuter(error, activation, ops); // [C, M_i]
            KpUpdate(_feedback[layer.Index]!, grad, _feedbackLearningRate, _weightDecay, ops);
        }
    }

    private bool IsInitializedFor(ICreditAssignmentContext<T> context)
    {
        if (_feedback is null || _shapeSignature is null) return false;
        var layers = context.Layers;
        if (_feedback.Length != layers.Count) return false;
        if (_shapeSignature[layers.Count] != layers[layers.Count - 1].FlatFeatureSize) return false;
        for (int i = 0; i < layers.Count; i++)
            if (_shapeSignature[i] != layers[i].FlatFeatureSize) return false;
        return true;
    }
}
