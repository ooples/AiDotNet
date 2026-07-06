using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// <b>Direct Feedback Alignment</b> (Nøkland, 2016; scaled to Transformers by Launay et al., 2020). Instead of
/// propagating the error sequentially through the network, DFA projects the <i>global</i> output error directly
/// onto every hidden layer through a per-layer fixed random matrix. Each layer's teaching signal depends only on
/// the output error — there is no backward chain — so the rule applies uniformly to dense, attention,
/// feed-forward, LayerNorm and embedding layers. The training engine turns each teaching signal into that layer's
/// parameter gradients via a local vector-Jacobian product (which supplies the layer's own activation Jacobian).
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
internal sealed class DirectFeedbackAlignmentCreditRule<T> : CreditRuleBase<T>
{
    // _feedback[i] maps the output error [B, C] to hidden layer i's flat feature space [B, M_i].
    // Shape: [C, M_i]. The output layer is trained with the exact loss gradient (no feedback matrix).
    private Matrix<T>?[]? _feedback;
    private int[]? _shapeSignature;

    public DirectFeedbackAlignmentCreditRule(int? seed = null) : base(seed) { }

    public override string Name => "DirectFeedbackAlignment";

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
            var layer = layers[i];
            _feedback[i] = layer.IsOutputLayer
                ? null
                : RandomGaussian(outputFeatures, layer.FlatFeatureSize, outputFeatures, random, context.NumOps);
            _shapeSignature[i] = layer.FlatFeatureSize;
        }
    }

    public override void ComputeTeachingSignals(ICreditAssignmentContext<T> context)
    {
        if (!IsInitializedFor(context)) Initialize(context);
        var error = ErrorMatrix(context); // [B, C]

        foreach (var layer in context.Layers)
        {
            if (layer.IsOutputLayer) continue; // engine uses the exact loss gradient here
            var projected = error.Multiply(_feedback![layer.Index]!); // [B, C] · [C, M_i] = [B, M_i]
            layer.TeachingSignal = ToTeachingSignal(projected, layer.OutputShape);
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
