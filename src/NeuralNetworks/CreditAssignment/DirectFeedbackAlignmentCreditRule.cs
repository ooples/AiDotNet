using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// <b>Direct Feedback Alignment</b> (Nøkland, 2016). Instead of propagating the error sequentially through the
/// network, DFA projects the <i>global</i> output error directly onto every hidden layer through a per-layer
/// fixed random matrix, then gates it by that layer's activation derivative. Each layer's teaching signal
/// depends only on the output error — there is no backward chain — so updates are local and parallelisable.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
internal sealed class DirectFeedbackAlignmentCreditRule<T> : CreditRuleBase<T>
{
    // _feedback[i] maps the output error [B, outputFeatures] to hidden layer i's units [B, outDim_i].
    // Shape: [outputFeatures, outDim_i]. The final (output) layer uses the true error and has no matrix.
    private Matrix<T>?[]? _feedback;
    private int[]? _shapeSignature;

    public override string Name => "DirectFeedbackAlignment";

    public override void Initialize(ICreditAssignmentContext<T> context)
    {
        if (IsInitializedFor(context)) return;

        var layers = context.Layers;
        int outputFeatures = layers[layers.Count - 1].OutputDim;
        _feedback = new Matrix<T>?[layers.Count];
        _shapeSignature = new int[layers.Count + 1];
        _shapeSignature[layers.Count] = outputFeatures;
        for (int i = 0; i < layers.Count; i++)
        {
            var layer = layers[i];
            _feedback[i] = layer.IsOutputLayer
                ? null
                : RandomGaussian(outputFeatures, layer.OutputDim, outputFeatures, context.Random, context.NumOps);
            _shapeSignature[i] = layer.OutputDim;
        }
    }

    public override void ComputeUpdates(ICreditAssignmentContext<T> context)
    {
        if (!IsInitializedFor(context)) Initialize(context);
        var ops = context.NumOps;
        var layers = context.Layers;
        var error = context.OutputError; // [B, outputFeatures]

        for (int i = 0; i < layers.Count; i++)
        {
            var layer = layers[i];
            Matrix<T> delta;
            if (layer.IsOutputLayer)
            {
                // Output layer's delta is the true error (= ∂L/∂z_last for a matched output loss).
                delta = error;
            }
            else
            {
                var projected = error.Multiply(_feedback![i]!);   // [B, outFeat] · [outFeat, outDim_i] = [B, outDim_i]
                var deriv = layer.ActivationDerivative();
                delta = Hadamard(projected, deriv, ops);
            }

            SetParameterGradients(layer, delta, ops);
        }
    }

    private bool IsInitializedFor(ICreditAssignmentContext<T> context)
    {
        if (_feedback is null || _shapeSignature is null) return false;
        var layers = context.Layers;
        if (_feedback.Length != layers.Count) return false;
        if (_shapeSignature[layers.Count] != layers[layers.Count - 1].OutputDim) return false;
        for (int i = 0; i < layers.Count; i++)
            if (_shapeSignature[i] != layers[i].OutputDim) return false;
        return true;
    }
}
