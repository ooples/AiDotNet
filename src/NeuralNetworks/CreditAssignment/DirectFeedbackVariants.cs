using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// <b>Direct Random Target Projection (DRTP)</b> — Frenkel, Lefebvre &amp; Bol, 2021 ("Learning without Feedback:
/// Fixed Random Learning Signals Allow for Feedforward Training of Deep Neural Networks"). Each hidden layer's
/// teaching signal is a <i>fixed random projection of the one-hot target</i> rather than of the output error — so
/// no backward error path is needed at all. Because the (negated) target supplies a valid descent direction on
/// average (the output error <c>prediction − target ≈ −target</c> early in training), projecting the target through
/// a fixed random matrix still trains the hidden layers. The rule holds only fixed random state, so it applies
/// uniformly to dense, attention, feed-forward, LayerNorm and embedding layers.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
internal sealed class DrtpCreditRule<T> : CreditRuleBase<T>
{
    // _feedback[i] : [outputFeatures, features_i], fixed random. The output layer has no feedback matrix.
    private Matrix<T>?[]? _feedback;
    private int[]? _shapeSignature;

    public DrtpCreditRule(int? seed = null) : base(seed) { }

    public override string Name => "DRTP";

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
        var target = TargetMatrix(context); // [B, C] one-hot
        var ops = context.NumOps;

        foreach (var layer in context.Layers)
        {
            if (layer.IsOutputLayer) continue;
            // Project the target, then negate: the descent direction matches DFA's error projection because
            // (prediction − target) ≈ −target while the prediction is still near-uniform.
            var projected = target.Multiply(_feedback![layer.Index]!); // [B, C] · [C, M_i] = [B, M_i]
            for (int b = 0; b < projected.Rows; b++)
                for (int j = 0; j < projected.Columns; j++)
                    projected[b, j] = ops.Negate(projected[b, j]);
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

/// <summary>
/// <b>Normalized Direct Feedback Alignment</b> (Launay et al., 2020 style). Vanilla DFA projects the global output
/// error onto every hidden layer through a fixed random matrix; without care the projected signal's magnitude
/// drifts with layer width and depth, which destabilizes training of deep / Transformer networks. This variant
/// keeps the magnitude stable: the feedback matrices are built with <b>unit-norm columns</b>, and each layer's
/// projected teaching signal is rescaled per sample to match the output error's magnitude. Otherwise identical to
/// Direct Feedback Alignment.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
internal sealed class NormalizedDfaCreditRule<T> : CreditRuleBase<T>
{
    private Matrix<T>?[]? _feedback;
    private int[]? _shapeSignature;

    public NormalizedDfaCreditRule(int? seed = null) : base(seed) { }

    public override string Name => "DFANormalized";

    public override void Initialize(ICreditAssignmentContext<T> context)
    {
        if (IsInitializedFor(context)) return;

        var layers = context.Layers;
        int outputFeatures = layers[layers.Count - 1].FlatFeatureSize;
        var random = ResolveRandom(context);
        var ops = context.NumOps;

        _feedback = new Matrix<T>?[layers.Count];
        _shapeSignature = new int[layers.Count + 1];
        _shapeSignature[layers.Count] = outputFeatures;
        for (int i = 0; i < layers.Count; i++)
        {
            if (layers[i].IsOutputLayer) { _feedback[i] = null; _shapeSignature[i] = layers[i].FlatFeatureSize; continue; }
            var b = RandomGaussian(outputFeatures, layers[i].FlatFeatureSize, outputFeatures, random, ops);
            NormalizeColumns(b, ops);
            _feedback[i] = b;
            _shapeSignature[i] = layers[i].FlatFeatureSize;
        }
    }

    public override void ComputeTeachingSignals(ICreditAssignmentContext<T> context)
    {
        if (!IsInitializedFor(context)) Initialize(context);
        var error = ErrorMatrix(context); // [B, C]
        var ops = context.NumOps;

        foreach (var layer in context.Layers)
        {
            if (layer.IsOutputLayer) continue;
            var projected = error.Multiply(_feedback![layer.Index]!); // [B, M_i]
            RescaleRowsToMatch(projected, error, ops);                 // depth-stable magnitude
            layer.TeachingSignal = ToTeachingSignal(projected, layer.OutputShape);
        }
    }

    /// <summary>Scales each column of <paramref name="m"/> to unit L2 norm (in place).</summary>
    private static void NormalizeColumns(Matrix<T> m, INumericOperations<T> ops)
    {
        for (int j = 0; j < m.Columns; j++)
        {
            double sq = 0;
            for (int i = 0; i < m.Rows; i++)
            {
                double v = ops.ToDouble(m[i, j]);
                sq += v * v;
            }
            double norm = Math.Sqrt(sq);
            if (norm <= 1e-12) continue;
            T inv = ops.FromDouble(1.0 / norm);
            for (int i = 0; i < m.Rows; i++)
                m[i, j] = ops.Multiply(m[i, j], inv);
        }
    }

    /// <summary>Rescales each row of <paramref name="signal"/> so its L2 norm matches the same row of <paramref name="reference"/>.</summary>
    private static void RescaleRowsToMatch(Matrix<T> signal, Matrix<T> reference, INumericOperations<T> ops)
    {
        for (int b = 0; b < signal.Rows; b++)
        {
            double sig = 0, refN = 0;
            for (int j = 0; j < signal.Columns; j++) { double v = ops.ToDouble(signal[b, j]); sig += v * v; }
            for (int c = 0; c < reference.Columns; c++) { double v = ops.ToDouble(reference[b, c]); refN += v * v; }
            double sigNorm = Math.Sqrt(sig);
            if (sigNorm <= 1e-12) continue;
            double scale = Math.Sqrt(refN) / sigNorm;
            T s = ops.FromDouble(scale);
            for (int j = 0; j < signal.Columns; j++)
                signal[b, j] = ops.Multiply(signal[b, j], s);
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
