using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.FederatedLearning.ServerOptimizers;

/// <summary>
/// FedYogi server optimizer — adaptive learning rates with controlled second-moment growth.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Yogi is similar to Adam but more stable with noisy or sparse updates.
/// While Adam's second moment can grow unboundedly (causing vanishingly small learning rates),
/// Yogi uses an additive update that only grows the second moment when the gradient magnitude
/// exceeds the current estimate, preventing premature convergence.</para>
///
/// <para>Reference: Reddi, S., et al. (2021). "Adaptive Federated Optimization." ICLR 2021.</para>
/// </remarks>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public sealed class FedYogiServerOptimizer<T> : FederatedServerOptimizerBase<T>
{
    private readonly double _learningRate;
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _epsilon;

    private Vector<T>? _m;
    private Vector<T>? _v;

    public FedYogiServerOptimizer(double learningRate = 1.0, double beta1 = 0.9, double beta2 = 0.99, double epsilon = 1e-3)
    {
        if (learningRate <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive.");
        }

        if (beta1 < 0.0 || beta1 >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(beta1), "Beta1 must be in [0, 1).");
        }

        if (beta2 < 0.0 || beta2 >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(beta2), "Beta2 must be in [0, 1).");
        }

        if (epsilon <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");
        }

        _learningRate = learningRate;
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
    }

    /// <inheritdoc />
    public override Vector<T> Step(Vector<T> currentGlobalParameters, Vector<T> aggregatedTargetParameters)
    {
        ValidateVectors(currentGlobalParameters, aggregatedTargetParameters);

        int n = currentGlobalParameters.Length;
        _m ??= new Vector<T>(n);
        if (_v is null)
        {
            // Reddi et al. 2021, Algorithm 2: v_{-1} >= tau^2.
            _v = new Vector<T>(n);
            _v.Fill(NumOps.FromDouble(_epsilon * _epsilon));
        }

        // Reddi et al. 2021, Algorithm 2 (FedYogi), no bias correction:
        // m = b1 m + (1 - b1) g;  v = v + (1 - b2) sign(g^2 - v) g^2;  x = x + lr m / (sqrt(v) + tau).
        var g = Engine.Subtract(aggregatedTargetParameters, currentGlobalParameters);
        var gSquared = Engine.Multiply(g, g);
        _m = Engine.Add(Engine.Multiply(_m, NumOps.FromDouble(_beta1)), Engine.Multiply(g, NumOps.FromDouble(1.0 - _beta1)));
        var direction = Engine.Sign(Engine.Subtract(gSquared, _v));
        _v = Engine.Add(_v, Engine.Multiply(Engine.Multiply(direction, gSquared), NumOps.FromDouble(1.0 - _beta2)));

        var denominator = Engine.TensorAddScalar(Engine.Sqrt(_v).AsTensor(), NumOps.FromDouble(_epsilon)).ToVector();
        var step = Engine.Multiply(Engine.Divide(_m, denominator), NumOps.FromDouble(_learningRate));
        return Engine.Add(currentGlobalParameters, step);
    }

    /// <inheritdoc />
    public override string GetOptimizerName() => "FedYogi";
}
