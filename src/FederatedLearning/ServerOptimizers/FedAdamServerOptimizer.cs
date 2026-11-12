using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.FederatedLearning.ServerOptimizers;

/// <summary>
/// FedAdam server optimizer — adaptive learning rates with momentum and second-moment estimation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Adam combines the benefits of Adagrad (adaptive learning rates)
/// and momentum (smoothed gradient direction). FedAdam applies this at the server level,
/// treating aggregated client updates as pseudo-gradients for server-side optimization.</para>
///
/// <para>Reference: Reddi, S., et al. (2021). "Adaptive Federated Optimization." ICLR 2021.</para>
/// </remarks>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public sealed class FedAdamServerOptimizer<T> : FederatedServerOptimizerBase<T>
{
    private readonly double _learningRate;
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _epsilon;

    private Vector<T>? _m;
    private Vector<T>? _v;

    public FedAdamServerOptimizer(double learningRate = 1.0, double beta1 = 0.9, double beta2 = 0.99, double epsilon = 1e-3)
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
        _v ??= new Vector<T>(n);

        // Reddi et al. 2021, Algorithm 2 (FedAdam): no bias correction.
        var g = Engine.Subtract(aggregatedTargetParameters, currentGlobalParameters);
        _m = Engine.Add(Engine.Multiply(_m, NumOps.FromDouble(_beta1)), Engine.Multiply(g, NumOps.FromDouble(1.0 - _beta1)));
        _v = Engine.Add(
            Engine.Multiply(_v, NumOps.FromDouble(_beta2)),
            Engine.Multiply(Engine.Multiply(g, g), NumOps.FromDouble(1.0 - _beta2)));

        var denominator = Engine.TensorAddScalar(Engine.Sqrt(_v).AsTensor(), NumOps.FromDouble(_epsilon)).ToVector();
        var step = Engine.Multiply(Engine.Divide(_m, denominator), NumOps.FromDouble(_learningRate));
        return Engine.Add(currentGlobalParameters, step);
    }

    /// <inheritdoc />
    public override string GetOptimizerName() => "FedAdam";
}

