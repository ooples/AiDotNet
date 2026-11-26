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
    private Vector<float>? _mWide;
    private Vector<float>? _vWide;

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

        // Half / BFloat16 keep their optimizer state in float (as the previous double state did): the squared
        // deltas and the accumulated moments pass Half's 65504 long before the parameters do.
        if (AiDotNet.FederatedLearning.Infrastructure.FederatedReductions.IsNarrowFloat<T>())
        {
            var mWide = _mWide ?? new Vector<float>(n);
            var vWide = _vWide ?? new Vector<float>(n);
            var updatedWide = StepCore(Widen(currentGlobalParameters), Widen(aggregatedTargetParameters), ref mWide, ref vWide);
            _mWide = mWide;
            _vWide = vWide;
            return updatedWide.AsTensor().Cast<T>().ToVector();
        }

        var m = _m ?? new Vector<T>(n);
        var v = _v ?? new Vector<T>(n);
        var updated = StepCore(currentGlobalParameters, aggregatedTargetParameters, ref m, ref v);
        _m = m;
        _v = v;
        return updated;
    }

    private static Vector<float> Widen(Vector<T> v) => v.AsTensor().Cast<float>().ToVector();

    /// <summary>
    /// Reddi et al. 2021, Algorithm 2 (FedAdam), no bias correction: m = b1 m + (1 - b1) d,
    /// v = b2 v + (1 - b2) d^2, x = x + lr m / (sqrt(v) + tau).
    /// </summary>
    private Vector<TV> StepCore<TV>(Vector<TV> x, Vector<TV> target, ref Vector<TV> m, ref Vector<TV> v)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<TV>();
        var g = Engine.Subtract(target, x);
        m = Engine.Add(Engine.Multiply(m, ops.FromDouble(_beta1)), Engine.Multiply(g, ops.FromDouble(1.0 - _beta1)));
        v = Engine.Add(
            Engine.Multiply(v, ops.FromDouble(_beta2)),
            Engine.Multiply(Engine.Multiply(g, g), ops.FromDouble(1.0 - _beta2)));

        var denominator = Engine.TensorAddScalar(Engine.Sqrt(v).AsTensor(), ops.FromDouble(_epsilon)).ToVector();
        var step = Engine.Multiply(Engine.Divide(m, denominator), ops.FromDouble(_learningRate));
        return Engine.Add(x, step);
    }

    /// <inheritdoc />
    public override string GetOptimizerName() => "FedAdam";
}

