using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.FederatedLearning.ServerOptimizers;

/// <summary>
/// FedAdagrad server optimizer — adaptive learning rates using accumulated squared gradients.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Adagrad automatically reduces the learning rate for parameters
/// that have been updated frequently, allowing rarely-updated parameters to learn faster.
/// In federated settings, this helps handle heterogeneous client updates.</para>
///
/// <para>Reference: Reddi, S., et al. (2021). "Adaptive Federated Optimization." ICLR 2021.</para>
/// </remarks>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public sealed class FedAdagradServerOptimizer<T> : FederatedServerOptimizerBase<T>
{
    private readonly double _learningRate;
    private readonly double _beta1;
    private readonly double _epsilon;
    private Vector<T>? _momentum;
    private Vector<T>? _accumulator;
    private Vector<float>? _momentumWide;
    private Vector<float>? _accumulatorWide;

    public FedAdagradServerOptimizer(double learningRate = 1.0, double epsilon = 1e-3, double beta1 = 0.9)
    {
        if (learningRate <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive.");
        }

        if (beta1 < 0.0 || beta1 >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(beta1), "Beta1 must be in [0, 1).");
        }

        if (epsilon <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");
        }

        _learningRate = learningRate;
        _beta1 = beta1;
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
            var mWide = _momentumWide ?? new Vector<float>(n);
            var vWide = _accumulatorWide ?? new Vector<float>(n);
            var updatedWide = StepCore(Widen(currentGlobalParameters), Widen(aggregatedTargetParameters), ref mWide, ref vWide);
            _momentumWide = mWide;
            _accumulatorWide = vWide;
            return updatedWide.AsTensor().Cast<T>().ToVector();
        }

        var m = _momentum ?? new Vector<T>(n);
        var v = _accumulator ?? new Vector<T>(n);
        var updated = StepCore(currentGlobalParameters, aggregatedTargetParameters, ref m, ref v);
        _momentum = m;
        _accumulator = v;
        return updated;
    }

    private static Vector<float> Widen(Vector<T> v) => v.AsTensor().Cast<float>().ToVector();

    /// <summary>
    /// Reddi et al. 2021, Algorithm 2 (FedAdagrad): m = b1 m + (1 - b1) d, v = v + d^2,
    /// x = x + lr m / (sqrt(v) + tau).
    /// </summary>
    private Vector<TV> StepCore<TV>(Vector<TV> x, Vector<TV> target, ref Vector<TV> m, ref Vector<TV> v)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<TV>();
        var g = Engine.Subtract(target, x);
        m = Engine.Add(Engine.Multiply(m, ops.FromDouble(_beta1)), Engine.Multiply(g, ops.FromDouble(1.0 - _beta1)));
        v = Engine.Add(v, Engine.Multiply(g, g));

        var denominator = Engine.TensorAddScalar(Engine.Sqrt(v).AsTensor(), ops.FromDouble(_epsilon)).ToVector();
        var step = Engine.Multiply(Engine.Divide(m, denominator), ops.FromDouble(_learningRate));
        return Engine.Add(x, step);
    }

    /// <inheritdoc />
    public override string GetOptimizerName() => "FedAdagrad";
}

