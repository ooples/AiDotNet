using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.FederatedLearning.ServerOptimizers;

/// <summary>
/// FedAvgM server optimizer — server-side momentum for stabilized federated averaging.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Momentum helps smooth updates across rounds. Instead of applying only the
/// current round's update, the server maintains a running "velocity" that accumulates updates.
/// This reduces oscillations caused by heterogeneous client data.</para>
///
/// <para>Reference: Hsu, T.-M. H., et al. (2019). "Measuring the Effects of Non-Identical Data
/// Distribution for Federated Visual Classification." NeurIPS Workshop 2019.</para>
/// </remarks>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public sealed class FedAvgMServerOptimizer<T> : FederatedServerOptimizerBase<T>
{
    private readonly double _learningRate;
    private readonly double _momentum;
    private Vector<T>? _velocity;
    private Vector<float>? _velocityWide;

    public FedAvgMServerOptimizer(double learningRate = 1.0, double momentum = 0.9)
    {
        if (learningRate <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive.");
        }

        if (momentum < 0.0 || momentum >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(momentum), "Momentum must be in [0, 1).");
        }

        _learningRate = learningRate;
        _momentum = momentum;
    }

    /// <inheritdoc />
    public override Vector<T> Step(Vector<T> currentGlobalParameters, Vector<T> aggregatedTargetParameters)
    {
        ValidateVectors(currentGlobalParameters, aggregatedTargetParameters);

        int n = currentGlobalParameters.Length;

        // Half / BFloat16 keep their optimizer state in float (as the previous double state did): the velocity
        // (up to 1 / (1 - momentum) times the delta) passes Half's 65504 long before the parameters do.
        if (AiDotNet.FederatedLearning.Infrastructure.FederatedReductions.IsNarrowFloat<T>())
        {
            var velocityWide = _velocityWide ?? new Vector<float>(n);
            var updatedWide = StepCore(
                currentGlobalParameters.AsTensor().Cast<float>().ToVector(),
                aggregatedTargetParameters.AsTensor().Cast<float>().ToVector(),
                ref velocityWide);
            _velocityWide = velocityWide;
            return updatedWide.AsTensor().Cast<T>().ToVector();
        }

        var velocity = _velocity ?? new Vector<T>(n);
        var updated = StepCore(currentGlobalParameters, aggregatedTargetParameters, ref velocity);
        _velocity = velocity;
        return updated;
    }

    /// <summary>velocity = momentum * velocity + delta; x = x + lr * velocity (Hsu et al. 2019).</summary>
    private Vector<TV> StepCore<TV>(Vector<TV> x, Vector<TV> target, ref Vector<TV> velocity)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<TV>();
        var delta = Engine.Subtract(target, x);
        velocity = Engine.Add(Engine.Multiply(velocity, ops.FromDouble(_momentum)), delta);
        return Engine.Add(x, Engine.Multiply(velocity, ops.FromDouble(_learningRate)));
    }

    /// <inheritdoc />
    public override string GetOptimizerName() => "FedAvgM";
}

