using AiDotNet.Attributes;
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
[ModelDomain(ModelDomain.General)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Low)]
public sealed class FedAvgMServerOptimizer<T> : FederatedServerOptimizerBase<T>
{
    private readonly double _learningRate;
    private readonly double _momentum;
    private double[]? _velocity;

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

    public override Vector<T> Step(Vector<T> currentGlobalParameters, Vector<T> aggregatedTargetParameters)
    {
        ValidateVectors(currentGlobalParameters, aggregatedTargetParameters);

        int n = currentGlobalParameters.Length;
        _velocity ??= new double[n];

        var updated = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            double current = NumOps.ToDouble(currentGlobalParameters[i]);
            double target = NumOps.ToDouble(aggregatedTargetParameters[i]);
            double delta = target - current;

            _velocity[i] = (_momentum * _velocity[i]) + delta;
            double next = current + (_learningRate * _velocity[i]);
            updated[i] = NumOps.FromDouble(next);
        }

        return updated;
    }

    public override string GetOptimizerName() => "FedAvgM";
}

