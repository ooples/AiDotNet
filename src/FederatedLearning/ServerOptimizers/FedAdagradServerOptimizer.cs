using AiDotNet.Interfaces;

namespace AiDotNet.FederatedLearning.ServerOptimizers;

/// <summary>
/// FedAdagrad server optimizer.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> FedAdagradServerOptimizer provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public sealed class FedAdagradServerOptimizer<T> : FederatedServerOptimizerBase<T>
{
    private readonly double _learningRate;
    private readonly double _epsilon;
    private double[]? _accumulator;

    public FedAdagradServerOptimizer(double learningRate = 1.0, double epsilon = 1e-8)
    {
        if (learningRate <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive.");
        }

        if (epsilon <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");
        }

        _learningRate = learningRate;
        _epsilon = epsilon;
    }

    public override Vector<T> Step(Vector<T> currentGlobalParameters, Vector<T> aggregatedTargetParameters)
    {
        ValidateVectors(currentGlobalParameters, aggregatedTargetParameters);

        int n = currentGlobalParameters.Length;
        _accumulator ??= new double[n];

        var updated = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            double current = NumOps.ToDouble(currentGlobalParameters[i]);
            double target = NumOps.ToDouble(aggregatedTargetParameters[i]);
            double g = target - current;

            _accumulator[i] += g * g;
            double step = _learningRate * g / (Math.Sqrt(_accumulator[i]) + _epsilon);
            updated[i] = NumOps.FromDouble(current + step);
        }

        return updated;
    }

    public override string GetOptimizerName() => "FedAdagrad";
}

