using AiDotNet.Attributes;
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
[ModelDomain(ModelDomain.General)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelPaper("Adaptive Federated Optimization", "https://arxiv.org/abs/2003.00295", Year = 2021, Authors = "Reddi et al.")]
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

