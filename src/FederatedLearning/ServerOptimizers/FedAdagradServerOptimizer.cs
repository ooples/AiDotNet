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
        _momentum ??= new Vector<T>(n);
        _accumulator ??= new Vector<T>(n);

        // Reddi et al. 2021, Algorithm 2 (FedAdagrad): m_t = b1 m + (1 - b1) d, v_t = v + d^2,
        // x = x + lr * m_t / (sqrt(v_t) + tau).
        var g = Engine.Subtract(aggregatedTargetParameters, currentGlobalParameters);
        _momentum = Engine.Add(
            Engine.Multiply(_momentum, NumOps.FromDouble(_beta1)), Engine.Multiply(g, NumOps.FromDouble(1.0 - _beta1)));
        _accumulator = Engine.Add(_accumulator, Engine.Multiply(g, g));

        var denominator = Engine.TensorAddScalar(Engine.Sqrt(_accumulator).AsTensor(), NumOps.FromDouble(_epsilon)).ToVector();
        var step = Engine.Multiply(Engine.Divide(_momentum, denominator), NumOps.FromDouble(_learningRate));
        return Engine.Add(currentGlobalParameters, step);
    }

    /// <inheritdoc />
    public override string GetOptimizerName() => "FedAdagrad";
}

