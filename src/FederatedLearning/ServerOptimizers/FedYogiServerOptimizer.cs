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

    // The moments are held in double: v accumulates g^2, which overflows Half (65504) once |g| passes 256, and an
    // infinite v freezes every later step at zero.
    private Tensor<double>? _m;
    private Tensor<double>? _v;

    /// <summary>Creates a FedYogi optimizer with the direct-constructor defaults.</summary>
    /// <remarks>The beta2=0.99 and epsilon=0.001 defaults apply only to direct construction.
    /// Trainers using shared server optimizer options use the values supplied by those options.</remarks>
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
        _m ??= new Tensor<double>(new[] { n });
        if (_v is null)
        {
            // Reddi et al. 2021, Algorithm 2: v_{-1} >= tau^2.
            _v = new Tensor<double>(new[] { n });
            _v.Fill(_epsilon * _epsilon);
        }

        // Reddi et al. 2021, Algorithm 2 (FedYogi), no bias correction:
        // m = b1 m + (1 - b1) g;  v = v + (1 - b2) sign(g^2 - v) g^2;  x = x + lr m / (sqrt(v) + tau).
        // Keep the step and addition in double; only the completed parameters are rounded to T.
        var current = currentGlobalParameters.AsTensor().Cast<double>();
        var g = Engine.TensorSubtract(aggregatedTargetParameters.AsTensor().Cast<double>(), current);
        var gSquared = Engine.TensorSquare(g);
        var m = Engine.TensorAdd(Engine.TensorMultiplyScalar(_m, _beta1), Engine.TensorMultiplyScalar(g, 1.0 - _beta1));
        var direction = Engine.TensorSign(Engine.TensorSubtract(gSquared, _v));
        var v = Engine.TensorAdd(_v, Engine.TensorMultiplyScalar(Engine.TensorMultiply(direction, gSquared), 1.0 - _beta2));

        // Kept across rounds, so copied out of the engine's buffers.
        _m = new Tensor<double>(m.ToArray(), new[] { n });
        _v = new Tensor<double>(v.ToArray(), new[] { n });

        var denominator = Engine.TensorAddScalar(Engine.TensorSqrt(_v), _epsilon);
        var step = Engine.TensorMultiplyScalar(Engine.TensorDivide(_m, denominator), _learningRate);
        return Engine.TensorAdd(current, step).Cast<T>().ToVector();
    }

    /// <inheritdoc />
    public override string GetOptimizerName() => "FedYogi";
}
