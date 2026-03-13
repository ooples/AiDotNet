using AiDotNet.Attributes;
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
[ModelDomain(ModelDomain.General)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelPaper("Adaptive Federated Optimization", "https://arxiv.org/abs/2003.00295", Year = 2021, Authors = "Reddi et al.")]
public sealed class FedYogiServerOptimizer<T> : FederatedServerOptimizerBase<T>
{
    private readonly double _learningRate;
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _epsilon;

    private double[]? _m;
    private double[]? _v;
    private int _t;

    public FedYogiServerOptimizer(double learningRate = 1.0, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
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

    public override Vector<T> Step(Vector<T> currentGlobalParameters, Vector<T> aggregatedTargetParameters)
    {
        ValidateVectors(currentGlobalParameters, aggregatedTargetParameters);

        int n = currentGlobalParameters.Length;
        _m ??= new double[n];
        _v ??= new double[n];
        _t++;

        double bias1 = 1.0 - Math.Pow(_beta1, _t);
        double bias2 = 1.0 - Math.Pow(_beta2, _t);

        var updated = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            double current = NumOps.ToDouble(currentGlobalParameters[i]);
            double target = NumOps.ToDouble(aggregatedTargetParameters[i]);
            double g = target - current;

            _m[i] = (_beta1 * _m[i]) + ((1.0 - _beta1) * g);

            // Yogi second moment update: v = v + (1-beta2) * sign(g^2 - v) * g^2
            _v[i] += (1.0 - _beta2) * Math.Sign((g * g) - _v[i]) * (g * g);

            double mHat = _m[i] / bias1;
            double vHat = _v[i] / bias2;

            double step = _learningRate * mHat / (Math.Sqrt(vHat) + _epsilon);
            updated[i] = NumOps.FromDouble(current + step);
        }

        return updated;
    }

    public override string GetOptimizerName() => "FedYogi";
}
