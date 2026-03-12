using AiDotNet.Interfaces;

namespace AiDotNet.FederatedLearning.ServerOptimizers;

/// <summary>
/// FedAdam server optimizer.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> FedAdamServerOptimizer provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public sealed class FedAdamServerOptimizer<T> : FederatedServerOptimizerBase<T>
{
    private readonly double _learningRate;
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _epsilon;

    private double[]? _m;
    private double[]? _v;
    private int _t;

    public FedAdamServerOptimizer(double learningRate = 1.0, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
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
            _v[i] = (_beta2 * _v[i]) + ((1.0 - _beta2) * g * g);

            double mHat = _m[i] / bias1;
            double vHat = _v[i] / bias2;

            double step = _learningRate * mHat / (Math.Sqrt(vHat) + _epsilon);
            updated[i] = NumOps.FromDouble(current + step);
        }

        return updated;
    }

    public override string GetOptimizerName() => "FedAdam";
}

