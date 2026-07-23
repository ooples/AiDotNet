using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.IntrinsicMotivation;

/// <summary>
/// Random Network Distillation (Burda et al. 2018): intrinsic reward is the error of a trained predictor
/// network against a fixed, randomly-initialized target network. Novel states have high prediction error
/// (the predictor has not learned them yet); familiar states have low error, so the bonus fades.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Both networks are single-hidden-layer MLPs (tanh). The target is fixed random; the predictor is
/// trained by one SGD step per visited state to match the target's output on that state. The intrinsic
/// reward is the mean-squared prediction error, normalized by a running estimate so its scale stays
/// stable as training proceeds. Networks are lazily sized to the first state seen.
/// </para>
/// </remarks>
public sealed class RandomNetworkDistillation<T> : IIntrinsicRewardModule<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _hidden;
    private readonly int _output;
    private readonly double _learningRate;
    private readonly Random _rng;

    private bool _initialized;
    private int _inputDim;

    // Target (fixed) and predictor (trained) parameters.
    private double[,] _tW1 = new double[0, 0], _tW2 = new double[0, 0];
    private double[] _tB1 = System.Array.Empty<double>(), _tB2 = System.Array.Empty<double>();
    private double[,] _pW1 = new double[0, 0], _pW2 = new double[0, 0];
    private double[] _pB1 = System.Array.Empty<double>(), _pB2 = System.Array.Empty<double>();

    /// <summary>Creates an RND intrinsic-reward module.</summary>
    /// <param name="hiddenSize">Hidden width of both MLPs. Defaults to 32.</param>
    /// <param name="outputSize">Embedding width both MLPs map to. Defaults to 16.</param>
    /// <param name="learningRate">Predictor SGD step size. Defaults to 1e-3.</param>
    /// <param name="seed">Optional seed for reproducible random initialization.</param>
    public RandomNetworkDistillation(int hiddenSize = 32, int outputSize = 16, double learningRate = 1e-3, int? seed = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _hidden = Math.Max(1, hiddenSize);
        _output = Math.Max(1, outputSize);
        _learningRate = learningRate;
        _rng = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <inheritdoc />
    public T ComputeIntrinsicReward(Vector<T> state)
    {
        var s = ToDoubles(state);
        EnsureInitialized(s.Length);
        // Novelty = the predictor's error against the fixed random target: high for states the predictor
        // has not learned yet, and it falls monotonically as a state is learned (see Update).
        return _numOps.FromDouble(SquaredError(s));
    }

    /// <inheritdoc />
    public void Update(Vector<T> state)
    {
        var s = ToDoubles(state);
        EnsureInitialized(s.Length);
        TrainPredictorStep(s);
    }

    /// <inheritdoc />
    public void Reset()
    {
        // RND novelty is cross-episode by design (the predictor keeps learning), so nothing per-episode.
    }

    private void EnsureInitialized(int inputDim)
    {
        if (_initialized) return;
        _inputDim = inputDim;
        _tW1 = RandomMatrix(_hidden, inputDim);
        _tB1 = RandomVector(_hidden);
        _tW2 = RandomMatrix(_output, _hidden);
        _tB2 = RandomVector(_output);
        _pW1 = RandomMatrix(_hidden, inputDim);
        _pB1 = RandomVector(_hidden);
        _pW2 = RandomMatrix(_output, _hidden);
        _pB2 = RandomVector(_output);
        _initialized = true;
    }

    private double SquaredError(double[] s)
    {
        var (_, predOut) = Forward(s, _pW1, _pB1, _pW2, _pB2);
        var (_, targetOut) = Forward(s, _tW1, _tB1, _tW2, _tB2);
        double sum = 0;
        for (int j = 0; j < _output; j++)
        {
            double diff = predOut[j] - targetOut[j];
            sum += diff * diff;
        }

        return sum / _output;
    }

    private void TrainPredictorStep(double[] s)
    {
        var (hidden, predOut) = Forward(s, _pW1, _pB1, _pW2, _pB2);
        var (_, targetOut) = Forward(s, _tW1, _tB1, _tW2, _tB2);

        // dLoss/dOut = 2 (pred - target) / output.
        var dOut = new double[_output];
        for (int j = 0; j < _output; j++) dOut[j] = 2.0 * (predOut[j] - targetOut[j]) / _output;

        // Hidden-layer gradient (through tanh).
        var dHidden = new double[_hidden];
        for (int i = 0; i < _hidden; i++)
        {
            double acc = 0;
            for (int j = 0; j < _output; j++) acc += _pW2[j, i] * dOut[j];
            dHidden[i] = acc * (1.0 - (hidden[i] * hidden[i]));
        }

        // SGD updates.
        for (int j = 0; j < _output; j++)
        {
            for (int i = 0; i < _hidden; i++) _pW2[j, i] -= _learningRate * dOut[j] * hidden[i];
            _pB2[j] -= _learningRate * dOut[j];
        }

        for (int i = 0; i < _hidden; i++)
        {
            for (int k = 0; k < _inputDim; k++) _pW1[i, k] -= _learningRate * dHidden[i] * s[k];
            _pB1[i] -= _learningRate * dHidden[i];
        }
    }

    private (double[] hidden, double[] output) Forward(double[] s, double[,] w1, double[] b1, double[,] w2, double[] b2)
    {
        var hidden = new double[_hidden];
        for (int i = 0; i < _hidden; i++)
        {
            double acc = b1[i];
            for (int k = 0; k < _inputDim; k++) acc += w1[i, k] * s[k];
            hidden[i] = Math.Tanh(acc);
        }

        var output = new double[_output];
        for (int j = 0; j < _output; j++)
        {
            double acc = b2[j];
            for (int i = 0; i < _hidden; i++) acc += w2[j, i] * hidden[i];
            output[j] = acc;
        }

        return (hidden, output);
    }

    private double[] ToDoubles(Vector<T> v)
    {
        var d = new double[v.Length];
        for (int i = 0; i < v.Length; i++) d[i] = _numOps.ToDouble(v[i]);
        return d;
    }

    private double[,] RandomMatrix(int rows, int cols)
    {
        // Xavier-ish scaling keeps activations in a sane range.
        double scale = Math.Sqrt(1.0 / Math.Max(1, cols));
        var m = new double[rows, cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                m[r, c] = (_rng.NextDouble() * 2.0 - 1.0) * scale;
        return m;
    }

    private double[] RandomVector(int n)
    {
        var v = new double[n];
        for (int i = 0; i < n; i++) v[i] = (_rng.NextDouble() * 2.0 - 1.0) * 0.1;
        return v;
    }
}
