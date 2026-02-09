using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;

namespace AiDotNet.ReinforcementLearning.Agents.Bandits;

/// <summary>
/// Thompson Sampling (Bayesian) Multi-Armed Bandit agent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ThompsonSamplingAgent<T> : ReinforcementLearningAgentBase<T>
{
    private ThompsonSamplingOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Random _random;
    private Vector<int> _successCounts;
    private Vector<int> _failureCounts;

    public ThompsonSamplingAgent(ThompsonSamplingOptions<T> options) : base(options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _random = RandomHelper.CreateSecureRandom();
        _successCounts = new Vector<int>(_options.NumArms);
        _failureCounts = new Vector<int>(_options.NumArms);
        for (int i = 0; i < _options.NumArms; i++)
        {
            _successCounts[i] = 1;  // Prior
            _failureCounts[i] = 1;  // Prior
        }
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        // Sample from Beta distribution for each arm
        int selectedArm = 0;
        double maxSample = double.NegativeInfinity;

        for (int a = 0; a < _options.NumArms; a++)
        {
            // Sample from Beta(successes, failures)
            double sample = SampleBeta(_successCounts[a], _failureCounts[a]);
            if (sample > maxSample)
            {
                maxSample = sample;
                selectedArm = a;
            }
        }

        var result = new Vector<T>(_options.NumArms);
        result[selectedArm] = NumOps.One;
        return result;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        int armIndex = ArgMax(action);
        double rewardValue = NumOps.ToDouble(reward);

        // Update Beta distribution parameters
        if (rewardValue > 0.5)  // Treat as success
        {
            _successCounts[armIndex]++;
        }
        else  // Treat as failure
        {
            _failureCounts[armIndex]++;
        }
    }

    private double SampleBeta(int alpha, int beta)
    {
        // Simplified Beta sampling using Gamma distribution ratio
        double x = SampleGamma(alpha);
        double y = SampleGamma(beta);
        return x / (x + y);
    }

    private double SampleGamma(int shape)
    {
        // Simplified Gamma sampling for integer shape parameter
        double sum = 0.0;
        for (int i = 0; i < shape; i++)
        {
            // Ensure NextDouble() never returns exactly 0 to avoid -infinity in log
            double u = _random.NextDouble();
            while (u == 0.0)
            {
                u = _random.NextDouble();
            }
            sum += -Math.Log(u);
        }
        return sum;
    }

    public override T Train() => NumOps.Zero;

    private int ArgMax(Vector<T> values)
    {
        int maxIndex = 0;
        T maxValue = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (NumOps.GreaterThan(values[i], maxValue))
            {
                maxValue = values[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public override Dictionary<string, T> GetMetrics()
    {
        var metrics = new Dictionary<string, T>();
        for (int i = 0; i < _options.NumArms; i++)
        {
            double mean = (double)_successCounts[i] / (_successCounts[i] + _failureCounts[i]);
            metrics[$"mean_arm_{i}"] = NumOps.FromDouble(mean);
            metrics[$"successes_arm_{i}"] = NumOps.FromDouble(_successCounts[i]);
            metrics[$"failures_arm_{i}"] = NumOps.FromDouble(_failureCounts[i]);
        }
        return metrics;
    }

    public override void ResetEpisode()
    {
        for (int i = 0; i < _options.NumArms; i++)
        {
            _successCounts[i] = 1;
            _failureCounts[i] = 1;
        }
    }

    public override Vector<T> Predict(Vector<T> input) => SelectAction(input, false);
    public Task<Vector<T>> PredictAsync(Vector<T> input) => Task.FromResult(Predict(input));
    public Task TrainAsync() { Train(); return Task.CompletedTask; }
    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T> { ModelType = ModelType.ReinforcementLearning, FeatureCount = this.FeatureCount, Complexity = ParameterCount };
    public override int ParameterCount => _options.NumArms * 2;
    public override int FeatureCount => 1;
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Write options
        writer.Write(_options.NumArms);

        // Write state
        for (int i = 0; i < _options.NumArms; i++)
        {
            writer.Write(_successCounts[i]);
            writer.Write(_failureCounts[i]);
        }

        return ms.ToArray();
    }

    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Read and validate options
        var numArms = reader.ReadInt32();

        if (numArms != _options.NumArms)
            throw new InvalidOperationException($"Serialized NumArms ({numArms}) doesn't match current options ({_options.NumArms})");

        // Read state
        for (int i = 0; i < _options.NumArms; i++)
        {
            _successCounts[i] = reader.ReadInt32();
            _failureCounts[i] = reader.ReadInt32();
        }
    }
    public override Vector<T> GetParameters()
    {
        int paramCount = _options.NumArms * 2; // success and failure counts for each arm
        var v = new Vector<T>(paramCount);
        int idx = 0;
        for (int i = 0; i < _options.NumArms; i++)
        {
            v[idx++] = NumOps.FromDouble(_successCounts[i]);
            v[idx++] = NumOps.FromDouble(_failureCounts[i]);
        }
        return v;
    }
    public override void SetParameters(Vector<T> parameters) { int idx = 0; for (int i = 0; i < _options.NumArms && idx + 1 < parameters.Length; i++) { _successCounts[i] = (int)NumOps.ToDouble(parameters[idx++]); _failureCounts[i] = (int)NumOps.ToDouble(parameters[idx++]); } }
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new ThompsonSamplingAgent<T>(_options);
        // Copy learned arm statistics to preserve trained state
        for (int i = 0; i < _options.NumArms; i++)
        {
            clone._successCounts[i] = _successCounts[i];
            clone._failureCounts[i] = _failureCounts[i];
        }
        return clone;
    }
    public override Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null) { var pred = Predict(input); var lf = lossFunction ?? LossFunction; var predMatrix = new Matrix<T>(new[] { pred }); var targetMatrix = new Matrix<T>(new[] { target }); var loss = lf.CalculateLoss(predMatrix.GetRow(0), targetMatrix.GetRow(0)); var grad = lf.CalculateDerivative(predMatrix.GetRow(0), targetMatrix.GetRow(0)); return grad; }
    public override void ApplyGradients(Vector<T> gradients, T learningRate) { }
    public override void SaveModel(string filepath) { var data = Serialize(); System.IO.File.WriteAllBytes(filepath, data); }
    public override void LoadModel(string filepath) { var data = System.IO.File.ReadAllBytes(filepath); Deserialize(data); }
}
