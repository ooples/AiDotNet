using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.ReinforcementLearning.Agents.Bandits;

/// <summary>
/// Thompson Sampling (Bayesian) Multi-Armed Bandit agent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ThompsonSamplingAgent<T> : ReinforcementLearningAgentBase<T>
{
    private ThompsonSamplingOptions<T> _options;
    private Vector<int> _successCounts;
    private Vector<int> _failureCounts;

    public ThompsonSamplingAgent(ThompsonSamplingOptions<T> options) : base(options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
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
            sum += -Math.Log(Random.NextDouble());
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
    public override Task<Vector<T>> PredictAsync(Vector<T> input) => Task.FromResult(Predict(input));
    public override Task TrainAsync() { Train(); return Task.CompletedTask; }
    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T> { ModelType = ModelType.ReinforcementLearning, FeatureCount = this.FeatureCount, Complexity = ParameterCount };
    public override int ParameterCount => _options.NumArms * 2;
    public override int FeatureCount => 1;
    public override byte[] Serialize() => throw new NotImplementedException();
    public override void Deserialize(byte[] data) => throw new NotImplementedException();
    public override Matrix<T> GetParameters() { var p = new List<T>(); for (int i = 0; i < _options.NumArms; i++) { p.Add(NumOps.FromDouble(_successCounts[i])); p.Add(NumOps.FromDouble(_failureCounts[i])); } var v = new Vector<T>(p.Count); for (int i = 0; i < p.Count; i++) v[i] = p[i]; return new Matrix<T>(new[] { v }); }
    public override void SetParameters(Matrix<T> parameters) { int idx = 0; for (int i = 0; i < _options.NumArms && idx + 1 < parameters.Columns; i++) { _successCounts[i] = (int)NumOps.ToDouble(parameters[0, idx++]); _failureCounts[i] = (int)NumOps.ToDouble(parameters[0, idx++]); } }
    public override IFullModel<T, Vector<T>, Vector<T>> Clone() => new ThompsonSamplingAgent<T>(_options);
    public override (Matrix<T> Gradients, T Loss) ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null) { var pred = Predict(input); var lf = lossFunction ?? LossFunction; var loss = lf.CalculateLoss(new Matrix<T>(new[] { pred }), new Matrix<T>(new[] { target })); var grad = lf.CalculateDerivative(new Matrix<T>(new[] { pred }), new Matrix<T>(new[] { target })); return (grad, loss); }
    public override void ApplyGradients(Matrix<T> gradients, T learningRate) { }
    public override void Save(string filepath) { var data = Serialize(); System.IO.File.WriteAllBytes(filepath, data); }
    public override void Load(string filepath) { var data = System.IO.File.ReadAllBytes(filepath); Deserialize(data); }
}
