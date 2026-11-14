using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;
using AiDotNet.Validation;

using AiDotNet.ReinforcementLearning.Parameters;

namespace AiDotNet.ReinforcementLearning.Agents.Bandits;

/// <summary>
/// Gradient Bandit agent using softmax action preferences.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Instead of estimating how good each action is (like epsilon-greedy),
/// the gradient bandit learns preferences for each action using gradient ascent. Actions with
/// higher preferences are selected more often via softmax probabilities. When an action does
/// better than average, its preference increases; when worse, it decreases. This approach
/// naturally handles the exploration-exploitation trade-off through the softmax distribution
/// without needing an explicit epsilon parameter.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a gradient bandit agent using softmax action selection
/// var options = new GradientBanditOptions&lt;double&gt; { NumArms = 10};
/// var agent = new GradientBanditAgent&lt;double&gt;(options);
///
/// // Select an arm based on learned preferences
/// var state = new Vector&lt;double&gt;(new double[] { 1.0 });
/// var action = agent.SelectAction(state);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Reinforcement Learning: An Introduction",
    "https://incompleteideas.net/book/the-book-2nd.html",
    Year = 2018,
    Authors = "Sutton, R. S. & Barto, A. G.")]
public partial class GradientBanditAgent<T> : ReinforcementLearningAgentBase<T>
{

    /// <inheritdoc />
    /// <remarks>The per-arm preferences this bandit learns.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(new VectorParameterSource<T>(() => _preferences));
    }
    private GradientBanditOptions<T> _options;

    private Random _random;
    private Vector<T> _preferences;  // H(a)
    private T _averageReward;
    private int _totalSteps;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public GradientBanditAgent()
        : this(new GradientBanditOptions<T>())
    {
    }

    public GradientBanditAgent(GradientBanditOptions<T> options) : base(options)
    {
        Guard.NotNull(options);
        _options = options;
        Options = _options;
        _random = Random;
        _preferences = Vector<T>.CreateDefault(_options.NumArms, NumOps.Zero);
        _averageReward = NumOps.Zero;
        _totalSteps = 0;
    }

    /// <inheritdoc />
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var result = new Vector<T>(_options.NumArms);

        if (!training)
        {
            // Evaluation: act greedily on the learned preferences. This makes the policy
            // deterministic (Predict called twice returns the same arm) and clone-stable
            // (a clone with identical preferences picks the same arm). Stochastic softmax
            // sampling below is only used while training/exploring.
            result[ArgMax(_preferences)] = NumOps.One;
            return result;
        }

        // Compute softmax probabilities
        var probs = ComputeSoftmax(_preferences);

        // Sample action according to probabilities
        double r = _random.NextDouble();
        int selectedArm = RLTensorOps.SampleFromCdf(
            probs.GetSubVector(0, Math.Min(probs.Length, _options.NumArms)), r, 0);

        result[selectedArm] = NumOps.One;
        return result;
    }

    /// <inheritdoc />
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        int armIndex = ArgMax(action);
        _totalSteps++;

        // Update average reward baseline. Per Sutton & Barto 2018 §2.8, R̄_t is the average of all
        // rewards up through and including time t, so it is updated before the preference step. A
        // consequence — by design, not a bug — is that a CONSTANT reward stream produces (R − R̄) = 0
        // and therefore no preference change: if every pull returns the same reward there is nothing
        // to learn and the softmax policy correctly stays uniform.
        if (_options.UseBaseline)
        {
            T alpha = NumOps.Divide(NumOps.One, NumOps.FromDouble(_totalSteps));
            T delta = NumOps.Subtract(reward, _averageReward);
            _averageReward = NumOps.Add(_averageReward, NumOps.Multiply(alpha, delta));
        }

        // Compute softmax probabilities
        var probs = ComputeSoftmax(_preferences);

        // Gradient update: H(a) ← H(a) + α(R - R̄)(1 - π(a)) for selected action
        //                  H(a) ← H(a) - α(R - R̄)π(a) for other actions
        T rewardDiff = NumOps.Subtract(reward, _averageReward);
        T stepSize = NumOps.FromDouble(_options.Alpha);

        // Both rules at once: H += alpha * (R - R_bar) * (onehot(a) - pi).
        var engine = AiDotNetEngine.Current;
        var indicator = new Vector<T>(_options.NumArms);
        indicator[armIndex] = NumOps.One;
        var direction = engine.Subtract(indicator, probs.GetSubVector(0, _options.NumArms));
        var update = engine.Multiply(engine.Multiply(direction, rewardDiff), stepSize);
        engine.Add(_preferences.GetSubVector(0, _options.NumArms), update).AsSpan().CopyTo(_preferences.AsWritableSpan());
    }

    private Vector<T> ComputeSoftmax(Vector<T> preferences)
    {
        // Find max for numerical stability
        // exp(H(a) - max H) / sum exp(H - max H)
        return RLTensorOps.StableSoftmax(preferences);
    }

    /// <inheritdoc />
    public override T Train() => NumOps.Zero;

    private int ArgMax(Vector<T> values)
    {
        return RLTensorOps.ArgMax(values);
    }

    /// <inheritdoc />
    public override Dictionary<string, T> GetMetrics()
    {
        var metrics = new Dictionary<string, T>();
        var probs = ComputeSoftmax(_preferences);
        RLTensorOps.AddIndexedMetrics(metrics, _options.NumArms,
            ("preference_arm_", _preferences), ("probability_arm_", probs));
        metrics["average_reward"] = _averageReward;
        return metrics;
    }

    /// <inheritdoc />
    public override void ResetEpisode()
    {
        _preferences.Fill(NumOps.Zero);
        _averageReward = NumOps.Zero;
        _totalSteps = 0;
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Vector<T> input) => SelectAction(input, false);
    public Task<Vector<T>> PredictAsync(Vector<T> input) => Task.FromResult(Predict(input));
    public Task TrainAsync() { Train(); return Task.CompletedTask; }
    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T> { FeatureCount = this.FeatureCount, Complexity = ParameterCount };
    /// <inheritdoc />
    public override int FeatureCount => 1;
    /// <inheritdoc />
    public override void SaveModel(string filepath) { var data = Serialize(); System.IO.File.WriteAllBytes(filepath, data); }
    /// <inheritdoc />
    public override void LoadModel(string filepath) { var data = System.IO.File.ReadAllBytes(filepath); Deserialize(data); }
}
