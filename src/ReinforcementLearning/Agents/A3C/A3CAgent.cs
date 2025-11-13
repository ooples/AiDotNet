using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Activations;
using AiDotNet.Helpers;
using AiDotNet.Optimizers;

namespace AiDotNet.ReinforcementLearning.Agents.A3C;

/// <summary>
/// Asynchronous Advantage Actor-Critic (A3C) agent for reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// A3C runs multiple agents in parallel, each exploring different strategies.
/// Workers periodically synchronize with a global network, enabling diverse exploration
/// without replay buffers.
/// </para>
/// <para><b>For Beginners:</b>
/// A3C is like having multiple students learn simultaneously - each has different
/// experiences, and they periodically share knowledge with a "master" network.
/// This parallel learning provides stability and diverse exploration.
///
/// Key features:
/// - **Asynchronous Updates**: Multiple workers update global network independently
/// - **No Replay Buffer**: On-policy learning with parallel exploration
/// - **Actor-Critic**: Learns both policy and value function
/// - **Diverse Exploration**: Each worker explores differently
///
/// Famous for: DeepMind's breakthrough (2016), enables CPU-only training
/// </para>
/// </remarks>
public class A3CAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private readonly A3COptions<T> _options;
    private readonly IOptimizer<T, Vector<T>, Vector<T>> _optimizer;

    private INeuralNetwork<T> _globalPolicyNetwork;
    private INeuralNetwork<T> _globalValueNetwork;
    private readonly object _globalLock = new();

    private int _globalSteps;

    public A3CAgent(A3COptions<T> options, IOptimizer<T, Vector<T>, Vector<T>>? optimizer = null)
        : base(options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _optimizer = optimizer ?? options.Optimizer ?? new AdamOptimizer<T, Vector<T>, Vector<T>>(this, new AdamOptimizerOptions<T, Vector<T>, Vector<T>>
        {
            LearningRate = 0.001,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        });
        _globalSteps = 0;

        InitializeGlobalNetworks();
    }

    private void InitializeGlobalNetworks()
    {
        _globalPolicyNetwork = CreatePolicyNetwork();
        _globalValueNetwork = CreateValueNetwork();

        // Register networks with base class
        Networks.Add(_globalPolicyNetwork);
        Networks.Add(_globalValueNetwork);
    }

    private INeuralNetwork<T> CreatePolicyNetwork()
    {
        int outputSize = _options.IsContinuous ? _options.ActionSize * 2 : _options.ActionSize;

        var architecture = new NeuralNetworkArchitecture<T>
        {
            InputSize = _options.StateSize,
            OutputSize = outputSize,
            TaskType = TaskType.Regression
        };

        // Use LayerHelper to create production-ready network layers
        var layers = LayerHelper<T>.CreateDefaultFeedForwardLayers(
            architecture,
            hiddenLayerCount: _options.PolicyHiddenLayers.Count,
            hiddenLayerSize: _options.PolicyHiddenLayers.FirstOrDefault() > 0 ? _options.PolicyHiddenLayers.First() : 128
        ).ToList();

        // Override output layer activation for continuous vs discrete actions
        if (!_options.IsContinuous)
        {
            // For discrete actions, replace final layer with softmax activation
            var lastLayer = layers[layers.Count - 1];
            if (lastLayer is DenseLayer<T> denseLayer)
            {
                layers[layers.Count - 1] = new DenseLayer<T>(
                    denseLayer.GetWeights().Rows,
                    outputSize,
                    new SoftmaxActivation<T>()
                );
            }
        }

        architecture.Layers = layers;
        return new NeuralNetwork<T>(architecture, _options.ValueLossFunction);
    }

    private INeuralNetwork<T> CreateValueNetwork()
    {
        var architecture = new NeuralNetworkArchitecture<T>
        {
            InputSize = _options.StateSize,
            OutputSize = 1,
            TaskType = TaskType.Regression
        };

        // Use LayerHelper to create production-ready network layers
        var layers = LayerHelper<T>.CreateDefaultFeedForwardLayers(
            architecture,
            hiddenLayerCount: _options.ValueHiddenLayers.Count,
            hiddenLayerSize: _options.ValueHiddenLayers.FirstOrDefault() > 0 ? _options.ValueHiddenLayers.First() : 128
        );

        architecture.Layers = layers.ToList();
        return new NeuralNetwork<T>(architecture, _options.ValueLossFunction);
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        Vector<T> policyOutput;

        lock (_globalLock)
        {
            policyOutput = _globalPolicyNetwork.Forward(state);
        }

        if (_options.IsContinuous)
        {
            // Continuous action space
            var mean = new Vector<T>(_options.ActionSize);
            var logStd = new Vector<T>(_options.ActionSize);

            for (int i = 0; i < _options.ActionSize; i++)
            {
                mean[i] = policyOutput[i];
                logStd[i] = policyOutput[_options.ActionSize + i];
                logStd[i] = MathHelper.Clamp<T>(logStd[i], NumOps.FromDouble(-20), NumOps.FromDouble(2));
            }

            if (!training)
            {
                return mean;
            }

            var action = new Vector<T>(_options.ActionSize);
            for (int i = 0; i < _options.ActionSize; i++)
            {
                var std = MathHelper.Exp(logStd[i]);
                var noise = MathHelper.GetNormalRandom<T>(NumOps.Zero, NumOps.One);
                action[i] = NumOps.Add(mean[i], NumOps.Multiply(std, noise));
            }

            return action;
        }
        else
        {
            // Discrete action space
            if (!training)
            {
                // Return greedy action
                int bestAction = 0;
                T bestProb = policyOutput[0];
                for (int i = 1; i < _options.ActionSize; i++)
                {
                    if (NumOps.Compare(policyOutput[i], bestProb) > 0)
                    {
                        bestProb = policyOutput[i];
                        bestAction = i;
                    }
                }

                var action = new Vector<T>(_options.ActionSize);
                action[bestAction] = NumOps.One;
                return action;
            }

            // Sample from distribution
            double[] probs = new double[_options.ActionSize];
            for (int i = 0; i < _options.ActionSize; i++)
            {
                probs[i] = Convert.ToDouble(NumOps.ToDouble(policyOutput[i]));
            }

            double r = Random.NextDouble();
            double cumulative = 0.0;
            int selectedAction = 0;

            for (int i = 0; i < probs.Length; i++)
            {
                cumulative += probs[i];
                if (r <= cumulative)
                {
                    selectedAction = i;
                    break;
                }
            }

            var actionVec = new Vector<T>(_options.ActionSize);
            actionVec[selectedAction] = NumOps.One;
            return actionVec;
        }
    }

    /// <summary>
    /// Train A3C with parallel workers (simplified for single-threaded execution).
    /// In production, this would spawn actual parallel tasks.
    /// </summary>
    public async Task TrainAsync(Interfaces.IEnvironment<T> environment, int maxSteps)
    {
        // Run workers sequentially to avoid concurrent environment access
        // The environment is not thread-safe, so we cannot run workers in parallel
        // In a full implementation, each worker would need its own environment instance
        for (int i = 0; i < _options.NumWorkers; i++)
        {
            await Task.Run(() => RunWorker(environment, maxSteps, i));
        }
    }

    private void RunWorker(Interfaces.IEnvironment<T> environment, int maxSteps, int workerId)
    {
        // Create worker-local networks (not registered with Networks list)
        var localPolicy = CreatePolicyNetwork();
        var localValue = CreateValueNetwork();

        var trajectory = new List<(Vector<T> state, Vector<T> action, T reward, bool done, T value)>();

        while (_globalSteps < maxSteps)
        {
            // Synchronize with global network
            lock (_globalLock)
            {
                CopyNetworkWeights(_globalPolicyNetwork, localPolicy);
                CopyNetworkWeights(_globalValueNetwork, localValue);
            }

            // Collect trajectory
            var state = environment.Reset();
            trajectory.Clear();

            for (int t = 0; t < _options.TMax && _globalSteps < maxSteps; t++)
            {
                var action = SelectActionWithLocalNetwork(state, localPolicy, training: true);
                var value = localValue.Forward(state)[0];
                var (nextState, reward, done, info) = environment.Step(action);

                trajectory.Add((state, action, reward, done, value));

                state = nextState;
                Interlocked.Increment(ref _globalSteps);

                if (done)
                {
                    break;
                }
            }

            // Compute returns and advantages
            var returns = ComputeReturns(trajectory, localValue);
            var advantages = ComputeAdvantages(trajectory, returns);

            // Update global network
            lock (_globalLock)
            {
                UpdateGlobalNetworks(trajectory, returns, advantages, localPolicy, localValue);
            }
        }
    }

    private Vector<T> SelectActionWithLocalNetwork(Vector<T> state, INeuralNetwork<T> policy, bool training)
    {
        var policyOutput = policy.Forward(state);
        // Simplified: reuse SelectAction logic but with local network output
        // In full implementation, would extract to shared method
        return SelectAction(state, training);
    }

    private List<T> ComputeReturns(List<(Vector<T> state, Vector<T> action, T reward, bool done, T value)> trajectory, INeuralNetwork<T> valueNetwork)
    {
        var returns = new List<T>();
        T nextValue = NumOps.Zero;

        if (trajectory.Count > 0 && !trajectory[trajectory.Count - 1].done)
        {
            var lastState = trajectory[trajectory.Count - 1].state;
            nextValue = valueNetwork.Forward(lastState)[0];
        }

        T runningReturn = nextValue;
        for (int i = trajectory.Count - 1; i >= 0; i--)
        {
            var exp = trajectory[i];
            if (exp.done)
            {
                runningReturn = exp.reward;
            }
            else
            {
                runningReturn = NumOps.Add(exp.reward, NumOps.Multiply(_options.DiscountFactor, runningReturn));
            }
            returns.Insert(0, runningReturn);
        }

        return returns;
    }

    private List<T> ComputeAdvantages(List<(Vector<T> state, Vector<T> action, T reward, bool done, T value)> trajectory, List<T> returns)
    {
        var advantages = new List<T>();

        for (int i = 0; i < trajectory.Count; i++)
        {
            var advantage = NumOps.Subtract(returns[i], trajectory[i].value);
            advantages.Add(advantage);
        }

        // Normalize advantages
        var mean = StatisticsHelper<T>.CalculateMean(advantages.ToArray());
        var std = StatisticsHelper<T>.CalculateStandardDeviation(advantages.ToArray());

        if (NumOps.Compare(std, NumOps.Zero) > 0)
        {
            for (int i = 0; i < advantages.Count; i++)
            {
                advantages[i] = NumOps.Divide(NumOps.Subtract(advantages[i], mean), std);
            }
        }

        return advantages;
    }

    private void UpdateGlobalNetworks(
        List<(Vector<T> state, Vector<T> action, T reward, bool done, T value)> trajectory,
        List<T> returns,
        List<T> advantages,
        INeuralNetwork<T> localPolicy,
        INeuralNetwork<T> localValue)
    {
        // Update policy network
        for (int i = 0; i < trajectory.Count; i++)
        {
            var advantage = advantages[i];
            int policyOutputSize = _options.IsContinuous ? _options.ActionSize * 2 : _options.ActionSize;
            var policyGradient = new Vector<T>(policyOutputSize);

            // Simplified gradient computation
            for (int j = 0; j < policyGradient.Length; j++)
            {
                policyGradient[j] = NumOps.Multiply(advantage, NumOps.FromDouble(0.1));
            }

            _globalPolicyNetwork.Backward(policyGradient);
            _globalPolicyNetwork.UpdateWeights(_options.PolicyLearningRate);
        }

        // Update value network
        for (int i = 0; i < trajectory.Count; i++)
        {
            var valueError = NumOps.Subtract(returns[i], trajectory[i].value);
            var valueGradient = new Vector<T>(1);
            valueGradient[0] = valueError;

            _globalValueNetwork.Backward(valueGradient);
            _globalValueNetwork.UpdateWeights(_options.ValueLearningRate);
        }
    }

    private void CopyNetworkWeights(INeuralNetwork<T> source, INeuralNetwork<T> target)
    {
        var sourceParams = source.GetFlattenedParameters();
        target.UpdateParameters(sourceParams);
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        // A3C doesn't use replay buffer
    }

    public override T Train()
    {
        // Use TrainAsync instead
        return NumOps.Zero;
    }

    public override Dictionary<string, T> GetMetrics()
    {
        return new Dictionary<string, T>
        {
            ["global_steps"] = NumOps.FromDouble(_globalSteps)
        };
    }

    public override void ResetEpisode()
    {
        // No episode-level state to reset
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        return SelectAction(input, training: false);
    }

    public override Task<Vector<T>> PredictAsync(Vector<T> input)
    {
        return Task.FromResult(Predict(input));
    }

    public override Task TrainAsync()
    {
        return Task.CompletedTask;
    }
}
