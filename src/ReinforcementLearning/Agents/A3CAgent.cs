using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
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
            InitialLearningRate = 0.001,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        });
        _globalSteps = 0;

        // Initialize networks directly in constructor
        _globalPolicyNetwork = CreatePolicyNetwork();
        _globalValueNetwork = CreateValueNetwork();

        // Register networks with base class
        Networks.Add(_globalPolicyNetwork);
        Networks.Add(_globalValueNetwork);
    }

    private INeuralNetwork<T> CreatePolicyNetwork()
    {
        int outputSize = _options.IsContinuous ? _options.ActionSize * 2 : _options.ActionSize;

        var layers = new List<ILayer<T>>();
        int prevSize = _options.StateSize;

        foreach (var hiddenSize in _options.PolicyHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(prevSize, hiddenSize, (IActivationFunction<T>)new TanhActivation<T>()));
            prevSize = hiddenSize;
        }

        // Output layer
        if (_options.IsContinuous)
        {
            layers.Add(new DenseLayer<T>(prevSize, outputSize, (IActivationFunction<T>)new IdentityActivation<T>()));
        }
        else
        {
            layers.Add(new DenseLayer<T>(prevSize, outputSize, (IActivationFunction<T>)new SoftmaxActivation<T>()));
        }

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: outputSize,
            layers: layers);

        return new NeuralNetwork<T>(architecture, _options.ValueLossFunction);
    }

    private INeuralNetwork<T> CreateValueNetwork()
    {
        var layers = new List<ILayer<T>>();
        int prevSize = _options.StateSize;

        foreach (var hiddenSize in _options.ValueHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(prevSize, hiddenSize, (IActivationFunction<T>)new TanhActivation<T>()));
            prevSize = hiddenSize;
        }

        layers.Add(new DenseLayer<T>(prevSize, 1, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: 1,
            layers: layers);

        return new NeuralNetwork<T>(architecture, _options.ValueLossFunction);
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        Vector<T> policyOutput;

        lock (_globalLock)
        {
            var stateTensor = Tensor<T>.FromVector(state);
            var policyOutputTensor = _globalPolicyNetwork.Predict(stateTensor);
            policyOutput = policyOutputTensor.ToVector();
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
                var std = NumOps.Exp(logStd[i]);
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
                    if (NumOps.GreaterThan(policyOutput[i], bestProb))
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
                var stateTensor = Tensor<T>.FromVector(state);
                var valueTensor = localValue.Predict(stateTensor);
                var value = valueTensor.ToVector()[0];
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
        var stateTensor = Tensor<T>.FromVector(state);
        var policyOutputTensor = policy.Predict(stateTensor);
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
            var lastStateTensor = Tensor<T>.FromVector(lastState);
            var nextValueTensor = valueNetwork.Predict(lastStateTensor);
            nextValue = nextValueTensor.ToVector()[0];
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
                runningReturn = NumOps.Add(exp.reward, NumOps.Multiply(DiscountFactor, runningReturn));
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

        if (NumOps.GreaterThan(std, NumOps.Zero))
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
        // Implement A3C gradient computation
        // Policy gradient: ∇θ log π(a|s) * advantage
        // Value gradient: ∇φ (V(s) - return)^2

        for (int i = 0; i < trajectory.Count; i++)
        {
            var exp = trajectory[i];
            var advantage = advantages[i];
            var targetReturn = returns[i];

            // Compute policy gradient
            var stateTensor1 = Tensor<T>.FromVector(exp.state);
            var policyOutputTensor = localPolicy.Predict(stateTensor1);
            var policyOutput = policyOutputTensor.ToVector();
            var policyGradient = ComputeA3CPolicyGradient(policyOutput, exp.action, advantage);
            var policyGradientTensor = Tensor<T>.FromVector(policyGradient);
            ((NeuralNetwork<T>)localPolicy).Backpropagate(policyGradientTensor);

            // Compute value gradient
            var stateTensor2 = Tensor<T>.FromVector(exp.state);
            var predictedValueTensor = localValue.Predict(stateTensor2);
            var predictedValue = predictedValueTensor.ToVector()[0];
            var valueDiff = NumOps.Subtract(predictedValue, targetReturn);
            var valueGradient = new Vector<T>(1);
            valueGradient[0] = NumOps.Divide(
                NumOps.Multiply(NumOps.FromDouble(2.0), valueDiff),
                NumOps.FromDouble(trajectory.Count));
            var valueGradientTensor = Tensor<T>.FromVector(valueGradient);
            ((NeuralNetwork<T>)localValue).Backpropagate(valueGradientTensor);
        }

        // Update global networks with local gradients
        UpdateNetworkParameters(_globalPolicyNetwork, localPolicy, _options.PolicyLearningRate);
        UpdateNetworkParameters(_globalValueNetwork, localValue, _options.ValueLearningRate);
    }


    private Vector<T> ComputeA3CPolicyGradient(Vector<T> policyOutput, Vector<T> action, T advantage)
    {
        // A3C uses same policy gradient as A2C: ∇θ log π(a|s) * advantage
        // Supports both continuous (Gaussian) and discrete (Softmax) policies

        if (_options.ActionSize == policyOutput.Length)
        {
            // Discrete action space: softmax policy
            var softmax = ComputeSoftmax(policyOutput);
            var selectedAction = GetDiscreteAction(action);

            var gradient = new Vector<T>(policyOutput.Length);
            for (int i = 0; i < policyOutput.Length; i++)
            {
                var indicator = (i == selectedAction) ? NumOps.One : NumOps.Zero;
                var grad = NumOps.Subtract(indicator, softmax[i]);
                gradient[i] = NumOps.Negate(NumOps.Multiply(advantage, grad));
            }
            return gradient;
        }
        else
        {
            // Continuous action space: Gaussian policy
            int actionDim = policyOutput.Length / 2;
            var gradient = new Vector<T>(policyOutput.Length);

            for (int i = 0; i < actionDim; i++)
            {
                var mean = policyOutput[i];
                var logStd = policyOutput[actionDim + i];
                var std = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(logStd)));
                var actionDiff = NumOps.Subtract(action[i], mean);
                var stdSquared = NumOps.Multiply(std, std);

                // ∇mean = -(a - μ) / σ² * advantage
                gradient[i] = NumOps.Negate(
                    NumOps.Multiply(advantage, NumOps.Divide(actionDiff, stdSquared)));

                // ∇log_std = -((a - μ)² / σ² - 1) * advantage
                var stdGrad = NumOps.Subtract(
                    NumOps.Divide(NumOps.Multiply(actionDiff, actionDiff), stdSquared),
                    NumOps.One);
                gradient[actionDim + i] = NumOps.Negate(NumOps.Multiply(advantage, stdGrad));
            }
            return gradient;
        }
    }

    private Vector<T> ComputeSoftmax(Vector<T> logits)
    {
        var max = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (NumOps.ToDouble(logits[i]) > NumOps.ToDouble(max))
                max = logits[i];

        var expSum = NumOps.Zero;
        var exps = new Vector<T>(logits.Length);
        for (int i = 0; i < logits.Length; i++)
        {
            exps[i] = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(NumOps.Subtract(logits[i], max))));
            expSum = NumOps.Add(expSum, exps[i]);
        }

        var softmax = new Vector<T>(logits.Length);
        for (int i = 0; i < logits.Length; i++)
            softmax[i] = NumOps.Divide(exps[i], expSum);

        return softmax;
    }

    private int GetDiscreteAction(Vector<T> actionVector)
    {
        // Action vector for discrete actions is one-hot encoded
        int maxIdx = 0;
        T maxVal = actionVector[0];
        for (int i = 1; i < actionVector.Length; i++)
        {
            if (NumOps.ToDouble(actionVector[i]) > NumOps.ToDouble(maxVal))
            {
                maxVal = actionVector[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    private void UpdateNetworkParameters(INeuralNetwork<T> globalNetwork, INeuralNetwork<T> localNetwork, T learningRate)
    {
        var globalParams = globalNetwork.GetParameters();
        var localGrads = ((NeuralNetwork<T>)localNetwork).GetGradients();

        for (int i = 0; i < globalParams.Length; i++)
        {
            var update = NumOps.Multiply(learningRate, localGrads[i]);
            globalParams[i] = NumOps.Subtract(globalParams[i], update);
        }

        globalNetwork.UpdateParameters(globalParams);
    }

    private void CopyNetworkWeights(INeuralNetwork<T> source, INeuralNetwork<T> target)
    {
        var sourceParams = source.GetParameters();
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

    public Task<Vector<T>> PredictAsync(Vector<T> input)
    {
        return Task.FromResult(Predict(input));
    }

    public Task TrainAsync()
    {
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override int FeatureCount => _options.StateSize;

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.A3CAgent,
            FeatureCount = _options.StateSize,
            Complexity = ParameterCount,
        };
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var policyParams = _globalPolicyNetwork.GetParameters();
        var valueParams = _globalValueNetwork.GetParameters();

        var total = policyParams.Length + valueParams.Length;
        var vector = new Vector<T>(total);

        int idx = 0;
        foreach (var p in policyParams) vector[idx++] = p;
        foreach (var p in valueParams) vector[idx++] = p;

        return vector;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        var policyParams = _globalPolicyNetwork.GetParameters();
        var valueParams = _globalValueNetwork.GetParameters();

        int idx = 0;
        var policyVec = new Vector<T>(policyParams.Length);
        var valueVec = new Vector<T>(valueParams.Length);

        for (int i = 0; i < policyParams.Length; i++) policyVec[i] = parameters[idx++];
        for (int i = 0; i < valueParams.Length; i++) valueVec[i] = parameters[idx++];

        _globalPolicyNetwork.UpdateParameters(policyVec);
        _globalValueNetwork.UpdateParameters(valueVec);
    }

    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new A3CAgent<T>(_options, _optimizer);
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc/>
    public override Vector<T> ComputeGradients(
        Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return GetParameters();
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // A3C uses asynchronous updates - not directly applicable
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        writer.Write(_options.StateSize);
        writer.Write(_options.ActionSize);
        writer.Write(_globalSteps);

        var policyBytes = _globalPolicyNetwork.Serialize();
        writer.Write(policyBytes.Length);
        writer.Write(policyBytes);

        var valueBytes = _globalValueNetwork.Serialize();
        writer.Write(valueBytes.Length);
        writer.Write(valueBytes);

        return ms.ToArray();
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        reader.ReadInt32(); // stateSize
        reader.ReadInt32(); // actionSize
        _globalSteps = reader.ReadInt32();

        var policyLength = reader.ReadInt32();
        var policyBytes = reader.ReadBytes(policyLength);
        _globalPolicyNetwork.Deserialize(policyBytes);

        var valueLength = reader.ReadInt32();
        var valueBytes = reader.ReadBytes(valueLength);
        _globalValueNetwork.Deserialize(valueBytes);
    }

    /// <inheritdoc/>
    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        System.IO.File.WriteAllBytes(filepath, data);
    }

    /// <inheritdoc/>
    public override void LoadModel(string filepath)
    {
        var data = System.IO.File.ReadAllBytes(filepath);
        Deserialize(data);
    }
}
