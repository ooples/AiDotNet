using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ReinforcementLearning.Common;

namespace AiDotNet.ReinforcementLearning.Agents.PPO;

/// <summary>
/// Proximal Policy Optimization (PPO) agent for reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PPO is a policy gradient method that uses a clipped surrogate objective to enable
/// multiple epochs of minibatch updates without destructively large policy changes.
/// It achieves state-of-the-art performance across many RL benchmarks while being
/// simpler and more robust than methods like TRPO.
/// </para>
/// <para><b>For Beginners:</b>
/// PPO is one of the most popular RL algorithms today. It's used by:
/// - OpenAI's ChatGPT (for RLHF training)
/// - Many robotics systems
/// - Game AI (including Dota 2 bots)
///
/// Key idea: Make small, safe policy improvements by clipping updates.
/// Think of it like driving - small steering adjustments work better than jerking the wheel.
///
/// PPO learns two things:
/// - A policy (actor): What action to take in each state
/// - A value function (critic): How good each state is
///
/// The critic helps the actor learn more efficiently.
/// </para>
/// <para><b>Reference:</b>
/// Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347.
/// </para>
/// </remarks>
public class PPOAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private PPOOptions<T> _ppoOptions;
    private readonly Trajectory<T> _trajectory;

    private NeuralNetwork<T> _policyNetwork;
    private NeuralNetwork<T> _valueNetwork;

    /// <inheritdoc/>
    public override int FeatureCount => _ppoOptions.StateSize;

    /// <summary>
    /// Initializes a new instance of the PPOAgent class.
    /// </summary>
    /// <param name="options">Configuration options for the PPO agent.</param>
    public PPOAgent(PPOOptions<T> options)
        : base(new ReinforcementLearningOptions<T>
        {
            LearningRate = options.PolicyLearningRate,
            DiscountFactor = options.DiscountFactor,
            LossFunction = new MeanSquaredErrorLoss<T>(),  // For policy, though we compute custom loss
            Seed = options.Seed,
            BatchSize = options.MiniBatchSize
        })
    {
        _ppoOptions = options ?? throw new ArgumentNullException(nameof(options));
        _trajectory = new Trajectory<T>();

        // Build policy network
        _policyNetwork = BuildPolicyNetwork();

        // Build value network
        _valueNetwork = BuildValueNetwork();

        // Register networks with base class
        Networks.Add(_policyNetwork);
        Networks.Add(_valueNetwork);
    }

    private NeuralNetwork<T> BuildPolicyNetwork()
    {
        var layers = new List<ILayer<T>>();
        int prevSize = _ppoOptions.StateSize;

        // Hidden layers
        foreach (var hiddenSize in _ppoOptions.PolicyHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(prevSize, hiddenSize, (IActivationFunction<T>)new TanhActivation<T>()));
            prevSize = hiddenSize;
        }

        // Output layer
        if (_ppoOptions.IsContinuous)
        {
            // For continuous: output mean and log_std for Gaussian policy
            layers.Add(new DenseLayer<T>(prevSize, _ppoOptions.ActionSize * 2, (IActivationFunction<T>)new IdentityActivation<T>()));
        }
        else
        {
            // For discrete: output action logits (softmax applied later)
            layers.Add(new DenseLayer<T>(prevSize, _ppoOptions.ActionSize, (IActivationFunction<T>)new IdentityActivation<T>()));
        }

        int finalOutputSize = _ppoOptions.IsContinuous ? _ppoOptions.ActionSize * 2 : _ppoOptions.ActionSize;
        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _ppoOptions.StateSize,
            outputSize: finalOutputSize,
            layers: layers);

        return new NeuralNetwork<T>(architecture);
    }

    private NeuralNetwork<T> BuildValueNetwork()
    {
        var layers = new List<ILayer<T>>();
        int prevSize = _ppoOptions.StateSize;

        // Hidden layers
        foreach (var hiddenSize in _ppoOptions.ValueHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(prevSize, hiddenSize, (IActivationFunction<T>)new TanhActivation<T>()));
            prevSize = hiddenSize;
        }

        // Output layer (single value)
        layers.Add(new DenseLayer<T>(prevSize, 1, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _ppoOptions.StateSize,
            outputSize: 1,
            layers: layers);

        return new NeuralNetwork<T>(architecture, _ppoOptions.ValueLossFunction);
    }

    /// <inheritdoc/>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var policyOutputTensor = _policyNetwork.Predict(stateTensor);
        var policyOutput = policyOutputTensor.ToVector();

        if (_ppoOptions.IsContinuous)
        {
            // Continuous action space: sample from Gaussian
            return SampleContinuousAction(policyOutput, training);
        }
        else
        {
            // Discrete action space: sample from categorical
            return SampleDiscreteAction(policyOutput, training);
        }
    }

    private Vector<T> SampleDiscreteAction(Vector<T> logits, bool training)
    {
        // Apply softmax to get probabilities
        var probs = Softmax(logits);

        int actionIndex;
        if (training)
        {
            // Sample from categorical distribution
            actionIndex = SampleCategorical(probs);
        }
        else
        {
            // Greedy: pick highest probability
            actionIndex = ArgMax(probs);
        }

        // Return one-hot encoded action
        var action = new Vector<T>(_ppoOptions.ActionSize);
        action[actionIndex] = NumOps.One;
        return action;
    }

    private Vector<T> SampleContinuousAction(Vector<T> output, bool training)
    {
        // First half is mean, second half is log_std
        var action = new Vector<T>(_ppoOptions.ActionSize);

        for (int i = 0; i < _ppoOptions.ActionSize; i++)
        {
            var mean = output[i];
            var logStd = output[_ppoOptions.ActionSize + i];
            var std = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(logStd)));

            if (training)
            {
                // Sample from Gaussian using MathHelper
                var noise = MathHelper.GetNormalRandom<T>(NumOps.Zero, NumOps.One);
                action[i] = NumOps.Add(mean, NumOps.Multiply(std, noise));
            }
            else
            {
                // Deterministic: use mean
                action[i] = mean;
            }
        }

        return action;
    }

    /// <inheritdoc/>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        // Get value estimate for current state
        var stateTensor = Tensor<T>.FromVector(state);
        var valueOutputTensor = _valueNetwork.Predict(stateTensor);
        var valueOutput = valueOutputTensor.ToVector();
        var value = valueOutput[0];

        // Get log probability of action
        var logProb = ComputeLogProb(state, action);

        _trajectory.AddStep(state, action, reward, value, logProb, done);
    }

    private T ComputeLogProb(Vector<T> state, Vector<T> action)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var policyOutputTensor = _policyNetwork.Predict(stateTensor);
        var policyOutput = policyOutputTensor.ToVector();

        if (_ppoOptions.IsContinuous)
        {
            return ComputeContinuousLogProb(policyOutput, action);
        }
        else
        {
            return ComputeDiscreteLogProb(policyOutput, action);
        }
    }

    private T ComputeDiscreteLogProb(Vector<T> logits, Vector<T> action)
    {
        var probs = Softmax(logits);
        int actionIndex = ArgMax(action);

        // Log probability of selected action
        var prob = probs[actionIndex];
        return NumOps.FromDouble(Math.Log(NumOps.ToDouble(prob) + 1e-10));
    }

    private T ComputeContinuousLogProb(Vector<T> output, Vector<T> action)
    {
        T totalLogProb = NumOps.Zero;

        for (int i = 0; i < _ppoOptions.ActionSize; i++)
        {
            var mean = output[i];
            var logStd = output[_ppoOptions.ActionSize + i];
            var std = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(logStd)));

            // Gaussian log probability
            var diff = NumOps.Subtract(action[i], mean);
            var variance = NumOps.Multiply(std, std);

            var logProb = NumOps.FromDouble(
                -0.5 * Math.Log(2 * Math.PI) -
                NumOps.ToDouble(logStd) -
                0.5 * NumOps.ToDouble(NumOps.Divide(NumOps.Multiply(diff, diff), variance))
            );

            totalLogProb = NumOps.Add(totalLogProb, logProb);
        }

        return totalLogProb;
    }

    /// <inheritdoc/>
    public override T Train()
    {
        // PPO trains when trajectory is full
        if (_trajectory.Length < _ppoOptions.StepsPerUpdate)
        {
            return NumOps.Zero;
        }

        TrainingSteps++;

        // Compute advantages and returns using GAE
        ComputeAdvantages();

        // Train for multiple epochs on collected data
        T totalLoss = NumOps.Zero;
        int numUpdates = 0;

        for (int epoch = 0; epoch < _ppoOptions.TrainingEpochs; epoch++)
        {
            // Shuffle indices for minibatch sampling
            var indices = Enumerable.Range(0, _trajectory.Length).OrderBy(_ => Random.Next()).ToList();

            for (int start = 0; start < _trajectory.Length; start += _ppoOptions.MiniBatchSize)
            {
                int end = Math.Min(start + _ppoOptions.MiniBatchSize, _trajectory.Length);
                var batchIndices = indices.Skip(start).Take(end - start).ToList();

                var loss = UpdateNetworks(batchIndices);
                totalLoss = NumOps.Add(totalLoss, loss);
                numUpdates++;
            }
        }

        var avgLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(numUpdates));
        LossHistory.Add(avgLoss);

        // Clear trajectory for next collection
        _trajectory.Clear();

        return avgLoss;
    }

    private void ComputeAdvantages()
    {
        // Compute advantages using GAE (Generalized Advantage Estimation)
        var advantages = new List<T>();
        var returns = new List<T>();

        T lastGae = NumOps.Zero;

        for (int t = _trajectory.Length - 1; t >= 0; t--)
        {
            T nextValue;
            if (t == _trajectory.Length - 1)
            {
                nextValue = _trajectory.Dones[t] ? NumOps.Zero : _trajectory.Values[t];
            }
            else
            {
                nextValue = _trajectory.Values[t + 1];
            }

            // TD error: delta = r + gamma * V(s') - V(s)
            var delta = NumOps.Add(
                _trajectory.Rewards[t],
                NumOps.Subtract(
                    NumOps.Multiply(DiscountFactor, nextValue),
                    _trajectory.Values[t]
                )
            );

            // GAE: A = delta + gamma * lambda * A_next
            lastGae = NumOps.Add(
                delta,
                NumOps.Multiply(
                    NumOps.Multiply(DiscountFactor, _ppoOptions.GaeLambda),
                    _trajectory.Dones[t] ? NumOps.Zero : lastGae
                )
            );

            advantages.Insert(0, lastGae);
            returns.Insert(0, NumOps.Add(lastGae, _trajectory.Values[t]));
        }

        // Normalize advantages using StatisticsHelper
        var stdAdv = StatisticsHelper<T>.CalculateStandardDeviation(advantages);
        T meanAdv = NumOps.Zero;
        foreach (var adv in advantages)
            meanAdv = NumOps.Add(meanAdv, adv);
        meanAdv = NumOps.Divide(meanAdv, NumOps.FromDouble(advantages.Count));

        for (int i = 0; i < advantages.Count; i++)
        {
            advantages[i] = NumOps.Divide(
                NumOps.Subtract(advantages[i], meanAdv),
                NumOps.Add(stdAdv, NumOps.FromDouble(1e-8))
            );
        }

        _trajectory.Advantages = advantages;
        _trajectory.Returns = returns;
    }

    private T UpdateNetworks(List<int> batchIndices)
    {
        T policyLoss = NumOps.Zero;
        T valueLoss = NumOps.Zero;
        T entropyLoss = NumOps.Zero;

        foreach (var idx in batchIndices)
        {
            var state = _trajectory.States[idx];
            var action = _trajectory.Actions[idx];
            var oldLogProb = _trajectory.LogProbs[idx];
            var advantage = _trajectory.Advantages![idx];
            var targetReturn = _trajectory.Returns![idx];

            // Policy loss (clipped objective)
            var newLogProb = ComputeLogProb(state, action);
            var ratio = NumOps.FromDouble(Math.Exp(
                NumOps.ToDouble(NumOps.Subtract(newLogProb, oldLogProb))
            ));

            var surr1 = NumOps.Multiply(ratio, advantage);
            var clippedRatio = MathHelper.Clamp<T>(ratio,
                NumOps.Subtract(NumOps.One, _ppoOptions.ClipEpsilon),
                NumOps.Add(NumOps.One, _ppoOptions.ClipEpsilon));
            var surr2 = NumOps.Multiply(clippedRatio, advantage);

            var minSurr = MathHelper.Min<T>(surr1, surr2);
            policyLoss = NumOps.Subtract(policyLoss, minSurr);  // Negative for gradient ascent

            // Value loss
            var stateTensor = Tensor<T>.FromVector(state);
            var valueOutputTensor = _valueNetwork.Predict(stateTensor);
            var valueOutput = valueOutputTensor.ToVector();
            var predictedValue = valueOutput[0];
            var valueDiff = NumOps.Subtract(predictedValue, targetReturn);
            valueLoss = NumOps.Add(valueLoss, NumOps.Multiply(valueDiff, valueDiff));

            // Entropy (for exploration)
            var entropy = ComputeEntropy(state);
            entropyLoss = NumOps.Subtract(entropyLoss, entropy);  // Negative to encourage entropy
        }

        // Average losses
        var batchSize = NumOps.FromDouble(batchIndices.Count);
        policyLoss = NumOps.Divide(policyLoss, batchSize);
        valueLoss = NumOps.Divide(valueLoss, batchSize);
        entropyLoss = NumOps.Divide(entropyLoss, batchSize);

        // Combined loss
        var totalLoss = NumOps.Add(policyLoss,
            NumOps.Add(
                NumOps.Multiply(_ppoOptions.ValueLossCoefficient, valueLoss),
                NumOps.Multiply(_ppoOptions.EntropyCoefficient, entropyLoss)
            )
        );

        // Update networks (simplified - in practice would use proper optimizers)
        UpdatePolicyNetwork(batchIndices);
        UpdateValueNetwork(batchIndices);

        return totalLoss;
    }

    private T ComputeEntropy(Vector<T> state)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var policyOutputTensor = _policyNetwork.Predict(stateTensor);
        var policyOutput = policyOutputTensor.ToVector();

        if (_ppoOptions.IsContinuous)
        {
            // Gaussian entropy: 0.5 * log(2 * pi * e * sigma^2)
            T entropy = NumOps.Zero;
            for (int i = 0; i < _ppoOptions.ActionSize; i++)
            {
                var logStd = policyOutput[_ppoOptions.ActionSize + i];
                entropy = NumOps.Add(entropy,
                    NumOps.Add(
                        NumOps.FromDouble(0.5 * Math.Log(2 * Math.PI * Math.E)),
                        logStd
                    )
                );
            }
            return entropy;
        }
        else
        {
            // Categorical entropy: -sum(p * log(p))
            var probs = Softmax(policyOutput);
            T entropy = NumOps.Zero;

            for (int i = 0; i < probs.Length; i++)
            {
                var p = NumOps.ToDouble(probs[i]);
                if (p > 1e-10)
                {
                    entropy = NumOps.Subtract(entropy,
                        NumOps.FromDouble(p * Math.Log(p))
                    );
                }
            }
            return entropy;
        }
    }

    private void UpdatePolicyNetwork(List<int> batchIndices)
    {
        // PPO clipped objective update
        var params_ = _policyNetwork.GetParameters();

        // Compute policy gradients using PPO clipped objective
        foreach (var idx in batchIndices)
        {
            var state = _trajectory.States[idx];
            var action = _trajectory.Actions[idx];
            var advantage = _trajectory.Advantages![idx];
            var oldLogProb = _trajectory.LogProbs![idx];

            // Forward pass to get current policy probabilities
            var stateTensor = Tensor<T>.FromVector(state);
            var currentProbs = _policyNetwork.Predict(stateTensor).ToVector();

            // Compute log probability of selected action under current policy
            // For discrete actions: log(prob[action])
            int selectedAction = 0;
            for (int i = 0; i < action.Length; i++)
            {
                if (NumOps.GreaterThan(action[i], NumOps.Zero))
                {
                    selectedAction = i;
                    break;
                }
            }

            // Clamp probability to avoid log(0)
            var currentProb = currentProbs[selectedAction];
            var clampedProb = MathHelper.Clamp(currentProb, NumOps.FromDouble(1e-8), NumOps.One);
            var currentLogProb = NumOps.FromDouble(Math.Log(NumOps.ToDouble(clampedProb)));

            // Compute importance sampling ratio: π_θ(a|s) / π_θ_old(a|s)
            // ratio = exp(log(π_θ) - log(π_θ_old))
            var logRatio = NumOps.Subtract(currentLogProb, oldLogProb);
            var ratio = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(logRatio)));

            // PPO clipped objective:
            // L^CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
            var epsilonValue = NumOps.ToDouble(_ppoOptions.ClipEpsilon);
            var clippedRatio = MathHelper.Clamp(ratio,
                NumOps.FromDouble(1.0 - epsilonValue),
                NumOps.FromDouble(1.0 + epsilonValue));

            var obj1 = NumOps.Multiply(ratio, advantage);
            var obj2 = NumOps.Multiply(clippedRatio, advantage);

            // Take minimum of clipped and unclipped objectives
            var policyLoss = NumOps.LessThan(obj1, obj2) ? obj1 : obj2;

            // Gradient is negative of loss (for gradient ascent)
            var gradOutput = new Vector<T>(currentProbs.Length);
            for (int i = 0; i < gradOutput.Length; i++)
            {
                // Only apply gradient to selected action
                if (i == selectedAction)
                {
                    gradOutput[i] = NumOps.Negate(policyLoss);
                }
                else
                {
                    gradOutput[i] = NumOps.Zero;
                }
            }

            var gradTensor = Tensor<T>.FromVector(gradOutput);
            _policyNetwork.Backpropagate(gradTensor);
        }

        // Apply gradients
        var grads = _policyNetwork.GetParameters();
        for (int i = 0; i < params_.Length; i++)
        {
            var update = NumOps.Multiply(_ppoOptions.PolicyLearningRate, grads[i]);
            params_[i] = NumOps.Add(params_[i], update);
        }

        _policyNetwork.UpdateParameters(params_);
    }

    private void UpdateValueNetwork(List<int> batchIndices)
    {
        // Simplified gradient update
        var params_ = _valueNetwork.GetParameters();

        foreach (var idx in batchIndices)
        {
            var state = _trajectory.States[idx];
            var targetReturn = _trajectory.Returns![idx];

            var stateTensor = Tensor<T>.FromVector(state);
            var valueOutputTensor = _valueNetwork.Predict(stateTensor);
            var valueOutput = valueOutputTensor.ToVector();
            var predicted = valueOutput[0];

            var target = new Vector<T>(1);
            target[0] = targetReturn;

            // Convert to vectors for loss function
            var gradientVector = _ppoOptions.ValueLossFunction.CalculateDerivative(valueOutput, target);
            var gradTensor = Tensor<T>.FromVector(gradientVector);
            _valueNetwork.Backpropagate(gradTensor);
        }

        var grads = _valueNetwork.GetParameters();
        for (int i = 0; i < params_.Length; i++)
        {
            var update = NumOps.Multiply(_ppoOptions.ValueLearningRate, grads[i]);
            params_[i] = NumOps.Subtract(params_[i], update);
        }

        _valueNetwork.UpdateParameters(params_);
    }

    /// <inheritdoc/>
    public override Dictionary<string, T> GetMetrics()
    {
        var baseMetrics = base.GetMetrics();
        baseMetrics["TrajectoryLength"] = NumOps.FromDouble(_trajectory.Length);
        return baseMetrics;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.PPOAgent,
            FeatureCount = _ppoOptions.StateSize,
        };
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        writer.Write(_ppoOptions.StateSize);
        writer.Write(_ppoOptions.ActionSize);
        writer.Write(_ppoOptions.IsContinuous);

        var policyBytes = _policyNetwork.Serialize();
        writer.Write(policyBytes.Length);
        writer.Write(policyBytes);

        var valueBytes = _valueNetwork.Serialize();
        writer.Write(valueBytes.Length);
        writer.Write(valueBytes);

        return ms.ToArray();
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        var stateSize = reader.ReadInt32();
        var actionSize = reader.ReadInt32();
        var isContinuous = reader.ReadBoolean();

        var policyLength = reader.ReadInt32();
        var policyBytes = reader.ReadBytes(policyLength);
        _policyNetwork.Deserialize(policyBytes);

        var valueLength = reader.ReadInt32();
        var valueBytes = reader.ReadBytes(valueLength);
        _valueNetwork.Deserialize(valueBytes);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var policyParams = _policyNetwork.GetParameters();
        var valueParams = _valueNetwork.GetParameters();

        var totalParams = policyParams.Length + valueParams.Length;
        var vector = new Vector<T>(totalParams);

        int idx = 0;
        for (int i = 0; i < policyParams.Length; i++)
            vector[idx++] = policyParams[i];
        for (int i = 0; i < valueParams.Length; i++)
            vector[idx++] = valueParams[i];

        return vector;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        var policyParams = _policyNetwork.GetParameters();
        var valueParams = _valueNetwork.GetParameters();

        var policyVector = new Vector<T>(policyParams.Length);
        var valueVector = new Vector<T>(valueParams.Length);

        int idx = 0;
        for (int i = 0; i < policyParams.Length; i++)
            policyVector[i] = parameters[idx++];
        for (int i = 0; i < valueParams.Length; i++)
            valueVector[i] = parameters[idx++];

        _policyNetwork.UpdateParameters(policyVector);
        _valueNetwork.UpdateParameters(valueVector);
    }

    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new PPOAgent<T>(_ppoOptions);
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc/>
    public override Vector<T> ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        // Not directly applicable for PPO (uses custom loss)
        return GetParameters();
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // Not directly applicable for PPO
    }

    // Helper methods
    private Vector<T> Softmax(Vector<T> logits)
    {
        var maxLogit = Max(logits);
        var exps = new Vector<T>(logits.Length);
        T sumExp = NumOps.Zero;

        for (int i = 0; i < logits.Length; i++)
        {
            var exp = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(NumOps.Subtract(logits[i], maxLogit))));
            exps[i] = exp;
            sumExp = NumOps.Add(sumExp, exp);
        }

        for (int i = 0; i < exps.Length; i++)
        {
            exps[i] = NumOps.Divide(exps[i], sumExp);
        }

        return exps;
    }

    private int SampleCategorical(Vector<T> probs)
    {
        double rand = Random.NextDouble();
        double cumProb = 0;

        for (int i = 0; i < probs.Length; i++)
        {
            cumProb += NumOps.ToDouble(probs[i]);
            if (rand < cumProb)
                return i;
        }

        return probs.Length - 1;
    }

    private int ArgMax(Vector<T> vector)
    {
        int maxIndex = 0;
        for (int i = 1; i < vector.Length; i++)
        {
            if (NumOps.ToDouble(vector[i]) > NumOps.ToDouble(vector[maxIndex]))
                maxIndex = i;
        }
        return maxIndex;
    }

    private T Max(Vector<T> vector)
    {
        T max = vector[0];
        for (int i = 1; i < vector.Length; i++)
        {
            if (NumOps.ToDouble(vector[i]) > NumOps.ToDouble(max))
                max = vector[i];
        }
        return max;
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
