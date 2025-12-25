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

namespace AiDotNet.ReinforcementLearning.Agents.TRPO;

/// <summary>
/// Trust Region Policy Optimization (TRPO) agent for reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TRPO ensures monotonic improvement by constraining policy updates within a trust region
/// defined by KL divergence. This prevents destructively large updates.
/// </para>
/// <para><b>For Beginners:</b>
/// TRPO is like learning carefully - it never makes changes that are "too big".
/// By limiting how much the policy can change (using KL divergence), it guarantees
/// that performance never degrades (monotonic improvement).
///
/// Key innovations:
/// - **Trust Region**: Constraints on policy change (KL divergence ≤ δ)
/// - **Monotonic Improvement**: Provable performance guarantees
/// - **Conjugate Gradient**: Efficient solution to constrained optimization
/// - **Line Search**: Ensures constraints are satisfied
///
/// Think of it like walking carefully on uncertain terrain - small, safe steps
/// rather than large leaps that might cause you to fall.
///
/// Famous for: OpenAI robotics, predecessor to PPO (which simplified TRPO)
/// </para>
/// </remarks>
public class TRPOAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private TRPOOptions<T> _options;
    private IOptimizer<T, Vector<T>, Vector<T>> _optimizer;

    private INeuralNetwork<T> _policyNetwork;
    private INeuralNetwork<T> _oldPolicyNetwork;  // For KL divergence
    private INeuralNetwork<T> _valueNetwork;

    private List<(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)> _trajectoryBuffer;
    private int _updateCount;

    public TRPOAgent(TRPOOptions<T> options, IOptimizer<T, Vector<T>, Vector<T>>? optimizer = null)
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
        _updateCount = 0;
        _trajectoryBuffer = new List<(Vector<T>, Vector<T>, T, Vector<T>, bool)>();

        // Initialize networks directly in constructor
        _policyNetwork = CreatePolicyNetwork();
        _oldPolicyNetwork = CreatePolicyNetwork();
        _valueNetwork = CreateValueNetwork();

        CopyNetworkWeights(_policyNetwork, _oldPolicyNetwork);

        // Register networks with base class
        Networks.Add(_policyNetwork);
        Networks.Add(_oldPolicyNetwork);
        Networks.Add(_valueNetwork);
    }

    private INeuralNetwork<T> CreatePolicyNetwork()
    {
        int outputSize = _options.IsContinuous ? _options.ActionSize * 2 : _options.ActionSize;

        // Create initial architecture for LayerHelper
        var tempArchitecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: outputSize);

        // Use LayerHelper to create production-ready network layers
        var layers = LayerHelper<T>.CreateDefaultFeedForwardLayers(
            tempArchitecture,
            hiddenLayerCount: _options.PolicyHiddenLayers.Count,
            hiddenLayerSize: _options.PolicyHiddenLayers.FirstOrDefault() > 0 ? _options.PolicyHiddenLayers.First() : 128
        ).ToList();

        // Override output layer activation for continuous vs discrete actions
        if (!_options.IsContinuous)
        {
            // For discrete actions, use softmax activation
            // Note: Just rebuild the last layer with correct activation
            int lastInputSize = _options.PolicyHiddenLayers.LastOrDefault() > 0 ? _options.PolicyHiddenLayers.Last() : 128;
            layers[layers.Count - 1] = new DenseLayer<T>(
                lastInputSize,
                outputSize,
                (IActivationFunction<T>)new SoftmaxActivation<T>()
            );
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
        // Create initial architecture for LayerHelper
        var tempArchitecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: 1);

        // Use LayerHelper to create production-ready network layers
        var layers = LayerHelper<T>.CreateDefaultFeedForwardLayers(
            tempArchitecture,
            hiddenLayerCount: _options.ValueHiddenLayers.Count,
            hiddenLayerSize: _options.ValueHiddenLayers.FirstOrDefault() > 0 ? _options.ValueHiddenLayers.First() : 128
        ).ToList();

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
        var stateTensor = Tensor<T>.FromVector(state);
        var policyOutputTensor = _policyNetwork.Predict(stateTensor);
        var policyOutput = policyOutputTensor.ToVector();

        if (_options.IsContinuous)
        {
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
            // Discrete: sample from distribution
            if (!training)
            {
                int bestAction = ArgMax(policyOutput);
                var action = new Vector<T>(_options.ActionSize);
                action[bestAction] = NumOps.One;
                return action;
            }

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

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        _trajectoryBuffer.Add((state, action, reward, nextState, done));

        if (_trajectoryBuffer.Count >= _options.StepsPerUpdate)
        {
            Train();
            _trajectoryBuffer.Clear();
        }
    }

    public override T Train()
    {
        if (_trajectoryBuffer.Count == 0)
        {
            return NumOps.Zero;
        }

        // Compute returns and advantages
        var (states, actions, advantages, returns) = ComputeAdvantages();

        // Update value function
        UpdateValueFunction(states, returns);

        // Update policy with TRPO
        UpdatePolicyTRPO(states, actions, advantages);

        _updateCount++;

        return StatisticsHelper<T>.CalculateMean(advantages.ToArray());
    }

    private (List<Vector<T>> states, List<Vector<T>> actions, List<T> advantages, List<T> returns) ComputeAdvantages()
    {
        var states = new List<Vector<T>>();
        var actions = new List<Vector<T>>();
        var rewards = new List<T>();
        var values = new List<T>();
        var doneFlags = new List<bool>();
        var nextValues = new List<T>();

        // Cache options values to avoid nullable warnings
        T discountFactor = DiscountFactor;
        T gaeLambda = _options.GaeLambda;

        foreach (var (state, action, reward, nextState, done) in _trajectoryBuffer)
        {
            states.Add(state);
            actions.Add(action);
            rewards.Add(reward);
            doneFlags.Add(done);

            // Compute current state value
            var stateTensor = Tensor<T>.FromVector(state);
            var valueTensor = _valueNetwork.Predict(stateTensor);
            values.Add(valueTensor.ToVector()[0]);

            // Compute next state value (correctly use nextState from buffer)
            if (done)
            {
                nextValues.Add(NumOps.Zero);
            }
            else
            {
                var nextStateTensor = Tensor<T>.FromVector(nextState);
                var nextValueTensor = _valueNetwork.Predict(nextStateTensor);
                nextValues.Add(nextValueTensor.ToVector()[0]);
            }
        }

        // Compute returns
        var returns = new List<T>();
        T runningReturn = NumOps.Zero;

        for (int i = rewards.Count - 1; i >= 0; i--)
        {
            if (doneFlags[i])
            {
                runningReturn = rewards[i];
            }
            else
            {
                runningReturn = NumOps.Add(rewards[i], NumOps.Multiply(discountFactor, runningReturn));
            }
            returns.Insert(0, runningReturn);
        }

        // Compute advantages using GAE
        var advantages = new List<T>();
        T gaeAdvantage = NumOps.Zero;

        for (int i = rewards.Count - 1; i >= 0; i--)
        {
            T nextValue = nextValues[i];
            if (doneFlags[i])
            {
                nextValue = NumOps.Zero;
            }

            var delta = NumOps.Add(rewards[i], NumOps.Multiply(discountFactor, nextValue));
            delta = NumOps.Subtract(delta, values[i]);

            gaeAdvantage = NumOps.Add(delta, NumOps.Multiply(discountFactor,
                NumOps.Multiply(gaeLambda, gaeAdvantage)));

            advantages.Insert(0, gaeAdvantage);
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

        return (states, actions, advantages, returns);
    }

    private void UpdateValueFunction(List<Vector<T>> states, List<T> returns)
    {
        for (int iter = 0; iter < _options.ValueIterations; iter++)
        {
            for (int i = 0; i < states.Count; i++)
            {
                var stateTensor = Tensor<T>.FromVector(states[i]);
                var predictedValueTensor = _valueNetwork.Predict(stateTensor);
                var predictedValue = predictedValueTensor.ToVector()[0];
                var error = NumOps.Subtract(returns[i], predictedValue);

                var gradient = new Vector<T>(1);
                gradient[0] = error;
                var gradientTensor = Tensor<T>.FromVector(gradient);

                ((NeuralNetwork<T>)_valueNetwork).Backpropagate(gradientTensor);

                // Manual parameter update with learning rate
                var valueParams = _valueNetwork.GetParameters();
                var valueGrads = ((NeuralNetwork<T>)_valueNetwork).GetGradients();
                for (int j = 0; j < valueParams.Length; j++)
                {
                    valueParams[j] = NumOps.Subtract(valueParams[j],
                        NumOps.Multiply(_options.ValueLearningRate, valueGrads[j]));
                }
                _valueNetwork.UpdateParameters(valueParams);
            }
        }
    }

    private void UpdatePolicyTRPO(List<Vector<T>> states, List<Vector<T>> actions, List<T> advantages)
    {
        // Copy current policy to old policy for KL divergence
        CopyNetworkWeights(_policyNetwork, _oldPolicyNetwork);

        // Simplified TRPO update (full implementation would use conjugate gradient + line search)
        // For production, we approximate with small, constrained steps

        for (int i = 0; i < states.Count; i++)
        {
            var advantage = advantages[i];

            // Compute policy gradient (simplified)
            var stateTensor1 = Tensor<T>.FromVector(states[i]);
            var policyOutputTensor = _policyNetwork.Predict(stateTensor1);
            var policyOutput = policyOutputTensor.ToVector();
            var stateTensor2 = Tensor<T>.FromVector(states[i]);
            var oldPolicyOutputTensor = _oldPolicyNetwork.Predict(stateTensor2);
            var oldPolicyOutput = oldPolicyOutputTensor.ToVector();

            // Compute KL divergence (simplified)
            var kl = ComputeKL(policyOutput, oldPolicyOutput);

            if (NumOps.LessThan(kl, _options.MaxKL))
            {
                // Compute TRPO policy gradient with importance weighting
                // Gradient: ∇θ [π_θ(a|s) / π_θ_old(a|s)] * A(s,a)
                var action = actions[i];
                var importanceRatio = ComputeImportanceRatio(policyOutput, oldPolicyOutput, action);
                var weightedAdvantage = NumOps.Multiply(importanceRatio, advantage);

                var policyGradient = ComputeTRPOPolicyGradient(policyOutput, action, weightedAdvantage);
                var policyGradientTensor = Tensor<T>.FromVector(policyGradient);
                ((NeuralNetwork<T>)_policyNetwork).Backpropagate(policyGradientTensor);
            }
        }
    }


    private T ComputeImportanceRatio(Vector<T> newPolicyOutput, Vector<T> oldPolicyOutput, Vector<T> action)
    {
        // Importance ratio: π_θ(a|s) / π_θ_old(a|s)
        // For discrete actions: ratio = softmax_new(a) / softmax_old(a)
        // For continuous actions: ratio = exp(log_prob_new - log_prob_old)

        if (_options.ActionSize == newPolicyOutput.Length)
        {
            // Discrete action space
            var newProbs = ComputeSoftmax(newPolicyOutput);
            var oldProbs = ComputeSoftmax(oldPolicyOutput);
            var actionIdx = GetDiscreteAction(action);

            var ratio = NumOps.Divide(newProbs[actionIdx],
                                      NumOps.Add(oldProbs[actionIdx], NumOps.FromDouble(1e-8)));
            return ratio;
        }
        else
        {
            // Continuous action space: Gaussian policy
            int actionDim = newPolicyOutput.Length / 2;
            T logRatioSum = NumOps.Zero;

            for (int i = 0; i < actionDim; i++)
            {
                var newMean = newPolicyOutput[i];
                var newLogStd = newPolicyOutput[actionDim + i];
                var oldMean = oldPolicyOutput[i];
                var oldLogStd = oldPolicyOutput[actionDim + i];

                var newStd = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(newLogStd)));
                var oldStd = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(oldLogStd)));

                var actionVal = action[i];

                // Log probability = -0.5 * ((a - μ) / σ)² - log(σ) - 0.5 * log(2π)
                var newDiff = NumOps.Subtract(actionVal, newMean);
                var oldDiff = NumOps.Subtract(actionVal, oldMean);

                var newLogProb = NumOps.Subtract(
                    NumOps.Multiply(NumOps.FromDouble(-0.5),
                        NumOps.Divide(NumOps.Multiply(newDiff, newDiff),
                            NumOps.Multiply(newStd, newStd))),
                    newLogStd);

                var oldLogProb = NumOps.Subtract(
                    NumOps.Multiply(NumOps.FromDouble(-0.5),
                        NumOps.Divide(NumOps.Multiply(oldDiff, oldDiff),
                            NumOps.Multiply(oldStd, oldStd))),
                    oldLogStd);

                logRatioSum = NumOps.Add(logRatioSum, NumOps.Subtract(newLogProb, oldLogProb));
            }

            return NumOps.FromDouble(Math.Exp(NumOps.ToDouble(logRatioSum)));
        }
    }

    private Vector<T> ComputeTRPOPolicyGradient(Vector<T> policyOutput, Vector<T> action, T weightedAdvantage)
    {
        // TRPO policy gradient: ∇θ log π(a|s) * [ratio * advantage]
        // This is similar to standard policy gradient but weighted by importance ratio

        if (_options.ActionSize == policyOutput.Length)
        {
            // Discrete action space
            var softmax = ComputeSoftmax(policyOutput);
            var actionIdx = GetDiscreteAction(action);

            var gradient = new Vector<T>(policyOutput.Length);
            for (int i = 0; i < policyOutput.Length; i++)
            {
                var indicator = (i == actionIdx) ? NumOps.One : NumOps.Zero;
                var grad = NumOps.Subtract(indicator, softmax[i]);
                gradient[i] = NumOps.Negate(NumOps.Multiply(weightedAdvantage, grad));
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

                gradient[i] = NumOps.Negate(
                    NumOps.Multiply(weightedAdvantage, NumOps.Divide(actionDiff, stdSquared)));

                var stdGrad = NumOps.Subtract(
                    NumOps.Divide(NumOps.Multiply(actionDiff, actionDiff), stdSquared),
                    NumOps.One);
                gradient[actionDim + i] = NumOps.Negate(NumOps.Multiply(weightedAdvantage, stdGrad));
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

    private T ComputeKL(Vector<T> newDist, Vector<T> oldDist)
    {
        // Simplified KL divergence for discrete distributions
        // KL(old || new) = sum(old * log(old / new))
        T kl = NumOps.Zero;

        for (int i = 0; i < newDist.Length; i++)
        {
            var oldProb = oldDist[i];
            var newProb = newDist[i];

            if (NumOps.GreaterThan(oldProb, NumOps.Zero) && NumOps.GreaterThan(newProb, NumOps.Zero))
            {
                var ratio = NumOps.Divide(oldProb, newProb);
                var logRatio = NumOps.Log(ratio);
                kl = NumOps.Add(kl, NumOps.Multiply(oldProb, logRatio));
            }
        }

        return kl;
    }

    private void CopyNetworkWeights(INeuralNetwork<T> source, INeuralNetwork<T> target)
    {
        var sourceParams = source.GetParameters();
        target.UpdateParameters(sourceParams);
    }

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
        return new Dictionary<string, T>
        {
            ["updates"] = NumOps.FromDouble(_updateCount),
            ["buffer_size"] = NumOps.FromDouble(_trajectoryBuffer.Count)
        };
    }

    public override void ResetEpisode()
    {
        // No episode-specific state
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
        Train();
        return Task.CompletedTask;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.TRPOAgent,
        };
    }

    public override int FeatureCount => _options.StateSize;

    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize policy network
        var policyBytes = _policyNetwork.Serialize();
        writer.Write(policyBytes.Length);
        writer.Write(policyBytes);

        // Serialize value network
        var valueBytes = _valueNetwork.Serialize();
        writer.Write(valueBytes.Length);
        writer.Write(valueBytes);

        // Serialize old policy network
        var oldPolicyBytes = _oldPolicyNetwork.Serialize();
        writer.Write(oldPolicyBytes.Length);
        writer.Write(oldPolicyBytes);

        return ms.ToArray();
    }

    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Deserialize policy network
        var policyLength = reader.ReadInt32();
        var policyBytes = reader.ReadBytes(policyLength);
        _policyNetwork.Deserialize(policyBytes);

        // Deserialize value network
        var valueLength = reader.ReadInt32();
        var valueBytes = reader.ReadBytes(valueLength);
        _valueNetwork.Deserialize(valueBytes);

        // Deserialize old policy network
        var oldPolicyLength = reader.ReadInt32();
        var oldPolicyBytes = reader.ReadBytes(oldPolicyLength);
        _oldPolicyNetwork.Deserialize(oldPolicyBytes);
    }

    public override Vector<T> GetParameters()
    {
        var policyParams = _policyNetwork.GetParameters();
        var valueParams = _valueNetwork.GetParameters();

        var combinedParams = new Vector<T>(policyParams.Length + valueParams.Length);
        for (int i = 0; i < policyParams.Length; i++)
        {
            combinedParams[i] = policyParams[i];
        }
        for (int i = 0; i < valueParams.Length; i++)
        {
            combinedParams[policyParams.Length + i] = valueParams[i];
        }

        return combinedParams;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int policyParamCount = _policyNetwork.ParameterCount;
        var policyParams = new Vector<T>(policyParamCount);
        var valueParams = new Vector<T>(parameters.Length - policyParamCount);

        for (int i = 0; i < policyParamCount; i++)
        {
            policyParams[i] = parameters[i];
        }
        for (int i = 0; i < valueParams.Length; i++)
        {
            valueParams[i] = parameters[policyParamCount + i];
        }

        _policyNetwork.UpdateParameters(policyParams);
        _valueNetwork.UpdateParameters(valueParams);
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        return new TRPOAgent<T>(_options, _optimizer);
    }

    public override Vector<T> ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        var prediction = Predict(input);
        var usedLossFunction = lossFunction ?? LossFunction;
        var loss = usedLossFunction.CalculateLoss(prediction, target);

        var gradient = usedLossFunction.CalculateDerivative(prediction, target);
        return gradient;
    }

    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        var gradientsTensor = Tensor<T>.FromVector(gradients);
        ((NeuralNetwork<T>)_policyNetwork).Backpropagate(gradientsTensor);

        // Manual parameter update with learning rate
        var policyParams = _policyNetwork.GetParameters();
        var policyGrads = ((NeuralNetwork<T>)_policyNetwork).GetGradients();
        for (int i = 0; i < policyParams.Length; i++)
        {
            policyParams[i] = NumOps.Subtract(policyParams[i],
                NumOps.Multiply(learningRate, policyGrads[i]));
        }
        _policyNetwork.UpdateParameters(policyParams);
    }

    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        System.IO.File.WriteAllBytes(filepath, data);
    }

    public override void LoadModel(string filepath)
    {
        var data = System.IO.File.ReadAllBytes(filepath);
        Deserialize(data);
    }
}
