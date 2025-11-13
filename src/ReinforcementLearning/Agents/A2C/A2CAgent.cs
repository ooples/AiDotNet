using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ActivationFunctions;
using AiDotNet.ReinforcementLearning.Common;
using AiDotNet.Helpers;

namespace AiDotNet.ReinforcementLearning.Agents.A2C;

/// <summary>
/// Advantage Actor-Critic (A2C) agent for reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// A2C is a synchronous, simpler version of A3C that combines policy gradients with value
/// function learning. It's the foundation for many modern RL algorithms including PPO.
/// </para>
/// <para><b>For Beginners:</b>
/// A2C learns two networks simultaneously:
/// - **Actor**: Decides which action to take (policy)
/// - **Critic**: Evaluates how good the current state is (value function)
///
/// The critic helps the actor learn faster by providing better feedback than rewards alone.
/// Think of it like having a coach (critic) give you targeted advice instead of just
/// saying "good" or "bad" after the game ends.
///
/// A2C is simpler than PPO but still very effective. Good starting point for actor-critic methods.
/// </para>
/// <para><b>Reference:</b>
/// Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning", 2016 (describes A3C, A2C is the synchronous version).
/// </para>
/// </remarks>
public class A2CAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private A2COptions<T> _a2cOptions;
    private readonly Trajectory<T> _trajectory;

    private NeuralNetwork<T> _policyNetwork;
    private NeuralNetwork<T> _valueNetwork;

    /// <inheritdoc/>
    public override int FeatureCount => _a2cOptions.StateSize;

    public A2CAgent(A2COptions<T> options)
        : base(new ReinforcementLearningOptions<T>
        {
            LearningRate = options.PolicyLearningRate,
            DiscountFactor = options.DiscountFactor,
            LossFunction = new MeanSquaredError<T>(),
            Seed = options.Seed
        })
    {
        _a2cOptions = options ?? throw new ArgumentNullException(nameof(options));
        _trajectory = new Trajectory<T>();

        _policyNetwork = BuildPolicyNetwork();
        _valueNetwork = BuildValueNetwork();

        Networks.Add(_policyNetwork);
        Networks.Add(_valueNetwork);
    }

    private NeuralNetwork<T> BuildPolicyNetwork()
    {
        var layers = new List<ILayer<T>>();
        int prevSize = _a2cOptions.StateSize;

        foreach (var hiddenSize in _a2cOptions.PolicyHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(prevSize, hiddenSize, new TanhActivation<T>()));
            prevSize = hiddenSize;
        }

        int outputSize = _a2cOptions.IsContinuous ? _a2cOptions.ActionSize * 2 : _a2cOptions.ActionSize;
        layers.Add(new DenseLayer<T>(prevSize, outputSize, new LinearActivation<T>()));

        return new NeuralNetwork<T>(new NeuralNetworkArchitecture<T>
        {
            InputSize = _a2cOptions.StateSize,
            OutputSize = outputSize,
            Layers = layers,
            TaskType = TaskType.Regression
        });
    }

    private NeuralNetwork<T> BuildValueNetwork()
    {
        var layers = new List<ILayer<T>>();
        int prevSize = _a2cOptions.StateSize;

        foreach (var hiddenSize in _a2cOptions.ValueHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(prevSize, hiddenSize, new TanhActivation<T>()));
            prevSize = hiddenSize;
        }

        layers.Add(new DenseLayer<T>(prevSize, 1, new LinearActivation<T>()));

        return new NeuralNetwork<T>(new NeuralNetworkArchitecture<T>
        {
            InputSize = _a2cOptions.StateSize,
            OutputSize = 1,
            Layers = layers,
            TaskType = TaskType.Regression
        }, _a2cOptions.ValueLossFunction);
    }

    /// <inheritdoc/>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var policyOutput = _policyNetwork.Forward(state);

        if (_a2cOptions.IsContinuous)
        {
            return SampleContinuousAction(policyOutput, training);
        }
        else
        {
            return SampleDiscreteAction(policyOutput, training);
        }
    }

    private Vector<T> SampleDiscreteAction(Vector<T> logits, bool training)
    {
        var probs = Softmax(logits);
        int actionIndex = training ? SampleCategorical(probs) : ArgMax(probs);

        var action = new Vector<T>(_a2cOptions.ActionSize);
        action[actionIndex] = NumOps.One;
        return action;
    }

    private Vector<T> SampleContinuousAction(Vector<T> output, bool training)
    {
        var action = new Vector<T>(_a2cOptions.ActionSize);

        for (int i = 0; i < _a2cOptions.ActionSize; i++)
        {
            var mean = output[i];
            var logStd = output[_a2cOptions.ActionSize + i];
            var std = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(logStd)));

            if (training)
            {
                // Sample from Gaussian using MathHelper
                var noise = MathHelper.GetNormalRandom<T>(NumOps.Zero, NumOps.One);
                action[i] = NumOps.Add(mean, NumOps.Multiply(std, noise));
            }
            else
            {
                action[i] = mean;
            }
        }

        return action;
    }

    /// <inheritdoc/>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        var value = _valueNetwork.Forward(state)[0];
        var logProb = ComputeLogProb(state, action);
        _trajectory.AddStep(state, action, reward, value, logProb, done);
    }

    private T ComputeLogProb(Vector<T> state, Vector<T> action)
    {
        var policyOutput = _policyNetwork.Forward(state);

        if (_a2cOptions.IsContinuous)
        {
            T totalLogProb = NumOps.Zero;
            for (int i = 0; i < _a2cOptions.ActionSize; i++)
            {
                var mean = policyOutput[i];
                var logStd = policyOutput[_a2cOptions.ActionSize + i];
                var std = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(logStd)));
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
        else
        {
            var probs = Softmax(policyOutput);
            int actionIndex = ArgMax(action);
            return NumOps.FromDouble(Math.Log(NumOps.ToDouble(probs[actionIndex]) + 1e-10));
        }
    }

    /// <inheritdoc/>
    public override T Train()
    {
        // A2C trains after fixed number of steps
        if (_trajectory.Length < _a2cOptions.StepsPerUpdate)
        {
            return NumOps.Zero;
        }

        TrainingSteps++;

        // Compute returns and advantages
        ComputeAdvantages();

        // Update networks
        T policyLoss = NumOps.Zero;
        T valueLoss = NumOps.Zero;
        T entropy = NumOps.Zero;

        for (int i = 0; i < _trajectory.Length; i++)
        {
            var state = _trajectory.States[i];
            var action = _trajectory.Actions[i];
            var advantage = _trajectory.Advantages![i];
            var targetReturn = _trajectory.Returns![i];

            // Policy loss: -log_prob * advantage
            var logProb = ComputeLogProb(state, action);
            policyLoss = NumOps.Subtract(policyLoss,
                NumOps.Multiply(logProb, advantage));

            // Value loss: (V - return)^2
            var predictedValue = _valueNetwork.Forward(state)[0];
            var valueDiff = NumOps.Subtract(predictedValue, targetReturn);
            valueLoss = NumOps.Add(valueLoss,
                NumOps.Multiply(valueDiff, valueDiff));

            // Entropy for exploration
            entropy = NumOps.Add(entropy, ComputeEntropy(state));
        }

        // Average losses
        var batchSize = NumOps.FromDouble(_trajectory.Length);
        policyLoss = NumOps.Divide(policyLoss, batchSize);
        valueLoss = NumOps.Divide(valueLoss, batchSize);
        entropy = NumOps.Divide(entropy, batchSize);

        // Combined loss
        var totalLoss = NumOps.Add(policyLoss,
            NumOps.Add(
                NumOps.Multiply(_a2cOptions.ValueLossCoefficient, valueLoss),
                NumOps.Multiply(_a2cOptions.EntropyCoefficient, NumOps.Negate(entropy))
            )
        );

        // Backpropagate through policy and value networks
        // We accumulate gradients over the batch before updating
        for (int i = 0; i < _trajectory.Length; i++)
        {
            var state = _trajectory.States[i];
            var action = _trajectory.Actions[i];
            var advantage = _trajectory.Advantages![i];
            var targetReturn = _trajectory.Returns![i];
            
            // Policy gradient: compute ∇ loss w.r.t. policy output
            var policyOutput = _policyNetwork.Forward(state);
            var policyGradient = ComputePolicyOutputGradient(policyOutput, action, advantage);
            _policyNetwork.Backward(policyGradient);
            
            // Value gradient: ∇ MSE w.r.t. value output = 2 * (V - target) / batchSize
            var predictedValue = _valueNetwork.Forward(state)[0];
            var valueDiff = NumOps.Subtract(predictedValue, targetReturn);
            var valueGradient = new Vector<T>(1);
            valueGradient[0] = NumOps.Divide(
                NumOps.Multiply(NumOps.FromDouble(2.0), valueDiff),
                NumOps.FromDouble(_trajectory.Length));
            _valueNetwork.Backward(valueGradient);
        }
        
        // Now update network parameters using accumulated gradients
        UpdatePolicyNetwork();
        UpdateValueNetwork();

        LossHistory.Add(totalLoss);
        _trajectory.Clear();

        return totalLoss;
    }

    private void ComputeAdvantages()
    {
        var advantages = new List<T>();
        var returns = new List<T>();

        T runningReturn = NumOps.Zero;

        for (int t = _trajectory.Length - 1; t >= 0; t--)
        {
            if (_trajectory.Dones[t])
            {
                runningReturn = _trajectory.Rewards[t];
            }
            else
            {
                runningReturn = NumOps.Add(
                    _trajectory.Rewards[t],
                    NumOps.Multiply(DiscountFactor, runningReturn)
                );
            }

            returns.Insert(0, runningReturn);
            var advantage = NumOps.Subtract(runningReturn, _trajectory.Values[t]);
            advantages.Insert(0, advantage);
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

    private void UpdatePolicyNetwork()
    {
        var params_ = _policyNetwork.GetFlattenedParameters();
        var grads = _policyNetwork.GetFlattenedGradients();

        for (int i = 0; i < params_.Length; i++)
        {
            var update = NumOps.Multiply(_a2cOptions.PolicyLearningRate, grads[i]);
            params_[i] = NumOps.Add(params_[i], update);
        }

        _policyNetwork.UpdateParameters(params_);
    }

    private void UpdateValueNetwork()
    {
        var params_ = _valueNetwork.GetFlattenedParameters();
        var grads = _valueNetwork.GetFlattenedGradients();

        for (int i = 0; i < params_.Length; i++)
        {
            var update = NumOps.Multiply(_a2cOptions.ValueLearningRate, grads[i]);
            params_[i] = NumOps.Subtract(params_[i], update);
        }

        _valueNetwork.UpdateParameters(params_);
    }

    private T ComputeEntropy(Vector<T> state)
    {
        var policyOutput = _policyNetwork.Forward(state);

        if (_a2cOptions.IsContinuous)
        {
            T entropy = NumOps.Zero;
            for (int i = 0; i < _a2cOptions.ActionSize; i++)
            {
                var logStd = policyOutput[_a2cOptions.ActionSize + i];
                entropy = NumOps.Add(entropy,
                    NumOps.Add(NumOps.FromDouble(0.5 * Math.Log(2 * Math.PI * Math.E)), logStd)
                );
            }
            return entropy;
        }
        else
        {
            var probs = Softmax(policyOutput);
            T entropy = NumOps.Zero;

            for (int i = 0; i < probs.Length; i++)
            {
                var p = NumOps.ToDouble(probs[i]);
                if (p > 1e-10)
                {
                    entropy = NumOps.Subtract(entropy, NumOps.FromDouble(p * Math.Log(p)));
                }
            }
            return entropy;
        }
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
            ModelType = ModelType.A2CAgent,
            FeatureCount = _a2cOptions.StateSize,
            TrainingSampleCount = TrainingSteps,
            Parameters = GetParameters()
        };
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        writer.Write(_a2cOptions.StateSize);
        writer.Write(_a2cOptions.ActionSize);

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

        reader.ReadInt32(); // stateSize
        reader.ReadInt32(); // actionSize

        var policyLength = reader.ReadInt32();
        var policyBytes = reader.ReadBytes(policyLength);
        _policyNetwork.Deserialize(policyBytes);

        var valueLength = reader.ReadInt32();
        var valueBytes = reader.ReadBytes(valueLength);
        _valueNetwork.Deserialize(valueBytes);
    }

    /// <inheritdoc/>
    public override Matrix<T> GetParameters()
    {
        var policyParams = _policyNetwork.GetFlattenedParameters();
        var valueParams = _valueNetwork.GetFlattenedParameters();

        var total = policyParams.Length + valueParams.Length;
        var matrix = new Matrix<T>(total, 1);

        int idx = 0;
        for (int i = 0; i < policyParams.Length; i++) matrix[idx++, 0] = policyParams[i];
        for (int i = 0; i < valueParams.Length; i++) matrix[idx++, 0] = valueParams[i];

        return matrix;
    }

    /// <inheritdoc/>
    public override void SetParameters(Matrix<T> parameters)
    {
        var policyParams = _policyNetwork.GetFlattenedParameters();
        var valueParams = _valueNetwork.GetFlattenedParameters();

        var policyVector = new Vector<T>(policyParams.Length);
        var valueVector = new Vector<T>(valueParams.Length);

        int idx = 0;
        for (int i = 0; i < policyParams.Length; i++) policyVector[i] = parameters[idx++, 0];
        for (int i = 0; i < valueParams.Length; i++) valueVector[i] = parameters[idx++, 0];

        _policyNetwork.UpdateParameters(policyVector);
        _valueNetwork.UpdateParameters(valueVector);
    }

    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new A2CAgent<T>(_a2cOptions);
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc/>
    public override (Matrix<T> Gradients, T Loss) ComputeGradients(
        Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return (GetParameters(), NumOps.Zero);
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Matrix<T> gradients, T learningRate)
    {
        // Not directly applicable
    }

    // Helper methods
    private Vector<T> Softmax(Vector<T> logits)
    {
        var maxLogit = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (NumOps.ToDouble(logits[i]) > NumOps.ToDouble(maxLogit))
                maxLogit = logits[i];
        }

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
            if (rand < cumProb) return i;
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

    private Vector<T> ComputePolicyOutputGradient(Vector<T> policyOutput, Vector<T> action, T advantage)
    {
        var gradient = new Vector<T>(policyOutput.Length);
        var scaledAdvantage = NumOps.Divide(advantage, NumOps.FromDouble(_trajectory.Length));
        
        if (_a2cOptions.IsContinuous)
        {
            // Continuous: Gaussian policy [mean, log_std]
            int actionSize = _a2cOptions.ActionSize;
            for (int i = 0; i < actionSize; i++)
            {
                var mean = policyOutput[i];
                var logStd = policyOutput[actionSize + i];
                var std = MathHelper.Exp(logStd);
                var actionDiff = NumOps.Subtract(action[i], mean);
                var stdSquared = NumOps.Multiply(std, std);
                
                // ∇μ: -(a - μ) / σ² * advantage
                gradient[i] = NumOps.Negate(
                    NumOps.Multiply(scaledAdvantage, NumOps.Divide(actionDiff, stdSquared)));
                    
                // ∇log_σ: -((a-μ)² / σ² - 1) * advantage
                var normalizedDiff = NumOps.Divide(actionDiff, std);
                var term = NumOps.Subtract(NumOps.Multiply(normalizedDiff, normalizedDiff), NumOps.One);
                gradient[actionSize + i] = NumOps.Negate(NumOps.Multiply(scaledAdvantage, term));
            }
        }
        else
        {
            // Discrete: softmax policy
            var softmax = ComputeSoftmax(policyOutput);
            int selectedAction = GetDiscreteAction(action);
            
            for (int i = 0; i < policyOutput.Length; i++)
            {
                var indicator = (i == selectedAction) ? NumOps.One : NumOps.Zero;
                var grad = NumOps.Subtract(indicator, softmax[i]);
                gradient[i] = NumOps.Negate(NumOps.Multiply(scaledAdvantage, grad));
            }
        }
        
        return gradient;
    }
    
    private Vector<T> ComputeSoftmax(Vector<T> logits)
    {
        var softmax = new Vector<T>(logits.Length);
        T maxLogit = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (NumOps.Compare(logits[i], maxLogit) > 0)
                maxLogit = logits[i];
        }
        
        T sumExp = NumOps.Zero;
        for (int i = 0; i < logits.Length; i++)
        {
            var exp = MathHelper.Exp(NumOps.Subtract(logits[i], maxLogit));
            softmax[i] = exp;
            sumExp = NumOps.Add(sumExp, exp);
        }
        
        for (int i = 0; i < softmax.Length; i++)
        {
            softmax[i] = NumOps.Divide(softmax[i], sumExp);
        }
        
        return softmax;
    }
    
    private int GetDiscreteAction(Vector<T> action)
    {
        for (int i = 0; i < action.Length; i++)
        {
            if (NumOps.Compare(action[i], NumOps.FromDouble(0.5)) > 0)
                return i;
        }
        return 0;
    }

}
