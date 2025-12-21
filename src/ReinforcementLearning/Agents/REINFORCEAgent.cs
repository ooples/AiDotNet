using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ReinforcementLearning.Common;

namespace AiDotNet.ReinforcementLearning.Agents.REINFORCE;

/// <summary>
/// REINFORCE (Monte Carlo Policy Gradient) agent for reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// REINFORCE is the simplest and most fundamental policy gradient algorithm. It directly
/// optimizes the policy by following the gradient of expected returns. Despite its simplicity,
/// it forms the foundation for many modern RL algorithms.
/// </para>
/// <para><b>For Beginners:</b>
/// REINFORCE is the "hello world" of policy gradient methods. The algorithm is beautifully simple:
///
/// 1. Play an entire episode
/// 2. Calculate total rewards for each action
/// 3. Make good actions more likely, bad actions less likely
///
/// Think of it like learning to play a game:
/// - You play a round
/// - At the end, you see your score
/// - You adjust your strategy to do better next time
///
/// **Pros**: Simple, works for any problem, easy to understand
/// **Cons**: High variance, slow learning, requires complete episodes
///
/// Modern algorithms like PPO and A2C improve on REINFORCE's core ideas.
/// </para>
/// <para><b>Reference:</b>
/// Williams, R. J. (1992). "Simple statistical gradient-following algorithms for connectionist RL."
/// </para>
/// </remarks>
public class REINFORCEAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private REINFORCEOptions<T> _reinforceOptions;
    private readonly Trajectory<T> _trajectory;

    private NeuralNetwork<T> _policyNetwork;

    /// <inheritdoc/>
    public override int FeatureCount => _reinforceOptions.StateSize;

    public REINFORCEAgent(REINFORCEOptions<T> options)
        : base(new ReinforcementLearningOptions<T>
        {
            LearningRate = options.LearningRate,
            DiscountFactor = options.DiscountFactor,
            LossFunction = new MeanSquaredErrorLoss<T>(),
            Seed = options.Seed
        })
    {
        _reinforceOptions = options ?? throw new ArgumentNullException(nameof(options));
        _trajectory = new Trajectory<T>();

        _policyNetwork = BuildPolicyNetwork();
        Networks.Add(_policyNetwork);
    }

    private NeuralNetwork<T> BuildPolicyNetwork()
    {
        var layers = new List<ILayer<T>>();
        int prevSize = _reinforceOptions.StateSize;

        foreach (var hiddenSize in _reinforceOptions.HiddenLayers)
        {
            layers.Add(new DenseLayer<T>(prevSize, hiddenSize, (IActivationFunction<T>)new TanhActivation<T>()));
            prevSize = hiddenSize;
        }

        int outputSize = _reinforceOptions.IsContinuous
            ? _reinforceOptions.ActionSize * 2  // Mean and log_std for Gaussian
            : _reinforceOptions.ActionSize;      // Logits for categorical

        layers.Add(new DenseLayer<T>(prevSize, outputSize, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _reinforceOptions.StateSize,
            outputSize: outputSize,
            layers: layers);

        return new NeuralNetwork<T>(architecture);
    }

    /// <inheritdoc/>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var policyOutputTensor = _policyNetwork.Predict(stateTensor);
        var policyOutput = policyOutputTensor.ToVector();

        if (_reinforceOptions.IsContinuous)
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

        var action = new Vector<T>(_reinforceOptions.ActionSize);
        action[actionIndex] = NumOps.One;
        return action;
    }

    private Vector<T> SampleContinuousAction(Vector<T> output, bool training)
    {
        var action = new Vector<T>(_reinforceOptions.ActionSize);

        for (int i = 0; i < _reinforceOptions.ActionSize; i++)
        {
            var mean = output[i];
            var logStd = output[_reinforceOptions.ActionSize + i];
            var std = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(logStd)));

            if (training)
            {
                // Sample from Gaussian using MathHelper
                var noise = MathHelper.GetNormalRandom<T>(NumOps.Zero, NumOps.One);
                action[i] = NumOps.Add(mean, NumOps.Multiply(std, noise));
            }
            else
            {
                action[i] = mean;  // Deterministic for evaluation
            }
        }

        return action;
    }

    /// <inheritdoc/>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        // REINFORCE only needs states, actions, and rewards
        var logProb = ComputeLogProb(state, action);
        _trajectory.AddStep(state, action, reward, NumOps.Zero, logProb, done);
    }

    private T ComputeLogProb(Vector<T> state, Vector<T> action)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var policyOutputTensor = _policyNetwork.Predict(stateTensor);
        var policyOutput = policyOutputTensor.ToVector();

        if (_reinforceOptions.IsContinuous)
        {
            T totalLogProb = NumOps.Zero;

            for (int i = 0; i < _reinforceOptions.ActionSize; i++)
            {
                var mean = policyOutput[i];
                var logStd = policyOutput[_reinforceOptions.ActionSize + i];
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
        // REINFORCE trains after each complete episode
        if (_trajectory.Length == 0)
        {
            return NumOps.Zero;
        }

        TrainingSteps++;

        // Compute discounted returns
        ComputeReturns();

        // Compute policy loss: -log_prob * return
        T totalLoss = NumOps.Zero;

        for (int t = 0; t < _trajectory.Length; t++)
        {
            var state = _trajectory.States[t];
            var action = _trajectory.Actions[t];
            var returnVal = _trajectory.Returns![t];

            var logProb = ComputeLogProb(state, action);

            // Policy gradient: -log_prob * return
            var loss = NumOps.Multiply(NumOps.Negate(logProb), returnVal);
            totalLoss = NumOps.Add(totalLoss, loss);

            // Compute output gradient for REINFORCE: ∇ loss w.r.t. policy output
            // For discrete: gradient is -G_t * (1_{a=a_t} - π(a|s)) for each action
            // For continuous: gradient depends on distribution type
            var stateTensor = Tensor<T>.FromVector(state);
            var policyOutputTensor = _policyNetwork.Predict(stateTensor);
            var policyOutput = policyOutputTensor.ToVector();
            var outputGradient = new Vector<T>(policyOutput.Length);

            if (_reinforceOptions.IsContinuous)
            {
                // Continuous action space: Gaussian policy with mean and log_std
                // Output is [mean_1, ..., mean_n, log_std_1, ..., log_std_n]
                int actionSize = _reinforceOptions.ActionSize;
                for (int i = 0; i < actionSize; i++)
                {
                    var mean = policyOutput[i];
                    var logStd = policyOutput[actionSize + i];
                    var std = NumOps.Exp(logStd);

                    // Gradient of -log π(a|s) * G_t w.r.t. mean: -(a - μ) / σ² * G_t
                    var actionDiff = NumOps.Subtract(action[i], mean);
                    var stdSquared = NumOps.Multiply(std, std);
                    outputGradient[i] = NumOps.Negate(
                        NumOps.Multiply(returnVal, NumOps.Divide(actionDiff, stdSquared)));

                    // Gradient w.r.t. log_std: -((a-μ)² / σ² - 1) * G_t
                    var normalizedDiff = NumOps.Divide(actionDiff, std);
                    var term = NumOps.Subtract(NumOps.Multiply(normalizedDiff, normalizedDiff), NumOps.One);
                    outputGradient[actionSize + i] = NumOps.Negate(NumOps.Multiply(returnVal, term));
                }
            }
            else
            {
                // Discrete action space: softmax policy
                // Gradient: -G_t * (1_{a=a_t} - softmax(logits))
                var softmax = ComputeSoftmax(policyOutput);
                int selectedAction = GetDiscreteAction(action);

                for (int i = 0; i < policyOutput.Length; i++)
                {
                    var indicator = (i == selectedAction) ? NumOps.One : NumOps.Zero;
                    var grad = NumOps.Subtract(indicator, softmax[i]);
                    outputGradient[i] = NumOps.Negate(NumOps.Multiply(returnVal, grad));
                }
            }

            // Backpropagate through policy network
            var outputGradientTensor = Tensor<T>.FromVector(outputGradient);
            _policyNetwork.Backpropagate(outputGradientTensor);
        }

        // Average loss
        var avgLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(_trajectory.Length));

        // Update policy network
        UpdatePolicyNetwork();

        LossHistory.Add(avgLoss);
        _trajectory.Clear();

        return avgLoss;
    }

    private void ComputeReturns()
    {
        var returns = new List<T>();
        T runningReturn = NumOps.Zero;

        // Compute discounted returns backwards
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
        }

        // Normalize returns (reduces variance) using StatisticsHelper
        var stdReturn = StatisticsHelper<T>.CalculateStandardDeviation(returns);
        T meanReturn = NumOps.Zero;
        foreach (var ret in returns)
            meanReturn = NumOps.Add(meanReturn, ret);
        meanReturn = NumOps.Divide(meanReturn, NumOps.FromDouble(returns.Count));

        for (int i = 0; i < returns.Count; i++)
        {
            returns[i] = NumOps.Divide(
                NumOps.Subtract(returns[i], meanReturn),
                NumOps.Add(stdReturn, NumOps.FromDouble(1e-8))
            );
        }

        _trajectory.Returns = returns;
    }

    private void UpdatePolicyNetwork()
    {
        var params_ = _policyNetwork.GetParameters();
        var grads = _policyNetwork.GetGradients();

        for (int i = 0; i < params_.Length; i++)
        {
            var update = NumOps.Multiply(LearningRate, grads[i]);
            params_[i] = NumOps.Subtract(params_[i], update);
        }

        _policyNetwork.UpdateParameters(params_);
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
            ModelType = ModelType.ReinforcementLearning,  // Generic RL type
            FeatureCount = _reinforceOptions.StateSize,
        };
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        writer.Write(_reinforceOptions.StateSize);
        writer.Write(_reinforceOptions.ActionSize);

        var policyBytes = _policyNetwork.Serialize();
        writer.Write(policyBytes.Length);
        writer.Write(policyBytes);

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
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        return _policyNetwork.GetParameters();
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        _policyNetwork.UpdateParameters(parameters);
    }

    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new REINFORCEAgent<T>(_reinforceOptions);
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

    private Vector<T> ComputeSoftmax(Vector<T> logits)
    {
        var softmax = new Vector<T>(logits.Length);
        T maxLogit = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (NumOps.GreaterThan(logits[i], maxLogit))
                maxLogit = logits[i];
        }

        T sumExp = NumOps.Zero;
        for (int i = 0; i < logits.Length; i++)
        {
            var exp = NumOps.Exp(NumOps.Subtract(logits[i], maxLogit));
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
        // Find the index of the action (assumes one-hot encoding or argmax)
        for (int i = 0; i < action.Length; i++)
        {
            if (NumOps.GreaterThan(action[i], NumOps.FromDouble(0.5)))
                return i;
        }
        return 0;
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
