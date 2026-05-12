using System;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ReinforcementLearning.Common;
using AiDotNet.Validation;

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
/// <example>
/// <code>
/// // Create a REINFORCE agent for Monte Carlo policy gradient learning
/// var options = new REINFORCEOptions&lt;double&gt; { StateSize = 4, ActionSize = 2, LearningRate = 0.001 };
/// var agent = new REINFORCEAgent&lt;double&gt;(options);
///
/// // Select an action sampled from the learned policy
/// var state = new Vector&lt;double&gt;(new double[] { 0.5, -0.3, 1.0, 0.2 });
/// var action = agent.SelectAction(state);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning",
    "https://link.springer.com/article/10.1007/BF00992696",
    Year = 1992,
    Authors = "Williams, R. J.")]
public class REINFORCEAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private REINFORCEOptions<T> _reinforceOptions;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _reinforceOptions;
    private readonly Trajectory<T> _trajectory;

    private readonly INeuralNetwork<T> _policyNetwork;

    /// <inheritdoc/>
    public override int FeatureCount => _reinforceOptions.StateSize;

    public REINFORCEAgent() : this(new REINFORCEOptions<T>()) { }

    public REINFORCEAgent(REINFORCEOptions<T> options)
        : base(new ReinforcementLearningOptions<T>
        {
            LearningRate = options.LearningRate,
            DiscountFactor = options.DiscountFactor,
            LossFunction = new MeanSquaredErrorLoss<T>(),
            Seed = options.Seed
        })
    {
        Guard.NotNull(options);
        _reinforceOptions = options;
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
            layers.Add(new DenseLayer<T>(hiddenSize, (IActivationFunction<T>)new TanhActivation<T>()));
            prevSize = hiddenSize;
        }

        int outputSize = _reinforceOptions.IsContinuous
            ? _reinforceOptions.ActionSize * 2  // Mean and log_std for Gaussian
            : _reinforceOptions.ActionSize;      // Logits for categorical

        layers.Add(new DenseLayer<T>(outputSize, (IActivationFunction<T>)new IdentityActivation<T>()));

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

        if (training)
        {
            // Training: sample a discrete action and return a one-hot
            // vector so StoreExperience records the commitment.
            int actionIndex = SampleCategorical(probs);
            var action = new Vector<T>(_reinforceOptions.ActionSize);
            action[actionIndex] = NumOps.One;
            return action;
        }

        // Inference: return the full softmax distribution π(·|s). Same
        // convention as A3C / MuZero / RainbowDQN inference output.
        return probs;
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
    /// <remarks>
    /// One REINFORCE update step (Williams 1992 "Simple statistical
    /// gradient-following algorithms for connectionist reinforcement
    /// learning" Eq. 5; Sutton &amp; Barto 2018 §13.3 Algorithm box):
    /// <para><c>θ ← θ + α · Σ_t G_t · ∇_θ log π(a_t | s_t; θ)</c></para>
    /// where <c>G_t</c> is the discounted return from step t and
    /// <c>π(·|s; θ)</c> is the parameterized policy. Implemented via
    /// tape-based autodiff: build a batched log-prob tensor against the
    /// sampled actions, multiply by returns, sum (NOT mean — that
    /// rescales the per-sample gradient by 1/N and isn't the paper's
    /// objective), negate to convert maximization to minimization,
    /// backprop through the policy network.
    /// </remarks>
    public override T Train()
    {
        if (_trajectory.Length == 0)
        {
            return NumOps.Zero;
        }

        TrainingSteps++;

        // Compute discounted returns
        ComputeReturns();

        int stateSize = _reinforceOptions.StateSize;
        int trajLen = _trajectory.Length;
        var returns = _trajectory.Returns ?? throw new InvalidOperationException("Returns not initialized.");

        var batchStates = new Tensor<T>([trajLen, stateSize]);
        for (int i = 0; i < trajLen; i++)
            for (int j = 0; j < stateSize; j++)
                batchStates[i, j] = _trajectory.States[i][j];

        var returnsTensor = new Tensor<T>([trajLen]);
        for (int i = 0; i < trajLen; i++)
            returnsTensor[i] = returns[i];

        // Policy gradient objective: maximize Σ_t G_t · log π(a_t | s_t).
        // Minimize the negation. All ops below are engine-routed so the
        // gradient tape captures them through to the policy network's
        // trainable parameters.
        var trainablePolicy = (NeuralNetworkBase<T>)_policyNetwork;
        T avgLoss = trainablePolicy.TrainWithCustomLoss(batchStates, policyOutput =>
        {
            var eng = Engine;
            Tensor<T> logProbs;

            if (_reinforceOptions.IsContinuous)
            {
                int actionSize = _reinforceOptions.ActionSize;
                var means = eng.TensorSlice(policyOutput, [0, 0], [trajLen, actionSize]);
                var logStds = eng.TensorSlice(policyOutput, [0, actionSize], [trajLen, actionSize * 2]);

                var actionsTensor = new Tensor<T>([trajLen, actionSize]);
                for (int i = 0; i < trajLen; i++)
                    for (int j = 0; j < actionSize; j++)
                        actionsTensor[i, j] = _trajectory.Actions[i][j];

                logProbs = PolicyDistributionHelper<T>.ComputeGaussianLogProb(eng, means, logStds, actionsTensor);
            }
            else
            {
                var actionIndices = new int[trajLen];
                for (int i = 0; i < trajLen; i++)
                    actionIndices[i] = GetDiscreteAction(_trajectory.Actions[i]);

                logProbs = PolicyDistributionHelper<T>.ComputeDiscreteLogProb(eng, policyOutput, actionIndices);
            }

            // weighted = G_t · log π(a_t|s_t), shape [trajLen]
            var weighted = eng.TensorMultiply(logProbs, returnsTensor);
            // Sum (NOT mean) per the paper's Σ_t — mean would shrink the
            // gradient by 1/N which silently changes the effective
            // learning rate as trajectory length varies.
            var allAxes = Enumerable.Range(0, weighted.Shape.Length).ToArray();
            var summed = eng.ReduceSum(weighted, allAxes, keepDims: false);
            return eng.TensorNegate(summed);
        });

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

    // UpdatePolicyNetwork removed — policy network now trained via TrainWithCustomLoss
    // which uses the configured optimizer and tape-based gradient computation.

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
    /// <remarks>
    /// Calls <see cref="NeuralNetworkBase{T}.SetParameters"/> (not
    /// <c>UpdateParameters</c>) so the call is a full state restore that
    /// round-trips with <see cref="GetParameters"/>. <c>UpdateParameters</c>
    /// on the underlying network is the optimizer-style apply-delta path
    /// and doesn't guarantee that the resulting parameter vector equals
    /// the input; Clone would then produce a network whose
    /// <c>Predict(state)</c> diverges from the original on every call.
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        _policyNetwork.SetParameters(parameters);
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
            if (NumOps.GreaterThan(logits[i], maxLogit))
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
            if (NumOps.GreaterThan(vector[i], vector[maxIndex]))
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
