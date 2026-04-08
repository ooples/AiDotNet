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
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Agents.SAC;

/// <summary>
/// Soft Actor-Critic (SAC) agent for continuous control reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SAC is a state-of-the-art off-policy actor-critic algorithm that achieves high sample
/// efficiency and robustness by incorporating maximum entropy reinforcement learning.
/// It's particularly effective for continuous control tasks.
/// </para>
/// <para><b>For Beginners:</b>
/// SAC is one of the best algorithms for continuous control (robot movement, etc.).
///
/// Key innovations:
/// - **Maximum Entropy**: Learns to be both effective AND diverse
/// - **Twin Q-Networks**: Two critics prevent overestimation
/// - **Automatic Tuning**: Adjusts exploration automatically
/// - **Off-Policy**: Very sample efficient
///
/// Think of it like learning to drive: you want to reach your destination (high reward)
/// but also maintain flexibility in how you drive (high entropy). This makes the policy
/// more robust and adaptable.
///
/// Used by: Boston Dynamics robots, autonomous vehicles, dexterous manipulation
/// </para>
/// <para><b>Reference:</b>
/// Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor", 2018.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Soft Actor-Critic agent for continuous control
/// var options = new SACOptions&lt;double&gt; { StateSize = 4, ActionSize = 2, AutoTuneAlpha = true };
/// var agent = new SACAgent&lt;double&gt;(options);
///
/// // Select a continuous action with entropy-driven exploration
/// var state = new Vector&lt;double&gt;(new double[] { 0.5, -0.3, 1.0, 0.2 });
/// var action = agent.SelectAction(state);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor",
    "https://arxiv.org/abs/1801.01290",
    Year = 2018,
    Authors = "Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S.")]
public class SACAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private SACOptions<T> _sacOptions;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _sacOptions;
    private readonly UniformReplayBuffer<T, Vector<T>, Vector<T>> _replayBuffer;

    private INeuralNetwork<T> _policyNetwork;      // Actor (stochastic policy)
    private INeuralNetwork<T> _q1Network;          // First Q-network (critic 1)
    private INeuralNetwork<T> _q2Network;          // Second Q-network (critic 2)
    private INeuralNetwork<T> _q1TargetNetwork;    // Target for Q1
    private INeuralNetwork<T> _q2TargetNetwork;    // Target for Q2

    private T _logAlpha;                          // Log of temperature parameter
    private int _steps;

    /// <inheritdoc/>
    public override int FeatureCount => _sacOptions.StateSize;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public SACAgent()
        : this(new SACOptions<T> { StateSize = 4, ActionSize = 2 })
    {
    }

    /// <summary>
    /// Initializes a new instance of the SACAgent class.
    /// </summary>
    public SACAgent(SACOptions<T> options)
        : base(new ReinforcementLearningOptions<T>
        {
            LearningRate = options.PolicyLearningRate,
            DiscountFactor = options.DiscountFactor,
            LossFunction = options.QLossFunction,
            Seed = options.Seed,
            BatchSize = options.BatchSize,
            ReplayBufferSize = options.ReplayBufferSize,
            WarmupSteps = options.WarmupSteps
        })
    {
        Guard.NotNull(options);
        _sacOptions = options;
        _replayBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(options.ReplayBufferSize, options.Seed);
        _steps = 0;
        _logAlpha = NumOps.FromDouble(Math.Log(NumOps.ToDouble(options.InitialTemperature)));

        // Build networks
        _policyNetwork = BuildPolicyNetwork();
        _q1Network = BuildQNetwork();
        _q2Network = BuildQNetwork();
        _q1TargetNetwork = BuildQNetwork();
        _q2TargetNetwork = BuildQNetwork();

        // Initialize target networks
        CopyNetworkWeights(_q1Network, _q1TargetNetwork);
        CopyNetworkWeights(_q2Network, _q2TargetNetwork);

        // Register networks
        Networks.Add(_policyNetwork);
        Networks.Add(_q1Network);
        Networks.Add(_q2Network);
        Networks.Add(_q1TargetNetwork);
        Networks.Add(_q2TargetNetwork);
    }

    private NeuralNetwork<T> BuildPolicyNetwork()
    {
        // Policy network outputs mean and log_std for Gaussian policy
        var layers = new List<ILayer<T>>();
        int prevSize = _sacOptions.StateSize;

        foreach (var hiddenSize in _sacOptions.PolicyHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(prevSize, hiddenSize, (IActivationFunction<T>)new ReLUActivation<T>()));
            prevSize = hiddenSize;
        }

        // Output: mean and log_std for each action dimension
        layers.Add(new DenseLayer<T>(prevSize, _sacOptions.ActionSize * 2, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _sacOptions.StateSize,
            outputSize: _sacOptions.ActionSize * 2,
            layers: layers
        );

        return new NeuralNetwork<T>(architecture);
    }

    private NeuralNetwork<T> BuildQNetwork()
    {
        // Q-network takes state and action as input
        var layers = new List<ILayer<T>>();
        int inputSize = _sacOptions.StateSize + _sacOptions.ActionSize;
        int prevSize = inputSize;

        foreach (var hiddenSize in _sacOptions.QHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(prevSize, hiddenSize, (IActivationFunction<T>)new ReLUActivation<T>()));
            prevSize = hiddenSize;
        }

        // Output: single Q-value
        layers.Add(new DenseLayer<T>(prevSize, 1, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: inputSize,
            outputSize: 1,
            layers: layers
        );

        return new NeuralNetwork<T>(architecture, lossFunction: _sacOptions.QLossFunction);
    }

    /// <inheritdoc/>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var policyOutputTensor = _policyNetwork.Predict(stateTensor);
        var policyOutput = policyOutputTensor.ToVector();
        var (action, _) = SampleAction(policyOutput, training);
        return action;
    }

    private (Vector<T> Action, T LogProb) SampleAction(Vector<T> policyOutput, bool training)
    {
        var action = new Vector<T>(_sacOptions.ActionSize);
        T totalLogProb = NumOps.Zero;

        for (int i = 0; i < _sacOptions.ActionSize; i++)
        {
            var mean = policyOutput[i];
            var logStd = policyOutput[_sacOptions.ActionSize + i];

            // Clip log_std for numerical stability using MathHelper
            logStd = MathHelper.Clamp<T>(logStd, NumOps.FromDouble(-20), NumOps.FromDouble(2));
            var std = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(logStd)));

            if (training)
            {
                // Sample from Gaussian using MathHelper
                var noise = MathHelper.GetNormalRandom<T>(NumOps.Zero, NumOps.One);
                var rawAction = NumOps.Add(mean, NumOps.Multiply(std, noise));

                // Apply tanh squashing using MathHelper
                action[i] = MathHelper.Tanh<T>(rawAction);

                // Compute log prob with tanh correction
                var gaussianLogProb = NumOps.FromDouble(
                    -0.5 * Math.Log(2 * Math.PI) -
                    NumOps.ToDouble(logStd) -
                    0.5 * NumOps.ToDouble(NumOps.Multiply(noise, noise))
                );

                // Tanh correction: log(1 - tanh^2(x))
                var tanhCorrection = NumOps.FromDouble(
                    Math.Log(1 - Math.Pow(NumOps.ToDouble(action[i]), 2) + 1e-6)
                );

                totalLogProb = NumOps.Add(totalLogProb,
                    NumOps.Subtract(gaussianLogProb, tanhCorrection));
            }
            else
            {
                // Deterministic: use mean with tanh using MathHelper
                action[i] = MathHelper.Tanh<T>(mean);
            }
        }

        return (action, totalLogProb);
    }

    /// <inheritdoc/>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        _replayBuffer.Add(new Experience<T, Vector<T>, Vector<T>>(state, action, reward, nextState, done));
    }

    /// <inheritdoc/>
    public override T Train()
    {
        _steps++;
        TrainingSteps++;

        if (_steps < _sacOptions.WarmupSteps || !_replayBuffer.CanSample(_sacOptions.BatchSize))
        {
            return NumOps.Zero;
        }

        T totalLoss = NumOps.Zero;

        // Multiple gradient steps per environment step (Haarnoja et al. 2018)
        for (int g = 0; g < _sacOptions.GradientSteps; g++)
        {
            var batch = _replayBuffer.Sample(_sacOptions.BatchSize);
            int stateSize = _sacOptions.StateSize;
            int actionSize = _sacOptions.ActionSize;
            int batchCount = batch.Count;
            T alpha = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(_logAlpha)));

            // Build batched state-action tensors for critic training
            var batchStateActions = new Tensor<T>([batchCount, stateSize + actionSize]);
            var batchStates = new Tensor<T>([batchCount, stateSize]);
            for (int i = 0; i < batchCount; i++)
            {
                for (int j = 0; j < stateSize; j++)
                {
                    batchStates[i, j] = batch[i].State[j];
                    batchStateActions[i, j] = batch[i].State[j];
                }
                for (int j = 0; j < actionSize; j++)
                    batchStateActions[i, stateSize + j] = batch[i].Action[j];
            }

            // --- Compute Q targets (outside tape — target networks not trained) ---
            // y = r + gamma * (1-done) * (min(Q_tgt1, Q_tgt2) - alpha * log_pi_next)
            var targetQ = new Tensor<T>([batchCount, 1]);
            for (int i = 0; i < batchCount; i++)
            {
                var (nextAction, nextLogProb) = SampleAction(batch[i].NextState, false);
                var nextSA = ConcatenateStateAction(batch[i].NextState, nextAction);
                var nextSATensor = Tensor<T>.FromVector(nextSA, [1, stateSize + actionSize]);
                var q1Next = _q1TargetNetwork.Predict(nextSATensor).ToVector()[0];
                var q2Next = _q2TargetNetwork.Predict(nextSATensor).ToVector()[0];
                var minQNext = NumOps.LessThan(q1Next, q2Next) ? q1Next : q2Next;
                var softV = NumOps.Subtract(minQNext, NumOps.Multiply(alpha, nextLogProb));
                var doneT = batch[i].Done ? NumOps.Zero : NumOps.One;
                targetQ[i, 0] = NumOps.Add(batch[i].Reward,
                    NumOps.Multiply(DiscountFactor, NumOps.Multiply(doneT, softV)));
            }

            // --- Update both Q-networks (MSE on TD targets) ---
            _q1Network.Train(batchStateActions, targetQ);
            T q1Loss = _q1Network.GetLastLoss();
            _q2Network.Train(batchStateActions, targetQ);
            T q2Loss = _q2Network.GetLastLoss();
            T qLoss = NumOps.Divide(NumOps.Add(q1Loss, q2Loss), NumOps.FromDouble(2.0));

            // --- Update actor (minimize alpha * log_pi - Q via engine ops) ---
            var trainableActor = (NeuralNetworkBase<T>)_policyNetwork;
            T policyLoss = trainableActor.TrainWithCustomLoss(batchStates, actorOutput =>
            {
                // actorOutput = [batchCount, actionSize*2] for continuous = [means, logStds]
                var means = Engine.TensorSlice(actorOutput, [0, 0], [batchCount, actionSize]);
                var logStds = Engine.TensorSlice(actorOutput, [0, actionSize], [batchCount, actionSize * 2]);

                // Compute log-probabilities via engine ops
                // Sample actions from policy for reparameterization
                var stds = Engine.TensorExp(logStds);
                var noise = new Tensor<T>([batchCount, actionSize]);
                var rng = Tensors.Helpers.RandomHelper.CreateSecureRandom();
                for (int i = 0; i < noise.Length; i++)
                    noise[i] = NumOps.FromDouble(rng.NextDouble() * 2.0 - 1.0);
                var sampledActions = Engine.TensorAdd(means, Engine.TensorMultiply(stds, noise));

                var logProbs = PolicyDistributionHelper<T>.ComputeGaussianLogProb(means, logStds, sampledActions);

                // Build state-action for Q evaluation
                var stateActionForQ = new Tensor<T>([batchCount, stateSize + actionSize]);
                for (int i = 0; i < batchCount; i++)
                {
                    for (int j = 0; j < stateSize; j++)
                        stateActionForQ[i, j] = batchStates[i, j];
                    for (int j = 0; j < actionSize; j++)
                        stateActionForQ[i, stateSize + j] = sampledActions[i * actionSize + j];
                }

                // Q values (detached — critics not updated here)
                var q1 = _q1Network.Predict(stateActionForQ);
                var q2 = _q2Network.Predict(stateActionForQ);
                // min(Q1, Q2): use -max(-Q1, -Q2)
                var negQ1 = Engine.TensorNegate(q1);
                var negQ2 = Engine.TensorNegate(q2);
                var minQ = Engine.TensorNegate(Engine.TensorMax(negQ1, negQ2));

                // policy loss = mean(alpha * log_pi - min_Q)
                var alphaLogPi = Engine.TensorMultiplyScalar(logProbs, alpha);
                var flatMinQ = new Tensor<T>([batchCount]);
                for (int i = 0; i < batchCount; i++)
                    flatMinQ[i] = minQ[i];

                var loss = Engine.TensorSubtract(alphaLogPi, flatMinQ);
                var allAxes = Enumerable.Range(0, loss.Shape.Length).ToArray();
                return Engine.ReduceMean(loss, allAxes, keepDims: false);
            });

            // Update temperature (alpha)
            if (_sacOptions.AutoTuneTemperature)
            {
                UpdateTemperature(batch);
            }

            // Soft update target networks (Polyak averaging)
            SoftUpdateTargets();

            totalLoss = NumOps.Add(totalLoss, NumOps.Add(qLoss, policyLoss));
        }

        var avgLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(_sacOptions.GradientSteps));
        LossHistory.Add(avgLoss);

        return avgLoss;
    }


    private Vector<T> ComputeSACPolicyGradient(
        Vector<T> policyOutput, Vector<T> action, T alpha, T logProb, T qValue)
    {
        // SAC gradient: ∇θ [α log π(a|s) - Q(s,a)]
        // Split into two terms:
        // 1. Entropy term: α * ∇θ log π(a|s)
        // 2. Q term: -∇θ Q(s, f_θ(s, ε)) via reparameterization

        var gradient = new Vector<T>(_sacOptions.ActionSize * 2);

        for (int i = 0; i < _sacOptions.ActionSize; i++)
        {
            var mean = policyOutput[i];
            var logStd = policyOutput[_sacOptions.ActionSize + i];
            var clippedLogStd = MathHelper.Clamp<T>(logStd, NumOps.FromDouble(-20), NumOps.FromDouble(2));
            var std = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(clippedLogStd)));

            // Reconstruct the noise used: ε = (atanh(a) - μ) / σ
            // Note: action[i] is already tanh-squashed
            var atanhAction = NumOps.FromDouble(Math.Log((1.0 + NumOps.ToDouble(action[i])) /
                                                           (1.0 - NumOps.ToDouble(action[i]) + 1e-6)) / 2.0);
            var rawAction = atanhAction; // This was tanh^(-1)(action)

            // Gradient of log probability w.r.t. mean and log_std
            // For Gaussian: ∂log_π/∂μ = (a - μ) / σ²
            //               ∂log_π/∂log_σ = -1 + (a - μ)² / σ²
            var actionDiff = NumOps.Subtract(rawAction, mean);
            var stdSquared = NumOps.Multiply(std, std);

            var dLogPi_dMean = NumOps.Divide(actionDiff, stdSquared);
            var dLogPi_dLogStd = NumOps.Subtract(
                NumOps.FromDouble(-1.0),
                NumOps.Divide(NumOps.Multiply(actionDiff, actionDiff), stdSquared)
            );

            // Entropy gradient: α * ∇θ log π
            var entropyGradMean = NumOps.Multiply(alpha, dLogPi_dMean);
            var entropyGradLogStd = NumOps.Multiply(alpha, dLogPi_dLogStd);

            // Q-value gradient using reparameterization trick
            // SAC gradient: ∇θ [α log π(a|s) - Q(s, f_θ(ε))]
            // The Q term requires ∇_μ Q and ∇_log_σ Q through the action
            // For tanh-squashed Gaussian: a = tanh(μ + σε)
            //
            // Approximation: Since we can't easily compute ∇_a Q analytically,
            // we use the fact that for policy gradient methods:
            // ∇θ Q(s, f_θ(ε)) ≈ ∇θ f_θ(ε) * ∇_a Q(s,a)
            //
            // In this simplified version, we scale the gradient by Q-value
            // A more accurate implementation would use finite differences or
            // automatic differentiation to compute ∇_a Q
            var qGradScale = NumOps.Negate(NumOps.Divide(qValue, NumOps.Add(std, NumOps.FromDouble(1e-6))));
            var qGradMean = qGradScale;  // Simplified: gradient flows through mean
            var qGradLogStd = NumOps.Multiply(qGradScale, std);  // Scaled by std

            // Total gradient: α * ∇θ log π - ∇θ Q
            // We negate the sum because networks minimize loss (gradient descent)
            // but we want to maximize J = E[Q - α log π]
            gradient[i] = NumOps.Negate(NumOps.Add(entropyGradMean, qGradMean));
            gradient[_sacOptions.ActionSize + i] = NumOps.Negate(NumOps.Add(entropyGradLogStd, qGradLogStd));
        }

        return gradient;
    }

    private void UpdateTemperature(List<Experience<T, Vector<T>, Vector<T>>> batch)
    {
        if (!_sacOptions.AutoTuneTemperature) return;

        T totalEntropy = NumOps.Zero;

        foreach (var exp in batch)
        {
            var stateTensor = Tensor<T>.FromVector(exp.State);
            var policyOutputTensor = _policyNetwork.Predict(stateTensor);
            var policyOutput = policyOutputTensor.ToVector();
            var (_, logProb) = SampleAction(policyOutput, training: true);
            totalEntropy = NumOps.Add(totalEntropy, logProb);
        }

        var avgEntropy = NumOps.Divide(totalEntropy, NumOps.FromDouble(batch.Count));
        var targetEntropy = _sacOptions.TargetEntropy ?? NumOps.FromDouble(-_sacOptions.ActionSize);

        // Alpha loss: -alpha * (log_prob + target_entropy)
        var alphaLoss = NumOps.Multiply(
            NumOps.FromDouble(-Math.Exp(NumOps.ToDouble(_logAlpha))),
            NumOps.Add(avgEntropy, targetEntropy)
        );

        // Update log_alpha
        var alphaGrad = NumOps.Multiply(_sacOptions.AlphaLearningRate, alphaLoss);
        _logAlpha = NumOps.Subtract(_logAlpha, alphaGrad);
    }

    private void SoftUpdateTargets()
    {
        SoftUpdateNetwork(_q1Network, _q1TargetNetwork);
        SoftUpdateNetwork(_q2Network, _q2TargetNetwork);
    }

    private void SoftUpdateNetwork(INeuralNetwork<T> source, INeuralNetwork<T> target)
    {
        var sourceParams = source.GetParameters();
        var targetParams = target.GetParameters();

        var tau = _sacOptions.TargetUpdateTau;
        var oneMinusTau = NumOps.Subtract(NumOps.One, tau);

        for (int i = 0; i < targetParams.Length; i++)
        {
            targetParams[i] = NumOps.Add(
                NumOps.Multiply(tau, sourceParams[i]),
                NumOps.Multiply(oneMinusTau, targetParams[i])
            );
        }

        target.UpdateParameters(targetParams);
    }

    // UpdateNetworkParameters removed — networks now trained via Train() and
    // TrainWithCustomLoss which use configured optimizers and tape-based gradients.

    private Vector<T> ConcatenateStateAction(Vector<T> state, Vector<T> action)
    {
        var combined = new Vector<T>(state.Length + action.Length);
        for (int i = 0; i < state.Length; i++)
            combined[i] = state[i];
        for (int i = 0; i < action.Length; i++)
            combined[state.Length + i] = action[i];
        return combined;
    }

    /// <inheritdoc/>
    public override Dictionary<string, T> GetMetrics()
    {
        var baseMetrics = base.GetMetrics();
        baseMetrics["Alpha"] = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(_logAlpha)));
        baseMetrics["ReplayBufferSize"] = NumOps.FromDouble(_replayBuffer.Count);
        return baseMetrics;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            FeatureCount = _sacOptions.StateSize,
        };
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        writer.Write(_sacOptions.StateSize);
        writer.Write(_sacOptions.ActionSize);
        writer.Write(NumOps.ToDouble(_logAlpha));

        void WriteNetwork(INeuralNetwork<T> net)
        {
            var bytes = net.Serialize();
            writer.Write(bytes.Length);
            writer.Write(bytes);
        }

        WriteNetwork(_policyNetwork);
        WriteNetwork(_q1Network);
        WriteNetwork(_q2Network);
        WriteNetwork(_q1TargetNetwork);
        WriteNetwork(_q2TargetNetwork);

        return ms.ToArray();
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        reader.ReadInt32(); // stateSize
        reader.ReadInt32(); // actionSize
        _logAlpha = NumOps.FromDouble(reader.ReadDouble());

        void ReadNetwork(INeuralNetwork<T> net)
        {
            var len = reader.ReadInt32();
            var bytes = reader.ReadBytes(len);
            net.Deserialize(bytes);
        }

        ReadNetwork(_policyNetwork);
        ReadNetwork(_q1Network);
        ReadNetwork(_q2Network);
        ReadNetwork(_q1TargetNetwork);
        ReadNetwork(_q2TargetNetwork);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var policyParams = _policyNetwork.GetParameters();
        var q1Params = _q1Network.GetParameters();
        var q2Params = _q2Network.GetParameters();

        var total = policyParams.Length + q1Params.Length + q2Params.Length;
        var vector = new Vector<T>(total);

        int idx = 0;
        foreach (var p in policyParams) vector[idx++] = p;
        foreach (var p in q1Params) vector[idx++] = p;
        foreach (var p in q2Params) vector[idx++] = p;

        return vector;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        var policyParams = _policyNetwork.GetParameters();
        var q1Params = _q1Network.GetParameters();
        var q2Params = _q2Network.GetParameters();

        int idx = 0;
        var policyVec = new Vector<T>(policyParams.Length);
        var q1Vec = new Vector<T>(q1Params.Length);
        var q2Vec = new Vector<T>(q2Params.Length);

        for (int i = 0; i < policyParams.Length; i++) policyVec[i] = parameters[idx++];
        for (int i = 0; i < q1Params.Length; i++) q1Vec[i] = parameters[idx++];
        for (int i = 0; i < q2Params.Length; i++) q2Vec[i] = parameters[idx++];

        _policyNetwork.UpdateParameters(policyVec);
        _q1Network.UpdateParameters(q1Vec);
        _q2Network.UpdateParameters(q2Vec);

        // Update targets
        CopyNetworkWeights(_q1Network, _q1TargetNetwork);
        CopyNetworkWeights(_q2Network, _q2TargetNetwork);
    }

    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new SACAgent<T>(_sacOptions);
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
        // SAC uses actor-critic architecture with separate policy and value networks.
        // Gradients are computed and applied internally during Train() for each network.
        // External gradient application is not applicable for this algorithm.
        throw new NotSupportedException("SAC applies gradients internally during training. Use Train() method instead.");
    }

    // Helper methods
    private void CopyNetworkWeights(INeuralNetwork<T> source, INeuralNetwork<T> target)
    {
        target.UpdateParameters(source.GetParameters());
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
