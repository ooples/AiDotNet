using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ReinforcementLearning.ReplayBuffers;

namespace AiDotNet.ReinforcementLearning.Agents.CQL;

/// <summary>
/// Conservative Q-Learning (CQL) agent for offline reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CQL is designed for offline RL, learning from fixed datasets without environment interaction.
/// It prevents overestimation by adding a conservative penalty that pushes down Q-values
/// for out-of-distribution actions while maintaining accuracy on in-distribution actions.
/// </para>
/// <para><b>For Beginners:</b>
/// Unlike online RL (which tries actions and learns), CQL learns only from recorded data.
/// This is crucial for domains where exploration is dangerous or expensive.
///
/// Key features:
/// - **Conservative Penalty**: Lowers Q-values for unseen state-action pairs
/// - **Offline Learning**: No environment interaction needed
/// - **Safe Policy Improvement**: Guarantees improvement over behavior policy
///
/// Example use cases:
/// - Learning from medical records (can't experiment on patients)
/// - Autonomous driving from dashcam data
/// - Robotics from demonstration datasets
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Conservative Q-Learning agent for offline RL from recorded data
/// var options = new CQLOptions&lt;double&gt; { StateSize = 4, ActionSize = 2, ConservativeWeight = 5.0 };
/// var agent = new CQLAgent&lt;double&gt;(options);
///
/// // Select an action given the current state
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
[ResearchPaper("Conservative Q-Learning for Offline Reinforcement Learning",
    "https://arxiv.org/abs/2006.04779",
    Year = 2020,
    Authors = "Kumar, A., Zhou, A., Tucker, G., & Levine, S.")]
public class CQLAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private CQLOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private readonly INumericOperations<T> _numOps;

    private INeuralNetwork<T> _policyNetwork;
    private INeuralNetwork<T> _q1Network;
    private INeuralNetwork<T> _q2Network;
    private INeuralNetwork<T> _targetQ1Network;
    private INeuralNetwork<T> _targetQ2Network;

    private UniformReplayBuffer<T, Vector<T>, Vector<T>> _offlineBuffer;  // Fixed offline dataset
    private Random _random;
    private T _logAlpha;
    private T _alpha;
    private int _updateCount;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public CQLAgent()
        : this(new CQLOptions<T> { StateSize = 4, ActionSize = 2 })
    {
    }

    public CQLAgent(CQLOptions<T> options) : base(CreateBaseOptions(options))
    {
        _options = options;
        _numOps = MathHelper.GetNumericOperations<T>();
        _random = options.Seed.HasValue ? RandomHelper.CreateSeededRandom(options.Seed.Value) : RandomHelper.CreateSecureRandom();
        _updateCount = 0;

        _logAlpha = NumOps.Log(_options.InitialTemperature);
        _alpha = _options.InitialTemperature;

        // Initialize networks directly in constructor
        _policyNetwork = CreatePolicyNetwork();
        _q1Network = CreateQNetwork();
        _q2Network = CreateQNetwork();
        _targetQ1Network = CreateQNetwork();
        _targetQ2Network = CreateQNetwork();

        CopyNetworkWeights(_q1Network, _targetQ1Network);
        CopyNetworkWeights(_q2Network, _targetQ2Network);

        // Initialize offline buffer
        _offlineBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(_options.BufferSize, _options.Seed);
    }

    private static ReinforcementLearningOptions<T> CreateBaseOptions(CQLOptions<T> options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        return new ReinforcementLearningOptions<T>
        {
            LearningRate = options.QLearningRate,
            DiscountFactor = options.DiscountFactor,
            LossFunction = options.QLossFunction,
            Seed = options.Seed,
            BatchSize = options.BatchSize,
            ReplayBufferSize = options.BufferSize
        };
    }

    private NeuralNetwork<T> CreatePolicyNetwork()
    {
        var layers = new List<ILayer<T>>();
        int previousSize = _options.StateSize;

        foreach (var layerSize in _options.PolicyHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(layerSize, (IActivationFunction<T>)new ReLUActivation<T>()));
            previousSize = layerSize;
        }

        // Output: mean and log_std for Gaussian policy
        layers.Add(new DenseLayer<T>(_options.ActionSize * 2, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: _options.ActionSize * 2,
            layers: layers
        );

        return new NeuralNetwork<T>(architecture, null);
    }

    private NeuralNetwork<T> CreateQNetwork()
    {
        var layers = new List<ILayer<T>>();
        int inputSize = _options.StateSize + _options.ActionSize;
        int previousSize = inputSize;

        foreach (var layerSize in _options.QHiddenLayers)
        {
            layers.Add(new DenseLayer<T>(layerSize, (IActivationFunction<T>)new ReLUActivation<T>()));
            previousSize = layerSize;
        }

        layers.Add(new DenseLayer<T>(1, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: inputSize,
            outputSize: 1,
            layers: layers
        );

        return new NeuralNetwork<T>(architecture, lossFunction: _options.QLossFunction);
    }

    private void InitializeBuffer()
    {
        _offlineBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(_options.BufferSize);
    }

    /// <summary>
    /// Load offline dataset into the replay buffer.
    /// </summary>
    public void LoadOfflineData(List<(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)> dataset)
    {
        foreach (var transition in dataset)
        {
            var experience = new Experience<T, Vector<T>, Vector<T>>(
                transition.state,
                transition.action,
                transition.reward,
                transition.nextState,
                transition.done);
            _offlineBuffer.Add(experience);
        }
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var policyOutputTensor = _policyNetwork.Predict(stateTensor);
        var policyOutput = policyOutputTensor.ToVector();

        // Extract mean and log_std
        var mean = new Vector<T>(_options.ActionSize);
        var logStd = new Vector<T>(_options.ActionSize);

        for (int i = 0; i < _options.ActionSize; i++)
        {
            mean[i] = policyOutput[i];
            logStd[i] = policyOutput[_options.ActionSize + i];

            // Clamp log_std for numerical stability
            logStd[i] = MathHelper.Clamp<T>(logStd[i], _numOps.FromDouble(-20), _numOps.FromDouble(2));
        }

        if (!training)
        {
            // Return mean action during evaluation
            for (int i = 0; i < mean.Length; i++)
            {
                mean[i] = MathHelper.Tanh<T>(mean[i]);
            }
            return mean;
        }

        // Sample action from Gaussian policy
        var action = new Vector<T>(_options.ActionSize);
        for (int i = 0; i < _options.ActionSize; i++)
        {
            var std = NumOps.Exp(logStd[i]);
            var noise = GetSeededNormalRandom(_numOps.Zero, _numOps.One, _random);
            var rawAction = _numOps.Add(mean[i], _numOps.Multiply(std, noise));
            action[i] = MathHelper.Tanh<T>(rawAction);
        }

        return action;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        // CQL is offline - data is loaded beforehand
        // This method is kept for interface compliance but not used in offline setting
        var experience = new Experience<T, Vector<T>, Vector<T>>(state, action, reward, nextState, done);
        _offlineBuffer.Add(experience);
    }

    public override T Train()
    {
        if (_offlineBuffer.Count < _options.BatchSize)
        {
            return _numOps.Zero;
        }

        var batch = _offlineBuffer.Sample(_options.BatchSize);
        int n = batch.Count;
        if (n == 0) return _numOps.Zero;

        int stateDim = _options.StateSize;
        int actionDim = _options.ActionSize;
        int saDim = stateDim + actionDim;
        T gamma = _options.DiscountFactor;

        // --- Twin-Q Bellman regression (Kumar et al. 2020, eq. for the standard TD term) ---
        // Build the data state-action inputs and the clipped double-Q TD targets:
        //   y = r + gamma * (1 - done) * min(Q'_1(s', a'), Q'_2(s', a')),  a' ~ pi(s').
        var dataSA = new Tensor<T>([n, saDim]);
        var tdTargets = new Tensor<T>([n, 1]);
        var randSA = new Tensor<T>([n, saDim]);
        var consTargets = new Tensor<T>([n, 1]);
        var policyInputs = new Tensor<T>([n, stateDim]);
        var policyTargets = new Tensor<T>([n, actionDim * 2]);

        for (int i = 0; i < n; i++)
        {
            var exp = batch[i];
            var stateTensor = Tensor<T>.FromVector(exp.State);

            // Next action from the (current) policy — deterministic mean for a stable target.
            var nextAction = SelectAction(exp.NextState, training: false);
            var nextSA = ConcatenateStateAction(exp.NextState, nextAction);
            var nextSATensor = Tensor<T>.FromVector(nextSA);
            T q1Next = _targetQ1Network.Predict(nextSATensor).ToVector()[0];
            T q2Next = _targetQ2Network.Predict(nextSATensor).ToVector()[0];
            T minNext = _numOps.LessThan(q1Next, q2Next) ? q1Next : q2Next;

            T y = exp.Reward;
            if (!exp.Done)
            {
                y = _numOps.Add(y, _numOps.Multiply(gamma, minNext));
            }

            var dataAction = ConcatenateStateAction(exp.State, exp.Action);
            for (int j = 0; j < saDim; j++) dataSA[i, j] = dataAction[j];
            tdTargets[i, 0] = y;

            // --- CQL conservative penalty: push Q DOWN on out-of-distribution (random) actions ---
            // so the agent does not over-estimate the value of actions absent from the dataset.
            // Realised as a regression of Q(s, a_random) toward (current value − CQLAlpha).
            var randomAction = new Vector<T>(actionDim);
            for (int k = 0; k < actionDim; k++)
                randomAction[k] = _numOps.FromDouble(_random.NextDouble() * 2.0 - 1.0);
            var randomSAVec = ConcatenateStateAction(exp.State, randomAction);
            var randomSATensor = Tensor<T>.FromVector(randomSAVec);
            T qRand = _q1Network.Predict(randomSATensor).ToVector()[0];
            for (int j = 0; j < saDim; j++) randSA[i, j] = randomSAVec[j];
            consTargets[i, 0] = _numOps.Subtract(qRand, _options.CQLAlpha);

            // --- Policy improvement (CQL is built on SAC; Kumar et al. 2020 §3.2): the actor
            // MAXIMISES the (conservative) Q. We take the squashed policy mean as the current
            // action, estimate ∇a min(Q1,Q2) by central finite differences (the deterministic
            // policy gradient), and regress the policy mean toward the Q-ascending action target
            // a + step·∇a Q. log_std targets 0. The conservative critic keeps this maximisation
            // from exploiting over-valued out-of-distribution actions. ---
            var polOut = _policyNetwork.Predict(stateTensor).ToVector();
            var aCur = new Vector<T>(actionDim);
            for (int k = 0; k < actionDim; k++) aCur[k] = MathHelper.Tanh<T>(polOut[k]);
            var qGrad = FiniteDiffMinQActionGradient(exp.State, aCur);
            T polStep = _numOps.FromDouble(0.05);
            for (int j = 0; j < stateDim; j++) policyInputs[i, j] = exp.State[j];
            for (int k = 0; k < actionDim; k++)
            {
                policyTargets[i, k] = MathHelper.Clamp<T>(
                    _numOps.Add(aCur[k], _numOps.Multiply(polStep, qGrad[k])),
                    _numOps.FromDouble(-1), _numOps.FromDouble(1));   // mean -> Q-ascending action
                policyTargets[i, actionDim + k] = _numOps.Zero;       // log_std -> 0
            }
        }

        // Apply the gradient updates (each Train() is a tape-based forward/backward/step).
        _q1Network.Train(dataSA, tdTargets);
        _q2Network.Train(dataSA, tdTargets);
        _q1Network.Train(randSA, consTargets);
        _q2Network.Train(randSA, consTargets);
        _policyNetwork.Train(policyInputs, policyTargets);

        T totalLoss = _numOps.Add(
            _numOps.Add(_q1Network.GetLastLoss(), _q2Network.GetLastLoss()),
            _policyNetwork.GetLastLoss());

        // Update temperature
        if (_options.AutoTuneTemperature)
        {
            UpdateTemperature(batch);
        }

        // Soft update target networks
        SoftUpdateTargetNetworks();

        _updateCount++;

        return _numOps.Divide(totalLoss, _numOps.FromDouble(3));
    }

    /// <summary>
    /// Central finite-difference estimate of ∇a min(Q1(s,a), Q2(s,a)) — the action gradient the
    /// SAC-based CQL actor ascends (the deterministic policy gradient), used because the critics
    /// expose no analytic gradient w.r.t. their action input.
    /// </summary>
    private Vector<T> FiniteDiffMinQActionGradient(Vector<T> state, Vector<T> action)
    {
        var grad = new Vector<T>(action.Length);
        T eps = _numOps.FromDouble(1e-3);
        T twoEps = _numOps.FromDouble(2e-3);
        for (int i = 0; i < action.Length; i++)
        {
            var aPlus = action.Clone();
            var aMinus = action.Clone();
            aPlus[i] = _numOps.Add(action[i], eps);
            aMinus[i] = _numOps.Subtract(action[i], eps);
            var saPlus = Tensor<T>.FromVector(ConcatenateStateAction(state, aPlus));
            var saMinus = Tensor<T>.FromVector(ConcatenateStateAction(state, aMinus));
            T qPlus1 = _q1Network.Predict(saPlus).ToVector()[0];
            T qPlus2 = _q2Network.Predict(saPlus).ToVector()[0];
            T qMinus1 = _q1Network.Predict(saMinus).ToVector()[0];
            T qMinus2 = _q2Network.Predict(saMinus).ToVector()[0];
            T qPlus = _numOps.LessThan(qPlus1, qPlus2) ? qPlus1 : qPlus2;
            T qMinus = _numOps.LessThan(qMinus1, qMinus2) ? qMinus1 : qMinus2;
            grad[i] = _numOps.Divide(_numOps.Subtract(qPlus, qMinus), twoEps);
        }
        return grad;
    }

    private T ComputeCQLPenalty(Vector<T> state, Vector<T> dataAction, T q1Value, T q2Value)
    {
        // CQL penalty: E[Q(s, a_random)] - Q(s, a_data)
        // This pushes down Q-values for random actions while keeping data actions accurate

        T randomQSum = _numOps.Zero;
        int numSamples = _options.CQLNumActions;

        for (int i = 0; i < numSamples; i++)
        {
            // Sample random action
            var randomAction = new Vector<T>(_options.ActionSize);
            for (int j = 0; j < _options.ActionSize; j++)
            {
                randomAction[j] = _numOps.FromDouble(_random.NextDouble() * 2 - 1);  // [-1, 1]
            }

            var stateAction = ConcatenateStateAction(state, randomAction);
            var stateActionTensor = Tensor<T>.FromVector(stateAction);
            var q1RandomTensor = _q1Network.Predict(stateActionTensor);
            var q2RandomTensor = _q2Network.Predict(stateActionTensor);
            var q1Random = q1RandomTensor.ToVector()[0];
            var q2Random = q2RandomTensor.ToVector()[0];

            var avgQRandom = _numOps.Divide(_numOps.Add(q1Random, q2Random), _numOps.FromDouble(2));
            randomQSum = _numOps.Add(randomQSum, avgQRandom);
        }

        var avgRandomQ = _numOps.Divide(randomQSum, _numOps.FromDouble(numSamples));
        var avgDataQ = _numOps.Divide(_numOps.Add(q1Value, q2Value), _numOps.FromDouble(2));

        // Penalty = alpha * (E[Q(s, a_random)] - Q(s, a_data))
        var gap = _numOps.Subtract(avgRandomQ, avgDataQ);
        return _numOps.Multiply(_options.CQLAlpha, gap);
    }

    private void UpdateTemperature(List<Experience<T, Vector<T>, Vector<T>>> batch)
    {
        // Temperature update using entropy target
        // Loss: alpha * (entropy - target_entropy)
        // Gradient: d_loss/d_log_alpha = alpha * (entropy - target_entropy)

        T avgEntropy = _numOps.Zero;
        foreach (var experience in batch)
        {
            var policyOutputTensor = _policyNetwork.Predict(Tensor<T>.FromVector(experience.State));
            var policyOutput = policyOutputTensor.ToVector();

            T entropy = _numOps.Zero;
            for (int tempIdx = 0; tempIdx < _options.ActionSize; tempIdx++)
            {
                var logStd = policyOutput[_options.ActionSize + tempIdx];
                logStd = MathHelper.Clamp<T>(logStd, _numOps.FromDouble(-20), _numOps.FromDouble(2));
                var gaussianConst = _numOps.FromDouble(0.5 * (1.0 + System.Math.Log(2.0 * System.Math.PI)));
                entropy = _numOps.Add(entropy, _numOps.Add(gaussianConst, logStd));
            }
            avgEntropy = _numOps.Add(avgEntropy, entropy);
        }
        avgEntropy = _numOps.Divide(avgEntropy, _numOps.FromDouble(batch.Count));

        // Target entropy: -dim(action_space)
        var targetEntropy = _numOps.FromDouble(-_options.ActionSize);
        var entropyGap = _numOps.Subtract(avgEntropy, targetEntropy);

        // Update log_alpha: log_alpha -= lr * alpha * entropy_gap
        var alphaLr = _numOps.FromDouble(0.0003);
        var alphaGrad = _numOps.Multiply(_alpha, entropyGap);
        var alphaUpdate = _numOps.Multiply(alphaLr, alphaGrad);
        _logAlpha = _numOps.Subtract(_logAlpha, alphaUpdate);

        // Update alpha from log_alpha
        _alpha = NumOps.Exp(_logAlpha);
    }

    private void SoftUpdateTargetNetworks()
    {
        SoftUpdateNetwork(_q1Network, _targetQ1Network);
        SoftUpdateNetwork(_q2Network, _targetQ2Network);
    }

    private void SoftUpdateNetwork(INeuralNetwork<T> source, INeuralNetwork<T> target)
    {
        var sourceParams = source.GetParameters();
        var targetParams = target.GetParameters();
        var oneMinusTau = _numOps.Subtract(_numOps.One, _options.TargetUpdateTau);

        var updatedParams = new Vector<T>(targetParams.Length);
        for (int softUpdateIdx = 0; softUpdateIdx < targetParams.Length; softUpdateIdx++)
        {
            var sourceContrib = _numOps.Multiply(_options.TargetUpdateTau, sourceParams[softUpdateIdx]);
            var targetContrib = _numOps.Multiply(oneMinusTau, targetParams[softUpdateIdx]);
            updatedParams[softUpdateIdx] = _numOps.Add(sourceContrib, targetContrib);
        }

        target.UpdateParameters(updatedParams);
    }

    private void CopyNetworkWeights(INeuralNetwork<T> source, INeuralNetwork<T> target)
    {
        var sourceParams = source.GetParameters();
        target.UpdateParameters(sourceParams);
    }

    private Vector<T> ConcatenateStateAction(Vector<T> state, Vector<T> action)
    {
        var result = new Vector<T>(state.Length + action.Length);
        for (int i = 0; i < state.Length; i++)
        {
            result[i] = state[i];
        }
        for (int i = 0; i < action.Length; i++)
        {
            result[state.Length + i] = action[i];
        }
        return result;
    }

    private T GetSeededNormalRandom(T mean, T stdDev, Random random)
    {
        double result = random.NextGaussian() * Convert.ToDouble(stdDev) + Convert.ToDouble(mean);
        return _numOps.FromDouble(result);
    }

    public override Dictionary<string, T> GetMetrics()
    {
        return new Dictionary<string, T>
        {
            ["updates"] = _numOps.FromDouble(_updateCount),
            ["buffer_size"] = _numOps.FromDouble(_offlineBuffer.Count),
            ["alpha"] = _alpha
        };
    }

    public override void ResetEpisode()
    {
        // CQL is offline - no episode reset needed
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

    /// <inheritdoc/>
    public override int FeatureCount => _options.StateSize;

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            FeatureCount = _options.StateSize,
            Complexity = ParameterCount,
        };
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        // Combine parameters from policy network and both Q-networks
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
    }

    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new CQLAgent<T>(_options);
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc/>
    public override Vector<T> ComputeGradients(
        Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // CQL uses custom gradient computation - return zero gradients as placeholder
        var parameters = GetParameters();
        var gradients = new Vector<T>(parameters.Length);
        for (int i = 0; i < gradients.Length; i++)
        {
            gradients[i] = _numOps.Zero;
        }
        return gradients;
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // CQL uses direct network updates - not directly applicable
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        writer.Write(_options.StateSize);
        writer.Write(_options.ActionSize);
        writer.Write(_updateCount);
        writer.Write(Convert.ToDouble(_alpha));

        var policyBytes = _policyNetwork.Serialize();
        writer.Write(policyBytes.Length);
        writer.Write(policyBytes);

        var q1Bytes = _q1Network.Serialize();
        writer.Write(q1Bytes.Length);
        writer.Write(q1Bytes);

        var q2Bytes = _q2Network.Serialize();
        writer.Write(q2Bytes.Length);
        writer.Write(q2Bytes);

        return ms.ToArray();
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        reader.ReadInt32(); // stateSize
        reader.ReadInt32(); // actionSize
        _updateCount = reader.ReadInt32();
        _alpha = _numOps.FromDouble(reader.ReadDouble());

        var policyLength = reader.ReadInt32();
        var policyBytes = reader.ReadBytes(policyLength);
        _policyNetwork.Deserialize(policyBytes);

        var q1Length = reader.ReadInt32();
        var q1Bytes = reader.ReadBytes(q1Length);
        _q1Network.Deserialize(q1Bytes);

        var q2Length = reader.ReadInt32();
        var q2Bytes = reader.ReadBytes(q2Length);
        _q2Network.Deserialize(q2Bytes);
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
