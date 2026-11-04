using AiDotNet.LearningRateSchedulers;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Agents.DecisionTransformer;

/// <summary>
/// Decision Transformer agent for offline reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Decision Transformer treats RL as sequence modeling, using transformer architecture
/// to predict actions conditioned on desired returns-to-go.
/// </para>
/// <para>
/// The network (section 3) embeds the last K timesteps as 3K tokens, (return-to-go, state, action) per timestep, with a
/// linear layer per modality plus a learned per-timestep embedding, normalizes them, and runs a causally masked GPT
/// (3 layers, 1 head, 128 dimensions, dropout 0.1 by default, the Gym model of appendix A.2); the hidden state of each
/// state token predicts that timestep's action. Training regresses the dataset's actions (mean-squared error, or
/// cross-entropy for discrete actions) on K-step segments; acting conditions on a target return that each reward
/// decrements.
/// </para>
/// <para><b>For Beginners:</b>
/// Instead of learning "what's the best action", Decision Transformer learns
/// "what action was taken when the outcome was X". At test time, you specify
/// the desired outcome, and it generates the action sequence.
///
/// Key innovation:
/// - **Return Conditioning**: Specify target return, get actions that achieve it
/// - **Sequence Modeling**: Uses transformers like GPT for temporal dependencies
/// - **No RL Updates**: Just supervised learning on (return, state, action) sequences
/// - **Offline-First**: Designed for learning from fixed datasets
///
/// Think of it as: "Show me examples of successful games, and I'll learn to
/// generate moves that lead to that level of success."
///
/// Famous for: Berkeley/Meta research simplifying RL to sequence modeling
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Decision Transformer for offline RL via sequence modeling
/// var options = new DecisionTransformerOptions&lt;double&gt; { StateSize = 4, ActionSize = 2, ContextLength = 20 };
/// var agent = new DecisionTransformerAgent&lt;double&gt;(options);
///
/// // Select an action conditioned on the desired return
/// var state = new Vector&lt;double&gt;(new double[] { 0.5, -0.3, 1.0, 0.2 });
/// var action = agent.SelectAction(state);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Decision Transformer: Reinforcement Learning via Sequence Modeling",
    "https://arxiv.org/abs/2106.01345",
    Year = 2021,
    Authors = "Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel, P., Srinivas, A., & Mordatch, I.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, WeightDecay = 1e-4,
                ReferenceBatchSize = 64, WarmupSteps = 100000,
                Schedule = LearningRateSchedulerType.LinearWarmup,
                Source = "Chen et al. 2021, Appendix A: AdamW following PyTorch defaults, trained for "
                        + "10^5 gradient steps with a learning rate of 1e-4, a weight decay of 1e-4, a "
                        + "batch size of 64 and linear warmup over the first 10^5 training steps. The "
                        + "exponents render as a bare 105 in extraction. Table 9: grad norm clip 0.25.",
                MaxGradientNorm = 0.25)]
public partial class DecisionTransformerAgent<T> : DeepReinforcementLearningAgentBase<T>, IGradientComputable<T, Vector<T>, Vector<T>>
{
    /// <inheritdoc />
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_transformerNetwork);
    }
    private DecisionTransformerOptions<T> _options;

    private IOptimizer<T, Vector<T>, Vector<T>> _optimizer;
    // AdamW with the paper's recipe (appendix A), on the transformer's tensors.
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _networkOptimizer;

    private DecisionTransformerNetwork<T> _transformerNetwork;
    // The offline dataset, trajectory after trajectory: each step's state, action, reward, return-to-go and timestep
    // (0 starts a trajectory).
    private List<(Vector<T> state, Vector<T> action, T reward, T returnToGo, int timestep)> _trajectoryBuffer;
    // The episode StoreExperience is collecting; it joins the dataset when it ends.
    private List<(Vector<T> state, Vector<T> action, T reward)> _episode;
    private int _updateCount;

    // Evaluation (Algorithm 1): the last K returns-to-go, states and actions, the episode's timestep, and the
    // return-to-go still to collect.
    private SequenceContext<T> _currentContext;
    private int _contextTimestep;
    private T _runningReturn;

    /// <summary>The transformer.</summary>
    internal DecisionTransformerNetwork<T> Network => _transformerNetwork;

    public DecisionTransformerAgent() : this(new DecisionTransformerOptions<T>()) { }

    public DecisionTransformerAgent(DecisionTransformerOptions<T> options, IOptimizer<T, Vector<T>, Vector<T>>? optimizer = null)
        : base(options)
    {
        Guard.NotNull(options);
        ValidateOptions(options);
        _options = options;
        Options = _options;
        _optimizer = optimizer ?? options.Optimizer ?? new AdamOptimizer<T, Vector<T>, Vector<T>>(this, new AdamOptimizerOptions<T, Vector<T>, Vector<T>>
        {
            InitialLearningRate = NumOps.ToDouble(LearningRate),
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        });
        _updateCount = 0;
        _trajectoryBuffer = new List<(Vector<T>, Vector<T>, T, T, int)>();
        _episode = new List<(Vector<T>, Vector<T>, T)>();
        _currentContext = new SequenceContext<T>();
        RecomputeStateStatistics();
        _runningReturn = InitialReturnToGo();

        _transformerNetwork = new DecisionTransformerNetwork<T>(_options.StateSize, _options.ActionSize, _options.EmbeddingDim,
            _options.NumLayers, _options.NumHeads, _options.ContextLength, _options.MaxEpisodeLength, _options.DropoutRate,
            _options.IsContinuous, NextNetworkInitializationSeed());
        _networkOptimizer = _options.OptimizerFactory?.Invoke() ?? CreatePaperOptimizer();

        Networks.Add(_transformerNetwork);
    }

    /// <summary>
    /// Loads offline trajectories into the dataset, computing each step's return-to-go (the sum of the rewards from
    /// that step to the end) and timestep.
    /// </summary>
    public void LoadOfflineData(List<List<(Vector<T> state, Vector<T> action, T reward)>> trajectories)
    {
        foreach (var trajectory in trajectories) AddTrajectory(trajectory);
        if (_currentContext.Length == 0) ResetEpisode();
    }

    private void AddTrajectory(List<(Vector<T> state, Vector<T> action, T reward)> trajectory)
    {
        var returnsToGo = new T[trajectory.Count];
        T returnToGo = NumOps.Zero;
        for (int i = trajectory.Count - 1; i >= 0; i--)
        {
            returnToGo = NumOps.Add(trajectory[i].reward, returnToGo);
            returnsToGo[i] = returnToGo;
        }
        for (int i = 0; i < trajectory.Count; i++)
            _trajectoryBuffer.Add((trajectory[i].state, trajectory[i].action, trajectory[i].reward, returnsToGo[i], i));
        if (trajectory.Count > 0 && (!_hasEpisodeReturn || NumOps.GreaterThan(returnToGo, _bestEpisodeReturn)))
        {
            _bestEpisodeReturn = returnToGo;
            _hasEpisodeReturn = true;
        }
        TrimCompletedTrajectories();
    }

    /// <summary>
    /// Selects an action for the episode's next state, conditioned on the return-to-go still to collect: the
    /// configured <see cref="DecisionTransformerOptions{T}.TargetReturn"/> at the start of an episode, decremented by
    /// each reward passed to <see cref="StoreExperience"/>.
    /// </summary>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        return SelectActionWithReturn(state, _runningReturn, training);
    }

    /// <summary>
    /// Select action conditioned on desired return-to-go (Algorithm 1's evaluation loop): appends (R, s, t) to the
    /// context, keeps its last K timesteps, and returns the action predicted at the newest state token.
    /// </summary>
    public Vector<T> SelectActionWithReturn(Vector<T> state, T targetReturn, bool training = true)
    {
        _currentContext.States.Add(state);
        _currentContext.ReturnsToGo.Add(targetReturn);
        _runningReturn = targetReturn;
        _contextTimestep++;
        while (_currentContext.Length > _options.ContextLength)
        {
            _currentContext.States.RemoveAt(0);
            _currentContext.ReturnsToGo.RemoveAt(0);
            if (_currentContext.Actions.Count > 0) _currentContext.Actions.RemoveAt(0);
        }

        // The newest action is not known yet: its slot is zero, and causal masking keeps it from the state token's
        // prediction anyway.
        int k = _currentContext.Length, firstTimestep = _contextTimestep - k;
        var input = new Tensor<T>(new[] { 1, k, _transformerNetwork.StepWidth });
        for (int i = 0; i < k; i++)
        {
            var action = i < _currentContext.Actions.Count ? _currentContext.Actions[i] : null;
            Pack(input, 0, i, _currentContext.ReturnsToGo[i], _currentContext.States[i], action, firstTimestep + i);
        }
        _transformerNetwork.SetTrainingMode(false);
        var output = _transformerNetwork.Predict(input);
        var predicted = new Vector<T>(_options.ActionSize);
        for (int j = 0; j < _options.ActionSize; j++) predicted[j] = output[(k - 1) * _options.ActionSize + j];
        var chosen = _options.IsContinuous ? predicted : OneHot(ArgMax(predicted));
        _currentContext.Actions.Add(chosen);
        return chosen;
    }

    private Vector<T> OneHot(int index)
    {
        var v = new Vector<T>(_options.ActionSize);
        v[index] = NumOps.One;
        return v;
    }

    private static int ArgMax(Vector<T> v)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (ops.GreaterThan(v[i], v[best])) best = i;
        return best;
    }

    // One packed timestep [R / scale, s, a, t]; a null action is zero.
    private void Pack(Tensor<T> input, int b, int position, T returnToGo, Vector<T> state, Vector<T>? action, int timestep)
    {
        int w = _transformerNetwork.StepWidth;
        var span = input;
        int offset = (b * input.Shape[1] + position) * w;
        span[offset] = NumOps.FromDouble(NumOps.ToDouble(returnToGo) / _options.ReturnScale);
        for (int j = 0; j < _options.StateSize; j++) span[offset + 1 + j] = _options.NormalizeStates
            ? NumOps.Divide(NumOps.Subtract(state[j], _stateMean[j]), _stateStd[j]) : state[j];
        if (action is not null) for (int j = 0; j < _options.ActionSize; j++) span[offset + 1 + _options.StateSize + j] = action[j];
        span[offset + w - 1] = NumOps.FromDouble(Math.Min(Math.Max(timestep, 0), _options.MaxEpisodeLength - 1));
    }

    /// <summary>
    /// Collects the episode; when it ends it joins the offline dataset (with its returns-to-go). The reward also
    /// decrements the return-to-go the next <see cref="SelectAction"/> conditions on (Algorithm 1).
    /// </summary>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        _episode.Add((state, action, reward));
        _runningReturn = NumOps.Subtract(_runningReturn, reward);
        if (done)
        {
            AddTrajectory(_episode);
            _episode = new List<(Vector<T>, Vector<T>, T)>();
            ResetEpisode();
        }
    }

    /// <summary>
    /// One gradient step (section 3, "Training"): a minibatch of K-step segments, each predicting its actions from
    /// the state tokens, with the mean-squared error (continuous) or cross-entropy (discrete) averaged over the
    /// timesteps. Segments shorter than K (at a trajectory's end) are padded on the left and the padding is left out of
    /// the loss.
    /// </summary>
    public override T Train()
    {
        if (_trajectoryBuffer.Count == 0)
        {
            return NumOps.Zero;
        }

        int batch = Math.Max(1, _options.BatchSize), k = 1, a = _options.ActionSize;
        var starts = new int[batch];
        var lengths = new int[batch];
        for (int b = 0; b < batch; b++)
        {
            starts[b] = Random.Next(_trajectoryBuffer.Count);
            int length = 1;
            while (length < _options.ContextLength && starts[b] + length < _trajectoryBuffer.Count
                && _trajectoryBuffer[starts[b] + length].timestep != 0) length++;
            lengths[b] = length;
            k = Math.Max(k, length);
        }
        var input = new Tensor<T>(new[] { batch, k, _transformerNetwork.StepWidth });
        var targets = new Tensor<T>(new[] { batch, k, a });
        var mask = new Tensor<T>(new[] { batch, k, a });
        int counted = 0;
        for (int b = 0; b < batch; b++)
        {
            // A uniformly drawn step starts the segment: the authors' trajectory-by-length then uniform-start sampling.
            int start = starts[b], length = lengths[b];
            int pad = k - length;
            for (int i = 0; i < length; i++)
            {
                var step = _trajectoryBuffer[start + i];
                Pack(input, b, pad + i, step.returnToGo, step.state, step.action, step.timestep);
                for (int j = 0; j < a; j++)
                {
                    targets[b, pad + i, j] = step.action[j];
                    mask[b, pad + i, j] = NumOps.One;
                }
                counted++;
            }
        }

        var scale = NumOps.FromDouble(1.0 / (counted * (_options.IsContinuous ? a : 1)));
        _transformerNetwork.SetTrainingMode(true);
        T loss = _transformerNetwork.TrainWithCustomLoss(input, output =>
        {
            var allAxes = new[] { 0, 1, 2 };
            if (_options.IsContinuous)
            {
                var diff = Engine.TensorMultiply(Engine.TensorSubtract(output, targets), mask);
                return Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorMultiply(diff, diff), allAxes, keepDims: false), scale);
            }
            var logProbabilities = Engine.TensorLogSoftmax(output, 2);
            var picked = Engine.TensorMultiply(Engine.TensorMultiply(logProbabilities, targets), mask);
            return Engine.TensorNegate(Engine.TensorMultiplyScalar(Engine.ReduceSum(picked, allAxes, keepDims: false), scale));
        }, _networkOptimizer);
        _transformerNetwork.SetTrainingMode(false);

        _updateCount++;
        return loss;
    }

    /// <summary>
    /// Supervised (online) training entry from <see cref="IFullModel{T,TInput,TOutput}"/>: one step predicting
    /// <paramref name="target"/> (the action) from a one-timestep sequence of <paramref name="state"/> at return-to-go 0
    /// and timestep 0. Decision Transformer is return-conditioned behavior cloning, so this is its objective on a
    /// single pair.
    /// </summary>
    public override void Train(Vector<T> state, Vector<T> target)
    {
        if (state is null) throw new ArgumentNullException(nameof(state));
        if (target is null) throw new ArgumentNullException(nameof(target));

        var input = SingleStep(state, training: true);
        var targetAction = new Tensor<T>(new[] { 1, 1, _options.ActionSize });
        int copyLen = Math.Min(target.Length, _options.ActionSize);
        for (int i = 0; i < copyLen; i++) targetAction[0, 0, i] = target[i];

        _transformerNetwork.SetTrainingMode(true);
        _transformerNetwork.TrainWithCustomLoss(input, output =>
        {
            var diff = Engine.TensorSubtract(output, targetAction);
            return Engine.ReduceMean(Engine.TensorMultiply(diff, diff), new[] { 0, 1, 2 }, keepDims: false);
        }, _networkOptimizer);
        _transformerNetwork.SetTrainingMode(false);
        _updateCount++;
    }

    // A one-timestep sequence: return-to-go 0, the state, no action, timestep 0.
    private Tensor<T> SingleStep(Vector<T> state, bool training = false)
    {
        var input = new Tensor<T>(new[] { 1, 1, _transformerNetwork.StepWidth });
        Pack(input, 0, 0, training ? NumOps.Zero : InitialReturnToGo(), state, null, 0);
        return input;
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
        _currentContext = new SequenceContext<T>();
        _contextTimestep = 0;
        _runningReturn = InitialReturnToGo();
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        // Pure, stateless single-step inference (the IFullModel.Predict contract): two Predict(state) calls must
        // return the same action, so this runs a one-timestep sequence without touching the episode's context.
        if (input is null) throw new ArgumentNullException(nameof(input));
        _transformerNetwork.SetTrainingMode(false);
        var output = _transformerNetwork.Predict(SingleStep(input));
        var action = new Vector<T>(_options.ActionSize);
        for (int j = 0; j < _options.ActionSize; j++) action[j] = output[j];
        return _options.IsContinuous ? action : OneHot(ArgMax(action));
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
        };
    }

    public override int FeatureCount => _options.StateSize;

    /// <summary>
    /// Computes gradients of the loss with respect to this agent's parameters, without updating them.
    /// </summary>
    /// <param name="input">The input state.</param>
    /// <param name="target">The target output.</param>
    /// <param name="lossFunction">The loss to differentiate, or null to use the agent's own.</param>
    /// <returns>A gradient vector the same length as <c>GetParameters()</c>.</returns>
    /// <remarks>
    /// The state runs as a one-timestep sequence (as in <see cref="Predict"/>) and the gradient comes from the network's
    /// own tape pass.
    /// </remarks>
    public Vector<T> ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (target is null) throw new ArgumentNullException(nameof(target));
        var targetAction = new Tensor<T>(new[] { 1, 1, _options.ActionSize });
        for (int i = 0; i < Math.Min(target.Length, _options.ActionSize); i++) targetAction[0, 0, i] = target[i];
        return _transformerNetwork.ComputeGradients(SingleStep(input, training: true), targetAction, lossFunction ?? LossFunction);
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
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer()
    {
        double rate = NumOps.ToDouble(LearningRate);
        if (double.IsNaN(rate) || double.IsInfinity(rate) || rate <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(LearningRate), rate, "LearningRate must be positive and finite.");

        double clip = _options.MaxGradientNorm;
        bool clipping = clip > 0.0 && !double.IsPositiveInfinity(clip);
        var optimizerOptions = new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = rate,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8,
            WeightDecay = _options.WeightDecay,
            UseAMSGrad = false,
            UseAdaptiveLearningRate = false,
            EnableGradientClipping = clipping,
            GradientClippingMethod = GradientClippingMethod.ByNorm,
        };
        if (clipping) optimizerOptions.MaxGradientNorm = clip;

        int warmup = _options.WarmupSteps;
        if (warmup > 1)
        {
            optimizerOptions.LearningRateScheduler =
                new LinearWarmupScheduler(rate, warmup - 1, warmupInitLr: rate / warmup);
            optimizerOptions.SchedulerStepMode = SchedulerStepMode.StepPerBatch;
        }

        return new AiDotNet.Optimizers.AdamWOptimizer<T, Tensor<T>, Tensor<T>>(null, optimizerOptions);
    }
}
