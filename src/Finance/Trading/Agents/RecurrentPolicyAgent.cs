using System;
using System.Collections.Generic;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Finance.Trading.Evaluation;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;

namespace AiDotNet.Finance.Trading.Agents;

/// <summary>
/// A RECURRENT continuous policy for the portfolio-manager harness: an LSTM carries hidden state across the
/// episode, so the policy conditions each action on the whole history it has seen — not just the fixed
/// observation window an MLP policy is limited to. This is the recurrent-policy path for the greenfield
/// trading-agent research: partial observability (regime, position history) is exactly where a recurrent policy
/// should beat a memoryless one, and the experiment runner ranks that on the untouched holdout.
/// <para>
/// Trained on-policy with REINFORCE + a return baseline, back-propagating through time over each episode's
/// rollout (the natural fit for recurrence — on-policy methods learn from full sequences, no sequence-replay
/// surgery). Exploration uses a fixed Gaussian σ, so the policy-gradient objective reduces to
/// <c>Σ advantageₜ · ½‖actionₜ − μₜ‖² / σ²</c> — a numerically robust loss (no exp/log of parameters). Reuses
/// the tape-trainable LSTM cell + Adam path proven by the DeepAR model. Implements the narrow
/// <see cref="IPortfolioAgent{T}"/> so it drops straight into <see cref="PortfolioExperimentRunner"/>.
/// </para>
/// </summary>
/// <typeparam name="T">Element type (float/double).</typeparam>
public sealed partial class RecurrentPolicyAgent<T> : IPortfolioAgent<T>, IDisposable
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static IEngine Engine => AiDotNetEngine.Current;

    private readonly int _stateDim;
    private readonly int _actionDim;
    private readonly int _hidden;
    private readonly double _sigma;
    private readonly double _gamma;
    private readonly Random _rng;

    private readonly DeepARLstmCellTape<T> _cell;   // recurrent core (tape-trainable)
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Tensor<T> _meanW;              // [actionDim, hidden]
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Tensor<T> _meanB;              // [actionDim, 1]
    private readonly IReadOnlyList<Tensor<T>> _trainable;
    private readonly AdamOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    // One episode's update on the shared fused session: the compiled plan when it applies, the eager tape otherwise.
    private readonly Training.TapeTrainingStepper<T> _stepper;

    /// <summary>Episode updates run on the fused compiled plan.</summary>
    internal int FusedTrainingStepCount => _stepper.FusedStepCount;

    /// <summary>Episode updates run on the eager tape.</summary>
    internal int EagerTrainingStepCount => _stepper.EagerStepCount;

    // Per-episode recurrent state (eager act path).
    [AiDotNet.Attributes.Scratch]
    private Tensor<T> _h;
    [AiDotNet.Attributes.Scratch]
    private Tensor<T> _c;

    // Current-episode rollout.
    private readonly List<T[]> _states = new();
    private readonly List<T[]> _actions = new();
    private readonly List<double> _rewards = new();
    private bool _episodeDone;
    private bool _disposed;

    public RecurrentPolicyAgent(
        int stateDim, int actionDim, int hidden = 32, double explorationSigma = 0.2,
        double learningRate = 1e-3, double gamma = 0.99, int seed = 0)
    {
        if (stateDim <= 0) throw new ArgumentOutOfRangeException(nameof(stateDim));
        if (actionDim <= 0) throw new ArgumentOutOfRangeException(nameof(actionDim));
        if (hidden <= 0) throw new ArgumentOutOfRangeException(nameof(hidden));

        _stateDim = stateDim;
        _actionDim = actionDim;
        _hidden = hidden;
        _sigma = explorationSigma > 1e-6 ? explorationSigma : 1e-6;
        _gamma = gamma;
        _rng = RandomHelper.CreateSeededRandom(seed);

        _cell = new DeepARLstmCellTape<T>(stateDim, hidden, seed);
        double stddev = Math.Sqrt(2.0 / hidden);
        _meanW = CreateRandom(new[] { actionDim, hidden }, stddev, seed + 7);
        _meanB = new Tensor<T>(new[] { actionDim, 1 });

        var trainable = new List<Tensor<T>>();
        trainable.AddRange(Training.TapeTrainingStep<T>.CollectParameters(new NeuralNetworks.Layers.LayerBase<T>[] { _cell }, -1));
        trainable.Add(_meanW);
        trainable.Add(_meanB);
        _trainable = trainable;

        _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            null, new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = learningRate });
        _stepper = new Training.TapeTrainingStepper<T>(this);

        ResetHidden();
    }

    private static Tensor<T> CreateRandom(int[] shape, double stddev, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++)
            t[i] = NumOps.FromDouble((rng.NextDouble() * 2 - 1) * stddev);
        return t;
    }

    [System.Diagnostics.CodeAnalysis.MemberNotNull(nameof(_h), nameof(_c))]
    private void ResetHidden()
    {
        _h = new Tensor<T>(new[] { _hidden, 1 });
        _c = new Tensor<T>(new[] { _hidden, 1 });
    }

    private static double ToD(T v) => Convert.ToDouble(v);

    /// <summary>Eager mean action for one observation, advancing the recurrent state. tanh-bounded to [-1, 1].</summary>
    private T[] Forward(T[] state)
    {
        var xt = new Tensor<T>(new[] { _stateDim, 1 });
        for (int i = 0; i < _stateDim; i++) xt[i, 0] = state[i];

        var (h, c) = _cell.Step(xt, _h, _c);
        _h = h;
        _c = c;

        var proj = Engine.TensorAdd(Engine.TensorMatMul(_meanW, h), _meanB); // [A,1]
        var mean = Engine.Tanh(proj);
        var result = new T[_actionDim];
        for (int a = 0; a < _actionDim; a++) result[a] = mean[a, 0];
        return result;
    }

    public Vector<T> SelectAction(Vector<T> state, bool explore)
    {
        var s = new T[_stateDim];
        for (int i = 0; i < _stateDim && i < state.Length; i++) s[i] = state[i];

        var mean = Forward(s); // advances the recurrent state (must happen on every act, incl. inference)
        var action = new T[_actionDim];
        for (int a = 0; a < _actionDim; a++)
        {
            double m = ToD(mean[a]);
            double v = explore ? m + _sigma * _rng.NextGaussian() : m;
            action[a] = NumOps.FromDouble(MathPolyfill.Clamp(v, -1.0, 1.0));
        }

        // The hidden state advances above (a recurrent policy must), but the rollout is NOT recorded here — so a
        // greedy backtest that never calls StoreExperience cannot grow or contaminate the training buffers.
        return new Vector<T>(action);
    }

    public void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        // Record the rollout HERE (owning a copy of the supplied state/action) — the transition the trainer
        // actually stored — so only true training steps feed the BPTT replay.
        var s = new T[_stateDim];
        for (int i = 0; i < _stateDim && i < state.Length; i++) s[i] = state[i];
        var a = new T[_actionDim];
        for (int i = 0; i < _actionDim && i < action.Length; i++) a[i] = action[i];

        _states.Add(s);
        _actions.Add(a);
        _rewards.Add(ToD(reward));
        _episodeDone = done;
    }

    public T Train()
    {
        // On-policy: only update at the end of a complete episode rollout.
        if (!_episodeDone || _states.Count == 0 || _states.Count != _rewards.Count)
        {
            return NumOps.Zero;
        }

        // BPTT: replay the rollout through the LSTM from a zero state; minimize Σ adv·½‖a−μ‖²/σ².
        var weights = PolicyGradientRollout<T>.LossWeights(_rewards, _gamma, _sigma);
        T reported = _stepper.Step(new Training.FusedTrainingStepRequest<T>
        {
            Layers = Array.Empty<ITrainableLayer<T>>(),
            Input = PolicyGradientRollout<T>.States(_states, _stateDim),
            Target = PolicyGradientRollout<T>.Target(_actions, weights, _actionDim),
            Forward = ReplayMeans,
            ComputeLoss = (means, target) => PolicyGradientRollout<T>.Loss(means, target, _actionDim),
            Optimizer = _optimizer,
            ExtraParameters = _trainable,
        });

        _states.Clear();
        _actions.Clear();
        _rewards.Clear();
        _episodeDone = false;
        return reported;
    }

    /// <summary>
    /// The policy means [actionDim, n] over a rollout's states [stateDim, n], replayed through the LSTM from a zero
    /// state. Each step's state is a column of the input, so a compiled plan replays each episode's own states.
    /// </summary>
    private Tensor<T> ReplayMeans(Tensor<T> states)
    {
        int n = states.Shape[1];
        var h = new Tensor<T>(new[] { _hidden, 1 });
        var c = new Tensor<T>(new[] { _hidden, 1 });
        var means = new Tensor<T>[n];
        for (int t = 0; t < n; t++)
        {
            var (hNew, cNew) = _cell.Step(Engine.TensorNarrow(states, 1, t, 1), h, c);
            h = hNew;
            c = cNew;
            means[t] = Engine.Tanh(Engine.TensorAdd(Engine.TensorMatMul(_meanW, h), _meanB)); // [A,1]
        }
        return n == 1 ? means[0] : Engine.TensorConcatenate(means, axis: 1);
    }

    public void ResetEpisode()
    {
        ResetHidden();
        _states.Clear();
        _actions.Clear();
        _rewards.Clear();
        _episodeDone = false;
    }

    /// <summary>
    /// Releases the LSTM cell this agent owns (its pool-rented weights and registered engine state).
    /// </summary>
    /// <remarks>
    /// Mirrors the RL agent bases: owned layers are released through the once-only guard, so a repeated
    /// call -- or the cell being released by another path -- is harmless. The head weights
    /// <c>_meanW</c>/<c>_meanB</c> are ordinary tensors with nothing to release.
    /// </remarks>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        DisposeOnceGuard.DisposeAll(new object?[] { _cell }, nameof(RecurrentPolicyAgent<T>));
    }
}
