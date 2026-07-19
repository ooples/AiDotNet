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
public sealed class RecurrentPolicyAgent<T> : IPortfolioAgent<T>
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
    private readonly Tensor<T> _meanW;              // [actionDim, hidden]
    private readonly Tensor<T> _meanB;              // [actionDim, 1]
    private readonly IReadOnlyList<Tensor<T>> _trainable;
    private readonly AdamOptimizer<T, Matrix<T>, Vector<T>> _optimizer;

    // Per-episode recurrent state (eager act path).
    private Tensor<T> _h = null!;
    private Tensor<T> _c = null!;

    // Current-episode rollout.
    private readonly List<T[]> _states = new();
    private readonly List<T[]> _actions = new();
    private readonly List<double> _rewards = new();
    private bool _episodeDone;

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

        _optimizer = new AdamOptimizer<T, Matrix<T>, Vector<T>>(
            null, new AdamOptimizerOptions<T, Matrix<T>, Vector<T>> { InitialLearningRate = learningRate });

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

        var proj = Engine.TensorBroadcastAdd(Engine.TensorMatMul(_meanW, h), _meanB); // [A,1]
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

        int n = _states.Count;

        // Discounted returns, then a whitened baseline (return − mean) as the advantage. Detached constants.
        var returns = new double[n];
        double g = 0;
        for (int t = n - 1; t >= 0; t--)
        {
            g = _rewards[t] + _gamma * g;
            returns[t] = g;
        }
        double mean = 0;
        for (int t = 0; t < n; t++) mean += returns[t];
        mean /= n;
        double var = 0;
        for (int t = 0; t < n; t++) var += (returns[t] - mean) * (returns[t] - mean);
        double std = Math.Sqrt(var / Math.Max(1, n)) + 1e-8;
        var advantage = new double[n];
        for (int t = 0; t < n; t++) advantage[t] = (returns[t] - mean) / std;

        // BPTT: replay the rollout through the LSTM under a tape; minimize Σ adv·½‖a−μ‖²/σ².
        double invSigma2 = 1.0 / (_sigma * _sigma);
        using var tape = new GradientTape<T>();
        var h = new Tensor<T>(new[] { _hidden, 1 });
        var c = new Tensor<T>(new[] { _hidden, 1 });
        Tensor<T>? loss = null;

        for (int t = 0; t < n; t++)
        {
            var xt = new Tensor<T>(new[] { _stateDim, 1 });
            for (int i = 0; i < _stateDim; i++) xt[i, 0] = _states[t][i];
            var (hNew, cNew) = _cell.Step(xt, h, c);
            h = hNew;
            c = cNew;

            var meanT = Engine.Tanh(Engine.TensorBroadcastAdd(Engine.TensorMatMul(_meanW, h), _meanB)); // [A,1]
            var actionT = new Tensor<T>(new[] { _actionDim, 1 });
            for (int a = 0; a < _actionDim; a++) actionT[a, 0] = _actions[t][a];

            var diff = Engine.TensorSubtract(actionT, meanT);
            var sq = Engine.ReduceSum(Engine.TensorMultiply(diff, diff), new[] { 0, 1 }, keepDims: false); // scalar
            // weight = advantage_t · 0.5 / σ²  (constant w.r.t. params)
            var term = Engine.TensorMultiplyScalar(sq, NumOps.FromDouble(0.5 * invSigma2 * advantage[t]));
            loss = loss is null ? term : Engine.TensorAdd(loss, term);
        }

        T reported = NumOps.Zero;
        if (loss is not null)
        {
            var grads = tape.ComputeGradients(loss, sources: null);
            var picked = new Dictionary<Tensor<T>, Tensor<T>>(Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
            foreach (var p in _trainable)
                if (grads.TryGetValue(p, out var gr)) picked[p] = gr;

            reported = loss.Length > 0 ? loss[0] : NumOps.Zero;
            _optimizer.Step(new TapeStepContext<T>(_trainable, picked, reported));
        }

        _states.Clear();
        _actions.Clear();
        _rewards.Clear();
        _episodeDone = false;
        return reported;
    }

    public void ResetEpisode()
    {
        ResetHidden();
        _states.Clear();
        _actions.Clear();
        _rewards.Clear();
        _episodeDone = false;
    }
}
