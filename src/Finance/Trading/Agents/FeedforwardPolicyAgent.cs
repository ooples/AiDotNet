using System;
using System.Collections.Generic;
using AiDotNet.Autodiff;
using AiDotNet.Finance.Trading.Evaluation;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Finance.Trading.Agents;

/// <summary>
/// A MEMORYLESS (feed-forward MLP) continuous policy for the portfolio-manager harness — the controlled
/// counterpart to <see cref="RecurrentPolicyAgent{T}"/>. It uses the SAME on-policy REINFORCE-with-baseline
/// training, exploration σ, and objective, differing ONLY in that the policy is a plain MLP over the current
/// observation with no hidden state carried across steps. That makes recurrent-vs-MLP a clean controlled
/// experiment: run both through <see cref="Evaluation.PortfolioExperimentRunner"/> / study and rank on the
/// untouched holdout — any gap is attributable to memory, not the algorithm.
/// </summary>
/// <typeparam name="T">Element type (float/double).</typeparam>
public sealed class FeedforwardPolicyAgent<T> : IPortfolioAgent<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static IEngine Engine => AiDotNetEngine.Current;

    private readonly int _stateDim;
    private readonly int _actionDim;
    private readonly double _sigma;
    private readonly double _gamma;
    private readonly Random _rng;

    private readonly Tensor<T> _w1;   // [hidden, stateDim]
    private readonly Tensor<T> _b1;   // [hidden, 1]
    private readonly Tensor<T> _meanW; // [actionDim, hidden]
    private readonly Tensor<T> _meanB; // [actionDim, 1]
    private readonly IReadOnlyList<Tensor<T>> _trainable;
    private readonly AdamOptimizer<T, Matrix<T>, Vector<T>> _optimizer;

    private readonly List<T[]> _states = new();
    private readonly List<T[]> _actions = new();
    private readonly List<double> _rewards = new();
    private bool _episodeDone;

    public FeedforwardPolicyAgent(
        int stateDim, int actionDim, int hidden = 32, double explorationSigma = 0.2,
        double learningRate = 1e-3, double gamma = 0.99, int seed = 0)
    {
        if (stateDim <= 0) throw new ArgumentOutOfRangeException(nameof(stateDim));
        if (actionDim <= 0) throw new ArgumentOutOfRangeException(nameof(actionDim));
        if (hidden <= 0) throw new ArgumentOutOfRangeException(nameof(hidden));

        _stateDim = stateDim;
        _actionDim = actionDim;
        _sigma = explorationSigma > 1e-6 ? explorationSigma : 1e-6;
        _gamma = gamma;
        _rng = RandomHelper.CreateSeededRandom(seed);

        _w1 = CreateRandom(new[] { hidden, stateDim }, Math.Sqrt(2.0 / stateDim), seed + 1);
        _b1 = new Tensor<T>(new[] { hidden, 1 });
        _meanW = CreateRandom(new[] { actionDim, hidden }, Math.Sqrt(2.0 / hidden), seed + 2);
        _meanB = new Tensor<T>(new[] { actionDim, 1 });
        _trainable = new List<Tensor<T>> { _w1, _b1, _meanW, _meanB };

        _optimizer = new AdamOptimizer<T, Matrix<T>, Vector<T>>(
            null, new AdamOptimizerOptions<T, Matrix<T>, Vector<T>> { InitialLearningRate = learningRate });
    }

    private static Tensor<T> CreateRandom(int[] shape, double stddev, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++)
            t[i] = NumOps.FromDouble((rng.NextDouble() * 2 - 1) * stddev);
        return t;
    }

    private static double ToD(T v) => Convert.ToDouble(v);

    /// <summary>Mean action tensor [actionDim,1] for one state tensor [stateDim,1]: tanh(meanW·tanh(W1·x+b1)+meanB).</summary>
    private Tensor<T> MeanOf(Tensor<T> x)
    {
        var h = Engine.Tanh(Engine.TensorBroadcastAdd(Engine.TensorMatMul(_w1, x), _b1));
        return Engine.Tanh(Engine.TensorBroadcastAdd(Engine.TensorMatMul(_meanW, h), _meanB));
    }

    public Vector<T> SelectAction(Vector<T> state, bool explore)
    {
        var s = new T[_stateDim];
        for (int i = 0; i < _stateDim && i < state.Length; i++) s[i] = state[i];

        var xt = new Tensor<T>(new[] { _stateDim, 1 });
        for (int i = 0; i < _stateDim; i++) xt[i, 0] = s[i];
        var mean = MeanOf(xt);

        var action = new T[_actionDim];
        for (int a = 0; a < _actionDim; a++)
        {
            double m = ToD(mean[a, 0]);
            double v = explore ? m + _sigma * _rng.NextGaussian() : m;
            action[a] = NumOps.FromDouble(Math.Clamp(v, -1.0, 1.0));
        }

        // Pure inference: no rollout is recorded here, so evaluating the policy (e.g. a greedy backtest that
        // never calls StoreExperience) cannot grow or contaminate the training buffers.
        return new Vector<T>(action);
    }

    public void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        // The rollout is recorded HERE (owning a copy of the supplied state/action) — the transition the trainer
        // actually stored — rather than in SelectAction, so only true training steps populate the buffers.
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
        if (!_episodeDone || _states.Count == 0 || _states.Count != _rewards.Count)
        {
            return NumOps.Zero;
        }

        int n = _states.Count;
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

        double invSigma2 = 1.0 / (_sigma * _sigma);
        using var tape = new GradientTape<T>();
        Tensor<T>? loss = null;
        for (int t = 0; t < n; t++)
        {
            var xt = new Tensor<T>(new[] { _stateDim, 1 });
            for (int i = 0; i < _stateDim; i++) xt[i, 0] = _states[t][i];
            var meanT = MeanOf(xt); // memoryless: each step independent

            var actionT = new Tensor<T>(new[] { _actionDim, 1 });
            for (int a = 0; a < _actionDim; a++) actionT[a, 0] = _actions[t][a];

            var diff = Engine.TensorSubtract(actionT, meanT);
            var sq = Engine.ReduceSum(Engine.TensorMultiply(diff, diff), new[] { 0, 1 }, keepDims: false);
            var term = Engine.TensorMultiplyScalar(sq, NumOps.FromDouble(0.5 * invSigma2 * advantage[t]));
            loss = loss is null ? term : Engine.TensorAdd(loss, term);
        }

        T reported = NumOps.Zero;
        if (loss is not null)
        {
            var grads = tape.ComputeGradients(loss, sources: null);
            var picked = new Dictionary<Tensor<T>, Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);
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
        _states.Clear();
        _actions.Clear();
        _rewards.Clear();
        _episodeDone = false;
    }
}
