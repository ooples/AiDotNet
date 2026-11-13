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
public sealed partial class FeedforwardPolicyAgent<T> : IPortfolioAgent<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static IEngine Engine => AiDotNetEngine.Current;

    private readonly int _stateDim;
    private readonly int _actionDim;
    private readonly double _sigma;
    private readonly T _gamma;
    private readonly Random _rng;

    [AiDotNet.Attributes.TrainableParameter]
    private readonly Tensor<T> _w1;   // [hidden, stateDim]
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Tensor<T> _b1;   // [hidden, 1]
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Tensor<T> _meanW; // [actionDim, hidden]
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Tensor<T> _meanB; // [actionDim, 1]
    private readonly IReadOnlyList<Tensor<T>> _trainable;
    private readonly AdamOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    // One episode's update on the shared fused session: the compiled plan when it applies, the eager tape otherwise.
    private readonly Training.TapeTrainingStepper<T> _stepper;

    /// <summary>Episode updates run on the fused compiled plan.</summary>
    internal int FusedTrainingStepCount => _stepper.FusedStepCount;

    /// <summary>Episode updates run on the eager tape.</summary>
    internal int EagerTrainingStepCount => _stepper.EagerStepCount;

    private readonly List<Tensor<T>> _states = new();
    private readonly List<Tensor<T>> _actions = new();
    private readonly List<T> _rewards = new();
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
        _gamma = NumOps.FromDouble(gamma);
        _rng = RandomHelper.CreateSeededRandom(seed);

        _w1 = CreateRandom(new[] { hidden, stateDim }, Math.Sqrt(2.0 / stateDim), seed + 1);
        _b1 = new Tensor<T>(new[] { hidden, 1 });
        _meanW = CreateRandom(new[] { actionDim, hidden }, Math.Sqrt(2.0 / hidden), seed + 2);
        _meanB = new Tensor<T>(new[] { actionDim, 1 });
        _trainable = new List<Tensor<T>> { _w1, _b1, _meanW, _meanB };

        _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            null, new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = learningRate });
        _stepper = new Training.TapeTrainingStepper<T>(this);
    }

    /// <summary>
    /// Weights uniform on <c>(-stddev, stddev)</c> from a generator seeded with <paramref name="seed"/>. Each draw is
    /// <c>(2u - 1) · stddev</c> formed in double and rounded to <typeparamref name="T"/> once; rounding u to
    /// <typeparamref name="T"/> before scaling would add a second rounding to every weight.
    /// </summary>
    internal static Tensor<T> CreateRandom(int[] shape, double stddev, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++)
            t[i] = NumOps.FromDouble((rng.NextDouble() * 2 - 1) * stddev);
        return t;
    }

    /// <summary>
    /// Mean actions [actionDim, n] for states [stateDim, n], one column per state: tanh(meanW·tanh(W1·x+b1)+meanB).
    /// </summary>
    private Tensor<T> MeanOf(Tensor<T> x)
    {
        var h = Engine.Tanh(Engine.TensorAdd(Engine.TensorMatMul(_w1, x), _b1));
        return Engine.Tanh(Engine.TensorAdd(Engine.TensorMatMul(_meanW, h), _meanB));
    }

    /// <inheritdoc />
    public Vector<T> SelectAction(Vector<T> state, bool explore)
    {
        var xt = Engine.Reshape(PolicyGradientRollout<T>.Column(state, _stateDim), new[] { _stateDim, 1 });
        var mean = MeanOf(xt);

        // The exploration noise sigma * z, one Gaussian draw per action in master's order (only when exploring), each
        // formed in double and rounded to T once; the action is then clip(mean + noise, -1, 1) on the whole tensor.
        Tensor<T>? noise = null;
        if (explore)
        {
            noise = new Tensor<T>(new[] { _actionDim, 1 });
            for (int a = 0; a < _actionDim; a++) noise[a, 0] = NumOps.FromDouble(_sigma * _rng.NextGaussian());
        }

        // Pure inference: no rollout is recorded here, so evaluating the policy (e.g. a greedy backtest that
        // never calls StoreExperience) cannot grow or contaminate the training buffers.
        return PolicyGradientRollout<T>.Act(mean, noise);
    }

    /// <inheritdoc />
    public void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        // The rollout is recorded HERE (owning a copy of the supplied state/action) — the transition the trainer
        // actually stored — rather than in SelectAction, so only true training steps populate the buffers.
        _states.Add(PolicyGradientRollout<T>.Column(state, _stateDim));
        _actions.Add(PolicyGradientRollout<T>.Column(action, _actionDim));
        _rewards.Add(reward);
        _episodeDone = done;
    }

    /// <inheritdoc />
    public T Train()
    {
        if (!_episodeDone || _states.Count == 0 || _states.Count != _rewards.Count)
        {
            return NumOps.Zero;
        }

        // The policy is memoryless, so the whole episode is one batched forward: states [stateDim, n] in, means
        // [actionDim, n] out, scored against the actions and their loss weights.
        var weights = PolicyGradientRollout<T>.LossWeights(_rewards, _gamma, _sigma);
        T reported = _stepper.Step(new Training.FusedTrainingStepRequest<T>
        {
            Layers = Array.Empty<ITrainableLayer<T>>(),
            Input = PolicyGradientRollout<T>.States(_states),
            Target = PolicyGradientRollout<T>.Target(_actions, weights),
            Forward = MeanOf,
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

    /// <inheritdoc />
    public void ResetEpisode()
    {
        _states.Clear();
        _actions.Clear();
        _rewards.Clear();
        _episodeDone = false;
    }
}
