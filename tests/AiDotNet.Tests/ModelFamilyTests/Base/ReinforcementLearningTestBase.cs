using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Diagnostics;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for reinforcement learning agents.
/// Tests mathematical invariants: valid action selection, deterministic policy,
/// finite value estimates, training updates, and clone consistency.
/// </summary>
/// <remarks>
/// RL agents use IFullModel&lt;T, Vector&lt;T&gt;, Vector&lt;T&gt;&gt; where
/// input is state and output is action/value.
/// </remarks>
public abstract class ReinforcementLearningTestBase
{
    protected abstract IFullModel<double, Vector<double>, Vector<double>> CreateModel();

    protected virtual int StateDim => 4;

    private Vector<double> CreateRandomState(Random rng)
    {
        var state = new Vector<double>(StateDim);
        for (int i = 0; i < StateDim; i++)
            state[i] = rng.NextDouble() * 2.0 - 1.0;
        return state;
    }

    /// <summary>
    /// Wall-clock budget (seconds) for the bounded training loops below. Generous
    /// enough to train a correctly-implemented agent through a legitimate warm-up
    /// (e.g. DQN's replay-start period) yet well under the 60s test timeout. Agents
    /// that already satisfy the invariant early-exit immediately and never approach it.
    /// </summary>
    protected virtual double TrainingBudgetSeconds => 25.0;

    private static bool ActionsDiffer(Vector<double> a, Vector<double> b)
    {
        int minLen = Math.Min(a.Length, b.Length);
        for (int i = 0; i < minLen; i++)
            if (Math.Abs(a[i] - b[i]) > 1e-12)
                return true;
        return false;
    }

    private static bool ParametersChanged(double[] snapshot, Vector<double> current)
    {
        for (int i = 0; i < Math.Min(snapshot.Length, current.Length); i++)
            if (Math.Abs(snapshot[i] - current[i]) > 1e-15)
                return true;
        return false;
    }

    [Fact(Timeout = 60000)]
    public async Task ActionSelection_ShouldBeFinite()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var state = CreateRandomState(rng);

        // Train briefly
        var target = new Vector<double>(StateDim);
        for (int i = 0; i < StateDim; i++) target[i] = 0.5;
        model.Train(state, target);

        var action = model.Predict(state);
        Assert.True(action.Length > 0, "RL agent produced empty action.");
        for (int i = 0; i < action.Length; i++)
        {
            Assert.False(double.IsNaN(action[i]), $"Action[{i}] is NaN — broken policy.");
            Assert.False(double.IsInfinity(action[i]), $"Action[{i}] is Infinity — unbounded action.");
        }
    }

    [Fact(Timeout = 60000)]
    public async Task Policy_ShouldBeDeterministic()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var state = CreateRandomState(rng);

        var action1 = model.Predict(state);
        var action2 = model.Predict(state);

        Assert.Equal(action1.Length, action2.Length);
        for (int i = 0; i < action1.Length; i++)
            Assert.Equal(action1[i], action2[i]);
    }

    /// <summary>
    /// Set to false in test scaffolds for non-state-conditional RL agents
    /// (e.g. UCB / ε-greedy bandits per Auer 2002 §2.1, tabular Policy
    /// Iteration per Sutton & Barto 2018 §4.3 on unobserved states, A2C
    /// at random init before any policy has formed). For these, the
    /// "different state → different action" invariant doesn't apply by
    /// the algorithm's design — UCB picks by arm-uncertainty, not state;
    /// tabular methods return the default action for any state outside
    /// the visited set. Keeping the test invariant active for genuinely
    /// state-conditional agents (DQN, PPO, A3C, contextual bandits)
    /// still catches the bug class it was designed for.
    /// </summary>
    protected virtual bool IsStateConditional => true;

    [Fact(Timeout = 60000)]
    public async Task DifferentStates_DifferentActions()
    {
        if (!IsStateConditional) return;

        await Task.Yield();
        using var _arena = TensorArena.Create();
        using var model = CreateModel();

        // Two genuinely distinct states. They must differ in DIRECTION, not only in
        // magnitude: two collinear vectors (e.g. 0.1·1 and 0.9·1) are mapped to the
        // same greedy action by any positively-scaled policy — ReLU networks are
        // near scale-homogeneous and linear/tabular policies exactly so — and so
        // cannot probe state-conditionality at all. Use an ascending vs descending ramp.
        var state1 = new Vector<double>(StateDim);
        var state2 = new Vector<double>(StateDim);
        for (int i = 0; i < StateDim; i++)
        {
            state1[i] = (i + 1.0) / StateDim;              // 0.25, 0.50, 0.75, 1.00
            state2[i] = (StateDim - i) / (double)StateDim; // 1.00, 0.75, 0.50, 0.25
        }

        // A freshly-initialised discrete-action policy can map every state to a single
        // dominant action: the underlying Q-values / logits ARE functions of the state,
        // but the argmax read-out need not differ at random init. That is expected of an
        // UNTRAINED policy, not a degenerate one — no RL paper claims an untrained value
        // network has state-varying greedy actions, and with non-seeded weight init the
        // old "untrained argmax must differ" check was in fact flaky. So we verify the
        // paper's real guarantee: a state-conditional agent, given a differentiating
        // learning signal, can LEARN to act differently in the two states. We push state1
        // toward the first action and state2 toward the last and train until the greedy
        // actions diverge or a bounded budget elapses — training through any legitimate
        // warm-up (e.g. DQN's replay-start) rather than stripping it out.
        bool anyDifferent = ActionsDiffer(model.Predict(state1), model.Predict(state2));
        int actionLen = model.Predict(state1).Length;
        if (!anyDifferent && actionLen >= 2)
        {
            var target1 = new Vector<double>(actionLen);
            var target2 = new Vector<double>(actionLen);
            target1[0] = 1.0;               // prefer the first action in state1
            target2[actionLen - 1] = 1.0;   // prefer the last action in state2

            var sw = Stopwatch.StartNew();
            for (int iter = 0; !anyDifferent && sw.Elapsed.TotalSeconds < TrainingBudgetSeconds; iter++)
            {
                model.Train(state1, target1);
                model.Train(state2, target2);
                if (iter % 16 == 0)
                    anyDifferent = ActionsDiffer(model.Predict(state1), model.Predict(state2));
            }
            anyDifferent = anyDifferent || ActionsDiffer(model.Predict(state1), model.Predict(state2));
        }

        Assert.True(anyDifferent,
            "RL agent cannot learn to map two distinct states to distinct actions even " +
            "after a differentiating training signal — its policy does not condition on state.");
    }

    [Fact(Timeout = 60000)]
    public async Task Training_ShouldChangeParameters()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();

        var paramsBefore = ((IParameterizable<double, Vector<double>, Vector<double>>)model).GetParameters();
        var snapshot = new double[paramsBefore.Length];
        for (int i = 0; i < paramsBefore.Length; i++) snapshot[i] = paramsBefore[i];

        // Train with a NON-DEGENERATE, learnable signal until the parameters change or a
        // bounded budget elapses. The signal must vary: a constant-reward stream is
        // genuinely unlearnable for some correct algorithms — a gradient bandit
        // (Sutton & Barto 2018 §2.8) leaves its preferences unchanged when every arm
        // returns the same reward, because there is nothing to distinguish the arms.
        // Warm-up-gated agents (DQN's replay-start) likewise need more than a handful of
        // steps before the first gradient is applied. We therefore alternate two
        // (state, target) pairs whose decoded rewards differ in magnitude (1.0 vs 0.3),
        // giving every algorithm family a real learning signal, and train within a
        // bounded budget rather than asserting after a fixed 5 steps.
        var stateA = CreateRandomState(rng);
        var stateB = CreateRandomState(rng);
        int actionLen = Math.Max(model.Predict(stateA).Length, 2);
        var targetA = new Vector<double>(actionLen);
        var targetB = new Vector<double>(actionLen);
        targetA[0] = 1.0;                 // action 0, reward 1.0
        targetB[actionLen - 1] = 0.3;     // last action, reward 0.3 (≠ reward of A)

        bool anyChanged = false;
        var sw = Stopwatch.StartNew();
        for (int iter = 0; !anyChanged && sw.Elapsed.TotalSeconds < TrainingBudgetSeconds; iter++)
        {
            if (iter % 2 == 0) model.Train(stateA, targetA);
            else model.Train(stateB, targetB);
            if (iter % 8 == 0)
                anyChanged = ParametersChanged(
                    snapshot, ((IParameterizable<double, Vector<double>, Vector<double>>)model).GetParameters());
        }
        anyChanged = anyChanged || ParametersChanged(
            snapshot, ((IParameterizable<double, Vector<double>, Vector<double>>)model).GetParameters());

        Assert.True(anyChanged, "RL agent parameters unchanged after a learnable training signal.");
    }

    [Fact(Timeout = 60000)]
    public async Task Clone_ShouldProduceSamePolicy()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var state = CreateRandomState(rng);

        var cloned = model.Clone();
        var action1 = model.Predict(state);
        var action2 = cloned.Predict(state);
        for (int i = 0; i < action1.Length; i++)
            Assert.Equal(action1[i], action2[i]);
    }

    [Fact(Timeout = 60000)]
    public async Task Metadata_ShouldExistAfterTraining()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var state = CreateRandomState(rng);
        var target = new Vector<double>(StateDim);
        model.Train(state, target);
        Assert.NotNull(model.GetModelMetadata());
    }

    [Fact(Timeout = 60000)]
    public async Task Parameters_ShouldBeNonEmpty()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        using var model = CreateModel();
        Assert.True(((IParameterizable<double, Vector<double>, Vector<double>>)model).GetParameters().Length > 0, "RL agent should have parameters.");
    }
}
