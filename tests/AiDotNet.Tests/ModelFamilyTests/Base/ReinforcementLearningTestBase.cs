using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
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
    /// Fixed iteration cap for the bounded training loops below. A DETERMINISTIC step
    /// count (not a wall-clock budget) is used so the result does not depend on machine
    /// speed or load: a warm-up-gated agent (e.g. DQN's default WarmupSteps = 1000)
    /// applies no gradient until the replay buffer is primed, so the loop must run enough
    /// steps to clear that warm-up plus a few hundred learning updates — guaranteed by a
    /// step count, only borderline under a wall clock. The per-step cost is small (warm-up
    /// steps just fill the buffer; post-warm-up steps backprop a tiny network), and agents
    /// that already satisfy the invariant early-exit immediately, so this stays well under
    /// the 60s test timeout. Expensive on-policy agents (PPO/TRPO) are opted out separately.
    /// </summary>
    protected virtual int TrainingIterationCap => 1500;

    private static bool ActionsDiffer(Vector<double> a, Vector<double> b)
    {
        int minLen = Math.Min(a.Length, b.Length);
        for (int i = 0; i < minLen; i++)
            if (Math.Abs(a[i] - b[i]) > 1e-12)
                return true;
        return false;
    }

    /// <summary>
    /// A battery of directionally-distinct states (ascending/descending ramps, two
    /// opposite alternating patterns, and two complementary one-hot-ish spikes), built
    /// deterministically so the test is reproducible.
    /// </summary>
    private Vector<double>[] BuildStateBattery()
    {
        var ascending = new Vector<double>(StateDim);
        var descending = new Vector<double>(StateDim);
        var altA = new Vector<double>(StateDim);
        var altB = new Vector<double>(StateDim);
        var spikeLow = new Vector<double>(StateDim);
        var spikeHigh = new Vector<double>(StateDim);
        for (int i = 0; i < StateDim; i++)
        {
            ascending[i] = (i + 1.0) / StateDim;               // 0.25, 0.50, 0.75, 1.00
            descending[i] = (StateDim - i) / (double)StateDim; // 1.00, 0.75, 0.50, 0.25
            altA[i] = (i % 2 == 0) ? 1.0 : -1.0;               // +,-,+,-
            altB[i] = (i % 2 == 0) ? -1.0 : 1.0;               // -,+,-,+
        }
        spikeLow[0] = 1.0;                 // weight on the first feature
        spikeHigh[StateDim - 1] = 1.0;     // weight on the last feature
        return new[] { ascending, descending, altA, altB, spikeLow, spikeHigh };
    }

    /// <summary>
    /// True if the agent's greedy action is not identical across every state in the
    /// battery — i.e. its policy conditions on the input for at least one pair.
    /// </summary>
    private static bool ActionsVaryAcross(IFullModel<double, Vector<double>, Vector<double>> model, Vector<double>[] states)
    {
        var first = model.Predict(states[0]);
        for (int i = 1; i < states.Length; i++)
            if (ActionsDiffer(first, model.Predict(states[i])))
                return true;
        return false;
    }

    private static bool ParametersChanged(double[] snapshot, Vector<double> current)
    {
        // A change in length is itself a parameter change: tabular agents grow their
        // Q-table lazily as new states are visited, so an agent that starts with an empty
        // parameter vector and acquires entries during training HAS changed its parameters.
        if (snapshot.Length != current.Length)
            return true;
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
    /// Set to false in test scaffolds for agents where the "different state →
    /// different action" invariant does not apply by the algorithm's design:
    /// non-contextual k-armed bandits (UCB / ε-greedy / Thompson / gradient —
    /// they pick by arm statistics, not state); tabular DP returning the default
    /// action for unobserved states (Policy/ModifiedPolicy Iteration, Sutton &
    /// Barto 2018 §4.3); actor-critic policy-gradient methods whose untrained
    /// policy is ~uniform and whose on-policy trajectory update the single-
    /// transition supervised adapter cannot drive (A2C / PPO / TRPO); on-policy
    /// SARSA(λ) which evaluates the action it actually took; and multi-agent
    /// QMIX which consumes a joint observation, not a single agent's state.
    /// The invariant stays active for the genuinely state-conditional,
    /// adapter-drivable agents (DQN family, REINFORCE, value/linear methods)
    /// where it catches the "policy ignores state" bug class it was designed for.
    /// </summary>
    protected virtual bool IsStateConditional => true;

    [Fact(Timeout = 60000)]
    public async Task DifferentStates_DifferentActions()
    {
        if (!IsStateConditional) return;

        await Task.Yield();
        using var _arena = TensorArena.Create();
        using var model = CreateModel();

        // For a VALUE-BASED agent the greedy action is argmax over Q(s,·) — a lossy projection that
        // can be constant across inputs at random init even when Q is genuinely state-conditional.
        // When the agent exposes its raw action-values, probe those directly: that signal is the
        // deterministic, non-projected evidence of state-conditionality and removes the random-init
        // flakiness of an argmax-only read-out (no reliance on a training fallback flipping the argmax).
        if (model is IActionValueProvider<double> valueProvider)
        {
            var qBattery = BuildStateBattery();
            var firstQ = valueProvider.GetActionValues(qBattery[0]);
            bool qDiffers = false;
            for (int i = 1; i < qBattery.Length && !qDiffers; i++)
                qDiffers = ActionsDiffer(firstQ, valueProvider.GetActionValues(qBattery[i]));
            Assert.True(qDiffers,
                "Value-based RL agent's action-values are identical across a diverse state battery — " +
                "its Q-function ignores the input (degenerate policy).");
            return;
        }

        // Probe a BATTERY of directionally-distinct states rather than a single pair.
        // A freshly-initialised discrete-action policy can map one particular state pair
        // to the same dominant action — the underlying Q-values / logits ARE functions of
        // the state, but the argmax read-out need not differ for that pair, and with
        // non-seeded weight init a single-pair check was flaky. A genuinely state-
        // conditional policy will, however, produce a different greedy action for SOME
        // pair among several diverse states; a policy that truly ignores its input returns
        // the same action for ALL of them. States must differ in DIRECTION, not only in
        // magnitude (a positively-scaled policy maps collinear states to the same action).
        var battery = BuildStateBattery();
        bool anyDifferent = ActionsVaryAcross(model, battery);

        // If no untrained pair diverges, verify the paper's real guarantee: a state-
        // conditional agent, given a differentiating learning signal, can LEARN to act
        // differently. Push battery[0] toward the first action and battery[1] toward the
        // last, training through any legitimate warm-up (e.g. DQN's replay-start) up to a
        // deterministic step cap, then re-probe the whole battery.
        int actionLen = model.Predict(battery[0]).Length;
        if (!anyDifferent && actionLen >= 2)
        {
            // Use a large reward so the reinforced action's learned value clearly exceeds
            // any other action's initial value, flipping the greedy action within a few
            // post-warm-up updates (fast early-exit) instead of inching past random init.
            var target1 = new Vector<double>(actionLen);
            var target2 = new Vector<double>(actionLen);
            target1[0] = 10.0;               // prefer the first action in battery[0]
            target2[actionLen - 1] = 10.0;   // prefer the last action in battery[1]

            for (int iter = 0; !anyDifferent && iter < TrainingIterationCap; iter++)
            {
                model.Train(battery[0], target1);
                model.Train(battery[1], target2);
                if (iter % 16 == 0)
                    anyDifferent = ActionsDiffer(model.Predict(battery[0]), model.Predict(battery[1]));
            }
            anyDifferent = anyDifferent || ActionsVaryAcross(model, battery);
        }

        Assert.True(anyDifferent,
            "RL agent returns the same action for every state in a diverse battery and cannot " +
            "learn to distinguish them after a differentiating signal — its policy ignores state.");
    }

    /// <summary>
    /// Set to false in test scaffolds for agents that cannot be trained through the
    /// generic single-transition <c>Train(state, target)</c> adapter, because their
    /// learning rule needs an input this harness does not provide. The parameter-change
    /// invariant then does not apply by the algorithm's design. Examples:
    /// multi-agent QMIX (Train consumes a joint observation across all agents, not a
    /// single agent's state) and TRPO (Sutton & Barto 2018 §13; Schulman et al. 2015 —
    /// its KL-constrained trust-region step is computed over whole on-policy trajectories
    /// with advantages, so a stream of isolated terminal transitions yields ~zero update).
    /// </summary>
    protected virtual bool TrainsViaSingleTransitionAdapter => true;

    [Fact(Timeout = 60000)]
    public async Task Training_ShouldChangeParameters()
    {
        if (!TrainsViaSingleTransitionAdapter) return;

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
        for (int iter = 0; !anyChanged && iter < TrainingIterationCap; iter++)
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
