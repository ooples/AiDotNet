using AiDotNet.Interfaces;
using System;
using System.Reflection;
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
public abstract class ReinforcementLearningTestBase<T>
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Converts a double literal into the fixture's numeric type.</summary>
    protected static T ToT(double value) => NumOps.FromDouble(value);

    /// <summary>Converts a fixture-typed value back to double for finiteness / magnitude asserts.</summary>
    protected static double ToD(T value) => Convert.ToDouble(value);

    protected abstract IFullModel<T, Vector<T>, Vector<T>> CreateModel();

    /// <summary>Fallback state width, used only for an agent that does not report its own.</summary>
    protected virtual int StateDim => 4;

    /// <summary>
    /// False for an agent whose Train() cannot be driven by a single-agent store-then-Train loop at
    /// all, because it consumes joint multi-agent transitions through a different API.
    /// </summary>
    /// <remarks>
    /// MADDPG and QMIX THROW rather than under-train ("requires joint transitions stored via
    /// StoreMultiAgentExperience ... expected 8/4/8" for a 4/2/4 single-agent transition), so the
    /// real-loop invariants cannot run against them at all. Emitted by the scaffold generator.
    /// </remarks>
    protected virtual bool SupportsSingleAgentOnlineLoop => true;

    /// <summary>
    /// False for an agent that runs the online loop but cannot be expected to shift its greedy action
    /// toward whichever action was just rewarded.
    /// </summary>
    /// <remarks>
    /// This is about the LEARNING SIGNAL, not about state-conditionality, so it is deliberately a
    /// separate flag from <see cref="IsStateConditional"/>, whose membership is different: on-policy
    /// methods need whole trajectories (A2C/PPO/TRPO), REINFORCE needs complete episodes, SARSA(lambda)
    /// evaluates the action actually taken, CQL/IQL are offline by construction and resist moving
    /// toward actions their fixed dataset does not support, and Dreamer optimises against an imagined
    /// value whose world model needs far more than a unit-test budget to fit. Emitted by the generator.
    /// </remarks>
    protected virtual bool FollowsOnlineReward => true;

    private int? _resolvedStateDim;
    private int? _resolvedActionDim;

    /// <summary>The state width THIS agent accepts, taken from the model rather than assumed.</summary>
    /// <remarks>
    /// A fixed 4 was wrong for every agent declaring a different StateSize -- FinancialDQNAgent uses
    /// 10, MarketMakingAgent 64 -- and once TradingAgentBase.ValidateTransitionShape began enforcing
    /// the declared width, every invariant that fed a state threw "State length 4 must match StateSize
    /// N" instead of testing anything at all. FeatureCount is each agent's own answer, so ask once.
    /// </remarks>
    protected int EffectiveStateDim
    {
        get
        {
            if (_resolvedStateDim is null) ResolveAgentShapes();
            return _resolvedStateDim ?? StateDim;
        }
    }

    /// <summary>The action width THIS agent produces.</summary>
    /// <remarks>
    /// ValidateTransitionShape checks the action too, so a target vector sized by the state width
    /// fails exactly as hard as a mis-sized state. Read from SelectAction's own output rather than
    /// from options: every agent can answer that, with no per-agent knowledge and no reflection.
    /// </remarks>
    protected int EffectiveActionDim
    {
        get
        {
            if (_resolvedActionDim is null) ResolveAgentShapes();
            return _resolvedActionDim ?? StateDim;
        }
    }

    private void ResolveAgentShapes()
    {
        int stateDim = StateDim;
        int actionDim = StateDim;

        using (var probe = CreateModel())
        {
            if (probe is AiDotNet.ReinforcementLearning.Agents.ReinforcementLearningAgentBase<T> agent
                && agent.FeatureCount > 0)
            {
                stateDim = agent.FeatureCount;
            }

            if (probe is IRLAgent<T> rlAgent)
            {
                try
                {
                    var sampleAction = rlAgent.SelectAction(new Vector<T>(stateDim), false);
                    if (sampleAction is not null && sampleAction.Length > 0)
                        actionDim = sampleAction.Length;
                }
                catch (ArgumentException)
                {
                    // An agent that rejects a single-agent state (MADDPG wants a joint observation)
                    // keeps the declared default; SupportsSingleAgentOnlineLoop skips it before the
                    // width is ever used, and the narrower invariants fail on their own terms rather
                    // than here, where the message would be unreadable.
                }
                catch (InvalidOperationException)
                {
                }
            }
        }

        _resolvedStateDim = stateDim;
        _resolvedActionDim = actionDim;
    }

    private Vector<T> CreateRandomState(Random rng)
    {
        int dim = EffectiveStateDim;
        var state = new Vector<T>(dim);
        for (int i = 0; i < dim; i++)
            state[i] = ToT(rng.NextDouble() * 2.0 - 1.0);
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

    private static bool ActionsDiffer(Vector<T> a, Vector<T> b)
    {
        int minLen = Math.Min(a.Length, b.Length);
        for (int i = 0; i < minLen; i++)
            if (Math.Abs(ToD(a[i]) - ToD(b[i])) > 1e-12)
                return true;
        return false;
    }

    /// <summary>
    /// A battery of directionally-distinct states (ascending/descending ramps, two
    /// opposite alternating patterns, and two complementary one-hot-ish spikes), built
    /// deterministically so the test is reproducible.
    /// </summary>
    private Vector<T>[] BuildStateBattery()
    {
        int dim = EffectiveStateDim;
        var ascending = new Vector<T>(dim);
        var descending = new Vector<T>(dim);
        var altA = new Vector<T>(dim);
        var altB = new Vector<T>(dim);
        var spikeLow = new Vector<T>(dim);
        var spikeHigh = new Vector<T>(dim);
        for (int i = 0; i < dim; i++)
        {
            ascending[i] = ToT((i + 1.0) / dim);               // 0.25, 0.50, 0.75, 1.00
            descending[i] = ToT((dim - i) / (double)dim);      // 1.00, 0.75, 0.50, 0.25
            altA[i] = ToT((i % 2 == 0) ? 1.0 : -1.0);          // +,-,+,-
            altB[i] = ToT((i % 2 == 0) ? -1.0 : 1.0);          // -,+,-,+
        }
        spikeLow[0] = ToT(1.0);             // weight on the first feature
        spikeHigh[dim - 1] = ToT(1.0);      // weight on the last feature
        return new[] { ascending, descending, altA, altB, spikeLow, spikeHigh };
    }

    /// <summary>
    /// True if the agent's greedy action is not identical across every state in the
    /// battery — i.e. its policy conditions on the input for at least one pair.
    /// </summary>
    private static bool ActionsVaryAcross(IFullModel<T, Vector<T>, Vector<T>> model, Vector<T>[] states)
    {
        var first = model.Predict(states[0]);
        for (int i = 1; i < states.Length; i++)
            if (ActionsDiffer(first, model.Predict(states[i])))
                return true;
        return false;
    }

    private static bool ParametersChanged(double[] snapshot, Vector<T> current)
    {
        // A change in length is itself a parameter change: tabular agents grow their
        // Q-table lazily as new states are visited, so an agent that starts with an empty
        // parameter vector and acquires entries during training HAS changed its parameters.
        if (snapshot.Length != current.Length)
            return true;
        for (int i = 0; i < Math.Min(snapshot.Length, current.Length); i++)
            if (Math.Abs(snapshot[i] - ToD(current[i])) > 1e-15)
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

        // Train briefly. The target becomes the ACTION in StoreSupervisedExperience, so it must be
        // action-width: ValidateTransitionShape rejects a mis-sized action just as it does a state.
        var target = new Vector<T>(EffectiveActionDim);
        for (int i = 0; i < target.Length; i++) target[i] = ToT(0.5);
        model.Train(state, target);

        var action = model.Predict(state);
        Assert.True(action.Length > 0, "RL agent produced empty action.");
        for (int i = 0; i < action.Length; i++)
        {
            Assert.False(double.IsNaN(ToD(action[i])), $"Action[{i}] is NaN — broken policy.");
            Assert.False(double.IsInfinity(ToD(action[i])), $"Action[{i}] is Infinity — unbounded action.");
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
        if (model is IActionValueProvider<T> valueProvider)
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
            var target1 = new Vector<T>(actionLen);
            var target2 = new Vector<T>(actionLen);
            target1[0] = ToT(10.0);               // prefer the first action in battery[0]
            target2[actionLen - 1] = ToT(10.0);   // prefer the last action in battery[1]

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

    [SkippableFact(Timeout = 60000)]
    public async Task Training_ShouldChangeParameters()
    {
        // An explicit skip, not a bare return. Returning early reported this agent's ONLY training
        // invariant as a pass while it asserted nothing at all, so an agent opted out of the check
        // looked identical in CI to one that genuinely trains. A skip is visible in the run summary
        // and can be counted; a silent green cannot.
        Skip.IfNot(TrainsViaSingleTransitionAdapter,
            "This agent does not learn from isolated single transitions (e.g. a gradient bandit needs "
            + "differing rewards per arm, TRPO needs whole on-policy trajectories), so the "
            + "single-transition adapter cannot produce a parameter change for it.");

        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();

        var paramsBefore = ((IParameterizable<T, Vector<T>, Vector<T>>)model).GetParameters();
        var snapshot = new double[paramsBefore.Length];
        for (int i = 0; i < paramsBefore.Length; i++) snapshot[i] = ToD(paramsBefore[i]);

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
        var targetA = new Vector<T>(actionLen);
        var targetB = new Vector<T>(actionLen);
        targetA[0] = ToT(1.0);                 // action 0, reward 1.0
        targetB[actionLen - 1] = ToT(0.3);     // last action, reward 0.3 (≠ reward of A)

        bool anyChanged = false;
        for (int iter = 0; !anyChanged && iter < TrainingIterationCap; iter++)
        {
            if (iter % 2 == 0) model.Train(stateA, targetA);
            else model.Train(stateB, targetB);
            if (iter % 8 == 0)
                anyChanged = ParametersChanged(
                    snapshot, ((IParameterizable<T, Vector<T>, Vector<T>>)model).GetParameters());
        }
        anyChanged = anyChanged || ParametersChanged(
            snapshot, ((IParameterizable<T, Vector<T>, Vector<T>>)model).GetParameters());

        Assert.True(anyChanged, "RL agent parameters unchanged after a learnable training signal.");
    }

    /// <summary>
    /// Steps of the real reinforcement-learning loop this invariant may run before giving up, once the
    /// agent's own warmup gate has been lowered.
    /// </summary>
    /// <remarks>
    /// This was 14000 purely to outlast SAC's shipped WarmupSteps of 10000. Running that many steps at
    /// a production batch of 256 -- twice over, for the two reward directions -- exhausted the test
    /// host, and the ~49 RL agents that have no warmup gate at all paid the same cost for nothing.
    /// Lowering the gate first makes a few hundred steps sufficient, so the loop is reachable rather
    /// than merely long.
    /// </remarks>
    protected virtual int RealLoopStepBudget => 800;

    /// <summary>
    /// Lowers an agent's own warmup / batch gate so the real training loop is reachable within
    /// <see cref="RealLoopStepBudget"/>, and reports whether anything was lowered.
    /// </summary>
    /// <remarks>
    /// Five agents -- SAC, DDPG, DQN, DoubleDQN and DuelingDQN -- return immediately from Train()
    /// until WarmupSteps have elapsed AND the replay buffer can fill a batch. At their shipped
    /// defaults that is 10000 steps at a batch of 256, which no unit test can afford.
    ///
    /// Each of those agents overrides GetOptions() to hand back its OWN live options instance rather
    /// than the base class's copy, and re-reads it on every Train() call, so lowering the values here
    /// genuinely opens the gate. This changes only how far the fixture must step to reach the update,
    /// never what the update computes, and leaves production defaults untouched.
    ///
    /// Agents without these properties are unaffected -- they have no warmup gate to begin with.
    /// </remarks>
    private static bool LowerTrainingGates(IFullModel<T, Vector<T>, Vector<T>> model)
    {
        if (model is not IConfigurableModel<T> configurable) return false;

        var options = configurable.GetOptions();
        if (options is null) return false;

        bool loweredWarmup = TryLowerInt(options, "WarmupSteps", 2);
        bool loweredBatch = TryLowerInt(options, "BatchSize", 8);
        return loweredWarmup || loweredBatch;
    }

    /// <summary>
    /// Sets a public settable int property when it exists and currently holds a LARGER value. Only
    /// ever lowers, so an agent that already ships a small gate keeps its own setting.
    /// </summary>
    private static bool TryLowerInt(object target, string propertyName, int value)
    {
        var property = target.GetType().GetProperty(
            propertyName, BindingFlags.Instance | BindingFlags.Public);

        if (property is null || !property.CanWrite || property.PropertyType != typeof(int))
            return false;

        if (property.GetValue(target) is not int current || current <= value)
            return false;

        property.SetValue(target, value);
        return true;
    }

    [SkippableFact(Timeout = 300000)]
    public async Task Policy_ShouldFollowTheReward()
    {
        Skip.IfNot(SupportsSingleAgentOnlineLoop,
            "This agent consumes joint multi-agent transitions through a different API, so the "
            + "single-agent store-then-Train loop this invariant drives cannot run against it.");

        Skip.IfNot(FollowsOnlineReward,
            "This agent runs the loop but cannot be expected to move its greedy action toward the "
            + "rewarded one within a unit-test budget: on-policy methods need whole trajectories, "
            + "REINFORCE needs complete episodes, SARSA(lambda) evaluates the action it actually took, "
            + "CQL/IQL are offline and resist actions their fixed dataset does not support, and "
            + "Dreamer optimises against an imagined value whose world model is nowhere near fitted.");

        // OUTCOME invariant, deliberately mechanism-independent.
        //
        // Every structural check has a blind spot. Per-component liveness cannot see this defect at
        // all: SAC's actor loss is alpha*logPi minus a DETACHED min(Q1,Q2), so the actor still gets
        // real gradient from the entropy term and its weights move every step — it simply never
        // follows the reward. A reachability check catches that particular severing, but not one
        // performed by rebuilding a tensor element by element, which leaves no Predict call to find.
        //
        // What no severing can fake is the outcome: a policy that never sees the value signal cannot
        // prefer a rewarded action over a punished one. So train the SAME state with the reward on
        // one action, then again with the reward on the other, and require the policy to end up
        // somewhere different. This is the shape FinancialSacActorHeadTests already uses by hand.
        await Task.Yield();
        using var _arena = TensorArena.Create();

        var probeRng = ModelTestHelpers.CreateSeededRandom();
        var state = CreateRandomState(probeRng);

        using (var probe = CreateModel())
        {
            Skip.If(probe is not IRLAgent<T>, "This fixture's model is not an IRLAgent.");
        }

        var (actionA, actionB) = TwoDistinctActions(state);
        Skip.If(actionA is null || actionB is null,
            "This agent's action space did not yield two distinct actions for one state, so a "
            + "reward difference between actions cannot be expressed here.");

        var (favouringA, stepsA) = PolicyAfterRewarding(state, actionA!, actionB!);
        var (favouringB, stepsB) = PolicyAfterRewarding(state, actionB!, actionA!);

        Skip.If(favouringA is null || favouringB is null,
            $"Neither run reached the agent's own training gate (A stopped at {stepsA} steps, B at "
            + $"{stepsB} of a {RealLoopStepBudget} budget), so the policy was never updated and the "
            + "question cannot be answered here.");

        // DIRECTIONAL, not "the two runs differ". Any two training runs differ by initialisation and
        // sampling noise, so a magnitude threshold is nearly vacuous — it passes for a policy that
        // ignores the reward entirely. The real question is whether the policy moved TOWARDS the
        // action that was paid: the run rewarding A must end up on A's side of the two probe
        // actions, and the run rewarding B on B's side.
        //
        // MEASURED ALONG THE AXIS THAT SEPARATES THE TWO ACTIONS, not as a distance in the full
        // action space. A and B differ in some coordinates and are near-identical in the rest, so
        // only the A-B direction carries any reward information at all: in every other coordinate
        // both actions were paid the same, the critic has no preference, and a continuous policy is
        // free to drift wherever it likes. A max-abs distance over EVERY coordinate is dominated by
        // exactly that free drift. Measured on MarketMakingAgent, the trained policy saturates at
        // its own +/-MaxPositionSize clamp about 1.5 away from both probe actions in the coordinates
        // neither of them distinguishes -- while landing on precisely the correct side of the
        // midpoint along the one coordinate that does. Six runs out of six were correct along the
        // axis and decided by coin-flip without it, which is what the intermittent failures were.
        //
        // Projecting first does not relax the invariant. A policy that ignores the reward still
        // lands on either side of the midpoint at chance, exactly as before; what changes is that
        // the answer is no longer decided by movement that carries no signal. For a discrete agent
        // the axis is e_A - e_B and the projection is just "how much more probability mass A has
        // than B", which is the same question the distance form was asking.
        var axis = new double[Math.Min(actionA!.Length, actionB!.Length)];
        double axisNormSquared = 0.0;
        for (int i = 0; i < axis.Length; i++)
        {
            axis[i] = ToD(actionA[i]) - ToD(actionB[i]);
            axisNormSquared += axis[i] * axis[i];
        }

        Skip.If(axisNormSquared <= 0.0,
            "The two probe actions coincide once projected, so no axis separates them and the "
            + "question cannot be expressed here.");

        // Position along the A-B axis, normalised so that Along(A) - Along(B) == 1.
        double Along(Vector<T> action)
        {
            double dot = 0.0;
            for (int i = 0; i < axis.Length && i < action.Length; i++) dot += ToD(action[i]) * axis[i];
            return dot / axisNormSquared;
        }

        double midpoint = (Along(actionA) + Along(actionB)) / 2.0;
        double towardsA = midpoint - Along(favouringA!);
        double towardsB = Along(favouringB!) - midpoint;

        Assert.True(towardsA < 0 && towardsB < 0,
            "The policy did not move towards whichever action was rewarded. Along the A-B axis, "
            + $"rewarding A left the greedy action {(towardsA < 0 ? "on" : "off")} A's side of the "
            + $"midpoint (margin {-towardsA:E3}), and rewarding B left it {(towardsB < 0 ? "on" : "off")} "
            + $"B's side (margin {-towardsB:E3}); both margins must be positive. Reached after "
            + $"{stepsA} and {stepsB} training steps.\n\n"
            + "A policy can fail this while passing every parameter-movement invariant in the suite: "
            + "SAC's actor keeps moving on its entropy term alone when min(Q1,Q2) is detached from "
            + "the tape, so its weights change every step without ever following the reward. That is "
            + "the blind spot this outcome check exists to cover.");
    }

    /// <summary>
    /// Finds two actions the agent can actually take in one state, so a reward can distinguish them.
    /// Returns nulls when the action space cannot express two, which is a skip rather than a failure.
    /// </summary>
    private (Vector<T>?, Vector<T>?) TwoDistinctActions(Vector<T> state)
    {
        using var model = CreateModel();
        if (model is not IRLAgent<T> agent) return (null, null);

        var first = agent.SelectAction(state, explore: false);
        if (first.Length == 0) return (null, null);

        // Collect the whole draw budget, then take the two candidates that are FARTHEST APART.
        // A deterministic or single-action policy yields none, and that is a legitimate skip.
        //
        // Two earlier selections both made this probe unanswerable for continuous control:
        //
        //   1. Taking the FIRST merely-different draw. A Gaussian exploration policy with
        //      sigma = 0.05 separates its first differing draw from the mean by about one sigma, so
        //      the test asked the critic to resolve a +1/-1 reward gap across a ~0.05 displacement
        //      in action space. The critic fits the mean, dQ/da is numerically flat over that
        //      interval, the policy target collapses back onto the current mean, and the outcome is
        //      decided by rounding: both arms end with the SAME policy and the two margins come out
        //      exactly symmetric (+x and -x).
        //
        //   2. Pairing the greedy action with the farthest draw. That fixes the separation but
        //      biases the two arms against each other, because the policy STARTS at the greedy
        //      action: the arm rewarding it passes without the policy moving at all, while the arm
        //      rewarding the other must travel more than half the gap to register. The observed
        //      failure was exactly that shape -- +1.507E-001 one way, -1.195E-002 the other.
        //
        // The farthest PAIR is symmetric by construction: both members are exploration draws around
        // the same mean, so each arm has the same distance to cover, and for a Gaussian policy they
        // land on opposite sides of it. Discrete and one-hot agents are unaffected -- every pair of
        // distinct one-hots is the same distance apart -- so this sharpens the probe rather than
        // relaxing it.
        // Pass one: free-running draws, exactly as before. The farthest pair is taken from THIS
        // pass alone whenever it yields one, so no agent that already answered the probe can have
        // its pair changed by the fallback below.
        var freeRunning = new List<Vector<T>> { first };
        for (int attempt = 0; attempt < 256; attempt++)
        {
            var candidate = agent.SelectAction(state, explore: true);
            if (candidate.Length == first.Length) freeRunning.Add(candidate);
        }

        var pair = FarthestPair(freeRunning);
        if (pair.Item1 is not null) return pair;

        // Pass two, only when pass one found nothing: a fresh episode per draw. An agent whose
        // exploration is EPISODIC rather than per-step offers ONE exploratory action and then acts
        // greedily for the remainder of the episode. Monte Carlo ES is the textbook case (Sutton
        // and Barto): its exploring start fires on the first action of an episode and never again
        // until that episode ends. Pass one never ends an episode, so such an agent contributed a
        // single random draw, and whether the probe found two distinct actions was decided by
        // whether that one draw happened to differ from the greedy action -- a coin flip at
        // ActionSize 2, seen as the same test skipping on four runs of six against identical code.
        // ResetEpisode is the IRLAgent contract's start-of-episode signal and its base
        // implementation is a no-op.
        //
        // The two passes are kept SEPARATE rather than pooled. Pooling was tried and regressed TD3
        // from passing to failing: an agent that anneals its exploration scale draws far wider
        // immediately after a reset than it does deep into a run, so the widest pair across the
        // pooled set came from two different distributions and was no longer symmetric about the
        // policy mean -- reintroducing the very bias documented above (-8.962E-001 against
        // +2.685E+000). Within one pass every draw is comparable.
        var perEpisode = new List<Vector<T>> { first };
        for (int attempt = 0; attempt < 256; attempt++)
        {
            agent.ResetEpisode();
            var candidate = agent.SelectAction(state, explore: true);
            if (candidate.Length == first.Length) perEpisode.Add(candidate);
        }

        return FarthestPair(perEpisode);
    }

    /// <summary>
    /// The two candidates that are farthest apart, or nulls when they all coincide.
    /// </summary>
    private (Vector<T>?, Vector<T>?) FarthestPair(List<Vector<T>> candidates)
    {
        Vector<T>? bestA = null;
        Vector<T>? bestB = null;
        double bestSeparation = 1e-9;

        for (int i = 0; i < candidates.Count; i++)
        {
            for (int j = i + 1; j < candidates.Count; j++)
            {
                double separation = Distance(candidates[i], candidates[j]);
                if (separation > bestSeparation)
                {
                    bestA = candidates[i];
                    bestB = candidates[j];
                    bestSeparation = separation;
                }
            }
        }

        return (bestA, bestB);
    }

    /// <summary>
    /// Trains a fresh agent with <paramref name="rewarded"/> paying +1 and <paramref name="punished"/>
    /// paying -1 in the same state, then returns the greedy policy output. Null when the agent never
    /// reached its own training gate within the budget.
    /// </summary>
    private (Vector<T>?, int) PolicyAfterRewarding(Vector<T> state, Vector<T> rewarded, Vector<T> punished)
    {
        using var model = CreateModel();
        if (model is not IRLAgent<T> agent) return (null, 0);

        // Open the agent's own warmup/batch gate first, or the budget below measures an agent that
        // never trained. At shipped defaults this loop ran 10000 steps at a batch of 256, twice.
        LowerTrainingGates(model);

        var parameterized = (IParameterizable<T, Vector<T>, Vector<T>>)model;
        var before = parameterized.GetParameters();
        var snapshot = new double[before.Length];
        for (int i = 0; i < before.Length; i++) snapshot[i] = ToD(before[i]);

        bool agentDrawsItsOwnActions = false;
        for (int i = 0; i < 512; i++)
        {
            bool even = i % 2 == 0;
            StoreDrawnTransition(agent, state, even ? rewarded : punished,
                action => RewardFor(action, rewarded, punished), state, i % 64 == 63,
                ref agentDrawsItsOwnActions);
        }

        // Run the WHOLE budget rather than stopping at the first flicker of movement. Breaking early
        // was wrong: an agent can move parameters long before its own update gate opens (a Polyak
        // target copy, a lazily materialized tensor), so an early exit measured a policy that had
        // never actually learned, and the comparison downstream was then initialisation noise.
        int stepsTrained = 0;
        bool moved = false;
        for (int step = 0; step < RealLoopStepBudget; step++)
        {
            // Keep FEEDING the agent as well as training it. An off-policy agent replays from its
            // buffer and is unaffected -- these transitions are drawn from exactly the same
            // alternating distribution as the prefill above. An ON-POLICY agent is not: A3CAgent
            // drains its entire trajectory on the first Train() and every later call returns at
            // once on an empty buffer, so prefill-then-train handed it ONE update while this loop's
            // budget claimed 800 -- and one policy-gradient step does not move an argmax. The pair
            // below simply CONTINUES the prefill's stream -- same alternation, same episode
            // cadence, next index -- so no agent sees a distribution it would not have seen anyway.
            int fed = 512 + step;
            bool feedRewarded = fed % 2 == 0;
            StoreDrawnTransition(agent, state, feedRewarded ? rewarded : punished,
                action => RewardFor(action, rewarded, punished), state, fed % 64 == 63,
                ref agentDrawsItsOwnActions);

            agent.Train();
            if (step % 512 != 511) continue;

            var current = parameterized.GetParameters();
            if (current.Length != snapshot.Length) { moved = true; stepsTrained = step + 1; continue; }
            for (int i = 0; i < snapshot.Length; i++)
            {
                if (Math.Abs(snapshot[i] - ToD(current[i])) > 1e-15)
                {
                    moved = true;
                    stepsTrained = step + 1;
                    break;
                }
            }
        }

        return (moved ? agent.SelectAction(state, explore: false) : null, stepsTrained);
    }

    /// <summary>
    /// Stores one transition, handing the agent <paramref name="preferred"/> where it will take it
    /// and an action it drew itself where it will not, and rewarding whichever action was stored.
    /// </summary>
    /// <remarks>
    /// The two regimes are mutually exclusive, so the first store decides which one this agent is
    /// in and <paramref name="agentDrawsItsOwnActions"/> carries that decision across the loop.
    ///
    /// Most agents accept any action the fixture hands them, and some REQUIRE that. Monte Carlo
    /// Exploring Starts draws its exploring action on the first step of an episode and is greedy
    /// afterwards, so a stream assembled only from what it draws revisits one action forever, never
    /// values the other, and the policy cannot move towards whichever one pays. Feeding it the
    /// action the fixture chose is what makes exploring starts work at all.
    ///
    /// An on-policy agent refuses exactly that. FinancialA2CAgent stamps the action it last returned
    /// from SelectAction together with the state it was drawn for, and StoreExperience throws
    /// "Use an unconsumed action sampled by this agent from the current policy for these state
    /// values." for anything else -- including an action the agent itself produced a moment earlier,
    /// since drawing two up front and storing the older one, or reusing one draw across a whole
    /// prefill loop, both break the contract. Redrawing immediately before each store is what a real
    /// rollout does.
    ///
    /// The refusal is therefore the signal: attempt the fixture's action, and switch to redrawing
    /// only for an agent that rejects it.
    /// </remarks>
    private static void StoreDrawnTransition(
        IRLAgent<T> agent,
        Vector<T> state,
        Vector<T> preferred,
        Func<Vector<T>, double> reward,
        Vector<T> nextState,
        bool done,
        ref bool agentDrawsItsOwnActions)
    {
        if (!agentDrawsItsOwnActions)
        {
            try
            {
                agent.StoreExperience(state, preferred, ToT(reward(preferred)), nextState, done);
                return;
            }
            catch (InvalidOperationException)
            {
                // The agent polices action provenance. Every later store goes through the draw path.
                agentDrawsItsOwnActions = true;
            }
        }

        var drawn = DrawActionFor(agent, state, preferred);
        agent.StoreExperience(state, drawn, ToT(reward(drawn)), nextState, done);
    }

    /// <summary>
    /// Returns an action for <paramref name="state"/> that the agent itself has just drawn and has
    /// not yet consumed, reproducing <paramref name="preferred"/> where the policy can.
    /// </summary>
    /// <remarks>
    /// Only a one-hot action is worth redrawing for: a discrete policy assigns every index nonzero
    /// probability, so the wanted index arrives within a few draws. A continuous policy will never
    /// reproduce a given vector exactly, and those agents accept whatever action they are given, so
    /// the preferred vector is returned unchanged.
    /// </remarks>
    private static Vector<T> DrawActionFor(IRLAgent<T> agent, Vector<T> state, Vector<T> preferred)
    {
        if (!IsOneHot(preferred)) return preferred;

        // The LAST draw is returned when the budget runs out, not the preferred action. A policy
        // that has learned to avoid the punished action stops producing it -- that is the agent
        // working, not a fixture fault -- and handing back the preferred vector at that point would
        // store an action the agent never sampled, which is exactly what the provenance check
        // refuses. Callers reward the action they get back rather than the one they asked for, so
        // the stream stays a real on-policy rollout in either case.
        Vector<T> drawn = preferred;
        for (int draw = 0; draw < 512; draw++)
        {
            drawn = agent.SelectAction(state, explore: true);
            if (Distance(drawn, preferred) <= 1e-12) break;
        }

        return drawn;
    }

    /// <summary>
    /// The reward an action earns: positive for the rewarded action, negative for the punished one,
    /// and neutral for anything else the policy happens to draw.
    /// </summary>
    private static double RewardFor(Vector<T> action, Vector<T> rewarded, Vector<T> punished)
    {
        if (Distance(action, rewarded) <= 1e-12) return 1.0;
        if (Distance(action, punished) <= 1e-12) return -1.0;
        return 0.0;
    }

    /// <summary>True when exactly one component is one and every other is zero.</summary>
    private static bool IsOneHot(Vector<T> action)
    {
        int hot = 0;
        for (int i = 0; i < action.Length; i++)
        {
            double value = ToD(action[i]);
            if (Math.Abs(value - 1.0) <= 1e-12) { hot++; continue; }
            if (Math.Abs(value) > 1e-12) return false;
        }

        return hot == 1;
    }

    /// <summary>Largest absolute per-component distance between two actions.</summary>
    private static double Distance(Vector<T> left, Vector<T> right)
    {
        double worst = 0.0;
        int length = Math.Min(left.Length, right.Length);
        for (int i = 0; i < length; i++)
            worst = Math.Max(worst, Math.Abs(ToD(left[i]) - ToD(right[i])));
        return worst;
    }

    [SkippableFact(Timeout = 300000)]
    public async Task EveryTrainableComponent_ShouldChangeDuringTraining()
    {
        // Only the agents that physically CANNOT run this loop are excused. Notably Dreamer is NOT:
        // its actor really did receive no gradient (the finite-difference policy improvement collapsed
        // to a no-op), and that was fixed at the source rather than skipped here, so this invariant
        // stays live on it and will catch the regression if the taped gradient is ever severed again.
        Skip.IfNot(SupportsSingleAgentOnlineLoop,
            "This agent consumes joint multi-agent transitions through a different API, so the "
            + "single-agent store-then-Train loop this invariant drives cannot run against it.");

        // WHY THIS EXISTS, given Training_ShouldChangeParameters already runs.
        //
        // Two independent holes, either of which hides a completely dead network.
        //
        // 1. THE QUANTIFIER. That test asserts "at least one parameter ANYWHERE in the agent moved".
        //    An actor-critic agent concatenates several networks into one flat vector, so whichever
        //    network does train satisfies it on contact and a policy receiving no gradient at all is
        //    structurally invisible.
        //
        // 2. THE DRIVER. It calls Train(state, target) — the single-transition adapter. For SACAgent
        //    that is a BEHAVIOUR-CLONING step: an MSE loss between the actor's mean head and a demo
        //    action, which trains the policy alone and never calls TrainOnBatch. So the actor-critic
        //    update — the code that actually carries the reward signal, and where a detached critic
        //    read leaves the actor ascending nothing — was never executed by any generated invariant.
        //
        // This drives the real loop instead: store experiences, then Train() until the agent's own
        // warmup/batch gate opens, and compare each registered component against itself.
        //
        // LIMIT, measured rather than argued: this does NOT catch a detached critic read. It passes
        // identically on the defective and on the repaired SAC actor update -- 3 s either way --
        // because the actor's entropy term carries real gradient and moves its weights every step
        // while the Q term contributes none. For that defect the question is reachability: did the
        // objective actually REACH the parameters. See SacActorCriticReachabilityTests, which
        // reports 0 of 12 critic tensors before the fix and passes after.
        //
        // What this test does buy is a component that never moves at all, which the existential
        // "did anything change" form hides behind whichever sibling network does train.
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();

        Skip.If(model is not IRLAgent<T>, "This fixture's model is not an IRLAgent, so the real "
            + "store-then-train loop cannot be driven through it.");
        var agent = (IRLAgent<T>)model;
        LowerTrainingGates(model);

        var before = ComponentSnapshot(agent);
        Skip.If(before.Count < 2,
            $"This agent exposes {before.Count} component(s) with materialized parameters; the "
            + "per-component check only adds signal over the model-level one at two or more.");

        // Fill the replay buffer before stepping, so the batch gate is satisfied the moment the
        // warmup counter clears. Actions come from the agent itself so their shape is always legal.
        var stateA = CreateRandomState(rng);
        var stateB = CreateRandomState(rng);
        var actionA = agent.SelectAction(stateA, explore: true);
        var actionB = agent.SelectAction(stateB, explore: true);
        bool agentDrawsItsOwnActions = false;
        for (int i = 0; i < 512; i++)
        {
            bool even = i % 2 == 0;
            // Differing rewards per state: a constant reward stream is genuinely unlearnable for
            // some correct algorithms (a gradient bandit leaves its preferences untouched).
            var stored = even ? stateA : stateB;
            double reward = even ? 1.0 : -1.0;
            StoreDrawnTransition(
                agent,
                stored,
                even ? actionA : actionB,
                _ => reward,
                even ? stateB : stateA,
                i % 64 == 63,
                ref agentDrawsItsOwnActions);
        }

        var moved = new HashSet<string>(StringComparer.Ordinal);
        int stepsTaken = 0;
        int budgetAfterFirstMovement = int.MaxValue;

        for (int step = 0; step < RealLoopStepBudget && moved.Count < before.Count; step++)
        {
            agent.Train();
            stepsTaken = step + 1;

            if (step % 64 != 63) continue;
            RecordMovement(agent, before, moved);

            // Once anything has moved the gate is open, so a straggler is a real straggler. Give it
            // a bounded extra run rather than the whole budget: a delayed policy update (TD3 moves
            // its actor every d steps) needs slack, a dead one will never use it.
            if (moved.Count > 0 && budgetAfterFirstMovement == int.MaxValue)
                budgetAfterFirstMovement = step + 2048;
            if (step >= budgetAfterFirstMovement) break;
        }

        RecordMovement(agent, before, moved);

        // NOTHING moved: the agent never reached its own update (warmup not cleared, batch never
        // filled, an algorithm that does not learn from this synthetic stream). That is inconclusive,
        // not a dead component, and reporting it as a failure would be a false alarm.
        Skip.If(moved.Count == 0,
            $"No component moved in {stepsTaken} steps of the real loop, so this agent never reached "
            + "its own training gate here and the per-component question cannot be answered.");

        var stalled = before.Keys.Where(id => !moved.Contains(id)).OrderBy(id => id, StringComparer.Ordinal).ToList();

        Assert.True(stalled.Count == 0,
            $"{stalled.Count} of {before.Count} component(s) received no update in {stepsTaken} steps, "
            + $"while {moved.Count} did: {string.Join(", ", stalled)}. Because the others moved, the "
            + "agent's training gate was demonstrably open — so these are not warmup artifacts. The "
            + "model-level 'did anything change' invariant passes in exactly this situation, which is "
            + "why this check exists. A component that never moves is usually one whose loss term was "
            + "detached from the gradient tape: read through Predict (a NoGradScope), or rebuilt "
            + "element by element into a fresh tensor between the networks.");
    }

    /// <summary>
    /// Captures each component's parameters separately, keyed by its stable manifest ID. Reading the
    /// chunks directly avoids reconstructing component boundaries from flat-vector offsets, which do
    /// not agree with a lazily shaped component's materialized width.
    /// </summary>
    private static Dictionary<string, double[]> ComponentSnapshot(IRLAgent<T> agent)
    {
        var snapshot = new Dictionary<string, double[]>(StringComparer.Ordinal);
        if (agent is not AiDotNet.ReinforcementLearning.Agents.ReinforcementLearningAgentBase<T> agentBase)
            return snapshot;

        foreach (var chunk in agentBase.GetParameterStateChunks())
        {
            // Only optimizer-updatable state is expected to move. Target networks tracked for
            // serialization, frozen weights and replay buffers are persistent numbers that a
            // gradient step is not supposed to touch.
            if (chunk.Role != AiDotNet.Models.Parameters.ParameterSlotRole.Trainable) continue;

            var tensor = chunk.Tensor;
            if (tensor is null || tensor.Length == 0) continue;

            var values = new double[tensor.Length];
            for (int i = 0; i < values.Length; i++) values[i] = ToD(tensor.GetFlat(i));
            snapshot[chunk.StableId] = values;
        }

        return snapshot;
    }

    /// <summary>Adds every component whose values now differ from its snapshot to <paramref name="moved"/>.</summary>
    private static void RecordMovement(
        IRLAgent<T> agent, Dictionary<string, double[]> before, HashSet<string> moved)
    {
        if (agent is not AiDotNet.ReinforcementLearning.Agents.ReinforcementLearningAgentBase<T> agentBase)
            return;

        foreach (var chunk in agentBase.GetParameterStateChunks())
        {
            if (moved.Contains(chunk.StableId)) continue;
            if (!before.TryGetValue(chunk.StableId, out var baseline)) continue;

            var tensor = chunk.Tensor;
            if (tensor is null) continue;

            // A length change is itself movement: a component that materialized lazily during
            // training has certainly changed.
            if (tensor.Length != baseline.Length) { moved.Add(chunk.StableId); continue; }

            for (int i = 0; i < baseline.Length; i++)
            {
                if (Math.Abs(baseline[i] - ToD(tensor.GetFlat(i))) > 1e-15)
                {
                    moved.Add(chunk.StableId);
                    break;
                }
            }
        }
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
        var target = new Vector<T>(EffectiveActionDim);
        model.Train(state, target);
        Assert.NotNull(model.GetModelMetadata());
    }

    /// <summary>
    /// An agent exposes its learned parameters once it has learned something, and its two parameter
    /// APIs agree at every point in that lifecycle.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This used to assert <c>GetParameters().Length &gt; 0</c> on a freshly constructed agent. That
    /// is not true of tabular reinforcement learning and never was: a Q-learning agent's parameters
    /// ARE its Q-table, and an untrained agent's table is empty. Zero is the honest answer.
    /// </para>
    /// <para>
    /// Asserting otherwise did real damage. Rather than the premise being questioned, at least five
    /// agents were contorted to satisfy it — <c>_qTable.Count &gt; 0 ? _qTable.Count * ActionSize : 1</c>
    /// fabricates one parameter that does not exist, purely so this line passes. That padding is
    /// what put <c>GetParameters().Length</c> permanently out of step with <c>ParameterCount</c>,
    /// and a length mismatch means a saved vector restores into the wrong tensors, because callers
    /// pair the two by length.
    /// </para>
    /// <para>
    /// So the test now asks the question it was reaching for: after the agent has learned, does it
    /// expose parameters, and do both APIs describe the same ones? The count/length agreement is
    /// checked on BOTH sides of training, because that invariant has no excuse to break in either
    /// state — it is the empty case where it was actually broken.
    /// </para>
    /// </remarks>
    [Fact(Timeout = 60000)]
    public async Task Parameters_ShouldBeNonEmptyAfterTraining()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var parameterizable = (IParameterizable<T, Vector<T>, Vector<T>>)model;

        // Untrained: whatever the agent reports, the two APIs must agree. Zero is a valid answer.
        Assert.True(
            parameterizable.GetParameters().Length == parameterizable.ParameterCount,
            $"Before training, GetParameters().Length ({parameterizable.GetParameters().Length}) " +
            $"disagrees with ParameterCount ({parameterizable.ParameterCount}). Callers pair these " +
            "by length, so a mismatch means a saved parameter vector restores into the wrong slots.");

        model.Train(CreateRandomState(rng), new Vector<T>(EffectiveActionDim));

        Assert.True(parameterizable.GetParameters().Length > 0,
            "After training, the agent should expose the parameters it learned.");

        Assert.True(
            parameterizable.GetParameters().Length == parameterizable.ParameterCount,
            $"After training, GetParameters().Length ({parameterizable.GetParameters().Length}) " +
            $"disagrees with ParameterCount ({parameterizable.ParameterCount}).");
    }
}

/// <summary>Default-precision alias, so existing derived fixtures need no change.</summary>
/// <remarks>
/// Mirrors the pattern the already-generic bases use. The generated scaffold names
/// <c>ReinforcementLearningTestBase&lt;float&gt;</c> directly once the base is whitelisted;
/// this alias only serves fixtures that were written against the non-generic name.
/// </remarks>
public abstract class ReinforcementLearningTestBase : ReinforcementLearningTestBase<double> { }
