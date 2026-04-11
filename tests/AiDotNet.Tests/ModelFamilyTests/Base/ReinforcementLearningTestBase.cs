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

    [Fact(Timeout = 60000)]
    public async Task ActionSelection_ShouldBeFinite()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
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
        var model = CreateModel();
        var state = CreateRandomState(rng);

        var action1 = model.Predict(state);
        var action2 = model.Predict(state);

        Assert.Equal(action1.Length, action2.Length);
        for (int i = 0; i < action1.Length; i++)
            Assert.Equal(action1[i], action2[i]);
    }

    [Fact(Timeout = 60000)]
    public async Task DifferentStates_DifferentActions()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var model = CreateModel();

        var state1 = new Vector<double>(StateDim);
        var state2 = new Vector<double>(StateDim);
        for (int i = 0; i < StateDim; i++)
        {
            state1[i] = 0.1;
            state2[i] = 0.9;
        }

        var action1 = model.Predict(state1);
        var action2 = model.Predict(state2);

        bool anyDifferent = false;
        int minLen = Math.Min(action1.Length, action2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(action1[i] - action2[i]) > 1e-12)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "RL agent produces identical actions for different states — policy is degenerate.");
    }

    [Fact(Timeout = 60000)]
    public async Task Training_ShouldChangeParameters()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();

        var paramsBefore = ((IParameterizable<double, Vector<double>, Vector<double>>)model).GetParameters();
        var snapshot = new double[paramsBefore.Length];
        for (int i = 0; i < paramsBefore.Length; i++) snapshot[i] = paramsBefore[i];

        var state = CreateRandomState(rng);
        var target = new Vector<double>(StateDim);
        for (int i = 0; i < StateDim; i++) target[i] = 1.0;
        for (int iter = 0; iter < 5; iter++)
            model.Train(state, target);

        var paramsAfter = ((IParameterizable<double, Vector<double>, Vector<double>>)model).GetParameters();
        bool anyChanged = false;
        for (int i = 0; i < Math.Min(snapshot.Length, paramsAfter.Length); i++)
        {
            if (Math.Abs(snapshot[i] - paramsAfter[i]) > 1e-15)
            {
                anyChanged = true;
                break;
            }
        }
        Assert.True(anyChanged, "RL agent parameters unchanged after training.");
    }

    [Fact(Timeout = 60000)]
    public async Task Clone_ShouldProduceSamePolicy()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
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
        var model = CreateModel();
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
        var model = CreateModel();
        Assert.True(((IParameterizable<double, Vector<double>, Vector<double>>)model).GetParameters().Length > 0, "RL agent should have parameters.");
    }
}
