using System;
using System.Threading.Tasks;
using AiDotNet.Data.Structures;
using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Family invariants for a model that a scalar meta-learner's <c>Adapt</c> returns (#2139).
/// </summary>
/// <remarks>
/// An adapted model has no constructor a fixture could call with anything real: it is the result of adapting a
/// meta-model to one task. So the base builds it the way a user gets one. The fixture names a meta-learner that
/// produces it, the base adapts that learner to an episodic task, and the invariants run on what comes back. The
/// learner's own invariants run too, since the composition is the same one the meta-learner family tests.
/// </remarks>
public abstract class AdaptedMetaModelTestBase : MetaLearnerTestBase
{
    /// <summary>The adapted model under test, as an open generic type definition.</summary>
    protected abstract Type AdaptedModelType { get; }

    private (IFullModel<double, Matrix<double>, Vector<double>> Model, MetaLearningTask<double, Matrix<double>, Vector<double>> Task) Adapt(int seed)
    {
        var task = CreateTask(seed);
        // Adapt returns an IModel; an adapted model is a full model, with parameters and a deep copy.
        var adapted = Assert.IsAssignableFrom<IFullModel<double, Matrix<double>, Vector<double>>>(CreateLearner().Adapt(task));
        return (adapted, task);
    }

    private static double[] Values(Vector<double> output) => output.ToArray();

    [Fact(Timeout = 120000)]
    public async Task Adapt_ReturnsTheModelUnderTest()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        AdaptedMetaModelChecks.ReturnsTheModelUnderTest(Adapt(seed: 11).Model, AdaptedModelType);
    }

    [Fact(Timeout = 120000)]
    public async Task AdaptedModel_PredictsDeterministically()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var (model, task) = Adapt(seed: 12);
        AdaptedMetaModelChecks.PredictsDeterministically(model, task.QuerySetX, Values);
    }

    [Fact(Timeout = 120000)]
    public async Task AdaptedModel_ParametersDescribeTheModel()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var (model, task) = Adapt(seed: 13);
        AdaptedMetaModelChecks.ParametersDescribeTheModel(model, task.QuerySetX, Values);
    }

    [Fact(Timeout = 120000)]
    public async Task AdaptedModel_DeepCopyIsIndependent()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var (model, task) = Adapt(seed: 14);
        AdaptedMetaModelChecks.DeepCopyIsIndependent(model, task.QuerySetX, Values);
    }

    [Fact(Timeout = 120000)]
    public async Task AdaptedModel_RefusesDirectTraining()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var (model, task) = Adapt(seed: 15);
        AdaptedMetaModelChecks.RefusesDirectTraining(model, task.QuerySetX, task.QuerySetY);
    }
}