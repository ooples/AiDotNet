using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.Data.Structures;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Family invariants for the adapted models a meta-learner's <c>Adapt</c> returns (<c>MetaLearningModelBase</c>):
/// <c>AdaptedMetaModel</c>, <c>MbPAAdaptedModel</c> and <c>NeuralProcessModel</c> (#2139).
/// </summary>
/// <remarks>
/// <para>
/// These wrap a meta-model with task-specific state, so nothing can construct one except adaptation, and the family
/// was excluded from generation. Each fixture here builds the meta-learner that produces its model, adapts it to a
/// seeded task, and checks the result as a model in its own right.
/// </para>
/// <para>
/// The invariants: predictions are finite and deterministic; the parameters are live (changing them changes the
/// predictions) and round-trip through <c>GetParameters</c>, <c>SetParameters</c> and <c>WithParameters</c>; a deep
/// copy is equivalent and independent; changing or using an adapted model leaves the meta-learner and its other
/// adapted models untouched; adaptation depends on the task (two tasks give two different models); and
/// <c>Train</c> is refused, since an adapted model is the output of adaptation, not something to train further.
/// </para>
/// </remarks>
/// <typeparam name="TOutput">The meta-learner's output type.</typeparam>
public abstract class AdaptedModelTestBase<TOutput>
{
    /// <summary>The adapted model type the fixture's meta-learner must return.</summary>
    protected abstract Type ExpectedType { get; }

    /// <summary>Builds the meta-learner that produces the adapted model.</summary>
    protected abstract IMetaLearner<double, Matrix<double>, TOutput> CreateLearner();

    /// <summary>A seeded task for the learner.</summary>
    protected abstract MetaLearningTask<double, Matrix<double>, TOutput> CreateTask(int seed);

    /// <summary>The prediction's values, flattened.</summary>
    protected abstract double[] Values(TOutput output);

    private IFullModel<double, Matrix<double>, TOutput> Adapt(int seed, out Matrix<double> query)
        => Adapt(CreateLearner(), seed, out query);

    private IFullModel<double, Matrix<double>, TOutput> Adapt(
        IMetaLearner<double, Matrix<double>, TOutput> learner, int seed, out Matrix<double> query)
    {
        var task = CreateTask(seed);
        query = task.QuerySetX;
        var adapted = learner.Adapt(task);
        Assert.IsType(ExpectedType, adapted);
        return Assert.IsAssignableFrom<IFullModel<double, Matrix<double>, TOutput>>(adapted);
    }

    private static void AssertSame(double[] expected, double[] actual, string what)
    {
        Assert.True(expected.Length == actual.Length, $"{what}: {expected.Length} values, then {actual.Length}.");
        for (int i = 0; i < expected.Length; i++)
            Assert.True(expected[i] == actual[i], $"{what}: value {i} was {expected[i]}, then {actual[i]}.");
    }

    private static double Distance(double[] a, double[] b)
    {
        double sum = 0;
        for (int i = 0; i < Math.Min(a.Length, b.Length); i++) sum += (a[i] - b[i]) * (a[i] - b[i]);
        return Math.Sqrt(sum);
    }

    private static Vector<double> Perturbed(Vector<double> parameters)
    {
        var copy = new Vector<double>(parameters.Length);
        for (int i = 0; i < parameters.Length; i++) copy[i] = parameters[i] + (0.25 * ((i % 3) - 1)) + 0.1;
        return copy;
    }

    [Fact(Timeout = 120000)]
    public async Task Predict_IsFiniteNonEmptyAndDeterministic()
    {
        await Task.Yield();
        var adapted = Adapt(11, out var query);
        var first = Values(adapted.Predict(query));
        Assert.NotEmpty(first);
        Assert.All(first, v => Assert.False(double.IsNaN(v) || double.IsInfinity(v), $"prediction {v} is not finite"));
        AssertSame(first, Values(adapted.Predict(query)), "a second Predict on the same query");
    }

    [Fact(Timeout = 120000)]
    public async Task Parameters_RoundTrip()
    {
        await Task.Yield();
        var adapted = Adapt(12, out var query);
        var before = Values(adapted.Predict(query));
        var parameters = adapted.GetParameters();
        Assert.True(parameters.Length > 0, "an adapted model with no parameters cannot carry its adaptation");

        AssertSame(before, Values(adapted.WithParameters(parameters).Predict(query)), "WithParameters(GetParameters())");
        adapted.SetParameters(parameters);
        AssertSame(before, Values(adapted.Predict(query)), "SetParameters(GetParameters())");

        // A no-op setter would pass the identity round trip above, so apply different values and read them back.
        var perturbed = Perturbed(parameters);
        var viaWith = Values(adapted.WithParameters(perturbed).Predict(query));
        adapted.SetParameters(perturbed);
        AssertSame(perturbed.ToArray(), adapted.GetParameters().ToArray(), "GetParameters after SetParameters(perturbed)");
        var viaSet = Values(adapted.Predict(query));
        Assert.True(Distance(before, viaSet) > 1e-9,
            $"{ExpectedType.Name}: SetParameters with every value changed left the predictions as they were.");
        AssertSame(viaWith, viaSet, "SetParameters(p) against WithParameters(p)");
    }

    [Fact(Timeout = 120000)]
    public async Task Parameters_AreLive()
    {
        await Task.Yield();
        var adapted = Adapt(13, out var query);
        var before = Values(adapted.Predict(query));
        var changed = Values(adapted.WithParameters(Perturbed(adapted.GetParameters())).Predict(query));
        Assert.True(Distance(before, changed) > 1e-9,
            $"{ExpectedType.Name}: changing every parameter did not change the predictions; its parameters are dead.");
    }

    [Fact(Timeout = 120000)]
    public async Task DeepCopy_IsEquivalentAndIndependent()
    {
        await Task.Yield();
        var adapted = Adapt(14, out var query);
        var before = Values(adapted.Predict(query));
        var copy = adapted.DeepCopy();
        AssertSame(before, Values(copy.Predict(query)), "the deep copy");

        copy.SetParameters(Perturbed(copy.GetParameters()));
        AssertSame(before, Values(adapted.Predict(query)), "the original after its copy's parameters changed");
    }

    [Fact(Timeout = 120000)]
    public async Task ChangingTheAdaptedModel_LeavesTheMetaLearnerUnchanged()
    {
        await Task.Yield();
        // Adapt both models first: some learners (MbPA) add each task's support set to a memory by design, so
        // adapting again would change the answer for a reason that has nothing to do with this invariant.
        var learner = CreateLearner();
        var adapted = Adapt(learner, 18, out var query);
        var sibling = Adapt(learner, 19, out _);
        var siblingBefore = Values(sibling.Predict(query));
        var meta = learner.GetMetaModel().GetParameters().ToArray();

        adapted.SetParameters(Perturbed(adapted.GetParameters()));
        adapted.Predict(query);

        Assert.Equal(meta, learner.GetMetaModel().GetParameters().ToArray());
        AssertSame(siblingBefore, Values(sibling.Predict(query)),
            "another model adapted from the same learner, after the first one's parameters changed");
    }

    [Fact(Timeout = 120000)]
    public async Task Adaptation_DependsOnTheTask()
    {
        await Task.Yield();
        var first = Adapt(15, out var query);
        var second = Adapt(16, out _);
        double distance = Distance(Values(first.Predict(query)), Values(second.Predict(query)));
        Assert.True(distance > 1e-9,
            $"{ExpectedType.Name}: models adapted to two different tasks predict the same query identically; adaptation ignored the task.");
    }

    [Fact(Timeout = 120000)]
    public async Task Train_IsRefused()
    {
        await Task.Yield();
        var task = CreateTask(17);
        var adapted = Adapt(17, out _);
        Assert.Throws<NotSupportedException>(() => adapted.Train(task.SupportSetX, task.SupportSetY));
    }
}