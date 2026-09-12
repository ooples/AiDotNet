using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Data.Structures;
using AiDotNet.Interfaces;
using AiDotNet.MetaLearning.Models;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Family invariants for meta-learners: MAML and its variants, Reptile, the metric learners, the neural
/// processes and the rest (#2139).
/// </summary>
/// <remarks>
/// <para>
/// A meta-learner wraps an inner model its caller chooses, so no model family could build one, and every
/// meta-learner was excluded from generation: a hundred shipped algorithms ran none of the family invariants.
/// </para>
/// <para>
/// <b>The base owns the composition.</b> It builds the inner model, a <c>LinearVectorModel</c> over
/// <see cref="FeatureCount"/> features, and hands it to <see cref="CreateLearner(IFullModel{double, Matrix{double}, Vector{double}})"/>,
/// which the generated fixture implements as <c>new X(new XOptions(innerModel))</c>. The episodic tasks come from
/// the same feature count, so a learner, its inner model and its tasks cannot disagree. Matrix in, vector out,
/// class indices as targets is the contract the meta-learning integration tests already run these algorithms
/// under.
/// </para>
/// <para>
/// The invariants hold for any meta-learner, whatever its paper. A meta-training step is finite and moves the
/// meta-model. Adaptation returns a model that predicts the query set and leaves the meta-model as it was, since
/// adapted parameters are per task (Finn et al. 2017: theta'_i is computed from theta, and theta moves only in the
/// meta-update). Evaluation is finite.
/// </para>
/// </remarks>
public abstract class MetaLearnerTestBase
{
    /// <summary>Features per sample, shared by the inner model and every task.</summary>
    protected virtual int FeatureCount => 3;

    /// <summary>Classes per task; the targets are class indices.</summary>
    protected virtual int NumWays => 2;

    private const int SupportRows = 4;
    private const int QueryRows = 4;

    /// <summary>Subclasses build the learner around exactly the inner model they are handed.</summary>
    protected abstract IMetaLearner<double, Matrix<double>, Vector<double>> CreateLearner(
        IFullModel<double, Matrix<double>, Vector<double>> innerModel);

    private IMetaLearner<double, Matrix<double>, Vector<double>> CreateLearner()
        => CreateLearner(new LinearVectorModel(FeatureCount));

    private MetaLearningTask<double, Matrix<double>, Vector<double>> CreateTask(int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);

        Matrix<double> Features(int rows)
        {
            var features = new Matrix<double>(rows, FeatureCount);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < FeatureCount; j++) features[i, j] = rng.NextDouble() - 0.5;
            }

            return features;
        }

        Vector<double> Labels(int rows)
        {
            var labels = new Vector<double>(rows);
            for (int i = 0; i < rows; i++) labels[i] = i % NumWays;
            return labels;
        }

        var supportX = Features(SupportRows);
        var queryX = Features(QueryRows);
        return new MetaLearningTask<double, Matrix<double>, Vector<double>>
        {
            SupportSetX = supportX,
            SupportSetY = Labels(SupportRows),
            QuerySetX = queryX,
            QuerySetY = Labels(QueryRows),
            NumWays = NumWays,
            NumShots = Math.Max(1, SupportRows / NumWays),
            NumQueryPerClass = Math.Max(1, QueryRows / NumWays),
            Name = $"family-task-{seed}",
        };
    }

    private TaskBatch<double, Matrix<double>, Vector<double>> CreateBatch(int seed, int count = 2)
        => new TaskBatch<double, Matrix<double>, Vector<double>>(
            Enumerable.Range(0, count).Select(i => CreateTask(seed + i)).ToArray());

    private static double[] MetaParameters(IMetaLearner<double, Matrix<double>, Vector<double>> learner)
    {
        var parameters = learner.GetMetaModel().GetParameters();
        return Enumerable.Range(0, parameters.Length).Select(i => parameters[i]).ToArray();
    }

    private static void AssertFinite(double value, string what)
        => Assert.False(double.IsNaN(value) || double.IsInfinity(value), $"{what} is {value}.");

    [Fact(Timeout = 120000)]
    public async Task MetaTrain_ReturnsAFiniteLoss()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var learner = CreateLearner();

        AssertFinite(learner.MetaTrain(CreateBatch(seed: 1)), "The meta-training loss");
    }

    /// <summary>
    /// True for a learner whose paper keeps the meta-model frozen and meta-trains other state instead - CAML's
    /// pretrained feature extractor beside its context module (Fifty et al. 2024).
    /// </summary>
    protected virtual bool MetaModelFrozenByDesign => false;

    [Fact(Timeout = 120000)]
    public async Task MetaTrain_MovesTheMetaModel()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var learner = CreateLearner();
        var before = MetaParameters(learner);
        var stateBefore = LearnedState(learner);

        learner.MetaTrain(CreateBatch(seed: 2));
        learner.MetaTrain(CreateBatch(seed: 4));

        var after = MetaParameters(learner);
        if (MetaModelFrozenByDesign)
        {
            // Frozen means frozen, and the state the paper does meta-train has to move instead.
            Assert.Equal(before, after);
            Assert.NotEqual(stateBefore, LearnedState(learner));
            return;
        }

        Assert.True(before.Length != after.Length || before.Zip(after, (a, b) => a != b).Any(changed => changed),
            "Two meta-training steps left every meta-model parameter where it was.");
    }

    [Fact(Timeout = 120000)]
    public async Task Adapt_PredictsOneValuePerQueryRow()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var learner = CreateLearner();
        var task = CreateTask(seed: 5);

        var predictions = learner.Adapt(task).Predict(task.QuerySetX);

        Assert.Equal(QueryRows, predictions.Length);
        for (int i = 0; i < predictions.Length; i++) AssertFinite(predictions[i], $"Query prediction {i}");
    }

    [Fact(Timeout = 120000)]
    public async Task Adapt_LeavesTheMetaModelUnchanged()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var learner = CreateLearner();
        var before = MetaParameters(learner);

        learner.Adapt(CreateTask(seed: 6));

        Assert.Equal(before, MetaParameters(learner));
    }

    /// <summary>The learner's parameters and declared state, as its own serializer writes them.</summary>
    private static byte[] LearnedState(IMetaLearner<double, Matrix<double>, Vector<double>> learner)
    {
        var serializer = Assert.IsAssignableFrom<IModelSerializer>(learner);
        using (AiDotNet.Helpers.ModelPersistenceGuard.InternalOperation())
        {
            return serializer.Serialize();
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Clone_IsIndependentAndEquivalent()
    {
        // A copy starts where the original is and shares nothing with it: meta-training the copy must leave the
        // original's meta-model and learned state exactly as they were.
        await Task.Yield();
        using var arena = TensorArena.Create();
        var learner = CreateLearner();
        learner.MetaTrain(CreateBatch(seed: 8));
        var parameters = MetaParameters(learner);
        var state = LearnedState(learner);

        var model = Assert.IsAssignableFrom<IFullModel<double, Matrix<double>, Vector<double>>>(learner);
        var copy = Assert.IsAssignableFrom<IMetaLearner<double, Matrix<double>, Vector<double>>>(model.DeepCopy());
        Assert.NotSame(learner.GetMetaModel(), copy.GetMetaModel());
        Assert.Equal(parameters, MetaParameters(copy));
        Assert.Equal(state, LearnedState(copy));

        copy.MetaTrain(CreateBatch(seed: 9));

        Assert.Equal(parameters, MetaParameters(learner));
        Assert.Equal(state, LearnedState(learner));
    }

    [Fact(Timeout = 120000)]
    public async Task Evaluate_ReturnsAFiniteLoss()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var learner = CreateLearner();

        AssertFinite(learner.Evaluate(CreateBatch(seed: 7, count: 3)), "The evaluation loss");
    }
}
