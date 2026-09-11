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
/// Family invariants for classifier meta-learners: prototypical, matching and relation networks, ANIL and BOIL,
/// the memory-augmented learners and the rest that classify each query example into one of the task's ways
/// (#2139).
/// </summary>
/// <remarks>
/// <para>
/// These learners work on one embedding per example and emit one score per class for each query example. The
/// scalar-per-row family base could not host them: it gave them a model with a single output per row and
/// compared their per-class scores against a vector of labels, so they failed on shapes instead of on anything
/// they compute - and several treated a whole batch's output as one example's features.
/// </para>
/// <para>
/// <b>The base owns the composition.</b> The inner model is a <see cref="LinearEmbeddingModel"/> mapping
/// <see cref="FeatureCount"/> features to an <see cref="EmbeddingDimension"/>-wide embedding per row. Tasks carry
/// class indices as a <c>[rows]</c> tensor. The generated fixture binds the options' class count to
/// <see cref="NumWays"/> and their embedding width to <see cref="EmbeddingDimension"/>.
/// </para>
/// <para>
/// The invariants hold for any classifier, whatever its paper: a meta-training step is finite and moves the
/// meta-model (unless the paper freezes it); adaptation returns one finite score row per query example,
/// <c>[queryRows, ways]</c>, and leaves the meta-model as it was; evaluation is finite; and a deep copy starts
/// equal and shares nothing.
/// </para>
/// </remarks>
public abstract class ClassificationMetaLearnerTestBase
{
    /// <summary>Features per example, the inner model's input width.</summary>
    protected virtual int FeatureCount => 3;

    /// <summary>Width of each example's embedding, the inner model's output width.</summary>
    protected virtual int EmbeddingDimension => 4;

    /// <summary>Classes per task; the targets are class indices.</summary>
    protected virtual int NumWays => 2;

    /// <summary>
    /// True for a learner whose paper keeps the meta-model frozen and meta-trains other state instead.
    /// </summary>
    protected virtual bool MetaModelFrozenByDesign => false;

    private const int ShotsPerClass = 2;
    private const int QueriesPerClass = 2;

    private int SupportRows => NumWays * ShotsPerClass;
    private int QueryRows => NumWays * QueriesPerClass;

    /// <summary>Subclasses build the learner around exactly the inner model they are handed.</summary>
    protected abstract IMetaLearner<double, Matrix<double>, Tensor<double>> CreateLearner(
        IFullModel<double, Matrix<double>, Tensor<double>> innerModel);

    private IMetaLearner<double, Matrix<double>, Tensor<double>> CreateLearner()
        => CreateLearner(new LinearEmbeddingModel(FeatureCount, EmbeddingDimension));

    /// <summary>
    /// A task whose classes are separable: each class's examples sit around its own centre, so a classifier has
    /// something to learn.
    /// </summary>
    private MetaLearningTask<double, Matrix<double>, Tensor<double>> CreateTask(int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var centres = Enumerable.Range(0, NumWays)
            .Select(_ => Enumerable.Range(0, FeatureCount).Select(_ => rng.NextDouble() * 2.0 - 1.0).ToArray())
            .ToArray();

        (Matrix<double> X, Tensor<double> Y) Rows(int perClass)
        {
            int rows = NumWays * perClass;
            var x = new Matrix<double>(rows, FeatureCount);
            var y = new Tensor<double>(new[] { rows });
            for (int i = 0; i < rows; i++)
            {
                int label = i % NumWays;
                for (int j = 0; j < FeatureCount; j++) x[i, j] = centres[label][j] + 0.1 * (rng.NextDouble() - 0.5);
                y[i] = label;
            }

            return (x, y);
        }

        var support = Rows(ShotsPerClass);
        var query = Rows(QueriesPerClass);
        return new MetaLearningTask<double, Matrix<double>, Tensor<double>>
        {
            SupportSetX = support.X,
            SupportSetY = support.Y,
            QuerySetX = query.X,
            QuerySetY = query.Y,
            NumWays = NumWays,
            NumShots = ShotsPerClass,
            NumQueryPerClass = QueriesPerClass,
            Name = $"classification-task-{seed}",
        };
    }

    private TaskBatch<double, Matrix<double>, Tensor<double>> CreateBatch(int seed, int count = 2)
        => new TaskBatch<double, Matrix<double>, Tensor<double>>(
            Enumerable.Range(0, count).Select(i => CreateTask(seed + i)).ToArray());

    private static double[] MetaParameters(IMetaLearner<double, Matrix<double>, Tensor<double>> learner)
    {
        var parameters = learner.GetMetaModel().GetParameters();
        return Enumerable.Range(0, parameters.Length).Select(i => parameters[i]).ToArray();
    }

    /// <summary>The learner's parameters and declared state, as its own serializer writes them.</summary>
    private static byte[] LearnedState(IMetaLearner<double, Matrix<double>, Tensor<double>> learner)
    {
        var serializer = Assert.IsAssignableFrom<IModelSerializer>(learner);
        using (AiDotNet.Helpers.ModelPersistenceGuard.InternalOperation())
        {
            return serializer.Serialize();
        }
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
            Assert.Equal(before, after);
            Assert.NotEqual(stateBefore, LearnedState(learner));
            return;
        }

        Assert.True(before.Length != after.Length || before.Zip(after, (a, b) => a != b).Any(changed => changed),
            "Two meta-training steps left every meta-model parameter where it was.");
    }

    [Fact(Timeout = 120000)]
    public async Task Adapt_ScoresEveryClassForEachQueryExample()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var learner = CreateLearner();
        var task = CreateTask(seed: 5);

        var scores = learner.Adapt(task).Predict(task.QuerySetX);

        Assert.Equal(new[] { QueryRows, NumWays }, scores.Shape.ToArray());
        for (int i = 0; i < scores.Length; i++) AssertFinite(scores[i], $"Score {i}");
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

    [Fact(Timeout = 120000)]
    public async Task Evaluate_ReturnsAFiniteLoss()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var learner = CreateLearner();

        AssertFinite(learner.Evaluate(CreateBatch(seed: 7, count: 3)), "The evaluation loss");
    }

    [Fact(Timeout = 120000)]
    public async Task Clone_IsIndependentAndEquivalent()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var learner = CreateLearner();
        learner.MetaTrain(CreateBatch(seed: 8));
        var parameters = MetaParameters(learner);
        var state = LearnedState(learner);

        var model = Assert.IsAssignableFrom<IFullModel<double, Matrix<double>, Tensor<double>>>(learner);
        var copy = Assert.IsAssignableFrom<IMetaLearner<double, Matrix<double>, Tensor<double>>>(model.DeepCopy());
        Assert.NotSame(learner.GetMetaModel(), copy.GetMetaModel());
        Assert.Equal(parameters, MetaParameters(copy));
        Assert.Equal(state, LearnedState(copy));

        copy.MetaTrain(CreateBatch(seed: 9));

        Assert.Equal(parameters, MetaParameters(learner));
        Assert.Equal(state, LearnedState(learner));
    }
}
