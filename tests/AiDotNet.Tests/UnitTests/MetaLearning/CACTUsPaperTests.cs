using System;
using System.Linq;
using AiDotNet.Data.Structures;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using AiDotNet.Tests.IntegrationTests.MetaLearning;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Reference tests for <see cref="UnsupervisedMetaLearnAlgorithm{T, TInput, TOutput}"/> against CACTUs (Hsu, Levine and
/// Finn, "Unsupervised Learning via Meta-Learning", ICLR 2019): pseudo-tasks come from k-means clusters of the
/// unlabeled inputs, and the given labels are never read during meta-training.
/// </summary>
public class CACTUsPaperTests
{
    // Two tight, far-apart groups of four points each: rows 0-3 near (0, 0), rows 4-7 near (10, 10).
    private static readonly double[,] Points =
    {
        { 0.0, 0.1 }, { 0.1, 0.0 }, { -0.1, 0.0 }, { 0.0, -0.1 },
        { 10.0, 10.1 }, { 10.1, 10.0 }, { 9.9, 10.0 }, { 10.0, 9.9 },
    };

    private static UnsupervisedMetaLearnAlgorithm<double, Matrix<double>, Vector<double>> Algorithm()
        => new(new UnsupervisedMetaLearnOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(2))
        {
            Seed = 17,
        });

    [Fact]
    public void KMeans_RecoversWellSeparatedGroups()
    {
        var assignment = Algorithm().KMeans(Tensor<double>.FromMatrix(new Matrix<double>(Points)), 2);

        Assert.All(assignment.Take(4), a => Assert.Equal(assignment[0], a));
        Assert.All(assignment.Skip(4), a => Assert.Equal(assignment[4], a));
        Assert.NotEqual(assignment[0], assignment[4]);
    }

    [Fact]
    public void PseudoTask_LabelsByCluster_AndIgnoresTheGivenLabels()
    {
        var pool = new Matrix<double>(Points);
        // Deliberately wrong labels: if they leaked into the pseudo-task, classes would mix the two groups.
        var slot = new MetaLearningTask<double, Matrix<double>, Vector<double>>
        {
            SupportSetX = pool, SupportSetY = new Vector<double>(new[] { 0.0, 1, 0, 1, 0, 1, 0, 1 }),
            QuerySetX = pool, QuerySetY = new Vector<double>(new[] { 0.0, 1, 0, 1, 0, 1, 0, 1 }),
            NumWays = 2, NumShots = 1, NumQueryPerClass = 2,
        };

        var task = Algorithm().BuildPseudoTask(pool, Tensor<double>.FromMatrix(pool), slot);

        Assert.True(task.HasValue);
        var (sx, sy, qx, qy) = task.Value;
        Assert.Equal(2, sx.Rows);
        Assert.Equal(4, qx.Rows);
        for (int i = 0; i < qx.Rows; i++)
        {
            // A query example's pseudo-label is the class of the support example from its own group.
            int match = (qx[i, 0] > 5) == (sx[0, 0] > 5) ? 0 : 1;
            Assert.Equal(sy[match], qy[i]);
        }
        Assert.NotEqual(sy[0], sy[1]);
    }

    [Fact]
    public void MetaTrain_RunsOnPseudoTasks()
    {
        var model = new LinearVectorModel(2);
        var algorithm = new UnsupervisedMetaLearnAlgorithm<double, Matrix<double>, Vector<double>>(
            new UnsupervisedMetaLearnOptions<double, Matrix<double>, Vector<double>>(model) { Seed = 4 });
        var pool = new Matrix<double>(Points);
        var labels = new Vector<double>(new[] { 0.0, 0, 0, 0, 1, 1, 1, 1 });
        var task = new MetaLearningTask<double, Matrix<double>, Vector<double>>
        {
            SupportSetX = pool, SupportSetY = labels, QuerySetX = pool, QuerySetY = labels,
            NumWays = 2, NumShots = 1, NumQueryPerClass = 2,
        };

        double loss = algorithm.MetaTrain(new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task }));

        Assert.False(double.IsNaN(loss) || double.IsInfinity(loss));
    }

    [Fact]
    public void KMeans_SeparatesLargeMagnitudeGroupsInHalf()
    {
        // Groups near (300, 300) and (-300, -300). In Half ||c||^2 and x . c (about 180000) overflow to inf, so
        // every distance was inf or inf - inf = NaN and the arg-min put every point in cluster 0.
        float[] offsets = { 0f, 2f, -2f, 0f };
        float[] other = { 2f, 0f, 0f, -2f };
        var data = new Half[16];
        for (int i = 0; i < 4; i++)
        {
            data[2 * i] = (Half)(300f + offsets[i]);
            data[2 * i + 1] = (Half)(300f + other[i]);
            data[8 + 2 * i] = (Half)(-300f - offsets[i]);
            data[8 + 2 * i + 1] = (Half)(-300f - other[i]);
        }

        var algorithm = new UnsupervisedMetaLearnAlgorithm<Half, Tensor<Half>, Tensor<Half>>(
            new UnsupervisedMetaLearnOptions<Half, Tensor<Half>, Tensor<Half>>(new MockFullModel<Half>()) { Seed = 17 });
        var assignment = algorithm.KMeans(Tensor<Half>.FromVector(new Vector<Half>(data), new[] { 8, 2 }), 2);

        Assert.All(assignment.Take(4), a => Assert.Equal(assignment[0], a));
        Assert.All(assignment.Skip(4), a => Assert.Equal(assignment[4], a));
        Assert.NotEqual(assignment[0], assignment[4]);
    }
}
