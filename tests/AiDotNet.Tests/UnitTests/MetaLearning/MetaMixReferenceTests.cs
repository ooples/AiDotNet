using AiDotNet.Data.Structures;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.IntegrationTests.MetaLearning;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Reference tests for <see cref="MetaTaskAlgorithm{T, TInput, TOutput}"/> against MetaMix
/// (Chen et al., "MetaMix: Improved Meta-Learning with Interpolation-based Consistency Regularization",
/// ICPR 2020, arXiv:2009.13735), Eq. 4-5 and Algorithm 1.
/// </summary>
public class MetaMixReferenceTests
{
    private static MetaLearningTask<double, Matrix<double>, Vector<double>> Task()
    {
        // Support values are in the hundreds, query values below 10, so a mixed row shows which set it came from.
        var sx = new Matrix<double>(new[,] { { 100.0, 101.0 }, { 102.0, 103.0 } });
        var sy = new Vector<double>(new[] { 200.0, 201.0 });
        var qx = new Matrix<double>(new[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 } });
        var qy = new Vector<double>(new[] { 7.0, 8.0, 9.0 });
        return new MetaLearningTask<double, Matrix<double>, Vector<double>>
        {
            SupportSetX = sx, SupportSetY = sy, QuerySetX = qx, QuerySetY = qy,
            NumWays = 1, NumShots = 2, NumQueryPerClass = 3,
        };
    }

    private static MetaTaskAlgorithm<double, Matrix<double>, Vector<double>> Algorithm(MetaMixSource source)
        => new(new MetaTaskOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(2))
        {
            MixSource = source,
            Seed = 5,
        });

    [Fact]
    public void Defaults_FollowThePaper()
    {
        // Sec. IV-B: α̌ = 1 (λ uniform); Table III: query-only mixing; Z defaults to the query size.
        var options = new MetaTaskOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(2));
        Assert.Equal(1.0, options.InterpolationAlpha);
        Assert.Equal(MetaMixSource.Query, options.MixSource);
        Assert.Equal(0, options.NumMixedPairs);
    }

    [Fact]
    public void MixedSet_IsEq4And5_OnQueryPairs()
    {
        var algorithm = Algorithm(MetaMixSource.Query);
        double lambda = 0.25;
        var (x, y) = algorithm.BuildMixedSet(Task(), new[] { 0, 2 }, new[] { 1, 0 }, lambda);

        // x̂ = λ x_m + (1-λ) x_n, ŷ = λ y_m + (1-λ) y_n with (m, n) = (0, 1) and (2, 0).
        Assert.Equal(2, x.Rows);
        Assert.Equal(0.25 * 1 + 0.75 * 3, x[0, 0], 12);
        Assert.Equal(0.25 * 2 + 0.75 * 4, x[0, 1], 12);
        Assert.Equal(0.25 * 5 + 0.75 * 1, x[1, 0], 12);
        Assert.Equal(0.25 * 6 + 0.75 * 2, x[1, 1], 12);
        Assert.Equal(0.25 * 7 + 0.75 * 8, y[0], 12);
        Assert.Equal(0.25 * 9 + 0.75 * 7, y[1], 12);
    }

    [Fact]
    public void SupportAndQuery_TakesTheFirstExampleFromTheSupportSet()
    {
        var algorithm = Algorithm(MetaMixSource.SupportAndQuery);
        var (x, y) = algorithm.BuildMixedSet(Task(), new[] { 1 }, new[] { 2 }, 0.5);

        Assert.Equal(0.5 * 102 + 0.5 * 5, x[0, 0], 12);
        Assert.Equal(0.5 * 103 + 0.5 * 6, x[0, 1], 12);
        Assert.Equal(0.5 * 201 + 0.5 * 9, y[0], 12);
    }

    [Fact]
    public void MetaTrain_ReturnsTheMixedQueryLoss()
    {
        // The outer objective is the loss on the mixed set only (Eq. 7); it must be finite and update θ.
        var model = new LinearVectorModel(2);
        var algorithm = new MetaTaskAlgorithm<double, Matrix<double>, Vector<double>>(
            new MetaTaskOptions<double, Matrix<double>, Vector<double>>(model) { Seed = 3, OuterLearningRate = 0.01 });
        var before = model.GetParameters().Clone();

        double loss = algorithm.MetaTrain(new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { Task() }));

        Assert.False(double.IsNaN(loss) || double.IsInfinity(loss));
        Assert.NotEqual(before[0], model.GetParameters()[0]);
    }
}
