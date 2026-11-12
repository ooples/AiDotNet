using AiDotNet.Data.Structures;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.IntegrationTests.MetaLearning;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Reference tests for <see cref="OMLAlgorithm{T, TInput, TOutput}"/> against Javed and White,
/// "Meta-Learning Representations for Continual Learning" (NeurIPS 2019), Algorithm 1.
/// </summary>
public class OMLPaperTests
{
    private static readonly double[] Theta = { 0.3, -0.2, 0.1, 0.05 };

    private static MetaLearningTask<double, Matrix<double>, Vector<double>> Task()
        => new()
        {
            SupportSetX = new Matrix<double>(new[,] { { 0.5, -1.0, 0.25 }, { 1.0, 0.5, -0.5 } }),
            SupportSetY = new Vector<double>(new[] { 1.0, -0.5 }),
            QuerySetX = new Matrix<double>(new[,] { { -0.25, 0.75, 1.0 }, { 0.5, 0.5, 0.5 }, { 1.0, 0.0, 1.0 } }),
            QuerySetY = new Vector<double>(new[] { 0.5, 1.5, -1.0 }),
            NumWays = 1, NumShots = 2, NumQueryPerClass = 3,
        };

    private static OMLAlgorithm<double, Matrix<double>, Vector<double>> Algorithm(LinearVectorModel model)
    {
        model.SetParameters(new Vector<double>(Theta));
        return new OMLAlgorithm<double, Matrix<double>, Vector<double>>(
            new OMLOptions<double, Matrix<double>, Vector<double>>(model)
            {
                LossFunction = new MeanSquaredErrorLoss<double>(),
                GradientClipThreshold = null,
                InnerLearningRate = 0.1,
                PlnFraction = 0.5,
            });
    }

    /// <summary>Algorithm 1's online inner loop: one SGD step per support example, on the PLN (last half) only.</summary>
    private static Vector<double> ReferenceAdapted()
    {
        var reference = new LinearVectorModel(3);
        reference.SetParameters(new Vector<double>(Theta));
        var loss = new MeanSquaredErrorLoss<double>();
        var task = Task();
        for (int j = 0; j < 2; j++)
        {
            var x = new Matrix<double>(1, 3);
            for (int c = 0; c < 3; c++) x[0, c] = task.SupportSetX[j, c];
            var g = reference.ComputeGradients(x, new Vector<double>(new[] { task.SupportSetY[j] }), loss);
            var p = reference.GetParameters();
            p[2] -= 0.1 * g[2];
            p[3] -= 0.1 * g[3];
            reference.SetParameters(p);
        }

        return reference.GetParameters();
    }

    [Fact]
    public void Defaults_FollowThePaper()
    {
        var options = new OMLOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(3));
        Assert.True(options.OnlineInnerLoop);
        Assert.Equal(1, options.AdaptationSteps);
        Assert.Equal(0.0, options.SparsityPenalty);
    }

    [Fact]
    public void InnerLoop_IsOnline_AndTouchesOnlyThePredictionNetwork()
    {
        var algorithm = Algorithm(new LinearVectorModel(3));
        var adapted = ((AdaptedMetaModel<double, Matrix<double>, Vector<double>>)algorithm.Adapt(Task())).GetParameters();
        var expected = ReferenceAdapted();

        Assert.Equal(Theta[0], adapted[0], 12);
        Assert.Equal(Theta[1], adapted[1], 12);
        Assert.Equal(expected[2], adapted[2], 10);
        Assert.Equal(expected[3], adapted[3], 10);
    }

    [Fact]
    public void MetaLoss_IsTheMeanOverTrajectoryAndRandomSamples()
    {
        // L(f(S_traj U X_rand; theta, W_k)) as a per-example mean over the 2 support and 3 query examples.
        var reference = new LinearVectorModel(3);
        reference.SetParameters(ReferenceAdapted());
        var loss = new MeanSquaredErrorLoss<double>();
        var task = Task();
        double expected = (2 * loss.CalculateLoss(reference.Predict(task.SupportSetX), task.SupportSetY)
                         + 3 * loss.CalculateLoss(reference.Predict(task.QuerySetX), task.QuerySetY)) / 5;

        double actual = Algorithm(new LinearVectorModel(3))
            .MetaTrain(new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task }));

        Assert.Equal(expected, actual, 10);
    }
}
