using AiDotNet.Data.Structures;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.IntegrationTests.MetaLearning;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Reference tests for <see cref="WarpGradAlgorithm{T, TInput, TOutput}"/> against Flennerhag et al., "Meta-Learning
/// with Warped Gradient Descent" (ICLR 2020): the warp meta-objective is J(P) = sum_k L_query(theta_k - alpha P g_k)
/// over the task's adaptation trajectory with every theta_k held fixed, so it never backpropagates through the inner loop.
/// </summary>
public class WarpGradPaperTests
{
    private const double Alpha = 0.1;
    private const int Steps = 2;
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

    private static WarpGradAlgorithm<double, Matrix<double>, Vector<double>> Algorithm()
    {
        var model = new LinearVectorModel(3);
        model.SetParameters(new Vector<double>(Theta));

        // Two warp layers of ONE shared parameter each: layer 0 scales coordinates 0-1, layer 1 scales 2-3.
        return new WarpGradAlgorithm<double, Matrix<double>, Vector<double>>(
            new WarpGradOptions<double, Matrix<double>, Vector<double>>(model)
            {
                LossFunction = new MeanSquaredErrorLoss<double>(),
                GradientClipThreshold = null,
                InnerLearningRate = Alpha,
                AdaptationSteps = Steps,
                NumWarpLayers = 2,
                UseDiagonalWarp = false,
                WarpLayerHiddenDim = 1,
                WarpInitScale = 0.2,
                Seed = 5,
            });
    }

    private static double[] Scale(double p0, double p1) => new[] { p0, p0, p1, p1 };

    private static Vector<double> Gradient(Vector<double> theta, Matrix<double> x, Vector<double> y)
    {
        var model = new LinearVectorModel(3);
        model.SetParameters(theta);
        return model.ComputeGradients(x, y, new MeanSquaredErrorLoss<double>());
    }

    private static double QueryLoss(Vector<double> theta)
    {
        var model = new LinearVectorModel(3);
        model.SetParameters(theta);
        var task = Task();
        return new MeanSquaredErrorLoss<double>().CalculateLoss(model.Predict(task.QuerySetX), task.QuerySetY);
    }

    private static Vector<double> Step(Vector<double> theta, Vector<double> g, double[] scale)
    {
        var next = new Vector<double>(theta.Length);
        for (int i = 0; i < theta.Length; i++) next[i] = theta[i] - (Alpha * scale[i] * g[i]);
        return next;
    }

    /// <summary>The trajectory points theta_0..theta_{K-1} and support gradients there, under preconditioner P.</summary>
    private static (Vector<double>[] Points, Vector<double>[] Gradients) Trajectory(double[] scale)
    {
        var task = Task();
        var points = new Vector<double>[Steps];
        var grads = new Vector<double>[Steps];
        var theta = new Vector<double>(Theta);
        for (int k = 0; k < Steps; k++)
        {
            points[k] = theta;
            grads[k] = Gradient(theta, task.SupportSetX, task.SupportSetY);
            theta = Step(theta, grads[k], scale);
        }

        return (points, grads);
    }

    /// <summary>J(P') with the trajectory frozen at the one P generated.</summary>
    private static double Objective((Vector<double>[] Points, Vector<double>[] Gradients) frozen, double[] scale)
    {
        double total = 0.0;
        for (int k = 0; k < Steps; k++) total += QueryLoss(Step(frozen.Points[k], frozen.Gradients[k], scale));
        return total;
    }

    [Fact]
    public void WarpGradient_IsTheGradientOfTheTrajectoryObjective()
    {
        var algorithm = Algorithm();
        double p0 = algorithm.WarpLayerParameters[0][0];
        double p1 = algorithm.WarpLayerParameters[1][0];
        var frozen = Trajectory(Scale(p0, p1));

        var warpGrads = algorithm.ComputeWarpGradients(Task(), new Vector<double>(Theta));

        const double h = 1e-6;
        double d0 = (Objective(frozen, Scale(p0 + h, p1)) - Objective(frozen, Scale(p0 - h, p1))) / (2 * h);
        double d1 = (Objective(frozen, Scale(p0, p1 + h)) - Objective(frozen, Scale(p0, p1 - h))) / (2 * h);

        Assert.Equal(2, warpGrads.Count);
        Assert.Single(warpGrads[0]);
        Assert.Single(warpGrads[1]);
        Assert.Equal(d0, warpGrads[0][0], 6);
        Assert.Equal(d1, warpGrads[1][0], 6);
    }

    [Fact]
    public void Adapt_StepsWithThePreconditionedGradient()
    {
        var algorithm = Algorithm();
        var scale = Scale(algorithm.WarpLayerParameters[0][0], algorithm.WarpLayerParameters[1][0]);
        var frozen = Trajectory(scale);
        var expected = Step(frozen.Points[Steps - 1], frozen.Gradients[Steps - 1], scale);

        var adapted = algorithm.Adapt(Task());
        var task = Task();
        var prediction = adapted.Predict(task.QuerySetX);

        var reference = new LinearVectorModel(3);
        reference.SetParameters(expected);
        var expectedPrediction = reference.Predict(task.QuerySetX);
        Assert.Equal(3, prediction.Length);
        Assert.Equal(expectedPrediction.Length, prediction.Length);
        for (int i = 0; i < prediction.Length; i++)
            Assert.Equal(expectedPrediction[i], prediction[i], 10);
    }
}
