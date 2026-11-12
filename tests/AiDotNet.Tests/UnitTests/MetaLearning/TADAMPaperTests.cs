using System;
using AiDotNet.Data.Structures;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.IntegrationTests.MetaLearning;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Reference tests for <see cref="TADAMAlgorithm{T, TInput, TOutput}"/> against Oreshkin, Rodriguez and Lacoste,
/// "TADAM: Task dependent adaptive metric for improved few-shot learning" (NeurIPS 2018): p(y = k | x) =
/// softmax(-alpha d(f(x), c_k)) with one learned scalar alpha (Eq. 3), and a task-embedding network whose
/// L2-penalized post-multiplier starts at zero (Sec. 2.2).
/// </summary>
public class TADAMPaperTests
{
    private static MetaLearningTask<double, Matrix<double>, Tensor<double>> Task()
        => new()
        {
            SupportSetX = new Matrix<double>(new[,] { { 0.5, -1.0 }, { 1.0, 0.5 }, { -0.5, 0.25 }, { 0.75, -0.25 } }),
            SupportSetY = new Tensor<double>(new[] { 0.0, 1.0, 0.0, 1.0 }, new[] { 4 }),
            QuerySetX = new Matrix<double>(new[,] { { -0.25, 0.75 }, { 0.5, 0.5 }, { 1.0, -1.0 } }),
            QuerySetY = new Tensor<double>(new[] { 1.0, 1.0, 0.0 }, new[] { 3 }),
            NumWays = 2, NumShots = 2, NumQueryPerClass = 1,
        };

    private static TADAMAlgorithm<double, Matrix<double>, Tensor<double>> Algorithm(bool conditioning)
        => new(new TADAMOptions<double, Matrix<double>, Tensor<double>>(new TensorEmbeddingModel(2, 2))
        {
            NumClasses = 2,
            EmbeddingDimension = 2,
            TaskEmbeddingDimension = 3,
            NormalizeEmbeddings = false,
            UseTaskConditioning = conditioning,
            GradientClipThreshold = null,
            Seed = 11,
        });

    /// <summary>softmax(-alpha ||f(q) - c_k||^2) per query row, with c_k the class means of the support features.</summary>
    private static double[,] Reference(double alpha)
    {
        var task = Task();
        var model = new TensorEmbeddingModel(2, 2);
        var support = model.Predict(task.SupportSetX);
        var query = model.Predict(task.QuerySetX);
        var prototypes = new double[2, 2];
        for (int s = 0; s < 4; s++)
            for (int d = 0; d < 2; d++) prototypes[(int)task.SupportSetY[s], d] += support[new[] { s, d }] / 2.0;

        var result = new double[3, 2];
        for (int q = 0; q < 3; q++)
        {
            double total = 0.0;
            for (int k = 0; k < 2; k++)
            {
                double distance = 0.0;
                for (int d = 0; d < 2; d++) distance += Math.Pow(query[new[] { q, d }] - prototypes[k, d], 2);
                result[q, k] = Math.Exp(-alpha * distance);
                total += result[q, k];
            }

            for (int k = 0; k < 2; k++) result[q, k] /= total;
        }

        return result;
    }

    [Fact]
    public void Defaults_PenalizeThePostMultiplierAsThePaperDoes()
    {
        var options = new TADAMOptions<double, Matrix<double>, Tensor<double>>(new TensorEmbeddingModel(2, 2));
        Assert.Equal(0.01, options.L2Regularization);
    }

    [Fact]
    public void Adapt_ScoresWithTheScaledEuclideanMetric()
    {
        var algorithm = Algorithm(conditioning: false);
        algorithm.MetricScaleForTesting = new Vector<double>(new[] { 2.5 });

        var probabilities = algorithm.Adapt(Task()).Predict(Task().QuerySetX);
        var expected = Reference(2.5);

        for (int q = 0; q < 3; q++)
            for (int k = 0; k < 2; k++) Assert.Equal(expected[q, k], probabilities[new[] { q, k }], 10);
    }

    [Fact]
    public void EpisodeLoss_IsTheQueryCrossEntropyOfTheMetric()
    {
        var algorithm = Algorithm(conditioning: false);
        algorithm.MetricScaleForTesting = new Vector<double>(new[] { 2.5 });
        var expected = Reference(2.5);
        var labels = Task().QuerySetY;

        double loss = 0.0;
        for (int q = 0; q < 3; q++) loss -= Math.Log(expected[q, (int)labels[q]]);

        Assert.Equal(loss / 3.0, algorithm.EpisodeLossForTesting(Task()), 8);
    }

    [Fact]
    public void UntrainedTaskEncoder_ConditionsNothing()
    {
        // gamma0 starts at 0, so the conditioned metric is exactly the unconditioned one.
        var conditioned = Algorithm(conditioning: true);
        var plain = Algorithm(conditioning: false);
        Assert.Equal(plain.EpisodeLossForTesting(Task()), conditioned.EpisodeLossForTesting(Task()), 12);
    }

    [Fact]
    public void MetricAndTaskEncoderGradients_MatchFiniteDifferences()
    {
        var algorithm = Algorithm(conditioning: true);
        algorithm.EpisodeLossForTesting(Task()); // sizes the task encoder
        var encoder = algorithm.TaskEncoderForTesting;
        encoder[encoder.Length - 1] = 0.4; // gamma0 != 0, so every TEN weight reaches the loss
        algorithm.TaskEncoderForTesting = encoder;
        algorithm.MetricScaleForTesting = new Vector<double>(new[] { 1.7 });

        var (alphaGradient, encoderGradient) = algorithm.MetricGradientForTesting(Task());

        const double h = 1e-6;
        algorithm.MetricScaleForTesting = new Vector<double>(new[] { 1.7 + h });
        double up = algorithm.EpisodeLossForTesting(Task());
        algorithm.MetricScaleForTesting = new Vector<double>(new[] { 1.7 - h });
        double down = algorithm.EpisodeLossForTesting(Task());
        algorithm.MetricScaleForTesting = new Vector<double>(new[] { 1.7 });
        Assert.Equal((up - down) / (2 * h), alphaGradient[0], 6);

        Assert.Equal(encoder.Length, encoderGradient.Length);
        foreach (int i in new[] { 0, 7, encoder.Length - 1 })
        {
            var plus = encoder.Clone();
            plus[i] += h;
            algorithm.TaskEncoderForTesting = plus;
            double lossUp = algorithm.EpisodeLossForTesting(Task());
            var minus = encoder.Clone();
            minus[i] -= h;
            algorithm.TaskEncoderForTesting = minus;
            double lossDown = algorithm.EpisodeLossForTesting(Task());
            Assert.Equal((lossUp - lossDown) / (2 * h), encoderGradient[i], 6);
        }
    }
}
