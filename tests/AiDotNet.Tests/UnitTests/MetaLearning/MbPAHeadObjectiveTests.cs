using AiDotNet.Data.Structures;
using AiDotNet.Enums;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Models;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

public class MbPAHeadObjectiveTests
{
    [Fact]
    public void ConfidentlyWrongCategoricalPredictionRetainsItsLossAndGradient()
    {
        var loss = new MbPAHeadLoss<double>(new Vector<double>([1000.0, -1000.0, 0.0, 0.0]),
            1, 2, MbPAOutputDistribution.Categorical);
        var features = new Tensor<double>([1, 1], new Vector<double>([1.0]));
        var target = new Tensor<double>([1, 2], new Vector<double>([0.0, 1.0]));
        Assert.Equal(2000.0, loss.CalculateLoss(features.ToVector(), target.ToVector()), 10);
        using var tape = new GradientTape<double>();
        var objective = loss.ComputeTapeLoss(features, target);
        Assert.Equal(2000.0, objective[0], 10);
        var gradient = tape.ComputeGradients(objective, new[] { features })[features];
        Assert.Equal(2000.0, gradient[0], 10);
        Assert.Equal(gradient[0], loss.CalculateDerivative(features.ToVector(), target.ToVector())[0], 10);
    }

    [Theory]
    [InlineData(MbPAOutputDistribution.Categorical)]
    [InlineData(MbPAOutputDistribution.Gaussian)]
    public void TapeObjectiveAndGradientMatchTheComposedHead(MbPAOutputDistribution distribution)
    {
        var loss = new MbPAHeadLoss<double>(new Vector<double>([0.2, -0.3, 0.4, 0.1, -0.2, 0.3]),
            2, 2, distribution);
        var features = new Tensor<double>([2, 2], new Vector<double>([0.5, -0.2, -0.4, 0.7]));
        var target = new Tensor<double>([2, 2], new Vector<double>([1.0, 0.0, 0.0, 1.0]));
        double expected = loss.CalculateLoss(features.ToVector(), target.ToVector());
        using var tape = new GradientTape<double>();
        var objective = loss.ComputeTapeLoss(features, target);
        Assert.Equal(expected, objective[0], 12);
        var gradient = tape.ComputeGradients(objective, new[] { features })[features];
        var expectedGradient = loss.CalculateDerivative(features.ToVector(), target.ToVector());
        for (int i = 0; i < features.Length; i++) Assert.Equal(expectedGradient[i], gradient[i], 10);
        const double epsilon = 1e-6;
        var flat = features.ToVector();
        for (int i = 0; i < flat.Length; i++)
        {
            double original = flat[i];
            flat[i] = original + epsilon;
            double plus = loss.CalculateLoss(flat, target.ToVector());
            flat[i] = original - epsilon;
            double minus = loss.CalculateLoss(flat, target.ToVector());
            flat[i] = original;
            Assert.Equal((plus - minus) / (2 * epsilon), gradient[i], 8);
        }
    }

    [Fact]
    public void ClassIndexTargetsMatchOneHotInBothLossPaths()
    {
        var loss = new MbPAHeadLoss<double>(new Vector<double>([0.2, -0.3, 0.4, 0.1, -0.2, 0.3]),
            2, 2, MbPAOutputDistribution.Categorical);
        var features = new Tensor<double>([2, 2], new Vector<double>([0.5, -0.2, -0.4, 0.7]));
        var indices = new Tensor<double>([2], new Vector<double>([0.0, 1.0]));
        var oneHot = new Tensor<double>([2, 2], new Vector<double>([1.0, 0.0, 0.0, 1.0]));
        Assert.Equal(loss.CalculateLoss(features.ToVector(), oneHot.ToVector()),
            loss.CalculateLoss(features.ToVector(), indices.ToVector()), 12);
        Assert.Equal(loss.ComputeTapeLoss(features, oneHot)[0], loss.ComputeTapeLoss(features, indices)[0], 12);
    }

    [Fact]
    public void ClassIndicesAndOneHotTargetsProduceTheSameAdaptation()
    {
        var inputs = new Matrix<double>(new double[,] { { -1.0, 0.2 }, { 0.7, 1.0 } });
        Tensor<double> AdaptAndPredict(Tensor<double> labels)
        {
            var learner = new MbPAAlgorithm<double, Matrix<double>, Tensor<double>>(
                new MbPAOptions<double, Matrix<double>, Tensor<double>>(new LinearEmbeddingModel(2, 2))
                { FeatureDimension = 2, OutputDimension = 2, RandomSeed = 1337 });
            var task = new MetaLearningTask<double, Matrix<double>, Tensor<double>>
            {
                SupportSetX = inputs, SupportSetY = labels, QuerySetX = inputs, QuerySetY = labels,
                NumWays = 2, NumShots = 1, NumQueryPerClass = 1
            };
            return learner.Adapt(task).Predict(inputs);
        }
        var indices = AdaptAndPredict(new Tensor<double>([2], new Vector<double>([0.0, 1.0])));
        var oneHot = AdaptAndPredict(new Tensor<double>([2, 2], new Vector<double>([1.0, 0.0, 0.0, 1.0])));
        Assert.Equal(oneHot.ToArray(), indices.ToArray());
    }
}
