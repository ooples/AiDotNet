using System;
using System.Collections.Generic;
using AiDotNet.KnowledgeDistillation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.KnowledgeDistillation;

/// <summary>
/// FitNets-style feature distillation (Romero et al. 2015): weighted MSE between paired teacher and student
/// features, averaged over the layer pairs, and its exact gradient.
/// </summary>
public class FeatureDistillationStrategyTests
{
    private static readonly Dictionary<string, double[]> Teacher = new()
    {
        ["t1"] = new[] { 1.0, 2.0, 3.0 },
        ["t2"] = new[] { -1.0, 0.5 },
    };

    private static Dictionary<string, double[]> Student() => new()
    {
        ["s1"] = new[] { 1.5, 1.0, 3.0 },
        ["s2"] = new[] { 0.0, 0.5 },
    };

    private static double Loss(FeatureDistillationStrategy<double> strategy, Dictionary<string, double[]> student) =>
        strategy.ComputeFeatureLoss(
            name => new Vector<double>(Teacher[name]),
            name => new Vector<double>(student[name]),
            new Vector<double>(1));

    [Fact]
    public void Loss_IsTheWeightedMeanOfThePairsMse()
    {
        var strategy = new FeatureDistillationStrategy<double>(new[] { "t1:s1", "t2:s2" }, featureWeight: 0.5);

        // Pair 1: (0.25 + 1 + 0) / 3; pair 2: (1 + 0) / 2; averaged over 2 pairs, times 0.5.
        double expected = 0.5 * ((1.25 / 3.0) + 0.5) / 2.0;
        Assert.Equal(expected, Loss(strategy, Student()), 12);
    }

    [Fact]
    public void Gradient_MatchesFiniteDifferencesOfTheLoss_WithSeveralPairs()
    {
        var strategy = new FeatureDistillationStrategy<double>(new[] { "t1:s1", "t2:s2" }, featureWeight: 0.7);
        var student = Student();
        var gradient = strategy.ComputeFeatureGradient(new Vector<double>(student["s1"]), new Vector<double>(Teacher["t1"]));

        const double h = 1e-6;
        for (int i = 0; i < student["s1"].Length; i++)
        {
            var plus = Student();
            plus["s1"][i] += h;
            var minus = Student();
            minus["s1"][i] -= h;
            double numeric = (Loss(strategy, plus) - Loss(strategy, minus)) / (2 * h);
            Assert.Equal(numeric, gradient[i], 8);
        }
    }

    [Theory]
    [InlineData("teacher")]
    [InlineData(":student")]
    [InlineData("a:b:c")]
    public void MalformedLayerPair_IsRejected(string pair)
    {
        Assert.Throws<ArgumentException>(() => new FeatureDistillationStrategy<double>(new[] { pair }));
    }

    [Fact]
    public void MismatchedFeatureWidths_AreRejected()
    {
        var strategy = new FeatureDistillationStrategy<double>(new[] { "t1:s2" });
        Assert.Throws<ArgumentException>(() => Loss(strategy, Student()));
    }
}
