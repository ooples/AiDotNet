using System;
using System.Linq;
using AiDotNet.AutoML;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Member = AiDotNet.MetaLearning.Models.LinearVectorModel;

namespace AiDotNet.Tests.UnitTests.AutoML;

/// <summary>
/// The ensemble AutoML returns from its best trials: a weighted average for regression and a weighted vote for
/// classification.
/// </summary>
public class AutoMLEnsembleModelTests
{
    private static readonly Matrix<double> Input = new(new double[,] { { 1.0 }, { 2.0 } });

    // A member whose prediction is the constant c for every row (zero weight, bias c).
    private static IFullModel<double, Matrix<double>, Vector<double>> Constant(double c)
    {
        var member = new Member(1);
        member.SetParameters(new Vector<double>(new[] { 0.0, c }));
        return member;
    }

    [Fact]
    public void Regression_IsTheNormalisedWeightedAverage()
    {
        var ensemble = new AutoMLEnsembleModel<double>(
            new[] { Constant(1.0), Constant(4.0) }, PredictionType.Regression, new[] { 1.0, 3.0 });

        var y = ensemble.Predict(Input);

        Assert.Equal(new[] { 0.25, 0.75 }, ensemble.Weights);
        Assert.All(y.ToArray(), v => Assert.Equal(0.25 * 1.0 + 0.75 * 4.0, v, 12));
    }

    [Fact]
    public void MultiClass_IsTheWeightedVote()
    {
        // Two members say class 2, one heavier member says class 5: the weight decides, not the head count.
        var ensemble = new AutoMLEnsembleModel<double>(
            new[] { Constant(2.0), Constant(2.0), Constant(5.0) }, PredictionType.MultiClass, new[] { 1.0, 1.0, 3.0 });

        Assert.All(ensemble.Predict(Input).ToArray(), v => Assert.Equal(5.0, v));
    }

    [Fact]
    public void WithoutWeights_MembersCountEqually()
    {
        var ensemble = new AutoMLEnsembleModel<double>(
            new[] { Constant(2.0), Constant(2.0), Constant(5.0) }, PredictionType.MultiClass);

        Assert.All(ensemble.Predict(Input).ToArray(), v => Assert.Equal(2.0, v));
        Assert.All(ensemble.Weights, w => Assert.Equal(1.0 / 3.0, w, 12));
    }

    [Fact]
    public void InvalidEnsembles_AreRejected()
    {
        Assert.Throws<ArgumentException>(() => new AutoMLEnsembleModel<double>(
            Enumerable.Empty<IFullModel<double, Matrix<double>, Vector<double>>>(), PredictionType.Regression));
        Assert.Throws<ArgumentException>(() => new AutoMLEnsembleModel<double>(
            new[] { Constant(1.0), Constant(2.0) }, PredictionType.Regression, new[] { 1.0 }));
        Assert.Throws<InvalidOperationException>(() => new AutoMLEnsembleModel<double>().Predict(Input));
    }
}
