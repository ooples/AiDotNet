using System;
using AiDotNet.Regression;
using AiDotNet.SurvivalAnalysis;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.SurvivalAnalysis;

/// <summary>
/// Covers <c>AiModelBuilder.Build(features, times, events)</c>, the three-argument overload that keeps
/// censoring out of the covariate matrix.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="SurvivalModelBase{T}"/> reads column 0 of its design matrix as the event indicator, which
/// works but leaves callers assembling that matrix by hand and one mistake away from a covariate being
/// read as censoring — the one error that changes the answer rather than throwing. This overload
/// assembles it instead, so the three signals stay three named arguments.
/// </para>
/// </remarks>
public class SurvivalBuildOverloadTests
{
    private const int Subjects = 8;

    private static Matrix<double> Features()
    {
        double[] ages = { 45, 52, 38, 61, 47, 55, 41, 58 };
        var m = new Matrix<double>(Subjects, 2);
        for (int i = 0; i < Subjects; i++)
        {
            m[i, 0] = ages[i];
            m[i, 1] = i % 2;
        }

        return m;
    }

    private static Vector<double> Times() =>
        new(new double[] { 5.0, 12.0, 3.0, 18.0, 9.0, 21.0, 7.0, 15.0 });

    private static Vector<int> Events() => new(new[] { 1, 0, 1, 0, 1, 0, 1, 0 });

    /// <summary>
    /// The overload's whole job: reach the same fit as the model's own three-argument entry point,
    /// while the caller passes three named arguments and never sees a design matrix.
    /// </summary>
    [Fact]
    public void ItFitsWhatTheExplicitThreeArgumentCallFits()
    {
        var features = Features();

        var viaBuilder = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new KaplanMeierEstimator<double>())
            .Build(features, Times(), Events());

        var direct = new KaplanMeierEstimator<double>();
        direct.FitSurvival(features, Times(), Events());

        var a = viaBuilder.Predict(features);
        var b = direct.Predict(features);

        Assert.Equal(b.Length, a.Length);
        for (int i = 0; i < b.Length; i++)
        {
            Assert.Equal(b[i], a[i]);
        }
    }

    /// <summary>
    /// The reason the overload hands the indicators over rather than folding them into X: prediction
    /// takes covariates. Whether the event occurred is the thing being predicted, so a caller could not
    /// supply a design matrix at predict time even if asked to.
    /// </summary>
    [Fact]
    public void PredictionTakesTheSameCovariatesTrainingDid()
    {
        var features = Features();

        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new KaplanMeierEstimator<double>())
            .Build(features, Times(), Events());

        var predictions = result.Predict(features);

        Assert.Equal(features.Rows, predictions.Length);
    }

    /// <summary>
    /// If the indicator were dropped on the way through, censoring would stop mattering.
    /// </summary>
    [Fact]
    public void CensoringReachesTheModel()
    {
        var features = Features();
        var allObserved = new Vector<int>(new[] { 1, 1, 1, 1, 1, 1, 1, 1 });

        var censored = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new KaplanMeierEstimator<double>())
            .Build(features, Times(), Events());

        var uncensored = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new KaplanMeierEstimator<double>())
            .Build(features, Times(), allObserved);

        var a = censored.Predict(features);
        var b = uncensored.Predict(features);

        bool differs = false;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i])
            {
                differs = true;
                break;
            }
        }

        Assert.True(differs, "Censoring half the subjects changed nothing, so the indicator is lost.");
    }

    /// <summary>
    /// The covariates must arrive intact — the overload prepends a column, and an off-by-one there would
    /// shift every feature without failing.
    /// </summary>
    [Fact]
    public void TheCovariatesAreNotShifted()
    {
        var features = Features();

        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new CoxProportionalHazards<double>())
            .Build(features, Times(), Events());

        var direct = new CoxProportionalHazards<double>();
        direct.FitSurvival(features, Times(), Events());

        var a = result.Predict(features);
        var b = direct.Predict(features);

        for (int i = 0; i < b.Length; i++)
        {
            Assert.Equal(b[i], a[i]);
        }
    }

    [Fact]
    public void ANonSurvivalModel_IsRejectedBeforeTraining()
    {
        var ex = Assert.Throws<InvalidOperationException>(() =>
            new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureModel(new SimpleRegression<double>())
                .Build(Features(), Times(), Events()));

        Assert.Contains("survival models", ex.Message, StringComparison.Ordinal);
        Assert.Contains(nameof(SimpleRegression<double>), ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void NoModelConfigured_SaysWhatToConfigure()
    {
        var ex = Assert.Throws<InvalidOperationException>(() =>
            new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .Build(Features(), Times(), Events()));

        Assert.Contains("KaplanMeierEstimator", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void MismatchedLengths_NameAllThreeCounts()
    {
        var shortEvents = new Vector<int>(new[] { 1, 0, 1 });

        var ex = Assert.Throws<ArgumentException>(() =>
            new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureModel(new KaplanMeierEstimator<double>())
                .Build(Features(), Times(), shortEvents));

        Assert.Contains("8 rows", ex.Message, StringComparison.Ordinal);
        Assert.Contains("3", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnIndicatorThatIsNotZeroOrOne_IsRejected()
    {
        var badEvents = new Vector<int>(new[] { 1, 0, 2, 0, 1, 0, 1, 0 });

        var ex = Assert.Throws<ArgumentException>(() =>
            new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureModel(new KaplanMeierEstimator<double>())
                .Build(Features(), Times(), badEvents));

        Assert.Contains("found 2 at index 2", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// A builder declared over tensors cannot take a covariate matrix and a time vector, and should say
    /// so rather than failing somewhere inside the pipeline.
    /// </summary>
    [Fact]
    public void ABuilderOverTheWrongTypes_SaysSo()
    {
        var ex = Assert.Throws<InvalidOperationException>(() =>
            new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
                .Build(Features(), Times(), Events()));

        Assert.Contains("needs a model", ex.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("KaplanMeierEstimator")]
    [InlineData("NelsonAalenEstimator")]
    [InlineData("CoxProportionalHazards")]
    [InlineData("WeibullAFT")]
    [InlineData("LogNormalAFT")]
    public void EverySurvivalModel_BuildsThroughTheOverload(string model)
    {
        SurvivalModelBase<double> Build() => model switch
        {
            "KaplanMeierEstimator" => new KaplanMeierEstimator<double>(),
            "NelsonAalenEstimator" => new NelsonAalenEstimator<double>(),
            "CoxProportionalHazards" => new CoxProportionalHazards<double>(),
            "WeibullAFT" => new WeibullAFT<double>(),
            "LogNormalAFT" => new LogNormalAFT<double>(),
            _ => throw new ArgumentOutOfRangeException(nameof(model))
        };

        var features = Features();

        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(Build())
            .Build(features, Times(), Events());

        var direct = Build();
        direct.FitSurvival(features, Times(), Events());

        var a = result.Predict(features);
        var b = direct.Predict(features);

        for (int i = 0; i < b.Length; i++)
        {
            Assert.Equal(b[i], a[i]);
        }
    }
}
