using System;
using AiDotNet.SurvivalAnalysis;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.SurvivalAnalysis;

/// <summary>
/// Pins that survival models learn who was censored.
/// </summary>
/// <remarks>
/// <para>
/// <c>SurvivalModelBase.Train(x, y)</c> used to set every event indicator to 1, which asserts that
/// nobody was censored. Censoring is the entire reason survival analysis is a separate field: a subject
/// still alive when the study ends has not had the event, and counting them as though they had biases
/// every survival estimate downward. That is worse than refusing the data, because the answer looks
/// reasonable.
/// </para>
/// <para>
/// Column 0 of X now carries the indicator, matching the convention
/// <c>CausalModelBase.Train(x, y)</c> already used to pass its treatment indicator through the same
/// two-argument contract.
/// </para>
/// </remarks>
public class SurvivalCensoringTests
{
    private const int Subjects = 8;

    /// <summary>The covariates on their own: one column of ages.</summary>
    private static Matrix<double> Covariates()
    {
        double[] ages = { 45, 52, 38, 61, 47, 55, 41, 58 };
        var m = new Matrix<double>(Subjects, 1);
        for (int i = 0; i < Subjects; i++)
        {
            m[i, 0] = ages[i];
        }

        return m;
    }

    private static Vector<double> Times() =>
        new(new double[] { 5.0, 12.0, 3.0, 18.0, 9.0, 21.0, 7.0, 15.0 });

    /// <summary>Half the subjects censored, so honouring the indicator has something to change.</summary>
    private static Vector<int> Events() => new(new[] { 1, 0, 1, 0, 1, 0, 1, 0 });

    /// <summary>The design matrix the two-argument Train takes: indicator, then covariates.</summary>
    private static Matrix<double> Design(Vector<int> events)
    {
        var covariates = Covariates();
        var design = new Matrix<double>(Subjects, covariates.Columns + 1);
        for (int i = 0; i < Subjects; i++)
        {
            design[i, 0] = events[i];
            for (int j = 0; j < covariates.Columns; j++)
            {
                design[i, j + 1] = covariates[i, j];
            }
        }

        return design;
    }

    /// <summary>
    /// The strongest statement available: splitting the design matrix has to produce exactly the fit the
    /// explicit three-argument entry point produces from the same pieces. If column 0 were read as a
    /// covariate, or the indicator misaligned by a row, these would diverge.
    /// </summary>
    [Fact]
    public void TheTwoArgumentPath_FitsTheSameModelAsTheExplicitPath()
    {
        var events = Events();

        var viaDesignMatrix = new KaplanMeierEstimator<double>();
        viaDesignMatrix.Train(Design(events), Times());

        var viaExplicitCall = new KaplanMeierEstimator<double>();
        viaExplicitCall.FitSurvival(Covariates(), Times(), events);

        var a = viaDesignMatrix.Predict(Covariates());
        var b = viaExplicitCall.Predict(Covariates());

        Assert.Equal(b.Length, a.Length);
        for (int i = 0; i < b.Length; i++)
        {
            Assert.Equal(b[i], a[i]);
        }
    }

    /// <summary>
    /// The defect this replaces, stated as a test: if the indicator were ignored and everyone treated as
    /// an observed event, these two fits would be identical. They must not be.
    /// </summary>
    [Fact]
    public void CensoredSubjects_ChangeTheFit()
    {
        var allObserved = new Vector<int>(new[] { 1, 1, 1, 1, 1, 1, 1, 1 });

        var withCensoring = new KaplanMeierEstimator<double>();
        withCensoring.Train(Design(Events()), Times());

        var withoutCensoring = new KaplanMeierEstimator<double>();
        withoutCensoring.Train(Design(allObserved), Times());

        var a = withCensoring.Predict(Covariates());
        var b = withoutCensoring.Predict(Covariates());

        bool differsSomewhere = false;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i])
            {
                differsSomewhere = true;
                break;
            }
        }

        Assert.True(
            differsSomewhere,
            "Censoring half the subjects produced the same fit as observing all of them, which means " +
            "the event indicator is not reaching the model.");
    }

    /// <summary>
    /// Censoring is not missing data — a censored subject is known to have survived at least that long.
    /// Counting them as an event instead understates survival, so honouring the indicator must not lower
    /// the survival curve.
    /// </summary>
    [Fact]
    public void TreatingCensoredSubjectsAsEvents_UnderstatesSurvival()
    {
        var allObserved = new Vector<int>(new[] { 1, 1, 1, 1, 1, 1, 1, 1 });

        var honest = new KaplanMeierEstimator<double>();
        honest.Train(Design(Events()), Times());

        var pessimistic = new KaplanMeierEstimator<double>();
        pessimistic.Train(Design(allObserved), Times());

        var honestCurve = honest.GetBaselineSurvival(Times());
        var pessimisticCurve = pessimistic.GetBaselineSurvival(Times());

        for (int i = 0; i < honestCurve.Length; i++)
        {
            Assert.True(
                honestCurve[i] >= pessimisticCurve[i],
                $"At time index {i}, honouring censoring gave survival {honestCurve[i]} but treating " +
                $"every subject as an observed event gave {pessimisticCurve[i]}. Censoring cannot lower " +
                "the estimated survival.");
        }
    }

    [Fact]
    public void ADesignMatrixWithNoCovariates_SaysWhatIsMissing()
    {
        var eventsOnly = new Matrix<double>(Subjects, 1);
        for (int i = 0; i < Subjects; i++)
        {
            eventsOnly[i, 0] = 1;
        }

        var ex = Assert.Throws<ArgumentException>(
            () => new KaplanMeierEstimator<double>().Train(eventsOnly, Times()));

        Assert.Contains("event indicator", ex.Message, StringComparison.Ordinal);
        Assert.Contains("censored", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// The one mistake the split cannot recover from is a covariate left in column 0, which would be
    /// read as censoring and silently change the answer. It has to be rejected, and the message has to
    /// say what to do about it.
    /// </summary>
    [Fact]
    public void ACovariateLeftInColumnZero_IsRejected()
    {
        // Two columns, so this reaches the indicator check rather than the arity check: someone has
        // passed a plain covariate matrix where the design matrix was expected.
        double[] ages = { 45, 52, 38, 61, 47, 55, 41, 58 };
        var plainCovariates = new Matrix<double>(Subjects, 2);
        for (int i = 0; i < Subjects; i++)
        {
            plainCovariates[i, 0] = ages[i];
            plainCovariates[i, 1] = i % 2;
        }

        var ex = Assert.Throws<ArgumentException>(
            () => new KaplanMeierEstimator<double>().Train(plainCovariates, Times()));

        Assert.Contains("must be 0 or 1", ex.Message, StringComparison.Ordinal);
        Assert.Contains("move it", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AMismatchedRowCount_IsRejected()
    {
        var shortTimes = new Vector<double>(new double[] { 5.0, 12.0 });

        var ex = Assert.Throws<ArgumentException>(
            () => new KaplanMeierEstimator<double>().Train(Design(Events()), shortTimes));

        Assert.Contains("Sample count mismatch", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// The convention has to hold for every survival model, not just the one the other tests use.
    /// </summary>
    [Theory]
    [InlineData("KaplanMeierEstimator")]
    [InlineData("NelsonAalenEstimator")]
    [InlineData("CoxProportionalHazards")]
    [InlineData("WeibullAFT")]
    [InlineData("LogNormalAFT")]
    public void EverySurvivalModel_ReadsTheEventIndicator(string model)
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

        var viaDesignMatrix = Build();
        viaDesignMatrix.Train(Design(Events()), Times());

        var viaExplicitCall = Build();
        viaExplicitCall.FitSurvival(Covariates(), Times(), Events());

        var a = viaDesignMatrix.Predict(Covariates());
        var b = viaExplicitCall.Predict(Covariates());

        for (int i = 0; i < b.Length; i++)
        {
            Assert.Equal(b[i], a[i]);
        }

        // And a design matrix whose column 0 is not an indicator is refused by all of them, not just
        // by whichever model the other tests happen to use.
        var wrongShape = new Matrix<double>(Subjects, 2);
        for (int i = 0; i < Subjects; i++)
        {
            wrongShape[i, 0] = 45 + i;
            wrongShape[i, 1] = i % 2;
        }

        Assert.Throws<ArgumentException>(() => Build().Train(wrongShape, Times()));
    }
}
