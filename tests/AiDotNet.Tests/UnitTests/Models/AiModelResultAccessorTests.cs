using System;
using AiDotNet.AutoML;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.CausalInference;
using AiDotNet.Enums;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Clustering.Spectral;
using AiDotNet.Data.Structures;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models;

/// <summary>
/// Covers the model-specific accessors on <see cref="AiModelResult{T, TInput, TOutput}"/>: the ones that
/// exist so a causal estimate, a series decomposition, a cluster assignment or a derived architecture can
/// be reached without holding the model and stepping around the builder.
/// </summary>
/// <remarks>
/// <para>
/// Two things are worth pinning for each accessor, and they fail differently. The first is that it hands
/// back exactly what the wrapped model produced — a delegation that silently returned a copy, a reordered
/// vector or a stale cache would be invisible at the call site. The second is that asking a model of the
/// wrong kind fails with a message naming what was actually built, because the alternative is a
/// NullReferenceException from inside a cast, which tells a caller nothing about what they did wrong.
/// </para>
/// <para>
/// The result is constructed directly rather than through <c>AiModelBuilder</c>. These accessors are pure
/// dispatch on the wrapped model, so routing each one through a full build would test the training
/// pipeline instead of the thing under test, and would make a fast unit test slow enough to skip.
/// </para>
/// </remarks>
public class AiModelResultAccessorTests
{
    /// <summary>
    /// Wraps a model in a result without running a build, so a test can pick the model kind.
    /// </summary>
    /// <remarks>
    /// The result takes its model from <c>OptimizationResult.BestSolution</c> rather than from the
    /// options' own <c>Model</c> property, which the standard construction path ignores.
    /// </remarks>
    private static AiModelResult<T, TInput, TOutput> Wrap<T, TInput, TOutput>(
        IFullModel<T, TInput, TOutput> model) =>
        new(new AiModelResultOptions<T, TInput, TOutput>
        {
            OptimizationResult = new OptimizationResult<T, TInput, TOutput> { BestSolution = model }
        });

    /// <summary>A result around a model that is none of the kinds the accessors dispatch on.</summary>
    private static AiModelResult<double, Matrix<double>, Vector<double>> WrongKind() =>
        Wrap<double, Matrix<double>, Vector<double>>(new SimpleRegression<double>());

    /// <summary>A short series with a period of 4, long enough to decompose and to fit.</summary>
    private static (Matrix<double> Features, Vector<double> Series) SeasonalSeries()
    {
        double[] values = { 10, 20, 30, 20, 12, 22, 32, 22, 14, 24, 34, 24, 16, 26, 36, 26 };
        var features = new Matrix<double>(values.Length, 1);
        var series = new Vector<double>(values.Length);
        for (int i = 0; i < values.Length; i++)
        {
            features[i, 0] = i;
            series[i] = values[i];
        }

        return (features, series);
    }

    /// <summary>
    /// A design matrix in the layout <see cref="CausalModelBase{T}"/> documents for the builder path:
    /// column 0 is the binary treatment indicator, columns 1.. are the covariates.
    /// </summary>
    private static (Matrix<double> Covariates, Vector<int> Treatment, Vector<double> Outcome) CausalSample()
    {
        double[,] covariateValues = { { 45, 1 }, { 52, 0 }, { 38, 1 }, { 61, 0 }, { 47, 1 }, { 55, 0 } };
        int[] treatmentValues = { 0, 1, 0, 1, 0, 1 };
        double[] outcomeValues = { 3.1, 7.2, 2.8, 8.0, 3.4, 7.6 };

        var covariates = new Matrix<double>(6, 2);
        for (int i = 0; i < 6; i++)
        {
            covariates[i, 0] = covariateValues[i, 0];
            covariates[i, 1] = covariateValues[i, 1];
        }

        return (covariates, new Vector<int>(treatmentValues), new Vector<double>(outcomeValues));
    }

    private static void AssertNamesTheBuiltModel(NotSupportedException ex)
    {
        // The whole point of the guard over a raw cast is that the caller learns what they built.
        Assert.Contains(nameof(SimpleRegression<double>), ex.Message, StringComparison.Ordinal);
    }

    // ── Causal effects ────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void EstimateATE_ReturnsWhatTheCausalModelComputed()
    {
        var (covariates, treatment, outcome) = CausalSample();
        var model = new InverseProbabilityWeighting<double>();
        model.Fit(covariates, treatment);

        var expected = model.EstimateATE(covariates, treatment, outcome);
        var actual = Wrap<double, Matrix<double>, Vector<double>>(model)
            .EstimateATE(covariates, treatment, outcome);

        Assert.Equal(expected.estimate, actual.estimate);

        // The standard error is bootstrapped from RandomHelper.CreateSecureRandom(), which takes no
        // seed, so two calls on the same fitted model disagree. Comparing it exactly would make this
        // test flaky for a reason that has nothing to do with the delegation under test.
        Assert.False(double.IsNaN(actual.standardError), "standard error should be a number");
        Assert.False(double.IsInfinity(actual.standardError), "standard error should be finite");
        Assert.True(actual.standardError >= 0, "standard error should not be negative");
    }

    [Fact]
    public void EstimateATT_ReturnsWhatTheCausalModelComputed()
    {
        var (covariates, treatment, outcome) = CausalSample();
        var model = new PropensityScoreMatching<double>();
        model.Fit(covariates, treatment);

        var expected = model.EstimateATT(covariates, treatment, outcome);
        var actual = Wrap<double, Matrix<double>, Vector<double>>(model)
            .EstimateATT(covariates, treatment, outcome);

        Assert.Equal(expected.estimate, actual.estimate);

        // The standard error is bootstrapped from RandomHelper.CreateSecureRandom(), which takes no
        // seed, so two calls on the same fitted model disagree. Comparing it exactly would make this
        // test flaky for a reason that has nothing to do with the delegation under test.
        Assert.False(double.IsNaN(actual.standardError), "standard error should be a number");
        Assert.False(double.IsInfinity(actual.standardError), "standard error should be finite");
        Assert.True(actual.standardError >= 0, "standard error should not be negative");
    }

    [Fact]
    public void EstimateTreatmentEffect_ReturnsOneEffectPerRow()
    {
        var (covariates, treatment, outcome) = CausalSample();
        var treatmentAsT = new Vector<double>(treatment.Length);
        for (int i = 0; i < treatment.Length; i++)
        {
            treatmentAsT[i] = treatment[i];
        }

        var model = new TLearner<double>();
        model.Fit(covariates, treatmentAsT, outcome);

        var expected = model.EstimateTreatmentEffect(covariates);
        var actual = Wrap<double, Matrix<double>, Vector<double>>(model)
            .EstimateTreatmentEffect(covariates);

        Assert.Equal(covariates.Rows, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i]);
        }
    }

    [Fact]
    public void EstimateATE_OnANonCausalModel_SaysWhatWasBuilt()
    {
        var (covariates, treatment, outcome) = CausalSample();
        var ex = Assert.Throws<NotSupportedException>(
            () => WrongKind().EstimateATE(covariates, treatment, outcome));
        AssertNamesTheBuiltModel(ex);
    }

    [Fact]
    public void EstimateATT_OnANonCausalModel_SaysWhatWasBuilt()
    {
        var (covariates, treatment, outcome) = CausalSample();
        var ex = Assert.Throws<NotSupportedException>(
            () => WrongKind().EstimateATT(covariates, treatment, outcome));
        AssertNamesTheBuiltModel(ex);
    }

    [Fact]
    public void EstimateTreatmentEffect_OnANonCausalModel_SaysWhatWasBuilt()
    {
        var (covariates, _, _) = CausalSample();
        var ex = Assert.Throws<NotSupportedException>(
            () => WrongKind().EstimateTreatmentEffect(covariates));
        AssertNamesTheBuiltModel(ex);
    }

    // ── Time-series components ────────────────────────────────────────────────────────────────────

    [Fact]
    public void GetTrend_And_GetSeasonal_ReturnTheDecompositionTheModelProduced()
    {
        var (features, series) = SeasonalSeries();
        var model = new STLDecomposition<double>(new STLDecompositionOptions<double> { SeasonalPeriod = 4 });
        model.Train(features, series);

        var result = Wrap<double, Matrix<double>, Vector<double>>(model);

        var expectedTrend = model.GetTrend();
        var actualTrend = result.GetTrend();
        Assert.Equal(expectedTrend.Length, actualTrend.Length);
        for (int i = 0; i < expectedTrend.Length; i++)
        {
            Assert.Equal(expectedTrend[i], actualTrend[i]);
        }

        var expectedSeasonal = model.GetSeasonal();
        var actualSeasonal = result.GetSeasonal();
        Assert.Equal(expectedSeasonal.Length, actualSeasonal.Length);
        for (int i = 0; i < expectedSeasonal.Length; i++)
        {
            Assert.Equal(expectedSeasonal[i], actualSeasonal[i]);
        }
    }

    [Fact]
    public void GetFrequencies_And_GetPeriodogram_LineUpWithEachOther()
    {
        var (features, series) = SeasonalSeries();
        var model = new SpectralAnalysisModel<double>();
        model.Train(features, series);

        var result = Wrap<double, Matrix<double>, Vector<double>>(model);

        var frequencies = result.GetFrequencies();
        var periodogram = result.GetPeriodogram();

        // They are read together — a power for each frequency — so a mismatch is a real defect.
        Assert.Equal(frequencies.Length, periodogram.Length);

        var expected = model.GetFrequencies();
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], frequencies[i]);
        }
    }

    [Fact]
    public void ComputeAnomalyScores_ReturnsOneScorePerPoint()
    {
        var (features, series) = SeasonalSeries();
        var model = new ARIMAModel<double>();
        model.Train(features, series);

        var actual = Wrap<double, Matrix<double>, Vector<double>>(model).ComputeAnomalyScores(series);
        var expected = model.ComputeAnomalyScores(series);

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i]);
        }
    }

    [Fact]
    public void Forecast_FromASuppliedHistory_ReturnsOneValuePerStep()
    {
        var (features, series) = SeasonalSeries();
        var model = new BayesianStructuralTimeSeriesModel<double>();
        model.Train(features, series);

        const int Horizon = 5;
        var actual = Wrap<double, Matrix<double>, Vector<double>>(model).Forecast(series, Horizon);

        Assert.Equal(Horizon, actual.Length);
    }

    [Fact]
    public void Forecast_FromTheFittedSeries_ReturnsOneValuePerStep()
    {
        var (features, series) = SeasonalSeries();
        var model = new UnobservedComponentsModel<double, Matrix<double>, Vector<double>>();
        model.Train(features, series);

        const int Horizon = 6;
        var actual = Wrap<double, Matrix<double>, Vector<double>>(model).Forecast(Horizon);

        Assert.Equal(Horizon, actual.Length);
    }

    [Fact]
    public void GetTrend_OnAModelThatDoesNotDecompose_SaysWhatWasBuilt() =>
        AssertNamesTheBuiltModel(Assert.Throws<NotSupportedException>(() => WrongKind().GetTrend()));

    [Fact]
    public void GetSeasonal_OnAModelThatDoesNotDecompose_SaysWhatWasBuilt() =>
        AssertNamesTheBuiltModel(Assert.Throws<NotSupportedException>(() => WrongKind().GetSeasonal()));

    [Fact]
    public void GetFrequencies_OnANonSpectralModel_SaysWhatWasBuilt() =>
        AssertNamesTheBuiltModel(Assert.Throws<NotSupportedException>(() => WrongKind().GetFrequencies()));

    [Fact]
    public void GetPeriodogram_OnANonSpectralModel_SaysWhatWasBuilt() =>
        AssertNamesTheBuiltModel(Assert.Throws<NotSupportedException>(() => WrongKind().GetPeriodogram()));

    [Fact]
    public void ComputeAnomalyScores_OnAModelThatDoesNotScoreThem_SaysWhatWasBuilt()
    {
        var (_, series) = SeasonalSeries();
        AssertNamesTheBuiltModel(
            Assert.Throws<NotSupportedException>(() => WrongKind().ComputeAnomalyScores(series)));
    }

    [Fact]
    public void Forecast_WithHistory_OnANonStructuralModel_PointsAtPredict()
    {
        var (_, series) = SeasonalSeries();
        var ex = Assert.Throws<NotSupportedException>(() => WrongKind().Forecast(series, horizon: 4));
        AssertNamesTheBuiltModel(ex);

        // A caller who wanted an ordinary sequence forecast should be sent to the method that gives one.
        Assert.Contains("Predict(lookback, horizon)", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Forecast_WithoutHistory_OnANonComponentsModel_PointsAtPredict()
    {
        var ex = Assert.Throws<NotSupportedException>(() => WrongKind().Forecast(horizon: 4));
        AssertNamesTheBuiltModel(ex);
        Assert.Contains("Predict(lookback, horizon)", ex.Message, StringComparison.Ordinal);
    }

    // ── Model-specific outputs ────────────────────────────────────────────────────────────────────

    [Fact]
    public void GetClusterLabels_ReturnsTheAssignmentTheModelMade()
    {
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 }, { 1.5, 1.8 }, { 5.0, 8.0 }, { 8.0, 8.0 },
            { 1.0, 0.6 }, { 9.0, 11.0 }, { 8.5, 9.5 }, { 0.5, 1.2 }
        });
        var ignored = new Vector<double>(data.Rows);

        var model = new SpectralClustering<double>();
        model.Train(data, ignored);

        var actual = Wrap<double, Matrix<double>, Vector<double>>(model).GetClusterLabels();

        Assert.NotNull(actual);
        Assert.Equal(data.Rows, actual!.Length);

        var expected = model.Labels;
        Assert.NotNull(expected);
        for (int i = 0; i < expected!.Length; i++)
        {
            Assert.Equal(expected[i], actual[i]);
        }
    }

    [Fact]
    public void GetClusterLabels_OnANonClusteringModel_SaysWhatWasBuilt() =>
        AssertNamesTheBuiltModel(Assert.Throws<NotSupportedException>(() => WrongKind().GetClusterLabels()));

    [Fact]
    public void DeriveArchitecture_ReturnsTheDesignTheSearchSettledOn()
    {
        var model = new SuperNet<float>(new SearchSpaceBase<float>(), numNodes: 4);

        var expected = model.DeriveArchitecture();
        var actual = Wrap<float, Tensor<float>, Tensor<float>>(model).DeriveArchitecture();

        Assert.NotNull(actual);
        Assert.Equal(expected.NodeCount, actual.NodeCount);
        Assert.Equal(expected.Operations.Count, actual.Operations.Count);
        for (int i = 0; i < expected.Operations.Count; i++)
        {
            Assert.Equal(expected.Operations[i], actual.Operations[i]);
        }
    }

    /// <summary>
    /// Dispatch has to reach the generator, not stop at the type test: an unfitted generator raises its
    /// own error, and seeing that rather than <see cref="NotSupportedException"/> is what proves the
    /// accessor delegated. Fitting a GAN is far too heavy for a unit test to do just to prove that.
    /// </summary>
    [Fact]
    public void GenerateSamples_OnAnUnfittedGenerator_SurfacesTheGeneratorsOwnError()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 8, outputSize: 8);
        var model = new MedGANGenerator<double>(architecture);

        var ex = Assert.Throws<InvalidOperationException>(
            () => Wrap<double, Tensor<double>, Tensor<double>>(model).GenerateSamples(numSamples: 4));

        Assert.Contains("not fitted", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// As above: what matters is that the call reached the meta-learner. Whatever it does with an empty
    /// task is the meta-learner's business, but it must not be the accessor's wrong-kind guard.
    /// </summary>
    [Fact]
    public void AdaptAndEvaluate_ReachesTheMetaLearner()
    {
        var metaModel = new NeuralNetwork<double>(
            new NeuralNetworkArchitecture<double>(inputFeatures: 4, outputSize: 2));
        var model = new MAMLAlgorithm<double, Tensor<double>, Tensor<double>>(
            new MAMLOptions<double, Tensor<double>, Tensor<double>>(metaModel));

        var result = Wrap<double, Tensor<double>, Tensor<double>>(model);
        var task = new MetaLearningTask<double, Tensor<double>, Tensor<double>>();

        var ex = Record.Exception(() => result.AdaptAndEvaluate(task));
        Assert.IsNotType<NotSupportedException>(ex);
    }

    [Fact]
    public void AdaptAndEvaluate_OnANonMetaLearner_SaysWhatWasBuilt()
    {
        var task = new MetaLearningTask<double, Matrix<double>, Vector<double>>();
        AssertNamesTheBuiltModel(
            Assert.Throws<NotSupportedException>(() => WrongKind().AdaptAndEvaluate(task)));
    }

    [Fact]
    public void GenerateSamples_OnANonGenerator_SaysWhatWasBuilt() =>
        AssertNamesTheBuiltModel(
            Assert.Throws<NotSupportedException>(() => WrongKind().GenerateSamples(numSamples: 10)));

    [Fact]
    public void DeriveArchitecture_OnANonSearchModel_SaysWhatWasBuilt() =>
        AssertNamesTheBuiltModel(
            Assert.Throws<NotSupportedException>(() => WrongKind().DeriveArchitecture()));

    /// <summary>
    /// Every accessor reaches the model through <c>EnsureModel</c>, which is what turns "nothing was
    /// built" into a message rather than a NullReferenceException from inside the type test.
    /// </summary>
    [Fact]
    public void AnAccessorOnAResultWithNoModel_ExplainsThatNothingWasBuilt()
    {
        var empty = new AiModelResult<double, Matrix<double>, Vector<double>>(
            new AiModelResultOptions<double, Matrix<double>, Vector<double>>
            {
                OptimizationResult = new OptimizationResult<double, Matrix<double>, Vector<double>>()
            });

        var ex = Assert.Throws<InvalidOperationException>(() => empty.GetTrend());
        Assert.Contains("has not been initialized", ex.Message, StringComparison.Ordinal);
    }
}
