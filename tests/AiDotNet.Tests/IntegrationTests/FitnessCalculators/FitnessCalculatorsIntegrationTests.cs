using AiDotNet.Enums;
using AiDotNet.FitnessCalculators;
using AiDotNet.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.FitnessCalculators;

/// <summary>
/// Integration tests for fitness calculator classes.
/// Tests construction, IsHigherScoreBetter property, and IsBetterFitness method for all fitness calculators.
/// </summary>
public class FitnessCalculatorsIntegrationTests
{
    #region MeanSquaredErrorFitnessCalculator Tests

    [Fact(Timeout = 120000)]
    public async Task MeanSquaredErrorFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new MeanSquaredErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact(Timeout = 120000)]
    public async Task MeanSquaredErrorFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new MeanSquaredErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact(Timeout = 120000)]
    public async Task MeanSquaredErrorFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new MeanSquaredErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region MeanAbsoluteErrorFitnessCalculator Tests

    [Fact(Timeout = 120000)]
    public async Task MeanAbsoluteErrorFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new MeanAbsoluteErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact(Timeout = 120000)]
    public async Task MeanAbsoluteErrorFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new MeanAbsoluteErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact(Timeout = 120000)]
    public async Task MeanAbsoluteErrorFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new MeanAbsoluteErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region RootMeanSquaredErrorFitnessCalculator Tests

    [Fact(Timeout = 120000)]
    public async Task RootMeanSquaredErrorFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact(Timeout = 120000)]
    public async Task RootMeanSquaredErrorFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact(Timeout = 120000)]
    public async Task RootMeanSquaredErrorFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region RSquaredFitnessCalculator Tests

    [Fact(Timeout = 120000)]
    public async Task RSquaredFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new RSquaredFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact(Timeout = 120000)]
    public async Task RSquaredFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        // Note: R² fitness calculator uses isHigherScoreBetter=false for optimization compatibility
        var calculator = new RSquaredFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact(Timeout = 120000)]
    public async Task RSquaredFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        // Note: For optimization consistency, all fitness calculators use lower-is-better comparison
        var calculator = new RSquaredFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region AdjustedRSquaredFitnessCalculator Tests

    [Fact(Timeout = 120000)]
    public async Task AdjustedRSquaredFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new AdjustedRSquaredFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact(Timeout = 120000)]
    public async Task AdjustedRSquaredFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        // Note: Adjusted R² fitness calculator uses isHigherScoreBetter=false for optimization compatibility
        var calculator = new AdjustedRSquaredFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact(Timeout = 120000)]
    public async Task AdjustedRSquaredFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        // Note: For optimization consistency, all fitness calculators use lower-is-better comparison
        var calculator = new AdjustedRSquaredFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region BinaryCrossEntropyLossFitnessCalculator Tests

    [Fact(Timeout = 120000)]
    public async Task BinaryCrossEntropyLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new BinaryCrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact(Timeout = 120000)]
    public async Task BinaryCrossEntropyLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new BinaryCrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact(Timeout = 120000)]
    public async Task BinaryCrossEntropyLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new BinaryCrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region CategoricalCrossEntropyLossFitnessCalculator Tests

    [Fact(Timeout = 120000)]
    public async Task CategoricalCrossEntropyLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new CategoricalCrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact(Timeout = 120000)]
    public async Task CategoricalCrossEntropyLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new CategoricalCrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact(Timeout = 120000)]
    public async Task CategoricalCrossEntropyLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new CategoricalCrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region CrossEntropyLossFitnessCalculator Tests

    [Fact(Timeout = 120000)]
    public async Task CrossEntropyLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new CrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact(Timeout = 120000)]
    public async Task CrossEntropyLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new CrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact(Timeout = 120000)]
    public async Task CrossEntropyLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new CrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region HuberLossFitnessCalculator Tests

    [Fact(Timeout = 120000)]
    public async Task HuberLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new HuberLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact(Timeout = 120000)]
    public async Task HuberLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new HuberLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact(Timeout = 120000)]
    public async Task HuberLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new HuberLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region HingeLossFitnessCalculator Tests

    [Fact(Timeout = 120000)]
    public async Task HingeLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new HingeLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact(Timeout = 120000)]
    public async Task HingeLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new HingeLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact(Timeout = 120000)]
    public async Task HingeLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new HingeLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region SquaredHingeLossFitnessCalculator Tests

    [Fact(Timeout = 120000)]
    public async Task SquaredHingeLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new SquaredHingeLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact(Timeout = 120000)]
    public async Task SquaredHingeLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new SquaredHingeLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact(Timeout = 120000)]
    public async Task SquaredHingeLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new SquaredHingeLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region LogCoshLossFitnessCalculator Tests

    [Fact(Timeout = 120000)]
    public async Task LogCoshLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new LogCoshLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact(Timeout = 120000)]
    public async Task LogCoshLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new LogCoshLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact(Timeout = 120000)]
    public async Task LogCoshLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new LogCoshLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region QuantileLossFitnessCalculator Tests

    [Fact(Timeout = 120000)]
    public async Task QuantileLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new QuantileLossFitnessCalculator<double, Matrix<double>, Vector<double>>(0.5);
        Assert.NotNull(calculator);
    }

    [Fact(Timeout = 120000)]
    public async Task QuantileLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new QuantileLossFitnessCalculator<double, Matrix<double>, Vector<double>>(0.5);
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact(Timeout = 120000)]
    public async Task QuantileLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new QuantileLossFitnessCalculator<double, Matrix<double>, Vector<double>>(0.5);
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region FocalLossFitnessCalculator Tests

    [Fact(Timeout = 120000)]
    public async Task FocalLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new FocalLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact(Timeout = 120000)]
    public async Task FocalLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new FocalLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact(Timeout = 120000)]
    public async Task FocalLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new FocalLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region CosineSimilarityLossFitnessCalculator Tests

    [Fact(Timeout = 120000)]
    public async Task CosineSimilarityLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new CosineSimilarityLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact(Timeout = 120000)]
    public async Task CosineSimilarityLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new CosineSimilarityLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact(Timeout = 120000)]
    public async Task CosineSimilarityLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new CosineSimilarityLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region ContrastiveLossFitnessCalculator Tests

    [Fact(Timeout = 120000)]
    public async Task ContrastiveLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new ContrastiveLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact(Timeout = 120000)]
    public async Task ContrastiveLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new ContrastiveLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact(Timeout = 120000)]
    public async Task ContrastiveLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new ContrastiveLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region TripletLossFitnessCalculator Tests

    [Fact(Timeout = 120000)]
    public async Task TripletLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new TripletLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact(Timeout = 120000)]
    public async Task TripletLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new TripletLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact(Timeout = 120000)]
    public async Task TripletLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new TripletLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region DiceLossFitnessCalculator Tests

    [Fact(Timeout = 120000)]
    public async Task DiceLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new DiceLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact(Timeout = 120000)]
    public async Task DiceLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new DiceLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact(Timeout = 120000)]
    public async Task DiceLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new DiceLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region JaccardLossFitnessCalculator Tests

    [Fact(Timeout = 120000)]
    public async Task JaccardLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new JaccardLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact(Timeout = 120000)]
    public async Task JaccardLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new JaccardLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact(Timeout = 120000)]
    public async Task JaccardLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new JaccardLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region KullbackLeiblerDivergenceFitnessCalculator Tests

    [Fact(Timeout = 120000)]
    public async Task KullbackLeiblerDivergenceFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact(Timeout = 120000)]
    public async Task KullbackLeiblerDivergenceFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact(Timeout = 120000)]
    public async Task KullbackLeiblerDivergenceFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region PoissonLossFitnessCalculator Tests

    [Fact(Timeout = 120000)]
    public async Task PoissonLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new PoissonLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact(Timeout = 120000)]
    public async Task PoissonLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new PoissonLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact(Timeout = 120000)]
    public async Task PoissonLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new PoissonLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region ExponentialLossFitnessCalculator Tests

    [Fact(Timeout = 120000)]
    public async Task ExponentialLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new ExponentialLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact(Timeout = 120000)]
    public async Task ExponentialLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new ExponentialLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact(Timeout = 120000)]
    public async Task ExponentialLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new ExponentialLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region ElasticNetLossFitnessCalculator Tests

    [Fact(Timeout = 120000)]
    public async Task ElasticNetLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new ElasticNetLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact(Timeout = 120000)]
    public async Task ElasticNetLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new ElasticNetLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact(Timeout = 120000)]
    public async Task ElasticNetLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new ElasticNetLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region ModifiedHuberLossFitnessCalculator Tests

    [Fact(Timeout = 120000)]
    public async Task ModifiedHuberLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new ModifiedHuberLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact(Timeout = 120000)]
    public async Task ModifiedHuberLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new ModifiedHuberLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact(Timeout = 120000)]
    public async Task ModifiedHuberLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new ModifiedHuberLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region OrdinalRegressionLossFitnessCalculator Tests

    [Fact(Timeout = 120000)]
    public async Task OrdinalRegressionLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new OrdinalRegressionLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact(Timeout = 120000)]
    public async Task OrdinalRegressionLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new OrdinalRegressionLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact(Timeout = 120000)]
    public async Task OrdinalRegressionLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new OrdinalRegressionLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region WeightedCrossEntropyLossFitnessCalculator Tests

    [Fact(Timeout = 120000)]
    public async Task WeightedCrossEntropyLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new WeightedCrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact(Timeout = 120000)]
    public async Task WeightedCrossEntropyLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new WeightedCrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact(Timeout = 120000)]
    public async Task WeightedCrossEntropyLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new WeightedCrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region Integration Tests

    [Fact(Timeout = 120000)]
    public async Task AllFitnessCalculators_DifferentDataSetTypes_CanBeConstructed()
    {
        // Training set
        var mseTraining = new MeanSquaredErrorFitnessCalculator<double, Matrix<double>, Vector<double>>(DataSetType.Training);
        Assert.NotNull(mseTraining);

        // Validation set (default)
        var mseValidation = new MeanSquaredErrorFitnessCalculator<double, Matrix<double>, Vector<double>>(DataSetType.Validation);
        Assert.NotNull(mseValidation);

        // Test set
        var mseTesting = new MeanSquaredErrorFitnessCalculator<double, Matrix<double>, Vector<double>>(DataSetType.Testing);
        Assert.NotNull(mseTesting);
    }

    [Fact(Timeout = 120000)]
    public async Task FitnessCalculators_EqualScores_IsBetterFitness_ReturnsFalse()
    {
        var mseCalculator = new MeanSquaredErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();
        var r2Calculator = new RSquaredFitnessCalculator<double, Matrix<double>, Vector<double>>();

        // Equal scores should not be "better"
        Assert.False(mseCalculator.IsBetterFitness(0.5, 0.5));
        Assert.False(r2Calculator.IsBetterFitness(0.5, 0.5));
    }

    [Fact(Timeout = 120000)]
    public async Task FitnessCalculators_ExtremeDifferences_HandleCorrectly()
    {
        var mseCalculator = new MeanSquaredErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();

        // Very small vs very large - all fitness calculators use lower-is-better
        Assert.True(mseCalculator.IsBetterFitness(0.0001, 1000.0));
    }

    [Fact(Timeout = 120000)]
    public async Task FitnessCalculators_NegativeValues_HandleCorrectly()
    {
        var mseCalculator = new MeanSquaredErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();

        // Negative values can occur - lower is still better for all fitness calculators
        Assert.True(mseCalculator.IsBetterFitness(-0.5, 0.5));
        Assert.True(mseCalculator.IsBetterFitness(-0.5, -0.1));
    }

    #endregion
}
