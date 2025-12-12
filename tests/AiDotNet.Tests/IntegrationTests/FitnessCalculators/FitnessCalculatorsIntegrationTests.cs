using AiDotNet.Enums;
using AiDotNet.FitnessCalculators;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.FitnessCalculators;

/// <summary>
/// Integration tests for fitness calculator classes.
/// Tests construction, IsHigherScoreBetter property, and IsBetterFitness method for all fitness calculators.
/// </summary>
public class FitnessCalculatorsIntegrationTests
{
    #region MeanSquaredErrorFitnessCalculator Tests

    [Fact]
    public void MeanSquaredErrorFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new MeanSquaredErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact]
    public void MeanSquaredErrorFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new MeanSquaredErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void MeanSquaredErrorFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new MeanSquaredErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region MeanAbsoluteErrorFitnessCalculator Tests

    [Fact]
    public void MeanAbsoluteErrorFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new MeanAbsoluteErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact]
    public void MeanAbsoluteErrorFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new MeanAbsoluteErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void MeanAbsoluteErrorFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new MeanAbsoluteErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region RootMeanSquaredErrorFitnessCalculator Tests

    [Fact]
    public void RootMeanSquaredErrorFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact]
    public void RootMeanSquaredErrorFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void RootMeanSquaredErrorFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new RootMeanSquaredErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region RSquaredFitnessCalculator Tests

    [Fact]
    public void RSquaredFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new RSquaredFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact]
    public void RSquaredFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        // Note: R² fitness calculator uses isHigherScoreBetter=false for optimization compatibility
        var calculator = new RSquaredFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void RSquaredFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        // Note: For optimization consistency, all fitness calculators use lower-is-better comparison
        var calculator = new RSquaredFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region AdjustedRSquaredFitnessCalculator Tests

    [Fact]
    public void AdjustedRSquaredFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new AdjustedRSquaredFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact]
    public void AdjustedRSquaredFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        // Note: Adjusted R² fitness calculator uses isHigherScoreBetter=false for optimization compatibility
        var calculator = new AdjustedRSquaredFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void AdjustedRSquaredFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        // Note: For optimization consistency, all fitness calculators use lower-is-better comparison
        var calculator = new AdjustedRSquaredFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region BinaryCrossEntropyLossFitnessCalculator Tests

    [Fact]
    public void BinaryCrossEntropyLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new BinaryCrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact]
    public void BinaryCrossEntropyLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new BinaryCrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void BinaryCrossEntropyLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new BinaryCrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region CategoricalCrossEntropyLossFitnessCalculator Tests

    [Fact]
    public void CategoricalCrossEntropyLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new CategoricalCrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact]
    public void CategoricalCrossEntropyLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new CategoricalCrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void CategoricalCrossEntropyLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new CategoricalCrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region CrossEntropyLossFitnessCalculator Tests

    [Fact]
    public void CrossEntropyLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new CrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact]
    public void CrossEntropyLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new CrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void CrossEntropyLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new CrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region HuberLossFitnessCalculator Tests

    [Fact]
    public void HuberLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new HuberLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact]
    public void HuberLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new HuberLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void HuberLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new HuberLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region HingeLossFitnessCalculator Tests

    [Fact]
    public void HingeLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new HingeLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact]
    public void HingeLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new HingeLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void HingeLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new HingeLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region SquaredHingeLossFitnessCalculator Tests

    [Fact]
    public void SquaredHingeLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new SquaredHingeLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact]
    public void SquaredHingeLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new SquaredHingeLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void SquaredHingeLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new SquaredHingeLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region LogCoshLossFitnessCalculator Tests

    [Fact]
    public void LogCoshLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new LogCoshLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact]
    public void LogCoshLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new LogCoshLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void LogCoshLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new LogCoshLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region QuantileLossFitnessCalculator Tests

    [Fact]
    public void QuantileLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new QuantileLossFitnessCalculator<double, Matrix<double>, Vector<double>>(0.5);
        Assert.NotNull(calculator);
    }

    [Fact]
    public void QuantileLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new QuantileLossFitnessCalculator<double, Matrix<double>, Vector<double>>(0.5);
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void QuantileLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new QuantileLossFitnessCalculator<double, Matrix<double>, Vector<double>>(0.5);
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region FocalLossFitnessCalculator Tests

    [Fact]
    public void FocalLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new FocalLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact]
    public void FocalLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new FocalLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void FocalLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new FocalLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region CosineSimilarityLossFitnessCalculator Tests

    [Fact]
    public void CosineSimilarityLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new CosineSimilarityLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact]
    public void CosineSimilarityLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new CosineSimilarityLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void CosineSimilarityLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new CosineSimilarityLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region ContrastiveLossFitnessCalculator Tests

    [Fact]
    public void ContrastiveLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new ContrastiveLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact]
    public void ContrastiveLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new ContrastiveLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void ContrastiveLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new ContrastiveLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region TripletLossFitnessCalculator Tests

    [Fact]
    public void TripletLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new TripletLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact]
    public void TripletLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new TripletLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void TripletLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new TripletLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region DiceLossFitnessCalculator Tests

    [Fact]
    public void DiceLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new DiceLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact]
    public void DiceLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new DiceLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void DiceLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new DiceLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region JaccardLossFitnessCalculator Tests

    [Fact]
    public void JaccardLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new JaccardLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact]
    public void JaccardLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new JaccardLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void JaccardLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new JaccardLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region KullbackLeiblerDivergenceFitnessCalculator Tests

    [Fact]
    public void KullbackLeiblerDivergenceFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact]
    public void KullbackLeiblerDivergenceFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void KullbackLeiblerDivergenceFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new KullbackLeiblerDivergenceFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region PoissonLossFitnessCalculator Tests

    [Fact]
    public void PoissonLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new PoissonLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact]
    public void PoissonLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new PoissonLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void PoissonLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new PoissonLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region ExponentialLossFitnessCalculator Tests

    [Fact]
    public void ExponentialLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new ExponentialLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact]
    public void ExponentialLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new ExponentialLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void ExponentialLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new ExponentialLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region ElasticNetLossFitnessCalculator Tests

    [Fact]
    public void ElasticNetLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new ElasticNetLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact]
    public void ElasticNetLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new ElasticNetLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void ElasticNetLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new ElasticNetLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region ModifiedHuberLossFitnessCalculator Tests

    [Fact]
    public void ModifiedHuberLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new ModifiedHuberLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact]
    public void ModifiedHuberLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new ModifiedHuberLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void ModifiedHuberLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new ModifiedHuberLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region OrdinalRegressionLossFitnessCalculator Tests

    [Fact]
    public void OrdinalRegressionLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new OrdinalRegressionLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact]
    public void OrdinalRegressionLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new OrdinalRegressionLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void OrdinalRegressionLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new OrdinalRegressionLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region WeightedCrossEntropyLossFitnessCalculator Tests

    [Fact]
    public void WeightedCrossEntropyLossFitnessCalculator_Construction_Succeeds()
    {
        var calculator = new WeightedCrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(calculator);
    }

    [Fact]
    public void WeightedCrossEntropyLossFitnessCalculator_IsHigherScoreBetter_ReturnsFalse()
    {
        var calculator = new WeightedCrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.False(calculator.IsHigherScoreBetter);
    }

    [Fact]
    public void WeightedCrossEntropyLossFitnessCalculator_IsBetterFitness_LowerIsBetter()
    {
        var calculator = new WeightedCrossEntropyLossFitnessCalculator<double, Matrix<double>, Vector<double>>();
        Assert.True(calculator.IsBetterFitness(0.1, 0.5));
        Assert.False(calculator.IsBetterFitness(0.5, 0.1));
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void AllFitnessCalculators_DifferentDataSetTypes_CanBeConstructed()
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

    [Fact]
    public void FitnessCalculators_EqualScores_IsBetterFitness_ReturnsFalse()
    {
        var mseCalculator = new MeanSquaredErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();
        var r2Calculator = new RSquaredFitnessCalculator<double, Matrix<double>, Vector<double>>();

        // Equal scores should not be "better"
        Assert.False(mseCalculator.IsBetterFitness(0.5, 0.5));
        Assert.False(r2Calculator.IsBetterFitness(0.5, 0.5));
    }

    [Fact]
    public void FitnessCalculators_ExtremeDifferences_HandleCorrectly()
    {
        var mseCalculator = new MeanSquaredErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();

        // Very small vs very large - all fitness calculators use lower-is-better
        Assert.True(mseCalculator.IsBetterFitness(0.0001, 1000.0));
    }

    [Fact]
    public void FitnessCalculators_NegativeValues_HandleCorrectly()
    {
        var mseCalculator = new MeanSquaredErrorFitnessCalculator<double, Matrix<double>, Vector<double>>();

        // Negative values can occur - lower is still better for all fitness calculators
        Assert.True(mseCalculator.IsBetterFitness(-0.5, 0.5));
        Assert.True(mseCalculator.IsBetterFitness(-0.5, -0.1));
    }

    #endregion
}
