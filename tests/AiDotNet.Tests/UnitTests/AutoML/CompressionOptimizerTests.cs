using System;
using System.Linq;
using AiDotNet.AutoML;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.AutoML
{
    public class CompressionOptimizerTests
    {
        #region Constructor Tests

        [Fact]
        public void Constructor_WithDefaultOptions_CreatesInstance()
        {
            // Arrange & Act
            var optimizer = new CompressionOptimizer<double>();

            // Assert
            Assert.NotNull(optimizer);
            Assert.Null(optimizer.BestTrial);
            Assert.Empty(optimizer.TrialHistory);
        }

        [Fact]
        public void Constructor_WithCustomOptions_CreatesInstance()
        {
            // Arrange
            var options = new CompressionOptimizerOptions
            {
                MaxTrials = 5,
                MaxAccuracyLoss = 0.05,
                MinCompressionRatio = 1.5,
                RandomSeed = 42
            };

            // Act
            var optimizer = new CompressionOptimizer<double>(options);

            // Assert
            Assert.NotNull(optimizer);
        }

        #endregion

        #region Options Tests

        [Fact]
        public void CompressionOptimizerOptions_DefaultValues_AreCorrect()
        {
            // Arrange & Act
            var options = new CompressionOptimizerOptions();

            // Assert
            Assert.Equal(20, options.MaxTrials);
            Assert.Equal(0.02, options.MaxAccuracyLoss);
            Assert.Equal(2.0, options.MinCompressionRatio);
            Assert.Equal(0.5, options.AccuracyWeight);
            Assert.Equal(0.3, options.CompressionWeight);
            Assert.Equal(0.2, options.SpeedWeight);
            Assert.True(options.IncludePruning);
            Assert.True(options.IncludeQuantization);
            Assert.True(options.IncludeEncoding);
            Assert.True(options.IncludeHybrid);
            Assert.Null(options.RandomSeed);
        }

        [Fact]
        public void CompressionOptimizerOptions_CustomValues_AreSet()
        {
            // Arrange & Act
            var options = new CompressionOptimizerOptions
            {
                MaxTrials = 10,
                MaxAccuracyLoss = 0.1,
                MinCompressionRatio = 3.0,
                AccuracyWeight = 0.6,
                CompressionWeight = 0.25,
                SpeedWeight = 0.15,
                IncludePruning = false,
                IncludeQuantization = true,
                IncludeEncoding = false,
                IncludeHybrid = false,
                RandomSeed = 123
            };

            // Assert
            Assert.Equal(10, options.MaxTrials);
            Assert.Equal(0.1, options.MaxAccuracyLoss);
            Assert.Equal(3.0, options.MinCompressionRatio);
            Assert.Equal(0.6, options.AccuracyWeight);
            Assert.Equal(0.25, options.CompressionWeight);
            Assert.Equal(0.15, options.SpeedWeight);
            Assert.False(options.IncludePruning);
            Assert.True(options.IncludeQuantization);
            Assert.False(options.IncludeEncoding);
            Assert.False(options.IncludeHybrid);
            Assert.Equal(123, options.RandomSeed);
        }

        #endregion

        #region Optimize Tests

        [Fact]
        public void Optimize_WithNullWeights_ThrowsException()
        {
            // Arrange
            var optimizer = new CompressionOptimizer<double>();
            Func<Vector<double>, double> evaluator = _ => 0.9;

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                optimizer.Optimize(null!, evaluator));
        }

        [Fact]
        public void Optimize_WithNullEvaluator_ThrowsException()
        {
            // Arrange
            var optimizer = new CompressionOptimizer<double>();
            var weights = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                optimizer.Optimize(weights, null!));
        }

        [Fact]
        public void Optimize_WithValidInputs_ReturnsResult()
        {
            // Arrange
            var options = new CompressionOptimizerOptions
            {
                MaxTrials = 3,
                IncludePruning = true,
                IncludeQuantization = false,
                IncludeEncoding = false,
                IncludeHybrid = false,
                RandomSeed = 42
            };
            var optimizer = new CompressionOptimizer<double>(options);

            var random = RandomHelper.CreateSeededRandom(42);
            var weights = new double[100];
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = random.NextDouble();
            }
            var weightsVector = new Vector<double>(weights);

            Func<Vector<double>, double> evaluator = w => 0.9; // Simple constant accuracy

            // Act
            var result = optimizer.Optimize(weightsVector, evaluator);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.Success);
        }

        [Fact]
        public void Optimize_RecordsTrialHistory()
        {
            // Arrange
            var options = new CompressionOptimizerOptions
            {
                MaxTrials = 3,
                IncludePruning = true,
                IncludeQuantization = false,
                IncludeEncoding = false,
                IncludeHybrid = false,
                RandomSeed = 42
            };
            var optimizer = new CompressionOptimizer<double>(options);

            var weights = new Vector<double>(new double[] {
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
            });
            Func<Vector<double>, double> evaluator = _ => 0.9;

            // Act
            optimizer.Optimize(weights, evaluator);

            // Assert
            Assert.NotEmpty(optimizer.TrialHistory);
            Assert.True(optimizer.TrialHistory.Count <= 3);
        }

        [Fact]
        public void Optimize_SetsBestTrial()
        {
            // Arrange
            var options = new CompressionOptimizerOptions
            {
                MaxTrials = 2,
                IncludePruning = true,
                IncludeQuantization = false,
                IncludeEncoding = false,
                IncludeHybrid = false,
                RandomSeed = 42,
                MaxAccuracyLoss = 1.0, // Allow any accuracy loss
                MinCompressionRatio = 0.1 // Very low threshold
            };
            var optimizer = new CompressionOptimizer<double>(options);

            var weights = new Vector<double>(new double[] {
                0.001, 0.5, 0.002, 0.8, 0.003, 0.7, 0.004, 0.9
            });
            Func<Vector<double>, double> evaluator = _ => 0.95;

            // Act
            optimizer.Optimize(weights, evaluator);

            // Assert
            Assert.NotNull(optimizer.BestTrial);
        }

        #endregion

        #region Trial Tests

        [Fact]
        public void CompressionTrial_DefaultValues_AreCorrect()
        {
            // Arrange & Act
            var trial = new CompressionTrial<double>();

            // Assert
            Assert.Equal(CompressionType.None, trial.Technique);
            Assert.NotNull(trial.Hyperparameters);
            Assert.Empty(trial.Hyperparameters);
            Assert.Null(trial.Metrics);
            Assert.False(trial.Success);
            Assert.Null(trial.ErrorMessage);
        }

        [Fact]
        public void CompressionTrial_CanSetProperties()
        {
            // Arrange & Act
            var trial = new CompressionTrial<double>
            {
                Technique = CompressionType.SparsePruning,
                Success = true,
                FitnessScore = 0.8,
                ErrorMessage = null
            };
            trial.Hyperparameters["sparsityTarget"] = 0.9;

            // Assert
            Assert.Equal(CompressionType.SparsePruning, trial.Technique);
            Assert.True(trial.Success);
            Assert.Equal(0.8, trial.FitnessScore);
            Assert.Single(trial.Hyperparameters);
        }

        #endregion

        #region GetSummary Tests

        [Fact]
        public void GetSummary_WithNoTrials_ReturnsValidSummary()
        {
            // Arrange
            var optimizer = new CompressionOptimizer<double>();

            // Act
            var summary = optimizer.GetSummary();

            // Assert
            Assert.Contains("Compression Optimization Summary", summary);
            Assert.Contains("Trials: 0", summary);
            Assert.Contains("No successful trials completed", summary);
        }

        [Fact]
        public void GetSummary_WithTrials_ContainsDetails()
        {
            // Arrange
            var options = new CompressionOptimizerOptions
            {
                MaxTrials = 2,
                IncludePruning = true,
                IncludeQuantization = false,
                IncludeEncoding = false,
                IncludeHybrid = false,
                RandomSeed = 42,
                MaxAccuracyLoss = 1.0,
                MinCompressionRatio = 0.1
            };
            var optimizer = new CompressionOptimizer<double>(options);

            var weights = new Vector<double>(new double[] {
                0.001, 0.5, 0.002, 0.8, 0.003, 0.7, 0.004, 0.9
            });
            Func<Vector<double>, double> evaluator = _ => 0.95;

            optimizer.Optimize(weights, evaluator);

            // Act
            var summary = optimizer.GetSummary();

            // Assert
            Assert.Contains("Compression Optimization Summary", summary);
            Assert.Contains("successful", summary);
        }

        #endregion

        #region Type-Specific Tests

        [Fact]
        public void Optimize_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var options = new CompressionOptimizerOptions
            {
                MaxTrials = 2,
                IncludePruning = true,
                IncludeQuantization = false,
                IncludeEncoding = false,
                IncludeHybrid = false,
                RandomSeed = 42
            };
            var optimizer = new CompressionOptimizer<float>(options);

            var weights = new Vector<float>(new float[] {
                0.001f, 0.5f, 0.002f, 0.8f, 0.003f, 0.7f, 0.004f, 0.9f
            });
            Func<Vector<float>, float> evaluator = _ => 0.9f;

            // Act
            var result = optimizer.Optimize(weights, evaluator);

            // Assert
            Assert.NotNull(result);
        }

        #endregion

        #region Edge Case Tests

        [Fact]
        public void Optimize_WithOnlyQuantization_WorksCorrectly()
        {
            // Arrange
            var options = new CompressionOptimizerOptions
            {
                MaxTrials = 2,
                IncludePruning = false,
                IncludeQuantization = true,
                IncludeEncoding = false,
                IncludeHybrid = false,
                RandomSeed = 42
            };
            var optimizer = new CompressionOptimizer<double>(options);

            var weights = new Vector<double>(new double[] {
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
            });
            Func<Vector<double>, double> evaluator = _ => 0.9;

            // Act
            var result = optimizer.Optimize(weights, evaluator);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(CompressionType.WeightClustering, result.Technique);
        }

        [Fact]
        public void Optimize_WithOnlyEncoding_WorksCorrectly()
        {
            // Arrange
            var options = new CompressionOptimizerOptions
            {
                MaxTrials = 2,
                IncludePruning = false,
                IncludeQuantization = false,
                IncludeEncoding = true,
                IncludeHybrid = false,
                RandomSeed = 42
            };
            var optimizer = new CompressionOptimizer<double>(options);

            var weights = new Vector<double>(new double[] {
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
            });
            Func<Vector<double>, double> evaluator = _ => 0.9;

            // Act
            var result = optimizer.Optimize(weights, evaluator);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(CompressionType.HuffmanEncoding, result.Technique);
        }

        #endregion
    }
}
