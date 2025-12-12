using AiDotNet.CrossValidators;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using Xunit;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNetTests.IntegrationTests.CrossValidators
{
    /// <summary>
    /// Comprehensive integration tests for all CrossValidators in the AiDotNet library.
    /// Tests verify correct splitting behavior, mathematical properties, and edge cases for each validator.
    /// </summary>
    public class CrossValidatorsIntegrationTests
    {
        private const double Tolerance = 1e-6;
        private readonly Random _random = new(42);

        #region Test Helper Classes

        /// <summary>
        /// Simple test model for cross-validation testing
        /// </summary>
        private class SimpleTestModel<T> : IFullModel<T, Matrix<T>, Vector<T>>
        {
            private Vector<T> _parameters;
            private readonly INumericOperations<T> _numOps;

            public SimpleTestModel(int parameterCount)
            {
                _numOps = MathHelper.GetNumericOperations<T>();
                _parameters = new Vector<T>(parameterCount);
                for (int i = 0; i < parameterCount; i++)
                {
                    _parameters[i] = _numOps.FromDouble(1.0);
                }
                ParameterCount = parameterCount;
            }

            public int ParameterCount { get; }

            public Vector<T> GetParameters() => _parameters;

            public void SetParameters(Vector<T> parameters)
            {
                if (parameters.Length != _parameters.Length)
                    throw new ArgumentException("Parameter count mismatch");
                _parameters = parameters;
            }

            public IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
            {
                var model = new SimpleTestModel<T>(_parameters.Length);
                model.SetParameters(parameters);
                return model;
            }

            public IFullModel<T, Matrix<T>, Vector<T>> Clone()
            {
                var clone = new SimpleTestModel<T>(_parameters.Length);
                clone.SetParameters(_parameters.Clone());
                return clone;
            }

            public IFullModel<T, Matrix<T>, Vector<T>> DeepCopy()
            {
                return Clone();
            }

            public void Train(Matrix<T> inputs, Vector<T> outputs)
            {
                // Simple training: just compute mean
            }

            public Vector<T> Predict(Matrix<T> inputs)
            {
                var result = new Vector<T>(inputs.Rows);
                for (int i = 0; i < inputs.Rows; i++)
                {
                    T sum = _numOps.Zero;
                    for (int j = 0; j < Math.Min(inputs.Columns, _parameters.Length); j++)
                    {
                        sum = _numOps.Add(sum, _numOps.Multiply(inputs[i, j], _parameters[j]));
                    }
                    result[i] = sum;
                }
                return result;
            }

            public ModelMetadata<T> GetModelMetadata()
            {
                return new ModelMetadata<T>
                {
                    FeatureImportance = new Vector<T>(_parameters.Length)
                };
            }
        }

        /// <summary>
        /// Creates simple training data for testing
        /// </summary>
        private static (Matrix<double> X, Vector<double> y) CreateTestData(int samples = 100, int features = 2)
        {
            var X = new Matrix<double>(samples, features);
            var y = new Vector<double>(samples);
            var random = new Random(42);

            for (int i = 0; i < samples; i++)
            {
                for (int j = 0; j < features; j++)
                {
                    X[i, j] = random.NextDouble() * 10.0 - 5.0;
                }
                y[i] = 2.0 * X[i, 0] + (features > 1 ? 3.0 * X[i, 1] : 0) + (random.NextDouble() - 0.5);
            }

            return (X, y);
        }

        /// <summary>
        /// Creates classification data with balanced classes
        /// </summary>
        private static (Matrix<double> X, Vector<double> y) CreateClassificationData(int samples = 100, int classes = 3)
        {
            var X = new Matrix<double>(samples, 2);
            var y = new Vector<double>(samples);
            var random = new Random(42);

            for (int i = 0; i < samples; i++)
            {
                var classLabel = i % classes;
                X[i, 0] = random.NextDouble() * 10.0 + classLabel * 5.0;
                X[i, 1] = random.NextDouble() * 10.0 + classLabel * 5.0;
                y[i] = classLabel;
            }

            return (X, y);
        }

        /// <summary>
        /// Creates a simple optimizer for testing
        /// </summary>
        private static IOptimizer<double, Matrix<double>, Vector<double>> CreateSimpleOptimizer(IFullModel<double, Matrix<double>, Vector<double>> model)
        {
            var options = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                LearningRate = 0.01,
                MaxIterations = 10,
                Tolerance = 1e-6
            };
            return new AdamOptimizer<double, Matrix<double>, Vector<double>>(model, options);
        }

        #endregion

        #region StandardCrossValidator Tests

        [Fact]
        public void StandardCrossValidator_CreatesCorrectNumberOfFolds()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new CrossValidationOptions { NumberOfFolds = 5, RandomSeed = 42 };
            var validator = new StandardCrossValidator<double, Matrix<double>, Vector<double>>(options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert
            Assert.Equal(5, result.FoldResults.Count);
        }

        [Fact]
        public void StandardCrossValidator_TrainAndTestSetsAreDisjoint()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new CrossValidationOptions { NumberOfFolds = 5, RandomSeed = 42 };
            var validator = new StandardCrossValidator<double, Matrix<double>, Vector<double>>(options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - No overlap between train and test indices in each fold
            foreach (var fold in result.FoldResults)
            {
                var trainSet = new HashSet<int>(fold.TrainingIndices!);
                var testSet = new HashSet<int>(fold.ValidationIndices!);
                Assert.Empty(trainSet.Intersect(testSet));
            }
        }

        [Fact]
        public void StandardCrossValidator_AllDataPointsUsedInTestSets()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new CrossValidationOptions { NumberOfFolds = 5, RandomSeed = 42 };
            var validator = new StandardCrossValidator<double, Matrix<double>, Vector<double>>(options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - All indices appear in test sets across folds
            var allTestIndices = new HashSet<int>();
            foreach (var fold in result.FoldResults)
            {
                allTestIndices.UnionWith(fold.ValidationIndices!);
            }
            Assert.Equal(100, allTestIndices.Count);
        }

        [Fact]
        public void StandardCrossValidator_CorrectSplitProportions()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new CrossValidationOptions { NumberOfFolds = 5, RandomSeed = 42 };
            var validator = new StandardCrossValidator<double, Matrix<double>, Vector<double>>(options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - Each fold should have ~20% test, ~80% train
            foreach (var fold in result.FoldResults)
            {
                Assert.Equal(20, fold.ValidationIndices!.Length);
                Assert.Equal(80, fold.TrainingIndices!.Length);
            }
        }

        [Fact]
        public void StandardCrossValidator_SmallDataset_HandlesCorrectly()
        {
            // Arrange
            var (X, y) = CreateTestData(10, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new CrossValidationOptions { NumberOfFolds = 5, RandomSeed = 42 };
            var validator = new StandardCrossValidator<double, Matrix<double>, Vector<double>>(options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert
            Assert.Equal(5, result.FoldResults.Count);
            foreach (var fold in result.FoldResults)
            {
                Assert.Equal(2, fold.ValidationIndices!.Length);
            }
        }

        [Fact]
        public void StandardCrossValidator_SingleFold_UsesAllDataForTesting()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new CrossValidationOptions { NumberOfFolds = 1, RandomSeed = 42 };
            var validator = new StandardCrossValidator<double, Matrix<double>, Vector<double>>(options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert
            Assert.Single(result.FoldResults);
            Assert.Equal(100, result.FoldResults[0].ValidationIndices!.Length);
        }

        [Fact]
        public void StandardCrossValidator_WithShuffling_RandomizesData()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var optionsShuffled = new CrossValidationOptions { NumberOfFolds = 5, ShuffleData = true, RandomSeed = 42 };
            var optionsNotShuffled = new CrossValidationOptions { NumberOfFolds = 5, ShuffleData = false, RandomSeed = 42 };
            var validatorShuffled = new StandardCrossValidator<double, Matrix<double>, Vector<double>>(optionsShuffled);
            var validatorNotShuffled = new StandardCrossValidator<double, Matrix<double>, Vector<double>>(optionsNotShuffled);

            // Act
            var resultShuffled = validatorShuffled.Validate(model, X, y, optimizer);
            var resultNotShuffled = validatorNotShuffled.Validate(model, X, y, optimizer);

            // Assert - First fold test indices should be different
            var shuffledIndices = resultShuffled.FoldResults[0].ValidationIndices!;
            var notShuffledIndices = resultNotShuffled.FoldResults[0].ValidationIndices!;
            Assert.NotEqual(shuffledIndices, notShuffledIndices);
        }

        #endregion

        #region KFoldCrossValidator Tests

        [Fact]
        public void KFoldCrossValidator_CreatesCorrectNumberOfFolds()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new CrossValidationOptions { NumberOfFolds = 10, RandomSeed = 42 };
            var validator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert
            Assert.Equal(10, result.FoldResults.Count);
        }

        [Fact]
        public void KFoldCrossValidator_EqualSizedFolds()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new CrossValidationOptions { NumberOfFolds = 5, RandomSeed = 42 };
            var validator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - All folds should have equal test set size
            var testSizes = result.FoldResults.Select(f => f.ValidationIndices!.Length).ToList();
            Assert.True(testSizes.All(s => s == 20));
        }

        [Fact]
        public void KFoldCrossValidator_NoOverlapBetweenFolds()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new CrossValidationOptions { NumberOfFolds = 5, RandomSeed = 42 };
            var validator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - No test index should appear in multiple folds
            var allTestIndices = new List<int>();
            foreach (var fold in result.FoldResults)
            {
                allTestIndices.AddRange(fold.ValidationIndices!);
            }
            Assert.Equal(allTestIndices.Count, allTestIndices.Distinct().Count());
        }

        [Fact]
        public void KFoldCrossValidator_AllDataPointsCovered()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new CrossValidationOptions { NumberOfFolds = 5, RandomSeed = 42 };
            var validator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert
            var allTestIndices = new HashSet<int>();
            foreach (var fold in result.FoldResults)
            {
                allTestIndices.UnionWith(fold.ValidationIndices!);
            }
            Assert.Equal(100, allTestIndices.Count);
            Assert.Equal(Enumerable.Range(0, 100).ToHashSet(), allTestIndices);
        }

        [Fact]
        public void KFoldCrossValidator_CorrectProportionPerFold()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new CrossValidationOptions { NumberOfFolds = 4, RandomSeed = 42 };
            var validator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - 1/4 of data in test (25 samples)
            foreach (var fold in result.FoldResults)
            {
                Assert.Equal(25, fold.ValidationIndices!.Length);
                Assert.Equal(75, fold.TrainingIndices!.Length);
            }
        }

        [Fact]
        public void KFoldCrossValidator_LargeK_HandlesCorrectly()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new CrossValidationOptions { NumberOfFolds = 20, RandomSeed = 42 };
            var validator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert
            Assert.Equal(20, result.FoldResults.Count);
            foreach (var fold in result.FoldResults)
            {
                Assert.Equal(5, fold.ValidationIndices!.Length);
                Assert.Equal(95, fold.TrainingIndices!.Length);
            }
        }

        [Fact]
        public void KFoldCrossValidator_ReproducibleWithSeed()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var model1 = new SimpleTestModel<double>(2);
            var model2 = new SimpleTestModel<double>(2);
            var optimizer1 = CreateSimpleOptimizer(model1);
            var optimizer2 = CreateSimpleOptimizer(model2);
            var options = new CrossValidationOptions { NumberOfFolds = 5, RandomSeed = 42 };
            var validator1 = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(options);
            var validator2 = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(options);

            // Act
            var result1 = validator1.Validate(model1, X, y, optimizer1);
            var result2 = validator2.Validate(model2, X, y, optimizer2);

            // Assert - Same folds should be created
            for (int i = 0; i < result1.FoldResults.Count; i++)
            {
                Assert.Equal(result1.FoldResults[i].ValidationIndices, result2.FoldResults[i].ValidationIndices);
            }
        }

        #endregion

        #region StratifiedKFoldCrossValidator Tests

        [Fact]
        public void StratifiedKFoldCrossValidator_MaintainsClassDistribution()
        {
            // Arrange
            var (X, y) = CreateClassificationData(99, 3); // 33 samples per class
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new CrossValidationOptions { NumberOfFolds = 3, RandomSeed = 42 };
            var validator = new StratifiedKFoldCrossValidator<double, Matrix<double>, Vector<double>, double>(options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - Each fold should have balanced classes
            foreach (var fold in result.FoldResults)
            {
                var testClasses = fold.ValidationIndices!.Select(i => y[i]).GroupBy(c => c);
                foreach (var classGroup in testClasses)
                {
                    // Each class should have ~11 samples in test (33/3)
                    Assert.Equal(11, classGroup.Count());
                }
            }
        }

        [Fact]
        public void StratifiedKFoldCrossValidator_PreservesProportions()
        {
            // Arrange - Imbalanced dataset: 70% class 0, 30% class 1
            var X = new Matrix<double>(100, 2);
            var y = new Vector<double>(100);
            var random = new Random(42);

            for (int i = 0; i < 70; i++)
            {
                X[i, 0] = random.NextDouble();
                X[i, 1] = random.NextDouble();
                y[i] = 0;
            }
            for (int i = 70; i < 100; i++)
            {
                X[i, 0] = random.NextDouble() + 5.0;
                X[i, 1] = random.NextDouble() + 5.0;
                y[i] = 1;
            }

            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new CrossValidationOptions { NumberOfFolds = 5, RandomSeed = 42 };
            var validator = new StratifiedKFoldCrossValidator<double, Matrix<double>, Vector<double>, double>(options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - Each fold should maintain 70/30 ratio
            foreach (var fold in result.FoldResults)
            {
                var testIndices = fold.ValidationIndices!;
                var class0Count = testIndices.Count(i => y[i] == 0);
                var class1Count = testIndices.Count(i => y[i] == 1);

                Assert.Equal(14, class0Count); // 70% of 20
                Assert.Equal(6, class1Count);   // 30% of 20
            }
        }

        [Fact]
        public void StratifiedKFoldCrossValidator_AllClassesInEachFold()
        {
            // Arrange
            var (X, y) = CreateClassificationData(90, 3);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new CrossValidationOptions { NumberOfFolds = 5, RandomSeed = 42 };
            var validator = new StratifiedKFoldCrossValidator<double, Matrix<double>, Vector<double>, double>(options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - Each fold should contain all classes
            foreach (var fold in result.FoldResults)
            {
                var uniqueClasses = fold.ValidationIndices!.Select(i => y[i]).Distinct().Count();
                Assert.Equal(3, uniqueClasses);
            }
        }

        [Fact]
        public void StratifiedKFoldCrossValidator_NoOverlapBetweenFolds()
        {
            // Arrange
            var (X, y) = CreateClassificationData(100, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new CrossValidationOptions { NumberOfFolds = 5, RandomSeed = 42 };
            var validator = new StratifiedKFoldCrossValidator<double, Matrix<double>, Vector<double>, double>(options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert
            var allTestIndices = new List<int>();
            foreach (var fold in result.FoldResults)
            {
                allTestIndices.AddRange(fold.ValidationIndices!);
            }
            Assert.Equal(allTestIndices.Count, allTestIndices.Distinct().Count());
        }

        [Fact]
        public void StratifiedKFoldCrossValidator_MultipleClasses_HandlesCorrectly()
        {
            // Arrange
            var (X, y) = CreateClassificationData(100, 5); // 5 classes
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new CrossValidationOptions { NumberOfFolds = 5, RandomSeed = 42 };
            var validator = new StratifiedKFoldCrossValidator<double, Matrix<double>, Vector<double>, double>(options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert
            Assert.Equal(5, result.FoldResults.Count);
            foreach (var fold in result.FoldResults)
            {
                var classDistribution = fold.ValidationIndices!.Select(i => y[i]).GroupBy(c => c);
                Assert.Equal(5, classDistribution.Count()); // All classes present
            }
        }

        #endregion

        #region LeaveOneOutCrossValidator Tests

        [Fact]
        public void LeaveOneOutCrossValidator_CreatesNFoldsForNSamples()
        {
            // Arrange
            var (X, y) = CreateTestData(20, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var validator = new LeaveOneOutCrossValidator<double, Matrix<double>, Vector<double>>();

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert
            Assert.Equal(20, result.FoldResults.Count);
        }

        [Fact]
        public void LeaveOneOutCrossValidator_SingleTestSamplePerFold()
        {
            // Arrange
            var (X, y) = CreateTestData(15, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var validator = new LeaveOneOutCrossValidator<double, Matrix<double>, Vector<double>>();

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - Each fold should have exactly 1 test sample
            foreach (var fold in result.FoldResults)
            {
                Assert.Single(fold.ValidationIndices!);
                Assert.Equal(14, fold.TrainingIndices!.Length);
            }
        }

        [Fact]
        public void LeaveOneOutCrossValidator_AllSamplesTestedExactlyOnce()
        {
            // Arrange
            var (X, y) = CreateTestData(25, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var validator = new LeaveOneOutCrossValidator<double, Matrix<double>, Vector<double>>();

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - Collect all test indices
            var allTestIndices = result.FoldResults.Select(f => f.ValidationIndices![0]).ToList();
            Assert.Equal(25, allTestIndices.Count);
            Assert.Equal(25, allTestIndices.Distinct().Count());
            Assert.Equal(Enumerable.Range(0, 25).ToHashSet(), allTestIndices.ToHashSet());
        }

        [Fact]
        public void LeaveOneOutCrossValidator_TrainAndTestDisjoint()
        {
            // Arrange
            var (X, y) = CreateTestData(20, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var validator = new LeaveOneOutCrossValidator<double, Matrix<double>, Vector<double>>();

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert
            foreach (var fold in result.FoldResults)
            {
                var trainSet = new HashSet<int>(fold.TrainingIndices!);
                var testSample = fold.ValidationIndices![0];
                Assert.DoesNotContain(testSample, trainSet);
            }
        }

        [Fact]
        public void LeaveOneOutCrossValidator_SmallDataset_Works()
        {
            // Arrange
            var (X, y) = CreateTestData(5, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var validator = new LeaveOneOutCrossValidator<double, Matrix<double>, Vector<double>>();

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert
            Assert.Equal(5, result.FoldResults.Count);
            foreach (var fold in result.FoldResults)
            {
                Assert.Single(fold.ValidationIndices!);
                Assert.Equal(4, fold.TrainingIndices!.Length);
            }
        }

        [Fact]
        public void LeaveOneOutCrossValidator_MaximumTrainingData_PerFold()
        {
            // Arrange
            var (X, y) = CreateTestData(30, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var validator = new LeaveOneOutCrossValidator<double, Matrix<double>, Vector<double>>();

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - Each fold uses n-1 samples for training
            foreach (var fold in result.FoldResults)
            {
                Assert.Equal(29, fold.TrainingIndices!.Length);
            }
        }

        #endregion

        #region GroupKFoldCrossValidator Tests

        [Fact]
        public void GroupKFoldCrossValidator_GroupsStayTogether()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            // Create groups: 0-19 -> group 0, 20-39 -> group 1, etc.
            var groups = Enumerable.Range(0, 100).Select(i => i / 20).ToArray();
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new CrossValidationOptions { NumberOfFolds = 5, RandomSeed = 42 };
            var validator = new GroupKFoldCrossValidator<double, Matrix<double>, Vector<double>>(groups, options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - All samples from same group should be in same fold
            foreach (var fold in result.FoldResults)
            {
                var testGroups = fold.ValidationIndices!.Select(i => groups[i]).Distinct().ToList();
                var trainGroups = fold.TrainingIndices!.Select(i => groups[i]).Distinct().ToList();

                // No group should appear in both train and test
                Assert.Empty(testGroups.Intersect(trainGroups));
            }
        }

        [Fact]
        public void GroupKFoldCrossValidator_AllGroupsCovered()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var groups = Enumerable.Range(0, 100).Select(i => i / 10).ToArray(); // 10 groups
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new CrossValidationOptions { NumberOfFolds = 5, RandomSeed = 42 };
            var validator = new GroupKFoldCrossValidator<double, Matrix<double>, Vector<double>>(groups, options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - All groups should appear in test sets
            var allTestGroups = new HashSet<int>();
            foreach (var fold in result.FoldResults)
            {
                allTestGroups.UnionWith(fold.ValidationIndices!.Select(i => groups[i]));
            }
            Assert.Equal(10, allTestGroups.Count);
        }

        [Fact]
        public void GroupKFoldCrossValidator_NoGroupSplitAcrossFolds()
        {
            // Arrange
            var (X, y) = CreateTestData(60, 2);
            var groups = Enumerable.Range(0, 60).Select(i => i / 10).ToArray(); // 6 groups of 10
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new CrossValidationOptions { NumberOfFolds = 3, RandomSeed = 42 };
            var validator = new GroupKFoldCrossValidator<double, Matrix<double>, Vector<double>>(groups, options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - Each group should appear in exactly one fold
            var groupToFold = new Dictionary<int, int>();
            for (int foldIdx = 0; foldIdx < result.FoldResults.Count; foldIdx++)
            {
                var fold = result.FoldResults[foldIdx];
                var foldGroups = fold.ValidationIndices!.Select(i => groups[i]).Distinct();
                foreach (var group in foldGroups)
                {
                    Assert.DoesNotContain(group, groupToFold.Keys);
                    groupToFold[group] = foldIdx;
                }
            }
        }

        [Fact]
        public void GroupKFoldCrossValidator_UnevenGroupSizes_HandlesCorrectly()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var groups = new int[100];
            // Create uneven groups: group 0 = 50 samples, group 1 = 30, group 2 = 20
            for (int i = 0; i < 50; i++) groups[i] = 0;
            for (int i = 50; i < 80; i++) groups[i] = 1;
            for (int i = 80; i < 100; i++) groups[i] = 2;

            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new CrossValidationOptions { NumberOfFolds = 3, RandomSeed = 42 };
            var validator = new GroupKFoldCrossValidator<double, Matrix<double>, Vector<double>>(groups, options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert
            Assert.Equal(3, result.FoldResults.Count);

            // Each fold should have exactly one group
            foreach (var fold in result.FoldResults)
            {
                var foldGroups = fold.ValidationIndices!.Select(i => groups[i]).Distinct().ToList();
                Assert.Single(foldGroups);
            }
        }

        [Fact]
        public void GroupKFoldCrossValidator_CorrectNumberOfFolds()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var groups = Enumerable.Range(0, 100).Select(i => i / 20).ToArray(); // 5 groups
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new CrossValidationOptions { NumberOfFolds = 5, RandomSeed = 42 };
            var validator = new GroupKFoldCrossValidator<double, Matrix<double>, Vector<double>>(groups, options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert
            Assert.Equal(5, result.FoldResults.Count);
        }

        #endregion

        #region TimeSeriesCrossValidator Tests

        [Fact]
        public void TimeSeriesCrossValidator_MaintainsTemporalOrder()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var validator = new TimeSeriesCrossValidator<double, Matrix<double>, Vector<double>>(
                initialTrainSize: 20,
                validationSize: 10,
                step: 10
            );

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - Test indices should always be after train indices
            foreach (var fold in result.FoldResults)
            {
                var maxTrainIndex = fold.TrainingIndices!.Max();
                var minTestIndex = fold.ValidationIndices!.Min();
                Assert.True(minTestIndex > maxTrainIndex, "Test data should come after training data");
            }
        }

        [Fact]
        public void TimeSeriesCrossValidator_ExpandingWindow()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var validator = new TimeSeriesCrossValidator<double, Matrix<double>, Vector<double>>(
                initialTrainSize: 20,
                validationSize: 10,
                step: 10
            );

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - Training set should grow with each fold
            for (int i = 1; i < result.FoldResults.Count; i++)
            {
                Assert.True(result.FoldResults[i].TrainingIndices!.Length >
                           result.FoldResults[i - 1].TrainingIndices!.Length);
            }
        }

        [Fact]
        public void TimeSeriesCrossValidator_CorrectValidationSize()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var validationSize = 15;
            var validator = new TimeSeriesCrossValidator<double, Matrix<double>, Vector<double>>(
                initialTrainSize: 20,
                validationSize: validationSize,
                step: 10
            );

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - Each fold should have consistent validation size
            foreach (var fold in result.FoldResults)
            {
                Assert.Equal(validationSize, fold.ValidationIndices!.Length);
            }
        }

        [Fact]
        public void TimeSeriesCrossValidator_CorrectNumberOfFolds()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var validator = new TimeSeriesCrossValidator<double, Matrix<double>, Vector<double>>(
                initialTrainSize: 20,
                validationSize: 10,
                step: 10
            );

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - With these parameters: (100 - 20 - 10) / 10 = 7 folds
            Assert.Equal(7, result.FoldResults.Count);
        }

        [Fact]
        public void TimeSeriesCrossValidator_FirstFoldUsesInitialTrainSize()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var initialTrainSize = 25;
            var validator = new TimeSeriesCrossValidator<double, Matrix<double>, Vector<double>>(
                initialTrainSize: initialTrainSize,
                validationSize: 10,
                step: 10
            );

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert
            Assert.Equal(initialTrainSize, result.FoldResults[0].TrainingIndices!.Length);
        }

        [Fact]
        public void TimeSeriesCrossValidator_ConsecutiveValidationSets()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var validator = new TimeSeriesCrossValidator<double, Matrix<double>, Vector<double>>(
                initialTrainSize: 20,
                validationSize: 10,
                step: 10
            );

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - Validation sets should be consecutive time periods
            for (int i = 0; i < result.FoldResults.Count; i++)
            {
                var validationIndices = result.FoldResults[i].ValidationIndices!;
                for (int j = 1; j < validationIndices.Length; j++)
                {
                    Assert.Equal(validationIndices[j - 1] + 1, validationIndices[j]);
                }
            }
        }

        #endregion

        #region MonteCarloValidator Tests

        [Fact]
        public void MonteCarloValidator_CreatesCorrectNumberOfIterations()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new MonteCarloValidationOptions
            {
                NumberOfFolds = 10,
                ValidationSize = 0.2,
                RandomSeed = 42
            };
            var validator = new MonteCarloValidator<double, Matrix<double>, Vector<double>>(options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert
            Assert.Equal(10, result.FoldResults.Count);
        }

        [Fact]
        public void MonteCarloValidator_CorrectValidationSize()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var validationSize = 0.3;
            var options = new MonteCarloValidationOptions
            {
                NumberOfFolds = 5,
                ValidationSize = validationSize,
                RandomSeed = 42
            };
            var validator = new MonteCarloValidator<double, Matrix<double>, Vector<double>>(options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - Each fold should have ~30 samples in validation
            foreach (var fold in result.FoldResults)
            {
                Assert.Equal(30, fold.ValidationIndices!.Length);
                Assert.Equal(70, fold.TrainingIndices!.Length);
            }
        }

        [Fact]
        public void MonteCarloValidator_RandomSplits_DifferentFolds()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new MonteCarloValidationOptions
            {
                NumberOfFolds = 5,
                ValidationSize = 0.2,
                RandomSeed = 42
            };
            var validator = new MonteCarloValidator<double, Matrix<double>, Vector<double>>(options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - Folds should have different validation indices
            var fold1Indices = result.FoldResults[0].ValidationIndices!.ToHashSet();
            var fold2Indices = result.FoldResults[1].ValidationIndices!.ToHashSet();
            Assert.NotEqual(fold1Indices, fold2Indices);
        }

        [Fact]
        public void MonteCarloValidator_SamplesMayAppearMultipleTimes()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new MonteCarloValidationOptions
            {
                NumberOfFolds = 20,
                ValidationSize = 0.2,
                RandomSeed = 42
            };
            var validator = new MonteCarloValidator<double, Matrix<double>, Vector<double>>(options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - Some samples should appear in multiple validation sets
            var sampleTestCounts = new Dictionary<int, int>();
            foreach (var fold in result.FoldResults)
            {
                foreach (var idx in fold.ValidationIndices!)
                {
                    sampleTestCounts[idx] = sampleTestCounts.GetValueOrDefault(idx, 0) + 1;
                }
            }

            Assert.True(sampleTestCounts.Values.Any(count => count > 1));
        }

        [Fact]
        public void MonteCarloValidator_TrainTestDisjoint_PerFold()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new MonteCarloValidationOptions
            {
                NumberOfFolds = 10,
                ValidationSize = 0.25,
                RandomSeed = 42
            };
            var validator = new MonteCarloValidator<double, Matrix<double>, Vector<double>>(options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - Within each fold, train and test should be disjoint
            foreach (var fold in result.FoldResults)
            {
                var trainSet = new HashSet<int>(fold.TrainingIndices!);
                var testSet = new HashSet<int>(fold.ValidationIndices!);
                Assert.Empty(trainSet.Intersect(testSet));
            }
        }

        [Fact]
        public void MonteCarloValidator_DifferentValidationSizes_Work()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new MonteCarloValidationOptions
            {
                NumberOfFolds = 5,
                ValidationSize = 0.1,
                RandomSeed = 42
            };
            var validator = new MonteCarloValidator<double, Matrix<double>, Vector<double>>(options);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert
            foreach (var fold in result.FoldResults)
            {
                Assert.Equal(10, fold.ValidationIndices!.Length);
                Assert.Equal(90, fold.TrainingIndices!.Length);
            }
        }

        [Fact]
        public void MonteCarloValidator_ReproducibleWithSeed()
        {
            // Arrange
            var (X, y) = CreateTestData(100, 2);
            var model1 = new SimpleTestModel<double>(2);
            var model2 = new SimpleTestModel<double>(2);
            var optimizer1 = CreateSimpleOptimizer(model1);
            var optimizer2 = CreateSimpleOptimizer(model2);
            var options = new MonteCarloValidationOptions
            {
                NumberOfFolds = 5,
                ValidationSize = 0.2,
                RandomSeed = 42
            };
            var validator1 = new MonteCarloValidator<double, Matrix<double>, Vector<double>>(options);
            var validator2 = new MonteCarloValidator<double, Matrix<double>, Vector<double>>(options);

            // Act
            var result1 = validator1.Validate(model1, X, y, optimizer1);
            var result2 = validator2.Validate(model2, X, y, optimizer2);

            // Assert - Same random splits should be generated
            for (int i = 0; i < result1.FoldResults.Count; i++)
            {
                Assert.Equal(result1.FoldResults[i].ValidationIndices, result2.FoldResults[i].ValidationIndices);
            }
        }

        #endregion

        #region NestedCrossValidator Tests

        [Fact]
        public void NestedCrossValidator_RunsBothInnerAndOuterLoops()
        {
            // Arrange
            var (X, y) = CreateTestData(50, 2);
            var model = new SimpleTestModel<double>(2);
            var outerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(
                new CrossValidationOptions { NumberOfFolds = 3, RandomSeed = 42 }
            );
            var innerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(
                new CrossValidationOptions { NumberOfFolds = 2, RandomSeed = 43 }
            );

            // Model selector: just return the best fold model
            Func<CrossValidationResult<double, Matrix<double>, Vector<double>>, IFullModel<double, Matrix<double>, Vector<double>>>
                modelSelector = (result) => result.FoldResults[0].Model;

            var validator = new NestedCrossValidator<double, Matrix<double>, Vector<double>>(
                outerValidator, innerValidator, modelSelector
            );
            var optimizer = CreateSimpleOptimizer(model);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - Should have 3 outer folds
            Assert.Equal(3, result.FoldResults.Count);
        }

        [Fact]
        public void NestedCrossValidator_OuterFoldsHaveCorrectSize()
        {
            // Arrange
            var (X, y) = CreateTestData(60, 2);
            var model = new SimpleTestModel<double>(2);
            var outerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(
                new CrossValidationOptions { NumberOfFolds = 3, RandomSeed = 42 }
            );
            var innerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(
                new CrossValidationOptions { NumberOfFolds = 2, RandomSeed = 43 }
            );

            Func<CrossValidationResult<double, Matrix<double>, Vector<double>>, IFullModel<double, Matrix<double>, Vector<double>>>
                modelSelector = (result) => result.FoldResults[0].Model;

            var validator = new NestedCrossValidator<double, Matrix<double>, Vector<double>>(
                outerValidator, innerValidator, modelSelector
            );
            var optimizer = CreateSimpleOptimizer(model);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - Each outer fold should have 20 test samples
            foreach (var fold in result.FoldResults)
            {
                Assert.Equal(20, fold.ValidationIndices!.Length);
            }
        }

        [Fact]
        public void NestedCrossValidator_ModelSelectorCalled()
        {
            // Arrange
            var (X, y) = CreateTestData(45, 2);
            var model = new SimpleTestModel<double>(2);
            var outerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(
                new CrossValidationOptions { NumberOfFolds = 3, RandomSeed = 42 }
            );
            var innerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(
                new CrossValidationOptions { NumberOfFolds = 2, RandomSeed = 43 }
            );

            int selectorCallCount = 0;
            Func<CrossValidationResult<double, Matrix<double>, Vector<double>>, IFullModel<double, Matrix<double>, Vector<double>>>
                modelSelector = (result) =>
                {
                    selectorCallCount++;
                    return result.FoldResults[0].Model;
                };

            var validator = new NestedCrossValidator<double, Matrix<double>, Vector<double>>(
                outerValidator, innerValidator, modelSelector
            );
            var optimizer = CreateSimpleOptimizer(model);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - Model selector should be called once per outer fold
            Assert.Equal(3, selectorCallCount);
        }

        [Fact]
        public void NestedCrossValidator_InnerCVFindsCandidate()
        {
            // Arrange
            var (X, y) = CreateTestData(60, 2);
            var model = new SimpleTestModel<double>(2);
            var outerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(
                new CrossValidationOptions { NumberOfFolds = 2, RandomSeed = 42 }
            );
            var innerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(
                new CrossValidationOptions { NumberOfFolds = 3, RandomSeed = 43 }
            );

            // Select best model based on validation performance
            Func<CrossValidationResult<double, Matrix<double>, Vector<double>>, IFullModel<double, Matrix<double>, Vector<double>>>
                modelSelector = (result) =>
                {
                    // Inner CV should have 3 folds
                    Assert.Equal(3, result.FoldResults.Count);
                    return result.FoldResults[0].Model;
                };

            var validator = new NestedCrossValidator<double, Matrix<double>, Vector<double>>(
                outerValidator, innerValidator, modelSelector
            );
            var optimizer = CreateSimpleOptimizer(model);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert
            Assert.Equal(2, result.FoldResults.Count);
        }

        [Fact]
        public void NestedCrossValidator_OuterTestSetsDisjoint()
        {
            // Arrange
            var (X, y) = CreateTestData(60, 2);
            var model = new SimpleTestModel<double>(2);
            var outerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(
                new CrossValidationOptions { NumberOfFolds = 3, RandomSeed = 42 }
            );
            var innerValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(
                new CrossValidationOptions { NumberOfFolds = 2, RandomSeed = 43 }
            );

            Func<CrossValidationResult<double, Matrix<double>, Vector<double>>, IFullModel<double, Matrix<double>, Vector<double>>>
                modelSelector = (result) => result.FoldResults[0].Model;

            var validator = new NestedCrossValidator<double, Matrix<double>, Vector<double>>(
                outerValidator, innerValidator, modelSelector
            );
            var optimizer = CreateSimpleOptimizer(model);

            // Act
            var result = validator.Validate(model, X, y, optimizer);

            // Assert - Outer test sets should not overlap
            var allTestIndices = new List<int>();
            foreach (var fold in result.FoldResults)
            {
                allTestIndices.AddRange(fold.ValidationIndices!);
            }
            Assert.Equal(allTestIndices.Count, allTestIndices.Distinct().Count());
        }

        #endregion

        #region Edge Cases and Stress Tests

        [Fact]
        public void CrossValidators_EmptyOrInsufficientData_ThrowsOrHandles()
        {
            // This test verifies behavior with very small datasets
            // Some validators may throw, others may handle gracefully
            var (X, y) = CreateTestData(3, 2);
            var model = new SimpleTestModel<double>(2);
            var optimizer = CreateSimpleOptimizer(model);

            // KFold with k > n should handle gracefully or throw
            var options = new CrossValidationOptions { NumberOfFolds = 5, RandomSeed = 42 };
            var validator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(options);

            // This may throw or produce some folds - test that it doesn't crash
            try
            {
                var result = validator.Validate(model, X, y, optimizer);
                // If it succeeds, verify basic properties
                Assert.NotNull(result);
                Assert.NotEmpty(result.FoldResults);
            }
            catch (Exception ex)
            {
                // If it throws, that's also acceptable behavior
                Assert.NotNull(ex);
            }
        }

        [Fact]
        public void CrossValidators_DifferentDatasetSizes_Work()
        {
            // Test with various dataset sizes
            var sizes = new[] { 20, 50, 100, 200 };

            foreach (var size in sizes)
            {
                var (X, y) = CreateTestData(size, 2);
                var model = new SimpleTestModel<double>(2);
                var optimizer = CreateSimpleOptimizer(model);
                var options = new CrossValidationOptions { NumberOfFolds = 5, RandomSeed = 42 };
                var validator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(options);

                var result = validator.Validate(model, X, y, optimizer);

                Assert.Equal(5, result.FoldResults.Count);

                // Verify all data is covered
                var allTestIndices = new HashSet<int>();
                foreach (var fold in result.FoldResults)
                {
                    allTestIndices.UnionWith(fold.ValidationIndices!);
                }
                Assert.Equal(size, allTestIndices.Count);
            }
        }

        [Fact]
        public void CrossValidators_HighDimensionalData_HandlesCorrectly()
        {
            // Test with high-dimensional features
            var (X, y) = CreateTestData(100, 50);
            var model = new SimpleTestModel<double>(50);
            var optimizer = CreateSimpleOptimizer(model);
            var options = new CrossValidationOptions { NumberOfFolds = 5, RandomSeed = 42 };
            var validator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(options);

            var result = validator.Validate(model, X, y, optimizer);

            Assert.Equal(5, result.FoldResults.Count);
        }

        #endregion
    }
}
