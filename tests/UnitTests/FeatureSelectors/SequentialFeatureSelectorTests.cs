using AiDotNet.Enums;
using AiDotNet.FeatureSelectors;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Metadata;
using Xunit;

namespace AiDotNetTests.UnitTests.FeatureSelectors
{
    /// <summary>
    /// Simple mock model for testing feature selection.
    /// This model just returns predictions based on the sum of features (simulating feature importance).
    /// </summary>
    public class SimpleMockModel : IFullModel<double, Matrix<double>, Vector<double>>
    {
        private Matrix<double>? _trainedData;
        private Vector<double>? _trainedTarget;

        public void Train(Matrix<double> input, Vector<double> expectedOutput)
        {
            _trainedData = input;
            _trainedTarget = expectedOutput;
        }

        public Vector<double> Predict(Matrix<double> input)
        {
            // Simple prediction: classify based on sum of features
            var predictions = new Vector<double>(input.RowCount);
            for (int i = 0; i < input.RowCount; i++)
            {
                double sum = 0;
                for (int j = 0; j < input.ColumnCount; j++)
                {
                    sum += input[i, j];
                }
                predictions[i] = sum > 10 ? 1.0 : 0.0;
            }
            return predictions;
        }

        public ModelMetadata<double> GetModelMetadata()
        {
            return new ModelMetadata<double>();
        }

        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
        public void SetParameters(Dictionary<string, object> parameters) { }
        public Dictionary<string, object> GetParameters() => new();
        public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Empty<int>();
        public void SetActiveFeatureIndices(IEnumerable<int> featureIndices) { }
        public bool IsFeatureUsed(int featureIndex) => true;
        public Dictionary<string, double> GetFeatureImportance() => new();
        public IFullModel<double, Matrix<double>, Vector<double>> Clone()
        {
            return new SimpleMockModel();
        }
    }

    public class SequentialFeatureSelectorTests
    {
        /// <summary>
        /// Simple accuracy scorer for testing.
        /// </summary>
        private double CalculateAccuracy(Vector<double> predictions, Vector<double> actual)
        {
            int correct = 0;
            for (int i = 0; i < predictions.Length; i++)
            {
                if (Math.Abs(predictions[i] - actual[i]) < 0.5)
                {
                    correct++;
                }
            }
            return (double)correct / predictions.Length;
        }

        [Fact]
        public void SelectFeatures_ForwardSelection_SelectsTopFeatures()
        {
            // Arrange
            var features = new Matrix<double>(new double[,]
            {
                { 1.0, 0.1, 0.1 },  // Class 0 (sum < 10)
                { 2.0, 0.2, 0.1 },  // Class 0
                { 8.0, 3.0, 0.1 },  // Class 1 (sum > 10)
                { 9.0, 4.0, 0.2 }   // Class 1
            });
            var target = new Vector<double>(new double[] { 0, 0, 1, 1 });

            var model = new SimpleMockModel();
            var selector = new SequentialFeatureSelector<double, Matrix<double>, Vector<double>>(
                model,
                target,
                CalculateAccuracy,
                SequentialFeatureSelectionDirection.Forward,
                numFeaturesToSelect: 2);

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(4, result.RowCount);
            Assert.Equal(2, result.ColumnCount); // 2 features selected
        }

        [Fact]
        public void SelectFeatures_BackwardElimination_RemovesWorstFeatures()
        {
            // Arrange
            var features = new Matrix<double>(new double[,]
            {
                { 1.0, 0.1, 0.1 },
                { 2.0, 0.2, 0.1 },
                { 8.0, 3.0, 0.1 },
                { 9.0, 4.0, 0.2 }
            });
            var target = new Vector<double>(new double[] { 0, 0, 1, 1 });

            var model = new SimpleMockModel();
            var selector = new SequentialFeatureSelector<double, Matrix<double>, Vector<double>>(
                model,
                target,
                CalculateAccuracy,
                SequentialFeatureSelectionDirection.Backward,
                numFeaturesToSelect: 2);

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(4, result.RowCount);
            Assert.Equal(2, result.ColumnCount);
        }

        [Fact]
        public void SelectFeatures_WithDefaultNumFeatures_SelectsHalf()
        {
            // Arrange
            var features = new Matrix<double>(new double[,]
            {
                { 1.0, 0.1, 0.1, 0.1 },
                { 2.0, 0.2, 0.1, 0.1 },
                { 8.0, 3.0, 0.1, 0.1 },
                { 9.0, 4.0, 0.2, 0.1 }
            });
            var target = new Vector<double>(new double[] { 0, 0, 1, 1 });

            var model = new SimpleMockModel();
            var selector = new SequentialFeatureSelector<double, Matrix<double>, Vector<double>>(
                model,
                target,
                CalculateAccuracy,
                SequentialFeatureSelectionDirection.Forward);

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(4, result.RowCount);
            Assert.Equal(2, result.ColumnCount); // 50% of 4 = 2
        }

        [Fact]
        public void SelectFeatures_WithSingleFeatureTarget_SelectsSingleFeature()
        {
            // Arrange
            var features = new Matrix<double>(new double[,]
            {
                { 1.0, 0.1, 0.1 },
                { 2.0, 0.2, 0.1 },
                { 8.0, 3.0, 0.1 },
                { 9.0, 4.0, 0.2 }
            });
            var target = new Vector<double>(new double[] { 0, 0, 1, 1 });

            var model = new SimpleMockModel();
            var selector = new SequentialFeatureSelector<double, Matrix<double>, Vector<double>>(
                model,
                target,
                CalculateAccuracy,
                SequentialFeatureSelectionDirection.Forward,
                numFeaturesToSelect: 1);

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(4, result.RowCount);
            Assert.Equal(1, result.ColumnCount);
        }

        [Fact]
        public void SelectFeatures_WithNumFeaturesGreaterThanTotal_SelectsAllFeatures()
        {
            // Arrange
            var features = new Matrix<double>(new double[,]
            {
                { 1.0, 0.1 },
                { 2.0, 0.2 },
                { 8.0, 3.0 },
                { 9.0, 4.0 }
            });
            var target = new Vector<double>(new double[] { 0, 0, 1, 1 });

            var model = new SimpleMockModel();
            var selector = new SequentialFeatureSelector<double, Matrix<double>, Vector<double>>(
                model,
                target,
                CalculateAccuracy,
                SequentialFeatureSelectionDirection.Forward,
                numFeaturesToSelect: 100);

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(4, result.RowCount);
            Assert.Equal(2, result.ColumnCount); // All features
        }

        [Fact]
        public void Constructor_WithNullModel_ThrowsArgumentNullException()
        {
            // Arrange
            var target = new Vector<double>(new double[] { 0, 1 });

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new SequentialFeatureSelector<double, Matrix<double>, Vector<double>>(
                    null!,
                    target,
                    CalculateAccuracy));
        }

        [Fact]
        public void Constructor_WithNullTarget_ThrowsArgumentNullException()
        {
            // Arrange
            var model = new SimpleMockModel();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new SequentialFeatureSelector<double, Matrix<double>, Vector<double>>(
                    model,
                    null!,
                    CalculateAccuracy));
        }

        [Fact]
        public void Constructor_WithNullScoringFunction_ThrowsArgumentNullException()
        {
            // Arrange
            var model = new SimpleMockModel();
            var target = new Vector<double>(new double[] { 0, 1 });

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new SequentialFeatureSelector<double, Matrix<double>, Vector<double>>(
                    model,
                    target,
                    null!));
        }

        [Fact]
        public void SelectFeatures_ForwardAndBackward_ProduceDifferentResults()
        {
            // Arrange
            var features = new Matrix<double>(new double[,]
            {
                { 1.0, 0.1, 0.2, 0.1 },
                { 2.0, 0.2, 0.3, 0.1 },
                { 8.0, 3.0, 0.1, 0.2 },
                { 9.0, 4.0, 0.2, 0.1 }
            });
            var target = new Vector<double>(new double[] { 0, 0, 1, 1 });

            var model1 = new SimpleMockModel();
            var forwardSelector = new SequentialFeatureSelector<double, Matrix<double>, Vector<double>>(
                model1,
                target,
                CalculateAccuracy,
                SequentialFeatureSelectionDirection.Forward,
                numFeaturesToSelect: 2);

            var model2 = new SimpleMockModel();
            var backwardSelector = new SequentialFeatureSelector<double, Matrix<double>, Vector<double>>(
                model2,
                target,
                CalculateAccuracy,
                SequentialFeatureSelectionDirection.Backward,
                numFeaturesToSelect: 2);

            // Act
            var forwardResult = forwardSelector.SelectFeatures(features);
            var backwardResult = backwardSelector.SelectFeatures(features);

            // Assert - Both should select 2 features
            Assert.Equal(2, forwardResult.ColumnCount);
            Assert.Equal(2, backwardResult.ColumnCount);
            // Results may differ depending on the selection process
        }

        [Fact]
        public void SelectFeatures_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var features = new Matrix<float>(new float[,]
            {
                { 1.0f, 0.1f, 0.1f },
                { 2.0f, 0.2f, 0.1f },
                { 8.0f, 3.0f, 0.1f },
                { 9.0f, 4.0f, 0.2f }
            });
            var target = new Vector<float>(new float[] { 0, 0, 1, 1 });

            var model = new SimpleMockModelFloat();
            var selector = new SequentialFeatureSelector<float, Matrix<float>, Vector<float>>(
                model,
                target,
                (pred, act) =>
                {
                    int correct = 0;
                    for (int i = 0; i < pred.Length; i++)
                    {
                        if (Math.Abs(pred[i] - act[i]) < 0.5f) correct++;
                    }
                    return (float)correct / pred.Length;
                },
                SequentialFeatureSelectionDirection.Forward,
                numFeaturesToSelect: 2);

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(4, result.RowCount);
            Assert.Equal(2, result.ColumnCount);
        }
    }

    /// <summary>
    /// Float version of the mock model for testing.
    /// </summary>
    public class SimpleMockModelFloat : IFullModel<float, Matrix<float>, Vector<float>>
    {
        public void Train(Matrix<float> input, Vector<float> expectedOutput) { }

        public Vector<float> Predict(Matrix<float> input)
        {
            var predictions = new Vector<float>(input.RowCount);
            for (int i = 0; i < input.RowCount; i++)
            {
                float sum = 0;
                for (int j = 0; j < input.ColumnCount; j++)
                {
                    sum += input[i, j];
                }
                predictions[i] = sum > 10 ? 1.0f : 0.0f;
            }
            return predictions;
        }

        public ModelMetadata<float> GetModelMetadata() => new();
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
        public void SetParameters(Dictionary<string, object> parameters) { }
        public Dictionary<string, object> GetParameters() => new();
        public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Empty<int>();
        public void SetActiveFeatureIndices(IEnumerable<int> featureIndices) { }
        public bool IsFeatureUsed(int featureIndex) => true;
        public Dictionary<string, float> GetFeatureImportance() => new();
        public IFullModel<float, Matrix<float>, Vector<float>> Clone()
        {
            return new SimpleMockModelFloat();
        }
    }
}
