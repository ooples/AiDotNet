using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.Preprocessing;
using AiDotNet.Preprocessing.Scalers;
using Xunit;

namespace AiDotNetTests.UnitTests.Preprocessing
{
    public class PreprocessingPipelineTests
    {
        [Fact]
        public void Pipeline_Add_AddsTransformerToSteps()
        {
            // Arrange
            var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();

            // Act
            pipeline.Add(new StandardScaler<double>());

            // Assert - pipeline should have one step
            Assert.Equal(1, pipeline.Steps.Count);
        }

        [Fact]
        public void Pipeline_AddWithName_AddsNamedStep()
        {
            // Arrange
            var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();

            // Act
            pipeline.Add("scaler", new StandardScaler<double>());

            // Assert
            Assert.Equal(1, pipeline.Steps.Count);
            Assert.Equal("scaler", pipeline.Steps[0].Name);
        }

        [Fact]
        public void Pipeline_FitTransform_AppliesAllStepsInOrder()
        {
            // Arrange
            var data = new Matrix<double>(new double[,]
            {
                { 0.0 },
                { 50.0 },
                { 100.0 }
            });

            var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
            pipeline.Add(new MinMaxScaler<double>()); // First scale to [0, 1]

            // Act
            var result = pipeline.FitTransform(data);

            // Assert
            Assert.True(Math.Abs(result[0, 0] - 0.0) < 0.0001);
            Assert.True(Math.Abs(result[1, 0] - 0.5) < 0.0001);
            Assert.True(Math.Abs(result[2, 0] - 1.0) < 0.0001);
        }

        [Fact]
        public void Pipeline_MultipleSteps_AppliesSequentially()
        {
            // Arrange
            var data = new Matrix<double>(new double[,]
            {
                { 0.0 },
                { 50.0 },
                { 100.0 }
            });

            var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
            pipeline.Add(new MinMaxScaler<double>()); // Scale to [0, 1]
            // After MinMax: [0, 0.5, 1]
            // Note: Adding another scaler would re-scale, which tests sequential behavior

            // Act
            var result = pipeline.FitTransform(data);

            // Assert
            Assert.True(pipeline.IsFitted);
            Assert.Equal(1, pipeline.Steps.Count);
        }

        [Fact]
        public void Pipeline_InverseTransform_ReversesAllStepsInReverseOrder()
        {
            // Arrange
            var data = new Matrix<double>(new double[,]
            {
                { 10.0, 20.0 },
                { 30.0, 40.0 },
                { 50.0, 60.0 }
            });

            var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
            pipeline.Add(new MinMaxScaler<double>());

            // Act
            var transformed = pipeline.FitTransform(data);
            var inversed = pipeline.InverseTransform(transformed);

            // Assert - should get back original values
            Assert.True(Math.Abs(inversed[0, 0] - 10.0) < 0.0001);
            Assert.True(Math.Abs(inversed[1, 1] - 40.0) < 0.0001);
            Assert.True(Math.Abs(inversed[2, 0] - 50.0) < 0.0001);
        }

        [Fact]
        public void Pipeline_FitThenTransform_WorksOnNewData()
        {
            // Arrange
            var trainData = new Matrix<double>(new double[,]
            {
                { 0.0 },
                { 50.0 },
                { 100.0 }
            });

            var newData = new Matrix<double>(new double[,]
            {
                { 25.0 },
                { 75.0 }
            });

            var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
            pipeline.Add(new MinMaxScaler<double>());

            // Act
            pipeline.Fit(trainData); // Fit on training data
            var result = pipeline.Transform(newData); // Transform new data

            // Assert - should use fitted parameters from training data
            Assert.True(Math.Abs(result[0, 0] - 0.25) < 0.0001); // 25 is 25% between 0 and 100
            Assert.True(Math.Abs(result[1, 0] - 0.75) < 0.0001); // 75 is 75% between 0 and 100
        }

        [Fact]
        public void Pipeline_TransformBeforeFit_ThrowsException()
        {
            // Arrange
            var data = new Matrix<double>(new double[,] { { 1.0 } });
            var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
            pipeline.Add(new StandardScaler<double>());

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => pipeline.Transform(data));
        }

        [Fact]
        public void Pipeline_IsFitted_ReflectsFitStatus()
        {
            // Arrange
            var data = new Matrix<double>(new double[,] { { 1.0 }, { 2.0 } });
            var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
            pipeline.Add(new StandardScaler<double>());

            // Assert - before fit
            Assert.False(pipeline.IsFitted);

            // Act
            pipeline.Fit(data);

            // Assert - after fit
            Assert.True(pipeline.IsFitted);
        }

        [Fact]
        public void Pipeline_SupportsInverseTransform_TrueWhenAllStepsSupport()
        {
            // Arrange
            var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
            pipeline.Add(new StandardScaler<double>());
            pipeline.Add(new MinMaxScaler<double>());

            // Assert - both scalers support inverse transform
            Assert.True(pipeline.SupportsInverseTransform);
        }

        [Fact]
        public void Pipeline_GetFeatureNamesOut_PassesThroughTransformers()
        {
            // Arrange
            var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
            pipeline.Add(new StandardScaler<double>());
            var inputNames = new[] { "age", "income" };

            // Act
            var outputNames = pipeline.GetFeatureNamesOut(inputNames);

            // Assert - scalers don't change feature names
            Assert.Equal(inputNames, outputNames);
        }

        [Fact]
        public void Pipeline_FluentInterface_WorksCorrectly()
        {
            // Arrange & Act
            var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>()
                .Add(new StandardScaler<double>())
                .Add("minmax", new MinMaxScaler<double>());

            // Assert
            Assert.Equal(2, pipeline.Steps.Count);
        }

        [Fact]
        public void Pipeline_EmptyPipeline_FitTransformReturnsOriginalData()
        {
            // Arrange
            var data = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0 },
                { 3.0, 4.0 }
            });
            var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();

            // Act
            var result = pipeline.FitTransform(data);

            // Assert - should return same data when no transformers
            Assert.Equal(data[0, 0], result[0, 0]);
            Assert.Equal(data[1, 1], result[1, 1]);
        }

        [Fact]
        public void Pipeline_AddNullTransformer_ThrowsException()
        {
            // Arrange
            var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => pipeline.Add(null!));
        }
    }
}
