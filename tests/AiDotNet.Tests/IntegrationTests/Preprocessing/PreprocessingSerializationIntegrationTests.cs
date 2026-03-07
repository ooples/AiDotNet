using AiDotNet.Preprocessing;
using AiDotNet.Preprocessing.Imputers;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Serialization;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Integration tests that verify preprocessing pipeline state is preserved
/// across serialization/deserialization round-trips using Newtonsoft.Json.
/// These tests use the same serialization settings as AiModelResult.Serialize/Deserialize.
/// </summary>
public class PreprocessingSerializationIntegrationTests
{
    /// <summary>
    /// Creates serializer settings matching those used by AiModelResult.
    /// </summary>
    private static JsonSerializerSettings CreateSettings()
    {
        return new JsonSerializerSettings
        {
            TypeNameHandling = TypeNameHandling.Auto,
            SerializationBinder = new SafeSerializationBinder(),
            Converters = JsonConverterRegistry.GetAllConverters(),
            ContractResolver = new AiModelResultContractResolver(),
            Formatting = Formatting.Indented
        };
    }

    /// <summary>
    /// Creates a simple test matrix with known values.
    /// </summary>
    private static Matrix<double> CreateTestMatrix()
    {
        return new Matrix<double>(new double[,]
        {
            { 1.0, 10.0, 100.0 },
            { 2.0, 20.0, 200.0 },
            { 3.0, 30.0, 300.0 },
            { 4.0, 40.0, 400.0 },
            { 5.0, 50.0, 500.0 }
        });
    }

    // ===================== StandardScaler Tests =====================

    [Fact]
    public void StandardScaler_RoundTrip_PreservesFittedState()
    {
        // Arrange
        var scaler = new StandardScaler<double>();
        var data = CreateTestMatrix();
        scaler.Fit(data);

        var settings = CreateSettings();

        // Act
        var json = JsonConvert.SerializeObject(scaler, settings);
        var deserialized = JsonConvert.DeserializeObject<StandardScaler<double>>(json, settings);

        // Assert
        Assert.NotNull(deserialized);
        Assert.True(deserialized.IsFitted);
        Assert.NotNull(deserialized.Mean);
        Assert.NotNull(deserialized.StandardDeviation);

        // Verify transform produces same results
        var originalResult = scaler.Transform(data);
        var deserializedResult = deserialized.Transform(data);

        for (int i = 0; i < originalResult.Rows; i++)
        {
            for (int j = 0; j < originalResult.Columns; j++)
            {
                Assert.Equal(originalResult[i, j], deserializedResult[i, j], 1e-10);
            }
        }
    }

    [Fact]
    public void StandardScaler_RoundTrip_PreservesConfiguration()
    {
        // Arrange - test with non-default options
        var scaler = new StandardScaler<double>(withMean: false, withStd: true);
        var data = CreateTestMatrix();
        scaler.Fit(data);

        var settings = CreateSettings();

        // Act
        var json = JsonConvert.SerializeObject(scaler, settings);
        var deserialized = JsonConvert.DeserializeObject<StandardScaler<double>>(json, settings);

        // Assert
        Assert.NotNull(deserialized);
        Assert.False(deserialized.WithMean);
        Assert.True(deserialized.WithStd);
    }

    // ===================== MinMaxScaler Tests =====================

    [Fact]
    public void MinMaxScaler_RoundTrip_PreservesFittedState()
    {
        // Arrange
        var scaler = new MinMaxScaler<double>();
        var data = CreateTestMatrix();
        scaler.Fit(data);

        var settings = CreateSettings();

        // Act
        var json = JsonConvert.SerializeObject(scaler, settings);
        var deserialized = JsonConvert.DeserializeObject<MinMaxScaler<double>>(json, settings);

        // Assert
        Assert.NotNull(deserialized);
        Assert.True(deserialized.IsFitted);
        Assert.NotNull(deserialized.DataMin);
        Assert.NotNull(deserialized.DataMax);

        // Verify transform produces same results
        var originalResult = scaler.Transform(data);
        var deserializedResult = deserialized.Transform(data);

        for (int i = 0; i < originalResult.Rows; i++)
        {
            for (int j = 0; j < originalResult.Columns; j++)
            {
                Assert.Equal(originalResult[i, j], deserializedResult[i, j], 1e-10);
            }
        }
    }

    [Fact]
    public void MinMaxScaler_CustomRange_RoundTrip_PreservesRange()
    {
        // Arrange
        var scaler = new MinMaxScaler<double>(-1.0, 1.0);
        var data = CreateTestMatrix();
        scaler.Fit(data);

        var settings = CreateSettings();

        // Act
        var json = JsonConvert.SerializeObject(scaler, settings);
        var deserialized = JsonConvert.DeserializeObject<MinMaxScaler<double>>(json, settings);

        // Assert
        Assert.NotNull(deserialized);

        // Verify the custom range [-1, 1] is preserved
        var originalResult = scaler.Transform(data);
        var deserializedResult = deserialized.Transform(data);

        // With range [-1, 1], min values should map to -1 and max to 1
        for (int j = 0; j < originalResult.Columns; j++)
        {
            // Find min and max in each column of transformed data
            double colMin = double.MaxValue, colMax = double.MinValue;
            for (int i = 0; i < deserializedResult.Rows; i++)
            {
                colMin = Math.Min(colMin, deserializedResult[i, j]);
                colMax = Math.Max(colMax, deserializedResult[i, j]);
            }
            Assert.Equal(-1.0, colMin, 1e-10);
            Assert.Equal(1.0, colMax, 1e-10);
        }

        // Verify transform produces same results
        for (int i = 0; i < originalResult.Rows; i++)
        {
            for (int j = 0; j < originalResult.Columns; j++)
            {
                Assert.Equal(originalResult[i, j], deserializedResult[i, j], 1e-10);
            }
        }
    }

    // ===================== RobustScaler Tests =====================

    [Fact]
    public void RobustScaler_RoundTrip_PreservesFittedState()
    {
        // Arrange
        var scaler = new RobustScaler<double>();
        var data = CreateTestMatrix();
        scaler.Fit(data);

        var settings = CreateSettings();

        // Act
        var json = JsonConvert.SerializeObject(scaler, settings);
        var deserialized = JsonConvert.DeserializeObject<RobustScaler<double>>(json, settings);

        // Assert
        Assert.NotNull(deserialized);
        Assert.True(deserialized.IsFitted);
        Assert.NotNull(deserialized.Median);
        Assert.NotNull(deserialized.InterquartileRange);

        // Verify transform produces same results
        var originalResult = scaler.Transform(data);
        var deserializedResult = deserialized.Transform(data);

        for (int i = 0; i < originalResult.Rows; i++)
        {
            for (int j = 0; j < originalResult.Columns; j++)
            {
                Assert.Equal(originalResult[i, j], deserializedResult[i, j], 1e-10);
            }
        }
    }

    // ===================== MaxAbsScaler Tests =====================

    [Fact]
    public void MaxAbsScaler_RoundTrip_PreservesFittedState()
    {
        // Arrange
        var scaler = new MaxAbsScaler<double>();
        var data = CreateTestMatrix();
        scaler.Fit(data);

        var settings = CreateSettings();

        // Act
        var json = JsonConvert.SerializeObject(scaler, settings);
        var deserialized = JsonConvert.DeserializeObject<MaxAbsScaler<double>>(json, settings);

        // Assert
        Assert.NotNull(deserialized);
        Assert.True(deserialized.IsFitted);
        Assert.NotNull(deserialized.MaxAbsolute);

        // Verify transform produces same results
        var originalResult = scaler.Transform(data);
        var deserializedResult = deserialized.Transform(data);

        for (int i = 0; i < originalResult.Rows; i++)
        {
            for (int j = 0; j < originalResult.Columns; j++)
            {
                Assert.Equal(originalResult[i, j], deserializedResult[i, j], 1e-10);
            }
        }
    }

    // ===================== SimpleImputer Tests =====================

    [Fact]
    public void SimpleImputer_RoundTrip_PreservesFittedState()
    {
        // Arrange
        var imputer = new SimpleImputer<double>(ImputationStrategy.Mean);
        var data = CreateTestMatrix();
        imputer.Fit(data);

        var settings = CreateSettings();

        // Act
        var json = JsonConvert.SerializeObject(imputer, settings);
        var deserialized = JsonConvert.DeserializeObject<SimpleImputer<double>>(json, settings);

        // Assert
        Assert.NotNull(deserialized);
        Assert.True(deserialized.IsFitted);
        Assert.Equal(ImputationStrategy.Mean, deserialized.Strategy);
        Assert.NotNull(deserialized.Statistics);

        // Verify transform produces same results
        var originalResult = imputer.Transform(data);
        var deserializedResult = deserialized.Transform(data);

        for (int i = 0; i < originalResult.Rows; i++)
        {
            for (int j = 0; j < originalResult.Columns; j++)
            {
                Assert.Equal(originalResult[i, j], deserializedResult[i, j], 1e-10);
            }
        }
    }

    // ===================== Pipeline Tests =====================

    [Fact]
    public void Pipeline_SingleStep_RoundTrip_PreservesFittedState()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add("scaler", new StandardScaler<double>());

        var data = CreateTestMatrix();
        pipeline.Fit(data);

        var settings = CreateSettings();

        // Act
        var json = JsonConvert.SerializeObject(pipeline, settings);
        var deserialized = JsonConvert.DeserializeObject<PreprocessingPipeline<double, Matrix<double>, Matrix<double>>>(json, settings);

        // Assert
        Assert.NotNull(deserialized);
        Assert.True(deserialized.IsFitted);
        Assert.Equal(1, deserialized.Count);
        Assert.Single(deserialized.Steps);
        Assert.Equal("scaler", deserialized.Steps[0].Name);

        // Verify transform produces same results
        var originalResult = pipeline.Transform(data);
        var deserializedResult = deserialized.Transform(data);

        for (int i = 0; i < originalResult.Rows; i++)
        {
            for (int j = 0; j < originalResult.Columns; j++)
            {
                Assert.Equal(originalResult[i, j], deserializedResult[i, j], 1e-10);
            }
        }
    }

    [Fact]
    public void Pipeline_MultiStep_RoundTrip_PreservesFittedState()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add("minmax", new MinMaxScaler<double>());
        pipeline.Add("standard", new StandardScaler<double>());

        var data = CreateTestMatrix();
        pipeline.Fit(data);

        var settings = CreateSettings();

        // Act
        var json = JsonConvert.SerializeObject(pipeline, settings);
        var deserialized = JsonConvert.DeserializeObject<PreprocessingPipeline<double, Matrix<double>, Matrix<double>>>(json, settings);

        // Assert
        Assert.NotNull(deserialized);
        Assert.True(deserialized.IsFitted);
        Assert.Equal(2, deserialized.Count);
        Assert.Equal(2, deserialized.Steps.Count);
        Assert.Equal("minmax", deserialized.Steps[0].Name);
        Assert.Equal("standard", deserialized.Steps[1].Name);

        // Verify transform produces same results
        var originalResult = pipeline.Transform(data);
        var deserializedResult = deserialized.Transform(data);

        for (int i = 0; i < originalResult.Rows; i++)
        {
            for (int j = 0; j < originalResult.Columns; j++)
            {
                Assert.Equal(originalResult[i, j], deserializedResult[i, j], 1e-10);
            }
        }
    }

    // ===================== PreprocessingInfo Tests =====================

    [Fact]
    public void PreprocessingInfo_RoundTrip_PreservesPipeline()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add("scaler", new StandardScaler<double>());

        var data = CreateTestMatrix();
        pipeline.Fit(data);

        var info = new PreprocessingInfo<double, Matrix<double>, Vector<double>>(pipeline);

        var settings = CreateSettings();

        // Act
        var json = JsonConvert.SerializeObject(info, settings);
        var deserialized = JsonConvert.DeserializeObject<PreprocessingInfo<double, Matrix<double>, Vector<double>>>(json, settings);

        // Assert
        Assert.NotNull(deserialized);
        Assert.NotNull(deserialized.Pipeline);
        Assert.True(deserialized.IsFitted);

        // Verify transform produces same results
        var originalResult = info.TransformFeatures(data);
        var deserializedResult = deserialized.TransformFeatures(data);

        for (int i = 0; i < originalResult.Rows; i++)
        {
            for (int j = 0; j < originalResult.Columns; j++)
            {
                Assert.Equal(originalResult[i, j], deserializedResult[i, j], 1e-10);
            }
        }
    }

    // ===================== InverseTransform Tests =====================

    [Fact]
    public void StandardScaler_InverseTransform_WorksAfterDeserialization()
    {
        // Arrange
        var scaler = new StandardScaler<double>();
        var data = CreateTestMatrix();
        scaler.Fit(data);

        var settings = CreateSettings();

        // Act - serialize, deserialize, then inverse transform
        var json = JsonConvert.SerializeObject(scaler, settings);
        var deserialized = JsonConvert.DeserializeObject<StandardScaler<double>>(json, settings);
        Assert.NotNull(deserialized);

        var transformed = deserialized.Transform(data);
        var recovered = deserialized.InverseTransform(transformed);

        // Assert - recovered data should match original data
        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                Assert.Equal(data[i, j], recovered[i, j], 1e-10);
            }
        }
    }

    [Fact]
    public void MinMaxScaler_InverseTransform_WorksAfterDeserialization()
    {
        // Arrange
        var scaler = new MinMaxScaler<double>();
        var data = CreateTestMatrix();
        scaler.Fit(data);

        var settings = CreateSettings();

        // Act
        var json = JsonConvert.SerializeObject(scaler, settings);
        var deserialized = JsonConvert.DeserializeObject<MinMaxScaler<double>>(json, settings);
        Assert.NotNull(deserialized);

        var transformed = deserialized.Transform(data);
        var recovered = deserialized.InverseTransform(transformed);

        // Assert
        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                Assert.Equal(data[i, j], recovered[i, j], 1e-10);
            }
        }
    }

    [Fact]
    public void Pipeline_InverseTransform_WorksAfterDeserialization()
    {
        // Arrange
        var pipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        pipeline.Add("minmax", new MinMaxScaler<double>());
        pipeline.Add("standard", new StandardScaler<double>());

        var data = CreateTestMatrix();
        pipeline.Fit(data);

        var settings = CreateSettings();

        // Act
        var json = JsonConvert.SerializeObject(pipeline, settings);
        var deserialized = JsonConvert.DeserializeObject<PreprocessingPipeline<double, Matrix<double>, Matrix<double>>>(json, settings);
        Assert.NotNull(deserialized);

        var transformed = deserialized.Transform(data);
        var recovered = deserialized.InverseTransform(transformed);

        // Assert
        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                Assert.Equal(data[i, j], recovered[i, j], 1e-8);
            }
        }
    }

    // ===================== DecimalScaler Tests =====================

    [Fact]
    public void DecimalScaler_RoundTrip_PreservesFittedState()
    {
        // Arrange
        var scaler = new DecimalScaler<double>();
        var data = CreateTestMatrix();
        scaler.Fit(data);

        var settings = CreateSettings();

        // Act
        var json = JsonConvert.SerializeObject(scaler, settings);
        var deserialized = JsonConvert.DeserializeObject<DecimalScaler<double>>(json, settings);

        // Assert
        Assert.NotNull(deserialized);
        Assert.True(deserialized.IsFitted);
        Assert.NotNull(deserialized.Scale);

        var originalResult = scaler.Transform(data);
        var deserializedResult = deserialized.Transform(data);

        for (int i = 0; i < originalResult.Rows; i++)
        {
            for (int j = 0; j < originalResult.Columns; j++)
            {
                Assert.Equal(originalResult[i, j], deserializedResult[i, j], 1e-10);
            }
        }
    }

    // ===================== LogScaler Tests =====================

    [Fact]
    public void LogScaler_RoundTrip_PreservesFittedState()
    {
        // Arrange
        var scaler = new LogScaler<double>();
        var data = CreateTestMatrix();
        scaler.Fit(data);

        var settings = CreateSettings();

        // Act
        var json = JsonConvert.SerializeObject(scaler, settings);
        var deserialized = JsonConvert.DeserializeObject<LogScaler<double>>(json, settings);

        // Assert
        Assert.NotNull(deserialized);
        Assert.True(deserialized.IsFitted);
        Assert.NotNull(deserialized.Shift);

        var originalResult = scaler.Transform(data);
        var deserializedResult = deserialized.Transform(data);

        for (int i = 0; i < originalResult.Rows; i++)
        {
            for (int j = 0; j < originalResult.Columns; j++)
            {
                Assert.Equal(originalResult[i, j], deserializedResult[i, j], 1e-10);
            }
        }
    }

    // ===================== LpNormScaler Tests =====================

    [Fact]
    public void LpNormScaler_RoundTrip_PreservesFittedState()
    {
        // Arrange
        var scaler = new LpNormScaler<double>(p: 2.0);
        var data = CreateTestMatrix();
        scaler.Fit(data);

        var settings = CreateSettings();

        // Act
        var json = JsonConvert.SerializeObject(scaler, settings);
        var deserialized = JsonConvert.DeserializeObject<LpNormScaler<double>>(json, settings);

        // Assert
        Assert.NotNull(deserialized);
        Assert.True(deserialized.IsFitted);
        Assert.Equal(2.0, deserialized.P);

        var originalResult = scaler.Transform(data);
        var deserializedResult = deserialized.Transform(data);

        for (int i = 0; i < originalResult.Rows; i++)
        {
            for (int j = 0; j < originalResult.Columns; j++)
            {
                Assert.Equal(originalResult[i, j], deserializedResult[i, j], 1e-10);
            }
        }
    }

    // ===================== Normalizer Tests =====================

    [Fact]
    public void Normalizer_RoundTrip_PreservesFittedState()
    {
        // Arrange
        var normalizer = new Normalizer<double>(NormType.L2);
        var data = CreateTestMatrix();
        normalizer.Fit(data);

        var settings = CreateSettings();

        // Act
        var json = JsonConvert.SerializeObject(normalizer, settings);
        var deserialized = JsonConvert.DeserializeObject<Normalizer<double>>(json, settings);

        // Assert
        Assert.NotNull(deserialized);
        Assert.True(deserialized.IsFitted);
        Assert.Equal(NormType.L2, deserialized.NormType);

        var originalResult = normalizer.Transform(data);
        var deserializedResult = deserialized.Transform(data);

        for (int i = 0; i < originalResult.Rows; i++)
        {
            for (int j = 0; j < originalResult.Columns; j++)
            {
                Assert.Equal(originalResult[i, j], deserializedResult[i, j], 1e-10);
            }
        }
    }

    // ===================== LogMeanVarianceScaler Tests =====================

    [Fact]
    public void LogMeanVarianceScaler_RoundTrip_PreservesFittedState()
    {
        // Arrange
        var scaler = new LogMeanVarianceScaler<double>();
        var data = CreateTestMatrix();
        scaler.Fit(data);

        var settings = CreateSettings();

        // Act
        var json = JsonConvert.SerializeObject(scaler, settings);
        var deserialized = JsonConvert.DeserializeObject<LogMeanVarianceScaler<double>>(json, settings);

        // Assert
        Assert.NotNull(deserialized);
        Assert.True(deserialized.IsFitted);
        Assert.NotNull(deserialized.Shift);
        Assert.NotNull(deserialized.LogMean);
        Assert.NotNull(deserialized.LogStdDev);

        var originalResult = scaler.Transform(data);
        var deserializedResult = deserialized.Transform(data);

        for (int i = 0; i < originalResult.Rows; i++)
        {
            for (int j = 0; j < originalResult.Columns; j++)
            {
                Assert.Equal(originalResult[i, j], deserializedResult[i, j], 1e-10);
            }
        }
    }

    // ===================== GlobalContrastScaler Tests =====================

    [Fact]
    public void GlobalContrastScaler_RoundTrip_PreservesFittedState()
    {
        // Arrange
        var scaler = new GlobalContrastScaler<double>();
        var data = CreateTestMatrix();
        scaler.Fit(data);

        var settings = CreateSettings();

        // Act
        var json = JsonConvert.SerializeObject(scaler, settings);
        var deserialized = JsonConvert.DeserializeObject<GlobalContrastScaler<double>>(json, settings);

        // Assert
        Assert.NotNull(deserialized);
        Assert.True(deserialized.IsFitted);
        Assert.NotNull(deserialized.Mean);
        Assert.NotNull(deserialized.StdDev);

        var originalResult = scaler.Transform(data);
        var deserializedResult = deserialized.Transform(data);

        for (int i = 0; i < originalResult.Rows; i++)
        {
            for (int j = 0; j < originalResult.Columns; j++)
            {
                Assert.Equal(originalResult[i, j], deserializedResult[i, j], 1e-10);
            }
        }
    }

    // ===================== ColumnIndices Round-Trip Test =====================

    [Fact]
    public void StandardScaler_WithColumnIndices_RoundTripPreservesIndices()
    {
        // Arrange: Create a scaler that only operates on columns 0 and 2
        var scaler = new StandardScaler<double>(columnIndices: new[] { 0, 2 });
        var data = CreateTestMatrix(); // 5 rows x 3 columns
        scaler.Fit(data);

        var settings = CreateSettings();

        // Act
        var json = JsonConvert.SerializeObject(scaler, settings);
        var deserialized = JsonConvert.DeserializeObject<StandardScaler<double>>(json, settings);

        // Assert: ColumnIndices round-tripped correctly
        Assert.NotNull(deserialized);
        Assert.NotNull(deserialized.ColumnIndices);
        Assert.Equal(new[] { 0, 2 }, deserialized.ColumnIndices);
        Assert.True(deserialized.IsFitted);

        // Transform should produce identical results
        var originalResult = scaler.Transform(data);
        var deserializedResult = deserialized.Transform(data);

        for (int i = 0; i < originalResult.Rows; i++)
        {
            for (int j = 0; j < originalResult.Columns; j++)
            {
                Assert.Equal(originalResult[i, j], deserializedResult[i, j], 1e-10);
            }
        }

        // Column 1 (not in ColumnIndices) should be unchanged
        for (int i = 0; i < data.Rows; i++)
        {
            Assert.Equal(data[i, 1], deserializedResult[i, 1], 1e-10);
        }
    }

    // ===================== TargetPipeline Round-Trip Test =====================

    [Fact]
    public void PreprocessingInfo_WithNullTargetPipeline_RoundTripPreservesNullTarget()
    {
        // Arrange: PreprocessingInfo with feature pipeline but null target pipeline
        var featurePipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        featurePipeline.Add("scaler", new StandardScaler<double>());
        var data = CreateTestMatrix();
        featurePipeline.Fit(data);

        var info = new PreprocessingInfo<double, Matrix<double>, Vector<double>>(
            featurePipeline,
            targetPipeline: null
        );

        var settings = CreateSettings();

        // Act
        var json = JsonConvert.SerializeObject(info, settings);
        var deserialized = JsonConvert.DeserializeObject<PreprocessingInfo<double, Matrix<double>, Vector<double>>>(json, settings);

        // Assert: null TargetPipeline preserved
        Assert.NotNull(deserialized);
        Assert.True(deserialized.IsFitted);
        Assert.Null(deserialized.TargetPipeline);
        Assert.False(deserialized.IsTargetFitted);

        // Feature pipeline still works
        var originalResult = info.TransformFeatures(data);
        var deserializedResult = deserialized.TransformFeatures(data);

        for (int i = 0; i < originalResult.Rows; i++)
        {
            for (int j = 0; j < originalResult.Columns; j++)
            {
                Assert.Equal(originalResult[i, j], deserializedResult[i, j], 1e-10);
            }
        }
    }

    [Fact]
    public void PreprocessingInfo_WithTargetPipeline_RoundTripPreservesTarget()
    {
        // Arrange: Create PreprocessingInfo where TOutput is also Matrix<double>
        // so we can use a StandardScaler as the target pipeline transformer
        var featurePipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        featurePipeline.Add("feature_scaler", new StandardScaler<double>());

        var targetPipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        targetPipeline.Add("target_scaler", new MinMaxScaler<double>());

        var featureData = CreateTestMatrix();
        var targetData = new Matrix<double>(new double[,] { { 10 }, { 20 }, { 30 }, { 40 }, { 50 } });

        featurePipeline.Fit(featureData);
        targetPipeline.Fit(targetData);

        var info = new PreprocessingInfo<double, Matrix<double>, Matrix<double>>(
            featurePipeline,
            targetPipeline
        );

        var settings = CreateSettings();

        // Act
        var json = JsonConvert.SerializeObject(info, settings);
        var deserialized = JsonConvert.DeserializeObject<PreprocessingInfo<double, Matrix<double>, Matrix<double>>>(json, settings);

        // Assert: Both pipelines survive round-trip
        Assert.NotNull(deserialized);
        Assert.True(deserialized.IsFitted);
        Assert.NotNull(deserialized.TargetPipeline);
        Assert.True(deserialized.IsTargetFitted);

        // Verify feature transform
        var originalFeature = info.TransformFeatures(featureData);
        var deserializedFeature = deserialized.TransformFeatures(featureData);
        for (int i = 0; i < originalFeature.Rows; i++)
            for (int j = 0; j < originalFeature.Columns; j++)
                Assert.Equal(originalFeature[i, j], deserializedFeature[i, j], 1e-10);

        // Verify target transform
        var originalTarget = targetPipeline.Transform(targetData);
        var deserializedTarget = deserialized.TargetPipeline.Transform(targetData);
        for (int i = 0; i < originalTarget.Rows; i++)
            for (int j = 0; j < originalTarget.Columns; j++)
                Assert.Equal(originalTarget[i, j], deserializedTarget[i, j], 1e-10);
    }
}
