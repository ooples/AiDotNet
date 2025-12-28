using Xunit;
using AiDotNet.FeatureSelectors;
using AiDotNet.Enums;

namespace AiDotNet.Tests.IntegrationTests.Helpers;

/// <summary>
/// Integration tests for FeatureSelectorHelper to verify feature selection utility operations.
/// </summary>
public class FeatureSelectorHelperIntegrationTests
{
    #region ExtractFeatureVector Tests - Matrix

    [Fact]
    public void ExtractFeatureVector_Matrix_ExtractsCorrectColumn()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 10, 100 },
            { 2, 20, 200 },
            { 3, 30, 300 },
            { 4, 40, 400 }
        });

        var featureVector = FeatureSelectorHelper<double, Matrix<double>>.ExtractFeatureVector(
            matrix,
            featureIndex: 1,
            numSamples: 4,
            higherDimensionStrategy: FeatureExtractionStrategy.Mean,
            dimensionWeights: new Dictionary<int, double>());

        Assert.Equal(4, featureVector.Length);
        Assert.Equal(10, featureVector[0]);
        Assert.Equal(20, featureVector[1]);
        Assert.Equal(30, featureVector[2]);
        Assert.Equal(40, featureVector[3]);
    }

    [Fact]
    public void ExtractFeatureVector_Matrix_FirstColumn_ReturnsCorrectValues()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 5, 10 },
            { 15, 20 },
            { 25, 30 }
        });

        var featureVector = FeatureSelectorHelper<double, Matrix<double>>.ExtractFeatureVector(
            matrix,
            featureIndex: 0,
            numSamples: 3,
            higherDimensionStrategy: FeatureExtractionStrategy.Mean,
            dimensionWeights: new Dictionary<int, double>());

        Assert.Equal(3, featureVector.Length);
        Assert.Equal(5, featureVector[0]);
        Assert.Equal(15, featureVector[1]);
        Assert.Equal(25, featureVector[2]);
    }

    [Fact]
    public void ExtractFeatureVector_Matrix_LastColumn_ReturnsCorrectValues()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 }
        });

        var featureVector = FeatureSelectorHelper<double, Matrix<double>>.ExtractFeatureVector(
            matrix,
            featureIndex: 2,
            numSamples: 2,
            higherDimensionStrategy: FeatureExtractionStrategy.Mean,
            dimensionWeights: new Dictionary<int, double>());

        Assert.Equal(2, featureVector.Length);
        Assert.Equal(3, featureVector[0]);
        Assert.Equal(6, featureVector[1]);
    }

    [Fact]
    public void ExtractFeatureVector_Float_Matrix_ReturnsCorrectValues()
    {
        var matrix = new Matrix<float>(new float[,]
        {
            { 1.5f, 2.5f },
            { 3.5f, 4.5f }
        });

        var featureVector = FeatureSelectorHelper<float, Matrix<float>>.ExtractFeatureVector(
            matrix,
            featureIndex: 0,
            numSamples: 2,
            higherDimensionStrategy: FeatureExtractionStrategy.Mean,
            dimensionWeights: new Dictionary<int, float>());

        Assert.Equal(1.5f, featureVector[0]);
        Assert.Equal(3.5f, featureVector[1]);
    }

    #endregion

    #region ExtractFeatureVector Tests - Tensor 2D

    [Fact]
    public void ExtractFeatureVector_Tensor2D_ExtractsCorrectColumn()
    {
        var tensor = new Tensor<double>(new[] { 3, 4 });
        // Row 0: 1, 2, 3, 4
        // Row 1: 5, 6, 7, 8
        // Row 2: 9, 10, 11, 12
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                tensor[i, j] = i * 4 + j + 1;

        var featureVector = FeatureSelectorHelper<double, Tensor<double>>.ExtractFeatureVector(
            tensor,
            featureIndex: 1,
            numSamples: 3,
            higherDimensionStrategy: FeatureExtractionStrategy.Mean,
            dimensionWeights: new Dictionary<int, double>());

        Assert.Equal(3, featureVector.Length);
        Assert.Equal(2, featureVector[0]);   // Second column of row 0
        Assert.Equal(6, featureVector[1]);   // Second column of row 1
        Assert.Equal(10, featureVector[2]);  // Second column of row 2
    }

    #endregion

    #region ExtractFeatureVector Tests - Tensor 3D with Mean Strategy

    [Fact]
    public void ExtractFeatureVector_Tensor3D_MeanStrategy_CalculatesAverage()
    {
        var tensor = new Tensor<double>(new[] { 2, 2, 3 });
        // Sample 0, Feature 0: [1, 2, 3] -> mean = 2
        // Sample 0, Feature 1: [4, 5, 6] -> mean = 5
        // Sample 1, Feature 0: [7, 8, 9] -> mean = 8
        // Sample 1, Feature 1: [10, 11, 12] -> mean = 11
        tensor[0, 0, 0] = 1; tensor[0, 0, 1] = 2; tensor[0, 0, 2] = 3;
        tensor[0, 1, 0] = 4; tensor[0, 1, 1] = 5; tensor[0, 1, 2] = 6;
        tensor[1, 0, 0] = 7; tensor[1, 0, 1] = 8; tensor[1, 0, 2] = 9;
        tensor[1, 1, 0] = 10; tensor[1, 1, 1] = 11; tensor[1, 1, 2] = 12;

        var featureVector = FeatureSelectorHelper<double, Tensor<double>>.ExtractFeatureVector(
            tensor,
            featureIndex: 0,
            numSamples: 2,
            higherDimensionStrategy: FeatureExtractionStrategy.Mean,
            dimensionWeights: new Dictionary<int, double>());

        Assert.Equal(2, featureVector.Length);
        Assert.Equal(2, featureVector[0], 5);  // Mean of [1, 2, 3]
        Assert.Equal(8, featureVector[1], 5);  // Mean of [7, 8, 9]
    }

    #endregion

    #region ExtractFeatureVector Tests - Tensor 3D with Max Strategy

    [Fact]
    public void ExtractFeatureVector_Tensor3D_MaxStrategy_FindsMaximum()
    {
        var tensor = new Tensor<double>(new[] { 2, 2, 3 });
        // Sample 0, Feature 0: [1, 5, 3] -> max = 5
        // Sample 1, Feature 0: [2, 8, 4] -> max = 8
        tensor[0, 0, 0] = 1; tensor[0, 0, 1] = 5; tensor[0, 0, 2] = 3;
        tensor[0, 1, 0] = 10; tensor[0, 1, 1] = 20; tensor[0, 1, 2] = 30;
        tensor[1, 0, 0] = 2; tensor[1, 0, 1] = 8; tensor[1, 0, 2] = 4;
        tensor[1, 1, 0] = 40; tensor[1, 1, 1] = 50; tensor[1, 1, 2] = 60;

        var featureVector = FeatureSelectorHelper<double, Tensor<double>>.ExtractFeatureVector(
            tensor,
            featureIndex: 0,
            numSamples: 2,
            higherDimensionStrategy: FeatureExtractionStrategy.Max,
            dimensionWeights: new Dictionary<int, double>());

        Assert.Equal(2, featureVector.Length);
        Assert.Equal(5, featureVector[0]);  // Max of [1, 5, 3]
        Assert.Equal(8, featureVector[1]);  // Max of [2, 8, 4]
    }

    #endregion

    #region ExtractFeatureVector Tests - Tensor 3D with Flatten Strategy

    [Fact]
    public void ExtractFeatureVector_Tensor3D_FlattenStrategy_GetsFirstElement()
    {
        var tensor = new Tensor<double>(new[] { 2, 2, 3 });
        // Sample 0, Feature 0: [100, 2, 3] -> first = 100
        // Sample 1, Feature 0: [200, 5, 6] -> first = 200
        tensor[0, 0, 0] = 100; tensor[0, 0, 1] = 2; tensor[0, 0, 2] = 3;
        tensor[0, 1, 0] = 4; tensor[0, 1, 1] = 5; tensor[0, 1, 2] = 6;
        tensor[1, 0, 0] = 200; tensor[1, 0, 1] = 8; tensor[1, 0, 2] = 9;
        tensor[1, 1, 0] = 10; tensor[1, 1, 1] = 11; tensor[1, 1, 2] = 12;

        var featureVector = FeatureSelectorHelper<double, Tensor<double>>.ExtractFeatureVector(
            tensor,
            featureIndex: 0,
            numSamples: 2,
            higherDimensionStrategy: FeatureExtractionStrategy.Flatten,
            dimensionWeights: new Dictionary<int, double>());

        Assert.Equal(2, featureVector.Length);
        Assert.Equal(100, featureVector[0]);
        Assert.Equal(200, featureVector[1]);
    }

    #endregion

    #region CreateFeatureSubset Tests - Matrix

    [Fact]
    public void CreateFeatureSubset_Matrix_SingleFeature_ReturnsCorrectSubset()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3, 4 },
            { 5, 6, 7, 8 },
            { 9, 10, 11, 12 }
        });

        var subset = FeatureSelectorHelper<double, Matrix<double>>.CreateFeatureSubset(
            matrix,
            new List<int> { 1 });

        Assert.Equal(3, subset.Rows);
        Assert.Equal(1, subset.Columns);
        Assert.Equal(2, subset[0, 0]);
        Assert.Equal(6, subset[1, 0]);
        Assert.Equal(10, subset[2, 0]);
    }

    [Fact]
    public void CreateFeatureSubset_Matrix_MultipleFeatures_ReturnsCorrectSubset()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3, 4 },
            { 5, 6, 7, 8 }
        });

        var subset = FeatureSelectorHelper<double, Matrix<double>>.CreateFeatureSubset(
            matrix,
            new List<int> { 0, 2 });

        Assert.Equal(2, subset.Rows);
        Assert.Equal(2, subset.Columns);
        Assert.Equal(1, subset[0, 0]);
        Assert.Equal(3, subset[0, 1]);
        Assert.Equal(5, subset[1, 0]);
        Assert.Equal(7, subset[1, 1]);
    }

    [Fact]
    public void CreateFeatureSubset_Matrix_ReorderedFeatures_ReordersColumns()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 }
        });

        var subset = FeatureSelectorHelper<double, Matrix<double>>.CreateFeatureSubset(
            matrix,
            new List<int> { 2, 0, 1 });

        Assert.Equal(3, subset[0, 0]);  // Originally column 2
        Assert.Equal(1, subset[0, 1]);  // Originally column 0
        Assert.Equal(2, subset[0, 2]);  // Originally column 1
    }

    [Fact]
    public void CreateFeatureSubset_Float_Matrix_ReturnsCorrectSubset()
    {
        var matrix = new Matrix<float>(new float[,]
        {
            { 1.5f, 2.5f, 3.5f }
        });

        var subset = FeatureSelectorHelper<float, Matrix<float>>.CreateFeatureSubset(
            matrix,
            new List<int> { 1 });

        Assert.Equal(2.5f, subset[0, 0]);
    }

    #endregion

    #region CreateFeatureSubset Tests - Tensor

    [Fact]
    public void CreateFeatureSubset_Tensor2D_SingleFeature_ReturnsCorrectSubset()
    {
        var tensor = new Tensor<double>(new[] { 3, 4 });
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                tensor[i, j] = i * 10 + j;

        var subset = FeatureSelectorHelper<double, Tensor<double>>.CreateFeatureSubset(
            tensor,
            new List<int> { 2 });

        Assert.Equal(3, subset.Shape[0]);
        Assert.Equal(1, subset.Shape[1]);
        Assert.Equal(2, subset[0, 0]);   // Row 0, original column 2
        Assert.Equal(12, subset[1, 0]);  // Row 1, original column 2
        Assert.Equal(22, subset[2, 0]);  // Row 2, original column 2
    }

    [Fact]
    public void CreateFeatureSubset_Tensor2D_MultipleFeatures_ReturnsCorrectSubset()
    {
        var tensor = new Tensor<double>(new[] { 2, 4 });
        tensor[0, 0] = 1; tensor[0, 1] = 2; tensor[0, 2] = 3; tensor[0, 3] = 4;
        tensor[1, 0] = 5; tensor[1, 1] = 6; tensor[1, 2] = 7; tensor[1, 3] = 8;

        var subset = FeatureSelectorHelper<double, Tensor<double>>.CreateFeatureSubset(
            tensor,
            new List<int> { 0, 3 });

        Assert.Equal(2, subset.Shape[0]);
        Assert.Equal(2, subset.Shape[1]);
        Assert.Equal(1, subset[0, 0]);
        Assert.Equal(4, subset[0, 1]);
        Assert.Equal(5, subset[1, 0]);
        Assert.Equal(8, subset[1, 1]);
    }

    #endregion

    #region CreateFilteredData Tests - Matrix

    [Fact]
    public void CreateFilteredData_Matrix_SingleFeature_ReturnsCorrectFiltered()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 }
        });

        var filtered = FeatureSelectorHelper<double, Matrix<double>>.CreateFilteredData(
            matrix,
            new List<int> { 1 });

        Assert.Equal(2, filtered.Rows);
        Assert.Equal(1, filtered.Columns);
        Assert.Equal(2, filtered[0, 0]);
        Assert.Equal(5, filtered[1, 0]);
    }

    [Fact]
    public void CreateFilteredData_Matrix_MultipleFeatures_ReturnsCorrectFiltered()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 10, 20, 30, 40, 50 },
            { 11, 21, 31, 41, 51 },
            { 12, 22, 32, 42, 52 }
        });

        var filtered = FeatureSelectorHelper<double, Matrix<double>>.CreateFilteredData(
            matrix,
            new List<int> { 0, 2, 4 });

        Assert.Equal(3, filtered.Rows);
        Assert.Equal(3, filtered.Columns);

        // First row: 10, 30, 50
        Assert.Equal(10, filtered[0, 0]);
        Assert.Equal(30, filtered[0, 1]);
        Assert.Equal(50, filtered[0, 2]);
    }

    [Fact]
    public void CreateFilteredData_Matrix_AllFeatures_ReturnsSameData()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 }
        });

        var filtered = FeatureSelectorHelper<double, Matrix<double>>.CreateFilteredData(
            matrix,
            new List<int> { 0, 1, 2 });

        Assert.Equal(2, filtered.Rows);
        Assert.Equal(3, filtered.Columns);
    }

    #endregion

    #region CreateFilteredData Tests - Tensor

    [Fact]
    public void CreateFilteredData_Tensor2D_ReturnsCorrectFiltered()
    {
        var tensor = new Tensor<double>(new[] { 2, 4 });
        tensor[0, 0] = 1; tensor[0, 1] = 2; tensor[0, 2] = 3; tensor[0, 3] = 4;
        tensor[1, 0] = 5; tensor[1, 1] = 6; tensor[1, 2] = 7; tensor[1, 3] = 8;

        var filtered = FeatureSelectorHelper<double, Tensor<double>>.CreateFilteredData(
            tensor,
            new List<int> { 1, 2 });

        Assert.Equal(2, filtered.Shape[0]);
        Assert.Equal(2, filtered.Shape[1]);
        Assert.Equal(2, filtered[0, 0]);
        Assert.Equal(3, filtered[0, 1]);
        Assert.Equal(6, filtered[1, 0]);
        Assert.Equal(7, filtered[1, 1]);
    }

    [Fact]
    public void CreateFilteredData_Tensor3D_ReturnsCorrectFiltered()
    {
        var tensor = new Tensor<double>(new[] { 2, 3, 2 });
        // Initialize with recognizable values
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 2; k++)
                    tensor[i, j, k] = i * 100 + j * 10 + k;

        var filtered = FeatureSelectorHelper<double, Tensor<double>>.CreateFilteredData(
            tensor,
            new List<int> { 0, 2 });

        Assert.Equal(2, filtered.Shape[0]);
        Assert.Equal(2, filtered.Shape[1]);
        Assert.Equal(2, filtered.Shape[2]);
    }

    #endregion

    #region CopyFeature Tests - Tensor 2D

    [Fact]
    public void CopyFeature_Tensor2D_CopiesCorrectValue()
    {
        var source = new Tensor<double>(new[] { 3, 4 });
        source[1, 2] = 42.0;

        var destination = new Tensor<double>(new[] { 3, 2 });

        FeatureSelectorHelper<double, Tensor<double>>.CopyFeature(source, destination, 1, 2, 0);

        Assert.Equal(42.0, destination[1, 0]);
    }

    [Fact]
    public void CopyFeature_Tensor2D_MultipleCopies_CopiesAllValues()
    {
        var source = new Tensor<double>(new[] { 2, 3 });
        source[0, 0] = 10; source[0, 1] = 20; source[0, 2] = 30;
        source[1, 0] = 40; source[1, 1] = 50; source[1, 2] = 60;

        var destination = new Tensor<double>(new[] { 2, 2 });

        // Copy feature 0 and 2 from source
        FeatureSelectorHelper<double, Tensor<double>>.CopyFeature(source, destination, 0, 0, 0);
        FeatureSelectorHelper<double, Tensor<double>>.CopyFeature(source, destination, 0, 2, 1);
        FeatureSelectorHelper<double, Tensor<double>>.CopyFeature(source, destination, 1, 0, 0);
        FeatureSelectorHelper<double, Tensor<double>>.CopyFeature(source, destination, 1, 2, 1);

        Assert.Equal(10, destination[0, 0]);
        Assert.Equal(30, destination[0, 1]);
        Assert.Equal(40, destination[1, 0]);
        Assert.Equal(60, destination[1, 1]);
    }

    #endregion

    #region CopyFeature Tests - Tensor 3D

    [Fact]
    public void CopyFeature_Tensor3D_CopiesAllElements()
    {
        var source = new Tensor<double>(new[] { 2, 3, 4 });
        for (int k = 0; k < 4; k++)
            source[1, 2, k] = k * 10.0;  // Values: 0, 10, 20, 30

        var destination = new Tensor<double>(new[] { 2, 1, 4 });

        FeatureSelectorHelper<double, Tensor<double>>.CopyFeature(source, destination, 1, 2, 0);

        Assert.Equal(0.0, destination[1, 0, 0]);
        Assert.Equal(10.0, destination[1, 0, 1]);
        Assert.Equal(20.0, destination[1, 0, 2]);
        Assert.Equal(30.0, destination[1, 0, 3]);
    }

    #endregion

    #region CopyFeature Tests - Tensor 4D

    [Fact]
    public void CopyFeature_Tensor4D_CopiesAllElements()
    {
        var source = new Tensor<double>(new[] { 2, 3, 2, 2 });
        // Set recognizable values for sample 0, feature 1
        source[0, 1, 0, 0] = 1;
        source[0, 1, 0, 1] = 2;
        source[0, 1, 1, 0] = 3;
        source[0, 1, 1, 1] = 4;

        var destination = new Tensor<double>(new[] { 2, 1, 2, 2 });

        FeatureSelectorHelper<double, Tensor<double>>.CopyFeature(source, destination, 0, 1, 0);

        Assert.Equal(1, destination[0, 0, 0, 0]);
        Assert.Equal(2, destination[0, 0, 0, 1]);
        Assert.Equal(3, destination[0, 0, 1, 0]);
        Assert.Equal(4, destination[0, 0, 1, 1]);
    }

    #endregion

    #region Unsupported Type Tests

    [Fact]
    public void ExtractFeatureVector_UnsupportedType_ThrowsInvalidOperationException()
    {
        Assert.Throws<InvalidOperationException>(() =>
            FeatureSelectorHelper<double, string>.ExtractFeatureVector(
                "invalid",
                0,
                5,
                FeatureExtractionStrategy.Mean,
                new Dictionary<int, double>()));
    }

    [Fact]
    public void CreateFeatureSubset_UnsupportedType_ThrowsInvalidOperationException()
    {
        Assert.Throws<InvalidOperationException>(() =>
            FeatureSelectorHelper<double, string>.CreateFeatureSubset(
                "invalid",
                new List<int> { 0 }));
    }

    [Fact]
    public void CreateFilteredData_UnsupportedType_ThrowsInvalidOperationException()
    {
        Assert.Throws<InvalidOperationException>(() =>
            FeatureSelectorHelper<double, string>.CreateFilteredData(
                "invalid",
                new List<int> { 0 }));
    }

    #endregion

    #region Edge Cases Tests

    [Fact]
    public void CreateFeatureSubset_EmptyFeatureList_ThrowsArgumentException()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 }
        });

        // Empty feature list causes exception when creating matrix from empty columns
        Assert.Throws<ArgumentException>(() =>
            FeatureSelectorHelper<double, Matrix<double>>.CreateFeatureSubset(
                matrix,
                new List<int>()));
    }

    [Fact]
    public void CreateFilteredData_EmptyFeatureList_ThrowsArgumentException()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2 },
            { 3, 4 }
        });

        // Empty feature list causes exception when creating matrix from empty columns
        Assert.Throws<ArgumentException>(() =>
            FeatureSelectorHelper<double, Matrix<double>>.CreateFilteredData(
                matrix,
                new List<int>()));
    }

    #endregion

    #region Large Dataset Tests

    [Fact]
    public void CreateFilteredData_LargeMatrix_PerformsCorrectly()
    {
        int rows = 1000;
        int cols = 100;
        var matrix = new Matrix<double>(rows, cols);

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = i * cols + j;

        // Select every 10th feature
        var selectedFeatures = Enumerable.Range(0, cols).Where(i => i % 10 == 0).ToList();

        var filtered = FeatureSelectorHelper<double, Matrix<double>>.CreateFilteredData(
            matrix,
            selectedFeatures);

        Assert.Equal(rows, filtered.Rows);
        Assert.Equal(10, filtered.Columns);
        Assert.Equal(0, filtered[0, 0]);   // First row, feature 0
        Assert.Equal(10, filtered[0, 1]);  // First row, feature 10
    }

    [Fact]
    public void ExtractFeatureVector_LargeMatrix_PerformsCorrectly()
    {
        int rows = 1000;
        int cols = 50;
        var matrix = new Matrix<double>(rows, cols);

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = i * 100 + j;

        var featureVector = FeatureSelectorHelper<double, Matrix<double>>.ExtractFeatureVector(
            matrix,
            featureIndex: 25,
            numSamples: rows,
            higherDimensionStrategy: FeatureExtractionStrategy.Mean,
            dimensionWeights: new Dictionary<int, double>());

        Assert.Equal(rows, featureVector.Length);
        Assert.Equal(25, featureVector[0]);     // Row 0, column 25
        Assert.Equal(125, featureVector[1]);    // Row 1, column 25
    }

    #endregion

    #region WeightedSum Strategy Tests

    [Fact]
    public void ExtractFeatureVector_Tensor3D_WeightedSum_CalculatesCorrectSum()
    {
        var tensor = new Tensor<double>(new[] { 1, 1, 3 });
        tensor[0, 0, 0] = 10;
        tensor[0, 0, 1] = 20;
        tensor[0, 0, 2] = 30;

        var weights = new Dictionary<int, double>
        {
            { 0, 1.0 },
            { 1, 2.0 },
            { 2, 3.0 }
        };

        var featureVector = FeatureSelectorHelper<double, Tensor<double>>.ExtractFeatureVector(
            tensor,
            featureIndex: 0,
            numSamples: 1,
            higherDimensionStrategy: FeatureExtractionStrategy.WeightedSum,
            dimensionWeights: weights);

        Assert.Equal(1, featureVector.Length);
        // Expected: 10*1 + 20*2 + 30*3 = 10 + 40 + 90 = 140
        Assert.Equal(140, featureVector[0], 5);
    }

    [Fact]
    public void ExtractFeatureVector_WeightedSum_NoWeights_ThrowsException()
    {
        var tensor = new Tensor<double>(new[] { 1, 1, 3 });

        Assert.Throws<InvalidOperationException>(() =>
            FeatureSelectorHelper<double, Tensor<double>>.ExtractFeatureVector(
                tensor,
                featureIndex: 0,
                numSamples: 1,
                higherDimensionStrategy: FeatureExtractionStrategy.WeightedSum,
                dimensionWeights: new Dictionary<int, double>()));  // Empty weights
    }

    #endregion
}
