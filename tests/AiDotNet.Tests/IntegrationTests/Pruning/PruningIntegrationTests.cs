using AiDotNet.Interfaces;
using AiDotNet.ModelCompression;
using AiDotNet.Pruning;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Pruning;

/// <summary>
/// Comprehensive integration tests for the Pruning module.
/// Tests MagnitudePruningStrategy, GradientPruningStrategy, StructuredPruningStrategy,
/// LotteryTicketPruningStrategy, PruningMask, and related types.
/// </summary>
public class PruningIntegrationTests
{
    #region MagnitudePruningStrategy Tests

    [Fact]
    public void MagnitudePruningStrategy_Properties_ReturnCorrectValues()
    {
        var strategy = new MagnitudePruningStrategy<double>();

        Assert.Equal("Magnitude", strategy.Name);
        Assert.False(strategy.RequiresGradients);
        Assert.False(strategy.IsStructured);
        Assert.Contains(SparsityPattern.Unstructured, strategy.SupportedPatterns);
        Assert.Contains(SparsityPattern.Structured2to4, strategy.SupportedPatterns);
        Assert.Contains(SparsityPattern.StructuredNtoM, strategy.SupportedPatterns);
    }

    [Fact]
    public void MagnitudePruningStrategy_ComputeImportanceScores_Vector_ReturnsAbsoluteValues()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Vector<double>(new[] { -0.5, 0.3, -0.8, 0.1, 0.9 });

        var scores = strategy.ComputeImportanceScores(weights);

        Assert.Equal(5, scores.Length);
        Assert.Equal(0.5, scores[0], 6);
        Assert.Equal(0.3, scores[1], 6);
        Assert.Equal(0.8, scores[2], 6);
        Assert.Equal(0.1, scores[3], 6);
        Assert.Equal(0.9, scores[4], 6);
    }

    [Fact]
    public void MagnitudePruningStrategy_ComputeImportanceScores_Matrix_ReturnsAbsoluteValues()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Matrix<double>(2, 3);
        weights[0, 0] = -0.5; weights[0, 1] = 0.3; weights[0, 2] = -0.8;
        weights[1, 0] = 0.1; weights[1, 1] = 0.9; weights[1, 2] = -0.2;

        var scores = strategy.ComputeImportanceScores(weights);

        Assert.Equal(2, scores.Rows);
        Assert.Equal(3, scores.Columns);
        Assert.Equal(0.5, scores[0, 0], 6);
        Assert.Equal(0.9, scores[1, 1], 6);
    }

    [Fact]
    public void MagnitudePruningStrategy_ComputeImportanceScores_Tensor_ReturnsAbsoluteValues()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Tensor<double>(new[] { 2, 2 });
        weights[0, 0] = -0.5; weights[0, 1] = 0.3;
        weights[1, 0] = 0.8; weights[1, 1] = -0.1;

        var scores = strategy.ComputeImportanceScores(weights);

        Assert.Equal(0.5, scores[0, 0], 6);
        Assert.Equal(0.3, scores[0, 1], 6);
        Assert.Equal(0.8, scores[1, 0], 6);
        Assert.Equal(0.1, scores[1, 1], 6);
    }

    [Fact]
    public void MagnitudePruningStrategy_CreateMask_Vector_PrunesSmallestWeights()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Vector<double>(new[] { 0.1, 0.5, 0.3, 0.9, 0.2 });
        var scores = strategy.ComputeImportanceScores(weights);

        // 40% sparsity = prune 2 out of 5 weights (the smallest: 0.1 and 0.2)
        var mask = strategy.CreateMask(scores, 0.4);

        Assert.Equal(0.4, mask.GetSparsity(), 6);
    }

    [Fact]
    public void MagnitudePruningStrategy_CreateMask_Matrix_PrunesSmallestWeights()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Matrix<double>(2, 3);
        weights[0, 0] = 0.1; weights[0, 1] = 0.5; weights[0, 2] = 0.3;
        weights[1, 0] = 0.9; weights[1, 1] = 0.2; weights[1, 2] = 0.8;

        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.CreateMask(scores, 0.5); // Prune 3 out of 6

        Assert.True(mask.GetSparsity() >= 0.4 && mask.GetSparsity() <= 0.6);
    }

    [Fact]
    public void MagnitudePruningStrategy_CreateMask_Tensor_PrunesSmallestWeights()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Tensor<double>(new[] { 2, 4 });
        var values = new[] { 0.1, 0.5, 0.3, 0.9, 0.2, 0.8, 0.4, 0.7 };
        for (int i = 0; i < values.Length; i++)
            weights[i] = values[i];

        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.CreateMask(scores, 0.5);

        Assert.True(mask.GetSparsity() >= 0.4 && mask.GetSparsity() <= 0.6);
    }

    [Fact]
    public void MagnitudePruningStrategy_CreateMask_ZeroSparsity_KeepsAllWeights()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Vector<double>(new[] { 0.1, 0.5, 0.3 });
        var scores = strategy.ComputeImportanceScores(weights);

        var mask = strategy.CreateMask(scores, 0.0);

        Assert.Equal(0.0, mask.GetSparsity(), 6);
    }

    [Fact]
    public void MagnitudePruningStrategy_CreateMask_FullSparsity_PrunesAllWeights()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Vector<double>(new[] { 0.1, 0.5, 0.3 });
        var scores = strategy.ComputeImportanceScores(weights);

        var mask = strategy.CreateMask(scores, 1.0);

        Assert.Equal(1.0, mask.GetSparsity(), 6);
    }

    [Fact]
    public void MagnitudePruningStrategy_CreateMask_InvalidSparsity_ThrowsArgumentException()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Vector<double>(new[] { 0.1, 0.5, 0.3 });
        var scores = strategy.ComputeImportanceScores(weights);

        Assert.Throws<ArgumentException>(() => strategy.CreateMask(scores, -0.1));
        Assert.Throws<ArgumentException>(() => strategy.CreateMask(scores, 1.1));
    }

    [Fact]
    public void MagnitudePruningStrategy_ApplyPruning_Vector_ZerosPrunedWeights()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Vector<double>(new[] { 0.1, 0.5, 0.3, 0.9, 0.2 });
        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.CreateMask(scores, 0.4);

        strategy.ApplyPruning(weights, mask);

        // Count zeros
        int zeroCount = 0;
        for (int i = 0; i < weights.Length; i++)
            if (weights[i] == 0.0) zeroCount++;

        Assert.Equal(2, zeroCount); // 40% of 5 = 2 zeros
    }

    [Fact]
    public void MagnitudePruningStrategy_ApplyPruning_Matrix_ZerosPrunedWeights()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Matrix<double>(2, 3);
        weights[0, 0] = 0.1; weights[0, 1] = 0.5; weights[0, 2] = 0.3;
        weights[1, 0] = 0.9; weights[1, 1] = 0.2; weights[1, 2] = 0.8;

        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.CreateMask(scores, 0.5);

        strategy.ApplyPruning(weights, mask);

        int zeroCount = 0;
        for (int i = 0; i < weights.Rows; i++)
            for (int j = 0; j < weights.Columns; j++)
                if (weights[i, j] == 0.0) zeroCount++;

        Assert.Equal(3, zeroCount); // 50% of 6 = 3 zeros
    }

    [Fact]
    public void MagnitudePruningStrategy_Create2to4Mask_CreatesCorrectPattern()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Tensor<double>(new[] { 2, 4 });
        var values = new[] { 0.1, 0.5, 0.3, 0.9, 0.2, 0.8, 0.4, 0.7 };
        for (int i = 0; i < values.Length; i++)
            weights[i] = values[i];

        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.Create2to4Mask(scores);

        // 2:4 means exactly 2 zeros per 4 elements = 50% sparsity
        Assert.Equal(0.5, mask.GetSparsity(), 6);
    }

    [Fact]
    public void MagnitudePruningStrategy_CreateNtoMMask_CreatesCorrectPattern()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Tensor<double>(new[] { 3, 6 });
        for (int i = 0; i < 18; i++)
            weights[i] = (i + 1) * 0.1;

        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.CreateNtoMMask(scores, 1, 3); // 1 zero per 3 elements

        // 1:3 = ~33% sparsity
        Assert.True(mask.GetSparsity() >= 0.3 && mask.GetSparsity() <= 0.35);
    }

    [Fact]
    public void MagnitudePruningStrategy_ToSparseFormat_COO_ReturnsValidResult()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Tensor<double>(new[] { 2, 3 });
        weights[0, 0] = 0.0; weights[0, 1] = 0.5; weights[0, 2] = 0.0;
        weights[1, 0] = 0.3; weights[1, 1] = 0.0; weights[1, 2] = 0.8;

        var result = strategy.ToSparseFormat(weights, SparseFormat.COO);

        Assert.Equal(SparseFormat.COO, result.Format);
        Assert.Equal(3, result.NonZeroCount); // 3 non-zero values
        Assert.NotNull(result.RowIndices);
        Assert.NotNull(result.ColumnIndices);
        Assert.Equal(new[] { 2, 3 }, result.OriginalShape);
    }

    [Fact]
    public void MagnitudePruningStrategy_ToSparseFormat_CSR_ReturnsValidResult()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Tensor<double>(new[] { 2, 3 });
        weights[0, 0] = 0.0; weights[0, 1] = 0.5; weights[0, 2] = 0.0;
        weights[1, 0] = 0.3; weights[1, 1] = 0.0; weights[1, 2] = 0.8;

        var result = strategy.ToSparseFormat(weights, SparseFormat.CSR);

        Assert.Equal(SparseFormat.CSR, result.Format);
        Assert.Equal(3, result.NonZeroCount);
        Assert.NotNull(result.RowPointers);
        Assert.NotNull(result.ColumnIndices);
    }

    [Fact]
    public void MagnitudePruningStrategy_ToSparseFormat_CSC_ReturnsValidResult()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Tensor<double>(new[] { 2, 3 });
        weights[0, 0] = 0.0; weights[0, 1] = 0.5; weights[0, 2] = 0.0;
        weights[1, 0] = 0.3; weights[1, 1] = 0.0; weights[1, 2] = 0.8;

        var result = strategy.ToSparseFormat(weights, SparseFormat.CSC);

        Assert.Equal(SparseFormat.CSC, result.Format);
        Assert.Equal(3, result.NonZeroCount);
        Assert.NotNull(result.ColumnPointers);
        Assert.NotNull(result.RowIndices);
    }

    [Fact]
    public void MagnitudePruningStrategy_ToSparseFormat_Structured2to4_ReturnsValidResult()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Tensor<double>(new[] { 2, 4 });
        weights[0] = 0.0; weights[1] = 0.5; weights[2] = 0.0; weights[3] = 0.9;
        weights[4] = 0.3; weights[5] = 0.0; weights[6] = 0.8; weights[7] = 0.0;

        var result = strategy.ToSparseFormat(weights, SparseFormat.Structured2to4);

        Assert.Equal(SparseFormat.Structured2to4, result.Format);
        Assert.Equal(2, result.SparsityN);
        Assert.Equal(4, result.SparsityM);
        Assert.NotNull(result.SparsityMask);
    }

    #endregion

    #region GradientPruningStrategy Tests

    [Fact]
    public void GradientPruningStrategy_Properties_ReturnCorrectValues()
    {
        var strategy = new GradientPruningStrategy<double>();

        Assert.Equal("Gradient", strategy.Name);
        Assert.True(strategy.RequiresGradients);
        Assert.False(strategy.IsStructured);
    }

    [Fact]
    public void GradientPruningStrategy_ComputeImportanceScores_Vector_ReturnsWeightGradientProduct()
    {
        var strategy = new GradientPruningStrategy<double>();
        var weights = new Vector<double>(new[] { 0.5, -0.3, 0.8 });
        var gradients = new Vector<double>(new[] { 0.1, 0.5, -0.2 });

        var scores = strategy.ComputeImportanceScores(weights, gradients);

        // |weight * gradient|
        Assert.Equal(0.05, scores[0], 6); // |0.5 * 0.1|
        Assert.Equal(0.15, scores[1], 6); // |-0.3 * 0.5|
        Assert.Equal(0.16, scores[2], 6); // |0.8 * -0.2|
    }

    [Fact]
    public void GradientPruningStrategy_ComputeImportanceScores_Matrix_ReturnsWeightGradientProduct()
    {
        var strategy = new GradientPruningStrategy<double>();
        var weights = new Matrix<double>(2, 2);
        weights[0, 0] = 0.5; weights[0, 1] = -0.3;
        weights[1, 0] = 0.8; weights[1, 1] = 0.1;

        var gradients = new Matrix<double>(2, 2);
        gradients[0, 0] = 0.2; gradients[0, 1] = 0.4;
        gradients[1, 0] = -0.1; gradients[1, 1] = 0.9;

        var scores = strategy.ComputeImportanceScores(weights, gradients);

        Assert.Equal(0.1, scores[0, 0], 6); // |0.5 * 0.2|
        Assert.Equal(0.12, scores[0, 1], 6); // |-0.3 * 0.4|
    }

    [Fact]
    public void GradientPruningStrategy_ComputeImportanceScores_NullGradients_ThrowsArgumentException()
    {
        var strategy = new GradientPruningStrategy<double>();
        var weights = new Vector<double>(new[] { 0.5, -0.3, 0.8 });

        Assert.Throws<ArgumentException>(() => strategy.ComputeImportanceScores(weights, null));
    }

    [Fact]
    public void GradientPruningStrategy_ComputeImportanceScores_MismatchedShapes_ThrowsArgumentException()
    {
        var strategy = new GradientPruningStrategy<double>();
        var weights = new Vector<double>(new[] { 0.5, -0.3, 0.8 });
        var gradients = new Vector<double>(new[] { 0.1, 0.5 }); // Different length

        Assert.Throws<ArgumentException>(() => strategy.ComputeImportanceScores(weights, gradients));
    }

    [Fact]
    public void GradientPruningStrategy_CreateMask_PrunesLowImportanceWeights()
    {
        var strategy = new GradientPruningStrategy<double>();
        var weights = new Vector<double>(new[] { 0.5, 0.1, 0.8, 0.3 });
        var gradients = new Vector<double>(new[] { 0.1, 0.9, 0.2, 0.5 });

        var scores = strategy.ComputeImportanceScores(weights, gradients);
        var mask = strategy.CreateMask(scores, 0.5);

        Assert.True(mask.GetSparsity() >= 0.4 && mask.GetSparsity() <= 0.6);
    }

    [Fact]
    public void GradientPruningStrategy_ApplyPruning_ZerosPrunedWeights()
    {
        var strategy = new GradientPruningStrategy<double>();
        var weights = new Vector<double>(new[] { 0.5, 0.1, 0.8, 0.3 });
        var gradients = new Vector<double>(new[] { 0.1, 0.9, 0.2, 0.5 });

        var scores = strategy.ComputeImportanceScores(weights, gradients);
        var mask = strategy.CreateMask(scores, 0.5);
        strategy.ApplyPruning(weights, mask);

        int zeroCount = 0;
        for (int i = 0; i < weights.Length; i++)
            if (weights[i] == 0.0) zeroCount++;

        Assert.Equal(2, zeroCount);
    }

    [Fact]
    public void GradientPruningStrategy_Create2to4Mask_CreatesCorrectPattern()
    {
        var strategy = new GradientPruningStrategy<double>();
        var weights = new Tensor<double>(new[] { 2, 4 });
        var gradients = new Tensor<double>(new[] { 2, 4 });
        for (int i = 0; i < 8; i++)
        {
            weights[i] = (i + 1) * 0.1;
            gradients[i] = (8 - i) * 0.1;
        }

        var scores = strategy.ComputeImportanceScores(weights, gradients);
        var mask = strategy.Create2to4Mask(scores);

        Assert.Equal(0.5, mask.GetSparsity(), 6);
    }

    [Fact]
    public void GradientPruningStrategy_ToSparseFormat_COO_ReturnsValidResult()
    {
        var strategy = new GradientPruningStrategy<double>();
        var weights = new Tensor<double>(new[] { 2, 3 });
        weights[0, 0] = 0.0; weights[0, 1] = 0.5; weights[0, 2] = 0.0;
        weights[1, 0] = 0.3; weights[1, 1] = 0.0; weights[1, 2] = 0.8;

        var result = strategy.ToSparseFormat(weights, SparseFormat.COO);

        Assert.Equal(SparseFormat.COO, result.Format);
        Assert.Equal(3, result.NonZeroCount);
    }

    #endregion

    #region StructuredPruningStrategy Tests

    [Fact]
    public void StructuredPruningStrategy_NeuronType_Properties_ReturnCorrectValues()
    {
        var strategy = new StructuredPruningStrategy<double>(
            StructuredPruningStrategy<double>.StructurePruningType.Neuron);

        Assert.Equal("Structured", strategy.Name);
        Assert.False(strategy.RequiresGradients);
        Assert.True(strategy.IsStructured);
        Assert.Contains(SparsityPattern.RowStructured, strategy.SupportedPatterns);
        Assert.Contains(SparsityPattern.ColumnStructured, strategy.SupportedPatterns);
    }

    [Fact]
    public void StructuredPruningStrategy_FilterType_Properties_ReturnCorrectValues()
    {
        var strategy = new StructuredPruningStrategy<double>(
            StructuredPruningStrategy<double>.StructurePruningType.Filter);

        Assert.Contains(SparsityPattern.FilterStructured, strategy.SupportedPatterns);
    }

    [Fact]
    public void StructuredPruningStrategy_ChannelType_Properties_ReturnCorrectValues()
    {
        var strategy = new StructuredPruningStrategy<double>(
            StructuredPruningStrategy<double>.StructurePruningType.Channel);

        Assert.Contains(SparsityPattern.ChannelStructured, strategy.SupportedPatterns);
    }

    [Fact]
    public void StructuredPruningStrategy_Neuron_ComputeImportanceScores_ReturnsColumnNorms()
    {
        var strategy = new StructuredPruningStrategy<double>(
            StructuredPruningStrategy<double>.StructurePruningType.Neuron);

        var weights = new Matrix<double>(3, 2);
        // Column 0: [0.3, 0.4, 0.0] -> L2 norm = sqrt(0.09 + 0.16) = 0.5
        // Column 1: [0.0, 0.6, 0.8] -> L2 norm = sqrt(0.36 + 0.64) = 1.0
        weights[0, 0] = 0.3; weights[0, 1] = 0.0;
        weights[1, 0] = 0.4; weights[1, 1] = 0.6;
        weights[2, 0] = 0.0; weights[2, 1] = 0.8;

        var scores = strategy.ComputeImportanceScores(weights);

        // All elements in same column should have same score
        Assert.Equal(scores[0, 0], scores[1, 0], 6);
        Assert.Equal(scores[0, 0], scores[2, 0], 6);
        Assert.Equal(scores[0, 1], scores[1, 1], 6);
        Assert.Equal(scores[0, 1], scores[2, 1], 6);
        // Column 1 should have higher score than column 0
        Assert.True(scores[0, 1] > scores[0, 0]);
    }

    [Fact]
    public void StructuredPruningStrategy_Filter_ComputeImportanceScores_ReturnsRowNorms()
    {
        var strategy = new StructuredPruningStrategy<double>(
            StructuredPruningStrategy<double>.StructurePruningType.Filter);

        var weights = new Matrix<double>(2, 3);
        weights[0, 0] = 0.3; weights[0, 1] = 0.4; weights[0, 2] = 0.0;
        weights[1, 0] = 0.0; weights[1, 1] = 0.6; weights[1, 2] = 0.8;

        var scores = strategy.ComputeImportanceScores(weights);

        // All elements in same row should have same score
        Assert.Equal(scores[0, 0], scores[0, 1], 6);
        Assert.Equal(scores[0, 0], scores[0, 2], 6);
        Assert.Equal(scores[1, 0], scores[1, 1], 6);
        Assert.Equal(scores[1, 0], scores[1, 2], 6);
    }

    [Fact]
    public void StructuredPruningStrategy_Neuron_CreateMask_PrunesEntireColumns()
    {
        var strategy = new StructuredPruningStrategy<double>(
            StructuredPruningStrategy<double>.StructurePruningType.Neuron);

        var weights = new Matrix<double>(3, 4);
        // Create columns with different norms
        for (int i = 0; i < 3; i++)
        {
            weights[i, 0] = 0.1; // Low importance column
            weights[i, 1] = 0.5; // Medium
            weights[i, 2] = 0.2; // Low
            weights[i, 3] = 0.9; // High
        }

        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.CreateMask(scores, 0.5); // Prune 2 columns

        strategy.ApplyPruning(weights, mask);

        // Check that entire columns are pruned (all zeros in a column)
        int zeroColumns = 0;
        for (int j = 0; j < 4; j++)
        {
            bool allZero = true;
            for (int i = 0; i < 3; i++)
            {
                if (weights[i, j] != 0.0) allZero = false;
            }
            if (allZero) zeroColumns++;
        }

        Assert.Equal(2, zeroColumns); // 50% of 4 columns = 2 pruned
    }

    [Fact]
    public void StructuredPruningStrategy_Filter_CreateMask_PrunesEntireRows()
    {
        var strategy = new StructuredPruningStrategy<double>(
            StructuredPruningStrategy<double>.StructurePruningType.Filter);

        var weights = new Matrix<double>(4, 3);
        // Create rows with different norms
        for (int j = 0; j < 3; j++)
        {
            weights[0, j] = 0.1;
            weights[1, j] = 0.9;
            weights[2, j] = 0.2;
            weights[3, j] = 0.8;
        }

        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.CreateMask(scores, 0.5);

        strategy.ApplyPruning(weights, mask);

        // Check that entire rows are pruned
        int zeroRows = 0;
        for (int i = 0; i < 4; i++)
        {
            bool allZero = true;
            for (int j = 0; j < 3; j++)
            {
                if (weights[i, j] != 0.0) allZero = false;
            }
            if (allZero) zeroRows++;
        }

        Assert.Equal(2, zeroRows);
    }

    [Fact]
    public void StructuredPruningStrategy_Filter_4DTensor_ComputeImportanceScores()
    {
        var strategy = new StructuredPruningStrategy<double>(
            StructuredPruningStrategy<double>.StructurePruningType.Filter);

        // 4D tensor: [filters=2, channels=2, height=2, width=2]
        var weights = new Tensor<double>(new[] { 2, 2, 2, 2 });
        for (int i = 0; i < 16; i++)
            weights[i] = (i + 1) * 0.1;

        var scores = strategy.ComputeImportanceScores(weights);

        // All elements in same filter should have same score
        Assert.Equal(scores.Shape, weights.Shape);
    }

    [Fact]
    public void StructuredPruningStrategy_Channel_4DTensor_ComputeImportanceScores()
    {
        var strategy = new StructuredPruningStrategy<double>(
            StructuredPruningStrategy<double>.StructurePruningType.Channel);

        var weights = new Tensor<double>(new[] { 2, 2, 2, 2 });
        for (int i = 0; i < 16; i++)
            weights[i] = (i + 1) * 0.1;

        var scores = strategy.ComputeImportanceScores(weights);

        Assert.Equal(scores.Shape, weights.Shape);
    }

    [Fact]
    public void StructuredPruningStrategy_CreateMask_InvalidSparsity_ThrowsArgumentException()
    {
        var strategy = new StructuredPruningStrategy<double>();
        var weights = new Matrix<double>(2, 3);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                weights[i, j] = 0.5;

        var scores = strategy.ComputeImportanceScores(weights);

        Assert.Throws<ArgumentException>(() => strategy.CreateMask(scores, -0.1));
        Assert.Throws<ArgumentException>(() => strategy.CreateMask(scores, 1.1));
        Assert.Throws<ArgumentException>(() => strategy.CreateMask(scores, double.NaN));
    }

    [Fact]
    public void StructuredPruningStrategy_ToSparseFormat_COO_ReturnsValidResult()
    {
        var strategy = new StructuredPruningStrategy<double>();
        var weights = new Tensor<double>(new[] { 2, 3 });
        weights[0, 0] = 0.0; weights[0, 1] = 0.5; weights[0, 2] = 0.0;
        weights[1, 0] = 0.3; weights[1, 1] = 0.0; weights[1, 2] = 0.8;

        var result = strategy.ToSparseFormat(weights, SparseFormat.COO);

        Assert.Equal(SparseFormat.COO, result.Format);
        Assert.Equal(3, result.NonZeroCount);
    }

    #endregion

    #region LotteryTicketPruningStrategy Tests

    [Fact]
    public void LotteryTicketPruningStrategy_Properties_ReturnCorrectValues()
    {
        var strategy = new LotteryTicketPruningStrategy<double>();

        Assert.Equal("LotteryTicket", strategy.Name);
        Assert.False(strategy.RequiresGradients);
        Assert.False(strategy.IsStructured);
    }

    [Fact]
    public void LotteryTicketPruningStrategy_Constructor_InvalidRounds_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new LotteryTicketPruningStrategy<double>(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LotteryTicketPruningStrategy<double>(-1));
    }

    [Fact]
    public void LotteryTicketPruningStrategy_StoreAndGetInitialWeights_WorksCorrectly()
    {
        var strategy = new LotteryTicketPruningStrategy<double>();
        var weights = new Matrix<double>(2, 3);
        weights[0, 0] = 0.1; weights[0, 1] = 0.2; weights[0, 2] = 0.3;
        weights[1, 0] = 0.4; weights[1, 1] = 0.5; weights[1, 2] = 0.6;

        strategy.StoreInitialWeights("layer1", weights);
        var retrieved = strategy.GetInitialWeights("layer1");

        // Should be a clone (not same reference)
        Assert.NotSame(weights, retrieved);

        // Values should match
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(weights[i, j], retrieved[i, j]);
    }

    [Fact]
    public void LotteryTicketPruningStrategy_GetInitialWeights_NoStoredWeights_ThrowsInvalidOperationException()
    {
        var strategy = new LotteryTicketPruningStrategy<double>();

        Assert.Throws<InvalidOperationException>(() => strategy.GetInitialWeights("nonexistent"));
    }

    [Fact]
    public void LotteryTicketPruningStrategy_ComputeImportanceScores_UseMagnitude()
    {
        var strategy = new LotteryTicketPruningStrategy<double>();
        var weights = new Vector<double>(new[] { -0.5, 0.3, -0.8, 0.1 });

        var scores = strategy.ComputeImportanceScores(weights);

        Assert.Equal(0.5, scores[0], 6);
        Assert.Equal(0.3, scores[1], 6);
        Assert.Equal(0.8, scores[2], 6);
        Assert.Equal(0.1, scores[3], 6);
    }

    [Fact]
    public void LotteryTicketPruningStrategy_CreateMask_IterativePruning_WorksCorrectly()
    {
        var strategy = new LotteryTicketPruningStrategy<double>(iterativeRounds: 3);
        var weights = new Vector<double>(new[] { 0.1, 0.5, 0.3, 0.9, 0.2, 0.8, 0.4, 0.7 });

        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.CreateMask(scores, 0.5);

        Assert.True(mask.GetSparsity() >= 0.4 && mask.GetSparsity() <= 0.6,
            $"Sparsity {mask.GetSparsity()} should be between 0.4 and 0.6");
    }

    [Fact]
    public void LotteryTicketPruningStrategy_CreateMask_Matrix_IterativePruning()
    {
        var strategy = new LotteryTicketPruningStrategy<double>(iterativeRounds: 3);
        var weights = new Matrix<double>(3, 4);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                weights[i, j] = (i * 4 + j + 1) * 0.05;

        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.CreateMask(scores, 0.5);

        Assert.True(mask.GetSparsity() >= 0.4 && mask.GetSparsity() <= 0.6);
    }

    [Fact]
    public void LotteryTicketPruningStrategy_ResetToInitialWeights_ResetsCorrectly()
    {
        var strategy = new LotteryTicketPruningStrategy<double>();

        // Store initial weights
        var initialWeights = new Matrix<double>(2, 3);
        initialWeights[0, 0] = 0.1; initialWeights[0, 1] = 0.2; initialWeights[0, 2] = 0.3;
        initialWeights[1, 0] = 0.4; initialWeights[1, 1] = 0.5; initialWeights[1, 2] = 0.6;
        strategy.StoreInitialWeights("layer1", initialWeights);

        // Simulate training (weights changed)
        var trainedWeights = new Matrix<double>(2, 3);
        trainedWeights[0, 0] = 0.8; trainedWeights[0, 1] = 0.1; trainedWeights[0, 2] = 0.9;
        trainedWeights[1, 0] = 0.2; trainedWeights[1, 1] = 0.7; trainedWeights[1, 2] = 0.05;

        // Create mask and reset
        var scores = strategy.ComputeImportanceScores(trainedWeights);
        var mask = strategy.CreateMask(scores, 0.5);

        strategy.ResetToInitialWeights("layer1", trainedWeights, mask);

        // Check that non-pruned weights are reset to initial values
        // and pruned weights are zero
        var maskData = mask.GetMaskData();
        int idx = 0;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                if (maskData[idx] != 0.0)
                {
                    // Weight should be reset to initial
                    Assert.Equal(initialWeights[i, j], trainedWeights[i, j], 6);
                }
                else
                {
                    // Weight should be zero (pruned)
                    Assert.Equal(0.0, trainedWeights[i, j], 6);
                }
                idx++;
            }
        }
    }

    [Fact]
    public void LotteryTicketPruningStrategy_ResetToInitialWeights_MismatchedDimensions_ThrowsArgumentException()
    {
        var strategy = new LotteryTicketPruningStrategy<double>();

        var initialWeights = new Matrix<double>(2, 3);
        strategy.StoreInitialWeights("layer1", initialWeights);

        var wrongSizeWeights = new Matrix<double>(3, 2); // Different size
        var mask = new PruningMask<double>(3, 2);

        Assert.Throws<ArgumentException>(() =>
            strategy.ResetToInitialWeights("layer1", wrongSizeWeights, mask));
    }

    [Fact]
    public void LotteryTicketPruningStrategy_Create2to4Mask_CreatesCorrectPattern()
    {
        var strategy = new LotteryTicketPruningStrategy<double>();
        var weights = new Tensor<double>(new[] { 2, 4 });
        for (int i = 0; i < 8; i++)
            weights[i] = (i + 1) * 0.1;

        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.Create2to4Mask(scores);

        Assert.Equal(0.5, mask.GetSparsity(), 6);
    }

    [Fact]
    public void LotteryTicketPruningStrategy_CreateNtoMMask_InvalidParameters_ThrowsException()
    {
        var strategy = new LotteryTicketPruningStrategy<double>();
        var weights = new Tensor<double>(new[] { 2, 4 });
        for (int i = 0; i < 8; i++)
            weights[i] = 0.5;
        var scores = strategy.ComputeImportanceScores(weights);

        Assert.Throws<ArgumentOutOfRangeException>(() => strategy.CreateNtoMMask(scores, 2, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => strategy.CreateNtoMMask(scores, -1, 4));
        Assert.Throws<ArgumentException>(() => strategy.CreateNtoMMask(scores, 5, 4)); // n > m
    }

    [Fact]
    public void LotteryTicketPruningStrategy_ToSparseFormat_AllFormats()
    {
        var strategy = new LotteryTicketPruningStrategy<double>();
        var weights = new Tensor<double>(new[] { 2, 3 });
        weights[0, 0] = 0.0; weights[0, 1] = 0.5; weights[0, 2] = 0.0;
        weights[1, 0] = 0.3; weights[1, 1] = 0.0; weights[1, 2] = 0.8;

        var coResult = strategy.ToSparseFormat(weights, SparseFormat.COO);
        var csrResult = strategy.ToSparseFormat(weights, SparseFormat.CSR);
        var cscResult = strategy.ToSparseFormat(weights, SparseFormat.CSC);

        Assert.Equal(SparseFormat.COO, coResult.Format);
        Assert.Equal(SparseFormat.CSR, csrResult.Format);
        Assert.Equal(SparseFormat.CSC, cscResult.Format);
        Assert.Equal(3, coResult.NonZeroCount);
        Assert.Equal(3, csrResult.NonZeroCount);
        Assert.Equal(3, cscResult.NonZeroCount);
    }

    #endregion

    #region PruningMask Tests

    [Fact]
    public void PruningMask_Constructor_RowsCols_InitializesAllOnes()
    {
        var mask = new PruningMask<double>(3, 4);

        Assert.Equal(new[] { 3, 4 }, mask.Shape);
        Assert.Equal(0.0, mask.GetSparsity(), 6); // No pruning = 0 sparsity
    }

    [Fact]
    public void PruningMask_Constructor_BoolArray1D_InitializesCorrectly()
    {
        var keepIndices = new[] { true, false, true, false };
        var mask = new PruningMask<double>(keepIndices);

        Assert.Equal(0.5, mask.GetSparsity(), 6);
    }

    [Fact]
    public void PruningMask_Constructor_BoolArray2D_InitializesCorrectly()
    {
        var keepIndices = new bool[2, 3];
        keepIndices[0, 0] = true; keepIndices[0, 1] = false; keepIndices[0, 2] = true;
        keepIndices[1, 0] = false; keepIndices[1, 1] = true; keepIndices[1, 2] = false;

        var mask = new PruningMask<double>(keepIndices);

        Assert.Equal(0.5, mask.GetSparsity(), 6);
    }

    [Fact]
    public void PruningMask_Constructor_Matrix_InitializesCorrectly()
    {
        var matrix = new Matrix<double>(2, 2);
        matrix[0, 0] = 1.0; matrix[0, 1] = 0.0;
        matrix[1, 0] = 0.0; matrix[1, 1] = 1.0;

        var mask = new PruningMask<double>(matrix);

        Assert.Equal(0.5, mask.GetSparsity(), 6);
    }

    [Fact]
    public void PruningMask_Apply_Vector_ReturnsCorrectResult()
    {
        var keepIndices = new[] { true, false, true, false };
        var mask = new PruningMask<double>(keepIndices);
        var weights = new Vector<double>(new[] { 0.5, 0.3, 0.8, 0.2 });

        var result = mask.Apply(weights);

        Assert.Equal(0.5, result[0], 6);
        Assert.Equal(0.0, result[1], 6);
        Assert.Equal(0.8, result[2], 6);
        Assert.Equal(0.0, result[3], 6);
    }

    [Fact]
    public void PruningMask_Apply_Vector_MismatchedLength_ThrowsArgumentException()
    {
        var keepIndices = new[] { true, false, true };
        var mask = new PruningMask<double>(keepIndices);
        var weights = new Vector<double>(new[] { 0.5, 0.3, 0.8, 0.2 }); // Wrong length

        Assert.Throws<ArgumentException>(() => mask.Apply(weights));
    }

    [Fact]
    public void PruningMask_Apply_Matrix_ReturnsCorrectResult()
    {
        var keepIndices = new bool[2, 2];
        keepIndices[0, 0] = true; keepIndices[0, 1] = false;
        keepIndices[1, 0] = false; keepIndices[1, 1] = true;

        var mask = new PruningMask<double>(keepIndices);
        var weights = new Matrix<double>(2, 2);
        weights[0, 0] = 0.5; weights[0, 1] = 0.3;
        weights[1, 0] = 0.8; weights[1, 1] = 0.2;

        var result = mask.Apply(weights);

        Assert.Equal(0.5, result[0, 0], 6);
        Assert.Equal(0.0, result[0, 1], 6);
        Assert.Equal(0.0, result[1, 0], 6);
        Assert.Equal(0.2, result[1, 1], 6);
    }

    [Fact]
    public void PruningMask_Apply_Matrix_MismatchedShape_ThrowsArgumentException()
    {
        var mask = new PruningMask<double>(2, 3);
        var weights = new Matrix<double>(3, 2); // Wrong shape

        Assert.Throws<ArgumentException>(() => mask.Apply(weights));
    }

    [Fact]
    public void PruningMask_Apply_Tensor2D_ReturnsCorrectResult()
    {
        var keepIndices = new bool[2, 2];
        keepIndices[0, 0] = true; keepIndices[0, 1] = false;
        keepIndices[1, 0] = false; keepIndices[1, 1] = true;

        var mask = new PruningMask<double>(keepIndices);
        var weights = new Tensor<double>(new[] { 2, 2 });
        weights[0, 0] = 0.5; weights[0, 1] = 0.3;
        weights[1, 0] = 0.8; weights[1, 1] = 0.2;

        var result = mask.Apply(weights);

        Assert.Equal(0.5, result[0, 0], 6);
        Assert.Equal(0.0, result[0, 1], 6);
        Assert.Equal(0.0, result[1, 0], 6);
        Assert.Equal(0.2, result[1, 1], 6);
    }

    [Fact]
    public void PruningMask_UpdateMask_BoolArray1D_UpdatesCorrectly()
    {
        var mask = new PruningMask<double>(1, 4);
        Assert.Equal(0.0, mask.GetSparsity(), 6);

        mask.UpdateMask(new[] { true, false, true, false });

        Assert.Equal(0.5, mask.GetSparsity(), 6);
    }

    [Fact]
    public void PruningMask_UpdateMask_BoolArray2D_UpdatesCorrectly()
    {
        var mask = new PruningMask<double>(2, 2);
        var newMask = new bool[2, 2];
        newMask[0, 0] = true; newMask[0, 1] = false;
        newMask[1, 0] = false; newMask[1, 1] = false;

        mask.UpdateMask(newMask);

        Assert.Equal(0.75, mask.GetSparsity(), 6);
    }

    [Fact]
    public void PruningMask_UpdateMask_MismatchedShape_ThrowsArgumentException()
    {
        var mask = new PruningMask<double>(2, 3);

        Assert.Throws<ArgumentException>(() =>
            mask.UpdateMask(new bool[3, 2])); // Wrong shape
    }

    [Fact]
    public void PruningMask_CombineWith_LogicalAND()
    {
        var keepIndices1 = new[] { true, true, false, false };
        var keepIndices2 = new[] { true, false, true, false };

        var mask1 = new PruningMask<double>(keepIndices1);
        var mask2 = new PruningMask<double>(keepIndices2);

        var combined = mask1.CombineWith(mask2);

        // AND: [true, false, false, false]
        Assert.Equal(0.75, combined.GetSparsity(), 6);
    }

    [Fact]
    public void PruningMask_CombineWith_MismatchedShape_ThrowsArgumentException()
    {
        var mask1 = new PruningMask<double>(2, 3);
        var mask2 = new PruningMask<double>(3, 2);

        Assert.Throws<ArgumentException>(() => mask1.CombineWith(mask2));
    }

    [Fact]
    public void PruningMask_GetKeptIndices_ReturnsCorrectIndices()
    {
        var keepIndices = new[] { true, false, true, false, true };
        var mask = new PruningMask<double>(keepIndices);

        var keptIndices = mask.GetKeptIndices();

        Assert.Equal(new[] { 0, 2, 4 }, keptIndices);
    }

    [Fact]
    public void PruningMask_GetPrunedIndices_ReturnsCorrectIndices()
    {
        var keepIndices = new[] { true, false, true, false, true };
        var mask = new PruningMask<double>(keepIndices);

        var prunedIndices = mask.GetPrunedIndices();

        Assert.Equal(new[] { 1, 3 }, prunedIndices);
    }

    [Fact]
    public void PruningMask_GetMaskData_ReturnsFlattenedData()
    {
        var keepIndices = new bool[2, 2];
        keepIndices[0, 0] = true; keepIndices[0, 1] = false;
        keepIndices[1, 0] = false; keepIndices[1, 1] = true;

        var mask = new PruningMask<double>(keepIndices);
        var data = mask.GetMaskData();

        Assert.Equal(4, data.Length);
        Assert.Equal(1.0, data[0], 6);
        Assert.Equal(0.0, data[1], 6);
        Assert.Equal(0.0, data[2], 6);
        Assert.Equal(1.0, data[3], 6);
    }

    [Fact]
    public void PruningMask_Pattern_ReturnsUnstructured()
    {
        var mask = new PruningMask<double>(2, 3);

        Assert.Equal(SparsityPattern.Unstructured, mask.Pattern);
    }

    #endregion

    #region PruningConfig Tests

    [Fact]
    public void PruningConfig_DefaultValues_AreCorrect()
    {
        var config = new PruningConfig();

        Assert.Equal(0.5, config.TargetSparsity);
        Assert.Equal(SparsityPattern.Unstructured, config.Pattern);
        Assert.Equal(2, config.SparsityN);
        Assert.Equal(4, config.SparsityM);
        Assert.False(config.GradualPruning);
        Assert.Equal(10, config.PruningIterations);
        Assert.Equal(0.0, config.InitialSparsity);
        Assert.False(config.LayerWiseSparsity);
        Assert.Null(config.LayerSparsityTargets);
        Assert.True(config.FineTuneAfterPruning);
        Assert.Equal(10, config.FineTuningEpochs);
        Assert.Equal(SparseFormat.CSR, config.OutputFormat);
    }

    [Fact]
    public void PruningConfig_SetProperties_WorksCorrectly()
    {
        var config = new PruningConfig
        {
            TargetSparsity = 0.9,
            Pattern = SparsityPattern.Structured2to4,
            SparsityN = 2,
            SparsityM = 4,
            GradualPruning = true,
            PruningIterations = 5,
            InitialSparsity = 0.1,
            LayerWiseSparsity = true,
            LayerSparsityTargets = new Dictionary<string, double> { ["layer1"] = 0.8 },
            FineTuneAfterPruning = false,
            FineTuningEpochs = 20,
            OutputFormat = SparseFormat.COO
        };

        Assert.Equal(0.9, config.TargetSparsity);
        Assert.Equal(SparsityPattern.Structured2to4, config.Pattern);
        Assert.True(config.GradualPruning);
        Assert.Equal(0.8, config.LayerSparsityTargets!["layer1"]);
        Assert.False(config.FineTuneAfterPruning);
    }

    #endregion

    #region SparseCompressionResult Tests

    [Fact]
    public void SparseCompressionResult_Sparsity_CalculatesCorrectly()
    {
        var result = new SparseCompressionResult<double>
        {
            Format = SparseFormat.COO,
            Values = new[] { 0.5, 0.3, 0.8 },
            RowIndices = new[] { 0, 1, 1 },
            ColumnIndices = new[] { 1, 0, 2 },
            OriginalShape = new[] { 2, 3 } // 6 total elements, 3 non-zero
        };

        Assert.Equal(0.5, result.Sparsity, 6); // 3/6 = 50% non-zero, so 50% sparse
    }

    [Fact]
    public void SparseCompressionResult_NonZeroCount_ReturnsValuesLength()
    {
        var result = new SparseCompressionResult<double>
        {
            Format = SparseFormat.COO,
            Values = new[] { 0.5, 0.3, 0.8, 0.2 },
            OriginalShape = new[] { 3, 3 }
        };

        Assert.Equal(4, result.NonZeroCount);
    }

    [Fact]
    public void SparseCompressionResult_GetCompressedSizeBytes_CalculatesCorrectly()
    {
        var result = new SparseCompressionResult<double>
        {
            Format = SparseFormat.COO,
            Values = new[] { 0.5, 0.3, 0.8 },
            RowIndices = new[] { 0, 1, 1 },
            ColumnIndices = new[] { 1, 0, 2 },
            OriginalShape = new[] { 2, 3 }
        };

        long size = result.GetCompressedSizeBytes(sizeof(double));

        // Values: 3 * 8 = 24
        // RowIndices: 3 * 4 = 12
        // ColumnIndices: 3 * 4 = 12
        // Metadata: (2 + 4) * 4 = 24
        Assert.True(size > 0);
    }

    #endregion

    #region Integration Scenarios

    [Fact]
    public void Integration_FullPruningWorkflow_MagnitudeStrategy()
    {
        var strategy = new MagnitudePruningStrategy<double>();

        // Create weights
        var weights = new Matrix<double>(4, 4);
        var random = new Random(42);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                weights[i, j] = random.NextDouble();

        // Compute importance scores
        var scores = strategy.ComputeImportanceScores(weights);

        // Create mask with 50% sparsity
        var mask = strategy.CreateMask(scores, 0.5);

        // Apply pruning
        strategy.ApplyPruning(weights, mask);

        // Convert to sparse format
        var tensor = new Tensor<double>(new[] { 4, 4 });
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                tensor[i, j] = weights[i, j];

        var sparse = strategy.ToSparseFormat(tensor, SparseFormat.CSR);

        // Verify results
        Assert.True(mask.GetSparsity() >= 0.4 && mask.GetSparsity() <= 0.6);
        Assert.Equal(SparseFormat.CSR, sparse.Format);
        Assert.True(sparse.Sparsity >= 0.4 && sparse.Sparsity <= 0.6);
    }

    [Fact]
    public void Integration_FullPruningWorkflow_GradientStrategy()
    {
        var strategy = new GradientPruningStrategy<double>();

        // Create weights and gradients
        var weights = new Matrix<double>(4, 4);
        var gradients = new Matrix<double>(4, 4);
        var random = new Random(42);
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                weights[i, j] = random.NextDouble();
                gradients[i, j] = random.NextDouble() - 0.5;
            }
        }

        // Compute importance scores (requires gradients)
        var scores = strategy.ComputeImportanceScores(weights, gradients);

        // Create mask and apply
        var mask = strategy.CreateMask(scores, 0.5);
        strategy.ApplyPruning(weights, mask);

        Assert.True(mask.GetSparsity() >= 0.4 && mask.GetSparsity() <= 0.6);
    }

    [Fact]
    public void Integration_FullPruningWorkflow_StructuredStrategy()
    {
        var strategy = new StructuredPruningStrategy<double>(
            StructuredPruningStrategy<double>.StructurePruningType.Neuron);

        // Create weights
        var weights = new Matrix<double>(4, 8);
        var random = new Random(42);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 8; j++)
                weights[i, j] = random.NextDouble();

        // Compute scores and create mask
        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.CreateMask(scores, 0.5);

        // Apply and verify entire columns are pruned
        strategy.ApplyPruning(weights, mask);

        // Check structured pruning - count zero columns
        int zeroColumns = 0;
        for (int j = 0; j < 8; j++)
        {
            bool allZero = true;
            for (int i = 0; i < 4; i++)
                if (weights[i, j] != 0.0) allZero = false;
            if (allZero) zeroColumns++;
        }

        Assert.Equal(4, zeroColumns); // 50% of 8 columns
    }

    [Fact]
    public void Integration_LotteryTicketHypothesis_Workflow()
    {
        var strategy = new LotteryTicketPruningStrategy<double>(iterativeRounds: 3);

        // Step 1: Store initial weights
        var initialWeights = new Matrix<double>(4, 4);
        var random = new Random(42);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                initialWeights[i, j] = random.NextDouble() * 0.1;

        strategy.StoreInitialWeights("layer1", initialWeights);

        // Step 2: Simulate training (weights become larger)
        var trainedWeights = new Matrix<double>(4, 4);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                trainedWeights[i, j] = initialWeights[i, j] + random.NextDouble();

        // Step 3: Find winning ticket
        var scores = strategy.ComputeImportanceScores(trainedWeights);
        var mask = strategy.CreateMask(scores, 0.5);

        // Step 4: Reset to initial weights
        strategy.ResetToInitialWeights("layer1", trainedWeights, mask);

        // Verify: Non-pruned weights should be at initial values
        var storedInitial = strategy.GetInitialWeights("layer1");
        var maskData = mask.GetMaskData();
        int idx = 0;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                if (maskData[idx] != 0.0)
                {
                    Assert.Equal(storedInitial[i, j], trainedWeights[i, j], 6);
                }
                else
                {
                    Assert.Equal(0.0, trainedWeights[i, j], 6);
                }
                idx++;
            }
        }
    }

    [Fact]
    public void Integration_CompareStrategies_SameSparsity()
    {
        var magnitudeStrategy = new MagnitudePruningStrategy<double>();
        var structuredStrategy = new StructuredPruningStrategy<double>();
        var lotteryStrategy = new LotteryTicketPruningStrategy<double>();

        // Same weights for all
        var weights = new Matrix<double>(4, 4);
        var random = new Random(42);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                weights[i, j] = random.NextDouble();

        // Compute masks with same target sparsity
        var magScores = magnitudeStrategy.ComputeImportanceScores(weights);
        var structScores = structuredStrategy.ComputeImportanceScores(weights);
        var lotScores = lotteryStrategy.ComputeImportanceScores(weights);

        var magMask = magnitudeStrategy.CreateMask(magScores, 0.5);
        var structMask = structuredStrategy.CreateMask(structScores, 0.5);
        var lotMask = lotteryStrategy.CreateMask(lotScores, 0.5);

        // All should achieve similar sparsity
        Assert.True(magMask.GetSparsity() >= 0.4 && magMask.GetSparsity() <= 0.6);
        Assert.True(structMask.GetSparsity() >= 0.4 && structMask.GetSparsity() <= 0.6);
        Assert.True(lotMask.GetSparsity() >= 0.4 && lotMask.GetSparsity() <= 0.6);
    }

    [Fact]
    public void Integration_SparseFormatRoundTrip()
    {
        var strategy = new MagnitudePruningStrategy<double>();

        // Create and prune weights
        var weights = new Tensor<double>(new[] { 3, 4 });
        for (int i = 0; i < 12; i++)
            weights[i] = (i + 1) * 0.1;

        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.CreateMask(scores, 0.5);
        strategy.ApplyPruning(weights, mask);

        // Convert to all sparse formats
        var cooResult = strategy.ToSparseFormat(weights, SparseFormat.COO);
        var csrResult = strategy.ToSparseFormat(weights, SparseFormat.CSR);
        var cscResult = strategy.ToSparseFormat(weights, SparseFormat.CSC);

        // All should have same non-zero count and sparsity
        Assert.Equal(cooResult.NonZeroCount, csrResult.NonZeroCount);
        Assert.Equal(csrResult.NonZeroCount, cscResult.NonZeroCount);
        Assert.Equal(cooResult.Sparsity, csrResult.Sparsity, 6);
        Assert.Equal(csrResult.Sparsity, cscResult.Sparsity, 6);
    }

    [Fact]
    public void Integration_2to4Sparsity_AllStrategies()
    {
        var strategies = new IPruningStrategy<double>[]
        {
            new MagnitudePruningStrategy<double>(),
            new GradientPruningStrategy<double>(),
            new StructuredPruningStrategy<double>(),
            new LotteryTicketPruningStrategy<double>()
        };

        var weights = new Tensor<double>(new[] { 2, 8 });
        for (int i = 0; i < 16; i++)
            weights[i] = (i + 1) * 0.1;

        var gradients = new Tensor<double>(new[] { 2, 8 });
        for (int i = 0; i < 16; i++)
            gradients[i] = (16 - i) * 0.05;

        foreach (var strategy in strategies)
        {
            Tensor<double>? grads = strategy.RequiresGradients ? gradients : null;
            var scores = strategy.ComputeImportanceScores(weights, grads);
            var mask = strategy.Create2to4Mask(scores);

            // All should achieve exactly 50% sparsity for 2:4 pattern
            Assert.Equal(0.5, mask.GetSparsity(), 6);
        }
    }

    [Fact]
    public void Integration_MaskCombination_MultipleRounds()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Vector<double>(new[] { 0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4 });

        // Round 1: 25% sparsity
        var scores1 = strategy.ComputeImportanceScores(weights);
        var mask1 = strategy.CreateMask(scores1, 0.25);

        // Apply first mask
        strategy.ApplyPruning(weights, mask1);

        // Round 2: Another 25% of remaining
        var scores2 = strategy.ComputeImportanceScores(weights);
        var mask2 = strategy.CreateMask(scores2, 0.333); // ~25% of remaining 75%

        // Combine masks
        var combinedMask = mask1.CombineWith(mask2);

        Assert.True(combinedMask.GetSparsity() >= 0.4 && combinedMask.GetSparsity() <= 0.6);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void EdgeCase_SingleElement_Pruning()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Vector<double>(new[] { 0.5 });

        var scores = strategy.ComputeImportanceScores(weights);
        var mask0 = strategy.CreateMask(scores, 0.0);
        var mask1 = strategy.CreateMask(scores, 1.0);

        Assert.Equal(0.0, mask0.GetSparsity(), 6);
        Assert.Equal(1.0, mask1.GetSparsity(), 6);
    }

    [Fact]
    public void EdgeCase_AllSameWeights_Pruning()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Vector<double>(new[] { 0.5, 0.5, 0.5, 0.5 });

        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.CreateMask(scores, 0.5);

        // Should still achieve target sparsity even when all weights are same
        Assert.Equal(0.5, mask.GetSparsity(), 6);
    }

    [Fact]
    public void EdgeCase_AllZeroWeights_Pruning()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0 });

        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.CreateMask(scores, 0.5);

        // Should work even with zero weights
        Assert.Equal(0.5, mask.GetSparsity(), 6);
    }

    [Fact]
    public void EdgeCase_VeryHighSparsity()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Vector<double>(new double[100]);
        var random = new Random(42);
        for (int i = 0; i < 100; i++)
            weights[i] = random.NextDouble();

        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.CreateMask(scores, 0.99);

        Assert.True(mask.GetSparsity() >= 0.98 && mask.GetSparsity() <= 1.0);
    }

    [Fact]
    public void EdgeCase_LargeMatrix_Performance()
    {
        var strategy = new MagnitudePruningStrategy<double>();
        var weights = new Matrix<double>(100, 100);
        var random = new Random(42);
        for (int i = 0; i < 100; i++)
            for (int j = 0; j < 100; j++)
                weights[i, j] = random.NextDouble();

        var scores = strategy.ComputeImportanceScores(weights);
        var mask = strategy.CreateMask(scores, 0.5);
        strategy.ApplyPruning(weights, mask);

        Assert.True(mask.GetSparsity() >= 0.4 && mask.GetSparsity() <= 0.6);
    }

    #endregion
}
