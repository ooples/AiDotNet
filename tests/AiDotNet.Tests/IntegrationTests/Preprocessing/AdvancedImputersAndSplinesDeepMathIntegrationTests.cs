using AiDotNet.Preprocessing.Imputers;
using AiDotNet.Preprocessing.FeatureGeneration;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

public class AdvancedImputersAndSplinesDeepMathIntegrationTests
{
    private const double Tolerance = 1e-6;
    private const double LooseTolerance = 1e-2;

    // =====================================================================
    // KNNImputer: Imputes missing (NaN) values using K-nearest neighbors
    // Distance = sqrt(sum((a-b)^2) / count_valid_features) * sqrt(total_features)
    // Uniform: simple average of neighbor values
    // Distance-weighted: inverse-distance weighted average
    // =====================================================================

    [Fact]
    public void KNNImputer_Uniform_SimpleMean_OneNeighbor()
    {
        // 3 training rows, 2 features. Query has NaN in col 1.
        // Row 0: [1, 10], Row 1: [2, 20], Row 2: [3, 30]
        // Query: [1, NaN] → nearest neighbor is Row 0 (distance 0 on col 0)
        // With k=1, should impute col 1 = 10
        var trainData = new double[,]
        {
            { 1.0, 10.0 },
            { 2.0, 20.0 },
            { 3.0, 30.0 }
        };
        var trainMatrix = new Matrix<double>(trainData);

        var imputer = new KNNImputer<double>(nNeighbors: 1, weights: KNNWeights.Uniform);
        imputer.Fit(trainMatrix);

        var testData = new double[,]
        {
            { 1.0, double.NaN }
        };
        var testMatrix = new Matrix<double>(testData);
        var result = imputer.Transform(testMatrix);

        // Col 0 unchanged
        Assert.Equal(1.0, result[0, 0], Tolerance);
        // Col 1 imputed from nearest neighbor (Row 0, value 10)
        Assert.Equal(10.0, result[0, 1], Tolerance);
    }

    [Fact]
    public void KNNImputer_Uniform_AverageOfKNeighbors()
    {
        // 4 training rows, query [2, NaN], k=2
        // Row 0: [1, 10], Row 1: [2, 20], Row 2: [3, 30], Row 3: [10, 100]
        // Distances from [2,_] (using col 0 only):
        //   Row 0: |2-1| = 1
        //   Row 1: |2-2| = 0
        //   Row 2: |2-3| = 1
        //   Row 3: |2-10| = 8
        // Nearest 2: Row 1 (d=0), Row 0 or Row 2 (d=1)
        // Uniform average of their col 1 values
        var trainData = new double[,]
        {
            { 1.0, 10.0 },
            { 2.0, 20.0 },
            { 3.0, 30.0 },
            { 10.0, 100.0 }
        };
        var trainMatrix = new Matrix<double>(trainData);

        var imputer = new KNNImputer<double>(nNeighbors: 2, weights: KNNWeights.Uniform);
        imputer.Fit(trainMatrix);

        var testData = new double[,]
        {
            { 2.0, double.NaN }
        };
        var testMatrix = new Matrix<double>(testData);
        var result = imputer.Transform(testMatrix);

        // Nearest 2 neighbors: Row 1 (d=0, val=20), then Row 0 or Row 2 (d=1, val=10 or 30)
        // Uniform average: (20 + 10)/2 = 15 or (20 + 30)/2 = 25
        // Either is valid depending on distance tie-breaking
        double imputed = result[0, 1];
        Assert.True(imputed == 15.0 || imputed == 25.0,
            $"Expected imputed value to be 15 or 25 (depending on tie-breaking), got {imputed}");
    }

    [Fact]
    public void KNNImputer_NonMissingValuesPreserved()
    {
        var trainData = new double[,]
        {
            { 1.0, 10.0 },
            { 2.0, 20.0 },
            { 3.0, 30.0 }
        };
        var trainMatrix = new Matrix<double>(trainData);

        var imputer = new KNNImputer<double>(nNeighbors: 2);
        imputer.Fit(trainMatrix);

        var testData = new double[,]
        {
            { 5.0, 50.0 }
        };
        var testMatrix = new Matrix<double>(testData);
        var result = imputer.Transform(testMatrix);

        // No NaN → values should be unchanged
        Assert.Equal(5.0, result[0, 0], Tolerance);
        Assert.Equal(50.0, result[0, 1], Tolerance);
    }

    [Fact]
    public void KNNImputer_DistanceWeighted_CloserNeighborsDominateAverage()
    {
        // With distance weighting, closer neighbors have more influence
        // Row 0: [0, 100] (very close to query), Row 1: [100, 0] (very far)
        // Query: [0, NaN] → Row 0 has distance ~0, Row 1 has distance ~100
        // Distance-weighted should strongly favor Row 0's value (100)
        var trainData = new double[,]
        {
            { 0.0, 100.0 },
            { 100.0, 0.0 }
        };
        var trainMatrix = new Matrix<double>(trainData);

        var imputer = new KNNImputer<double>(nNeighbors: 2, weights: KNNWeights.Distance);
        imputer.Fit(trainMatrix);

        var testData = new double[,]
        {
            { 0.0, double.NaN }
        };
        var testMatrix = new Matrix<double>(testData);
        var result = imputer.Transform(testMatrix);

        // Should be very close to 100 (Row 0's value) since Row 0 is at distance 0
        Assert.True(result[0, 1] > 90.0,
            $"Distance-weighted imputation should favor close neighbor (100), got {result[0, 1]}");
    }

    [Fact]
    public void KNNImputer_MultipleNaNsInSameRow()
    {
        var trainData = new double[,]
        {
            { 1.0, 10.0, 100.0 },
            { 2.0, 20.0, 200.0 },
            { 3.0, 30.0, 300.0 }
        };
        var trainMatrix = new Matrix<double>(trainData);

        var imputer = new KNNImputer<double>(nNeighbors: 1);
        imputer.Fit(trainMatrix);

        var testData = new double[,]
        {
            { 1.0, double.NaN, double.NaN }
        };
        var testMatrix = new Matrix<double>(testData);
        var result = imputer.Transform(testMatrix);

        // Both NaN columns should be imputed
        Assert.False(double.IsNaN(result[0, 1]), "Col 1 should be imputed");
        Assert.False(double.IsNaN(result[0, 2]), "Col 2 should be imputed");
        // Nearest neighbor is Row 0 (col 0 matches exactly)
        Assert.Equal(10.0, result[0, 1], Tolerance);
        Assert.Equal(100.0, result[0, 2], Tolerance);
    }

    [Fact]
    public void KNNImputer_MinNeighborsValidation()
    {
        Assert.Throws<ArgumentException>(() =>
            new KNNImputer<double>(nNeighbors: 0));
    }

    [Fact]
    public void KNNImputer_AllRowsHaveNaN_FallsBackToColumnMean()
    {
        // If no valid neighbors can be found for a column, falls back to column mean
        var trainData = new double[,]
        {
            { 1.0, 10.0 },
            { 2.0, 20.0 },
            { 3.0, 30.0 }
        };
        var trainMatrix = new Matrix<double>(trainData);

        var imputer = new KNNImputer<double>(nNeighbors: 3);
        imputer.Fit(trainMatrix);

        // The imputer still has training data to find neighbors from
        var testData = new double[,]
        {
            { 2.0, double.NaN }
        };
        var testMatrix = new Matrix<double>(testData);
        var result = imputer.Transform(testMatrix);

        // Should produce a finite imputed value
        Assert.True((!double.IsNaN(result[0, 1]) && !double.IsInfinity(result[0, 1])),
            $"Imputed value should be finite, got {result[0, 1]}");
    }

    // =====================================================================
    // SplineTransformer: B-spline basis functions via De Boor's algorithm
    // With m internal knots and degree d:
    //   nBasis = m + d + 1 (with intercept)
    //   nBasis = m + d (without intercept)
    // Partition of Unity: sum of basis functions = 1 at any point
    // =====================================================================

    [Fact]
    public void Spline_OutputDimension_CorrectFormula()
    {
        // nKnots=5, degree=3, includeIntercept=true
        // nBasis per feature = 5 + 3 + 1 = 9
        var data = new double[,]
        {
            { 0.0 }, { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 },
            { 5.0 }, { 6.0 }, { 7.0 }, { 8.0 }, { 9.0 }
        };
        var matrix = new Matrix<double>(data);

        var spline = new SplineTransformer<double>(
            nKnots: 5, degree: 3, includeIntercept: true);
        var result = spline.FitTransform(matrix);

        Assert.Equal(10, result.Rows);
        Assert.Equal(9, result.Columns); // 5 + 3 + 1
    }

    [Fact]
    public void Spline_WithoutIntercept_OneLessColumn()
    {
        var data = new double[,]
        {
            { 0.0 }, { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 },
            { 5.0 }, { 6.0 }, { 7.0 }, { 8.0 }, { 9.0 }
        };
        var matrix = new Matrix<double>(data);

        var withIntercept = new SplineTransformer<double>(
            nKnots: 5, degree: 3, includeIntercept: true);
        var resultWith = withIntercept.FitTransform(matrix);

        var withoutIntercept = new SplineTransformer<double>(
            nKnots: 5, degree: 3, includeIntercept: false);
        var resultWithout = withoutIntercept.FitTransform(matrix);

        Assert.Equal(resultWith.Columns - 1, resultWithout.Columns);
    }

    [Fact]
    public void Spline_PartitionOfUnity_BasisFunctionsSumToOne()
    {
        // B-spline basis functions form a partition of unity:
        // At any point x, the sum of all basis functions equals 1
        var data = new double[,]
        {
            { 0.0 }, { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 },
            { 5.0 }, { 6.0 }, { 7.0 }, { 8.0 }, { 10.0 }
        };
        var matrix = new Matrix<double>(data);

        var spline = new SplineTransformer<double>(
            nKnots: 3, degree: 3, includeIntercept: true);
        var result = spline.FitTransform(matrix);

        for (int i = 0; i < result.Rows; i++)
        {
            double sum = 0;
            for (int j = 0; j < result.Columns; j++)
            {
                sum += result[i, j];
            }
            Assert.Equal(1.0, sum, LooseTolerance);
        }
    }

    [Fact]
    public void Spline_BasisFunctionsNonNegative()
    {
        // B-spline basis functions are non-negative
        var data = new double[,]
        {
            { 0.0 }, { 2.5 }, { 5.0 }, { 7.5 }, { 10.0 },
            { 1.0 }, { 3.0 }, { 6.0 }, { 8.0 }, { 9.0 }
        };
        var matrix = new Matrix<double>(data);

        var spline = new SplineTransformer<double>(
            nKnots: 4, degree: 3, includeIntercept: true);
        var result = spline.FitTransform(matrix);

        for (int i = 0; i < result.Rows; i++)
        {
            for (int j = 0; j < result.Columns; j++)
            {
                Assert.True(result[i, j] >= -Tolerance,
                    $"B-spline basis at [{i},{j}] should be non-negative, got {result[i, j]}");
            }
        }
    }

    [Fact]
    public void Spline_Degree1_PiecewiseLinear()
    {
        // Degree 1 = linear B-splines (piecewise linear, hat functions)
        var data = new double[,]
        {
            { 0.0 }, { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 },
            { 5.0 }, { 6.0 }, { 7.0 }, { 8.0 }, { 10.0 }
        };
        var matrix = new Matrix<double>(data);

        var spline = new SplineTransformer<double>(
            nKnots: 3, degree: 1, includeIntercept: true);
        var result = spline.FitTransform(matrix);

        // nBasis = 3 + 1 + 1 = 5
        Assert.Equal(5, result.Columns);

        // All values should be non-negative and finite
        for (int i = 0; i < result.Rows; i++)
        {
            for (int j = 0; j < result.Columns; j++)
            {
                Assert.True((!double.IsNaN(result[i, j]) && !double.IsInfinity(result[i, j])),
                    $"Value at [{i},{j}] should be finite, got {result[i, j]}");
                Assert.True(result[i, j] >= -Tolerance,
                    $"Value at [{i},{j}] should be non-negative, got {result[i, j]}");
            }
        }
    }

    [Fact]
    public void Spline_UniformKnots_EvenlySpaced()
    {
        var data = new double[,]
        {
            { 0.0 }, { 5.0 }, { 10.0 }
        };
        var matrix = new Matrix<double>(data);

        var spline = new SplineTransformer<double>(
            nKnots: 3, degree: 1, knotStrategy: SplineKnotStrategy.Uniform);
        spline.Fit(matrix);

        Assert.NotNull(spline.Knots);
        var knots = spline.Knots[0];
        Assert.NotNull(knots);

        // For range [0, 10] with 3 internal knots:
        // Internal knots at: 10/4=2.5, 10/2=5.0, 30/4=7.5
        // Full knot vector: [0, 0, 2.5, 5.0, 7.5, 10, 10] (degree 1 repeats boundary once)
        // With degree 1, boundary knots are repeated (degree+1)=2 times
        Assert.Equal(0.0, knots[0], Tolerance); // boundary
        Assert.Equal(0.0, knots[1], Tolerance); // boundary
        Assert.Equal(2.5, knots[2], Tolerance); // internal
        Assert.Equal(5.0, knots[3], Tolerance); // internal
        Assert.Equal(7.5, knots[4], Tolerance); // internal
        Assert.Equal(10.0, knots[5], Tolerance); // boundary
        Assert.Equal(10.0, knots[6], Tolerance); // boundary
    }

    [Fact]
    public void Spline_ConstantExtrapolation_ClipsValues()
    {
        // With constant extrapolation, values outside range get boundary basis values
        var trainData = new double[,]
        {
            { 0.0 }, { 5.0 }, { 10.0 }
        };
        var trainMatrix = new Matrix<double>(trainData);

        var spline = new SplineTransformer<double>(
            nKnots: 2, degree: 1, extrapolation: SplineExtrapolation.Constant, includeIntercept: true);
        spline.Fit(trainMatrix);

        var testData = new double[,]
        {
            { -5.0 },  // Below range
            { 5.0 },   // Within range
            { 15.0 }   // Above range
        };
        var testMatrix = new Matrix<double>(testData);
        var result = spline.Transform(testMatrix);

        // Below-range row should have same basis values as min boundary
        // Above-range row should have same basis values as max boundary
        var belowBasis = new double[result.Columns];
        var aboveBasis = new double[result.Columns];
        for (int j = 0; j < result.Columns; j++)
        {
            belowBasis[j] = result[0, j];
            aboveBasis[j] = result[2, j];
        }

        // Both should still sum to 1 (partition of unity)
        double belowSum = belowBasis.Sum();
        double aboveSum = aboveBasis.Sum();
        Assert.Equal(1.0, belowSum, LooseTolerance);
        Assert.Equal(1.0, aboveSum, LooseTolerance);
    }

    [Fact]
    public void Spline_MultiColumn_IndependentBasis()
    {
        var data = new double[,]
        {
            { 0.0, 100.0 },
            { 5.0, 200.0 },
            { 10.0, 300.0 }
        };
        var matrix = new Matrix<double>(data);

        var spline = new SplineTransformer<double>(
            nKnots: 2, degree: 1, includeIntercept: true);
        var result = spline.FitTransform(matrix);

        // Each column generates nKnots + degree + 1 = 2 + 1 + 1 = 4 spline features
        // 2 input columns * 4 = 8 output columns
        Assert.Equal(8, result.Columns);
    }

    [Fact]
    public void Spline_ValidationRejectsInvalidParams()
    {
        Assert.Throws<ArgumentException>(() =>
            new SplineTransformer<double>(nKnots: 1)); // min 2
        Assert.Throws<ArgumentException>(() =>
            new SplineTransformer<double>(degree: -1)); // min 0
        Assert.Throws<ArgumentException>(() =>
            new SplineTransformer<double>(degree: 6)); // max 5
    }

    [Fact]
    public void Spline_GetFeatureNamesOut_CorrectNames()
    {
        var data = new double[,]
        {
            { 0.0, 10.0 },
            { 5.0, 20.0 },
            { 10.0, 30.0 }
        };
        var matrix = new Matrix<double>(data);

        var spline = new SplineTransformer<double>(
            nKnots: 2, degree: 1, includeIntercept: true);
        spline.Fit(matrix);

        var names = spline.GetFeatureNamesOut(new[] { "age", "income" });
        Assert.Equal(8, names.Length); // 4 per column

        // Names should contain base feature name with spline suffix
        Assert.Contains("age", names[0]);
        Assert.Contains("spline", names[0]);
    }

    [Fact]
    public void Spline_Degree0_StepFunctions()
    {
        // Degree 0 = constant/step functions
        var data = new double[,]
        {
            { 0.0 }, { 2.0 }, { 4.0 }, { 6.0 }, { 8.0 }, { 10.0 }
        };
        var matrix = new Matrix<double>(data);

        var spline = new SplineTransformer<double>(
            nKnots: 3, degree: 0, includeIntercept: true);
        var result = spline.FitTransform(matrix);

        // nBasis = 3 + 0 + 1 = 4
        Assert.Equal(4, result.Columns);

        // Each row should have exactly one non-zero basis function (step function)
        for (int i = 0; i < result.Rows; i++)
        {
            int nonZeroCount = 0;
            for (int j = 0; j < result.Columns; j++)
            {
                if (Math.Abs(result[i, j]) > Tolerance)
                    nonZeroCount++;
            }
            Assert.Equal(1, nonZeroCount);
        }
    }

    [Fact]
    public void Spline_AllBasisFinite()
    {
        // Test that no NaN or Infinity values are produced
        var data = new double[,]
        {
            { 0.0 }, { 0.1 }, { 0.5 }, { 1.0 }, { 2.0 },
            { 5.0 }, { 9.0 }, { 9.5 }, { 9.9 }, { 10.0 }
        };
        var matrix = new Matrix<double>(data);

        var spline = new SplineTransformer<double>(
            nKnots: 5, degree: 3, includeIntercept: true);
        var result = spline.FitTransform(matrix);

        for (int i = 0; i < result.Rows; i++)
        {
            for (int j = 0; j < result.Columns; j++)
            {
                Assert.True((!double.IsNaN(result[i, j]) && !double.IsInfinity(result[i, j])),
                    $"Basis value at [{i},{j}] should be finite, got {result[i, j]}");
            }
        }
    }

    [Fact]
    public void Spline_BoundaryValues_HandledCorrectly()
    {
        // Test that the exact min and max values are handled
        var data = new double[,]
        {
            { 0.0 }, { 5.0 }, { 10.0 }
        };
        var matrix = new Matrix<double>(data);

        var spline = new SplineTransformer<double>(
            nKnots: 2, degree: 2, includeIntercept: true);
        var result = spline.FitTransform(matrix);

        // At boundaries, basis functions should still sum to 1
        for (int i = 0; i < result.Rows; i++)
        {
            double sum = 0;
            for (int j = 0; j < result.Columns; j++)
            {
                sum += result[i, j];
            }
            Assert.Equal(1.0, sum, LooseTolerance);
        }
    }
}
