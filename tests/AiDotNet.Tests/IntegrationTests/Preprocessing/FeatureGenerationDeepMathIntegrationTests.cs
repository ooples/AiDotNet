using AiDotNet.Preprocessing.Discretizers;
using AiDotNet.Preprocessing.FeatureGeneration;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Deep math-correctness integration tests for feature generation
/// (PolynomialFeatures, Binarizer).
/// Each test hand-computes expected values and verifies code matches.
/// </summary>
public class FeatureGenerationDeepMathIntegrationTests
{
    private const double Tol = 1e-8;

    private static Matrix<double> MakeMatrix(double[,] data) => new(data);

    // ========================================================================
    // PolynomialFeatures - Degree 2, Single Feature
    // ========================================================================

    [Fact]
    public void Poly_Degree2_SingleFeature_WithBias()
    {
        // Input [x] with degree=2, includeBias=true
        // Output: [1, x, x^2]
        // For x=3: [1, 3, 9]
        var data = MakeMatrix(new double[,] { { 3 } });
        var poly = new PolynomialFeatures<double>(degree: 2, includeBias: true);
        poly.Fit(data);

        Assert.Equal(3, poly.NOutputFeatures); // 1, x, x^2

        var result = poly.Transform(data);
        Assert.Equal(1.0, result[0, 0], Tol); // bias
        Assert.Equal(3.0, result[0, 1], Tol); // x
        Assert.Equal(9.0, result[0, 2], Tol); // x^2
    }

    [Fact]
    public void Poly_Degree2_SingleFeature_NoBias()
    {
        // Input [x] with degree=2, includeBias=false
        // Output: [x, x^2]
        // For x=4: [4, 16]
        var data = MakeMatrix(new double[,] { { 4 } });
        var poly = new PolynomialFeatures<double>(degree: 2, includeBias: false);
        poly.Fit(data);

        Assert.Equal(2, poly.NOutputFeatures); // x, x^2

        var result = poly.Transform(data);
        Assert.Equal(4.0, result[0, 0], Tol);  // x
        Assert.Equal(16.0, result[0, 1], Tol); // x^2
    }

    [Fact]
    public void Poly_Degree2_TwoFeatures_WithBias()
    {
        // Input [a, b] with degree=2, includeBias=true
        // Should produce 6 features containing: 1, a, b, a^2, ab, b^2
        // For a=2, b=3: values should be {1, 2, 3, 4, 6, 9}
        var data = MakeMatrix(new double[,] { { 2, 3 } });
        var poly = new PolynomialFeatures<double>(degree: 2, includeBias: true);
        poly.Fit(data);

        Assert.Equal(6, poly.NOutputFeatures);

        var result = poly.Transform(data);
        Assert.Equal(1.0, result[0, 0], Tol); // bias is always first

        // All expected polynomial values should be present
        var outputValues = new List<double>();
        for (int j = 0; j < 6; j++)
            outputValues.Add(result[0, j]);

        Assert.Contains(1.0, outputValues); // 1 (bias)
        Assert.Contains(2.0, outputValues); // a
        Assert.Contains(3.0, outputValues); // b
        Assert.Contains(4.0, outputValues); // a^2
        Assert.Contains(6.0, outputValues); // ab
        Assert.Contains(9.0, outputValues); // b^2
    }

    [Fact]
    public void Poly_Degree2_TwoFeatures_NoBias()
    {
        // Input [a, b] with degree=2, includeBias=false
        // Should produce 5 features containing: a, b, a^2, ab, b^2
        // For a=2, b=3: values should be {2, 3, 4, 6, 9}
        var data = MakeMatrix(new double[,] { { 2, 3 } });
        var poly = new PolynomialFeatures<double>(degree: 2, includeBias: false);
        poly.Fit(data);

        Assert.Equal(5, poly.NOutputFeatures);

        var result = poly.Transform(data);
        var outputValues = new List<double>();
        for (int j = 0; j < 5; j++)
            outputValues.Add(result[0, j]);

        Assert.Contains(2.0, outputValues); // a
        Assert.Contains(3.0, outputValues); // b
        Assert.Contains(4.0, outputValues); // a^2
        Assert.Contains(6.0, outputValues); // ab
        Assert.Contains(9.0, outputValues); // b^2
    }

    [Fact]
    public void Poly_Degree1_IsIdentityWithBias()
    {
        // Degree 1 with bias: outputs 3 features
        // The recursive combination generator produces [0,1] before [1,0]
        // So order is: [1, b, a] (bias, then features in reverse-ish order)
        var data = MakeMatrix(new double[,] { { 5, 7 } });
        var poly = new PolynomialFeatures<double>(degree: 1, includeBias: true);
        poly.Fit(data);

        Assert.Equal(3, poly.NOutputFeatures);

        var result = poly.Transform(data);
        Assert.Equal(1.0, result[0, 0], Tol); // bias

        // Both feature values should appear in the output (order may vary)
        var outputValues = new HashSet<double> { result[0, 1], result[0, 2] };
        Assert.Contains(5.0, outputValues);
        Assert.Contains(7.0, outputValues);
    }

    [Fact]
    public void Poly_Degree3_SingleFeature()
    {
        // Input [x] with degree=3, includeBias=true
        // Output: [1, x, x^2, x^3]
        // For x=2: [1, 2, 4, 8]
        var data = MakeMatrix(new double[,] { { 2 } });
        var poly = new PolynomialFeatures<double>(degree: 3, includeBias: true);
        poly.Fit(data);

        Assert.Equal(4, poly.NOutputFeatures);

        var result = poly.Transform(data);
        Assert.Equal(1.0, result[0, 0], Tol);
        Assert.Equal(2.0, result[0, 1], Tol);
        Assert.Equal(4.0, result[0, 2], Tol);
        Assert.Equal(8.0, result[0, 3], Tol);
    }

    [Fact]
    public void Poly_InteractionOnly_Degree2()
    {
        // Input [a, b] with degree=2, interactionOnly=true, includeBias=true
        // Should produce: [1, a, b, ab] (no a^2 or b^2)
        var data = MakeMatrix(new double[,] { { 2, 3 } });
        var poly = new PolynomialFeatures<double>(
            degree: 2, interactionOnly: true, includeBias: true);
        poly.Fit(data);

        Assert.Equal(4, poly.NOutputFeatures);

        var result = poly.Transform(data);
        var outputValues = new List<double>();
        for (int j = 0; j < 4; j++)
            outputValues.Add(result[0, j]);

        Assert.Contains(1.0, outputValues); // bias
        Assert.Contains(2.0, outputValues); // a
        Assert.Contains(3.0, outputValues); // b
        Assert.Contains(6.0, outputValues); // ab

        // Verify no squared terms (4 or 9)
        Assert.DoesNotContain(4.0, outputValues);
        Assert.DoesNotContain(9.0, outputValues);
    }

    [Fact]
    public void Poly_MultipleRows_IndependentComputation()
    {
        // Multiple rows computed independently
        var data = MakeMatrix(new double[,] { { 2 }, { 3 }, { 4 } });
        var poly = new PolynomialFeatures<double>(degree: 2, includeBias: true);
        poly.Fit(data);

        var result = poly.Transform(data);

        // Row 0: x=2 => [1, 2, 4]
        Assert.Equal(1.0, result[0, 0], Tol);
        Assert.Equal(2.0, result[0, 1], Tol);
        Assert.Equal(4.0, result[0, 2], Tol);

        // Row 1: x=3 => [1, 3, 9]
        Assert.Equal(1.0, result[1, 0], Tol);
        Assert.Equal(3.0, result[1, 1], Tol);
        Assert.Equal(9.0, result[1, 2], Tol);

        // Row 2: x=4 => [1, 4, 16]
        Assert.Equal(1.0, result[2, 0], Tol);
        Assert.Equal(4.0, result[2, 1], Tol);
        Assert.Equal(16.0, result[2, 2], Tol);
    }

    [Fact]
    public void Poly_ZeroInput_CorrectOutput()
    {
        // x=0: [1, 0, 0]
        var data = MakeMatrix(new double[,] { { 0 } });
        var poly = new PolynomialFeatures<double>(degree: 2, includeBias: true);
        poly.Fit(data);
        var result = poly.Transform(data);

        Assert.Equal(1.0, result[0, 0], Tol); // bias
        Assert.Equal(0.0, result[0, 1], Tol); // 0
        Assert.Equal(0.0, result[0, 2], Tol); // 0^2
    }

    [Fact]
    public void Poly_NegativeInput_CorrectSquare()
    {
        // x=-3: [1, -3, 9]
        var data = MakeMatrix(new double[,] { { -3 } });
        var poly = new PolynomialFeatures<double>(degree: 2, includeBias: true);
        poly.Fit(data);
        var result = poly.Transform(data);

        Assert.Equal(1.0, result[0, 0], Tol);
        Assert.Equal(-3.0, result[0, 1], Tol);
        Assert.Equal(9.0, result[0, 2], Tol);
    }

    [Fact]
    public void Poly_NegativeInput_CubeIsNegative()
    {
        // x=-2: [1, -2, 4, -8]
        var data = MakeMatrix(new double[,] { { -2 } });
        var poly = new PolynomialFeatures<double>(degree: 3, includeBias: true);
        poly.Fit(data);
        var result = poly.Transform(data);

        Assert.Equal(1.0, result[0, 0], Tol);
        Assert.Equal(-2.0, result[0, 1], Tol);
        Assert.Equal(4.0, result[0, 2], Tol);
        Assert.Equal(-8.0, result[0, 3], Tol);
    }

    [Fact]
    public void Poly_InvalidDegree_Throws()
    {
        Assert.Throws<ArgumentException>(() => new PolynomialFeatures<double>(degree: 0));
    }

    [Fact]
    public void Poly_FeatureNamesOut_HasCorrectCount()
    {
        var data = MakeMatrix(new double[,] { { 1, 2 } });
        var poly = new PolynomialFeatures<double>(degree: 2, includeBias: true);
        poly.Fit(data);

        var names = poly.GetFeatureNamesOut();
        Assert.Equal(poly.NOutputFeatures, names.Length);
    }

    [Fact]
    public void Poly_NInputFeatures_IsCorrect()
    {
        var data = MakeMatrix(new double[,] { { 1, 2, 3 } });
        var poly = new PolynomialFeatures<double>(degree: 2);
        poly.Fit(data);

        Assert.Equal(3, poly.NInputFeatures);
    }

    [Fact]
    public void Poly_InverseTransform_Throws()
    {
        var data = MakeMatrix(new double[,] { { 1 } });
        var poly = new PolynomialFeatures<double>(degree: 2);
        poly.Fit(data);
        var result = poly.Transform(data);

        Assert.Throws<NotSupportedException>(() => poly.InverseTransform(result));
    }

    // ========================================================================
    // Binarizer - Threshold-Based Binary Conversion
    // ========================================================================

    [Fact]
    public void Binarizer_DefaultThreshold0_HandComputed()
    {
        // Default threshold=0: positive -> 1, zero/negative -> 0
        // [-2, -1, 0, 1, 2] -> [0, 0, 0, 1, 1]
        var data = MakeMatrix(new double[,] { { -2 }, { -1 }, { 0 }, { 1 }, { 2 } });
        var binarizer = new Binarizer<double>();
        binarizer.Fit(data);
        var result = binarizer.Transform(data);

        Assert.Equal(0.0, result[0, 0], Tol);
        Assert.Equal(0.0, result[1, 0], Tol);
        Assert.Equal(0.0, result[2, 0], Tol);
        Assert.Equal(1.0, result[3, 0], Tol);
        Assert.Equal(1.0, result[4, 0], Tol);
    }

    [Fact]
    public void Binarizer_CustomThreshold5()
    {
        // Threshold=5: [3, 5, 6, 8] -> [0, 0, 1, 1]
        var data = MakeMatrix(new double[,] { { 3 }, { 5 }, { 6 }, { 8 } });
        var binarizer = new Binarizer<double>(threshold: 5.0);
        binarizer.Fit(data);
        var result = binarizer.Transform(data);

        Assert.Equal(0.0, result[0, 0], Tol); // 3 <= 5
        Assert.Equal(0.0, result[1, 0], Tol); // 5 <= 5 (not strictly greater)
        Assert.Equal(1.0, result[2, 0], Tol); // 6 > 5
        Assert.Equal(1.0, result[3, 0], Tol); // 8 > 5
    }

    [Fact]
    public void Binarizer_AllAboveThreshold()
    {
        var data = MakeMatrix(new double[,] { { 10 }, { 20 }, { 30 } });
        var binarizer = new Binarizer<double>(threshold: 5.0);
        binarizer.Fit(data);
        var result = binarizer.Transform(data);

        Assert.Equal(1.0, result[0, 0], Tol);
        Assert.Equal(1.0, result[1, 0], Tol);
        Assert.Equal(1.0, result[2, 0], Tol);
    }

    [Fact]
    public void Binarizer_AllBelowThreshold()
    {
        var data = MakeMatrix(new double[,] { { 1 }, { 2 }, { 3 } });
        var binarizer = new Binarizer<double>(threshold: 10.0);
        binarizer.Fit(data);
        var result = binarizer.Transform(data);

        Assert.Equal(0.0, result[0, 0], Tol);
        Assert.Equal(0.0, result[1, 0], Tol);
        Assert.Equal(0.0, result[2, 0], Tol);
    }

    [Fact]
    public void Binarizer_NegativeThreshold()
    {
        // Threshold=-1: [-3, -1, 0, 2] -> [0, 0, 1, 1]
        var data = MakeMatrix(new double[,] { { -3 }, { -1 }, { 0 }, { 2 } });
        var binarizer = new Binarizer<double>(threshold: -1.0);
        binarizer.Fit(data);
        var result = binarizer.Transform(data);

        Assert.Equal(0.0, result[0, 0], Tol); // -3 <= -1
        Assert.Equal(0.0, result[1, 0], Tol); // -1 <= -1
        Assert.Equal(1.0, result[2, 0], Tol); // 0 > -1
        Assert.Equal(1.0, result[3, 0], Tol); // 2 > -1
    }

    [Fact]
    public void Binarizer_MultipleColumns()
    {
        // Both columns binarized with threshold=0
        var data = MakeMatrix(new double[,] {
            { -1, 2 },
            { 3, -4 }
        });
        var binarizer = new Binarizer<double>();
        binarizer.Fit(data);
        var result = binarizer.Transform(data);

        Assert.Equal(0.0, result[0, 0], Tol); // -1 <= 0
        Assert.Equal(1.0, result[0, 1], Tol); // 2 > 0
        Assert.Equal(1.0, result[1, 0], Tol); // 3 > 0
        Assert.Equal(0.0, result[1, 1], Tol); // -4 <= 0
    }

    [Fact]
    public void Binarizer_SpecificColumns_OnlyBinarizesSelected()
    {
        // Only binarize column 0
        var data = MakeMatrix(new double[,] {
            { -1, 99 },
            { 3, 88 }
        });
        var binarizer = new Binarizer<double>(threshold: 0.0, columnIndices: new[] { 0 });
        binarizer.Fit(data);
        var result = binarizer.Transform(data);

        Assert.Equal(0.0, result[0, 0], Tol);  // -1 binarized to 0
        Assert.Equal(99.0, result[0, 1], Tol);  // 99 unchanged
        Assert.Equal(1.0, result[1, 0], Tol);   // 3 binarized to 1
        Assert.Equal(88.0, result[1, 1], Tol);  // 88 unchanged
    }

    [Fact]
    public void Binarizer_InverseTransform_Throws()
    {
        var data = MakeMatrix(new double[,] { { 1 } });
        var binarizer = new Binarizer<double>();
        binarizer.Fit(data);
        var result = binarizer.Transform(data);

        Assert.Throws<NotSupportedException>(() => binarizer.InverseTransform(result));
    }

    [Fact]
    public void Binarizer_Idempotent_AlreadyBinary()
    {
        // Already binary data should stay the same
        var data = MakeMatrix(new double[,] { { 0 }, { 1 }, { 0 }, { 1 } });
        var binarizer = new Binarizer<double>(threshold: 0.5);
        binarizer.Fit(data);
        var result = binarizer.Transform(data);

        Assert.Equal(0.0, result[0, 0], Tol);
        Assert.Equal(1.0, result[1, 0], Tol);
        Assert.Equal(0.0, result[2, 0], Tol);
        Assert.Equal(1.0, result[3, 0], Tol);

        // Second transform should give same result
        var result2 = binarizer.Transform(result);
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(result[i, 0], result2[i, 0], Tol);
        }
    }

    [Fact]
    public void Binarizer_OutputOnlyContains0sAnd1s()
    {
        var data = MakeMatrix(new double[,] {
            { -100 }, { -0.001 }, { 0 }, { 0.001 }, { 100 }
        });
        var binarizer = new Binarizer<double>();
        binarizer.Fit(data);
        var result = binarizer.Transform(data);

        for (int i = 0; i < 5; i++)
        {
            Assert.True(result[i, 0] == 0.0 || result[i, 0] == 1.0,
                $"Expected 0 or 1, got {result[i, 0]}");
        }
    }

    // ========================================================================
    // Cross-Component: PolynomialFeatures Properties
    // ========================================================================

    [Fact]
    public void Poly_BiasColumnAlwaysOne()
    {
        // Bias column should always be 1 regardless of input
        var data = MakeMatrix(new double[,] { { -5 }, { 0 }, { 100 } });
        var poly = new PolynomialFeatures<double>(degree: 2, includeBias: true);
        poly.Fit(data);
        var result = poly.Transform(data);

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(1.0, result[i, 0], Tol);
        }
    }

    [Fact]
    public void Poly_Degree2_ThreeFeatures_OutputCount()
    {
        // [a, b, c] with degree=2, bias=true
        // Degree 0: [1]
        // Degree 1: [a, b, c]
        // Degree 2: [a^2, ab, ac, b^2, bc, c^2]
        // Total: 1 + 3 + 6 = 10
        var data = MakeMatrix(new double[,] { { 1, 2, 3 } });
        var poly = new PolynomialFeatures<double>(degree: 2, includeBias: true);
        poly.Fit(data);

        Assert.Equal(10, poly.NOutputFeatures);
    }

    [Fact]
    public void Poly_FitTransform_EquivalentToSeparate()
    {
        var data = MakeMatrix(new double[,] { { 2, 3 } });

        var poly1 = new PolynomialFeatures<double>(degree: 2, includeBias: true);
        poly1.Fit(data);
        var result1 = poly1.Transform(data);

        var poly2 = new PolynomialFeatures<double>(degree: 2, includeBias: true);
        var result2 = poly2.FitTransform(data);

        for (int j = 0; j < result1.Columns; j++)
        {
            Assert.Equal(result1[0, j], result2[0, j], Tol);
        }
    }
}
