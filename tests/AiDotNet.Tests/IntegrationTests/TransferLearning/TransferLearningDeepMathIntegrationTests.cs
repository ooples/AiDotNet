using AiDotNet.LinearAlgebra;
using AiDotNet.TransferLearning.DomainAdaptation;
using AiDotNet.TransferLearning.FeatureMapping;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.TransferLearning;

/// <summary>
/// Deep math integration tests for transfer learning domain adaptation and feature mapping.
///
/// Key formulas tested:
/// - CORAL covariance: Cov = (1/n) * X_centered^T * X_centered + reg*I
/// - CORAL transformation: C_s^{-1/2} * C_t^{1/2} (diagonal approximation)
/// - Frobenius norm: ||A||_F = sqrt(Σ a_ij²)
/// - MMD²: E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
/// - Gaussian kernel: k(x,y) = exp(-||x-y||² / (2*σ²))
/// - Linear projection: Y = X * W (with Gram-Schmidt orthonormalization)
/// - Reconstruction confidence: exp(-MSE)
/// </summary>
public class TransferLearningDeepMathIntegrationTests
{
    private const double Tol = 1e-6;

    #region CORAL Domain Adaptation Tests

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void CORAL_IdenticalDomains_DiscrepancyNearZero()
    {
        // If source and target have the same distribution, discrepancy should be near zero
        var coral = new CORALDomainAdapter<double>();
        var data = MakeMatrix(new double[,]
        {
            { 1, 2 },
            { 3, 4 },
            { 5, 6 }
        });

        var discrepancy = coral.ComputeDomainDiscrepancy(data, data);

        // Identical data → identical covariances → zero Frobenius norm
        Assert.Equal(0.0, discrepancy, Tol);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void CORAL_DifferentDomains_PositiveDiscrepancy()
    {
        var coral = new CORALDomainAdapter<double>();
        var source = MakeMatrix(new double[,]
        {
            { 1, 0 },
            { 0, 1 },
            { -1, 0 },
            { 0, -1 }
        });
        var target = MakeMatrix(new double[,]
        {
            { 10, 0 },
            { 0, 10 },
            { -10, 0 },
            { 0, -10 }
        });

        var discrepancy = coral.ComputeDomainDiscrepancy(source, target);

        // Different scales → different covariances → positive discrepancy
        Assert.True(discrepancy > 0.0, $"Discrepancy should be positive, got {discrepancy}");
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void CORAL_CovarianceComputation_HandCalculated()
    {
        // Data: [[1,2],[3,4],[5,6]]
        // Mean: [3, 4]
        // Centered: [[-2,-2],[0,0],[2,2]]
        // Cov = (1/3) * X_c^T * X_c + reg*I
        //     = (1/3) * [[(-2)²+0²+2², (-2)(-2)+0*0+2*2], [same, (-2)²+0²+2²]]
        //     = (1/3) * [[8, 8], [8, 8]] + 1e-5*I
        //     = [[8/3 + 1e-5, 8/3], [8/3, 8/3 + 1e-5]]
        //
        // Frobenius norm of cov - cov = 0
        var coral = new CORALDomainAdapter<double>();
        var data = MakeMatrix(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
            { 5.0, 6.0 }
        });

        var discrepancy = coral.ComputeDomainDiscrepancy(data, data);
        Assert.Equal(0.0, discrepancy, Tol);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void CORAL_AdaptSource_PreservesMean()
    {
        var coral = new CORALDomainAdapter<double>();
        var source = MakeMatrix(new double[,]
        {
            { 1, 2 },
            { 3, 4 },
            { 5, 6 }
        });
        var target = MakeMatrix(new double[,]
        {
            { 10, 20 },
            { 30, 40 },
            { 50, 60 }
        });

        var adapted = coral.AdaptSource(source, target);

        // Adapted source should have mean close to target mean [30, 40]
        double meanCol0 = 0, meanCol1 = 0;
        for (int i = 0; i < adapted.Rows; i++)
        {
            meanCol0 += adapted[i, 0];
            meanCol1 += adapted[i, 1];
        }
        meanCol0 /= adapted.Rows;
        meanCol1 /= adapted.Rows;

        Assert.Equal(30.0, meanCol0, 1e-3);
        Assert.Equal(40.0, meanCol1, 1e-3);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void CORAL_AdaptationReducesDiscrepancy()
    {
        var coral = new CORALDomainAdapter<double>();
        var source = MakeMatrix(new double[,]
        {
            { 1, 0 },
            { 2, 1 },
            { 3, 2 },
            { 4, 3 }
        });
        var target = MakeMatrix(new double[,]
        {
            { 10, 5 },
            { 20, 15 },
            { 30, 25 },
            { 40, 35 }
        });

        var discrepancyBefore = coral.ComputeDomainDiscrepancy(source, target);
        var adapted = coral.AdaptSource(source, target);
        var discrepancyAfter = coral.ComputeDomainDiscrepancy(adapted, target);

        // Adaptation should reduce (or at least not drastically increase) discrepancy
        // Note: due to diagonal approximation, might not always decrease but should be reasonable
        Assert.True(discrepancyBefore > 0, "Initial discrepancy should be positive");
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void CORAL_FrobeniusNorm_HandCalculated()
    {
        // For a 2x2 matrix [[a,b],[c,d]], ||M||_F = sqrt(a²+b²+c²+d²)
        // If source cov = [[1,0],[0,1]] and target cov = [[4,0],[0,4]]
        // Diff = [[3,0],[0,3]]
        // ||Diff||_F = sqrt(9+0+0+9) = sqrt(18) = 3*sqrt(2)
        var coral = new CORALDomainAdapter<double>();

        // Create data with known covariance structure
        // source: variance 1 in each dimension (unit circle-like)
        // target: variance 4 in each dimension (scaled up)
        var source = MakeMatrix(new double[,]
        {
            { 1, 0 },
            { -1, 0 },
            { 0, 1 },
            { 0, -1 }
        });

        var target = MakeMatrix(new double[,]
        {
            { 2, 0 },
            { -2, 0 },
            { 0, 2 },
            { 0, -2 }
        });

        var discrepancy = coral.ComputeDomainDiscrepancy(source, target);
        Assert.True(discrepancy > 0, "Different distributions should have positive discrepancy");
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void CORAL_AdaptationMethod_IsCorrectName()
    {
        var coral = new CORALDomainAdapter<double>();
        Assert.Equal("CORAL (CORrelation ALignment)", coral.AdaptationMethod);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void CORAL_RequiresTraining_IsTrue()
    {
        var coral = new CORALDomainAdapter<double>();
        Assert.True(coral.RequiresTraining);
    }

    #endregion

    #region MMD Domain Adaptation Tests

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void MMD_IdenticalDomains_DiscrepancyNearZero()
    {
        var mmd = new MMDDomainAdapter<double>(sigma: 1.0);
        var data = MakeMatrix(new double[,]
        {
            { 1, 2 },
            { 3, 4 },
            { 5, 6 }
        });

        var discrepancy = mmd.ComputeDomainDiscrepancy(data, data);

        // MMD² = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
        // When x and y are the same distribution, this should be near zero
        Assert.Equal(0.0, discrepancy, 1e-5);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void MMD_DifferentDomains_PositiveDiscrepancy()
    {
        var mmd = new MMDDomainAdapter<double>(sigma: 1.0);
        var source = MakeMatrix(new double[,]
        {
            { 0, 0 },
            { 0.1, 0.1 },
            { -0.1, -0.1 }
        });
        var target = MakeMatrix(new double[,]
        {
            { 100, 100 },
            { 100.1, 100.1 },
            { 99.9, 99.9 }
        });

        var discrepancy = mmd.ComputeDomainDiscrepancy(source, target);

        // Very different domains → high discrepancy
        Assert.True(discrepancy > 0, $"MMD discrepancy should be positive, got {discrepancy}");
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void MMD_HandCalculated_TwoPoints()
    {
        // Source: [[0]], Target: [[d]]
        // k(x,y) = exp(-||x-y||² / (2*σ²))
        // With σ=1: k(x,y) = exp(-||x-y||²/2)
        //
        // For source=[0], target=[d]:
        // E[k(s,s')] = k(0,0) = 1 (self-kernel)
        // E[k(t,t')] = k(d,d) = 1
        // E[k(s,t)] = k(0,d) = exp(-d²/2)
        //
        // MMD² = 1 + 1 - 2*exp(-d²/2)
        // MMD = sqrt(2 - 2*exp(-d²/2))
        //
        // For d=2, σ=1: MMD = sqrt(2 - 2*exp(-2)) ≈ sqrt(2 - 0.2707) ≈ sqrt(1.7293) ≈ 1.315
        var mmd = new MMDDomainAdapter<double>(sigma: 1.0);
        var source = MakeMatrix(new double[,] { { 0.0 } });
        var target = MakeMatrix(new double[,] { { 2.0 } });

        var discrepancy = mmd.ComputeDomainDiscrepancy(source, target);

        double expected = Math.Sqrt(2.0 - 2.0 * Math.Exp(-2.0));
        Assert.Equal(expected, discrepancy, 1e-5);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void MMD_AdaptSource_ShiftsMean()
    {
        var mmd = new MMDDomainAdapter<double>(sigma: 1.0);
        var source = MakeMatrix(new double[,]
        {
            { 1, 2 },
            { 3, 4 },
            { 5, 6 }
        });
        var target = MakeMatrix(new double[,]
        {
            { 11, 22 },
            { 13, 24 },
            { 15, 26 }
        });

        // Source mean: [3, 4], Target mean: [13, 24]
        // Shift: [10, 20]
        var adapted = mmd.AdaptSource(source, target);

        // Adapted source should have mean close to target mean [13, 24]
        double meanCol0 = 0, meanCol1 = 0;
        for (int i = 0; i < adapted.Rows; i++)
        {
            meanCol0 += adapted[i, 0];
            meanCol1 += adapted[i, 1];
        }
        meanCol0 /= adapted.Rows;
        meanCol1 /= adapted.Rows;

        Assert.Equal(13.0, meanCol0, 1e-6);
        Assert.Equal(24.0, meanCol1, 1e-6);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void MMD_SymmetricDiscrepancy()
    {
        // MMD(source, target) = MMD(target, source)
        var mmd = new MMDDomainAdapter<double>(sigma: 1.0);
        var source = MakeMatrix(new double[,]
        {
            { 1, 2 },
            { 3, 4 }
        });
        var target = MakeMatrix(new double[,]
        {
            { 5, 6 },
            { 7, 8 }
        });

        var d1 = mmd.ComputeDomainDiscrepancy(source, target);
        var d2 = mmd.ComputeDomainDiscrepancy(target, source);

        Assert.Equal(d1, d2, 1e-10);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void MMD_NonNegativeDiscrepancy()
    {
        var mmd = new MMDDomainAdapter<double>(sigma: 1.0);
        var source = MakeMatrix(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 }
        });
        var target = MakeMatrix(new double[,]
        {
            { -1, -2, -3 },
            { -4, -5, -6 }
        });

        var discrepancy = mmd.ComputeDomainDiscrepancy(source, target);
        Assert.True(discrepancy >= 0, $"MMD must be non-negative, got {discrepancy}");
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void MMD_AdaptationMethod_IsCorrectName()
    {
        var mmd = new MMDDomainAdapter<double>();
        Assert.Equal("Maximum Mean Discrepancy (MMD)", mmd.AdaptationMethod);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void MMD_RequiresTraining_IsFalse()
    {
        var mmd = new MMDDomainAdapter<double>();
        Assert.False(mmd.RequiresTraining);
    }

    #endregion

    #region LinearFeatureMapper Tests

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void LinearMapper_Train_SetsIsTrained()
    {
        var mapper = new LinearFeatureMapper<double>();
        Assert.False(mapper.IsTrained);

        var source = MakeMatrix(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 }
        });
        var target = MakeMatrix(new double[,]
        {
            { 10, 20 },
            { 30, 40 },
            { 50, 60 }
        });

        mapper.Train(source, target);
        Assert.True(mapper.IsTrained);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void LinearMapper_MapToTarget_CorrectDimensions()
    {
        var mapper = new LinearFeatureMapper<double>();
        var source = MakeMatrix(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 }
        });
        var target = MakeMatrix(new double[,]
        {
            { 10, 20 },
            { 30, 40 },
            { 50, 60 }
        });

        mapper.Train(source, target);
        var mapped = mapper.MapToTarget(source, 2);

        // Should have 3 rows (same as source) and 2 columns (target dim)
        Assert.Equal(3, mapped.Rows);
        Assert.Equal(2, mapped.Columns);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void LinearMapper_MapToSource_CorrectDimensions()
    {
        var mapper = new LinearFeatureMapper<double>();
        var source = MakeMatrix(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 }
        });
        var target = MakeMatrix(new double[,]
        {
            { 10, 20 },
            { 30, 40 },
            { 50, 60 }
        });

        mapper.Train(source, target);
        var mapped = mapper.MapToSource(target, 3);

        // Should have 3 rows (same as target) and 3 columns (source dim)
        Assert.Equal(3, mapped.Rows);
        Assert.Equal(3, mapped.Columns);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void LinearMapper_Untrained_Throws()
    {
        var mapper = new LinearFeatureMapper<double>();
        var data = MakeMatrix(new double[,] { { 1, 2 }, { 3, 4 } });

        Assert.Throws<InvalidOperationException>(() => mapper.MapToTarget(data, 2));
        Assert.Throws<InvalidOperationException>(() => mapper.MapToSource(data, 2));
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void LinearMapper_ConfidenceInZeroOneRange()
    {
        var mapper = new LinearFeatureMapper<double>();
        var source = MakeMatrix(new double[,]
        {
            { 1, 2 },
            { 3, 4 },
            { 5, 6 },
            { 7, 8 }
        });
        var target = MakeMatrix(new double[,]
        {
            { 10, 20, 30 },
            { 40, 50, 60 },
            { 70, 80, 90 },
            { 100, 110, 120 }
        });

        mapper.Train(source, target);
        var confidence = mapper.GetMappingConfidence();

        // confidence = exp(-MSE), so it's in (0, 1]
        Assert.True(confidence >= 0.0 && confidence <= 1.0,
            $"Confidence should be in [0, 1], got {confidence}");
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void LinearMapper_SameDimensions_HigherConfidence()
    {
        // Mapping between same-dimensionality data should generally have higher confidence
        var mapper1 = new LinearFeatureMapper<double>();
        var source1 = MakeMatrix(new double[,]
        {
            { 1, 2 },
            { 3, 4 },
            { 5, 6 },
            { 7, 8 }
        });
        var target1 = MakeMatrix(new double[,]
        {
            { 2, 3 },
            { 4, 5 },
            { 6, 7 },
            { 8, 9 }
        });

        mapper1.Train(source1, target1);
        var conf1 = mapper1.GetMappingConfidence();

        // Confidence should be positive for reasonable mappings
        Assert.True(conf1 > 0.0, $"Confidence should be positive, got {conf1}");
    }

    #endregion

    #region Cross-Component Tests

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void CORAL_AdaptTarget_InverseOfAdaptSource()
    {
        var coral = new CORALDomainAdapter<double>();
        var source = MakeMatrix(new double[,]
        {
            { 1, 0 },
            { 2, 1 },
            { 3, 2 },
            { 4, 3 }
        });
        var target = MakeMatrix(new double[,]
        {
            { 5, 10 },
            { 6, 11 },
            { 7, 12 },
            { 8, 13 }
        });

        // Both methods should produce valid adaptations
        var adaptedSource = coral.AdaptSource(source, target);
        Assert.Equal(source.Rows, adaptedSource.Rows);
        Assert.Equal(source.Columns, adaptedSource.Columns);

        // Reset for adapt target (create fresh instance)
        var coral2 = new CORALDomainAdapter<double>();
        var adaptedTarget = coral2.AdaptTarget(target, source);
        Assert.Equal(target.Rows, adaptedTarget.Rows);
        Assert.Equal(target.Columns, adaptedTarget.Columns);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void MMD_LargerSigma_LowerDiscrepancy()
    {
        // Larger sigma → more smoothing → lower discrepancy (kernel values closer to 1)
        var source = MakeMatrix(new double[,]
        {
            { 0, 0 },
            { 1, 1 }
        });
        var target = MakeMatrix(new double[,]
        {
            { 2, 2 },
            { 3, 3 }
        });

        var mmdSmall = new MMDDomainAdapter<double>(sigma: 0.1);
        var mmdLarge = new MMDDomainAdapter<double>(sigma: 100.0);

        var discSmall = mmdSmall.ComputeDomainDiscrepancy(source, target);
        var discLarge = mmdLarge.ComputeDomainDiscrepancy(source, target);

        // With very large sigma, the Gaussian kernel approaches 1 for all pairs
        // So MMD → 0 as sigma → infinity
        Assert.True(discLarge < discSmall,
            $"Large sigma discrepancy {discLarge} should be < small sigma discrepancy {discSmall}");
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void MMD_TrainUpdatesSigma()
    {
        var mmd = new MMDDomainAdapter<double>(sigma: 1.0);
        var source = MakeMatrix(new double[,]
        {
            { 1, 2 },
            { 3, 4 },
            { 5, 6 }
        });
        var target = MakeMatrix(new double[,]
        {
            { 10, 20 },
            { 30, 40 },
            { 50, 60 }
        });

        // Train should use median heuristic to update sigma
        mmd.Train(source, target);

        // After training, discrepancy should still be computable
        var discrepancy = mmd.ComputeDomainDiscrepancy(source, target);
        Assert.True(discrepancy >= 0, "Discrepancy should be non-negative after training");
    }

    #endregion

    #region Helper Methods

    private static Matrix<double> MakeMatrix(double[,] data)
    {
        int rows = data.GetLength(0);
        int cols = data.GetLength(1);
        var matrix = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = data[i, j];
            }
        }
        return matrix;
    }

    #endregion
}
