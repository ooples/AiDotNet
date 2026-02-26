using AiDotNet.Preprocessing.PowerTransforms;
using AiDotNet.Preprocessing.Discretizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

public class PowerTransformsAndDiscretizersDeepMathIntegrationTests
{
    private const double Tolerance = 1e-6;
    private const double LooseTolerance = 1e-3;

    // =====================================================================
    // Box-Cox Transform: formula (x^lambda - 1) / lambda, lambda != 0
    //                     ln(x), lambda == 0
    // Requires strictly positive data
    // =====================================================================

    [Fact]
    public void BoxCox_Lambda1_IsLinearShift()
    {
        // Box-Cox with lambda=1: (x^1 - 1)/1 = x - 1
        // We can't directly set lambda, but we can verify the formula
        // by constructing data where lambda=1 is optimal (already Gaussian data + 1)
        var data = new double[,]
        {
            { 2.0 }, { 3.0 }, { 4.0 }, { 5.0 }, { 6.0 },
            { 7.0 }, { 8.0 }, { 9.0 }, { 10.0 }, { 11.0 }
        };
        var matrix = new Matrix<double>(data);

        var transformer = new PowerTransformer<double>(
            method: PowerTransformMethod.BoxCox, standardize: false);
        var result = transformer.FitTransform(matrix);

        // Lambda should be found via grid search [-2, 2] with step 0.1
        Assert.NotNull(transformer.Lambdas);
        double lambda = transformer.Lambdas[0];

        // Verify transformed values match formula
        for (int i = 0; i < 10; i++)
        {
            double x = data[i, 0];
            double expected;
            if (Math.Abs(lambda) < 1e-10)
                expected = Math.Log(x);
            else
                expected = (Math.Pow(x, lambda) - 1) / lambda;

            Assert.Equal(expected, result[i, 0], Tolerance);
        }
    }

    [Fact]
    public void BoxCox_RejectsNonPositiveData()
    {
        var data = new double[,]
        {
            { 1.0 }, { 0.0 }, { 3.0 }, { 4.0 }, { 5.0 }
        };
        var matrix = new Matrix<double>(data);

        var transformer = new PowerTransformer<double>(
            method: PowerTransformMethod.BoxCox, standardize: false);

        Assert.Throws<ArgumentException>(() => transformer.FitTransform(matrix));
    }

    [Fact]
    public void BoxCox_RejectsNegativeData()
    {
        var data = new double[,]
        {
            { 1.0 }, { -1.0 }, { 3.0 }, { 4.0 }, { 5.0 }
        };
        var matrix = new Matrix<double>(data);

        var transformer = new PowerTransformer<double>(
            method: PowerTransformMethod.BoxCox, standardize: false);

        Assert.Throws<ArgumentException>(() => transformer.FitTransform(matrix));
    }

    [Fact]
    public void BoxCox_TransformedDataHasSmallVariance()
    {
        // Highly skewed data should have reduced variance after transformation
        var data = new double[,]
        {
            { 1.0 }, { 2.0 }, { 3.0 }, { 5.0 }, { 10.0 },
            { 20.0 }, { 50.0 }, { 100.0 }, { 200.0 }, { 500.0 }
        };
        var matrix = new Matrix<double>(data);

        // Compute original variance
        double origMean = 0;
        for (int i = 0; i < 10; i++) origMean += data[i, 0];
        origMean /= 10;
        double origVar = 0;
        for (int i = 0; i < 10; i++)
        {
            double diff = data[i, 0] - origMean;
            origVar += diff * diff;
        }
        origVar /= 10;

        var transformer = new PowerTransformer<double>(
            method: PowerTransformMethod.BoxCox, standardize: false);
        var result = transformer.FitTransform(matrix);

        // Compute transformed variance
        double txMean = 0;
        for (int i = 0; i < 10; i++) txMean += result[i, 0];
        txMean /= 10;
        double txVar = 0;
        for (int i = 0; i < 10; i++)
        {
            double diff = result[i, 0] - txMean;
            txVar += diff * diff;
        }
        txVar /= 10;

        // Coefficient of variation should be much smaller after transform
        double origCV = Math.Sqrt(origVar) / Math.Abs(origMean);
        double txCV = Math.Sqrt(txVar) / Math.Max(Math.Abs(txMean), 1e-10);
        Assert.True(txCV < origCV,
            $"Coefficient of variation should decrease: original={origCV}, transformed={txCV}");
    }

    [Fact]
    public void BoxCox_InverseTransform_RoundTrips()
    {
        var data = new double[,]
        {
            { 1.0 }, { 2.0 }, { 5.0 }, { 10.0 }, { 20.0 },
            { 50.0 }, { 100.0 }, { 150.0 }, { 200.0 }, { 300.0 }
        };
        var matrix = new Matrix<double>(data);

        var transformer = new PowerTransformer<double>(
            method: PowerTransformMethod.BoxCox, standardize: true);
        var transformed = transformer.FitTransform(matrix);
        var reconstructed = transformer.InverseTransform(transformed);

        for (int i = 0; i < 10; i++)
        {
            Assert.Equal(data[i, 0], reconstructed[i, 0], LooseTolerance);
        }
    }

    // =====================================================================
    // Yeo-Johnson Transform:
    //   x >= 0: ((x+1)^lambda - 1)/lambda  (lambda != 0), ln(x+1) (lambda==0)
    //   x < 0:  -((-x+1)^(2-lambda) - 1)/(2-lambda) (lambda!=2), -ln(1-x) (lambda==2)
    // Works with negative values
    // =====================================================================

    [Fact]
    public void YeoJohnson_PositiveValues_MatchFormula()
    {
        var data = new double[,]
        {
            { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 }, { 5.0 },
            { 6.0 }, { 7.0 }, { 8.0 }, { 9.0 }, { 10.0 }
        };
        var matrix = new Matrix<double>(data);

        var transformer = new PowerTransformer<double>(
            method: PowerTransformMethod.YeoJohnson, standardize: false);
        var result = transformer.FitTransform(matrix);

        Assert.NotNull(transformer.Lambdas);
        double lambda = transformer.Lambdas[0];

        for (int i = 0; i < 10; i++)
        {
            double x = data[i, 0];
            double expected;
            if (Math.Abs(lambda) < 1e-10)
                expected = Math.Log(x + 1);
            else
                expected = (Math.Pow(x + 1, lambda) - 1) / lambda;

            Assert.Equal(expected, result[i, 0], Tolerance);
        }
    }

    [Fact]
    public void YeoJohnson_NegativeValues_MatchFormula()
    {
        // Yeo-Johnson handles negative values
        var data = new double[,]
        {
            { -5.0 }, { -4.0 }, { -3.0 }, { -2.0 }, { -1.0 },
            { 0.0 }, { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 }
        };
        var matrix = new Matrix<double>(data);

        var transformer = new PowerTransformer<double>(
            method: PowerTransformMethod.YeoJohnson, standardize: false);
        var result = transformer.FitTransform(matrix);

        Assert.NotNull(transformer.Lambdas);
        double lambda = transformer.Lambdas[0];

        for (int i = 0; i < 10; i++)
        {
            double x = data[i, 0];
            double expected;
            if (x >= 0)
            {
                if (Math.Abs(lambda) < 1e-10)
                    expected = Math.Log(x + 1);
                else
                    expected = (Math.Pow(x + 1, lambda) - 1) / lambda;
            }
            else
            {
                if (Math.Abs(lambda - 2) < 1e-10)
                    expected = -Math.Log(1 - x);
                else
                    expected = -(Math.Pow(1 - x, 2 - lambda) - 1) / (2 - lambda);
            }

            Assert.Equal(expected, result[i, 0], Tolerance);
        }
    }

    [Fact]
    public void YeoJohnson_AcceptsZeros()
    {
        var data = new double[,]
        {
            { 0.0 }, { 0.0 }, { 1.0 }, { 2.0 }, { 3.0 },
            { 4.0 }, { 5.0 }, { 6.0 }, { 7.0 }, { 8.0 }
        };
        var matrix = new Matrix<double>(data);

        var transformer = new PowerTransformer<double>(
            method: PowerTransformMethod.YeoJohnson, standardize: false);
        var result = transformer.FitTransform(matrix);

        // Should not throw and all values should be finite
        for (int i = 0; i < 10; i++)
        {
            Assert.True(double.IsFinite(result[i, 0]),
                $"Transformed value at row {i} should be finite, got {result[i, 0]}");
        }
    }

    [Fact]
    public void YeoJohnson_InverseTransform_RoundTrips()
    {
        var data = new double[,]
        {
            { -3.0 }, { -1.5 }, { -0.5 }, { 0.0 }, { 0.5 },
            { 1.5 }, { 3.0 }, { 5.0 }, { 8.0 }, { 12.0 }
        };
        var matrix = new Matrix<double>(data);

        var transformer = new PowerTransformer<double>(
            method: PowerTransformMethod.YeoJohnson, standardize: true);
        var transformed = transformer.FitTransform(matrix);
        var reconstructed = transformer.InverseTransform(transformed);

        for (int i = 0; i < 10; i++)
        {
            Assert.Equal(data[i, 0], reconstructed[i, 0], LooseTolerance);
        }
    }

    [Fact]
    public void PowerTransformer_Standardize_MeanZeroUnitVariance()
    {
        var data = new double[,]
        {
            { 1.0 }, { 2.0 }, { 4.0 }, { 8.0 }, { 16.0 },
            { 32.0 }, { 64.0 }, { 128.0 }, { 256.0 }, { 512.0 }
        };
        var matrix = new Matrix<double>(data);

        var transformer = new PowerTransformer<double>(
            method: PowerTransformMethod.YeoJohnson, standardize: true);
        var result = transformer.FitTransform(matrix);

        // Compute mean of transformed data
        double mean = 0;
        for (int i = 0; i < 10; i++) mean += result[i, 0];
        mean /= 10;

        // Compute variance (population) of transformed data
        double variance = 0;
        for (int i = 0; i < 10; i++)
        {
            double diff = result[i, 0] - mean;
            variance += diff * diff;
        }
        variance /= 10;

        Assert.Equal(0.0, mean, LooseTolerance);
        Assert.Equal(1.0, variance, LooseTolerance);
    }

    [Fact]
    public void PowerTransformer_LambdaGridSearch_BoundsAreRespected()
    {
        // Lambda search is in [-2, 2] with step 0.1
        var data = new double[,]
        {
            { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 }, { 5.0 },
            { 6.0 }, { 7.0 }, { 8.0 }, { 9.0 }, { 10.0 }
        };
        var matrix = new Matrix<double>(data);

        var transformer = new PowerTransformer<double>(
            method: PowerTransformMethod.YeoJohnson, standardize: false);
        transformer.FitTransform(matrix);

        Assert.NotNull(transformer.Lambdas);
        double lambda = transformer.Lambdas[0];
        Assert.InRange(lambda, -2.0, 2.0);
    }

    [Fact]
    public void PowerTransformer_MultiColumn_IndependentLambdas()
    {
        // Two columns with very different distributions should get different lambdas
        var data = new double[,]
        {
            { 1.0, 100.0 }, { 2.0, 200.0 }, { 3.0, 400.0 }, { 4.0, 800.0 }, { 5.0, 1600.0 },
            { 6.0, 3200.0 }, { 7.0, 6400.0 }, { 8.0, 12800.0 }, { 9.0, 25600.0 }, { 10.0, 51200.0 }
        };
        var matrix = new Matrix<double>(data);

        var transformer = new PowerTransformer<double>(
            method: PowerTransformMethod.BoxCox, standardize: false);
        transformer.FitTransform(matrix);

        Assert.NotNull(transformer.Lambdas);
        // The first column is nearly linear (lambda near 1),
        // the second is exponential (lambda should be much smaller)
        double lambda0 = transformer.Lambdas[0];
        double lambda1 = transformer.Lambdas[1];

        // They should be different since the distributions are very different
        Assert.NotEqual(lambda0, lambda1, 3);
    }

    [Fact]
    public void PowerTransformer_PreservesMonotonicity()
    {
        // Power transforms are monotonic
        var data = new double[,]
        {
            { 1.0 }, { 3.0 }, { 5.0 }, { 7.0 }, { 9.0 },
            { 11.0 }, { 15.0 }, { 20.0 }, { 30.0 }, { 50.0 }
        };
        var matrix = new Matrix<double>(data);

        var transformer = new PowerTransformer<double>(
            method: PowerTransformMethod.BoxCox, standardize: false);
        var result = transformer.FitTransform(matrix);

        for (int i = 1; i < 10; i++)
        {
            Assert.True(result[i, 0] > result[i - 1, 0],
                $"Transform should be monotonically increasing: result[{i}]={result[i, 0]} <= result[{i - 1}]={result[i - 1, 0]}");
        }
    }

    // =====================================================================
    // QuantileTransformer Tests
    // =====================================================================

    [Fact]
    public void QuantileTransformer_Uniform_OutputInZeroOne()
    {
        var data = new double[,]
        {
            { 1.0 }, { 5.0 }, { 10.0 }, { 20.0 }, { 50.0 },
            { 100.0 }, { 200.0 }, { 500.0 }, { 1000.0 }, { 5000.0 }
        };
        var matrix = new Matrix<double>(data);

        var transformer = new QuantileTransformer<double>(
            outputDistribution: OutputDistributionType.Uniform, nQuantiles: 10);
        var result = transformer.FitTransform(matrix);

        for (int i = 0; i < 10; i++)
        {
            Assert.InRange(result[i, 0], 0.0, 1.0);
        }
    }

    [Fact]
    public void QuantileTransformer_Uniform_PreservesOrder()
    {
        var data = new double[,]
        {
            { 3.0 }, { 1.0 }, { 7.0 }, { 2.0 }, { 9.0 },
            { 4.0 }, { 8.0 }, { 6.0 }, { 5.0 }, { 10.0 }
        };
        var matrix = new Matrix<double>(data);

        var transformer = new QuantileTransformer<double>(
            outputDistribution: OutputDistributionType.Uniform, nQuantiles: 10);
        var result = transformer.FitTransform(matrix);

        // Extract values and corresponding transformed values
        var pairs = new List<(double orig, double transformed)>();
        for (int i = 0; i < 10; i++)
            pairs.Add((data[i, 0], result[i, 0]));

        // Sort by original value; transformed values should also be sorted
        pairs.Sort((a, b) => a.orig.CompareTo(b.orig));
        for (int i = 1; i < pairs.Count; i++)
        {
            Assert.True(pairs[i].transformed >= pairs[i - 1].transformed,
                $"Quantile transform should preserve order: at rank {i}, " +
                $"orig={pairs[i].orig} tx={pairs[i].transformed} < prev tx={pairs[i - 1].transformed}");
        }
    }

    [Fact]
    public void QuantileTransformer_Uniform_ExtremeValuesClipped()
    {
        // Fit on [1..10], transform values outside the range
        var trainData = new double[,]
        {
            { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 }, { 5.0 },
            { 6.0 }, { 7.0 }, { 8.0 }, { 9.0 }, { 10.0 }
        };
        var trainMatrix = new Matrix<double>(trainData);

        var transformer = new QuantileTransformer<double>(
            outputDistribution: OutputDistributionType.Uniform, nQuantiles: 10);
        transformer.Fit(trainMatrix);

        var testData = new double[,]
        {
            { -100.0 }, { 0.5 }, { 5.0 }, { 10.5 }, { 1000.0 }
        };
        var testMatrix = new Matrix<double>(testData);
        var result = transformer.Transform(testMatrix);

        // Values below minimum → 0
        Assert.Equal(0.0, result[0, 0], Tolerance);
        Assert.Equal(0.0, result[1, 0], Tolerance);
        // Values above maximum → 1
        Assert.Equal(1.0, result[3, 0], Tolerance);
        Assert.Equal(1.0, result[4, 0], Tolerance);
    }

    [Fact]
    public void QuantileTransformer_Uniform_ConstantFeature_MapToHalf()
    {
        // All identical values → 0.5
        var data = new double[,]
        {
            { 5.0 }, { 5.0 }, { 5.0 }, { 5.0 }, { 5.0 },
            { 5.0 }, { 5.0 }, { 5.0 }, { 5.0 }, { 5.0 }
        };
        var matrix = new Matrix<double>(data);

        var transformer = new QuantileTransformer<double>(
            outputDistribution: OutputDistributionType.Uniform, nQuantiles: 10);
        var result = transformer.FitTransform(matrix);

        for (int i = 0; i < 10; i++)
        {
            Assert.Equal(0.5, result[i, 0], Tolerance);
        }
    }

    [Fact]
    public void QuantileTransformer_Normal_OutputSymmetricAroundZero()
    {
        // Symmetric input should give symmetric output in normal mode
        var data = new double[,]
        {
            { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 }, { 5.0 },
            { 6.0 }, { 7.0 }, { 8.0 }, { 9.0 }, { 10.0 }
        };
        var matrix = new Matrix<double>(data);

        var transformer = new QuantileTransformer<double>(
            outputDistribution: OutputDistributionType.Normal, nQuantiles: 10);
        var result = transformer.FitTransform(matrix);

        // With normal output, extreme values should be large in magnitude
        // Minimum → large negative, maximum → large positive
        Assert.True(result[0, 0] < 0,
            $"Min value should map to negative z-score, got {result[0, 0]}");
        Assert.True(result[9, 0] > 0,
            $"Max value should map to positive z-score, got {result[9, 0]}");
    }

    [Fact]
    public void QuantileTransformer_Normal_ExtremesClippedToEight()
    {
        // Values outside quantile range should be clipped to ±8 for normal output
        var trainData = new double[,]
        {
            { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 }, { 5.0 },
            { 6.0 }, { 7.0 }, { 8.0 }, { 9.0 }, { 10.0 }
        };
        var trainMatrix = new Matrix<double>(trainData);

        var transformer = new QuantileTransformer<double>(
            outputDistribution: OutputDistributionType.Normal, nQuantiles: 10);
        transformer.Fit(trainMatrix);

        var testData = new double[,]
        {
            { -1000.0 }, { 10000.0 }
        };
        var testMatrix = new Matrix<double>(testData);
        var result = transformer.Transform(testMatrix);

        Assert.Equal(-8.0, result[0, 0], Tolerance);
        Assert.Equal(8.0, result[1, 0], Tolerance);
    }

    [Fact]
    public void QuantileTransformer_QuantilesAreSorted()
    {
        var data = new double[,]
        {
            { 10.0 }, { 3.0 }, { 7.0 }, { 1.0 }, { 5.0 },
            { 9.0 }, { 2.0 }, { 8.0 }, { 4.0 }, { 6.0 },
            { 15.0 }, { 20.0 }, { 25.0 }, { 30.0 }, { 50.0 }
        };
        var matrix = new Matrix<double>(data);

        var transformer = new QuantileTransformer<double>(
            outputDistribution: OutputDistributionType.Uniform, nQuantiles: 15);
        transformer.Fit(matrix);

        Assert.NotNull(transformer.Quantiles);
        var quantiles = transformer.Quantiles[0];
        for (int i = 1; i < quantiles.Length; i++)
        {
            Assert.True(quantiles[i] >= quantiles[i - 1],
                $"Quantiles must be non-decreasing: q[{i}]={quantiles[i]} < q[{i - 1}]={quantiles[i - 1]}");
        }
    }

    [Fact]
    public void QuantileTransformer_InverseTransform_RoundTrips()
    {
        var data = new double[,]
        {
            { 1.0 }, { 3.0 }, { 5.0 }, { 7.0 }, { 9.0 },
            { 11.0 }, { 15.0 }, { 20.0 }, { 30.0 }, { 50.0 },
            { 2.0 }, { 4.0 }, { 6.0 }, { 8.0 }, { 10.0 }
        };
        var matrix = new Matrix<double>(data);

        var transformer = new QuantileTransformer<double>(
            outputDistribution: OutputDistributionType.Uniform, nQuantiles: 100);
        var transformed = transformer.FitTransform(matrix);
        var reconstructed = transformer.InverseTransform(transformed);

        for (int i = 0; i < 15; i++)
        {
            // Quantile inverse is approximate, but should be close for training data
            Assert.Equal(data[i, 0], reconstructed[i, 0], 1.0); // within 1.0 tolerance
        }
    }

    [Fact]
    public void QuantileTransformer_NQuantilesValidation()
    {
        Assert.Throws<ArgumentException>(() =>
            new QuantileTransformer<double>(nQuantiles: 5)); // minimum is 10
    }

    // =====================================================================
    // KBinsDiscretizer Tests
    // =====================================================================

    [Fact]
    public void KBins_Uniform_EqualWidthEdges()
    {
        var data = new double[,]
        {
            { 0.0 }, { 2.0 }, { 4.0 }, { 6.0 }, { 8.0 }, { 10.0 }
        };
        var matrix = new Matrix<double>(data);

        var discretizer = new KBinsDiscretizer<double>(
            nBins: 5, strategy: BinningStrategy.Uniform, encode: EncodeMode.Ordinal);
        discretizer.Fit(matrix);

        Assert.NotNull(discretizer.BinEdges);
        var edges = discretizer.BinEdges[0];

        // 5 bins from [0, 10] → edges at 0, 2, 4, 6, 8, 10
        Assert.Equal(6, edges.Length); // nBins + 1 edges
        Assert.Equal(0.0, edges[0], Tolerance);
        Assert.Equal(2.0, edges[1], Tolerance);
        Assert.Equal(4.0, edges[2], Tolerance);
        Assert.Equal(6.0, edges[3], Tolerance);
        Assert.Equal(8.0, edges[4], Tolerance);
        Assert.Equal(10.0, edges[5], Tolerance);
    }

    [Fact]
    public void KBins_Uniform_OrdinalEncoding_CorrectBins()
    {
        var data = new double[,]
        {
            { 0.0 }, { 3.0 }, { 5.0 }, { 7.0 }, { 10.0 }
        };
        var matrix = new Matrix<double>(data);

        var discretizer = new KBinsDiscretizer<double>(
            nBins: 5, strategy: BinningStrategy.Uniform, encode: EncodeMode.Ordinal);
        var result = discretizer.FitTransform(matrix);

        // Range [0, 10], 5 bins: [0,2], (2,4], (4,6], (6,8], (8,10]
        Assert.Equal(0.0, result[0, 0], Tolerance); // 0 → bin 0
        Assert.Equal(1.0, result[1, 0], Tolerance); // 3 → bin 1 (3 <= 4)
        Assert.Equal(2.0, result[2, 0], Tolerance); // 5 → bin 2 (5 <= 6)
        Assert.Equal(3.0, result[3, 0], Tolerance); // 7 → bin 3 (7 <= 8)
        Assert.Equal(4.0, result[4, 0], Tolerance); // 10 → bin 4 (last bin)
    }

    [Fact]
    public void KBins_Uniform_NormalizedEncoding_InZeroOne()
    {
        var data = new double[,]
        {
            { 0.0 }, { 5.0 }, { 10.0 }
        };
        var matrix = new Matrix<double>(data);

        var discretizer = new KBinsDiscretizer<double>(
            nBins: 3, strategy: BinningStrategy.Uniform, encode: EncodeMode.Normalized);
        var result = discretizer.FitTransform(matrix);

        // Normalized: bin_index / (nBins - 1)
        // Bin 0: 0 / 2 = 0.0
        // Bin 1: 1 / 2 = 0.5
        // Bin 2: 2 / 2 = 1.0
        for (int i = 0; i < 3; i++)
        {
            Assert.InRange(result[i, 0], 0.0, 1.0);
        }
        Assert.Equal(0.0, result[0, 0], Tolerance);
        Assert.Equal(1.0, result[2, 0], Tolerance);
    }

    [Fact]
    public void KBins_Quantile_EqualFrequencyBins()
    {
        // With 10 values and 5 bins, each bin should have ~2 values
        var data = new double[,]
        {
            { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 }, { 5.0 },
            { 6.0 }, { 7.0 }, { 8.0 }, { 9.0 }, { 10.0 }
        };
        var matrix = new Matrix<double>(data);

        var discretizer = new KBinsDiscretizer<double>(
            nBins: 5, strategy: BinningStrategy.Quantile, encode: EncodeMode.Ordinal);
        var result = discretizer.FitTransform(matrix);

        // All bin indices should be in [0, 4]
        for (int i = 0; i < 10; i++)
        {
            double binIdx = result[i, 0];
            Assert.InRange(binIdx, 0.0, 4.0);
        }

        // First value should be in bin 0, last in bin 4
        Assert.Equal(0.0, result[0, 0], Tolerance);
        Assert.Equal(4.0, result[9, 0], Tolerance);
    }

    [Fact]
    public void KBins_Quantile_BinEdgesAreNonDecreasing()
    {
        var data = new double[,]
        {
            { 1.0 }, { 1.0 }, { 5.0 }, { 10.0 }, { 100.0 },
            { 100.0 }, { 100.0 }, { 200.0 }, { 500.0 }, { 1000.0 }
        };
        var matrix = new Matrix<double>(data);

        var discretizer = new KBinsDiscretizer<double>(
            nBins: 4, strategy: BinningStrategy.Quantile, encode: EncodeMode.Ordinal);
        discretizer.Fit(matrix);

        Assert.NotNull(discretizer.BinEdges);
        var edges = discretizer.BinEdges[0];

        for (int i = 1; i < edges.Length; i++)
        {
            Assert.True(edges[i] >= edges[i - 1],
                $"Bin edges must be non-decreasing: edge[{i}]={edges[i]} < edge[{i - 1}]={edges[i - 1]}");
        }
    }

    [Fact]
    public void KBins_InverseTransform_ReturnsMidpoints()
    {
        var data = new double[,]
        {
            { 0.0 }, { 5.0 }, { 10.0 }
        };
        var matrix = new Matrix<double>(data);

        var discretizer = new KBinsDiscretizer<double>(
            nBins: 2, strategy: BinningStrategy.Uniform, encode: EncodeMode.Ordinal);
        var transformed = discretizer.FitTransform(matrix);
        var inverse = discretizer.InverseTransform(transformed);

        // Edges: [0, 5, 10]
        // Bin 0 midpoint: (0 + 5) / 2 = 2.5
        // Bin 1 midpoint: (5 + 10) / 2 = 7.5
        // 0 → bin 0 → midpoint 2.5
        Assert.Equal(2.5, inverse[0, 0], Tolerance);
        // 10 → bin 1 → midpoint 7.5
        Assert.Equal(7.5, inverse[2, 0], Tolerance);
    }

    [Fact]
    public void KBins_MinBinsValidation()
    {
        Assert.Throws<ArgumentException>(() =>
            new KBinsDiscretizer<double>(nBins: 1));
    }

    [Fact]
    public void KBins_MultiColumn_IndependentBins()
    {
        var data = new double[,]
        {
            { 0.0, 100.0 },
            { 5.0, 200.0 },
            { 10.0, 300.0 }
        };
        var matrix = new Matrix<double>(data);

        var discretizer = new KBinsDiscretizer<double>(
            nBins: 3, strategy: BinningStrategy.Uniform, encode: EncodeMode.Ordinal);
        discretizer.Fit(matrix);

        Assert.NotNull(discretizer.BinEdges);
        // Column 0: edges at [0, 3.33, 6.67, 10]
        // Column 1: edges at [100, 166.67, 233.33, 300]
        var edges0 = discretizer.BinEdges[0];
        var edges1 = discretizer.BinEdges[1];

        Assert.Equal(0.0, edges0[0], Tolerance);
        Assert.Equal(10.0, edges0[3], Tolerance);
        Assert.Equal(100.0, edges1[0], Tolerance);
        Assert.Equal(300.0, edges1[3], Tolerance);
    }

    [Fact]
    public void KBins_ValueAboveMax_GoesToLastBin()
    {
        var trainData = new double[,]
        {
            { 0.0 }, { 5.0 }, { 10.0 }
        };
        var trainMatrix = new Matrix<double>(trainData);

        var discretizer = new KBinsDiscretizer<double>(
            nBins: 2, strategy: BinningStrategy.Uniform, encode: EncodeMode.Ordinal);
        discretizer.Fit(trainMatrix);

        var testData = new double[,]
        {
            { 100.0 }
        };
        var testMatrix = new Matrix<double>(testData);
        var result = discretizer.Transform(testMatrix);

        // Value above max → last bin (bin 1)
        Assert.Equal(1.0, result[0, 0], Tolerance);
    }

    [Fact]
    public void KBins_Uniform_EdgeWidthsAreEqual()
    {
        var data = new double[,]
        {
            { 0.0 }, { 10.0 }, { 20.0 }, { 30.0 }, { 40.0 },
            { 50.0 }, { 60.0 }, { 70.0 }, { 80.0 }, { 100.0 }
        };
        var matrix = new Matrix<double>(data);

        var discretizer = new KBinsDiscretizer<double>(
            nBins: 4, strategy: BinningStrategy.Uniform, encode: EncodeMode.Ordinal);
        discretizer.Fit(matrix);

        Assert.NotNull(discretizer.BinEdges);
        var edges = discretizer.BinEdges[0];

        // All bin widths should be equal: 100/4 = 25
        double expectedWidth = (100.0 - 0.0) / 4.0;
        for (int i = 1; i < edges.Length; i++)
        {
            double width = edges[i] - edges[i - 1];
            Assert.Equal(expectedWidth, width, Tolerance);
        }
    }

    [Fact]
    public void KBins_OrdinalEncoding_IntegerValues()
    {
        var data = new double[,]
        {
            { 0.0 }, { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 },
            { 5.0 }, { 6.0 }, { 7.0 }, { 8.0 }, { 9.0 }
        };
        var matrix = new Matrix<double>(data);

        var discretizer = new KBinsDiscretizer<double>(
            nBins: 5, strategy: BinningStrategy.Uniform, encode: EncodeMode.Ordinal);
        var result = discretizer.FitTransform(matrix);

        for (int i = 0; i < 10; i++)
        {
            double val = result[i, 0];
            // All ordinal outputs should be non-negative integers < nBins
            Assert.Equal(Math.Floor(val), val, Tolerance);
            Assert.InRange(val, 0.0, 4.0);
        }
    }
}
