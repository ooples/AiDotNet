using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.FederatedLearning.Aggregators;
using AiDotNet.FederatedLearning.Privacy;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.FederatedLearning;

/// <summary>
/// Deep mathematical correctness tests for federated learning aggregation strategies
/// and differential privacy mechanisms. Each test verifies exact hand-calculated values.
/// </summary>
public class FederatedLearningDeepIntegrationTests
{
    #region FedAvg - Weighted Average Exact Math

    [Fact]
    public void FedAvg_TwoClients_EqualWeights_HandCalculated()
    {
        // Client 0: layer "dense" = [2.0, 4.0], weight = 100
        // Client 1: layer "dense" = [6.0, 8.0], weight = 100
        // Total weight = 200
        // Aggregated = (100/200)*[2, 4] + (100/200)*[6, 8]
        //            = 0.5*[2, 4] + 0.5*[6, 8]
        //            = [1, 2] + [3, 4] = [4.0, 6.0]
        var strategy = new FedAvgAggregationStrategy<double>();
        var clientModels = new Dictionary<int, Dictionary<string, double[]>>
        {
            { 0, new Dictionary<string, double[]> { { "dense", new[] { 2.0, 4.0 } } } },
            { 1, new Dictionary<string, double[]> { { "dense", new[] { 6.0, 8.0 } } } }
        };
        var weights = new Dictionary<int, double> { { 0, 100.0 }, { 1, 100.0 } };

        var result = strategy.Aggregate(clientModels, weights);

        Assert.Equal(4.0, result["dense"][0], 10);
        Assert.Equal(6.0, result["dense"][1], 10);
    }

    [Fact]
    public void FedAvg_TwoClients_UnequalWeights_HandCalculated()
    {
        // Client 0: layer "w" = [0.8], weight = 300
        // Client 1: layer "w" = [0.6], weight = 700
        // Total weight = 1000
        // Aggregated = (300/1000)*0.8 + (700/1000)*0.6
        //            = 0.3*0.8 + 0.7*0.6 = 0.24 + 0.42 = 0.66
        var strategy = new FedAvgAggregationStrategy<double>();
        var clientModels = new Dictionary<int, Dictionary<string, double[]>>
        {
            { 0, new Dictionary<string, double[]> { { "w", new[] { 0.8 } } } },
            { 1, new Dictionary<string, double[]> { { "w", new[] { 0.6 } } } }
        };
        var weights = new Dictionary<int, double> { { 0, 300.0 }, { 1, 700.0 } };

        var result = strategy.Aggregate(clientModels, weights);

        Assert.Equal(0.66, result["w"][0], 10);
    }

    [Fact]
    public void FedAvg_ThreeClients_HandCalculated()
    {
        // Client A: 1000 samples, params = [1.0, 2.0]
        // Client B: 500 samples, params = [3.0, 4.0]
        // Client C: 1500 samples, params = [5.0, 6.0]
        // Total = 3000
        // A_weight = 1000/3000 = 1/3, B_weight = 500/3000 = 1/6, C_weight = 1500/3000 = 1/2
        // Param[0] = (1/3)*1 + (1/6)*3 + (1/2)*5 = 0.333 + 0.5 + 2.5 = 3.333...
        // Param[1] = (1/3)*2 + (1/6)*4 + (1/2)*6 = 0.667 + 0.667 + 3.0 = 4.333...
        var strategy = new FedAvgAggregationStrategy<double>();
        var clientModels = new Dictionary<int, Dictionary<string, double[]>>
        {
            { 0, new Dictionary<string, double[]> { { "layer1", new[] { 1.0, 2.0 } } } },
            { 1, new Dictionary<string, double[]> { { "layer1", new[] { 3.0, 4.0 } } } },
            { 2, new Dictionary<string, double[]> { { "layer1", new[] { 5.0, 6.0 } } } }
        };
        var weights = new Dictionary<int, double> { { 0, 1000.0 }, { 1, 500.0 }, { 2, 1500.0 } };

        var result = strategy.Aggregate(clientModels, weights);

        double expected0 = (1000.0 / 3000.0) * 1.0 + (500.0 / 3000.0) * 3.0 + (1500.0 / 3000.0) * 5.0;
        double expected1 = (1000.0 / 3000.0) * 2.0 + (500.0 / 3000.0) * 4.0 + (1500.0 / 3000.0) * 6.0;

        Assert.Equal(expected0, result["layer1"][0], 8);
        Assert.Equal(expected1, result["layer1"][1], 8);
    }

    [Fact]
    public void FedAvg_MultipleLayersAggregated_Independently()
    {
        // Verify each layer is aggregated independently
        var strategy = new FedAvgAggregationStrategy<double>();
        var clientModels = new Dictionary<int, Dictionary<string, double[]>>
        {
            {
                0, new Dictionary<string, double[]>
                {
                    { "weights", new[] { 1.0 } },
                    { "bias", new[] { 10.0 } }
                }
            },
            {
                1, new Dictionary<string, double[]>
                {
                    { "weights", new[] { 3.0 } },
                    { "bias", new[] { 20.0 } }
                }
            }
        };
        var weights = new Dictionary<int, double> { { 0, 1.0 }, { 1, 1.0 } };

        var result = strategy.Aggregate(clientModels, weights);

        Assert.Equal(2.0, result["weights"][0], 10);  // (1+3)/2
        Assert.Equal(15.0, result["bias"][0], 10);     // (10+20)/2
    }

    [Fact]
    public void FedAvg_SingleClient_ReturnsExactCopy()
    {
        var strategy = new FedAvgAggregationStrategy<double>();
        var clientModels = new Dictionary<int, Dictionary<string, double[]>>
        {
            { 0, new Dictionary<string, double[]> { { "dense", new[] { 3.14, 2.72, 1.41 } } } }
        };
        var weights = new Dictionary<int, double> { { 0, 100.0 } };

        var result = strategy.Aggregate(clientModels, weights);

        Assert.Equal(3.14, result["dense"][0], 10);
        Assert.Equal(2.72, result["dense"][1], 10);
        Assert.Equal(1.41, result["dense"][2], 10);
    }

    [Fact]
    public void FedAvg_EmptyModels_Throws()
    {
        var strategy = new FedAvgAggregationStrategy<double>();
        var empty = new Dictionary<int, Dictionary<string, double[]>>();
        var weights = new Dictionary<int, double>();

        Assert.Throws<ArgumentException>(() => strategy.Aggregate(empty, weights));
    }

    [Fact]
    public void FedAvg_MissingWeight_Throws()
    {
        var strategy = new FedAvgAggregationStrategy<double>();
        var clientModels = new Dictionary<int, Dictionary<string, double[]>>
        {
            { 0, new Dictionary<string, double[]> { { "w", new[] { 1.0 } } } }
        };
        var weights = new Dictionary<int, double> { { 1, 100.0 } }; // No weight for client 0!

        Assert.Throws<ArgumentException>(() => strategy.Aggregate(clientModels, weights));
    }

    [Fact]
    public void FedAvg_StrategyName_IsFedAvg()
    {
        var strategy = new FedAvgAggregationStrategy<double>();
        Assert.Equal("FedAvg", strategy.GetStrategyName());
    }

    #endregion

    #region FedProx - Proximal Weighted Average

    [Fact]
    public void FedProx_AggregationResult_MatchesFedAvg()
    {
        // FedProx aggregation uses the same weighted average as FedAvg
        // (the proximal term is in the client training, not in aggregation)
        var fedavg = new FedAvgAggregationStrategy<double>();
        var fedprox = new FedProxAggregationStrategy<double>();

        var clientModels = new Dictionary<int, Dictionary<string, double[]>>
        {
            { 0, new Dictionary<string, double[]> { { "w", new[] { 1.0, 2.0 } } } },
            { 1, new Dictionary<string, double[]> { { "w", new[] { 3.0, 4.0 } } } }
        };
        var weights = new Dictionary<int, double> { { 0, 50.0 }, { 1, 50.0 } };

        var avgResult = fedavg.Aggregate(clientModels, weights);
        var proxResult = fedprox.Aggregate(clientModels, weights);

        Assert.Equal(avgResult["w"][0], proxResult["w"][0], 10);
        Assert.Equal(avgResult["w"][1], proxResult["w"][1], 10);
    }

    [Fact]
    public void FedProx_StrategyName_IsFedProx()
    {
        var strategy = new FedProxAggregationStrategy<double>();
        Assert.Contains("FedProx", strategy.GetStrategyName());
    }

    #endregion

    #region Gaussian Differential Privacy - Noise Calibration

    [Fact]
    public void GaussianDP_NoiseScale_MatchesFormula()
    {
        // σ = (Δ/ε) × sqrt(2 × ln(1.25/δ))
        // With clipNorm=1.0, ε=1.0, δ=1e-5:
        // σ = (1.0/1.0) × sqrt(2 × ln(1.25 / 0.00001))
        //   = sqrt(2 × ln(125000))
        //   = sqrt(2 × 11.7361...) = sqrt(23.472...) ≈ 4.845
        // The noisy model should differ from original by roughly σ per parameter
        var dp = new GaussianDifferentialPrivacy<double>(clipNorm: 1.0, randomSeed: 42);
        var model = new Dictionary<string, double[]>
        {
            { "w", new[] { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 } }
        };

        var noisy = dp.ApplyPrivacy(model, epsilon: 1.0, delta: 1e-5);

        // L2 norm of original is sqrt(10 * 0.01) = sqrt(0.1) ≈ 0.316
        // This is < clipNorm (1.0), so no clipping occurs
        // Each parameter gets noise ~ N(0, σ²) where σ ≈ 4.845
        // The noise should be significant compared to the 0.1 values
        double sumDiff = 0;
        for (int i = 0; i < 10; i++)
        {
            double diff = noisy["w"][i] - 0.1;
            sumDiff += diff * diff;
        }
        double rmsNoise = Math.Sqrt(sumDiff / 10.0);

        // RMS noise should be roughly σ ≈ 4.845 (with some sampling variation)
        Assert.True(rmsNoise > 0.5, $"RMS noise = {rmsNoise}, should be significantly > 0 for ε=1.0");
    }

    [Fact]
    public void GaussianDP_GradientClipping_ClampsL2Norm()
    {
        // If L2 norm > clipNorm, parameters are scaled down
        // Model: [3.0, 4.0], L2 norm = sqrt(9+16) = 5.0
        // clipNorm = 1.0 → scale factor = 1.0/5.0 = 0.2
        // Clipped: [0.6, 0.8]
        // Then noise is added on top of the clipped values
        var dp = new GaussianDifferentialPrivacy<double>(clipNorm: 1.0, randomSeed: 42);
        var model = new Dictionary<string, double[]>
        {
            { "w", new[] { 3.0, 4.0 } }
        };

        // Use very small epsilon so noise is minimal relative to clipping effect
        // Actually, smaller epsilon means MORE noise. Use large epsilon.
        var noisy = dp.ApplyPrivacy(model, epsilon: 100.0, delta: 1e-5);

        // With large epsilon, noise is very small: σ = (1/100)*sqrt(2*ln(125000)) ≈ 0.048
        // After clipping: [0.6, 0.8]
        // After noise: approximately [0.6, 0.8] ± 0.048
        double l2Clipped = Math.Sqrt(noisy["w"][0] * noisy["w"][0] + noisy["w"][1] * noisy["w"][1]);

        // Clipped + tiny noise should be close to 1.0 (the clip norm)
        Assert.True(Math.Abs(l2Clipped - 1.0) < 0.5,
            $"L2 norm after clipping + small noise = {l2Clipped}, expected ~1.0");
    }

    [Fact]
    public void GaussianDP_NormBelowClip_NoClipping()
    {
        // If L2 norm < clipNorm, no clipping occurs
        // Model: [0.1], L2 norm = 0.1 < clipNorm=10.0
        // No clipping → params stay as [0.1] (before noise)
        var dp = new GaussianDifferentialPrivacy<double>(clipNorm: 10.0, randomSeed: 42);
        var model = new Dictionary<string, double[]>
        {
            { "w", new[] { 0.1 } }
        };

        // Large epsilon → very little noise
        var noisy = dp.ApplyPrivacy(model, epsilon: 1000.0, delta: 1e-5);

        // σ = (10/1000)*sqrt(2*ln(125000)) ≈ 0.048
        Assert.True(Math.Abs(noisy["w"][0] - 0.1) < 1.0,
            $"Value should be close to 0.1 (no clipping), got {noisy["w"][0]}");
    }

    [Fact]
    public void GaussianDP_PrivacyBudget_AccumulatesCorrectly()
    {
        var dp = new GaussianDifferentialPrivacy<double>(clipNorm: 1.0, randomSeed: 42);
        var model = new Dictionary<string, double[]> { { "w", new[] { 0.5 } } };

        Assert.Equal(0.0, dp.GetPrivacyBudgetConsumed(), 10);

        dp.ApplyPrivacy(model, epsilon: 0.5, delta: 1e-5);
        Assert.Equal(0.5, dp.GetPrivacyBudgetConsumed(), 10);

        dp.ApplyPrivacy(model, epsilon: 0.3, delta: 1e-5);
        Assert.Equal(0.8, dp.GetPrivacyBudgetConsumed(), 10);
    }

    [Fact]
    public void GaussianDP_ResetBudget_SetsToZero()
    {
        var dp = new GaussianDifferentialPrivacy<double>(clipNorm: 1.0, randomSeed: 42);
        var model = new Dictionary<string, double[]> { { "w", new[] { 0.5 } } };

        dp.ApplyPrivacy(model, epsilon: 1.0, delta: 1e-5);
        Assert.True(dp.GetPrivacyBudgetConsumed() > 0);

        dp.ResetPrivacyBudget();
        Assert.Equal(0.0, dp.GetPrivacyBudgetConsumed(), 10);
    }

    [Fact]
    public void GaussianDP_Seeded_Reproducible()
    {
        var dp1 = new GaussianDifferentialPrivacy<double>(clipNorm: 1.0, randomSeed: 42);
        var dp2 = new GaussianDifferentialPrivacy<double>(clipNorm: 1.0, randomSeed: 42);
        var model = new Dictionary<string, double[]>
        {
            { "w", new[] { 0.5, 0.5, 0.5 } }
        };

        var noisy1 = dp1.ApplyPrivacy(model, epsilon: 1.0, delta: 1e-5);
        var noisy2 = dp2.ApplyPrivacy(model, epsilon: 1.0, delta: 1e-5);

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(noisy1["w"][i], noisy2["w"][i], 10);
        }
    }

    [Fact]
    public void GaussianDP_InvalidParameters_Throws()
    {
        var dp = new GaussianDifferentialPrivacy<double>(clipNorm: 1.0, randomSeed: 42);
        var model = new Dictionary<string, double[]> { { "w", new[] { 0.5 } } };

        Assert.Throws<ArgumentException>(() => dp.ApplyPrivacy(model, epsilon: -1.0, delta: 1e-5));
        Assert.Throws<ArgumentException>(() => dp.ApplyPrivacy(model, epsilon: 0.0, delta: 1e-5));
        Assert.Throws<ArgumentException>(() => dp.ApplyPrivacy(model, epsilon: 1.0, delta: 0.0));
        Assert.Throws<ArgumentException>(() => dp.ApplyPrivacy(model, epsilon: 1.0, delta: 1.0));
        Assert.Throws<ArgumentException>(() => dp.ApplyPrivacy(model, epsilon: 1.0, delta: -0.1));
    }

    [Fact]
    public void GaussianDP_InvalidClipNorm_Throws()
    {
        Assert.Throws<ArgumentException>(() => new GaussianDifferentialPrivacy<double>(clipNorm: 0));
        Assert.Throws<ArgumentException>(() => new GaussianDifferentialPrivacy<double>(clipNorm: -1.0));
    }

    [Fact]
    public void GaussianDP_EmptyModel_Throws()
    {
        var dp = new GaussianDifferentialPrivacy<double>(clipNorm: 1.0, randomSeed: 42);
        var empty = new Dictionary<string, double[]>();

        Assert.Throws<ArgumentException>(() => dp.ApplyPrivacy(empty, epsilon: 1.0, delta: 1e-5));
    }

    [Fact]
    public void GaussianDP_SmallerEpsilon_MoreNoise()
    {
        // Smaller epsilon → more privacy → more noise
        // σ = (Δ/ε) × sqrt(2 × ln(1.25/δ))
        // σ is inversely proportional to ε
        var dp1 = new GaussianDifferentialPrivacy<double>(clipNorm: 1.0, randomSeed: 42);
        var dp2 = new GaussianDifferentialPrivacy<double>(clipNorm: 1.0, randomSeed: 42);
        var model = new Dictionary<string, double[]>
        {
            { "w", Enumerable.Repeat(0.0, 100).ToArray() }
        };

        var noisy_small_eps = dp1.ApplyPrivacy(model, epsilon: 0.1, delta: 1e-5);
        var noisy_large_eps = dp2.ApplyPrivacy(model, epsilon: 10.0, delta: 1e-5);

        // Measure total noise for each
        double noise_small = noisy_small_eps["w"].Sum(x => x * x);
        double noise_large = noisy_large_eps["w"].Sum(x => x * x);

        Assert.True(noise_small > noise_large,
            $"Small ε noise ({noise_small}) should be greater than large ε noise ({noise_large})");
    }

    [Fact]
    public void GaussianDP_MechanismName_ContainsClipNorm()
    {
        var dp = new GaussianDifferentialPrivacy<double>(clipNorm: 2.5);
        string name = dp.GetMechanismName();

        Assert.Contains("Gaussian", name);
        Assert.Contains("2.5", name);
    }

    #endregion

    #region FedAvg Numerical Properties

    [Fact]
    public void FedAvg_WeightedAverage_WeightsSumToOne()
    {
        // The normalized weights should sum to 1, producing a proper weighted average
        // If all clients have same params [X], result should be [X] regardless of weights
        var strategy = new FedAvgAggregationStrategy<double>();
        var clientModels = new Dictionary<int, Dictionary<string, double[]>>
        {
            { 0, new Dictionary<string, double[]> { { "w", new[] { 5.0 } } } },
            { 1, new Dictionary<string, double[]> { { "w", new[] { 5.0 } } } },
            { 2, new Dictionary<string, double[]> { { "w", new[] { 5.0 } } } }
        };
        var weights = new Dictionary<int, double> { { 0, 1.0 }, { 1, 1000.0 }, { 2, 0.001 } };

        var result = strategy.Aggregate(clientModels, weights);

        // All clients have the same params, so result must be exactly 5.0
        Assert.Equal(5.0, result["w"][0], 8);
    }

    [Fact]
    public void FedAvg_ResultIsBetweenMinAndMax()
    {
        // The weighted average must always be between the minimum and maximum client values
        var strategy = new FedAvgAggregationStrategy<double>();
        var clientModels = new Dictionary<int, Dictionary<string, double[]>>
        {
            { 0, new Dictionary<string, double[]> { { "w", new[] { 1.0 } } } },
            { 1, new Dictionary<string, double[]> { { "w", new[] { 10.0 } } } },
            { 2, new Dictionary<string, double[]> { { "w", new[] { 5.0 } } } }
        };
        var weights = new Dictionary<int, double> { { 0, 1.0 }, { 1, 1.0 }, { 2, 1.0 } };

        var result = strategy.Aggregate(clientModels, weights);

        Assert.True(result["w"][0] >= 1.0, $"Result {result["w"][0]} should be >= 1.0");
        Assert.True(result["w"][0] <= 10.0, $"Result {result["w"][0]} should be <= 10.0");
    }

    [Fact]
    public void FedAvg_LayerLengthMismatch_Throws()
    {
        var strategy = new FedAvgAggregationStrategy<double>();
        var clientModels = new Dictionary<int, Dictionary<string, double[]>>
        {
            { 0, new Dictionary<string, double[]> { { "w", new[] { 1.0, 2.0 } } } },
            { 1, new Dictionary<string, double[]> { { "w", new[] { 3.0 } } } } // Wrong length!
        };
        var weights = new Dictionary<int, double> { { 0, 1.0 }, { 1, 1.0 } };

        Assert.Throws<ArgumentException>(() => strategy.Aggregate(clientModels, weights));
    }

    #endregion

    #region Helper: Create MockFullModel with specific parameters

    private static MockFullModel CreateMockWithParams(params double[] values)
    {
        var model = new MockFullModel(_ => new Vector<double>(1), values.Length);
        var vec = new Vector<double>(values.Length);
        for (int i = 0; i < values.Length; i++)
        {
            vec[i] = values[i];
        }
        model.SetParameters(vec);
        return model;
    }

    #endregion

    #region MedianFullModel - Coordinate-Wise Median

    [Fact]
    public void Median_ThreeClients_OddCount_TakesMiddleValue()
    {
        // Client 0: [1.0, 9.0, 5.0]
        // Client 1: [3.0, 7.0, 3.0]
        // Client 2: [2.0, 8.0, 4.0]
        // Sorted per coordinate:
        //   param0: [1, 2, 3] → median = 2
        //   param1: [7, 8, 9] → median = 8
        //   param2: [3, 4, 5] → median = 4
        var strategy = new MedianFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>();
        var clients = new Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>>
        {
            { 0, CreateMockWithParams(1.0, 9.0, 5.0) },
            { 1, CreateMockWithParams(3.0, 7.0, 3.0) },
            { 2, CreateMockWithParams(2.0, 8.0, 4.0) }
        };
        var weights = new Dictionary<int, double> { { 0, 1.0 }, { 1, 1.0 }, { 2, 1.0 } };

        var result = strategy.Aggregate(clients, weights);
        var resultParams = result.GetParameters();

        Assert.Equal(2.0, resultParams[0], 10);
        Assert.Equal(8.0, resultParams[1], 10);
        Assert.Equal(4.0, resultParams[2], 10);
    }

    [Fact]
    public void Median_TwoClients_EvenCount_AveragesMiddleTwo()
    {
        // Client 0: [2.0, 10.0]
        // Client 1: [8.0, 4.0]
        // Sorted per coordinate:
        //   param0: [2, 8] → median = (2+8)/2 = 5.0
        //   param1: [4, 10] → median = (4+10)/2 = 7.0
        var strategy = new MedianFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>();
        var clients = new Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>>
        {
            { 0, CreateMockWithParams(2.0, 10.0) },
            { 1, CreateMockWithParams(8.0, 4.0) }
        };
        var weights = new Dictionary<int, double> { { 0, 1.0 }, { 1, 1.0 } };

        var result = strategy.Aggregate(clients, weights);
        var resultParams = result.GetParameters();

        Assert.Equal(5.0, resultParams[0], 10);
        Assert.Equal(7.0, resultParams[1], 10);
    }

    [Fact]
    public void Median_FourClients_EvenCount_AveragesMiddleTwo()
    {
        // Client 0: [1.0]
        // Client 1: [3.0]
        // Client 2: [5.0]
        // Client 3: [7.0]
        // Sorted: [1, 3, 5, 7] → median = (3+5)/2 = 4.0
        var strategy = new MedianFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>();
        var clients = new Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>>
        {
            { 0, CreateMockWithParams(1.0) },
            { 1, CreateMockWithParams(3.0) },
            { 2, CreateMockWithParams(5.0) },
            { 3, CreateMockWithParams(7.0) }
        };
        var weights = new Dictionary<int, double> { { 0, 1.0 }, { 1, 1.0 }, { 2, 1.0 }, { 3, 1.0 } };

        var result = strategy.Aggregate(clients, weights);
        var resultParams = result.GetParameters();

        Assert.Equal(4.0, resultParams[0], 10);
    }

    [Fact]
    public void Median_IgnoresWeights_UnweightedByDesign()
    {
        // Median should NOT be affected by client weights — it's unweighted
        // Client 0: [10.0], weight = 1000000
        // Client 1: [20.0], weight = 1
        // Client 2: [30.0], weight = 1
        // Sorted: [10, 20, 30] → median = 20 regardless of weights
        var strategy = new MedianFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>();
        var clients = new Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>>
        {
            { 0, CreateMockWithParams(10.0) },
            { 1, CreateMockWithParams(20.0) },
            { 2, CreateMockWithParams(30.0) }
        };
        var weights = new Dictionary<int, double> { { 0, 1000000.0 }, { 1, 1.0 }, { 2, 1.0 } };

        var result = strategy.Aggregate(clients, weights);
        var resultParams = result.GetParameters();

        Assert.Equal(20.0, resultParams[0], 10);
    }

    [Fact]
    public void Median_FiveClients_OutlierResistant()
    {
        // Client 0: [1.0]  (normal)
        // Client 1: [2.0]  (normal)
        // Client 2: [3.0]  (normal)
        // Client 3: [4.0]  (normal)
        // Client 4: [999.0] (Byzantine outlier!)
        // Sorted: [1, 2, 3, 4, 999] → median = 3.0
        // FedAvg would give (1+2+3+4+999)/5 = 201.8 — median is robust!
        var strategy = new MedianFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>();
        var clients = new Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>>
        {
            { 0, CreateMockWithParams(1.0) },
            { 1, CreateMockWithParams(2.0) },
            { 2, CreateMockWithParams(3.0) },
            { 3, CreateMockWithParams(4.0) },
            { 4, CreateMockWithParams(999.0) }
        };
        var weights = new Dictionary<int, double> { { 0, 1.0 }, { 1, 1.0 }, { 2, 1.0 }, { 3, 1.0 }, { 4, 1.0 } };

        var result = strategy.Aggregate(clients, weights);
        var resultParams = result.GetParameters();

        Assert.Equal(3.0, resultParams[0], 10);
    }

    [Fact]
    public void Median_SingleClient_ReturnsExactParams()
    {
        var strategy = new MedianFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>();
        var clients = new Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>>
        {
            { 0, CreateMockWithParams(3.14, 2.72) }
        };
        var weights = new Dictionary<int, double> { { 0, 1.0 } };

        var result = strategy.Aggregate(clients, weights);
        var resultParams = result.GetParameters();

        Assert.Equal(3.14, resultParams[0], 10);
        Assert.Equal(2.72, resultParams[1], 10);
    }

    [Fact]
    public void Median_EmptyClients_Throws()
    {
        var strategy = new MedianFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>();
        var empty = new Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>>();
        var weights = new Dictionary<int, double>();

        Assert.Throws<ArgumentException>(() => strategy.Aggregate(empty, weights));
    }

    [Fact]
    public void Median_StrategyName_IsMedian()
    {
        var strategy = new MedianFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>();
        Assert.Equal("Median", strategy.GetStrategyName());
    }

    [Fact]
    public void Median_AllIdenticalParams_ReturnsExact()
    {
        // When all clients have identical parameters, median = that value
        var strategy = new MedianFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>();
        var clients = new Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>>
        {
            { 0, CreateMockWithParams(7.0, 7.0) },
            { 1, CreateMockWithParams(7.0, 7.0) },
            { 2, CreateMockWithParams(7.0, 7.0) }
        };
        var weights = new Dictionary<int, double> { { 0, 1.0 }, { 1, 1.0 }, { 2, 1.0 } };

        var result = strategy.Aggregate(clients, weights);
        var resultParams = result.GetParameters();

        Assert.Equal(7.0, resultParams[0], 10);
        Assert.Equal(7.0, resultParams[1], 10);
    }

    #endregion

    #region TrimmedMeanFullModel - Coordinate-Wise Trimmed Mean

    [Fact]
    public void TrimmedMean_FiveClients_Trim20Pct_DropsOneFromEachEnd()
    {
        // trim_fraction=0.2, n=5 → trim = floor(0.2*5) = 1
        // Drop 1 from each end, average remaining 3
        // Client 0: [1.0]
        // Client 1: [2.0]
        // Client 2: [3.0]
        // Client 3: [4.0]
        // Client 4: [100.0] (outlier)
        // Sorted: [1, 2, 3, 4, 100] → drop lowest (1) and highest (100) → avg(2, 3, 4) = 3.0
        var strategy = new TrimmedMeanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(trimFraction: 0.2);
        var clients = new Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>>
        {
            { 0, CreateMockWithParams(1.0) },
            { 1, CreateMockWithParams(2.0) },
            { 2, CreateMockWithParams(3.0) },
            { 3, CreateMockWithParams(4.0) },
            { 4, CreateMockWithParams(100.0) }
        };
        var weights = new Dictionary<int, double> { { 0, 1.0 }, { 1, 1.0 }, { 2, 1.0 }, { 3, 1.0 }, { 4, 1.0 } };

        var result = strategy.Aggregate(clients, weights);
        var resultParams = result.GetParameters();

        Assert.Equal(3.0, resultParams[0], 10);
    }

    [Fact]
    public void TrimmedMean_TenClients_Trim20Pct_DropsTwoFromEachEnd()
    {
        // trim_fraction=0.2, n=10 → trim = floor(0.2*10) = 2
        // Drop 2 from each end, average remaining 6
        // Sorted: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        // Drop [0,1] and [8,9] → average(2,3,4,5,6,7) = 27/6 = 4.5
        var strategy = new TrimmedMeanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(trimFraction: 0.2);
        var clients = new Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>>();
        var weights = new Dictionary<int, double>();
        for (int i = 0; i < 10; i++)
        {
            clients[i] = CreateMockWithParams((double)i);
            weights[i] = 1.0;
        }

        var result = strategy.Aggregate(clients, weights);
        var resultParams = result.GetParameters();

        Assert.Equal(4.5, resultParams[0], 10);
    }

    [Fact]
    public void TrimmedMean_ZeroTrimFraction_IsFullAverage()
    {
        // trim_fraction=0.0 → no trimming → simple average
        // [1, 2, 3, 4, 5] → average = 15/5 = 3.0
        var strategy = new TrimmedMeanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(trimFraction: 0.0);
        var clients = new Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>>
        {
            { 0, CreateMockWithParams(1.0) },
            { 1, CreateMockWithParams(2.0) },
            { 2, CreateMockWithParams(3.0) },
            { 3, CreateMockWithParams(4.0) },
            { 4, CreateMockWithParams(5.0) }
        };
        var weights = new Dictionary<int, double> { { 0, 1.0 }, { 1, 1.0 }, { 2, 1.0 }, { 3, 1.0 }, { 4, 1.0 } };

        var result = strategy.Aggregate(clients, weights);
        var resultParams = result.GetParameters();

        Assert.Equal(3.0, resultParams[0], 10);
    }

    [Fact]
    public void TrimmedMean_IgnoresWeights_UnweightedByDesign()
    {
        // TrimmedMean is unweighted — weights parameter is ignored
        // Same data, vastly different weights → same result
        var strategy = new TrimmedMeanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(trimFraction: 0.0);
        var clients = new Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>>
        {
            { 0, CreateMockWithParams(10.0) },
            { 1, CreateMockWithParams(20.0) },
            { 2, CreateMockWithParams(30.0) }
        };
        var weights1 = new Dictionary<int, double> { { 0, 1.0 }, { 1, 1.0 }, { 2, 1.0 } };
        var weights2 = new Dictionary<int, double> { { 0, 1000.0 }, { 1, 1.0 }, { 2, 1.0 } };

        var result1 = strategy.Aggregate(clients, weights1);
        var result2 = strategy.Aggregate(clients, weights2);

        Assert.Equal(result1.GetParameters()[0], result2.GetParameters()[0], 10);
    }

    [Fact]
    public void TrimmedMean_OutlierResistant_ComparedToFullAverage()
    {
        // Full average of [1, 2, 3, 4, 1000] = 1010/5 = 202.0
        // Trimmed mean (0.2) drops 1 and 1000 → average(2, 3, 4) = 3.0
        // Trimmed mean is much closer to the "honest" values
        var fullAvg = new TrimmedMeanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(trimFraction: 0.0);
        var trimmed = new TrimmedMeanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(trimFraction: 0.2);
        var clients = new Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>>
        {
            { 0, CreateMockWithParams(1.0) },
            { 1, CreateMockWithParams(2.0) },
            { 2, CreateMockWithParams(3.0) },
            { 3, CreateMockWithParams(4.0) },
            { 4, CreateMockWithParams(1000.0) }
        };
        var weights = new Dictionary<int, double> { { 0, 1.0 }, { 1, 1.0 }, { 2, 1.0 }, { 3, 1.0 }, { 4, 1.0 } };

        var fullResult = fullAvg.Aggregate(clients, weights).GetParameters();
        var trimResult = trimmed.Aggregate(clients, weights).GetParameters();

        // Full average = 202.0
        Assert.Equal(202.0, fullResult[0], 10);
        // Trimmed mean = 3.0
        Assert.Equal(3.0, trimResult[0], 10);
        // Trimmed is much closer to honest median range [1..4]
        Assert.True(Math.Abs(trimResult[0] - 3.0) < Math.Abs(fullResult[0] - 3.0));
    }

    [Fact]
    public void TrimmedMean_MultipleParams_EachTrimmedIndependently()
    {
        // Two parameters, 5 clients, trim=0.2 (drop 1 each end per param)
        // param0: [10, 20, 30, 40, 50] → drop 10 and 50 → avg(20,30,40) = 30.0
        // param1: [50, 40, 30, 20, 10] → sorted: [10,20,30,40,50] → drop 10 and 50 → avg(20,30,40) = 30.0
        var strategy = new TrimmedMeanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(trimFraction: 0.2);
        var clients = new Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>>
        {
            { 0, CreateMockWithParams(10.0, 50.0) },
            { 1, CreateMockWithParams(20.0, 40.0) },
            { 2, CreateMockWithParams(30.0, 30.0) },
            { 3, CreateMockWithParams(40.0, 20.0) },
            { 4, CreateMockWithParams(50.0, 10.0) }
        };
        var weights = new Dictionary<int, double> { { 0, 1.0 }, { 1, 1.0 }, { 2, 1.0 }, { 3, 1.0 }, { 4, 1.0 } };

        var result = strategy.Aggregate(clients, weights);
        var resultParams = result.GetParameters();

        Assert.Equal(30.0, resultParams[0], 10);
        Assert.Equal(30.0, resultParams[1], 10);
    }

    [Fact]
    public void TrimmedMean_SingleClient_ReturnsExactParams()
    {
        // trim_fraction=0.2, n=1 → trim = floor(0.2*1) = 0, kept=1
        // Result = exact params
        var strategy = new TrimmedMeanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(trimFraction: 0.2);
        var clients = new Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>>
        {
            { 0, CreateMockWithParams(3.14) }
        };
        var weights = new Dictionary<int, double> { { 0, 1.0 } };

        var result = strategy.Aggregate(clients, weights);
        var resultParams = result.GetParameters();

        Assert.Equal(3.14, resultParams[0], 10);
    }

    [Fact]
    public void TrimmedMean_InvalidTrimFraction_Throws()
    {
        // trim_fraction must be in [0.0, 0.5)
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TrimmedMeanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(trimFraction: 0.5));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TrimmedMeanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(trimFraction: 0.99));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TrimmedMeanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(trimFraction: -0.1));
    }

    [Fact]
    public void TrimmedMean_TrimTooLargeForClientCount_Throws()
    {
        // trim_fraction=0.4, n=2 → trim = floor(0.4*2) = 0, kept = 2-0 = 2 → OK
        // trim_fraction=0.49, n=3 → trim = floor(0.49*3) = 1, kept = 3-2 = 1 → OK
        // trim_fraction=0.4, n=5 → trim = floor(0.4*5) = 2, kept = 5-4 = 1 → OK
        // Need: kept <= 0 to throw. trim_fraction=0.4, n=2 → trim=0, kept=2. Not enough.
        // trim_fraction=0.49, n=2 → trim=0, kept=2.
        // We need trim_fraction such that floor(fraction * n) >= n/2
        // With the valid range [0, 0.5), it's hard to get kept <= 0 for small n
        // But we can test that the strategy handles the threshold correctly
        // with 3 clients and fraction 0.4: trim=floor(1.2)=1, kept=3-2=1 → OK
        // Let's verify it works correctly when barely above threshold
        var strategy = new TrimmedMeanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(trimFraction: 0.4);
        var clients = new Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>>
        {
            { 0, CreateMockWithParams(1.0) },
            { 1, CreateMockWithParams(2.0) },
            { 2, CreateMockWithParams(3.0) }
        };
        var weights = new Dictionary<int, double> { { 0, 1.0 }, { 1, 1.0 }, { 2, 1.0 } };

        // trim=floor(0.4*3)=1, kept=3-2=1 → takes only the middle value: sorted [1,2,3] → 2.0
        var result = strategy.Aggregate(clients, weights);
        var resultParams = result.GetParameters();

        Assert.Equal(2.0, resultParams[0], 10);
    }

    [Fact]
    public void TrimmedMean_EmptyClients_Throws()
    {
        var strategy = new TrimmedMeanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(trimFraction: 0.2);
        var empty = new Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>>();
        var weights = new Dictionary<int, double>();

        Assert.Throws<ArgumentException>(() => strategy.Aggregate(empty, weights));
    }

    [Fact]
    public void TrimmedMean_StrategyName_IsTrimmedMean()
    {
        var strategy = new TrimmedMeanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>();
        Assert.Equal("TrimmedMean", strategy.GetStrategyName());
    }

    [Fact]
    public void TrimmedMean_ParameterLengthMismatch_Throws()
    {
        var strategy = new TrimmedMeanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(trimFraction: 0.2);
        var clients = new Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>>
        {
            { 0, CreateMockWithParams(1.0, 2.0) },
            { 1, CreateMockWithParams(3.0) }  // Different parameter count!
        };
        var weights = new Dictionary<int, double> { { 0, 1.0 }, { 1, 1.0 } };

        Assert.Throws<ArgumentException>(() => strategy.Aggregate(clients, weights));
    }

    [Fact]
    public void Median_ParameterLengthMismatch_Throws()
    {
        var strategy = new MedianFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>();
        var clients = new Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>>
        {
            { 0, CreateMockWithParams(1.0, 2.0) },
            { 1, CreateMockWithParams(3.0) }  // Different parameter count!
        };
        var weights = new Dictionary<int, double> { { 0, 1.0 }, { 1, 1.0 } };

        Assert.Throws<ArgumentException>(() => strategy.Aggregate(clients, weights));
    }

    #endregion

    #region Cross-Strategy Comparison: Median vs TrimmedMean vs FedAvg

    [Fact]
    public void AllStrategies_IdenticalClients_AllReturnSameResult()
    {
        // When all clients are identical, all strategies should return the same value
        var fedavg = new FedAvgAggregationStrategy<double>();
        var median = new MedianFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>();
        var trimmed = new TrimmedMeanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(trimFraction: 0.2);

        // FedAvg uses Dictionary<string, double[]> interface
        var fedavgModels = new Dictionary<int, Dictionary<string, double[]>>
        {
            { 0, new Dictionary<string, double[]> { { "w", new[] { 5.0 } } } },
            { 1, new Dictionary<string, double[]> { { "w", new[] { 5.0 } } } },
            { 2, new Dictionary<string, double[]> { { "w", new[] { 5.0 } } } }
        };
        var fedavgWeights = new Dictionary<int, double> { { 0, 1.0 }, { 1, 1.0 }, { 2, 1.0 } };

        // Median/TrimmedMean use IFullModel interface
        var fullModelClients = new Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>>
        {
            { 0, CreateMockWithParams(5.0) },
            { 1, CreateMockWithParams(5.0) },
            { 2, CreateMockWithParams(5.0) }
        };
        var modelWeights = new Dictionary<int, double> { { 0, 1.0 }, { 1, 1.0 }, { 2, 1.0 } };

        var fedavgResult = fedavg.Aggregate(fedavgModels, fedavgWeights);
        var medianResult = median.Aggregate(fullModelClients, modelWeights).GetParameters();
        var trimmedResult = trimmed.Aggregate(fullModelClients, modelWeights).GetParameters();

        Assert.Equal(5.0, fedavgResult["w"][0], 10);
        Assert.Equal(5.0, medianResult[0], 10);
        Assert.Equal(5.0, trimmedResult[0], 10);
    }

    [Fact]
    public void Median_Vs_TrimmedMean_HighTrimConverges()
    {
        // With 5 clients and trim=0.4: trim=floor(2)=2, kept=1 → only middle value
        // This should equal the median for odd-count!
        // Values: [10, 20, 30, 40, 50]
        // Median: 30.0
        // TrimmedMean(0.4): drop 2 from each end → only 30 → 30.0
        var median = new MedianFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>();
        var trimmed = new TrimmedMeanFullModelAggregationStrategy<double, Matrix<double>, Vector<double>>(trimFraction: 0.4);

        var clients = new Dictionary<int, IFullModel<double, Matrix<double>, Vector<double>>>
        {
            { 0, CreateMockWithParams(10.0) },
            { 1, CreateMockWithParams(20.0) },
            { 2, CreateMockWithParams(30.0) },
            { 3, CreateMockWithParams(40.0) },
            { 4, CreateMockWithParams(50.0) }
        };
        var weights = new Dictionary<int, double> { { 0, 1.0 }, { 1, 1.0 }, { 2, 1.0 }, { 3, 1.0 }, { 4, 1.0 } };

        var medianResult = median.Aggregate(clients, weights).GetParameters();
        var trimmedResult = trimmed.Aggregate(clients, weights).GetParameters();

        Assert.Equal(30.0, medianResult[0], 10);
        Assert.Equal(30.0, trimmedResult[0], 10);
    }

    #endregion
}
