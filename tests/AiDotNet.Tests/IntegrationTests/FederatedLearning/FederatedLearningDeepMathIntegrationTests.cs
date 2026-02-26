using AiDotNet.Autodiff;
using AiDotNet.FederatedLearning.Aggregators;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.FederatedLearning;

/// <summary>
/// Deep math integration tests for federated learning aggregation strategies.
/// Tests the full aggregation pipeline with hand-computed expected values.
///
/// Key formulas tested:
/// - FedAvg: θ_agg = Σ (w_i / Σw_j) * θ_i  (weighted average)
/// - Median: coordinate-wise median (sort + middle value)
/// - TrimmedMean: sort, drop extreme trim%, average remainder
/// - WinsorizedMean: sort, clip extremes to boundary, average all
/// - Krum: argmin_i Σ_{j in N(i)} ||θ_i - θ_j||² (sum of n-f-2 nearest)
/// - MultiKrum: select m clients with lowest Krum scores, average them
/// - Bulyan: MultiKrum selection + trimmed aggregation within selected set
/// - RFA (Geometric Median): Weiszfeld iterations minimizing Σ||θ - θ_i||
/// </summary>
public class FederatedLearningDeepMathIntegrationTests
{
    private const double Tol = 1e-10;

    #region FedAvg Tests

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void FedAvg_EqualWeights_IsSimpleAverage()
    {
        // 3 clients, equal weights, 2 parameters each
        // θ₁ = [1, 4], θ₂ = [2, 5], θ₃ = [3, 6], w = [1,1,1]
        // Expected: [(1+2+3)/3, (4+5+6)/3] = [2, 5]
        var agg = new FedAvgFullModelAggregationStrategy<double, double[], double[]>();
        var models = MakeModels(
            (0, new[] { 1.0, 4.0 }),
            (1, new[] { 2.0, 5.0 }),
            (2, new[] { 3.0, 6.0 }));
        var weights = new Dictionary<int, double> { [0] = 1.0, [1] = 1.0, [2] = 1.0 };

        var result = agg.Aggregate(models, weights);
        var p = result.GetParameters();

        Assert.Equal(2.0, p[0], Tol);
        Assert.Equal(5.0, p[1], Tol);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void FedAvg_UnequalWeights_HandCalculated()
    {
        // θ₁ = [10, 20], θ₂ = [30, 40], weights = [100, 300]
        // normalized: w₁ = 100/400 = 0.25, w₂ = 300/400 = 0.75
        // Expected: [0.25*10 + 0.75*30, 0.25*20 + 0.75*40] = [25, 35]
        var agg = new FedAvgFullModelAggregationStrategy<double, double[], double[]>();
        var models = MakeModels(
            (0, new[] { 10.0, 20.0 }),
            (1, new[] { 30.0, 40.0 }));
        var weights = new Dictionary<int, double> { [0] = 100.0, [1] = 300.0 };

        var result = agg.Aggregate(models, weights);
        var p = result.GetParameters();

        Assert.Equal(25.0, p[0], Tol);
        Assert.Equal(35.0, p[1], Tol);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void FedAvg_SingleClient_ReturnsSameParameters()
    {
        var agg = new FedAvgFullModelAggregationStrategy<double, double[], double[]>();
        var models = MakeModels((0, new[] { 7.0, 8.0, 9.0 }));
        var weights = new Dictionary<int, double> { [0] = 1.0 };

        var result = agg.Aggregate(models, weights);
        var p = result.GetParameters();

        Assert.Equal(7.0, p[0], Tol);
        Assert.Equal(8.0, p[1], Tol);
        Assert.Equal(9.0, p[2], Tol);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void FedAvg_ThreeClients_WeightedAverage()
    {
        // θ₁=[1], θ₂=[2], θ₃=[6], w=[1,2,1]
        // total = 4, normalized: 0.25, 0.5, 0.25
        // Expected: 0.25*1 + 0.5*2 + 0.25*6 = 0.25 + 1.0 + 1.5 = 2.75
        var agg = new FedAvgFullModelAggregationStrategy<double, double[], double[]>();
        var models = MakeModels(
            (0, new[] { 1.0 }),
            (1, new[] { 2.0 }),
            (2, new[] { 6.0 }));
        var weights = new Dictionary<int, double> { [0] = 1.0, [1] = 2.0, [2] = 1.0 };

        var result = agg.Aggregate(models, weights);
        var p = result.GetParameters();

        Assert.Equal(2.75, p[0], Tol);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void FedAvg_EmptyModels_Throws()
    {
        var agg = new FedAvgFullModelAggregationStrategy<double, double[], double[]>();
        var models = new Dictionary<int, IFullModel<double, double[], double[]>>();
        var weights = new Dictionary<int, double>();

        Assert.Throws<ArgumentException>(() => agg.Aggregate(models, weights));
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void FedAvg_StrategyName_IsFedAvg()
    {
        var agg = new FedAvgFullModelAggregationStrategy<double, double[], double[]>();
        Assert.Equal("FedAvg", agg.GetStrategyName());
    }

    #endregion

    #region Median Tests

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void Median_OddClients_HandCalculated()
    {
        // 3 clients: θ₁=[1,10], θ₂=[5,20], θ₃=[3,30]
        // Sorted per coordinate:
        //   param0: [1,3,5] → median = 3
        //   param1: [10,20,30] → median = 20
        var agg = new MedianFullModelAggregationStrategy<double, double[], double[]>();
        var models = MakeModels(
            (0, new[] { 1.0, 10.0 }),
            (1, new[] { 5.0, 20.0 }),
            (2, new[] { 3.0, 30.0 }));
        var weights = new Dictionary<int, double> { [0] = 1.0, [1] = 1.0, [2] = 1.0 };

        var result = agg.Aggregate(models, weights);
        var p = result.GetParameters();

        Assert.Equal(3.0, p[0], Tol);
        Assert.Equal(20.0, p[1], Tol);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void Median_EvenClients_AverageOfMiddleTwo()
    {
        // 4 clients: θ₁=[10], θ₂=[20], θ₃=[30], θ₄=[40]
        // Sorted: [10,20,30,40] → median = (20+30)/2 = 25
        var agg = new MedianFullModelAggregationStrategy<double, double[], double[]>();
        var models = MakeModels(
            (0, new[] { 10.0 }),
            (1, new[] { 20.0 }),
            (2, new[] { 30.0 }),
            (3, new[] { 40.0 }));
        var weights = new Dictionary<int, double> { [0] = 1.0, [1] = 1.0, [2] = 1.0, [3] = 1.0 };

        var result = agg.Aggregate(models, weights);
        var p = result.GetParameters();

        Assert.Equal(25.0, p[0], Tol);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void Median_OutlierResistant_HandCalculated()
    {
        // 5 clients: θ₁=[1], θ₂=[2], θ₃=[3], θ₄=[4], θ₅=[1000]
        // Sorted: [1,2,3,4,1000] → median = 3 (outlier doesn't affect)
        var agg = new MedianFullModelAggregationStrategy<double, double[], double[]>();
        var models = MakeModels(
            (0, new[] { 1.0 }),
            (1, new[] { 2.0 }),
            (2, new[] { 3.0 }),
            (3, new[] { 4.0 }),
            (4, new[] { 1000.0 }));
        var weights = new Dictionary<int, double> { [0] = 1.0, [1] = 1.0, [2] = 1.0, [3] = 1.0, [4] = 1.0 };

        var result = agg.Aggregate(models, weights);
        var p = result.GetParameters();

        Assert.Equal(3.0, p[0], Tol);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void Median_StrategyName_IsMedian()
    {
        var agg = new MedianFullModelAggregationStrategy<double, double[], double[]>();
        Assert.Equal("Median", agg.GetStrategyName());
    }

    #endregion

    #region TrimmedMean Tests

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void TrimmedMean_20Pct_FiveClients_HandCalculated()
    {
        // 5 clients, trim=20%: floor(0.2*5) = 1 from each side
        // θ values for param0: [1, 2, 3, 4, 100]
        // Sorted: [1, 2, 3, 4, 100], trim 1 each side → [2, 3, 4]
        // Average of kept: (2+3+4)/3 = 3.0
        var agg = new TrimmedMeanFullModelAggregationStrategy<double, double[], double[]>(trimFraction: 0.2);
        var models = MakeModels(
            (0, new[] { 1.0 }),
            (1, new[] { 2.0 }),
            (2, new[] { 3.0 }),
            (3, new[] { 4.0 }),
            (4, new[] { 100.0 }));
        var weights = new Dictionary<int, double> { [0] = 1.0, [1] = 1.0, [2] = 1.0, [3] = 1.0, [4] = 1.0 };

        var result = agg.Aggregate(models, weights);
        var p = result.GetParameters();

        Assert.Equal(3.0, p[0], Tol);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void TrimmedMean_ZeroTrim_IsSameAsSimpleAverage()
    {
        // trim=0%: no trimming → simple average
        // θ: [10, 20, 30], average = 20
        var agg = new TrimmedMeanFullModelAggregationStrategy<double, double[], double[]>(trimFraction: 0.0);
        var models = MakeModels(
            (0, new[] { 10.0 }),
            (1, new[] { 20.0 }),
            (2, new[] { 30.0 }));
        var weights = new Dictionary<int, double> { [0] = 1.0, [1] = 1.0, [2] = 1.0 };

        var result = agg.Aggregate(models, weights);
        var p = result.GetParameters();

        Assert.Equal(20.0, p[0], Tol);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void TrimmedMean_MultipleParams_HandCalculated()
    {
        // 5 clients, 2 params, trim=20% (trim 1 each side)
        // param0: [10, -5, 3, 7, 100] → sorted [-5, 3, 7, 10, 100] → kept [3, 7, 10] → avg = 20/3
        // param1: [1, 2, 3, 4, 5] → sorted [1, 2, 3, 4, 5] → kept [2, 3, 4] → avg = 3
        var agg = new TrimmedMeanFullModelAggregationStrategy<double, double[], double[]>(trimFraction: 0.2);
        var models = MakeModels(
            (0, new[] { 10.0, 1.0 }),
            (1, new[] { -5.0, 2.0 }),
            (2, new[] { 3.0, 3.0 }),
            (3, new[] { 7.0, 4.0 }),
            (4, new[] { 100.0, 5.0 }));
        var weights = new Dictionary<int, double> { [0] = 1.0, [1] = 1.0, [2] = 1.0, [3] = 1.0, [4] = 1.0 };

        var result = agg.Aggregate(models, weights);
        var p = result.GetParameters();

        Assert.Equal(20.0 / 3.0, p[0], Tol);
        Assert.Equal(3.0, p[1], Tol);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void TrimmedMean_InvalidFraction_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TrimmedMeanFullModelAggregationStrategy<double, double[], double[]>(trimFraction: 0.5));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TrimmedMeanFullModelAggregationStrategy<double, double[], double[]>(trimFraction: -0.1));
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void TrimmedMean_StrategyName_IsTrimmedMean()
    {
        var agg = new TrimmedMeanFullModelAggregationStrategy<double, double[], double[]>();
        Assert.Equal("TrimmedMean", agg.GetStrategyName());
    }

    #endregion

    #region WinsorizedMean Tests

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void WinsorizedMean_20Pct_FiveClients_HandCalculated()
    {
        // 5 clients, winsorize=20%: w=floor(0.2*5)=1
        // lowerIndex = min(max(0,1), max(0,4)) = 1
        // upperIndex = max(0, min(4-1, 4)) = 3
        // param0 values: [1, 2, 3, 4, 100]
        // Sorted: [1, 2, 3, 4, 100]
        // lower = buffer[1] = 2, upper = buffer[3] = 4
        // Clipped: [2, 2, 3, 4, 4]  (1→2, 100→4)
        // Average: (2+2+3+4+4)/5 = 15/5 = 3.0
        var agg = new WinsorizedMeanFullModelAggregationStrategy<double, double[], double[]>(winsorizeFraction: 0.2);
        var models = MakeModels(
            (0, new[] { 1.0 }),
            (1, new[] { 2.0 }),
            (2, new[] { 3.0 }),
            (3, new[] { 4.0 }),
            (4, new[] { 100.0 }));
        var weights = new Dictionary<int, double> { [0] = 1.0, [1] = 1.0, [2] = 1.0, [3] = 1.0, [4] = 1.0 };

        var result = agg.Aggregate(models, weights);
        var p = result.GetParameters();

        Assert.Equal(3.0, p[0], Tol);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void WinsorizedMean_ZeroFraction_IsSameAsSimpleAverage()
    {
        // winsorize=0%: no clipping → simple average
        // θ: [10, 20, 30] → average = 20
        var agg = new WinsorizedMeanFullModelAggregationStrategy<double, double[], double[]>(winsorizeFraction: 0.0);
        var models = MakeModels(
            (0, new[] { 10.0 }),
            (1, new[] { 20.0 }),
            (2, new[] { 30.0 }));
        var weights = new Dictionary<int, double> { [0] = 1.0, [1] = 1.0, [2] = 1.0 };

        var result = agg.Aggregate(models, weights);
        var p = result.GetParameters();

        Assert.Equal(20.0, p[0], Tol);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void WinsorizedMean_InvalidFraction_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new WinsorizedMeanFullModelAggregationStrategy<double, double[], double[]>(winsorizeFraction: 0.5));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new WinsorizedMeanFullModelAggregationStrategy<double, double[], double[]>(winsorizeFraction: -0.1));
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void WinsorizedMean_StrategyName_IsWinsorizedMean()
    {
        var agg = new WinsorizedMeanFullModelAggregationStrategy<double, double[], double[]>();
        Assert.Equal("WinsorizedMean", agg.GetStrategyName());
    }

    #endregion

    #region Krum Tests

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void Krum_SelectsMostCentralClient_HandCalculated()
    {
        // 4 clients, f=1: neighborsToSum = 4-1-2 = 1
        // θ₀=[0,0], θ₁=[1,0], θ₂=[0,1], θ₃=[100,100]
        //
        // Squared L2 distances (pairwise):
        // d(0,1)= 1, d(0,2)= 1, d(0,3)= 20000
        // d(1,0)= 1, d(1,2)= 2, d(1,3)= 19802 (99²+100²)
        // d(2,0)= 1, d(2,1)= 2, d(2,3)= 19802 (100²+99²)
        // d(3,0)= 20000, d(3,1)= 19802, d(3,2)= 19802
        //
        // Krum score (sum of 1 nearest):
        // client0: min(1, 1, 20000) = 1
        // client1: min(1, 2, 19802) = 1
        // client2: min(1, 2, 19802) = 1
        // client3: min(20000, 19802, 19802) = 19802
        //
        // client0 wins (score=1, lowest clientId tiebreaker)
        var agg = new KrumFullModelAggregationStrategy<double, double[], double[]>(byzantineClientCount: 1);
        var models = MakeModels(
            (0, new[] { 0.0, 0.0 }),
            (1, new[] { 1.0, 0.0 }),
            (2, new[] { 0.0, 1.0 }),
            (3, new[] { 100.0, 100.0 }));
        var weights = new Dictionary<int, double> { [0] = 1.0, [1] = 1.0, [2] = 1.0, [3] = 1.0 };

        var result = agg.Aggregate(models, weights);
        var p = result.GetParameters();

        // Should select client0 (or client1 or client2 - all have score 1)
        // Implementation sorts by score then by clientId, so client0 wins
        Assert.Equal(0.0, p[0], Tol);
        Assert.Equal(0.0, p[1], Tol);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void Krum_TooFewClients_Throws()
    {
        // n=3, f=1 → neighborsToSum = 3-1-2 = 0 → throws
        var agg = new KrumFullModelAggregationStrategy<double, double[], double[]>(byzantineClientCount: 1);
        var models = MakeModels(
            (0, new[] { 1.0 }),
            (1, new[] { 2.0 }),
            (2, new[] { 3.0 }));
        var weights = new Dictionary<int, double> { [0] = 1.0, [1] = 1.0, [2] = 1.0 };

        Assert.Throws<InvalidOperationException>(() => agg.Aggregate(models, weights));
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void Krum_NegativeByzantine_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new KrumFullModelAggregationStrategy<double, double[], double[]>(byzantineClientCount: -1));
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void Krum_StrategyName_IsKrum()
    {
        var agg = new KrumFullModelAggregationStrategy<double, double[], double[]>();
        Assert.Equal("Krum", agg.GetStrategyName());
    }

    #endregion

    #region MultiKrum Tests

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void MultiKrum_SelectsTwoBestAndAverages_HandCalculated()
    {
        // 5 clients, f=1, selectionCount=2
        // neighborsToSum = 5-1-2 = 2
        // θ₀=[0], θ₁=[1], θ₂=[2], θ₃=[3], θ₄=[100]
        //
        // Squared distances (1D):
        // d(0,1)=1, d(0,2)=4, d(0,3)=9, d(0,4)=10000
        // d(1,0)=1, d(1,2)=1, d(1,3)=4, d(1,4)=9801
        // d(2,0)=4, d(2,1)=1, d(2,3)=1, d(2,4)=9604
        // d(3,0)=9, d(3,1)=4, d(3,2)=1, d(3,4)=9409
        // d(4,0)=10000, d(4,1)=9801, d(4,2)=9604, d(4,3)=9409
        //
        // Krum scores (sum of 2 nearest):
        // client0: sorted [1,4,9,10000] → 1+4 = 5
        // client1: sorted [1,1,4,9801] → 1+1 = 2
        // client2: sorted [1,1,4,9604] → 1+1 = 2
        // client3: sorted [1,4,9,9409] → 1+4 = 5
        // client4: sorted [9409,9604,9801,10000] → 9409+9604 = 19013
        //
        // Select top 2 by score (ascending): client1(2), client2(2)
        // Average: (1+2)/2 = 1.5
        var agg = new MultiKrumFullModelAggregationStrategy<double, double[], double[]>(
            byzantineClientCount: 1, selectionCount: 2);
        var models = MakeModels(
            (0, new[] { 0.0 }),
            (1, new[] { 1.0 }),
            (2, new[] { 2.0 }),
            (3, new[] { 3.0 }),
            (4, new[] { 100.0 }));
        var weights = new Dictionary<int, double> { [0] = 1.0, [1] = 1.0, [2] = 1.0, [3] = 1.0, [4] = 1.0 };

        var result = agg.Aggregate(models, weights);
        var p = result.GetParameters();

        Assert.Equal(1.5, p[0], Tol);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void MultiKrum_StrategyName_IsMultiKrum()
    {
        var agg = new MultiKrumFullModelAggregationStrategy<double, double[], double[]>();
        Assert.Equal("MultiKrum", agg.GetStrategyName());
    }

    #endregion

    #region Bulyan Tests

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void Bulyan_RequiresMinimumClients_HandCalculated()
    {
        // Bulyan requires n >= 4f+3. For f=1: n >= 7
        // 7 clients, f=1
        // selectionSize = n - 2f = 7 - 2 = 5
        // trim = min(f, (m-1)/2) = min(1, 2) = 1
        // kept = 5 - 2*1 = 3
        //
        // θ₀=[0], θ₁=[1], θ₂=[2], θ₃=[3], θ₄=[4], θ₅=[5], θ₆=[100]
        //
        // First: MultiKrum selects 5 clients (excludes 2 with highest Krum scores)
        // The outlier (client6=100) will have highest score and be excluded
        //
        // After selection, trimmed aggregation on the 5 selected:
        // trim=1 from each side of sorted values
        var agg = new BulyanFullModelAggregationStrategy<double, double[], double[]>(byzantineClientCount: 1);
        var models = MakeModels(
            (0, new[] { 0.0 }),
            (1, new[] { 1.0 }),
            (2, new[] { 2.0 }),
            (3, new[] { 3.0 }),
            (4, new[] { 4.0 }),
            (5, new[] { 5.0 }),
            (6, new[] { 100.0 }));
        var weights = new Dictionary<int, double>
            { [0] = 1.0, [1] = 1.0, [2] = 1.0, [3] = 1.0, [4] = 1.0, [5] = 1.0, [6] = 1.0 };

        var result = agg.Aggregate(models, weights);
        var p = result.GetParameters();

        // The result should be close to the middle values (robust against outlier 100)
        // After MultiKrum selection of 5 (likely excluding client6 + another boundary),
        // then trimmed mean on those 5 (trim 1 each side, keep 3)
        // Exact result depends on MultiKrum selection, but should be reasonable
        Assert.True(p[0] >= 0.0 && p[0] <= 5.0,
            $"Bulyan result {p[0]} should be in [0, 5] (robust against outlier)");
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void Bulyan_TooFewClients_Throws()
    {
        // Bulyan requires n >= 4f+3. For f=1: n >= 7
        // 6 clients, f=1 → should throw
        var agg = new BulyanFullModelAggregationStrategy<double, double[], double[]>(byzantineClientCount: 1);
        var models = MakeModels(
            (0, new[] { 0.0 }),
            (1, new[] { 1.0 }),
            (2, new[] { 2.0 }),
            (3, new[] { 3.0 }),
            (4, new[] { 4.0 }),
            (5, new[] { 5.0 }));
        var weights = new Dictionary<int, double>
            { [0] = 1.0, [1] = 1.0, [2] = 1.0, [3] = 1.0, [4] = 1.0, [5] = 1.0 };

        Assert.Throws<InvalidOperationException>(() => agg.Aggregate(models, weights));
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void Bulyan_StrategyName_IsBulyan()
    {
        var agg = new BulyanFullModelAggregationStrategy<double, double[], double[]>();
        Assert.Equal("Bulyan", agg.GetStrategyName());
    }

    #endregion

    #region RFA (Geometric Median) Tests

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void RFA_IdenticalClients_ReturnsExactValue()
    {
        // All clients have same parameters → geometric median = that value
        var agg = new RfaFullModelAggregationStrategy<double, double[], double[]>();
        var models = MakeModels(
            (0, new[] { 5.0, 10.0 }),
            (1, new[] { 5.0, 10.0 }),
            (2, new[] { 5.0, 10.0 }));
        var weights = new Dictionary<int, double> { [0] = 1.0, [1] = 1.0, [2] = 1.0 };

        var result = agg.Aggregate(models, weights);
        var p = result.GetParameters();

        Assert.Equal(5.0, p[0], 1e-6);
        Assert.Equal(10.0, p[1], 1e-6);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void RFA_OutlierResistant_HandCalculated()
    {
        // 5 clients with one extreme outlier
        // θ₀=[1], θ₁=[2], θ₂=[3], θ₃=[4], θ₄=[1000]
        // Geometric median should be close to 3 (robust against outlier)
        // Mean would be ~202, but geometric median resists this
        var agg = new RfaFullModelAggregationStrategy<double, double[], double[]>(
            maxIterations: 100, tolerance: 1e-10);
        var models = MakeModels(
            (0, new[] { 1.0 }),
            (1, new[] { 2.0 }),
            (2, new[] { 3.0 }),
            (3, new[] { 4.0 }),
            (4, new[] { 1000.0 }));
        var weights = new Dictionary<int, double> { [0] = 1.0, [1] = 1.0, [2] = 1.0, [3] = 1.0, [4] = 1.0 };

        var result = agg.Aggregate(models, weights);
        var p = result.GetParameters();

        // Geometric median of [1,2,3,4,1000] in 1D is just the regular median = 3
        // (for 1D, geometric median equals median)
        Assert.Equal(3.0, p[0], 1e-4);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void RFA_TwoClients_ConvergesToMidpoint()
    {
        // 2 clients: geometric median of 2 points converges to midpoint
        // θ₀=[0, 0], θ₁=[10, 20]
        // Midpoint = [5, 10]
        var agg = new RfaFullModelAggregationStrategy<double, double[], double[]>(
            maxIterations: 100, tolerance: 1e-10);
        var models = MakeModels(
            (0, new[] { 0.0, 0.0 }),
            (1, new[] { 10.0, 20.0 }));
        var weights = new Dictionary<int, double> { [0] = 1.0, [1] = 1.0 };

        var result = agg.Aggregate(models, weights);
        var p = result.GetParameters();

        // With 2 points, Weiszfeld converges to midpoint (both have equal weight)
        Assert.Equal(5.0, p[0], 1e-4);
        Assert.Equal(10.0, p[1], 1e-4);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void RFA_StrategyName_IsRFA()
    {
        var agg = new RfaFullModelAggregationStrategy<double, double[], double[]>();
        Assert.Equal("RFA", agg.GetStrategyName());
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void RFA_InvalidParameters_Throw()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new RfaFullModelAggregationStrategy<double, double[], double[]>(maxIterations: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new RfaFullModelAggregationStrategy<double, double[], double[]>(tolerance: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new RfaFullModelAggregationStrategy<double, double[], double[]>(epsilon: 0));
    }

    #endregion

    #region FedProx Tests

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void FedProx_AggregationIsSameAsFedAvg()
    {
        // FedProx aggregation is identical to FedAvg (proximal term only affects local training)
        var fedProx = new FedProxFullModelAggregationStrategy<double, double[], double[]>(mu: 0.01);
        var fedAvg = new FedAvgFullModelAggregationStrategy<double, double[], double[]>();

        var models = MakeModels(
            (0, new[] { 1.0, 2.0 }),
            (1, new[] { 3.0, 4.0 }));
        var weights = new Dictionary<int, double> { [0] = 1.0, [1] = 3.0 };

        var proxResult = fedProx.Aggregate(models, weights).GetParameters();
        var avgResult = fedAvg.Aggregate(models, weights).GetParameters();

        Assert.Equal(avgResult[0], proxResult[0], Tol);
        Assert.Equal(avgResult[1], proxResult[1], Tol);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void FedProx_MuAccessible()
    {
        var fedProx = new FedProxFullModelAggregationStrategy<double, double[], double[]>(mu: 0.05);
        Assert.Equal(0.05, fedProx.GetMu(), Tol);
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void FedProx_NegativeMu_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new FedProxFullModelAggregationStrategy<double, double[], double[]>(mu: -0.1));
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void FedProx_StrategyName_IsFedProx()
    {
        var agg = new FedProxFullModelAggregationStrategy<double, double[], double[]>();
        Assert.Equal("FedProx", agg.GetStrategyName());
    }

    #endregion

    #region Cross-Strategy Comparison Tests

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void AllStrategies_IdenticalClients_ReturnSameResult()
    {
        // When all clients have identical parameters, all strategies should return the same
        var models = MakeModels(
            (0, new[] { 5.0, 10.0 }),
            (1, new[] { 5.0, 10.0 }),
            (2, new[] { 5.0, 10.0 }),
            (3, new[] { 5.0, 10.0 }),
            (4, new[] { 5.0, 10.0 }),
            (5, new[] { 5.0, 10.0 }),
            (6, new[] { 5.0, 10.0 }));
        var weights = new Dictionary<int, double>
            { [0] = 1.0, [1] = 1.0, [2] = 1.0, [3] = 1.0, [4] = 1.0, [5] = 1.0, [6] = 1.0 };

        var strategies = new (string Name, Func<IFullModel<double, double[], double[]>> Agg)[]
        {
            ("FedAvg", () => new FedAvgFullModelAggregationStrategy<double, double[], double[]>().Aggregate(models, weights)),
            ("Median", () => new MedianFullModelAggregationStrategy<double, double[], double[]>().Aggregate(models, weights)),
            ("TrimmedMean", () => new TrimmedMeanFullModelAggregationStrategy<double, double[], double[]>().Aggregate(models, weights)),
            ("WinsorizedMean", () => new WinsorizedMeanFullModelAggregationStrategy<double, double[], double[]>().Aggregate(models, weights)),
            ("Krum", () => new KrumFullModelAggregationStrategy<double, double[], double[]>().Aggregate(models, weights)),
            ("Bulyan", () => new BulyanFullModelAggregationStrategy<double, double[], double[]>().Aggregate(models, weights)),
            ("RFA", () => new RfaFullModelAggregationStrategy<double, double[], double[]>().Aggregate(models, weights)),
        };

        foreach (var (name, aggFunc) in strategies)
        {
            var p = aggFunc().GetParameters();
            Assert.Equal(5.0, p[0], 1e-4);
            Assert.Equal(10.0, p[1], 1e-4);
        }
    }

    [Fact]
    [Trait("Category", "IntegrationTest")]
    public void RobustStrategies_OutlierResistant_AllNearMedian()
    {
        // 7 clients with one outlier
        // θ: [1, 2, 3, 4, 5, 6, 1000]
        // Robust strategies should all produce results near the median (3-4 range)
        var models = MakeModels(
            (0, new[] { 1.0 }),
            (1, new[] { 2.0 }),
            (2, new[] { 3.0 }),
            (3, new[] { 4.0 }),
            (4, new[] { 5.0 }),
            (5, new[] { 6.0 }),
            (6, new[] { 1000.0 }));
        var weights = new Dictionary<int, double>
            { [0] = 1.0, [1] = 1.0, [2] = 1.0, [3] = 1.0, [4] = 1.0, [5] = 1.0, [6] = 1.0 };

        var median = new MedianFullModelAggregationStrategy<double, double[], double[]>().Aggregate(models, weights).GetParameters();
        var trimmed = new TrimmedMeanFullModelAggregationStrategy<double, double[], double[]>().Aggregate(models, weights).GetParameters();
        var winsorized = new WinsorizedMeanFullModelAggregationStrategy<double, double[], double[]>().Aggregate(models, weights).GetParameters();
        var rfa = new RfaFullModelAggregationStrategy<double, double[], double[]>(maxIterations: 100).Aggregate(models, weights).GetParameters();

        // Median of [1,2,3,4,5,6,1000] = 4
        Assert.Equal(4.0, median[0], Tol);

        // Trimmed (20%, trim 1 each side): [2,3,4,5,6] → (2+3+4+5+6)/5 = 4.0
        Assert.Equal(4.0, trimmed[0], Tol);

        // All robust results should be in [1, 6] range
        Assert.True(winsorized[0] >= 1.0 && winsorized[0] <= 6.0,
            $"Winsorized result {winsorized[0]} should be in [1, 6]");
        Assert.True(rfa[0] >= 1.0 && rfa[0] <= 7.0,
            $"RFA result {rfa[0]} should be in [1, 7]");

        // FedAvg would be skewed by outlier: (1+2+3+4+5+6+1000)/7 ≈ 145.9
        var fedAvg = new FedAvgFullModelAggregationStrategy<double, double[], double[]>().Aggregate(models, weights).GetParameters();
        Assert.True(fedAvg[0] > 100.0, "FedAvg should be heavily influenced by outlier");
    }

    #endregion

    #region Helper Methods and Mock Model

    private static Dictionary<int, IFullModel<double, double[], double[]>> MakeModels(
        params (int ClientId, double[] Parameters)[] clients)
    {
        var dict = new Dictionary<int, IFullModel<double, double[], double[]>>();
        foreach (var (clientId, parameters) in clients)
        {
            dict[clientId] = new FLMockModel(parameters);
        }
        return dict;
    }

    /// <summary>
    /// Minimal mock model for federated learning aggregation tests.
    /// Implements IFullModel with controllable parameters.
    /// </summary>
    private class FLMockModel : IFullModel<double, double[], double[]>
    {
        private double[] _parameters;
        private List<int> _activeFeatures;

        public FLMockModel(double[] parameters)
        {
            _parameters = (double[])parameters.Clone();
            _activeFeatures = Enumerable.Range(0, parameters.Length).ToList();
        }

        // IModel
        public void Train(double[] input, double[] expectedOutput) { }
        public double[] Predict(double[] input) => _parameters;
        public ModelMetadata<double> GetModelMetadata() => new();

        // IParameterizable
        public Vector<double> GetParameters() => new(_parameters);
        public void SetParameters(Vector<double> parameters)
        {
            for (int i = 0; i < _parameters.Length; i++)
                _parameters[i] = parameters[i];
        }
        public int ParameterCount => _parameters.Length;
        public IFullModel<double, double[], double[]> WithParameters(Vector<double> p)
        {
            var newParams = new double[p.Length];
            for (int i = 0; i < p.Length; i++) newParams[i] = p[i];
            return new FLMockModel(newParams);
        }

        // IFeatureAware & IFeatureImportance
        public IEnumerable<int> GetActiveFeatureIndices() => _activeFeatures;
        public void SetActiveFeatureIndices(IEnumerable<int> indices) => _activeFeatures = indices.ToList();
        public bool IsFeatureUsed(int featureIndex) => _activeFeatures.Contains(featureIndex);
        public Dictionary<string, double> GetFeatureImportance() => new();

        // ICloneable
        public IFullModel<double, double[], double[]> DeepCopy() => new FLMockModel((double[])_parameters.Clone());
        public IFullModel<double, double[], double[]> Clone() => DeepCopy();

        // IGradientComputable
        public Vector<double> ComputeGradients(double[] input, double[] target, ILossFunction<double>? lossFunction = null)
            => new(_parameters.Length);
        public void ApplyGradients(Vector<double> gradients, double learningRate) { }

        // IJitCompilable
        public ComputationNode<double> ExportComputationGraph(List<ComputationNode<double>> inputNodes)
            => throw new NotSupportedException();
        public bool SupportsJitCompilation => false;

        // IModelSerializer
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }

        // ICheckpointableModel
        public void SaveState(Stream stream) { }
        public void LoadState(Stream stream) { }

        // ILossFunction default
        public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();
    }

    #endregion
}
