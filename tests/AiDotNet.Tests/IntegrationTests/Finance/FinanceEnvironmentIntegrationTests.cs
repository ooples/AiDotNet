using System;
using System.Threading.Tasks;
using AiDotNet.Finance.Data;
using AiDotNet.Finance.Trading.Environments;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Finance;

public class FinanceEnvironmentIntegrationTests
{
    [Fact]
    public async Task FinancialDataLoader_Float_LoadsAndSplits()
    {
        var series = FinanceTestHelpers.CreateMarketSeries<float>(30);
        var loader = new FinancialDataLoader<float>(series, sequenceLength: 5, predictionHorizon: 2);

        await loader.LoadAsync();
        Assert.True(loader.TotalCount > 0);

        var batch = loader.GetNextBatch();
        Assert.NotNull(batch.Features);
        Assert.NotNull(batch.Labels);

        var splits = loader.Split();
        Assert.NotNull(splits.Train);
        Assert.NotNull(splits.Validation);
        Assert.NotNull(splits.Test);
    }

    [Fact]
    public async Task FinancialDataLoader_Double_LoadsAndSplits()
    {
        var series = FinanceTestHelpers.CreateMarketSeries<double>(30);
        var loader = new FinancialDataLoader<double>(series, sequenceLength: 5, predictionHorizon: 2);

        await loader.LoadAsync();
        Assert.True(loader.TotalCount > 0);

        var batch = loader.GetNextBatch();
        Assert.NotNull(batch.Features);
        Assert.NotNull(batch.Labels);

        var splits = loader.Split();
        Assert.NotNull(splits.Train);
        Assert.NotNull(splits.Validation);
        Assert.NotNull(splits.Test);
    }

    [Fact]
    public async Task FinancialDataLoader_Float_NormalizesAndPredictsReturns()
    {
        var series = FinanceTestHelpers.CreateMarketSeries<float>(40);
        var loader = new FinancialDataLoader<float>(
            series,
            sequenceLength: 6,
            predictionHorizon: 3,
            includeVolume: true,
            includeReturns: true,
            predictReturns: true,
            normalizeMinMax: true);

        await loader.LoadAsync();
        Assert.True(loader.TotalCount > 0);
        Assert.True(loader.FeatureCount > 0);
        Assert.True(loader.OutputDimension > 0);

        var batch = loader.GetNextBatch();
        Assert.NotNull(batch.Features);
        Assert.NotNull(batch.Labels);
    }

    [Fact]
    public void MarketDataProvider_CanSliceAndTensorize()
    {
        var series = FinanceTestHelpers.CreateMarketSeries<double>(10);
        var provider = new MarketDataProvider<double>();
        provider.AddRange(series);

        Assert.Equal(10, provider.Count);

        var start = series[2].Timestamp;
        var end = series[6].Timestamp;
        var range = provider.GetRange(start, end);
        Assert.True(range.Count > 0);

        var tensor = provider.ToTensor();
        Assert.Equal(10, tensor.Shape[0]);
        Assert.True(tensor.Shape[1] >= 4);

        var noVolumeTensor = provider.ToTensor(includeVolume: false);
        Assert.Equal(10, noVolumeTensor.Shape[0]);
        Assert.Equal(4, noVolumeTensor.Shape[1]);

        var window = provider.GetWindow(startIndex: 0, length: 3);
        Assert.Equal(3, window.Count);

        provider.Clear();
        Assert.Equal(0, provider.Count);
    }

    [Fact]
    public void FinancialPreprocessor_CreatesFeaturesAndNormalizes()
    {
        var series = FinanceTestHelpers.CreateMarketSeries<double>(25);
        var preprocessor = new FinancialPreprocessor<double>();

        var features = preprocessor.CreateFeatureTensor(series, includeVolume: true, includeReturns: true);
        Assert.Equal(series.Count, features.Shape[0]);
        Assert.Equal(preprocessor.GetFeatureCount(includeVolume: true, includeReturns: true), features.Shape[1]);

        var normalized = preprocessor.NormalizeMinMax(features, out var minMax);
        Assert.Equal(features.Shape[0], normalized.Shape[0]);
        Assert.Equal(features.Shape[1], normalized.Shape[1]);
        Assert.Equal(minMax.Min.Length, minMax.Max.Length);

        var zScore = preprocessor.NormalizeZScore(features, out var stats);
        Assert.Equal(features.Shape[0], zScore.Shape[0]);
        Assert.Equal(features.Shape[1], zScore.Shape[1]);
        Assert.Equal(stats.Mean.Length, stats.StdDev.Length);

        var (supervisedFeatures, supervisedTargets) = preprocessor.CreateSupervisedLearningTensors(
            series,
            sequenceLength: 4,
            predictionHorizon: 2,
            includeVolume: true,
            includeReturns: true,
            predictReturns: true);

        Assert.Equal(3, supervisedFeatures.Shape.Length);
        Assert.Equal(3, supervisedTargets.Shape.Length);
        Assert.True(supervisedFeatures.Shape[0] > 0);
        Assert.True(supervisedTargets.Shape[0] > 0);
    }

    [Fact]
    public void TradingEnvironments_Float_Smoke()
    {
        RunEnvironmentSmokeTest<float>();
    }

    [Fact]
    public void TradingEnvironments_Double_Smoke()
    {
        RunEnvironmentSmokeTest<double>();
    }

    [Fact]
    public void TradingEnvironment_Seed_ReproducibleRandomStart()
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        var data = FinanceTestHelpers.CreatePriceTensor<double>(steps: 50, assets: 1);
        var env = new StockTradingEnvironment<double>(
            data,
            windowSize: 5,
            initialCapital: numOps.FromDouble(1000),
            tradeSize: numOps.One,
            randomStart: true);

        env.Seed(123);
        var state1 = env.Reset();
        env.Seed(123);
        var state2 = env.Reset();

        Assert.Equal(state1.Length, state2.Length);
        for (int i = 0; i < state1.Length; i++)
        {
            Assert.Equal(state1[i], state2[i]);
        }

        env.Close();
    }

    [Fact]
    public async Task FinancialDataLoaderFactory_CreatesLoaderFromProvider()
    {
        var series = FinanceTestHelpers.CreateMarketSeries<float>(30);
        var provider = new MarketDataProvider<float>();
        provider.AddRange(series);

        var loader = FinancialDataLoaderFactory.FromProvider(provider, sequenceLength: 5, predictionHorizon: 2);

        await loader.LoadAsync();
        Assert.True(loader.TotalCount > 0);
    }

    [Fact]
    public void TradingEnvironmentFactory_CreatesPortfolioEnvironment()
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        var start = DateTime.UtcNow;
        var seriesA = FinanceTestHelpers.CreateMarketSeries<double>(20, start);
        var seriesB = FinanceTestHelpers.CreateMarketSeries<double>(20, start);
        var series = new System.Collections.Generic.List<System.Collections.Generic.IReadOnlyList<MarketDataPoint<double>>>
        {
            seriesA,
            seriesB
        };

        var env = TradingEnvironmentFactory.CreatePortfolioTradingEnvironment(
            series,
            windowSize: 5,
            initialCapital: numOps.FromDouble(1000));

        var state = env.Reset();
        Assert.Equal(env.ObservationSpaceDimension, state.Length);
    }

    private static void RunEnvironmentSmokeTest<T>()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var singleAssetData = FinanceTestHelpers.CreatePriceTensor<T>(steps: 40, assets: 1);
        var multiAssetData = FinanceTestHelpers.CreatePriceTensor<T>(steps: 40, assets: 2);

        var stockEnv = new StockTradingEnvironment<T>(
            singleAssetData,
            windowSize: 5,
            initialCapital: numOps.FromDouble(1000),
            tradeSize: numOps.One,
            randomStart: false);

        var state = stockEnv.Reset();
        Assert.Equal(stockEnv.ObservationSpaceDimension, state.Length);

        var action = new Vector<T>(new[] { numOps.FromDouble(1) });
        var step = stockEnv.Step(action);
        Assert.Equal(stockEnv.ObservationSpaceDimension, step.NextState.Length);

        var portfolioEnv = new PortfolioTradingEnvironment<T>(
            multiAssetData,
            windowSize: 5,
            initialCapital: numOps.FromDouble(1000),
            allowShortSelling: false);

        var portfolioState = portfolioEnv.Reset();
        Assert.Equal(portfolioEnv.ObservationSpaceDimension, portfolioState.Length);

        var weights = new Vector<T>(new[] { numOps.FromDouble(0.5), numOps.FromDouble(0.5) });
        var portfolioStep = portfolioEnv.Step(weights);
        Assert.Equal(portfolioEnv.ObservationSpaceDimension, portfolioStep.NextState.Length);

        var marketMakingEnv = new MarketMakingEnvironment<T>(
            singleAssetData,
            windowSize: 5,
            initialCapital: numOps.FromDouble(1000),
            tradeSize: numOps.One);

        var mmState = marketMakingEnv.Reset();
        Assert.Equal(marketMakingEnv.ObservationSpaceDimension, mmState.Length);

        var mmAction = new Vector<T>(new[] { numOps.FromDouble(0.01), numOps.FromDouble(0.01) });
        var mmStep = marketMakingEnv.Step(mmAction);
        Assert.Equal(marketMakingEnv.ObservationSpaceDimension, mmStep.NextState.Length);
    }

}
