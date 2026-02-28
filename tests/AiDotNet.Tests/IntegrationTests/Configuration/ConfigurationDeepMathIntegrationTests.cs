using AiDotNet.Configuration;
using AiDotNet.Enums;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Configuration;

/// <summary>
/// Deep integration tests for Configuration classes:
/// ResNetConfiguration (block counts, expansion, conv layer math),
/// EfficientNetConfiguration (compound scaling, multipliers, dropout),
/// RL configs (exploration schedule, prioritized replay, target network,
/// reward clipping, early stopping, episode/step metrics, training summary).
/// </summary>
public class ConfigurationDeepMathIntegrationTests
{
    // ============================
    // ResNetConfiguration Block Count Tests
    // ============================

    [Fact]
    public void ResNet18_BlockCounts_Are_2_2_2_2()
    {
        var config = new ResNetConfiguration(ResNetVariant.ResNet18, 1000);

        Assert.Equal(new[] { 2, 2, 2, 2 }, config.BlockCounts);
    }

    [Fact]
    public void ResNet34_BlockCounts_Are_3_4_6_3()
    {
        var config = new ResNetConfiguration(ResNetVariant.ResNet34, 1000);

        Assert.Equal(new[] { 3, 4, 6, 3 }, config.BlockCounts);
    }

    [Fact]
    public void ResNet50_BlockCounts_Are_3_4_6_3()
    {
        var config = new ResNetConfiguration(ResNetVariant.ResNet50, 1000);

        Assert.Equal(new[] { 3, 4, 6, 3 }, config.BlockCounts);
    }

    [Fact]
    public void ResNet101_BlockCounts_Are_3_4_23_3()
    {
        var config = new ResNetConfiguration(ResNetVariant.ResNet101, 1000);

        Assert.Equal(new[] { 3, 4, 23, 3 }, config.BlockCounts);
    }

    [Fact]
    public void ResNet152_BlockCounts_Are_3_8_36_3()
    {
        var config = new ResNetConfiguration(ResNetVariant.ResNet152, 1000);

        Assert.Equal(new[] { 3, 8, 36, 3 }, config.BlockCounts);
    }

    // ============================
    // ResNet Bottleneck/BasicBlock Tests
    // ============================

    [Fact]
    public void ResNet18_UsesBasicBlock()
    {
        var config = new ResNetConfiguration(ResNetVariant.ResNet18, 1000);

        Assert.False(config.UsesBottleneck);
        Assert.Equal(1, config.Expansion);
    }

    [Fact]
    public void ResNet34_UsesBasicBlock()
    {
        var config = new ResNetConfiguration(ResNetVariant.ResNet34, 1000);

        Assert.False(config.UsesBottleneck);
        Assert.Equal(1, config.Expansion);
    }

    [Fact]
    public void ResNet50_UsesBottleneck()
    {
        var config = new ResNetConfiguration(ResNetVariant.ResNet50, 1000);

        Assert.True(config.UsesBottleneck);
        Assert.Equal(4, config.Expansion);
    }

    [Fact]
    public void ResNet101_UsesBottleneck()
    {
        var config = new ResNetConfiguration(ResNetVariant.ResNet101, 1000);

        Assert.True(config.UsesBottleneck);
        Assert.Equal(4, config.Expansion);
    }

    [Fact]
    public void ResNet152_UsesBottleneck()
    {
        var config = new ResNetConfiguration(ResNetVariant.ResNet152, 1000);

        Assert.True(config.UsesBottleneck);
        Assert.Equal(4, config.Expansion);
    }

    // ============================
    // ResNet NumConvLayers Tests (hand-computed)
    // ============================

    [Fact]
    public void ResNet18_NumConvLayers_HandComputed()
    {
        // ResNet18: blocks = [2,2,2,2], sum = 8
        // BasicBlock has 2 convs per block
        // NumConvLayers = 1 (initial) + 8 * 2 = 17
        var config = new ResNetConfiguration(ResNetVariant.ResNet18, 1000);

        Assert.Equal(17, config.NumConvLayers);
    }

    [Fact]
    public void ResNet34_NumConvLayers_HandComputed()
    {
        // ResNet34: blocks = [3,4,6,3], sum = 16
        // BasicBlock has 2 convs per block
        // NumConvLayers = 1 + 16 * 2 = 33
        var config = new ResNetConfiguration(ResNetVariant.ResNet34, 1000);

        Assert.Equal(33, config.NumConvLayers);
    }

    [Fact]
    public void ResNet50_NumConvLayers_HandComputed()
    {
        // ResNet50: blocks = [3,4,6,3], sum = 16
        // BottleneckBlock has 3 convs per block
        // NumConvLayers = 1 + 16 * 3 = 49
        var config = new ResNetConfiguration(ResNetVariant.ResNet50, 1000);

        Assert.Equal(49, config.NumConvLayers);
    }

    [Fact]
    public void ResNet101_NumConvLayers_HandComputed()
    {
        // ResNet101: blocks = [3,4,23,3], sum = 33
        // BottleneckBlock has 3 convs per block
        // NumConvLayers = 1 + 33 * 3 = 100
        var config = new ResNetConfiguration(ResNetVariant.ResNet101, 1000);

        Assert.Equal(100, config.NumConvLayers);
    }

    [Fact]
    public void ResNet152_NumConvLayers_HandComputed()
    {
        // ResNet152: blocks = [3,8,36,3], sum = 50
        // BottleneckBlock has 3 convs per block
        // NumConvLayers = 1 + 50 * 3 = 151
        var config = new ResNetConfiguration(ResNetVariant.ResNet152, 1000);

        Assert.Equal(151, config.NumConvLayers);
    }

    // ============================
    // ResNet NumWeightLayers Tests
    // ============================

    [Fact]
    public void ResNet18_NumWeightLayers_Is18()
    {
        // NumWeightLayers = NumConvLayers + 1 (FC) = 17 + 1 = 18
        var config = new ResNetConfiguration(ResNetVariant.ResNet18, 1000);

        Assert.Equal(18, config.NumWeightLayers);
    }

    [Fact]
    public void ResNet34_NumWeightLayers_Is34()
    {
        // NumWeightLayers = 33 + 1 = 34
        var config = new ResNetConfiguration(ResNetVariant.ResNet34, 1000);

        Assert.Equal(34, config.NumWeightLayers);
    }

    [Fact]
    public void ResNet50_NumWeightLayers_Is50()
    {
        // NumWeightLayers = 49 + 1 = 50
        var config = new ResNetConfiguration(ResNetVariant.ResNet50, 1000);

        Assert.Equal(50, config.NumWeightLayers);
    }

    [Fact]
    public void ResNet101_NumWeightLayers_Is101()
    {
        // NumWeightLayers = 100 + 1 = 101
        var config = new ResNetConfiguration(ResNetVariant.ResNet101, 1000);

        Assert.Equal(101, config.NumWeightLayers);
    }

    [Fact]
    public void ResNet152_NumWeightLayers_Is152()
    {
        // NumWeightLayers = 151 + 1 = 152
        var config = new ResNetConfiguration(ResNetVariant.ResNet152, 1000);

        Assert.Equal(152, config.NumWeightLayers);
    }

    // ============================
    // ResNet Input Shape & Size Tests
    // ============================

    [Fact]
    public void ResNet_DefaultInputShape_Is_3_224_224()
    {
        var config = new ResNetConfiguration(ResNetVariant.ResNet50, 10);

        Assert.Equal(new[] { 3, 224, 224 }, config.InputShape);
    }

    [Fact]
    public void ResNet_TotalInputSize_HandComputed()
    {
        // 3 * 224 * 224 = 150,528
        var config = new ResNetConfiguration(ResNetVariant.ResNet50, 10);

        Assert.Equal(150528, config.TotalInputSize);
    }

    [Fact]
    public void ResNet_CIFARInputShape_Is_3_32_32()
    {
        var config = ResNetConfiguration.CreateForCIFAR(ResNetVariant.ResNet18, 10);

        Assert.Equal(new[] { 3, 32, 32 }, config.InputShape);
        Assert.Equal(3072, config.TotalInputSize); // 3 * 32 * 32 = 3072
    }

    [Fact]
    public void ResNet_GrayscaleInput_HasOneChannel()
    {
        var config = new ResNetConfiguration(ResNetVariant.ResNet18, 10, inputChannels: 1);

        Assert.Equal(new[] { 1, 224, 224 }, config.InputShape);
        Assert.Equal(50176, config.TotalInputSize); // 1 * 224 * 224
    }

    [Fact]
    public void ResNet_BaseChannels_Are_64_128_256_512()
    {
        var config = new ResNetConfiguration(ResNetVariant.ResNet50, 10);

        Assert.Equal(new[] { 64, 128, 256, 512 }, config.BaseChannels);
    }

    // ============================
    // ResNet Validation Tests
    // ============================

    [Fact]
    public void ResNet_ZeroNumClasses_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ResNetConfiguration(ResNetVariant.ResNet50, 0));
    }

    [Fact]
    public void ResNet_NegativeNumClasses_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ResNetConfiguration(ResNetVariant.ResNet50, -1));
    }

    [Fact]
    public void ResNet_TooSmallHeight_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ResNetConfiguration(ResNetVariant.ResNet50, 10, inputHeight: 16));
    }

    [Fact]
    public void ResNet_ZeroChannels_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ResNetConfiguration(ResNetVariant.ResNet50, 10, inputChannels: 0));
    }

    [Fact]
    public void ResNet_MinimumSize32_Passes()
    {
        var config = new ResNetConfiguration(ResNetVariant.ResNet18, 10, inputHeight: 32, inputWidth: 32);

        Assert.Equal(32, config.InputHeight);
        Assert.Equal(32, config.InputWidth);
    }

    // ============================
    // ResNet Factory Methods
    // ============================

    [Fact]
    public void CreateResNet50_UsesResNet50Variant()
    {
        var config = ResNetConfiguration.CreateResNet50(1000);

        Assert.Equal(ResNetVariant.ResNet50, config.Variant);
        Assert.Equal(1000, config.NumClasses);
    }

    [Fact]
    public void CreateLightweight_UsesResNet18()
    {
        var config = ResNetConfiguration.CreateLightweight(10);

        Assert.Equal(ResNetVariant.ResNet18, config.Variant);
        Assert.Equal(10, config.NumClasses);
    }

    [Fact]
    public void CreateForTesting_UsesMinimalConfig()
    {
        var config = ResNetConfiguration.CreateForTesting(10);

        Assert.Equal(ResNetVariant.ResNet18, config.Variant);
        Assert.Equal(32, config.InputHeight);
        Assert.Equal(32, config.InputWidth);
    }

    [Fact]
    public void ResNet_DefaultFlags()
    {
        var config = new ResNetConfiguration(ResNetVariant.ResNet50, 10);

        Assert.True(config.IncludeClassifier);
        Assert.True(config.ZeroInitResidual);
        Assert.False(config.UseAutodiff);
    }

    // ============================
    // EfficientNet Input Height Tests (compound scaling)
    // ============================

    [Fact]
    public void EfficientNetB0_InputHeight_224()
    {
        var config = new EfficientNetConfiguration(EfficientNetVariant.B0, 1000);
        Assert.Equal(224, config.GetInputHeight());
    }

    [Fact]
    public void EfficientNetB1_InputHeight_240()
    {
        var config = new EfficientNetConfiguration(EfficientNetVariant.B1, 1000);
        Assert.Equal(240, config.GetInputHeight());
    }

    [Fact]
    public void EfficientNetB2_InputHeight_260()
    {
        var config = new EfficientNetConfiguration(EfficientNetVariant.B2, 1000);
        Assert.Equal(260, config.GetInputHeight());
    }

    [Fact]
    public void EfficientNetB3_InputHeight_300()
    {
        var config = new EfficientNetConfiguration(EfficientNetVariant.B3, 1000);
        Assert.Equal(300, config.GetInputHeight());
    }

    [Fact]
    public void EfficientNetB4_InputHeight_380()
    {
        var config = new EfficientNetConfiguration(EfficientNetVariant.B4, 1000);
        Assert.Equal(380, config.GetInputHeight());
    }

    [Fact]
    public void EfficientNetB5_InputHeight_456()
    {
        var config = new EfficientNetConfiguration(EfficientNetVariant.B5, 1000);
        Assert.Equal(456, config.GetInputHeight());
    }

    [Fact]
    public void EfficientNetB6_InputHeight_528()
    {
        var config = new EfficientNetConfiguration(EfficientNetVariant.B6, 1000);
        Assert.Equal(528, config.GetInputHeight());
    }

    [Fact]
    public void EfficientNetB7_InputHeight_600()
    {
        var config = new EfficientNetConfiguration(EfficientNetVariant.B7, 1000);
        Assert.Equal(600, config.GetInputHeight());
    }

    [Fact]
    public void EfficientNet_InputWidthEqualsHeight()
    {
        var config = new EfficientNetConfiguration(EfficientNetVariant.B3, 1000);
        Assert.Equal(config.GetInputHeight(), config.GetInputWidth());
    }

    // ============================
    // EfficientNet Width/Depth Multiplier Tests
    // ============================

    [Fact]
    public void EfficientNetB0_WidthMultiplier_1_0()
    {
        var config = new EfficientNetConfiguration(EfficientNetVariant.B0, 10);
        Assert.Equal(1.0, config.GetWidthMultiplier());
    }

    [Fact]
    public void EfficientNetB7_WidthMultiplier_2_0()
    {
        var config = new EfficientNetConfiguration(EfficientNetVariant.B7, 10);
        Assert.Equal(2.0, config.GetWidthMultiplier());
    }

    [Fact]
    public void EfficientNetB0_DepthMultiplier_1_0()
    {
        var config = new EfficientNetConfiguration(EfficientNetVariant.B0, 10);
        Assert.Equal(1.0, config.GetDepthMultiplier());
    }

    [Fact]
    public void EfficientNetB7_DepthMultiplier_3_1()
    {
        var config = new EfficientNetConfiguration(EfficientNetVariant.B7, 10);
        Assert.Equal(3.1, config.GetDepthMultiplier());
    }

    [Fact]
    public void EfficientNet_MultipliersIncreaseMonotonically()
    {
        var variants = new[]
        {
            EfficientNetVariant.B0, EfficientNetVariant.B1, EfficientNetVariant.B2,
            EfficientNetVariant.B3, EfficientNetVariant.B4, EfficientNetVariant.B5,
            EfficientNetVariant.B6, EfficientNetVariant.B7
        };

        for (int i = 1; i < variants.Length; i++)
        {
            var prev = new EfficientNetConfiguration(variants[i - 1], 10);
            var curr = new EfficientNetConfiguration(variants[i], 10);

            Assert.True(curr.GetWidthMultiplier() >= prev.GetWidthMultiplier(),
                $"Width multiplier should increase: B{i - 1}={prev.GetWidthMultiplier()}, B{i}={curr.GetWidthMultiplier()}");
            Assert.True(curr.GetDepthMultiplier() >= prev.GetDepthMultiplier(),
                $"Depth multiplier should increase: B{i - 1}={prev.GetDepthMultiplier()}, B{i}={curr.GetDepthMultiplier()}");
            Assert.True(curr.GetInputHeight() >= prev.GetInputHeight(),
                $"Input height should increase: B{i - 1}={prev.GetInputHeight()}, B{i}={curr.GetInputHeight()}");
        }
    }

    // ============================
    // EfficientNet Dropout Rate Tests
    // ============================

    [Fact]
    public void EfficientNetB0_DropoutRate_0_2()
    {
        var config = new EfficientNetConfiguration(EfficientNetVariant.B0, 10);
        Assert.Equal(0.2, config.GetDropoutRate());
    }

    [Fact]
    public void EfficientNetB7_DropoutRate_0_5()
    {
        var config = new EfficientNetConfiguration(EfficientNetVariant.B7, 10);
        Assert.Equal(0.5, config.GetDropoutRate());
    }

    [Fact]
    public void EfficientNet_DropoutRateIncreases_WithVariant()
    {
        var b0 = new EfficientNetConfiguration(EfficientNetVariant.B0, 10);
        var b7 = new EfficientNetConfiguration(EfficientNetVariant.B7, 10);

        Assert.True(b7.GetDropoutRate() >= b0.GetDropoutRate());
    }

    // ============================
    // EfficientNet Validation Tests
    // ============================

    [Fact]
    public void EfficientNet_ZeroClasses_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new EfficientNetConfiguration(EfficientNetVariant.B0, 0));
    }

    [Fact]
    public void EfficientNet_ZeroChannels_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new EfficientNetConfiguration(EfficientNetVariant.B0, 10, inputChannels: 0));
    }

    [Fact]
    public void EfficientNet_CustomVariant_RequiresCustomParams()
    {
        Assert.Throws<ArgumentException>(() =>
            new EfficientNetConfiguration(EfficientNetVariant.Custom, 10));
    }

    [Fact]
    public void EfficientNet_CustomVariant_WithParams_Succeeds()
    {
        var config = new EfficientNetConfiguration(
            EfficientNetVariant.Custom, 10,
            customInputHeight: 64,
            customWidthMultiplier: 1.5,
            customDepthMultiplier: 1.2);

        Assert.Equal(64, config.GetInputHeight());
        Assert.Equal(1.5, config.GetWidthMultiplier());
        Assert.Equal(1.2, config.GetDepthMultiplier());
    }

    [Fact]
    public void EfficientNet_InputShape_IsCorrect()
    {
        var config = new EfficientNetConfiguration(EfficientNetVariant.B0, 10);

        Assert.Equal(new[] { 3, 224, 224 }, config.InputShape);
    }

    [Fact]
    public void EfficientNet_CreateForTesting_UsesCustom32x32()
    {
        var config = EfficientNetConfiguration.CreateForTesting(10);

        Assert.Equal(EfficientNetVariant.Custom, config.Variant);
        Assert.Equal(32, config.GetInputHeight());
    }

    // ============================
    // RL ExplorationScheduleConfig Tests
    // ============================

    [Fact]
    public void ExplorationSchedule_Defaults_AreCorrect()
    {
        var config = new ExplorationScheduleConfig<double>();

        Assert.Equal(1.0, config.InitialEpsilon);
        Assert.Equal(0.01, config.FinalEpsilon);
        Assert.Equal(100000, config.DecaySteps);
        Assert.Equal(ExplorationDecayType.Linear, config.DecayType);
    }

    [Fact]
    public void ExplorationSchedule_InitialGreaterThanFinal()
    {
        var config = new ExplorationScheduleConfig<double>();

        Assert.True(Convert.ToDouble(config.InitialEpsilon) > Convert.ToDouble(config.FinalEpsilon));
    }

    [Fact]
    public void ExplorationSchedule_CustomValues_Stored()
    {
        var config = new ExplorationScheduleConfig<double>
        {
            InitialEpsilon = 0.5,
            FinalEpsilon = 0.05,
            DecaySteps = 50000,
            DecayType = ExplorationDecayType.Exponential
        };

        Assert.Equal(0.5, config.InitialEpsilon);
        Assert.Equal(0.05, config.FinalEpsilon);
        Assert.Equal(50000, config.DecaySteps);
        Assert.Equal(ExplorationDecayType.Exponential, config.DecayType);
    }

    // ============================
    // PrioritizedReplayConfig Tests
    // ============================

    [Fact]
    public void PrioritizedReplay_Defaults_AreCorrect()
    {
        var config = new PrioritizedReplayConfig<double>();

        Assert.Equal(0.6, config.Alpha);
        Assert.Equal(0.4, config.InitialBeta);
        Assert.Equal(1.0, config.FinalBeta);
        Assert.Equal(1e-6, config.PriorityEpsilon);
        Assert.Equal(100000, config.BetaAnnealingSteps);
    }

    [Fact]
    public void PrioritizedReplay_AlphaInRange_0To1()
    {
        var config = new PrioritizedReplayConfig<double>();

        Assert.True(Convert.ToDouble(config.Alpha) >= 0.0);
        Assert.True(Convert.ToDouble(config.Alpha) <= 1.0);
    }

    [Fact]
    public void PrioritizedReplay_BetaAnnealsUpward()
    {
        var config = new PrioritizedReplayConfig<double>();

        Assert.True(Convert.ToDouble(config.InitialBeta) < Convert.ToDouble(config.FinalBeta));
    }

    [Fact]
    public void PrioritizedReplay_PriorityEpsilon_IsSmallPositive()
    {
        var config = new PrioritizedReplayConfig<double>();

        Assert.True(Convert.ToDouble(config.PriorityEpsilon) > 0);
        Assert.True(Convert.ToDouble(config.PriorityEpsilon) < 1e-3);
    }

    // ============================
    // TargetNetworkConfig Tests
    // ============================

    [Fact]
    public void TargetNetwork_Defaults_AreCorrect()
    {
        var config = new TargetNetworkConfig<double>();

        Assert.Equal(1000, config.UpdateFrequency);
        Assert.False(config.UseSoftUpdate);
        Assert.Equal(0.005, config.Tau);
    }

    [Fact]
    public void TargetNetwork_Tau_InRange_0To1()
    {
        var config = new TargetNetworkConfig<double>();

        Assert.True(Convert.ToDouble(config.Tau) > 0);
        Assert.True(Convert.ToDouble(config.Tau) <= 1.0);
    }

    // ============================
    // RewardClippingConfig Tests
    // ============================

    [Fact]
    public void RewardClipping_Defaults_AreCorrect()
    {
        var config = new RewardClippingConfig<double>();

        Assert.Equal(-1.0, config.MinReward);
        Assert.Equal(1.0, config.MaxReward);
        Assert.True(config.UseClipping);
    }

    [Fact]
    public void RewardClipping_RangeIsSymmetric()
    {
        var config = new RewardClippingConfig<double>();

        Assert.Equal(-Convert.ToDouble(config.MaxReward), Convert.ToDouble(config.MinReward));
    }

    [Fact]
    public void RewardClipping_MinLessThanMax()
    {
        var config = new RewardClippingConfig<double>();

        Assert.True(Convert.ToDouble(config.MinReward) < Convert.ToDouble(config.MaxReward));
    }

    // ============================
    // RLEarlyStoppingConfig Tests
    // ============================

    [Fact]
    public void RLEarlyStopping_Defaults_AreCorrect()
    {
        var config = new RLEarlyStoppingConfig<double>();

        // T? for double defaults to 0 (not null) due to generic type handling
        Assert.Equal(0.0, config.RewardThreshold);
        Assert.Equal(100, config.PatienceEpisodes);
        Assert.Equal(0.01, config.MinImprovement);
    }

    [Fact]
    public void RLEarlyStopping_CustomValues_Stored()
    {
        var config = new RLEarlyStoppingConfig<double>
        {
            RewardThreshold = 100.0,
            PatienceEpisodes = 50,
            MinImprovement = 0.001
        };

        Assert.Equal(100.0, config.RewardThreshold);
        Assert.Equal(50, config.PatienceEpisodes);
        Assert.Equal(0.001, config.MinImprovement);
    }

    // ============================
    // RLEpisodeMetrics Tests
    // ============================

    [Fact]
    public void RLEpisodeMetrics_Defaults_AreZero()
    {
        var metrics = new RLEpisodeMetrics<double>();

        Assert.Equal(0.0, metrics.TotalReward);
        Assert.Equal(0.0, metrics.AverageLoss);
        Assert.Equal(0.0, metrics.AverageRewardRecent);
    }

    [Fact]
    public void RLEpisodeMetrics_CustomValues_Stored()
    {
        var metrics = new RLEpisodeMetrics<double>
        {
            Episode = 42,
            TotalReward = 150.5,
            Steps = 200,
            AverageLoss = 0.05,
            TerminatedNaturally = true,
            AverageRewardRecent = 140.0,
            ElapsedTime = TimeSpan.FromMinutes(5)
        };

        Assert.Equal(42, metrics.Episode);
        Assert.Equal(150.5, metrics.TotalReward);
        Assert.Equal(200, metrics.Steps);
        Assert.Equal(0.05, metrics.AverageLoss);
        Assert.True(metrics.TerminatedNaturally);
        Assert.Equal(140.0, metrics.AverageRewardRecent);
        Assert.Equal(TimeSpan.FromMinutes(5), metrics.ElapsedTime);
    }

    // ============================
    // RLStepMetrics Tests
    // ============================

    [Fact]
    public void RLStepMetrics_Defaults_AreZero()
    {
        var metrics = new RLStepMetrics<double>();

        Assert.Equal(0.0, metrics.Reward);
        // T? for double defaults to 0 (not null) due to generic type handling
        Assert.Equal(0.0, metrics.Loss);
        Assert.False(metrics.DidTrain);
    }

    [Fact]
    public void RLStepMetrics_CustomValues_Stored()
    {
        var metrics = new RLStepMetrics<double>
        {
            Episode = 10,
            Step = 50,
            TotalSteps = 1050,
            Reward = 1.5,
            Loss = 0.01,
            DidTrain = true
        };

        Assert.Equal(10, metrics.Episode);
        Assert.Equal(50, metrics.Step);
        Assert.Equal(1050, metrics.TotalSteps);
        Assert.Equal(1.5, metrics.Reward);
        Assert.Equal(0.01, metrics.Loss);
        Assert.True(metrics.DidTrain);
    }

    // ============================
    // RLTrainingSummary Tests
    // ============================

    [Fact]
    public void RLTrainingSummary_Defaults_AreZero()
    {
        var summary = new RLTrainingSummary<double>();

        Assert.Equal(0.0, summary.AverageReward);
        Assert.Equal(0.0, summary.BestReward);
        Assert.Equal(0.0, summary.FinalAverageReward);
        Assert.Equal(0.0, summary.AverageLoss);
        Assert.False(summary.EarlyStopTriggered);
    }

    [Fact]
    public void RLTrainingSummary_CustomValues_Stored()
    {
        var summary = new RLTrainingSummary<double>
        {
            TotalEpisodes = 1000,
            TotalSteps = 500000,
            AverageReward = 75.3,
            BestReward = 210.0,
            FinalAverageReward = 195.0,
            AverageLoss = 0.02,
            TotalTime = TimeSpan.FromHours(2),
            EarlyStopTriggered = true
        };

        Assert.Equal(1000, summary.TotalEpisodes);
        Assert.Equal(500000, summary.TotalSteps);
        Assert.Equal(75.3, summary.AverageReward);
        Assert.Equal(210.0, summary.BestReward);
        Assert.Equal(195.0, summary.FinalAverageReward);
        Assert.Equal(0.02, summary.AverageLoss);
        Assert.Equal(TimeSpan.FromHours(2), summary.TotalTime);
        Assert.True(summary.EarlyStopTriggered);
    }

    [Fact]
    public void RLTrainingSummary_BestReward_GreaterOrEqualAverage()
    {
        // Logically, best >= average always holds
        var summary = new RLTrainingSummary<double>
        {
            AverageReward = 75.3,
            BestReward = 210.0
        };

        Assert.True(Convert.ToDouble(summary.BestReward) >= Convert.ToDouble(summary.AverageReward));
    }
}
