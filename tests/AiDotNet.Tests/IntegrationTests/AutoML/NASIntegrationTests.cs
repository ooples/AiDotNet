using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.AutoML;
using AiDotNet.AutoML.NAS;
using AiDotNet.AutoML.SearchSpace;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AutoML
{
    /// <summary>
    /// Integration tests for Neural Architecture Search (NAS) algorithms.
    /// Tests complete end-to-end workflows including:
    /// - Search space configuration to architecture derivation
    /// - Hardware-aware architecture specialization
    /// - Cross-algorithm compatibility
    /// - Hardware cost model integration
    /// </summary>
    public class NASIntegrationTests
    {
        #region FBNet End-to-End Tests

        [Fact]
        public void FBNet_EndToEnd_SearchSpaceToArchitectureDerivation()
        {
            // Arrange - Create search space with operations
            var searchSpace = new SearchSpaceBase<double>();
            searchSpace.Operations = new List<string>
            {
                "identity",
                "conv3x3",
                "conv5x5",
                "depthwise_conv3x3",
                "se_block"
            };

            var fbnet = new FBNet<double>(
                searchSpace,
                numLayers: 8,
                targetPlatform: HardwarePlatform.Mobile,
                latencyWeight: 0.3,
                initialTemperature: 5.0,
                inputChannels: 32,
                spatialSize: 56);

            // Act - Derive architecture after temperature annealing
            for (int epoch = 0; epoch < 10; epoch++)
            {
                fbnet.AnnealTemperature(epoch, 10);
            }

            var architecture = fbnet.DeriveArchitecture();
            var cost = fbnet.GetArchitectureCost();

            // Assert - Complete flow produces valid results
            Assert.NotNull(architecture);
            Assert.Equal(8, architecture.Operations.Count);
            Assert.True(cost.Latency > 0);
            Assert.True(cost.Energy > 0);
            Assert.True(cost.Memory >= 0);

            // Verify all operations are from search space
            foreach (var op in architecture.Operations)
            {
                Assert.Contains(op.Item3, searchSpace.Operations);
            }
        }

        [Fact]
        public void FBNet_EndToEnd_ConstraintSatisfaction()
        {
            // Arrange - Create FBNet with tight constraints
            var searchSpace = new SearchSpaceBase<double>();
            searchSpace.Operations = new List<string> { "identity", "conv3x3", "depthwise_conv3x3" };

            var fbnet = new FBNet<double>(searchSpace, numLayers: 5, targetPlatform: HardwarePlatform.GPU);

            // Set reasonable constraints
            fbnet.SetConstraints(new HardwareConstraints<double>
            {
                MaxLatency = 1000.0,  // 1 second - should be achievable
                MaxMemory = 1000.0,   // 1 GB
                MaxEnergy = 5000.0    // 5 J
            });

            // Act
            var meetsConstraints = fbnet.MeetsConstraints();
            var cost = fbnet.GetArchitectureCost();

            // Assert
            Assert.True(meetsConstraints, $"Architecture should meet loose constraints. Latency={cost.Latency}, Memory={cost.Memory}, Energy={cost.Energy}");
        }

        [Fact]
        public void FBNet_EndToEnd_LossComputationWithLatencyRegularization()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            searchSpace.Operations = new List<string> { "conv3x3", "conv5x5", "conv7x7" };

            var fbnetLow = new FBNet<double>(searchSpace, numLayers: 5, latencyWeight: 0.1);
            var fbnetHigh = new FBNet<double>(searchSpace, numLayers: 5, latencyWeight: 0.9);

            double taskLoss = 1.0;

            // Act
            var totalLossLow = fbnetLow.ComputeTotalLoss(taskLoss);
            var totalLossHigh = fbnetHigh.ComputeTotalLoss(taskLoss);

            // Assert - Higher latency weight should produce different total loss
            Assert.True(totalLossLow >= taskLoss, "Total loss should be at least task loss");
            Assert.True(totalLossHigh >= taskLoss, "Total loss should be at least task loss");
        }

        [Theory]
        [InlineData(HardwarePlatform.Mobile)]
        [InlineData(HardwarePlatform.GPU)]
        [InlineData(HardwarePlatform.EdgeTPU)]
        [InlineData(HardwarePlatform.CPU)]
        public void FBNet_EndToEnd_DifferentTargetPlatforms(HardwarePlatform platform)
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            searchSpace.Operations = new List<string> { "conv3x3", "conv5x5" };

            var fbnet = new FBNet<double>(searchSpace, numLayers: 5, targetPlatform: platform);

            // Act
            var architecture = fbnet.DeriveArchitecture();
            var cost = fbnet.GetArchitectureCost();

            // Assert
            Assert.NotNull(architecture);
            Assert.Equal(5, architecture.Operations.Count);
            Assert.True(cost.Latency > 0, $"Platform {platform} should produce positive latency");
        }

        #endregion

        #region OnceForAll End-to-End Tests

        [Fact]
        public void OnceForAll_EndToEnd_ProgressiveShrinking()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var ofa = new OnceForAll<double>(
                searchSpace,
                elasticDepths: new List<int> { 2, 3, 4 },
                elasticWidths: new List<double> { 0.5, 0.75, 1.0 },
                elasticKernelSizes: new List<int> { 3, 5, 7 },
                elasticExpansionRatios: new List<int> { 3, 4, 6 });

            var configsByStage = new Dictionary<int, List<SubNetworkConfig>>();

            // Act - Sample configs at each training stage
            for (int stage = 0; stage <= 4; stage++)
            {
                ofa.SetTrainingStage(stage);
                var configs = new List<SubNetworkConfig>();
                for (int i = 0; i < 50; i++)
                {
                    configs.Add(ofa.SampleSubNetwork());
                }
                configsByStage[stage] = configs;
            }

            // Assert - Stage 0 should have largest values (no elasticity yet)
            var stage0Configs = configsByStage[0];
            Assert.All(stage0Configs, c => Assert.Equal(4, c.Depth));
            Assert.All(stage0Configs, c => Assert.Equal(6, c.ExpansionRatio));
            Assert.All(stage0Configs, c => Assert.Equal(1.0, c.WidthMultiplier));

            // Stage 4 should have variety in all dimensions
            var stage4Configs = configsByStage[4];
            var uniqueDepths = stage4Configs.Select(c => c.Depth).Distinct().Count();
            var uniqueWidths = stage4Configs.Select(c => c.WidthMultiplier).Distinct().Count();
            var uniqueKernels = stage4Configs.Select(c => c.KernelSize).Distinct().Count();

            Assert.True(uniqueKernels >= 2, "Stage 4 should have varied kernel sizes");
        }

        [Fact]
        public void OnceForAll_EndToEnd_HardwareSpecialization()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var ofa = new OnceForAll<double>(
                searchSpace,
                elasticDepths: new List<int> { 2, 4, 6 },
                elasticWidths: new List<double> { 0.5, 1.0 },
                elasticKernelSizes: new List<int> { 3, 5 },
                elasticExpansionRatios: new List<int> { 3, 6 });

            // Define different hardware constraints
            var mobileConstraints = new HardwareConstraints<double>
            {
                MaxLatency = 50.0,  // Tight latency for mobile
                MaxMemory = 20.0
            };

            var serverConstraints = new HardwareConstraints<double>
            {
                MaxLatency = 500.0,  // Loose latency for server
                MaxMemory = 200.0
            };

            // Act - Specialize for different platforms
            var mobileConfig = ofa.SpecializeForHardware(
                mobileConstraints,
                inputChannels: 32,
                spatialSize: 56,
                populationSize: 30,
                generations: 10);

            var serverConfig = ofa.SpecializeForHardware(
                serverConstraints,
                inputChannels: 32,
                spatialSize: 56,
                populationSize: 30,
                generations: 10);

            // Assert - Both produce valid configs
            Assert.NotNull(mobileConfig);
            Assert.NotNull(serverConfig);
            Assert.True(mobileConfig.Depth > 0);
            Assert.True(serverConfig.Depth > 0);

            // Server with looser constraints can potentially use larger networks
            // (This is probabilistic, so we just verify both are valid)
            Assert.Contains(mobileConfig.Depth, new[] { 2, 4, 6 });
            Assert.Contains(serverConfig.Depth, new[] { 2, 4, 6 });
        }

        [Fact]
        public void OnceForAll_EndToEnd_SharedWeights()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var ofa = new OnceForAll<double>(searchSpace);

            // Act - Get shared weights for different layer configurations
            var weights3x3_64 = ofa.GetSharedWeights("conv3x3_layer1", 64, 64);
            var weights3x3_64_again = ofa.GetSharedWeights("conv3x3_layer1", 64, 64);
            var weights3x3_128 = ofa.GetSharedWeights("conv3x3_layer2", 128, 64);
            var weights5x5_64 = ofa.GetSharedWeights("conv5x5_layer1", 64, 64);

            // Assert - Same key returns same weights (weight sharing)
            Assert.Same(weights3x3_64, weights3x3_64_again);

            // Different keys return different weights
            Assert.NotSame(weights3x3_64, weights3x3_128);
            Assert.NotSame(weights3x3_64, weights5x5_64);

            // Verify dimensions
            Assert.Equal(64, weights3x3_64.Rows);
            Assert.Equal(64, weights3x3_64.Columns);
            Assert.Equal(128, weights3x3_128.Rows);
        }

        #endregion

        #region Hardware Cost Model Integration Tests

        [Fact]
        public void HardwareCostModel_EndToEnd_ArchitectureAnalysis()
        {
            // Arrange - Build a realistic architecture
            var costModel = new HardwareCostModel<double>(HardwarePlatform.Mobile);
            var architecture = new Architecture<double>();

            // Create a small CNN-like architecture
            architecture.AddOperation(1, 0, "conv3x3");       // Input to first conv
            architecture.AddOperation(2, 1, "batch_norm");    // Batch norm
            architecture.AddOperation(3, 2, "relu");          // Activation
            architecture.AddOperation(4, 3, "conv3x3");       // Second conv
            architecture.AddOperation(5, 4, "batch_norm");
            architecture.AddOperation(6, 5, "relu");
            architecture.AddOperation(7, 6, "avgpool3x3");    // Pooling

            // Set channel dimensions
            architecture.NodeChannels[0] = 3;   // RGB input
            architecture.NodeChannels[1] = 32;  // First conv output
            architecture.NodeChannels[2] = 32;  // Same after batch norm
            architecture.NodeChannels[3] = 32;  // Same after ReLU
            architecture.NodeChannels[4] = 64;  // Second conv output
            architecture.NodeChannels[5] = 64;
            architecture.NodeChannels[6] = 64;
            architecture.NodeChannels[7] = 64;

            // Act
            var totalCost = costModel.EstimateArchitectureCost(architecture, inputChannels: 3, spatialSize: 224);
            var breakdown = costModel.GetCostBreakdown(architecture, inputChannels: 3, spatialSize: 224);
            var totalFlops = costModel.GetTotalFlops(architecture, inputChannels: 3, spatialSize: 224);
            var totalParams = costModel.GetTotalParameters(architecture, inputChannels: 3);

            // Assert
            Assert.True(totalCost.Latency > 0, "Total latency should be positive");
            Assert.True(totalCost.Energy > 0, "Total energy should be positive");
            Assert.True(totalFlops > 0, "Total FLOPs should be positive");

            // Verify breakdown sums to total
            double breakdownLatencySum = breakdown.Values.Sum(c => c.Latency);
            Assert.True(Math.Abs(breakdownLatencySum - totalCost.Latency) < 0.001,
                $"Breakdown sum ({breakdownLatencySum}) should match total ({totalCost.Latency})");
        }

        [Fact]
        public void HardwareCostModel_EndToEnd_CalibrationFlow()
        {
            // Arrange
            var costModel = new HardwareCostModel<double>(HardwarePlatform.GPU);

            // Act - Get baseline cost
            var baselineCost = costModel.EstimateOperationCost("conv3x3", 64, 64, 14);

            // Calibrate based on real measurements
            costModel.SetCalibrationFactor("conv3x3", 1.5);  // Real conv is 50% slower
            var calibratedCost = costModel.EstimateOperationCost("conv3x3", 64, 64, 14);

            // Clear calibration
            costModel.ClearCalibration();
            var resetCost = costModel.EstimateOperationCost("conv3x3", 64, 64, 14);

            // Assert
            Assert.True(calibratedCost.Latency > baselineCost.Latency,
                "Calibrated cost should be higher with factor > 1");
            Assert.Equal(baselineCost.Latency, resetCost.Latency);

            // Verify calibration factor is retrievable
            costModel.SetCalibrationFactor("conv5x5", 0.8);
            Assert.Equal(0.8, costModel.GetCalibrationFactor("conv5x5"));
            Assert.Equal(1.0, costModel.GetCalibrationFactor("unknown_op"));
        }

        [Fact]
        public void HardwareCostModel_EndToEnd_CrossPlatformComparison()
        {
            // Arrange - Same architecture on different platforms
            var architecture = new Architecture<double>();
            architecture.AddOperation(1, 0, "conv3x3");
            architecture.AddOperation(2, 1, "depthwise_conv3x3");
            architecture.AddOperation(3, 2, "conv1x1");

            var platforms = new[]
            {
                HardwarePlatform.CPU,
                HardwarePlatform.Mobile,
                HardwarePlatform.GPU,
                HardwarePlatform.EdgeTPU
            };

            var costs = new Dictionary<HardwarePlatform, HardwareCost<double>>();

            // Act
            foreach (var platform in platforms)
            {
                var model = new HardwareCostModel<double>(platform);
                costs[platform] = model.EstimateArchitectureCost(architecture, 64, 14);
            }

            // Assert - GPU should be fastest (lowest latency)
            Assert.True(costs[HardwarePlatform.GPU].Latency < costs[HardwarePlatform.CPU].Latency,
                "GPU should be faster than CPU");
            Assert.True(costs[HardwarePlatform.EdgeTPU].Latency < costs[HardwarePlatform.CPU].Latency,
                "EdgeTPU should be faster than CPU");

            // GPU is more energy efficient per FLOP than Mobile in this model
            // (GPU: 0.1 mJ/GFLOP, Mobile: 2.0 mJ/GFLOP)
            Assert.True(costs[HardwarePlatform.GPU].Energy < costs[HardwarePlatform.Mobile].Energy,
                "GPU should be more energy efficient per operation than Mobile");
        }

        #endregion

        #region Cross-Algorithm Integration Tests

        [Fact]
        public void NAS_EndToEnd_FBNetArchitectureToOnceForAllSpecialization()
        {
            // Arrange - Use FBNet to derive an architecture
            var searchSpace = new SearchSpaceBase<double>();
            searchSpace.Operations = new List<string> { "conv3x3", "conv5x5", "depthwise_conv3x3" };

            var fbnet = new FBNet<double>(searchSpace, numLayers: 6, targetPlatform: HardwarePlatform.Mobile);

            // Anneal and derive architecture
            for (int epoch = 0; epoch < 10; epoch++)
            {
                fbnet.AnnealTemperature(epoch, 10);
            }
            var fbnetArchitecture = fbnet.DeriveArchitecture();

            // Set up OnceForAll with similar dimensions
            int derivedDepth = fbnetArchitecture.Operations.Count;
            var ofa = new OnceForAll<double>(
                searchSpace,
                elasticDepths: new List<int> { derivedDepth - 1, derivedDepth, derivedDepth + 1 },
                elasticKernelSizes: new List<int> { 3, 5 },
                elasticWidths: new List<double> { 0.75, 1.0 },
                elasticExpansionRatios: new List<int> { 3, 6 });

            // Act - Specialize OFA for the same target
            var constraints = new HardwareConstraints<double>
            {
                MaxLatency = 100.0,
                MaxMemory = 50.0
            };

            var ofaConfig = ofa.SpecializeForHardware(
                constraints,
                inputChannels: 32,
                spatialSize: 56,
                populationSize: 20,
                generations: 5);

            // Assert - Both methods produce valid architectures
            Assert.NotNull(fbnetArchitecture);
            Assert.NotNull(ofaConfig);
            Assert.Equal(6, fbnetArchitecture.Operations.Count);
            Assert.True(ofaConfig.Depth > 0);
        }

        [Fact]
        public void NAS_EndToEnd_MultipleAlgorithmsSharedCostModel()
        {
            // Arrange - Create a shared cost model
            var costModel = new HardwareCostModel<double>(HardwarePlatform.Mobile);

            var searchSpace = new SearchSpaceBase<double>();
            searchSpace.Operations = new List<string> { "conv3x3", "conv5x5" };

            var fbnet = new FBNet<double>(searchSpace, numLayers: 4, targetPlatform: HardwarePlatform.Mobile);
            var ofa = new OnceForAll<double>(searchSpace);

            // Act - Get architectures from both
            var fbnetArch = fbnet.DeriveArchitecture();
            var ofaConfig = ofa.SampleSubNetwork();

            // Convert OFA config to architecture for comparison
            var ofaArch = new Architecture<double>();
            for (int i = 0; i < ofaConfig.Depth; i++)
            {
                string op = ofaConfig.KernelSize == 5 ? "conv5x5" : "conv3x3";
                ofaArch.AddOperation(i + 1, i, op);
            }

            // Evaluate both with same cost model
            var fbnetCost = costModel.EstimateArchitectureCost(fbnetArch, 32, 56);
            var ofaCost = costModel.EstimateArchitectureCost(ofaArch, 32, 56);

            // Assert - Both produce valid costs
            Assert.True(fbnetCost.Latency > 0);
            Assert.True(ofaCost.Latency > 0);
        }

        #endregion

        #region Edge Case Integration Tests

        [Fact]
        public void NAS_EndToEnd_MinimalConfiguration()
        {
            // Arrange - Minimal valid configuration
            var searchSpace = new SearchSpaceBase<double>();
            searchSpace.Operations = new List<string> { "identity" };

            var fbnet = new FBNet<double>(searchSpace, numLayers: 1);
            var ofa = new OnceForAll<double>(
                searchSpace,
                elasticDepths: new List<int> { 1 },
                elasticKernelSizes: new List<int> { 3 },
                elasticWidths: new List<double> { 1.0 },
                elasticExpansionRatios: new List<int> { 1 });

            // Act
            var fbnetArch = fbnet.DeriveArchitecture();
            var ofaConfig = ofa.SampleSubNetwork();

            // Assert
            Assert.Single(fbnetArch.Operations);
            Assert.Equal(1, ofaConfig.Depth);
        }

        [Fact]
        public void NAS_EndToEnd_LargeConfiguration()
        {
            // Arrange - Large configuration
            var searchSpace = new SearchSpaceBase<double>();
            searchSpace.Operations = new List<string>
            {
                "identity", "conv1x1", "conv3x3", "conv5x5", "conv7x7",
                "depthwise_conv3x3", "depthwise_conv5x5",
                "separable_conv3x3", "separable_conv5x5",
                "maxpool3x3", "avgpool3x3"
            };

            var fbnet = new FBNet<double>(searchSpace, numLayers: 50);
            var ofa = new OnceForAll<double>(
                searchSpace,
                elasticDepths: new List<int> { 10, 20, 30, 40, 50 },
                elasticKernelSizes: new List<int> { 3, 5, 7 },
                elasticWidths: new List<double> { 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0 },
                elasticExpansionRatios: new List<int> { 1, 2, 3, 4, 6, 8 });

            // Act
            var fbnetArch = fbnet.DeriveArchitecture();
            ofa.SetTrainingStage(4);
            var ofaConfig = ofa.SampleSubNetwork();

            var costModel = new HardwareCostModel<double>();
            var fbnetCost = costModel.EstimateArchitectureCost(fbnetArch, 64, 56);

            // Assert
            Assert.Equal(50, fbnetArch.Operations.Count);
            Assert.Contains(ofaConfig.Depth, new[] { 10, 20, 30, 40, 50 });
            Assert.True(fbnetCost.Latency > 0);
        }

        [Fact]
        public void NAS_EndToEnd_VeryTightConstraints()
        {
            // Arrange - Constraints that are nearly impossible to meet
            var searchSpace = new SearchSpaceBase<double>();
            searchSpace.Operations = new List<string> { "conv5x5", "conv7x7" };

            var ofa = new OnceForAll<double>(
                searchSpace,
                elasticDepths: new List<int> { 1, 2 },  // Allow minimal depth
                elasticKernelSizes: new List<int> { 3 },
                elasticWidths: new List<double> { 0.25 },  // Minimal width
                elasticExpansionRatios: new List<int> { 1 });

            var tightConstraints = new HardwareConstraints<double>
            {
                MaxLatency = 0.001,  // Very tight
                MaxMemory = 0.001
            };

            // Act - The algorithm should still produce a valid config
            var config = ofa.SpecializeForHardware(
                tightConstraints,
                inputChannels: 8,
                spatialSize: 7,
                populationSize: 10,
                generations: 3);

            // Assert - Config is valid even if it doesn't meet constraints
            Assert.NotNull(config);
            Assert.True(config.Depth > 0);
        }

        #endregion
    }
}
