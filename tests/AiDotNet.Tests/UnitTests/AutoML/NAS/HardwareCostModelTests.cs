using System;
using System.Collections.Generic;
using AiDotNet.AutoML;
using AiDotNet.AutoML.NAS;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AutoML.NAS
{
    /// <summary>
    /// Unit tests for the HardwareCostModel class.
    /// </summary>
    public class HardwareCostModelTests
    {
        [Fact]
        public void HardwareCostModel_Constructor_WithDefaultPlatform_InitializesCorrectly()
        {
            // Arrange & Act
            var model = new HardwareCostModel<double>();

            // Assert
            Assert.NotNull(model);
        }

        [Theory]
        [InlineData(HardwarePlatform.Mobile)]
        [InlineData(HardwarePlatform.GPU)]
        [InlineData(HardwarePlatform.EdgeTPU)]
        [InlineData(HardwarePlatform.CPU)]
        public void HardwareCostModel_Constructor_WithDifferentPlatforms_InitializesCorrectly(HardwarePlatform platform)
        {
            // Arrange & Act
            var model = new HardwareCostModel<double>(platform);

            // Assert
            Assert.NotNull(model);
        }

        [Theory]
        [InlineData("identity")]
        [InlineData("conv3x3")]
        [InlineData("conv5x5")]
        [InlineData("conv1x1")]
        [InlineData("depthwise_conv3x3")]
        [InlineData("maxpool3x3")]
        [InlineData("avgpool3x3")]
        [InlineData("se_block")]
        public void HardwareCostModel_EstimateOperationCost_KnownOperations_ReturnsValidCost(string operation)
        {
            // Arrange
            var model = new HardwareCostModel<double>();
            int inputChannels = 64;
            int outputChannels = 128;
            int spatialSize = 14;

            // Act
            var cost = model.EstimateOperationCost(operation, inputChannels, outputChannels, spatialSize);

            // Assert
            Assert.NotNull(cost);
            Assert.True(cost.Latency >= 0.0);
            Assert.True(cost.Energy >= 0.0);
            Assert.True(cost.Memory >= 0.0);
        }

        [Fact]
        public void HardwareCostModel_EstimateOperationCost_UnknownOperation_ReturnsConservativeEstimate()
        {
            // Arrange
            var model = new HardwareCostModel<double>();
            int inputChannels = 32;
            int outputChannels = 64;
            int spatialSize = 14;

            // Act
            var cost = model.EstimateOperationCost("unknown_operation", inputChannels, outputChannels, spatialSize);
            var conv3x3Cost = model.EstimateOperationCost("conv3x3", inputChannels, outputChannels, spatialSize);

            // Assert - unknown operations use conservative conv3x3 estimate for safety
            Assert.NotNull(cost);
            Assert.True(cost.Latency > 0.0, "Unknown operation should have positive latency");
            Assert.True(cost.Energy > 0.0, "Unknown operation should have positive energy");
            Assert.True(cost.Memory > 0.0, "Unknown operation should have positive memory");

            // The conservative estimate should match conv3x3 (used as fallback for unknown ops)
            Assert.Equal(conv3x3Cost.Latency, cost.Latency);
            Assert.Equal(conv3x3Cost.Energy, cost.Energy);
            Assert.Equal(conv3x3Cost.Memory, cost.Memory);
        }

        [Fact]
        public void HardwareCostModel_EstimateOperationCost_ScalesWithChannels()
        {
            // Arrange
            var model = new HardwareCostModel<double>();
            int spatialSize = 14;

            // Act - compare costs for different channel counts
            var costSmall = model.EstimateOperationCost("conv3x3", 32, 32, spatialSize);
            var costLarge = model.EstimateOperationCost("conv3x3", 128, 128, spatialSize);

            // Assert - larger channels should have higher cost
            Assert.True(costLarge.Latency > costSmall.Latency);
            Assert.True(costLarge.Energy > costSmall.Energy);
        }

        [Fact]
        public void HardwareCostModel_EstimateOperationCost_ScalesWithSpatialSize()
        {
            // Arrange
            var model = new HardwareCostModel<double>();
            int inputChannels = 64;
            int outputChannels = 64;

            // Act - compare costs for different spatial sizes
            var costSmall = model.EstimateOperationCost("conv3x3", inputChannels, outputChannels, 7);
            var costLarge = model.EstimateOperationCost("conv3x3", inputChannels, outputChannels, 28);

            // Assert - larger spatial size should have higher cost
            Assert.True(costLarge.Latency > costSmall.Latency);
            Assert.True(costLarge.Energy > costSmall.Energy);
        }

        [Fact]
        public void HardwareCostModel_EstimateOperationCost_GPUFasterThanMobile()
        {
            // Arrange
            var mobileModel = new HardwareCostModel<double>(HardwarePlatform.Mobile);
            var gpuModel = new HardwareCostModel<double>(HardwarePlatform.GPU);

            // Act
            var mobileCost = mobileModel.EstimateOperationCost("conv3x3", 64, 64, 14);
            var gpuCost = gpuModel.EstimateOperationCost("conv3x3", 64, 64, 14);

            // Assert - GPU should be faster (lower latency)
            Assert.True(gpuCost.Latency < mobileCost.Latency);
        }

        [Fact]
        public void HardwareCostModel_EstimateArchitectureCost_EmptyArchitecture_ReturnsZeroCost()
        {
            // Arrange
            var model = new HardwareCostModel<double>();
            var architecture = new Architecture<double>();

            // Act
            var cost = model.EstimateArchitectureCost(architecture, inputChannels: 32, spatialSize: 14);

            // Assert
            Assert.Equal(0.0, cost.Latency);
            Assert.Equal(0.0, cost.Energy);
            Assert.Equal(0.0, cost.Memory);
        }

        [Fact]
        public void HardwareCostModel_EstimateArchitectureCost_SingleOperation_ReturnsValidCost()
        {
            // Arrange
            var model = new HardwareCostModel<double>();
            var architecture = new Architecture<double>();
            architecture.Operations.Add((1, 0, "conv3x3"));

            // Act
            var cost = model.EstimateArchitectureCost(architecture, inputChannels: 32, spatialSize: 14);

            // Assert
            Assert.True(cost.Latency > 0.0);
            Assert.True(cost.Energy > 0.0);
        }

        [Fact]
        public void HardwareCostModel_EstimateArchitectureCost_MultipleOperations_SumsCosts()
        {
            // Arrange
            var model = new HardwareCostModel<double>();
            var architectureSingle = new Architecture<double>();
            architectureSingle.Operations.Add((1, 0, "conv3x3"));

            var architectureDouble = new Architecture<double>();
            architectureDouble.Operations.Add((1, 0, "conv3x3"));
            architectureDouble.Operations.Add((2, 1, "conv3x3"));

            // Act
            var costSingle = model.EstimateArchitectureCost(architectureSingle, inputChannels: 32, spatialSize: 14);
            var costDouble = model.EstimateArchitectureCost(architectureDouble, inputChannels: 32, spatialSize: 14);

            // Assert - two operations should have higher cost
            Assert.True(costDouble.Latency > costSingle.Latency);
        }

        [Fact]
        public void HardwareCostModel_EstimateArchitectureCost_UsesNodeChannels()
        {
            // Arrange
            var model = new HardwareCostModel<double>();
            var architecture = new Architecture<double>();
            architecture.Operations.Add((1, 0, "conv3x3"));
            architecture.NodeChannels[0] = 32;
            architecture.NodeChannels[1] = 64;

            // Act
            var cost = model.EstimateArchitectureCost(architecture, inputChannels: 16, spatialSize: 14);

            // Assert - should use node channels, not input channels
            Assert.True(cost.Latency > 0.0);
        }

        [Fact]
        public void HardwareCostModel_MeetsConstraints_NullConstraints_ReturnsTrue()
        {
            // Arrange
            var model = new HardwareCostModel<double>();
            var architecture = new Architecture<double>();
            architecture.Operations.Add((1, 0, "conv3x3"));
            var constraints = new HardwareConstraints<double>();

            // Act
            var meets = model.MeetsConstraints(architecture, constraints, inputChannels: 32, spatialSize: 14);

            // Assert - with no constraints set, should always meet
            Assert.True(meets);
        }

        [Fact]
        public void HardwareCostModel_MeetsConstraints_LooseLatencyConstraint_ReturnsTrue()
        {
            // Arrange
            var model = new HardwareCostModel<double>();
            var architecture = new Architecture<double>();
            architecture.Operations.Add((1, 0, "conv3x3"));
            var constraints = new HardwareConstraints<double>
            {
                MaxLatency = 1000000.0 // Very loose constraint
            };

            // Act
            var meets = model.MeetsConstraints(architecture, constraints, inputChannels: 32, spatialSize: 14);

            // Assert
            Assert.True(meets);
        }

        [Fact]
        public void HardwareCostModel_MeetsConstraints_TightLatencyConstraint_ReturnsFalse()
        {
            // Arrange
            var model = new HardwareCostModel<double>();
            var architecture = new Architecture<double>();
            // Add multiple heavy operations
            for (int i = 1; i <= 10; i++)
            {
                architecture.Operations.Add((i, i - 1, "conv5x5"));
            }
            var constraints = new HardwareConstraints<double>
            {
                MaxLatency = 0.0001 // Very tight constraint
            };

            // Act
            var meets = model.MeetsConstraints(architecture, constraints, inputChannels: 256, spatialSize: 56);

            // Assert
            Assert.False(meets);
        }

        [Fact]
        public void HardwareCostModel_MeetsConstraints_TightMemoryConstraint_ReturnsFalse()
        {
            // Arrange
            var model = new HardwareCostModel<double>();
            var architecture = new Architecture<double>();
            for (int i = 1; i <= 10; i++)
            {
                architecture.Operations.Add((i, i - 1, "conv5x5"));
            }
            var constraints = new HardwareConstraints<double>
            {
                MaxMemory = 0.0001 // Very tight constraint
            };

            // Act
            var meets = model.MeetsConstraints(architecture, constraints, inputChannels: 256, spatialSize: 56);

            // Assert
            Assert.False(meets);
        }

        [Fact]
        public void HardwareCostModel_MeetsConstraints_TightEnergyConstraint_ReturnsFalse()
        {
            // Arrange
            var model = new HardwareCostModel<double>();
            var architecture = new Architecture<double>();
            for (int i = 1; i <= 10; i++)
            {
                architecture.Operations.Add((i, i - 1, "conv5x5"));
            }
            var constraints = new HardwareConstraints<double>
            {
                MaxEnergy = 0.0001 // Very tight constraint
            };

            // Act
            var meets = model.MeetsConstraints(architecture, constraints, inputChannels: 256, spatialSize: 56);

            // Assert
            Assert.False(meets);
        }

        [Fact]
        public void HardwareCostModel_MeetsConstraints_MultipleConstraints_AllMustBeMet()
        {
            // Arrange
            var model = new HardwareCostModel<double>();
            var architecture = new Architecture<double>();
            architecture.Operations.Add((1, 0, "conv3x3"));

            // One tight constraint, others loose
            var constraints = new HardwareConstraints<double>
            {
                MaxLatency = 1000000.0,  // Loose
                MaxEnergy = 1000000.0,   // Loose
                MaxMemory = 0.0000001    // Tight
            };

            // Act
            var meets = model.MeetsConstraints(architecture, constraints, inputChannels: 128, spatialSize: 28);

            // Assert - should fail because memory constraint is not met
            Assert.False(meets);
        }

        [Fact]
        public void HardwareCostModel_DifferentOperations_HaveDifferentCosts()
        {
            // Arrange
            var model = new HardwareCostModel<double>();

            // Act
            var identityCost = model.EstimateOperationCost("identity", 64, 64, 14);
            var conv3x3Cost = model.EstimateOperationCost("conv3x3", 64, 64, 14);
            var conv5x5Cost = model.EstimateOperationCost("conv5x5", 64, 64, 14);

            // Assert - more complex operations should have higher costs
            Assert.True(identityCost.Latency < conv3x3Cost.Latency);
            Assert.True(conv3x3Cost.Latency < conv5x5Cost.Latency);
        }

        [Fact]
        public void HardwareCostModel_PoolingOperations_HaveSimilarCosts()
        {
            // Arrange
            var model = new HardwareCostModel<double>();

            // Act
            var maxPoolCost = model.EstimateOperationCost("maxpool3x3", 64, 64, 14);
            var avgPoolCost = model.EstimateOperationCost("avgpool3x3", 64, 64, 14);

            // Assert - pooling operations should have similar costs
            Assert.True(Math.Abs(maxPoolCost.Latency - avgPoolCost.Latency) < 0.01);
        }

        [Fact]
        public void HardwareCostModel_EdgeTPU_FasterThanCPU()
        {
            // Arrange
            var cpuModel = new HardwareCostModel<double>(HardwarePlatform.CPU);
            var edgeModel = new HardwareCostModel<double>(HardwarePlatform.EdgeTPU);

            // Act
            var cpuCost = cpuModel.EstimateOperationCost("conv3x3", 64, 64, 14);
            var edgeCost = edgeModel.EstimateOperationCost("conv3x3", 64, 64, 14);

            // Assert - EdgeTPU should be faster than CPU
            Assert.True(edgeCost.Latency < cpuCost.Latency);
        }

        [Fact]
        public void HardwareCostModel_Float_WorksCorrectly()
        {
            // Arrange & Act
            var model = new HardwareCostModel<float>();
            var cost = model.EstimateOperationCost("conv3x3", 64, 64, 14);

            // Assert
            Assert.True(cost.Latency > 0.0f);
        }

        [Fact]
        public void HardwareCost_DefaultValues()
        {
            // Arrange & Act
            var cost = new HardwareCost<double>();

            // Assert
            Assert.Equal(default(double), cost.Latency);
            Assert.Equal(default(double), cost.Energy);
            Assert.Equal(default(double), cost.Memory);
        }

        [Fact]
        public void HardwareConstraints_DefaultValues()
        {
            // Arrange & Act
            var constraints = new HardwareConstraints<double>();

            // Assert
            Assert.Null(constraints.MaxLatency);
            Assert.Null(constraints.MaxEnergy);
            Assert.Null(constraints.MaxMemory);
        }
    }
}
