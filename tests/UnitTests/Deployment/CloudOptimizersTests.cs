using AiDotNet.Deployment.CloudOptimizers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Deployment
{
    /// <summary>
    /// Comprehensive unit tests for AWS, GCP, and Azure cloud optimizers.
    /// Tests verify dictionary initialization, service configuration, and optimization behavior.
    /// </summary>
    public class CloudOptimizersTests
    {
        #region AWSOptimizer Tests

        [Fact]
        public void AWSOptimizer_Constructor_InitializesServiceConfigs()
        {
            // Arrange & Act
            var optimizer = new AWSOptimizer<double[], double[], object>();

            // Assert
            Assert.NotNull(optimizer);
            Assert.Equal("AWS Optimizer", optimizer.Name);
            Assert.Equal(DeploymentTarget.Cloud, optimizer.Target);
        }

        [Fact]
        public void AWSOptimizer_ServiceConfigs_ContainsAllExpectedServices()
        {
            // Arrange
            var optimizer = new AWSOptimizer<double[], double[], object>();

            // Act - Use reflection to access private ServiceConfigs
            var type = optimizer.GetType();
            var serviceConfigsProperty = type.GetProperty("ServiceConfigs",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var serviceConfigs = serviceConfigsProperty?.GetValue(optimizer) as System.Collections.IDictionary;

            // Assert
            Assert.NotNull(serviceConfigs);
            Assert.Equal(4, serviceConfigs.Count);
            Assert.True(serviceConfigs.Contains("SageMaker"));
            Assert.True(serviceConfigs.Contains("Lambda"));
            Assert.True(serviceConfigs.Contains("EC2"));
            Assert.True(serviceConfigs.Contains("Batch"));
        }

        [Fact]
        public void AWSOptimizer_SageMakerConfig_HasCorrectFormats()
        {
            // Arrange
            var optimizer = new AWSOptimizer<double[], double[], object>();

            // Act - Access SageMaker configuration via reflection
            var type = optimizer.GetType();
            var serviceConfigsProperty = type.GetProperty("ServiceConfigs",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var serviceConfigs = serviceConfigsProperty?.GetValue(optimizer) as System.Collections.IDictionary;
            var sageMakerConfig = serviceConfigs?["SageMaker"];

            // Assert
            Assert.NotNull(sageMakerConfig);
            var formatsProperty = sageMakerConfig.GetType().GetProperty("SupportedFormats");
            var formats = formatsProperty?.GetValue(sageMakerConfig) as string[];

            Assert.NotNull(formats);
            Assert.Contains("TensorFlow", formats);
            Assert.Contains("PyTorch", formats);
            Assert.Contains("MXNet", formats);
            Assert.Contains("XGBoost", formats);
            Assert.DoesNotContain("Tensor<double>Flow", formats); // Verify bug fix
        }

        [Fact]
        public void AWSOptimizer_LambdaConfig_HasCorrectFormats()
        {
            // Arrange
            var optimizer = new AWSOptimizer<double[], double[], object>();

            // Act - Access Lambda configuration via reflection
            var type = optimizer.GetType();
            var serviceConfigsProperty = type.GetProperty("ServiceConfigs",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var serviceConfigs = serviceConfigsProperty?.GetValue(optimizer) as System.Collections.IDictionary;
            var lambdaConfig = serviceConfigs?["Lambda"];

            // Assert
            Assert.NotNull(lambdaConfig);
            var formatsProperty = lambdaConfig.GetType().GetProperty("SupportedFormats");
            var formats = formatsProperty?.GetValue(lambdaConfig) as string[];

            Assert.NotNull(formats);
            Assert.Contains("TensorFlow Lite", formats);
            Assert.Contains("ONNX", formats);
            Assert.DoesNotContain("Tensor<double>Flow Lite", formats); // Verify bug fix
        }

        [Fact]
        public void AWSOptimizer_Configuration_HasCorrectDefaults()
        {
            // Arrange
            var optimizer = new AWSOptimizer<double[], double[], object>();

            // Act
            var config = optimizer.Configuration;

            // Assert
            Assert.NotNull(config);
            Assert.NotNull(config.PlatformSpecificSettings);
            Assert.Equal("us-east-1", config.PlatformSpecificSettings["Region"]);
            Assert.Equal(true, config.PlatformSpecificSettings["EnableElasticInference"]);
            Assert.Equal(true, config.PlatformSpecificSettings["EnableNeuron"]);
            Assert.Equal(false, config.PlatformSpecificSettings["EnableGraviton"]);
        }

        #endregion

        #region GCPOptimizer Tests

        [Fact]
        public void GCPOptimizer_Constructor_InitializesServiceConfigs()
        {
            // Arrange & Act
            var optimizer = new GCPOptimizer<double[], double[], object>();

            // Assert
            Assert.NotNull(optimizer);
            Assert.Equal("GCP Optimizer", optimizer.Name);
            Assert.Equal(DeploymentTarget.Cloud, optimizer.Target);
        }

        [Fact]
        public void GCPOptimizer_ServiceConfigs_ContainsAllExpectedServices()
        {
            // Arrange
            var optimizer = new GCPOptimizer<double[], double[], object>();

            // Act - Use reflection to access private ServiceConfigs
            var type = optimizer.GetType();
            var serviceConfigsProperty = type.GetProperty("ServiceConfigs",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var serviceConfigs = serviceConfigsProperty?.GetValue(optimizer) as System.Collections.IDictionary;

            // Assert
            Assert.NotNull(serviceConfigs);
            Assert.Equal(4, serviceConfigs.Count);
            Assert.True(serviceConfigs.Contains("VertexAI"));
            Assert.True(serviceConfigs.Contains("CloudFunctions"));
            Assert.True(serviceConfigs.Contains("CloudRun"));
            Assert.True(serviceConfigs.Contains("AIOptimizedVMs"));
        }

        [Fact]
        public void GCPOptimizer_VertexAIConfig_HasCorrectFormats()
        {
            // Arrange
            var optimizer = new GCPOptimizer<double[], double[], object>();

            // Act - Access VertexAI configuration via reflection
            var type = optimizer.GetType();
            var serviceConfigsProperty = type.GetProperty("ServiceConfigs",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var serviceConfigs = serviceConfigsProperty?.GetValue(optimizer) as System.Collections.IDictionary;
            var vertexConfig = serviceConfigs?["VertexAI"];

            // Assert
            Assert.NotNull(vertexConfig);
            var formatsProperty = vertexConfig.GetType().GetProperty("SupportedFormats");
            var formats = formatsProperty?.GetValue(vertexConfig) as string[];

            Assert.NotNull(formats);
            Assert.Contains("TensorFlow", formats);
            Assert.Contains("PyTorch", formats);
            Assert.Contains("XGBoost", formats);
            Assert.Contains("Scikit-learn", formats);
            Assert.Contains("ONNX", formats);
            Assert.DoesNotContain("Tensor<double>Flow", formats); // Verify bug fix
        }

        [Fact]
        public void GCPOptimizer_CloudFunctionsConfig_HasCorrectFormats()
        {
            // Arrange
            var optimizer = new GCPOptimizer<double[], double[], object>();

            // Act - Access CloudFunctions configuration via reflection
            var type = optimizer.GetType();
            var serviceConfigsProperty = type.GetProperty("ServiceConfigs",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var serviceConfigs = serviceConfigsProperty?.GetValue(optimizer) as System.Collections.IDictionary;
            var cloudFunctionsConfig = serviceConfigs?["CloudFunctions"];

            // Assert
            Assert.NotNull(cloudFunctionsConfig);
            var formatsProperty = cloudFunctionsConfig.GetType().GetProperty("SupportedFormats");
            var formats = formatsProperty?.GetValue(cloudFunctionsConfig) as string[];

            Assert.NotNull(formats);
            Assert.Contains("TensorFlow Lite", formats);
            Assert.Contains("ONNX", formats);
            Assert.DoesNotContain("Tensor<double>Flow Lite", formats); // Verify bug fix
        }

        [Fact]
        public void GCPOptimizer_Configuration_HasCorrectDefaults()
        {
            // Arrange
            var optimizer = new GCPOptimizer<double[], double[], object>();

            // Act
            var config = optimizer.Configuration;

            // Assert
            Assert.NotNull(config);
            Assert.NotNull(config.PlatformSpecificSettings);
            Assert.Equal("us-central1", config.PlatformSpecificSettings["Region"]);
            Assert.Equal("your-project-id", config.PlatformSpecificSettings["ProjectId"]);
            Assert.Equal(true, config.PlatformSpecificSettings["EnableTPU"]);
            Assert.Equal(true, config.PlatformSpecificSettings["EnableTensorRT"]);
            Assert.Equal(false, config.PlatformSpecificSettings["EnableEdgeTPU"]);
        }

        #endregion

        #region AzureOptimizer Tests

        [Fact]
        public void AzureOptimizer_Constructor_InitializesServiceConfigs()
        {
            // Arrange & Act
            var optimizer = new AzureOptimizer<double[], double[], object>();

            // Assert
            Assert.NotNull(optimizer);
            Assert.Equal("Azure Optimizer", optimizer.Name);
            Assert.Equal(DeploymentTarget.Cloud, optimizer.Target);
        }

        [Fact]
        public void AzureOptimizer_ServiceConfigs_ContainsAllExpectedServices()
        {
            // Arrange
            var optimizer = new AzureOptimizer<double[], double[], object>();

            // Act - Use reflection to access private ServiceConfigs
            var type = optimizer.GetType();
            var serviceConfigsProperty = type.GetProperty("ServiceConfigs",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var serviceConfigs = serviceConfigsProperty?.GetValue(optimizer) as System.Collections.IDictionary;

            // Assert
            Assert.NotNull(serviceConfigs);
            Assert.Equal(4, serviceConfigs.Count);
            Assert.True(serviceConfigs.Contains("MachineLearning"));
            Assert.True(serviceConfigs.Contains("Functions"));
            Assert.True(serviceConfigs.Contains("ContainerInstances"));
            Assert.True(serviceConfigs.Contains("CognitiveServices"));
        }

        [Fact]
        public void AzureOptimizer_MachineLearningConfig_HasCorrectFormats()
        {
            // Arrange
            var optimizer = new AzureOptimizer<double[], double[], object>();

            // Act - Access MachineLearning configuration via reflection
            var type = optimizer.GetType();
            var serviceConfigsProperty = type.GetProperty("ServiceConfigs",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var serviceConfigs = serviceConfigsProperty?.GetValue(optimizer) as System.Collections.IDictionary;
            var mlConfig = serviceConfigs?["MachineLearning"];

            // Assert
            Assert.NotNull(mlConfig);
            var formatsProperty = mlConfig.GetType().GetProperty("SupportedFormats");
            var formats = formatsProperty?.GetValue(mlConfig) as string[];

            Assert.NotNull(formats);
            Assert.Contains("TensorFlow", formats);
            Assert.Contains("PyTorch", formats);
            Assert.Contains("ONNX", formats);
            Assert.Contains("Scikit-learn", formats);
            Assert.DoesNotContain("Tensor<double>Flow", formats); // Verify bug fix
        }

        [Fact]
        public void AzureOptimizer_FunctionsConfig_HasCorrectFormats()
        {
            // Arrange
            var optimizer = new AzureOptimizer<double[], double[], object>();

            // Act - Access Functions configuration via reflection
            var type = optimizer.GetType();
            var serviceConfigsProperty = type.GetProperty("ServiceConfigs",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var serviceConfigs = serviceConfigsProperty?.GetValue(optimizer) as System.Collections.IDictionary;
            var functionsConfig = serviceConfigs?["Functions"];

            // Assert
            Assert.NotNull(functionsConfig);
            var formatsProperty = functionsConfig.GetType().GetProperty("SupportedFormats");
            var formats = formatsProperty?.GetValue(functionsConfig) as string[];

            Assert.NotNull(formats);
            Assert.Contains("ONNX", formats);
            Assert.Contains("TensorFlow Lite", formats);
            Assert.DoesNotContain("Tensor<double>Flow Lite", formats); // Verify bug fix
        }

        [Fact]
        public void AzureOptimizer_Configuration_HasCorrectDefaults()
        {
            // Arrange
            var optimizer = new AzureOptimizer<double[], double[], object>();

            // Act
            var config = optimizer.Configuration;

            // Assert
            Assert.NotNull(config);
            Assert.NotNull(config.PlatformSpecificSettings);
            Assert.Equal("eastus", config.PlatformSpecificSettings["Region"]);
            Assert.Equal(true, config.PlatformSpecificSettings["EnableONNXRuntime"]);
            Assert.Equal(false, config.PlatformSpecificSettings["EnableFPGA"]);
            Assert.Equal(true, config.PlatformSpecificSettings["EnableGPU"]);
            Assert.Equal("your-subscription-id", config.PlatformSpecificSettings["SubscriptionId"]);
            Assert.Equal("ml-resources", config.PlatformSpecificSettings["ResourceGroup"]);
        }

        #endregion

        #region Cross-Platform Consistency Tests

        [Fact]
        public void AllOptimizers_HaveConsistentTarget()
        {
            // Arrange
            var awsOptimizer = new AWSOptimizer<double[], double[], object>();
            var gcpOptimizer = new GCPOptimizer<double[], double[], object>();
            var azureOptimizer = new AzureOptimizer<double[], double[], object>();

            // Assert - All should target Cloud deployment
            Assert.Equal(DeploymentTarget.Cloud, awsOptimizer.Target);
            Assert.Equal(DeploymentTarget.Cloud, gcpOptimizer.Target);
            Assert.Equal(DeploymentTarget.Cloud, azureOptimizer.Target);
        }

        [Fact]
        public void AllOptimizers_HaveNonEmptyNames()
        {
            // Arrange
            var awsOptimizer = new AWSOptimizer<double[], double[], object>();
            var gcpOptimizer = new GCPOptimizer<double[], double[], object>();
            var azureOptimizer = new AzureOptimizer<double[], double[], object>();

            // Assert - All should have descriptive names
            Assert.False(string.IsNullOrEmpty(awsOptimizer.Name));
            Assert.False(string.IsNullOrEmpty(gcpOptimizer.Name));
            Assert.False(string.IsNullOrEmpty(azureOptimizer.Name));
        }

        [Fact]
        public void AllOptimizers_HaveConfiguredSettings()
        {
            // Arrange
            var awsOptimizer = new AWSOptimizer<double[], double[], object>();
            var gcpOptimizer = new GCPOptimizer<double[], double[], object>();
            var azureOptimizer = new AzureOptimizer<double[], double[], object>();

            // Assert - All should have platform-specific settings configured
            Assert.NotEmpty(awsOptimizer.Configuration.PlatformSpecificSettings);
            Assert.NotEmpty(gcpOptimizer.Configuration.PlatformSpecificSettings);
            Assert.NotEmpty(azureOptimizer.Configuration.PlatformSpecificSettings);
        }

        #endregion
    }
}
