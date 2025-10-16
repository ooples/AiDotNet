using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Deployment.CloudOptimizers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace AiDotNetTests.UnitTests.Deployment
{
    /// <summary>
    /// Unit tests for cloud optimizer classes to verify dictionary initialization and service configuration.
    /// Tests the fix for BUG-002: dictionary initialization syntax errors.
    /// </summary>
    [TestClass]
    public class CloudOptimizersTests
    {
        #region AWSOptimizer Tests

        [TestMethod]
        public void AWSOptimizer_Constructor_InitializesServiceConfigs()
        {
            // Arrange & Act
            var optimizer = new AWSOptimizer<double[], double[], Dictionary<string, object>>();

            // Assert
            Assert.IsNotNull(optimizer);
            Assert.AreEqual("AWS Optimizer", optimizer.Name);
            Assert.AreEqual(DeploymentTarget.Cloud, optimizer.Target);
        }

        [TestMethod]
        public void AWSOptimizer_ServiceConfigs_ContainsExpectedServices()
        {
            // Arrange
            var optimizer = new AWSOptimizer<double[], double[], Dictionary<string, object>>();
            var expectedServices = new[] { "SageMaker", "Lambda", "EC2", "Batch" };

            // Act - Access through reflection since ServiceConfigs is private
            var serviceConfigsProperty = optimizer.GetType().GetProperty("ServiceConfigs",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var serviceConfigs = serviceConfigsProperty?.GetValue(optimizer) as dynamic;

            // Assert
            Assert.IsNotNull(serviceConfigs, "ServiceConfigs should be initialized");

            foreach (var service in expectedServices)
            {
                Assert.IsTrue(((IDictionary<string, object>)serviceConfigs).ContainsKey(service),
                    $"ServiceConfigs should contain {service}");
            }
        }

        [TestMethod]
        public void AWSOptimizer_SageMakerConfig_HasCorrectProperties()
        {
            // Arrange
            var optimizer = new AWSOptimizer<double[], double[], Dictionary<string, object>>();

            // Act - Access through reflection
            var serviceConfigsProperty = optimizer.GetType().GetProperty("ServiceConfigs",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var serviceConfigs = serviceConfigsProperty?.GetValue(optimizer) as dynamic;
            var sageMakerConfig = ((IDictionary<string, object>)serviceConfigs)["SageMaker"];

            // Assert
            Assert.IsNotNull(sageMakerConfig);
            var serviceNameProp = sageMakerConfig.GetType().GetProperty("ServiceName");
            var serviceName = serviceNameProp?.GetValue(sageMakerConfig) as string;
            Assert.AreEqual("Amazon SageMaker", serviceName);
        }

        [TestMethod]
        public void AWSOptimizer_LambdaConfig_HasCorrectProperties()
        {
            // Arrange
            var optimizer = new AWSOptimizer<double[], double[], Dictionary<string, object>>();

            // Act - Access through reflection
            var serviceConfigsProperty = optimizer.GetType().GetProperty("ServiceConfigs",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var serviceConfigs = serviceConfigsProperty?.GetValue(optimizer) as dynamic;
            var lambdaConfig = ((IDictionary<string, object>)serviceConfigs)["Lambda"];

            // Assert
            Assert.IsNotNull(lambdaConfig);
            var serviceNameProp = lambdaConfig.GetType().GetProperty("ServiceName");
            var serviceName = serviceNameProp?.GetValue(lambdaConfig) as string;
            Assert.AreEqual("AWS Lambda", serviceName);
        }

        #endregion

        #region GCPOptimizer Tests

        [TestMethod]
        public void GCPOptimizer_Constructor_InitializesServiceConfigs()
        {
            // Arrange & Act
            var optimizer = new GCPOptimizer<double[], double[], Dictionary<string, object>>();

            // Assert
            Assert.IsNotNull(optimizer);
            Assert.AreEqual("GCP Optimizer", optimizer.Name);
            Assert.AreEqual(DeploymentTarget.Cloud, optimizer.Target);
        }

        [TestMethod]
        public void GCPOptimizer_ServiceConfigs_ContainsExpectedServices()
        {
            // Arrange
            var optimizer = new GCPOptimizer<double[], double[], Dictionary<string, object>>();
            var expectedServices = new[] { "VertexAI", "CloudFunctions", "CloudRun", "AIOptimizedVMs" };

            // Act - Access through reflection
            var serviceConfigsProperty = optimizer.GetType().GetProperty("ServiceConfigs",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var serviceConfigs = serviceConfigsProperty?.GetValue(optimizer) as dynamic;

            // Assert
            Assert.IsNotNull(serviceConfigs, "ServiceConfigs should be initialized");

            foreach (var service in expectedServices)
            {
                Assert.IsTrue(((IDictionary<string, object>)serviceConfigs).ContainsKey(service),
                    $"ServiceConfigs should contain {service}");
            }
        }

        [TestMethod]
        public void GCPOptimizer_VertexAIConfig_HasCorrectProperties()
        {
            // Arrange
            var optimizer = new GCPOptimizer<double[], double[], Dictionary<string, object>>();

            // Act - Access through reflection
            var serviceConfigsProperty = optimizer.GetType().GetProperty("ServiceConfigs",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var serviceConfigs = serviceConfigsProperty?.GetValue(optimizer) as dynamic;
            var vertexAIConfig = ((IDictionary<string, object>)serviceConfigs)["VertexAI"];

            // Assert
            Assert.IsNotNull(vertexAIConfig);
            var serviceNameProp = vertexAIConfig.GetType().GetProperty("ServiceName");
            var serviceName = serviceNameProp?.GetValue(vertexAIConfig) as string;
            Assert.AreEqual("Vertex AI", serviceName);
        }

        [TestMethod]
        public void GCPOptimizer_CloudFunctionsConfig_HasCorrectProperties()
        {
            // Arrange
            var optimizer = new GCPOptimizer<double[], double[], Dictionary<string, object>>();

            // Act - Access through reflection
            var serviceConfigsProperty = optimizer.GetType().GetProperty("ServiceConfigs",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var serviceConfigs = serviceConfigsProperty?.GetValue(optimizer) as dynamic;
            var cloudFunctionsConfig = ((IDictionary<string, object>)serviceConfigs)["CloudFunctions"];

            // Assert
            Assert.IsNotNull(cloudFunctionsConfig);
            var serviceNameProp = cloudFunctionsConfig.GetType().GetProperty("ServiceName");
            var serviceName = serviceNameProp?.GetValue(cloudFunctionsConfig) as string;
            Assert.AreEqual("Cloud Functions", serviceName);
        }

        #endregion

        #region AzureOptimizer Tests

        [TestMethod]
        public void AzureOptimizer_Constructor_InitializesServiceConfigs()
        {
            // Arrange & Act
            var optimizer = new AzureOptimizer<double[], double[], Dictionary<string, object>>();

            // Assert
            Assert.IsNotNull(optimizer);
            Assert.AreEqual("Azure Optimizer", optimizer.Name);
            Assert.AreEqual(DeploymentTarget.Cloud, optimizer.Target);
        }

        [TestMethod]
        public void AzureOptimizer_ServiceConfigs_ContainsExpectedServices()
        {
            // Arrange
            var optimizer = new AzureOptimizer<double[], double[], Dictionary<string, object>>();
            var expectedServices = new[] { "MachineLearning", "Functions", "ContainerInstances", "CognitiveServices" };

            // Act - Access through reflection
            var serviceConfigsProperty = optimizer.GetType().GetProperty("ServiceConfigs",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var serviceConfigs = serviceConfigsProperty?.GetValue(optimizer) as dynamic;

            // Assert
            Assert.IsNotNull(serviceConfigs, "ServiceConfigs should be initialized");

            foreach (var service in expectedServices)
            {
                Assert.IsTrue(((IDictionary<string, object>)serviceConfigs).ContainsKey(service),
                    $"ServiceConfigs should contain {service}");
            }
        }

        [TestMethod]
        public void AzureOptimizer_MachineLearningConfig_HasCorrectProperties()
        {
            // Arrange
            var optimizer = new AzureOptimizer<double[], double[], Dictionary<string, object>>();

            // Act - Access through reflection
            var serviceConfigsProperty = optimizer.GetType().GetProperty("ServiceConfigs",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var serviceConfigs = serviceConfigsProperty?.GetValue(optimizer) as dynamic;
            var mlConfig = ((IDictionary<string, object>)serviceConfigs)["MachineLearning"];

            // Assert
            Assert.IsNotNull(mlConfig);
            var serviceNameProp = mlConfig.GetType().GetProperty("ServiceName");
            var serviceName = serviceNameProp?.GetValue(mlConfig) as string;
            Assert.AreEqual("Azure Machine Learning", serviceName);
        }

        [TestMethod]
        public void AzureOptimizer_FunctionsConfig_HasCorrectProperties()
        {
            // Arrange
            var optimizer = new AzureOptimizer<double[], double[], Dictionary<string, object>>();

            // Act - Access through reflection
            var serviceConfigsProperty = optimizer.GetType().GetProperty("ServiceConfigs",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var serviceConfigs = serviceConfigsProperty?.GetValue(optimizer) as dynamic;
            var functionsConfig = ((IDictionary<string, object>)serviceConfigs)["Functions"];

            // Assert
            Assert.IsNotNull(functionsConfig);
            var serviceNameProp = functionsConfig.GetType().GetProperty("ServiceName");
            var serviceName = serviceNameProp?.GetValue(functionsConfig) as string;
            Assert.AreEqual("Azure Functions", serviceName);
        }

        #endregion

        #region Dictionary Initialization Tests

        [TestMethod]
        public void AllOptimizers_ServiceConfigs_CountMatchesExpected()
        {
            // Arrange
            var awsOptimizer = new AWSOptimizer<double[], double[], Dictionary<string, object>>();
            var gcpOptimizer = new GCPOptimizer<double[], double[], Dictionary<string, object>>();
            var azureOptimizer = new AzureOptimizer<double[], double[], Dictionary<string, object>>();

            // Act - Access through reflection
            var awsConfigs = GetServiceConfigsCount(awsOptimizer);
            var gcpConfigs = GetServiceConfigsCount(gcpOptimizer);
            var azureConfigs = GetServiceConfigsCount(azureOptimizer);

            // Assert
            Assert.AreEqual(4, awsConfigs, "AWS should have 4 service configurations");
            Assert.AreEqual(4, gcpConfigs, "GCP should have 4 service configurations");
            Assert.AreEqual(4, azureConfigs, "Azure should have 4 service configurations");
        }

        private int GetServiceConfigsCount(object optimizer)
        {
            var serviceConfigsProperty = optimizer.GetType().GetProperty("ServiceConfigs",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var serviceConfigs = serviceConfigsProperty?.GetValue(optimizer) as dynamic;
            return ((IDictionary<string, object>)serviceConfigs).Count;
        }

        #endregion
    }
}
