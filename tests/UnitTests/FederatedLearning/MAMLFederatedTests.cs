using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Xunit;
using AiDotNet.FederatedLearning.MetaLearning;
using AiDotNet.FederatedLearning.MetaLearning.Models;
using AiDotNet.FederatedLearning.MetaLearning.Parameters;
using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Tests.UnitTests.FederatedLearning
{
    /// <summary>
    /// Unit tests for MAMLFederated class
    /// </summary>
    public class MAMLFederatedTests
    {
        /// <summary>
        /// Mock model for testing that implements required interfaces
        /// </summary>
        private class MockMetaModel : IFullModel<double, Matrix<double>, Vector<double>>,
                                      ICloneable,
                                      IParameterizable<double>,
                                      IPredictiveModel<double, Matrix<double>, Vector<double>>,
                                      IGradientModel<double>
        {
            private Dictionary<string, Vector<double>> _parameters;

            public MockMetaModel()
            {
                _parameters = new Dictionary<string, Vector<double>>
                {
                    ["weights"] = new Vector<double>(new double[] { 1.0, 2.0, 3.0 }),
                    ["bias"] = new Vector<double>(new double[] { 0.5 })
                };
            }

            public Vector<double> Predict(Matrix<double> input)
            {
                var weights = _parameters["weights"];
                var bias = _parameters["bias"][0];
                var predictions = new double[input.Rows];

                for (int i = 0; i < input.Rows; i++)
                {
                    var sum = bias;
                    for (int j = 0; j < Math.Min(input.Columns, weights.Length); j++)
                    {
                        sum += input[i, j] * weights[j];
                    }
                    predictions[i] = sum;
                }

                return new Vector<double>(predictions);
            }

            public Dictionary<string, Vector<double>> ComputeGradients(Matrix<double> data, Vector<double> labels)
            {
                var predictions = Predict(data);
                var gradients = new Dictionary<string, Vector<double>>();

                // Simple gradient computation for testing
                var weightGrads = new double[_parameters["weights"].Length];
                double biasGrad = 0;

                for (int i = 0; i < data.Rows; i++)
                {
                    var error = predictions[i] - labels[i];
                    biasGrad += 2 * error / data.Rows;

                    for (int j = 0; j < Math.Min(data.Columns, weightGrads.Length); j++)
                    {
                        weightGrads[j] += 2 * error * data[i, j] / data.Rows;
                    }
                }

                gradients["weights"] = new Vector<double>(weightGrads);
                gradients["bias"] = new Vector<double>(new double[] { biasGrad });

                return gradients;
            }

            public Dictionary<string, Vector<double>> GetParameters()
            {
                return new Dictionary<string, Vector<double>>(_parameters);
            }

            public void SetParameters(Dictionary<string, Vector<double>> parameters)
            {
                _parameters = new Dictionary<string, Vector<double>>(parameters);
            }

            public object Clone()
            {
                var clone = new MockMetaModel();
                clone.SetParameters(GetParameters());
                return clone;
            }

            // Minimal implementations for other interface members
            public void Train(Matrix<double> input, Vector<double> expectedOutput) { }
            public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Range(0, 3);
            public bool IsFeatureUsed(int featureIndex) => true;
            public void SetActiveFeatureIndices(IEnumerable<int> featureIndices) { }
            public void SaveModel(string path) { }
            public void LoadModel(string path) { }
            public int ParameterCount => _parameters.Sum(p => p.Value.Length);
            public Dictionary<string, double> GetFeatureImportance() => new Dictionary<string, double>();
        }

        [Fact]
        public void Constructor_WithValidModel_Initializes Successfully()
        {
            // Arrange
            var model = new MockMetaModel();
            var parameters = new MAMLParameters
            {
                InnerLearningRate = 0.01,
                OuterLearningRate = 0.001,
                InnerSteps = 5
            };

            // Act
            var maml = new MAMLFederated(model, parameters);

            // Assert
            Assert.NotNull(maml);
            Assert.NotNull(maml.MetaModel);
            Assert.NotNull(maml.Parameters);
            Assert.Equal(0.01, maml.Parameters.InnerLearningRate);
            Assert.Equal(0.001, maml.Parameters.OuterLearningRate);
        }

        [Fact]
        public void Constructor_WithNullModel_ThrowsArgumentNullException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentNullException>(() => new MAMLFederated(null!));
        }

        [Fact]
        public void RegisterClientTask_WithValidTask_RegistersSuccessfully()
        {
            // Arrange
            var model = new MockMetaModel();
            var maml = new MAMLFederated(model);
            var task = new FederatedTask
            {
                TaskId = "task1",
                SupportSet = new Matrix<double>(5, 3),
                SupportLabels = new Vector<double>(5),
                QuerySet = new Matrix<double>(5, 3),
                QueryLabels = new Vector<double>(5)
            };

            // Act
            maml.RegisterClientTask("client1", task);

            // Assert
            Assert.Contains("client1", maml.ClientWeights.Keys);
        }

        [Fact]
        public void RegisterClientTask_WithNullClientId_ThrowsArgumentNullException()
        {
            // Arrange
            var model = new MockMetaModel();
            var maml = new MAMLFederated(model);
            var task = new FederatedTask
            {
                TaskId = "task1",
                SupportSet = new Matrix<double>(5, 3),
                SupportLabels = new Vector<double>(5),
                QuerySet = new Matrix<double>(5, 3),
                QueryLabels = new Vector<double>(5)
            };

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => maml.RegisterClientTask(null!, task));
        }

        [Fact]
        public void RegisterClientTask_WithNullTask_ThrowsArgumentNullException()
        {
            // Arrange
            var model = new MockMetaModel();
            var maml = new MAMLFederated(model);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => maml.RegisterClientTask("client1", null!));
        }

        [Fact]
        public async Task PerformMetaLearningRoundAsync_WithValidClients_CompletesSuccessfully()
        {
            // Arrange
            var model = new MockMetaModel();
            var maml = new MAMLFederated(model, new MAMLParameters
            {
                InnerLearningRate = 0.01,
                OuterLearningRate = 0.001,
                InnerSteps = 2
            });

            // Create test data
            var supportData = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0, 3.0 },
                { 2.0, 3.0, 4.0 },
                { 3.0, 4.0, 5.0 }
            });
            var supportLabels = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var queryData = new Matrix<double>(new double[,]
            {
                { 1.5, 2.5, 3.5 },
                { 2.5, 3.5, 4.5 }
            });
            var queryLabels = new Vector<double>(new double[] { 1.5, 2.5 });

            var task = new FederatedTask
            {
                TaskId = "task1",
                SupportSet = supportData,
                SupportLabels = supportLabels,
                QuerySet = queryData,
                QueryLabels = queryLabels
            };

            maml.RegisterClientTask("client1", task);

            // Act
            var result = await maml.PerformMetaLearningRoundAsync(new List<string> { "client1" });

            // Assert
            Assert.NotNull(result);
            Assert.Equal(0, result.Round);
            Assert.Contains("client1", result.ParticipatingClients);
            Assert.True(result.MetaGradientNorm >= 0);
            Assert.Single(maml.MetaHistory);
        }

        [Fact]
        public async Task PerformMetaLearningRoundAsync_WithNoClients_ThrowsArgumentException()
        {
            // Arrange
            var model = new MockMetaModel();
            var maml = new MAMLFederated(model);

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentException>(() =>
                maml.PerformMetaLearningRoundAsync(new List<string>()));
        }

        [Fact]
        public void ApplyDifferentialPrivacy_WithValidParameters_AppliesNoise()
        {
            // Arrange
            var model = new MockMetaModel();
            var maml = new MAMLFederated(model);
            var parameters = new Dictionary<string, Vector<double>>
            {
                ["weights"] = new Vector<double>(new double[] { 1.0, 2.0, 3.0 }),
                ["bias"] = new Vector<double>(new double[] { 0.5 })
            };

            // Act
            var privatizedParams = maml.ApplyDifferentialPrivacy(parameters, epsilon: 1.0, delta: 0.01);

            // Assert
            Assert.NotNull(privatizedParams);
            Assert.Equal(parameters.Count, privatizedParams.Count);

            // Check that values have changed (noise was added)
            var originalWeights = parameters["weights"];
            var privatizedWeights = privatizedParams["weights"];
            Assert.NotEqual(originalWeights[0], privatizedWeights[0]);
        }

        [Fact]
        public async Task ExportResultsAsync_WithMetaHistory_CreatesFile()
        {
            // Arrange
            var model = new MockMetaModel();
            var maml = new MAMLFederated(model);
            var tempFile = System.IO.Path.GetTempFileName();

            // Add some history
            maml.MetaHistory.Add(new MetaLearningRound
            {
                Round = 0,
                ParticipatingTasks = 1,
                AverageTaskAccuracy = 0.8,
                MetaLoss = 0.5,
                Success = true
            });

            try
            {
                // Act
                await maml.ExportResultsAsync(tempFile);

                // Assert
                Assert.True(System.IO.File.Exists(tempFile));
                var content = await System.IO.File.ReadAllTextAsync(tempFile);
                Assert.Contains("History", content);
                Assert.Contains("FinalAccuracy", content);
            }
            finally
            {
                // Cleanup
                if (System.IO.File.Exists(tempFile))
                {
                    System.IO.File.Delete(tempFile);
                }
            }
        }

        [Theory]
        [InlineData(0.01, 0.001, 5)]
        [InlineData(0.1, 0.01, 10)]
        [InlineData(0.001, 0.0001, 3)]
        public void Parameters_WithDifferentValues_AreSetCorrectly(
            double innerLR, double outerLR, int innerSteps)
        {
            // Arrange
            var model = new MockMetaModel();
            var parameters = new MAMLParameters
            {
                InnerLearningRate = innerLR,
                OuterLearningRate = outerLR,
                InnerSteps = innerSteps
            };

            // Act
            var maml = new MAMLFederated(model, parameters);

            // Assert
            Assert.Equal(innerLR, maml.Parameters.InnerLearningRate);
            Assert.Equal(outerLR, maml.Parameters.OuterLearningRate);
            Assert.Equal(innerSteps, maml.Parameters.InnerSteps);
        }

        [Fact]
        public void AggregateParameters_WithValidUpdates_ReturnsAggregatedParameters()
        {
            // Arrange
            var model = new MockMetaModel();
            var maml = new MAMLFederated(model);

            var clientUpdates = new Dictionary<string, Dictionary<string, Vector<double>>>
            {
                ["client1"] = new Dictionary<string, Vector<double>>
                {
                    ["weights"] = new Vector<double>(new double[] { 1.0, 2.0, 3.0 })
                },
                ["client2"] = new Dictionary<string, Vector<double>>
                {
                    ["weights"] = new Vector<double>(new double[] { 2.0, 3.0, 4.0 })
                }
            };

            maml.SetClientWeight("client1", 1.0);
            maml.SetClientWeight("client2", 1.0);

            // Act
            var aggregated = maml.AggregateParameters(clientUpdates,
                AiDotNet.Enums.FederatedAggregationStrategy.FederatedAveraging);

            // Assert
            Assert.NotNull(aggregated);
            Assert.Contains("weights", aggregated.Keys);
        }

        [Fact]
        public void Constructor_WithSeed_ProducesReproducibleResults()
        {
            // Arrange
            var model1 = new MockMetaModel();
            var model2 = new MockMetaModel();
            var seed = 42;

            // Act
            var maml1 = new MAMLFederated(model1, seed: seed);
            var maml2 = new MAMLFederated(model2, seed: seed);

            var params1 = maml1.ApplyDifferentialPrivacy(
                new Dictionary<string, Vector<double>>
                {
                    ["test"] = new Vector<double>(new double[] { 1.0, 2.0, 3.0 })
                }, 1.0, 0.01);

            var params2 = maml2.ApplyDifferentialPrivacy(
                new Dictionary<string, Vector<double>>
                {
                    ["test"] = new Vector<double>(new double[] { 1.0, 2.0, 3.0 })
                }, 1.0, 0.01);

            // Assert - results should be identical with same seed
            Assert.Equal(params1["test"][0], params2["test"][0]);
            Assert.Equal(params1["test"][1], params2["test"][1]);
            Assert.Equal(params1["test"][2], params2["test"][2]);
        }

        [Fact]
        public void MetaHistory_StartsEmpty_AndGrowsWithRounds()
        {
            // Arrange
            var model = new MockMetaModel();
            var maml = new MAMLFederated(model);

            // Assert initial state
            Assert.Empty(maml.MetaHistory);

            // Act - add history entries
            maml.MetaHistory.Add(new MetaLearningRound { Round = 0 });
            maml.MetaHistory.Add(new MetaLearningRound { Round = 1 });

            // Assert
            Assert.Equal(2, maml.MetaHistory.Count);
            Assert.Equal(0, maml.MetaHistory[0].Round);
            Assert.Equal(1, maml.MetaHistory[1].Round);
        }
    }
}
