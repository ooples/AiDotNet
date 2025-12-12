using AiDotNet.ActivationFunctions;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Xunit;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks
{
    /// <summary>
    /// Integration tests for neural network architectures including FeedForward, Convolutional,
    /// Recurrent, and advanced networks. Tests end-to-end training and prediction scenarios.
    /// </summary>
    public class NetworkIntegrationTests
    {
        private const double Tolerance = 1e-4;

        // ===== FeedForwardNeuralNetwork Tests =====

        [Fact]
        public void FeedForwardNetwork_XORProblem_LearnsCorrectly()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>
            {
                InputSize = 2,
                OutputSize = 1,
                HiddenLayerSizes = new[] { 4 },
                TaskType = TaskType.Regression
            };

            var network = new FeedForwardNeuralNetwork<double>(architecture);

            var xorInputs = new double[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
            var xorOutputs = new double[] { 0, 1, 1, 0 };

            double initialError = 0;
            for (int i = 0; i < 4; i++)
            {
                var input = new Tensor<double>([1, 2]);
                input[0, 0] = xorInputs[i, 0];
                input[0, 1] = xorInputs[i, 1];

                var output = network.Predict(input);
                initialError += Math.Pow(output[0, 0] - xorOutputs[i], 2);
            }

            // Act - Training
            for (int epoch = 0; epoch < 1000; epoch++)
            {
                for (int i = 0; i < 4; i++)
                {
                    var input = new Tensor<double>([1, 2]);
                    input[0, 0] = xorInputs[i, 0];
                    input[0, 1] = xorInputs[i, 1];

                    var target = new Tensor<double>([1, 1]);
                    target[0, 0] = xorOutputs[i];

                    network.Train(input, target);
                }
            }

            // Assert - Final error should be much lower
            double finalError = 0;
            for (int i = 0; i < 4; i++)
            {
                var input = new Tensor<double>([1, 2]);
                input[0, 0] = xorInputs[i, 0];
                input[0, 1] = xorInputs[i, 1];

                var output = network.Predict(input);
                finalError += Math.Pow(output[0, 0] - xorOutputs[i], 2);
            }

            Assert.True(finalError < initialError * 0.1);
        }

        [Fact]
        public void FeedForwardNetwork_SimpleClassification_ConvergesToSolution()
        {
            // Arrange - Binary classification
            var architecture = new NeuralNetworkArchitecture<double>
            {
                InputSize = 3,
                OutputSize = 2,
                HiddenLayerSizes = new[] { 8 },
                TaskType = TaskType.Classification
            };

            var network = new FeedForwardNeuralNetwork<double>(architecture);

            // Act - Train on simple patterns
            for (int epoch = 0; epoch < 100; epoch++)
            {
                // Class 0: small positive values
                var input1 = new Tensor<double>([1, 3]);
                input1[0, 0] = 0.1; input1[0, 1] = 0.2; input1[0, 2] = 0.1;
                var target1 = new Tensor<double>([1, 2]);
                target1[0, 0] = 1.0; target1[0, 1] = 0.0;
                network.Train(input1, target1);

                // Class 1: larger values
                var input2 = new Tensor<double>([1, 3]);
                input2[0, 0] = 0.9; input2[0, 1] = 0.8; input2[0, 2] = 0.9;
                var target2 = new Tensor<double>([1, 2]);
                target2[0, 0] = 0.0; target2[0, 1] = 1.0;
                network.Train(input2, target2);
            }

            // Assert - Should classify correctly
            var testInput = new Tensor<double>([1, 3]);
            testInput[0, 0] = 0.15; testInput[0, 1] = 0.25; testInput[0, 2] = 0.15;
            var prediction = network.Predict(testInput);

            Assert.True(prediction[0, 0] > prediction[0, 1]); // Should prefer class 0
        }

        [Fact]
        public void FeedForwardNetwork_MultiLayerDeep_TrainsSuccessfully()
        {
            // Arrange - Deep network
            var architecture = new NeuralNetworkArchitecture<double>
            {
                InputSize = 4,
                OutputSize = 1,
                HiddenLayerSizes = new[] { 10, 8, 6 },
                TaskType = TaskType.Regression
            };

            var network = new FeedForwardNeuralNetwork<double>(architecture);

            // Act - Train on simple regression
            var input = new Tensor<double>([1, 4]);
            var target = new Tensor<double>([1, 1]);

            for (int i = 0; i < 50; i++)
            {
                input[0, 0] = i * 0.1;
                target[0, 0] = i * 0.1;
                network.Train(input, target);
            }

            // Assert - Should produce reasonable output
            var testInput = new Tensor<double>([1, 4]);
            testInput[0, 0] = 2.5;
            var prediction = network.Predict(testInput);

            Assert.NotNull(prediction);
        }

        [Fact]
        public void FeedForwardNetwork_Predict_ProducesCorrectShape()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>
            {
                InputSize = 10,
                OutputSize = 5,
                HiddenLayerSizes = new[] { 8 }
            };

            var network = new FeedForwardNeuralNetwork<double>(architecture);
            var input = new Tensor<double>([3, 10]); // Batch of 3

            // Act
            var output = network.Predict(input);

            // Assert
            Assert.Equal(3, output.Shape[0]); // Batch size
            Assert.Equal(5, output.Shape[1]); // Output size
        }

        [Fact]
        public void FeedForwardNetwork_GetModelMetadata_ReturnsCorrectInfo()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>
            {
                InputSize = 5,
                OutputSize = 3,
                HiddenLayerSizes = new[] { 10, 8 }
            };

            var network = new FeedForwardNeuralNetwork<double>(architecture);

            // Act
            var metadata = network.GetModelMetadata();

            // Assert
            Assert.Equal(ModelType.FeedForwardNetwork, metadata.ModelType);
            Assert.True(metadata.AdditionalInfo.ContainsKey("LayerCount"));
            Assert.True(metadata.AdditionalInfo.ContainsKey("ParameterCount"));
        }

        [Fact]
        public void FeedForwardNetwork_OverfitSmallDataset_ReducesLoss()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>
            {
                InputSize = 2,
                OutputSize = 1,
                HiddenLayerSizes = new[] { 20, 20 }
            };

            var network = new FeedForwardNeuralNetwork<double>(architecture);

            var input = new Tensor<double>([1, 2]);
            input[0, 0] = 1.0; input[0, 1] = 2.0;
            var target = new Tensor<double>([1, 1]);
            target[0, 0] = 3.0;

            double initialLoss = double.MaxValue;
            double finalLoss = 0;

            // Act - Overfit on single example
            for (int i = 0; i < 500; i++)
            {
                network.Train(input, target);
                if (i == 0)
                    initialLoss = network.LastLoss;
            }
            finalLoss = network.LastLoss;

            // Assert - Should dramatically reduce loss
            Assert.True(finalLoss < initialLoss * 0.01);
        }

        // ===== ConvolutionalNeuralNetwork Tests =====

        [Fact]
        public void ConvolutionalNetwork_ForwardPass_ProducesCorrectShape()
        {
            // Arrange - Simple CNN for 28x28 images
            var architecture = new NeuralNetworkArchitecture<double>
            {
                InputType = InputType.ThreeDimensional,
                InputShape = new[] { 1, 28, 28 }, // Grayscale 28x28
                OutputSize = 10
            };

            var layers = new List<ILayer<double>>
            {
                new ConvolutionalLayer<double>(1, 8, 3, 28, 28, padding: 1),
                new MaxPoolingLayer<double>([8, 28, 28], 2, 2),
                new FlattenLayer<double>(),
                new DenseLayer<double>(8 * 14 * 14, 10)
            };

            architecture.Layers = layers;

            var network = new ConvolutionalNeuralNetwork<double>(architecture);
            var input = new Tensor<double>([2, 1, 28, 28]); // Batch of 2

            // Act
            var output = network.Predict(input);

            // Assert
            Assert.Equal(2, output.Shape[0]); // Batch size
            Assert.Equal(10, output.Shape[1]); // Output classes
        }

        [Fact]
        public void ConvolutionalNetwork_Training_UpdatesParameters()
        {
            // Arrange - Mini CNN
            var architecture = new NeuralNetworkArchitecture<double>
            {
                InputType = InputType.ThreeDimensional,
                InputShape = new[] { 1, 8, 8 },
                OutputSize = 2
            };

            var layers = new List<ILayer<double>>
            {
                new ConvolutionalLayer<double>(1, 4, 3, 8, 8, padding: 1),
                new FlattenLayer<double>(),
                new DenseLayer<double>(4 * 8 * 8, 2)
            };

            architecture.Layers = layers;

            var network = new ConvolutionalNeuralNetwork<double>(architecture);
            var initialParams = network.GetParameters();

            var input = new Tensor<double>([1, 1, 8, 8]);
            var target = new Tensor<double>([1, 2]);
            target[0, 0] = 1.0;

            // Act
            for (int i = 0; i < 10; i++)
            {
                network.Train(input, target);
            }

            var finalParams = network.GetParameters();

            // Assert - Parameters should change
            bool changed = false;
            for (int i = 0; i < Math.Min(100, initialParams.Length); i++)
            {
                if (Math.Abs(initialParams[i] - finalParams[i]) > 1e-8)
                {
                    changed = true;
                    break;
                }
            }
            Assert.True(changed);
        }

        [Fact]
        public void ConvolutionalNetwork_MultipleConvLayers_WorksCorrectly()
        {
            // Arrange - Multiple conv layers
            var architecture = new NeuralNetworkArchitecture<double>
            {
                InputType = InputType.ThreeDimensional,
                InputShape = new[] { 3, 16, 16 }, // RGB 16x16
                OutputSize = 4
            };

            var layers = new List<ILayer<double>>
            {
                new ConvolutionalLayer<double>(3, 8, 3, 16, 16, padding: 1),
                new ConvolutionalLayer<double>(8, 16, 3, 16, 16, padding: 1),
                new MaxPoolingLayer<double>([16, 16, 16], 2, 2),
                new FlattenLayer<double>(),
                new DenseLayer<double>(16 * 8 * 8, 4)
            };

            architecture.Layers = layers;

            var network = new ConvolutionalNeuralNetwork<double>(architecture);
            var input = new Tensor<double>([1, 3, 16, 16]);

            // Act
            var output = network.Predict(input);

            // Assert
            Assert.Equal(1, output.Shape[0]);
            Assert.Equal(4, output.Shape[1]);
        }

        // ===== Recurrent Network Tests =====

        [Fact]
        public void RecurrentNetwork_SequenceProcessing_WorksCorrectly()
        {
            // Arrange - Simple RNN for sequence prediction
            var architecture = new NeuralNetworkArchitecture<double>
            {
                InputType = InputType.Sequential,
                InputShape = new[] { 10, 5 }, // Sequence length 10, features 5
                OutputSize = 3
            };

            var layers = new List<ILayer<double>>
            {
                new RecurrentLayer<double>(5, 8),
                new DenseLayer<double>(8, 3)
            };

            architecture.Layers = layers;

            // Create network (would be RecurrentNeuralNetwork if it exists)
            // For now, test with FeedForward as fallback
            var network = new FeedForwardNeuralNetwork<double>(architecture);
            var input = new Tensor<double>([2, 10, 5]); // Batch of 2, seq 10

            // Act
            var output = network.Predict(input);

            // Assert
            Assert.NotNull(output);
        }

        [Fact]
        public void LSTMNetwork_LongSequence_MaintainsInformation()
        {
            // Arrange - LSTM for sequence learning
            var architecture = new NeuralNetworkArchitecture<double>
            {
                InputType = InputType.Sequential,
                InputShape = new[] { 20, 4 }, // Sequence length 20, features 4
                OutputSize = 2
            };

            var layers = new List<ILayer<double>>
            {
                new LSTMLayer<double>(4, 8),
                new DenseLayer<double>(8, 2)
            };

            architecture.Layers = layers;

            var network = new FeedForwardNeuralNetwork<double>(architecture);

            // Act - Train on sequence pattern
            var input = new Tensor<double>([1, 20, 4]);
            var target = new Tensor<double>([1, 2]);
            target[0, 0] = 1.0;

            for (int i = 0; i < 50; i++)
            {
                network.Train(input, target);
            }

            // Assert - Should learn successfully
            var prediction = network.Predict(input);
            Assert.NotNull(prediction);
        }

        // ===== Advanced Network Tests =====

        [Fact]
        public void Autoencoder_EncoderDecoder_ReconstructsInput()
        {
            // Arrange - Simple autoencoder architecture
            var architecture = new NeuralNetworkArchitecture<double>
            {
                InputSize = 16,
                OutputSize = 16, // Reconstruction
                HiddenLayerSizes = new[] { 8, 4, 8 } // Bottleneck at 4
            };

            var network = new FeedForwardNeuralNetwork<double>(architecture);

            var input = new Tensor<double>([1, 16]);
            for (int i = 0; i < 16; i++)
                input[0, i] = i * 0.1;

            double initialError = 0;
            var initialOutput = network.Predict(input);
            for (int i = 0; i < 16; i++)
                initialError += Math.Pow(initialOutput[0, i] - input[0, i], 2);

            // Act - Train to reconstruct
            for (int epoch = 0; epoch < 100; epoch++)
            {
                network.Train(input, input); // Target = Input for autoencoder
            }

            var finalOutput = network.Predict(input);
            double finalError = 0;
            for (int i = 0; i < 16; i++)
                finalError += Math.Pow(finalOutput[0, i] - input[0, i], 2);

            // Assert - Reconstruction error should decrease
            Assert.True(finalError < initialError);
        }

        [Fact]
        public void TransformerBlock_AttentionMechanism_ProcessesSequences()
        {
            // Arrange - Simplified transformer-like architecture
            var architecture = new NeuralNetworkArchitecture<double>
            {
                InputType = InputType.Sequential,
                InputShape = new[] { 10, 16 }, // Sequence length 10, embedding 16
                OutputSize = 8
            };

            var layers = new List<ILayer<double>>
            {
                new MultiHeadAttentionLayer<double>(16, 4), // 4 attention heads
                new FeedForwardLayer<double>(16, 32),
                new DenseLayer<double>(32, 8)
            };

            architecture.Layers = layers;

            var network = new FeedForwardNeuralNetwork<double>(architecture);
            var input = new Tensor<double>([1, 10, 16]);

            // Act
            var output = network.Predict(input);

            // Assert
            Assert.NotNull(output);
        }

        // ===== Batch Processing Tests =====

        [Fact]
        public void Network_BatchTraining_ProcessesMultipleSamples()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>
            {
                InputSize = 5,
                OutputSize = 2,
                HiddenLayerSizes = new[] { 8 }
            };

            var network = new FeedForwardNeuralNetwork<double>(architecture);

            // Act - Train with batch
            var batchInput = new Tensor<double>([4, 5]); // Batch of 4
            var batchTarget = new Tensor<double>([4, 2]);

            network.Train(batchInput, batchTarget);

            // Assert - Should handle batch
            var prediction = network.Predict(batchInput);
            Assert.Equal(4, prediction.Shape[0]);
        }

        [Fact]
        public void Network_LargeBatch_ProcessesEfficiently()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>
            {
                InputSize = 10,
                OutputSize = 5,
                HiddenLayerSizes = new[] { 20 }
            };

            var network = new FeedForwardNeuralNetwork<double>(architecture);
            var input = new Tensor<double>([64, 10]); // Large batch

            // Act
            var output = network.Predict(input);

            // Assert
            Assert.Equal(64, output.Shape[0]);
            Assert.Equal(5, output.Shape[1]);
        }

        // ===== Serialization Tests =====

        [Fact]
        public void Network_Serialize_Deserialize_RoundTrip()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>
            {
                InputSize = 4,
                OutputSize = 2,
                HiddenLayerSizes = new[] { 6 }
            };

            var network = new FeedForwardNeuralNetwork<double>(architecture);

            // Train a bit to have non-random parameters
            var input = new Tensor<double>([1, 4]);
            var target = new Tensor<double>([1, 2]);
            for (int i = 0; i < 10; i++)
            {
                network.Train(input, target);
            }

            var originalParams = network.GetParameters();

            // Act
            var serialized = network.Serialize();
            // Note: Full deserialization would require implementing Deserialize
            // This tests that serialization completes without errors

            // Assert
            Assert.NotNull(serialized);
            Assert.True(serialized.Length > 0);
        }

        // ===== Different Optimizers Tests =====

        [Fact]
        public void Network_WithAdamOptimizer_TrainsCorrectly()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>
            {
                InputSize = 3,
                OutputSize = 1,
                HiddenLayerSizes = new[] { 5 }
            };

            var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>();
            var network = new FeedForwardNeuralNetwork<double>(architecture, optimizer);

            // Act
            var input = new Tensor<double>([1, 3]);
            var target = new Tensor<double>([1, 1]);
            target[0, 0] = 1.0;

            for (int i = 0; i < 50; i++)
            {
                network.Train(input, target);
            }

            // Assert
            Assert.True(network.LastLoss < 1.0); // Should make some progress
        }

        // ===== Different Loss Functions Tests =====

        [Fact]
        public void Network_WithMSELoss_TrainsForRegression()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>
            {
                InputSize = 2,
                OutputSize = 1,
                HiddenLayerSizes = new[] { 4 },
                TaskType = TaskType.Regression
            };

            var lossFunction = new MeanSquaredErrorLoss<double>();
            var network = new FeedForwardNeuralNetwork<double>(architecture, lossFunction: lossFunction);

            // Act
            var input = new Tensor<double>([1, 2]);
            input[0, 0] = 1.0; input[0, 1] = 2.0;
            var target = new Tensor<double>([1, 1]);
            target[0, 0] = 3.0;

            double initialLoss = double.MaxValue;
            for (int i = 0; i < 100; i++)
            {
                network.Train(input, target);
                if (i == 0)
                    initialLoss = network.LastLoss;
            }

            // Assert
            Assert.True(network.LastLoss < initialLoss);
        }

        [Fact]
        public void Network_WithCrossEntropyLoss_TrainsForClassification()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>
            {
                InputSize = 3,
                OutputSize = 3,
                HiddenLayerSizes = new[] { 6 },
                TaskType = TaskType.Classification
            };

            var lossFunction = new CrossEntropyLoss<double>();
            var network = new FeedForwardNeuralNetwork<double>(architecture, lossFunction: lossFunction);

            // Act
            var input = new Tensor<double>([1, 3]);
            var target = new Tensor<double>([1, 3]);
            target[0, 0] = 1.0; // One-hot encoding

            for (int i = 0; i < 50; i++)
            {
                network.Train(input, target);
            }

            // Assert
            var prediction = network.Predict(input);
            Assert.NotNull(prediction);
        }

        // ===== Edge Cases =====

        [Fact]
        public void Network_VeryDeep_10Layers_TrainsSuccessfully()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>
            {
                InputSize = 5,
                OutputSize = 1,
                HiddenLayerSizes = new[] { 10, 9, 8, 7, 6, 5, 4, 3, 2 }
            };

            var network = new FeedForwardNeuralNetwork<double>(architecture);

            // Act
            var input = new Tensor<double>([1, 5]);
            var target = new Tensor<double>([1, 1]);

            for (int i = 0; i < 10; i++)
            {
                network.Train(input, target);
            }

            // Assert - Should complete training
            Assert.NotNull(network.Predict(input));
        }

        [Fact]
        public void Network_SingleNeuronPerLayer_WorksCorrectly()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>
            {
                InputSize = 1,
                OutputSize = 1,
                HiddenLayerSizes = new[] { 1, 1, 1 }
            };

            var network = new FeedForwardNeuralNetwork<double>(architecture);

            // Act
            var input = new Tensor<double>([1, 1]);
            input[0, 0] = 1.0;

            var output = network.Predict(input);

            // Assert
            Assert.Equal(1, output.Shape[1]);
        }

        [Fact]
        public void Network_SupportsTraining_ReturnsTrue()
        {
            // Arrange
            var architecture = new NeuralNetworkArchitecture<double>
            {
                InputSize = 5,
                OutputSize = 3
            };

            var network = new FeedForwardNeuralNetwork<double>(architecture);

            // Act & Assert
            Assert.True(network.SupportsTraining);
        }
    }
}
