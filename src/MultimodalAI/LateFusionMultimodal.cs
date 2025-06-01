using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.MultimodalAI
{
    /// <summary>
    /// Late fusion multimodal model that processes each modality separately before combining
    /// </summary>
    public class LateFusionMultimodal : MultimodalModelBase
    {
        private readonly Dictionary<string, FeedForwardNeuralNetwork<double>> _modalityNetworks;
        private FeedForwardNeuralNetwork<double> _fusionNetwork;
        private readonly int _modalityHiddenSize;
        private readonly int _fusionHiddenSize;
        private readonly double _learningRate;
        private readonly string _aggregationMethod;

        /// <summary>
        /// Initializes a new instance of LateFusionMultimodal
        /// </summary>
        /// <param name="fusedDimension">Dimension of the fused representation</param>
        /// <param name="modalityHiddenSize">Hidden size for modality-specific networks</param>
        /// <param name="fusionHiddenSize">Hidden size for fusion network</param>
        /// <param name="learningRate">Learning rate for training</param>
        /// <param name="aggregationMethod">Method for aggregating modality outputs (mean, max, weighted)</param>
        public LateFusionMultimodal(int fusedDimension, int modalityHiddenSize = 128,
                                  int fusionHiddenSize = 256, double learningRate = 0.001,
                                  string aggregationMethod = "weighted")
            : base("late_fusion", fusedDimension)
        {
            _modalityNetworks = new Dictionary<string, FeedForwardNeuralNetwork<double>>();
            _modalityHiddenSize = modalityHiddenSize;
            _fusionHiddenSize = fusionHiddenSize;
            _learningRate = learningRate;
            _aggregationMethod = aggregationMethod;
        }

        /// <summary>
        /// Adds a modality encoder and creates a corresponding network
        /// </summary>
        public override void AddModalityEncoder(string modalityName, IModalityEncoder encoder)
        {
            base.AddModalityEncoder(modalityName, encoder);
            
            // Create a modality-specific network
            var network = new FeedForwardNeuralNetwork<double>(new[] 
            { 
                encoder.OutputDimension, 
                _modalityHiddenSize, 
                _modalityHiddenSize / 2 
            });
            
            _modalityNetworks[modalityName] = network;
        }

        /// <summary>
        /// Processes multimodal input data using late fusion
        /// </summary>
        /// <param name="modalityData">Dictionary mapping modality names to their data</param>
        /// <returns>Fused representation</returns>
        public override Vector<double> ProcessMultimodal(Dictionary<string, object> modalityData)
        {
            ValidateModalityData(modalityData);

            var modalityOutputs = new Dictionary<string, Vector<double>>();
            var modalityWeights = new Dictionary<string, double>();

            // Process each modality independently
            foreach (var kvp in modalityData)
            {
                if (_modalityEncoders.ContainsKey(kvp.Key))
                {
                    // Encode modality
                    var encoded = EncodeModality(kvp.Key, kvp.Value);

                    // Process through modality-specific network
                    if (_modalityNetworks.ContainsKey(kvp.Key))
                    {
                        var output = _modalityNetworks[kvp.Key].Forward(encoded);
                        modalityOutputs[kvp.Key] = output;
                        
                        // Calculate modality confidence/weight
                        modalityWeights[kvp.Key] = CalculateModalityWeight(output);
                    }
                }
            }

            // Aggregate modality outputs
            Vector<double> aggregated = AggregateModalityOutputs(modalityOutputs, modalityWeights);

            // Initialize fusion network if needed
            if (_fusionNetwork == null)
            {
                InitializeFusionNetwork(aggregated.Dimension);
            }

            // Final fusion processing
            var fused = _fusionNetwork.Forward(aggregated);

            // Project to target dimension if needed
            if (fused.Dimension != _fusedDimension)
            {
                fused = ProjectToTargetDimension(fused, _fusedDimension);
            }

            return NormalizeFused(fused);
        }

        /// <summary>
        /// Trains the late fusion model
        /// </summary>
        public override void Train(Matrix<double> inputs, Vector<double> targets)
        {
            // Note: This is a simplified training approach
            // In practice, would need separate training data for each modality

            int epochs = 100;
            var random = new Random();

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double totalLoss = 0;

                for (int i = 0; i < inputs.Rows; i++)
                {
                    var input = inputs.GetRow(i);
                    var target = new Vector<double>(1) { [0] = targets[i] };

                    // Split input by modality (simplified - assumes equal split)
                    var modalityInputs = SplitInputByModality(input);

                    // Forward pass through modality networks
                    var modalityOutputs = new Dictionary<string, Vector<double>>();
                    foreach (var kvp in modalityInputs)
                    {
                        if (_modalityNetworks.ContainsKey(kvp.Key))
                        {
                            modalityOutputs[kvp.Key] = _modalityNetworks[kvp.Key].Forward(kvp.Value);
                        }
                    }

                    // Aggregate and forward through fusion network
                    var aggregated = AggregateModalityOutputs(modalityOutputs, null);
                    
                    if (_fusionNetwork == null)
                    {
                        InitializeFusionNetwork(aggregated.Dimension);
                    }
                    
                    var output = _fusionNetwork.Forward(aggregated);

                    // Calculate loss
                    var loss = CalculateLoss(output, target);
                    totalLoss += loss;

                    // Backward pass (simplified)
                    var gradient = CalculateLossGradient(output, target);
                    
                    // Backpropagate through fusion network
                    _fusionNetwork.Backward(gradient);
                    
                    // Backpropagate through modality networks
                    BackpropagateToModalityNetworks(gradient, modalityOutputs);
                }

                // Update weights
                UpdateAllNetworks(_learningRate);

                if (epoch % 10 == 0)
                {
                    Console.WriteLine($"Epoch {epoch}, Loss: {totalLoss / inputs.Rows:F4}");
                }
            }

            _isTrained = true;
        }

        /// <summary>
        /// Creates a copy of the model
        /// </summary>
        public override IFullModel<double, Dictionary<string, object>, Vector<double>> Clone()
        {
            var clone = new LateFusionMultimodal(_fusedDimension, _modalityHiddenSize,
                                               _fusionHiddenSize, _learningRate, _aggregationMethod);

            // Copy encoders and networks
            foreach (var kvp in _modalityEncoders)
            {
                clone.AddModalityEncoder(kvp.Key, kvp.Value);
            }

            clone._isTrained = _isTrained;
            clone.Name = Name;

            return clone;
        }

        /// <summary>
        /// Aggregates modality outputs based on the specified method
        /// </summary>
        private Vector<double> AggregateModalityOutputs(Dictionary<string, Vector<double>> outputs, Dictionary<string, double> weights)
        {
            if (outputs.Count == 0)
                throw new ArgumentException("No modality outputs to aggregate");

            int dimension = outputs.First().Value.Dimension;
            var aggregated = new Vector<double>(dimension);

            switch (_aggregationMethod.ToLower())
            {
                case "mean":
                    // Simple mean aggregation
                    foreach (var output in outputs.Values)
                    {
                        for (int i = 0; i < dimension; i++)
                        {
                            aggregated[i] += output[i];
                        }
                    }
                    aggregated = aggregated / outputs.Count;
                    break;

                case "max":
                    // Max pooling aggregation
                    for (int i = 0; i < dimension; i++)
                    {
                        aggregated[i] = outputs.Values.Max(v => v[i]);
                    }
                    break;

                case "weighted":
                    // Weighted aggregation
                    if (weights == null || weights.Count == 0)
                    {
                        // Equal weights if not provided
                        weights = outputs.ToDictionary(kvp => kvp.Key, kvp => 1.0 / outputs.Count);
                    }

                    // Normalize weights
                    double totalWeight = weights.Values.Sum();
                    
                    foreach (var kvp in outputs)
                    {
                        double weight = weights.ContainsKey(kvp.Key) ? weights[kvp.Key] / totalWeight : 0;
                        for (int i = 0; i < dimension; i++)
                        {
                            aggregated[i] += weight * kvp.Value[i];
                        }
                    }
                    break;

                case "concat":
                    // Concatenation (results in larger dimension)
                    var allValues = new List<double>();
                    foreach (var output in outputs.Values)
                    {
                        for (int i = 0; i < output.Dimension; i++)
                        {
                            allValues.Add(output[i]);
                        }
                    }
                    aggregated = new Vector<double>(allValues.ToArray());
                    break;

                default:
                    throw new ArgumentException($"Unknown aggregation method: {_aggregationMethod}");
            }

            return aggregated;
        }

        /// <summary>
        /// Calculates weight/confidence for a modality output
        /// </summary>
        private double CalculateModalityWeight(Vector<double> output)
        {
            // Simple confidence based on output magnitude
            // In practice, could use learned attention weights
            return output.Magnitude();
        }

        /// <summary>
        /// Initializes the fusion network
        /// </summary>
        private void InitializeFusionNetwork(int inputDimension)
        {
            _fusionNetwork = new FeedForwardNeuralNetwork<double>(new[]
            {
                inputDimension,
                _fusionHiddenSize,
                _fusionHiddenSize / 2,
                _fusedDimension
            });
        }

        /// <summary>
        /// Splits input vector by modality (simplified)
        /// </summary>
        private Dictionary<string, Vector<double>> SplitInputByModality(Vector<double> input)
        {
            var result = new Dictionary<string, Vector<double>>();
            var modalities = _modalityEncoders.Keys.ToList();
            
            if (modalities.Count == 0)
                return result;

            int dimensionPerModality = input.Dimension / modalities.Count;
            
            for (int i = 0; i < modalities.Count; i++)
            {
                int start = i * dimensionPerModality;
                int end = (i == modalities.Count - 1) ? input.Dimension : (i + 1) * dimensionPerModality;
                
                var modalityInput = new Vector<double>(end - start);
                for (int j = 0; j < modalityInput.Dimension; j++)
                {
                    modalityInput[j] = input[start + j];
                }
                
                result[modalities[i]] = modalityInput;
            }

            return result;
        }

        /// <summary>
        /// Backpropagates gradients to modality networks
        /// </summary>
        private void BackpropagateToModalityNetworks(Vector<double> fusionGradient, Dictionary<string, Vector<double>> modalityOutputs)
        {
            // Simplified backpropagation
            // In practice, would need to properly compute gradients through aggregation
            foreach (var kvp in modalityOutputs)
            {
                if (_modalityNetworks.ContainsKey(kvp.Key))
                {
                    // Create modality-specific gradient (simplified)
                    var modalityGradient = new Vector<double>(kvp.Value.Dimension);
                    for (int i = 0; i < Math.Min(fusionGradient.Dimension, modalityGradient.Dimension); i++)
                    {
                        modalityGradient[i] = fusionGradient[i] / modalityOutputs.Count;
                    }
                    
                    _modalityNetworks[kvp.Key].Backward(modalityGradient);
                }
            }
        }

        /// <summary>
        /// Updates weights for all networks
        /// </summary>
        private void UpdateAllNetworks(double learningRate)
        {
            // Update modality networks
            foreach (var network in _modalityNetworks.Values)
            {
                // This would be implemented in the neural network class
            }

            // Update fusion network
            if (_fusionNetwork != null)
            {
                // This would be implemented in the neural network class
            }
        }

        /// <summary>
        /// Calculates loss (MSE)
        /// </summary>
        private double CalculateLoss(Vector<double> output, Vector<double> target)
        {
            double sum = 0;
            for (int i = 0; i < Math.Min(output.Dimension, target.Dimension); i++)
            {
                double diff = output[i] - target[i];
                sum += diff * diff;
            }
            return sum / Math.Min(output.Dimension, target.Dimension);
        }

        /// <summary>
        /// Calculates loss gradient
        /// </summary>
        private Vector<double> CalculateLossGradient(Vector<double> output, Vector<double> target)
        {
            var gradient = new Vector<double>(output.Dimension);
            for (int i = 0; i < Math.Min(output.Dimension, target.Dimension); i++)
            {
                gradient[i] = 2 * (output[i] - target[i]) / output.Dimension;
            }
            return gradient;
        }

        /// <summary>
        /// Gets parameters of the model
        /// </summary>
        public override Dictionary<string, object> GetParameters()
        {
            var parameters = base.GetParameters();
            parameters["ModalityHiddenSize"] = _modalityHiddenSize;
            parameters["FusionHiddenSize"] = _fusionHiddenSize;
            parameters["LearningRate"] = _learningRate;
            parameters["AggregationMethod"] = _aggregationMethod;
            parameters["NumModalityNetworks"] = _modalityNetworks.Count;
            return parameters;
        }
    }
}