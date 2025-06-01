using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.MultimodalAI
{
    /// <summary>
    /// Early fusion multimodal model that concatenates modality features before processing
    /// </summary>
    public class EarlyFusionMultimodal : MultimodalModelBase
    {
        private FeedForwardNeuralNetwork<double> _fusionNetwork;
        private readonly int _hiddenLayerSize;
        private readonly int _numHiddenLayers;
        private readonly double _learningRate;
        private readonly double _dropoutRate;

        /// <summary>
        /// Initializes a new instance of EarlyFusionMultimodal
        /// </summary>
        /// <param name="fusedDimension">Dimension of the fused representation</param>
        /// <param name="hiddenLayerSize">Size of hidden layers in fusion network</param>
        /// <param name="numHiddenLayers">Number of hidden layers</param>
        /// <param name="learningRate">Learning rate for training</param>
        /// <param name="dropoutRate">Dropout rate for regularization</param>
        public EarlyFusionMultimodal(int fusedDimension, int hiddenLayerSize = 256, 
                                    int numHiddenLayers = 2, double learningRate = 0.001,
                                    double dropoutRate = 0.2)
            : base("early_fusion", fusedDimension)
        {
            _hiddenLayerSize = hiddenLayerSize;
            _numHiddenLayers = numHiddenLayers;
            _learningRate = learningRate;
            _dropoutRate = dropoutRate;
        }

        /// <summary>
        /// Processes multimodal input data using early fusion
        /// </summary>
        /// <param name="modalityData">Dictionary mapping modality names to their data</param>
        /// <returns>Fused representation</returns>
        public override Vector<double> ProcessMultimodal(Dictionary<string, object> modalityData)
        {
            ValidateModalityData(modalityData);

            // Encode each modality
            var encodedModalities = new List<Vector<double>>();
            foreach (var kvp in modalityData)
            {
                if (_modalityEncoders.ContainsKey(kvp.Key))
                {
                    var encoded = EncodeModality(kvp.Key, kvp.Value);
                    encodedModalities.Add(encoded);
                }
            }

            // Concatenate all encoded modalities (early fusion)
            var concatenated = ConcatenateVectors(encodedModalities);

            // Initialize fusion network if needed
            if (_fusionNetwork == null)
            {
                InitializeFusionNetwork(concatenated.Dimension);
            }

            // Process through fusion network
            var fused = _fusionNetwork.Forward(concatenated);

            // Project to target dimension if needed
            if (fused.Dimension != _fusedDimension)
            {
                fused = ProjectToTargetDimension(fused, _fusedDimension);
            }

            // Normalize the output
            return NormalizeFused(fused);
        }

        /// <summary>
        /// Trains the early fusion model
        /// </summary>
        /// <param name="inputs">Training inputs (each row is a concatenated feature vector)</param>
        /// <param name="targets">Target outputs</param>
        public override void Train(Matrix<double> inputs, Vector<double> targets)
        {
            if (_fusionNetwork == null)
            {
                InitializeFusionNetwork(inputs.Columns);
            }

            // Training loop (simplified)
            int epochs = 100;
            int batchSize = 32;
            var random = new Random();

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double totalLoss = 0;
                int numBatches = (inputs.Rows + batchSize - 1) / batchSize;

                // Shuffle data
                var indices = Enumerable.Range(0, inputs.Rows).OrderBy(x => random.Next()).ToList();

                for (int batch = 0; batch < numBatches; batch++)
                {
                    int start = batch * batchSize;
                    int end = Math.Min(start + batchSize, inputs.Rows);
                    double batchLoss = 0;

                    for (int i = start; i < end; i++)
                    {
                        int idx = indices[i];
                        var input = inputs.GetRow(idx);
                        var target = new Vector<double>(1) { [0] = targets[idx] };

                        // Forward pass
                        var output = _fusionNetwork.Forward(input);

                        // Calculate loss (MSE for simplicity)
                        var loss = CalculateLoss(output, target);
                        batchLoss += loss;

                        // Backward pass
                        var gradients = CalculateLossGradient(output, target);
                        _fusionNetwork.Backward(gradients);
                    }

                    // Update weights
                    UpdateWeights(_learningRate);
                    totalLoss += batchLoss;
                }

                if (epoch % 10 == 0)
                {
                    Console.WriteLine($"Epoch {epoch}, Loss: {totalLoss / inputs.Rows:F4}");
                }
            }

            _isTrained = true;
        }

        /// <summary>
        /// Makes predictions using the early fusion model
        /// </summary>
        /// <param name="inputs">Input data</param>
        /// <returns>Predictions</returns>
        public override Vector<double> Predict(Matrix<double> inputs)
        {
            if (!_isTrained || _fusionNetwork == null)
                throw new InvalidOperationException("Model must be trained before making predictions");

            var predictions = new Vector<double>(inputs.Rows);

            for (int i = 0; i < inputs.Rows; i++)
            {
                var input = inputs.GetRow(i);
                var output = _fusionNetwork.Forward(input);
                predictions[i] = output[0]; // Assuming single output
            }

            return predictions;
        }

        /// <summary>
        /// Creates a copy of the model
        /// </summary>
        /// <returns>A copy of the model</returns>
        public override IFullModel<double, Dictionary<string, object>, Vector<double>> Clone()
        {
            var clone = new EarlyFusionMultimodal(_fusedDimension, _hiddenLayerSize, 
                                                 _numHiddenLayers, _learningRate, _dropoutRate);

            // Copy encoders
            foreach (var kvp in _modalityEncoders)
            {
                clone.AddModalityEncoder(kvp.Key, kvp.Value);
            }

            // Copy fusion network if exists
            if (_fusionNetwork != null)
            {
                // Would need to implement network cloning
                clone._fusionNetwork = _fusionNetwork;
            }

            clone._isTrained = _isTrained;
            clone.Name = Name;

            return clone;
        }

        /// <summary>
        /// Concatenates multiple vectors into one
        /// </summary>
        private Vector<double> ConcatenateVectors(List<Vector<double>> vectors)
        {
            int totalDimension = vectors.Sum(v => v.Dimension);
            var concatenated = new Vector<double>(totalDimension);

            int offset = 0;
            foreach (var vector in vectors)
            {
                for (int i = 0; i < vector.Dimension; i++)
                {
                    concatenated[offset + i] = vector[i];
                }
                offset += vector.Dimension;
            }

            return concatenated;
        }

        /// <summary>
        /// Initializes the fusion network
        /// </summary>
        private void InitializeFusionNetwork(int inputDimension)
        {
            var layers = new List<int> { inputDimension };

            // Add hidden layers
            for (int i = 0; i < _numHiddenLayers; i++)
            {
                layers.Add(_hiddenLayerSize);
            }

            // Add output layer
            layers.Add(_fusedDimension);

            _fusionNetwork = new FeedForwardNeuralNetwork<double>(layers.ToArray());
        }

        /// <summary>
        /// Calculates loss (MSE)
        /// </summary>
        private double CalculateLoss(Vector<double> output, Vector<double> target)
        {
            double sum = 0;
            for (int i = 0; i < output.Dimension; i++)
            {
                double diff = output[i] - target[i];
                sum += diff * diff;
            }
            return sum / output.Dimension;
        }

        /// <summary>
        /// Calculates loss gradient
        /// </summary>
        private Vector<double> CalculateLossGradient(Vector<double> output, Vector<double> target)
        {
            var gradient = new Vector<double>(output.Dimension);
            for (int i = 0; i < output.Dimension; i++)
            {
                gradient[i] = 2 * (output[i] - target[i]) / output.Dimension;
            }
            return gradient;
        }

        /// <summary>
        /// Updates network weights
        /// </summary>
        private void UpdateWeights(double learningRate)
        {
            // This would be implemented in the neural network class
            // For now, this is a placeholder
        }

        /// <summary>
        /// Gets parameters of the model
        /// </summary>
        public override Dictionary<string, object> GetParameters()
        {
            var parameters = base.GetParameters();
            parameters["HiddenLayerSize"] = _hiddenLayerSize;
            parameters["NumHiddenLayers"] = _numHiddenLayers;
            parameters["LearningRate"] = _learningRate;
            parameters["DropoutRate"] = _dropoutRate;
            return parameters;
        }
    }
}