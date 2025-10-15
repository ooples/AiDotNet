using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ActivationFunctions;
using AiDotNet.Statistics;
using AiDotNet.Extensions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Threading.Tasks;

namespace AiDotNet.MultimodalAI
{
    /// <summary>
    /// Early fusion multimodal model that concatenates modality features before processing
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    /// <remarks>
    /// This model implements early fusion strategy where features from different modalities
    /// are concatenated at the input level before being processed by a shared neural network.
    /// This approach allows the model to learn cross-modal interactions from the beginning.
    /// </remarks>
    [Serializable]
    public class EarlyFusionMultimodal<T> : MultimodalModelBase<T>, IDisposable
    {
        private NeuralNetwork<T>? _fusionNetwork;
        private readonly int _hiddenLayerSize;
        private readonly int _numHiddenLayers;
        private readonly T _learningRate;
        private readonly T _dropoutRate;
        private readonly Random _random;
        private bool _disposed;
        private readonly object _lockObject = new object();
        private readonly INumericOperations<T> _ops;
        
        // Training history
        private readonly List<T> _trainingLosses = new List<T>();
        private readonly List<T> _validationLosses = new List<T>();

        /// <summary>
        /// Initializes a new instance of EarlyFusionMultimodal
        /// </summary>
        /// <param name="fusedDimension">Dimension of the fused representation</param>
        /// <param name="hiddenLayerSize">Size of hidden layers in fusion network</param>
        /// <param name="numHiddenLayers">Number of hidden layers</param>
        /// <param name="learningRate">Learning rate for training</param>
        /// <param name="dropoutRate">Dropout rate for regularization</param>
        /// <param name="randomSeed">Random seed for reproducibility</param>
        /// <param name="numericOps">Numeric operations for type T</param>
        public EarlyFusionMultimodal(int fusedDimension, int hiddenLayerSize = 256, 
                                    int numHiddenLayers = 2, double learningRate = 0.001,
                                    double dropoutRate = 0.2, int? randomSeed = null,
                                    INumericOperations<T>? numericOps = null)
            : base("early_fusion", fusedDimension, numericOps ?? MathHelper.GetNumericOperations<T>())
        {
            if (fusedDimension <= 0)
                throw new ArgumentException("Fused dimension must be positive", nameof(fusedDimension));
            if (hiddenLayerSize <= 0)
                throw new ArgumentException("Hidden layer size must be positive", nameof(hiddenLayerSize));
            if (numHiddenLayers < 0)
                throw new ArgumentException("Number of hidden layers must be non-negative", nameof(numHiddenLayers));
            if (learningRate <= 0 || learningRate > 1)
                throw new ArgumentException("Learning rate must be in (0, 1]", nameof(learningRate));
            if (dropoutRate < 0 || dropoutRate >= 1)
                throw new ArgumentException("Dropout rate must be in [0, 1)", nameof(dropoutRate));

            _ops = numericOps ?? MathHelper.GetNumericOperations<T>();
            _hiddenLayerSize = hiddenLayerSize;
            _numHiddenLayers = numHiddenLayers;
            _learningRate = _ops.FromDouble(learningRate);
            _dropoutRate = _ops.FromDouble(dropoutRate);
            _random = randomSeed.HasValue ? new Random(randomSeed.Value) : new Random();
        }

        /// <summary>
        /// Processes multimodal input data using early fusion
        /// </summary>
        /// <param name="modalityData">Dictionary mapping modality names to their data</param>
        /// <returns>Fused representation</returns>
        public override Vector<T> ProcessMultimodal(Dictionary<string, object> modalityData)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(EarlyFusionMultimodal<T>));

            ValidateModalityData(modalityData);

            lock (_lockObject)
            {
                try
                {
                    // Encode each modality
                    var encodedModalities = new List<Vector<T>>();
                    var encodingTasks = new List<Task<(string, Vector<T>)>>();

                    // Parallel encoding for better performance
                    foreach (var kvp in modalityData)
                    {
                        if (_modalityEncoders.ContainsKey(kvp.Key))
                        {
                            var modalityName = kvp.Key;
                            var data = kvp.Value;
                            
                            var task = Task.Run(() => 
                            {
                                var encoded = EncodeModality(modalityName, data);
                                return (modalityName, encoded);
                            });
                            
                            encodingTasks.Add(task);
                        }
                    }

                    // Wait for all encodings to complete
                    Task.WaitAll(encodingTasks.ToArray());

                    // Collect results in consistent order
                    var orderedModalities = modalityData.Keys.OrderBy(k => k).ToList();
                    var encodingResults = new Dictionary<string, Vector<T>>();
                    foreach (var task in encodingTasks)
                    {
                        var result = task.Result;
                        encodingResults[result.Item1] = result.Item2;
                    }
                    
                    foreach (var modality in orderedModalities)
                    {
                        if (encodingResults.ContainsKey(modality))
                        {
                            encodedModalities.Add(encodingResults[modality]);
                        }
                    }

                    if (encodedModalities.Count == 0)
                        throw new InvalidOperationException("No modalities were successfully encoded");

                    // Concatenate all encoded modalities (early fusion)
                    var concatenated = ConcatenateVectors(encodedModalities);

                    // Initialize fusion network if needed
                    if (_fusionNetwork == null && _modalityEncoders.Count > 0)
                    {
                        InitializeFusionNetwork(concatenated.Length);
                    }

                    // Process through fusion network
                    var output = ProcessThroughNetwork(concatenated);

                    // Project to target dimension if needed
                    if (output.Length != _fusedDimension)
                    {
                        output = ProjectToTargetDimension(output, _fusedDimension);
                    }

                    // Normalize the output
                    return NormalizeFused(output);
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Error processing multimodal data: {ex.Message}", ex);
                }
            }
        }

        /// <summary>
        /// Processes input through the neural network
        /// </summary>
        /// <param name="input">Input vector</param>
        /// <returns>Output vector</returns>
        private Vector<T> ProcessThroughNetwork(Vector<T> input)
        {
            if (_fusionNetwork == null)
                throw new InvalidOperationException("Fusion network not initialized");

            // Convert to tensor for neural network
            var inputTensor = new Tensor<T>(new[] { input.Length }, input);
            var outputTensor = _fusionNetwork.Predict(inputTensor);
            
            // Extract the vector from the output tensor
            var outputArray = outputTensor.ToArray();
            var output = new Vector<T>(outputArray.Length);
            for (int i = 0; i < outputArray.Length; i++)
            {
                output[i] = outputArray[i];
            }

            return output;
        }

        /// <summary>
        /// Trains the early fusion model
        /// </summary>
        /// <param name="inputs">Training inputs (each row is a concatenated feature vector)</param>
        /// <param name="targets">Target outputs</param>
        public override void Train(Matrix<T> inputs, Vector<T> targets)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(EarlyFusionMultimodal<T>));

            if (inputs == null)
                throw new ArgumentNullException(nameof(inputs));
            if (targets == null)
                throw new ArgumentNullException(nameof(targets));
            if (inputs.Rows != targets.Length)
                throw new ArgumentException("Number of input samples must match number of targets");
            if (inputs.Rows == 0)
                throw new ArgumentException("Training data cannot be empty");

            lock (_lockObject)
            {
                try
                {
                    if (_fusionNetwork == null)
                    {
                        InitializeFusionNetwork(inputs.Columns);
                    }

                    // Convert to tensors for neural network
                    var inputTensor = new Tensor<T>(new[] { inputs.Rows, inputs.Columns }, inputs.ToColumnVector());
                    var targetTensor = new Tensor<T>(new[] { targets.Length }, targets);

                    // Train the neural network
                    _fusionNetwork!.Train(inputTensor, targetTensor);

                    _isTrained = true;

                    // Log training completion
                    Console.WriteLine($"Early fusion model training completed on {inputs.Rows} samples");
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Training failed: {ex.Message}", ex);
                }
            }
        }

        /// <summary>
        /// Makes predictions using the early fusion model
        /// </summary>
        /// <param name="inputs">Input data</param>
        /// <returns>Predictions</returns>
        public override Vector<T> Predict(Matrix<T> inputs)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(EarlyFusionMultimodal<T>));

            if (!_isTrained || _fusionNetwork == null)
                throw new InvalidOperationException("Model must be trained before making predictions");

            if (inputs == null)
                throw new ArgumentNullException(nameof(inputs));

            if (inputs.Rows == 0)
                return new Vector<T>(0);

            lock (_lockObject)
            {
                try
                {
                    // Convert to tensor and predict
                    var inputTensor = new Tensor<T>(new[] { inputs.Rows, inputs.Columns }, inputs.ToColumnVector());
                    var outputTensor = _fusionNetwork.Predict(inputTensor);

                    // Convert output tensor to vector
                    var outputArray = outputTensor.ToArray();
                    var predictions = new Vector<T>(outputArray.Length);
                    for (int i = 0; i < outputArray.Length; i++)
                    {
                        predictions[i] = outputArray[i];
                    }

                    return predictions;
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Prediction failed: {ex.Message}", ex);
                }
            }
        }

        /// <summary>
        /// Creates a copy of the model
        /// </summary>
        /// <returns>A copy of the model</returns>
        public override IFullModel<T, Dictionary<string, object>, Vector<T>> Clone()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(EarlyFusionMultimodal<T>));

            lock (_lockObject)
            {
                var clone = new EarlyFusionMultimodal<T>(_fusedDimension, _hiddenLayerSize, 
                                                     _numHiddenLayers, Convert.ToDouble(_learningRate), 
                                                     Convert.ToDouble(_dropoutRate), null, _ops);

                // Copy encoders
                foreach (var kvp in _modalityEncoders)
                {
                    clone.AddModalityEncoder(kvp.Key, kvp.Value);
                }

                // Deep copy fusion network if exists
                if (_fusionNetwork != null && _isTrained)
                {
                    // Initialize the clone's network with the same architecture
                    clone.InitializeFusionNetwork(_fusionNetwork.Architecture.InputSize);
                    
                    // Copy parameters if possible
                    try
                    {
                        var parameters = _fusionNetwork.GetParameters();
                        clone._fusionNetwork!.SetParameters(parameters);
                    }
                    catch
                    {
                        // If parameter copying fails, at least we have the architecture
                    }
                }

                clone._isTrained = _isTrained;
                clone.Name = Name;

                // Copy training history
                clone._trainingLosses.AddRange(_trainingLosses);
                clone._validationLosses.AddRange(_validationLosses);

                return clone;
            }
        }

        /// <summary>
        /// Concatenates multiple vectors into one
        /// </summary>
        private Vector<T> ConcatenateVectors(List<Vector<T>> vectors)
        {
            int totalDimension = vectors.Sum(v => v.Length);
            var concatenated = new Vector<T>(totalDimension);

            int offset = 0;
            foreach (var vector in vectors)
            {
                for (int i = 0; i < vector.Length; i++)
                {
                    concatenated[offset + i] = vector[i];
                }
                offset += vector.Length;
            }

            return concatenated;
        }

        /// <summary>
        /// Initializes the fusion network
        /// </summary>
        private void InitializeFusionNetwork(int inputDimension)
        {
            if (inputDimension <= 0)
                throw new ArgumentException("Input dimension must be positive", nameof(inputDimension));

            // Create layers explicitly
            var layerList = new List<ILayer<T>>();
            
            // Input to first hidden layer
            layerList.Add(new FullyConnectedLayer<T>(inputDimension, _hiddenLayerSize, null as IActivationFunction<T>));
            layerList.Add(new ActivationLayer<T>(new[] { _hiddenLayerSize }, new ReLUActivation<T>() as IActivationFunction<T>));
            
            // Add additional hidden layers
            for (int i = 1; i < _numHiddenLayers; i++)
            {
                layerList.Add(new FullyConnectedLayer<T>(_hiddenLayerSize, _hiddenLayerSize, null as IActivationFunction<T>));
                layerList.Add(new ActivationLayer<T>(new[] { _hiddenLayerSize }, new ReLUActivation<T>() as IActivationFunction<T>));
            }
            
            // Output layer
            layerList.Add(new FullyConnectedLayer<T>(_hiddenLayerSize, _fusedDimension, null as IActivationFunction<T>));

            // Create neural network architecture with explicit layers
            var architecture = new NeuralNetworkArchitecture<T>(
                complexity: NetworkComplexity.Medium,
                taskType: Enums.NeuralNetworkTaskType.Regression,
                shouldReturnFullSequence: false,
                layers: layerList,
                isDynamicSampleCount: true,
                isPlaceholder: false);

            _fusionNetwork = new NeuralNetwork<T>(architecture);
        }

        /// <summary>
        /// Gets training losses
        /// </summary>
        public IReadOnlyList<T> TrainingLosses => _trainingLosses.AsReadOnly();

        /// <summary>
        /// Gets validation losses
        /// </summary>
        public IReadOnlyList<T> ValidationLosses => _validationLosses.AsReadOnly();

        /// <summary>
        /// Gets parameters of the model
        /// </summary>
        public override Dictionary<string, object> GetParametersDictionary()
        {
            var parameters = base.GetParametersDictionary();
            parameters["HiddenLayerSize"] = _hiddenLayerSize;
            parameters["NumHiddenLayers"] = _numHiddenLayers;
            parameters["LearningRate"] = Convert.ToDouble(_learningRate);
            parameters["DropoutRate"] = Convert.ToDouble(_dropoutRate);
            parameters["IsTrained"] = _isTrained;
            
            if (_fusionNetwork != null)
            {
                parameters["NetworkInputSize"] = _fusionNetwork.Architecture.InputSize;
                parameters["NetworkOutputSize"] = _fusionNetwork.Architecture.OutputSize;
                parameters["TotalParameters"] = CalculateTotalParameters();
            }
            
            return parameters;
        }

        /// <summary>
        /// Calculates the total number of parameters in the network
        /// </summary>
        private int CalculateTotalParameters()
        {
            if (_fusionNetwork == null)
                return 0;

            int totalParams = 0;
            var inputSize = _fusionNetwork.Architecture.InputSize;
            var outputSize = _fusionNetwork.Architecture.OutputSize;
            // Get layer sizes from architecture
            var layerSizes = new List<int> { inputSize };
            
            // Add hidden layer sizes
            for (int i = 0; i < _numHiddenLayers; i++)
            {
                layerSizes.Add(_hiddenLayerSize);
            }
            layerSizes.Add(outputSize);
            
            // Calculate total parameters
            for (int i = 0; i < layerSizes.Count - 1; i++)
            {
                // Weights + biases for each layer
                totalParams += layerSizes[i] * layerSizes[i + 1] + layerSizes[i + 1];
            }
            
            return totalParams;
        }

        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Releases unmanaged and - optionally - managed resources.
        /// </summary>
        /// <param name="disposing">true to release both managed and unmanaged resources; false to release only unmanaged resources.</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    lock (_lockObject)
                    {
                        // Dispose managed resources
                        if (_fusionNetwork is IDisposable disposableNetwork)
                        {
                            disposableNetwork.Dispose();
                        }
                        
                        // Clear collections
                        _trainingLosses.Clear();
                        _validationLosses.Clear();
                        
                        // Dispose encoders if they implement IDisposable
                        foreach (var encoder in _modalityEncoders.Values)
                        {
                            if (encoder is IDisposable disposableEncoder)
                            {
                                disposableEncoder.Dispose();
                            }
                        }
                        _modalityEncoders.Clear();
                    }
                }

                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer
        /// </summary>
        ~EarlyFusionMultimodal()
        {
            Dispose(false);
        }
    }
}