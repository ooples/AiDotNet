using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.NumericOperations;
using AiDotNet.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Threading.Tasks;

namespace AiDotNet.MultimodalAI
{
    /// <summary>
    /// Late fusion multimodal model that processes each modality separately before combining
    /// </summary>
    /// <remarks>
    /// This model implements late fusion strategy where each modality is processed
    /// independently through separate neural networks before combining their outputs.
    /// This approach preserves modality-specific characteristics and allows for
    /// specialized processing of each input type.
    /// </remarks>
    [Serializable]
    public class LateFusionMultimodal<T> : MultimodalModelBase<T>, IDisposable
    {
        private readonly INumericOperations<T> _ops;
        private readonly Dictionary<string, NeuralNetwork<T>> _modalityNetworks;
        private NeuralNetwork<T>? _fusionNetwork;
        private readonly int _modalityHiddenSize;
        private readonly int _fusionHiddenSize;
        private readonly T _learningRate;
        private readonly string _aggregationMethod;
        private readonly Random _random;
        private readonly Dictionary<string, T> _modalityWeights;
        private bool _disposed;
        private readonly object _lockObject = new object();

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
                                  string aggregationMethod = "weighted", int? randomSeed = null,
                                  INumericOperations<T>? ops = null)
            : base("late_fusion", fusedDimension, ops ?? MathHelper.GetNumericOperations<T>())
        {
            if (fusedDimension <= 0)
                throw new ArgumentException("Fused dimension must be positive", nameof(fusedDimension));
            if (modalityHiddenSize <= 0)
                throw new ArgumentException("Modality hidden size must be positive", nameof(modalityHiddenSize));
            if (fusionHiddenSize <= 0)
                throw new ArgumentException("Fusion hidden size must be positive", nameof(fusionHiddenSize));
            if (learningRate <= 0 || learningRate > 1)
                throw new ArgumentException("Learning rate must be in (0, 1]", nameof(learningRate));
            if (!new[] { "mean", "max", "weighted", "concat" }.Contains(aggregationMethod.ToLowerInvariant()))
                throw new ArgumentException("Invalid aggregation method. Use 'mean', 'max', 'weighted', or 'concat'", nameof(aggregationMethod));

            _ops = ops ?? throw new ArgumentNullException(nameof(ops));
            _modalityNetworks = new Dictionary<string, NeuralNetwork<T>>();
            _modalityHiddenSize = modalityHiddenSize;
            _fusionHiddenSize = fusionHiddenSize;
            _learningRate = _ops.FromDouble(learningRate);
            _aggregationMethod = aggregationMethod.ToLowerInvariant();
            _random = randomSeed.HasValue ? new Random(randomSeed.Value) : new Random();
            _modalityWeights = new Dictionary<string, T>();
        }

        /// <summary>
        /// Adds a modality encoder and creates a corresponding network
        /// </summary>
        public override void AddModalityEncoder(string modalityName, IModalityEncoder<T> encoder)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(LateFusionMultimodal<T>));

            base.AddModalityEncoder(modalityName, encoder);
            
            lock (_lockObject)
            {
                // Create a modality-specific network with explicit layers
                var layers = new List<ILayer<T>>
                {
                    new FullyConnectedLayer<T>(encoder.OutputDimension, _modalityHiddenSize, null as IActivationFunction<T>),
                    new ActivationLayer<T>(new[] { _modalityHiddenSize }, new ReLUActivation<T>() as IActivationFunction<T>),
                    new FullyConnectedLayer<T>(_modalityHiddenSize, _modalityHiddenSize / 2, null as IActivationFunction<T>),
                    new ActivationLayer<T>(new[] { _modalityHiddenSize / 2 }, new ReLUActivation<T>() as IActivationFunction<T>)
                };
                
                var architecture = new NeuralNetworkArchitecture<T>(
                    complexity: NetworkComplexity.Medium,
                    taskType: NeuralNetworkTaskType.Regression,
                    shouldReturnFullSequence: false,
                    layers: layers,
                    isDynamicSampleCount: true,
                    isPlaceholder: false);
                
                var network = new NeuralNetwork<T>(architecture);
                _modalityNetworks[modalityName] = network;
                
                // Initialize weight for weighted aggregation
                _modalityWeights[modalityName] = _ops.FromDouble(1.0 / (_modalityWeights.Count + 1));
                
                // Normalize weights
                var totalWeight = _ops.Zero;
                foreach (var weight in _modalityWeights.Values)
                {
                    totalWeight = _ops.Add(totalWeight, weight);
                }
                foreach (var key in _modalityWeights.Keys.ToList())
                {
                    _modalityWeights[key] = _ops.Divide(_modalityWeights[key], totalWeight);
                }
            }
        }

        /// <summary>
        /// Processes multimodal input data using late fusion
        /// </summary>
        /// <param name="modalityData">Dictionary mapping modality names to their data</param>
        /// <returns>Fused representation</returns>
        public override Vector<T> ProcessMultimodal(Dictionary<string, object> modalityData)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(LateFusionMultimodal<T>));

            ValidateModalityData(modalityData);

            lock (_lockObject)
            {
                try
                {
                    var modalityOutputs = new Dictionary<string, Vector<T>>();
                    var processingTasks = new List<Task<(string, Vector<T>)>>();

                    // Process each modality independently in parallel
                    foreach (var kvp in modalityData)
                    {
                        if (_modalityEncoders.ContainsKey(kvp.Key) && _modalityNetworks.ContainsKey(kvp.Key))
                        {
                            var modalityName = kvp.Key;
                            var data = kvp.Value;
                            
                            var task = Task.Run(() =>
                            {
                                // Encode modality
                                var encoded = EncodeModality(modalityName, data);
                                
                                // Process through modality-specific network
                                var inputTensor = new Tensor<T>(new[] { encoded.Length }, encoded);
                                var outputTensor = _modalityNetworks[modalityName].Predict(inputTensor);
                                
                                // Extract output vector
                                var outputArray = outputTensor.ToArray();
                                var output = new Vector<T>(outputArray.Length);
                                for (int i = 0; i < outputArray.Length; i++)
                                {
                                    output[i] = outputArray[i];
                                }
                                
                                return (modalityName, output);
                            });
                            
                            processingTasks.Add(task);
                        }
                    }

                    // Wait for all modality processing to complete
                    Task.WaitAll(processingTasks.ToArray());
                    
                    // Collect results
                    foreach (var task in processingTasks)
                    {
                        var (modalityName, output) = task.Result;
                        modalityOutputs[modalityName] = output;
                    }

                    if (modalityOutputs.Count == 0)
                        throw new InvalidOperationException("No modalities were successfully processed");

                    // Aggregate modality outputs
                    var aggregated = AggregateModalityOutputs(modalityOutputs);

                    // Initialize fusion network if needed
                    if (_fusionNetwork == null)
                    {
                        InitializeFusionNetwork(aggregated.Length);
                    }

                    // Final fusion processing
                    var fusionInputTensor = new Tensor<T>(new[] { aggregated.Length }, aggregated);
                    var fusedTensor = _fusionNetwork!.Predict(fusionInputTensor);
                    
                    // Extract fused vector
                    var fusedArray = fusedTensor.ToArray();
                    var fused = new Vector<T>(fusedArray.Length);
                    for (int i = 0; i < fusedArray.Length; i++)
                    {
                        fused[i] = fusedArray[i];
                    }

                    // Project to target dimension if needed
                    if (fused.Length != _fusedDimension)
                    {
                        fused = ProjectToTargetDimension(fused, _fusedDimension);
                    }

                    return NormalizeFused(fused);
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Error processing multimodal data: {ex.Message}", ex);
                }
            }
        }

        /// <summary>
        /// Trains the late fusion model
        /// </summary>
        public override void Train(Matrix<T> inputs, Vector<T> targets)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(LateFusionMultimodal<T>));

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
                    // For late fusion, we need to train each modality network separately
                    // This is a simplified approach - in practice, you'd want separate training data for each modality
                    
                    // Split input by modality
                    var modalityInputs = SplitInputsByModality(inputs);
                    
                    // Train each modality network
                    foreach (var kvp in modalityInputs)
                    {
                        if (_modalityNetworks.ContainsKey(kvp.Key))
                        {
                            Console.WriteLine($"Training {kvp.Key} modality network...");
                            
                            // Create target matrix for this modality
                            var modalityTargets = new Matrix<T>(targets.Length, _modalityHiddenSize / 2);
                            
                            // For simplicity, we'll use the same targets transformed
                            // In practice, you'd have modality-specific intermediate targets
                            for (int i = 0; i < targets.Length; i++)
                            {
                                for (int j = 0; j < modalityTargets.Columns; j++)
                                {
                                    var factor = _ops.FromDouble((double)(j + 1) / modalityTargets.Columns);
                                    modalityTargets[i, j] = _ops.Multiply(targets[i], factor);
                                }
                            }
                            
                            var inputTensor = new Tensor<T>(new[] { kvp.Value.Rows, kvp.Value.Columns }, kvp.Value.ToColumnVector());
                            var targetTensor = new Tensor<T>(new[] { modalityTargets.Rows, modalityTargets.Columns }, modalityTargets.ToColumnVector());
                            _modalityNetworks[kvp.Key].Train(inputTensor, targetTensor);
                        }
                    }
                    
                    // Now train the fusion network
                    Console.WriteLine("Training fusion network...");
                    
                    // Process all inputs through modality networks to get fusion inputs
                    var fusionInputs = new Matrix<T>(inputs.Rows, 0);
                    
                    for (int i = 0; i < inputs.Rows; i++)
                    {
                        var modalityOutputs = new Dictionary<string, Vector<T>>();
                        
                        foreach (var kvp in modalityInputs)
                        {
                            if (_modalityNetworks.ContainsKey(kvp.Key))
                            {
                                var modalityInput = new Vector<T>(kvp.Value.Columns);
                                for (int j = 0; j < kvp.Value.Columns; j++)
                                {
                                    modalityInput[j] = kvp.Value[i, j];
                                }
                                
                                var inputTensor = new Tensor<T>(new[] { modalityInput.Length }, modalityInput);
                                var modalityOutput = _modalityNetworks[kvp.Key].Predict(inputTensor);
                                var outputArray = modalityOutput.ToArray();
                                var outputVector = new Vector<T>(outputArray.Length);
                                for (int j = 0; j < outputArray.Length; j++)
                                {
                                    outputVector[j] = outputArray[j];
                                }
                                
                                modalityOutputs[kvp.Key] = outputVector;
                            }
                        }
                        
                        var aggregated = AggregateModalityOutputs(modalityOutputs);
                        
                        if (i == 0)
                        {
                            fusionInputs = new Matrix<T>(inputs.Rows, aggregated.Length);
                            if (_fusionNetwork == null)
                            {
                                InitializeFusionNetwork(aggregated.Length);
                            }
                        }
                        
                        for (int j = 0; j < aggregated.Length; j++)
                        {
                            fusionInputs[i, j] = aggregated[j];
                        }
                    }
                    
                    // Convert targets to matrix format
                    var targetMatrix = new Matrix<T>(targets.Length, 1);
                    for (int i = 0; i < targets.Length; i++)
                    {
                        targetMatrix[i, 0] = targets[i];
                    }
                    
                    var fusionInputTensor = new Tensor<T>(new[] { fusionInputs.Rows, fusionInputs.Columns }, fusionInputs.ToColumnVector());
                    var fusionTargetTensor = new Tensor<T>(new[] { targetMatrix.Rows, targetMatrix.Columns }, targetMatrix.ToColumnVector());
                    _fusionNetwork!.Train(fusionInputTensor, fusionTargetTensor);
                    
                    _isTrained = true;
                    Console.WriteLine("Late fusion model training completed");
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Training failed: {ex.Message}", ex);
                }
            }
        }

        /// <summary>
        /// Creates a copy of the model
        /// </summary>
        public override IFullModel<T, Dictionary<string, object>, Vector<T>> Clone()
        {
            var clone = new LateFusionMultimodal<T>(_fusedDimension, _modalityHiddenSize,
                                               _fusionHiddenSize, Convert.ToDouble(_learningRate), _aggregationMethod, null, _ops);

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
        private Vector<T> AggregateModalityOutputs(Dictionary<string, Vector<T>> outputs, Dictionary<string, T>? weights = null)
        {
            if (outputs.Count == 0)
                throw new ArgumentException("No modality outputs to aggregate");

            int dimension = outputs.First().Value.Length;
            var aggregated = new Vector<T>(dimension);

            switch (_aggregationMethod.ToLower())
            {
                case "mean":
                    // Simple mean aggregation
                    foreach (var output in outputs.Values)
                    {
                        for (int i = 0; i < dimension; i++)
                        {
                            aggregated[i] = _ops.Add(aggregated[i], output[i]);
                        }
                    }
                    var divisor = _ops.FromDouble(outputs.Count);
                    for (int i = 0; i < dimension; i++)
                    {
                        aggregated[i] = _ops.Divide(aggregated[i], divisor);
                    }
                    break;

                case "max":
                    // Max pooling aggregation
                    for (int i = 0; i < dimension; i++)
                    {
                        var maxVal = outputs.Values.First()[i];
                        foreach (var output in outputs.Values)
                        {
                            if (_ops.GreaterThan(output[i], maxVal))
                                maxVal = output[i];
                        }
                        aggregated[i] = maxVal;
                    }
                    break;

                case "weighted":
                    // Weighted aggregation
                    if (weights == null || weights.Count == 0)
                    {
                        // Use stored modality weights or equal weights
                        if (_modalityWeights.Count > 0)
                        {
                            weights = _modalityWeights;
                        }
                        else
                        {
                            weights = new Dictionary<string, T>();
                            var defaultWeight = _ops.FromDouble(1.0 / outputs.Count);
                            foreach (var key in outputs.Keys)
                            {
                                weights[key] = defaultWeight;
                            }
                        }
                    }

                    // Normalize weights
                    var totalWeight = _ops.Zero;
                    foreach (var weight in weights.Values)
                    {
                        totalWeight = _ops.Add(totalWeight, weight);
                    }
                    
                    foreach (var kvp in outputs)
                    {
                        var weight = weights.ContainsKey(kvp.Key) ? _ops.Divide(weights[kvp.Key], totalWeight) : _ops.Zero;
                        for (int i = 0; i < dimension; i++)
                        {
                            aggregated[i] = _ops.Add(aggregated[i], _ops.Multiply(weight, kvp.Value[i]));
                        }
                    }
                    break;

                case "concat":
                    // Concatenation (results in larger dimension)
                    var allValues = new List<T>();
                    foreach (var output in outputs.Values)
                    {
                        for (int i = 0; i < output.Length; i++)
                        {
                            allValues.Add(output[i]);
                        }
                    }
                    aggregated = new Vector<T>(allValues.ToArray());
                    break;

                default:
                    throw new ArgumentException($"Unknown aggregation method: {_aggregationMethod}");
            }

            return aggregated;
        }

        /// <summary>
        /// Calculates weight/confidence for a modality output
        /// </summary>
        private T CalculateModalityWeight(Vector<T> output)
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
            if (inputDimension <= 0)
                throw new ArgumentException("Input dimension must be positive", nameof(inputDimension));

            // Create layers explicitly
            var layers = new List<ILayer<T>>
            {
                new FullyConnectedLayer<T>(inputDimension, _fusionHiddenSize, null as IActivationFunction<T>),
                new ActivationLayer<T>(new[] { _fusionHiddenSize }, new ReLUActivation<T>() as IActivationFunction<T>),
                new FullyConnectedLayer<T>(_fusionHiddenSize, _fusionHiddenSize / 2, null as IActivationFunction<T>),
                new ActivationLayer<T>(new[] { _fusionHiddenSize / 2 }, new ReLUActivation<T>() as IActivationFunction<T>),
                new FullyConnectedLayer<T>(_fusionHiddenSize / 2, _fusedDimension, null as IActivationFunction<T>)
            };

            var architecture = new NeuralNetworkArchitecture<T>(
                complexity: NetworkComplexity.Medium,
                taskType: NeuralNetworkTaskType.Regression,
                shouldReturnFullSequence: false,
                layers: layers,
                isDynamicSampleCount: true,
                isPlaceholder: false);
            
            _fusionNetwork = new NeuralNetwork<T>(architecture);
        }

        /// <summary>
        /// Splits input matrix by modality
        /// </summary>
        private Dictionary<string, Matrix<T>> SplitInputsByModality(Matrix<T> inputs)
        {
            var result = new Dictionary<string, Matrix<T>>();
            var modalities = _modalityEncoders.Keys.OrderBy(k => k).ToList();
            
            if (modalities.Count == 0 || inputs.Columns == 0)
                return result;

            int colsPerModality = inputs.Columns / modalities.Count;
            int remainder = inputs.Columns % modalities.Count;
            
            int currentCol = 0;
            for (int i = 0; i < modalities.Count; i++)
            {
                int modalityCols = colsPerModality + (i < remainder ? 1 : 0);
                var modalityData = new Matrix<T>(inputs.Rows, modalityCols);
                
                for (int row = 0; row < inputs.Rows; row++)
                {
                    for (int col = 0; col < modalityCols; col++)
                    {
                        modalityData[row, col] = inputs[row, currentCol + col];
                    }
                }
                
                result[modalities[i]] = modalityData;
                currentCol += modalityCols;
            }
            
            return result;
        }

        /// <summary>
        /// Splits input vector by modality (simplified)
        /// </summary>
        private Dictionary<string, Vector<T>> SplitInputByModality(Vector<T> input)
        {
            var result = new Dictionary<string, Vector<T>>();
            var modalities = _modalityEncoders.Keys.ToList();
            
            if (modalities.Count == 0)
                return result;

            int dimensionPerModality = input.Length / modalities.Count;
            
            for (int i = 0; i < modalities.Count; i++)
            {
                int start = i * dimensionPerModality;
                int end = (i == modalities.Count - 1) ? input.Length : (i + 1) * dimensionPerModality;
                
                var modalityInput = new Vector<T>(end - start);
                for (int j = 0; j < modalityInput.Length; j++)
                {
                    modalityInput[j] = input[start + j];
                }
                
                result[modalities[i]] = modalityInput;
            }

            return result;
        }


        /// <summary>
        /// Gets parameters of the model
        /// </summary>
        public override Dictionary<string, object> GetParametersDictionary()
        {
            var parameters = base.GetParametersDictionary();
            parameters["ModalityHiddenSize"] = _modalityHiddenSize;
            parameters["FusionHiddenSize"] = _fusionHiddenSize;
            parameters["LearningRate"] = Convert.ToDouble(_learningRate);
            parameters["AggregationMethod"] = _aggregationMethod;
            parameters["NumModalityNetworks"] = _modalityNetworks.Count;
            parameters["IsTrained"] = _isTrained;
            
            if (_modalityWeights.Count > 0)
            {
                var doubleWeights = new Dictionary<string, double>();
                foreach (var kvp in _modalityWeights)
                {
                    doubleWeights[kvp.Key] = Convert.ToDouble(kvp.Value);
                }
                parameters["ModalityWeights"] = doubleWeights;
            }
            
            return parameters;
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
                        // Dispose modality networks
                        foreach (var network in _modalityNetworks.Values)
                        {
                            if (network is IDisposable disposableNetwork)
                            {
                                disposableNetwork.Dispose();
                            }
                        }
                        _modalityNetworks.Clear();
                        
                        // Dispose fusion network
                        if (_fusionNetwork is IDisposable disposableFusion)
                        {
                            disposableFusion.Dispose();
                        }
                        
                        // Dispose encoders if they implement IDisposable
                        foreach (var encoder in _modalityEncoders.Values)
                        {
                            if (encoder is IDisposable disposableEncoder)
                            {
                                disposableEncoder.Dispose();
                            }
                        }
                        _modalityEncoders.Clear();
                        
                        // Clear weights
                        _modalityWeights.Clear();
                    }
                }

                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer
        /// </summary>
        ~LateFusionMultimodal()
        {
            Dispose(false);
        }
    }
}