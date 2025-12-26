using System;
using System.IO;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;

namespace AiDotNet.PhysicsInformed.NeuralOperators
{
    /// <summary>
    /// Implements Deep Operator Network (DeepONet) for learning operators.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// DeepONet is another approach to learning operators (like FNO), but with a different architecture.
    ///
    /// Universal Approximation Theorem for Operators:
    /// Just as neural networks can approximate any function, DeepONet can approximate any operator!
    /// This is based on a theorem by Chen and Chen (1995).
    ///
    /// The Key Idea - Decomposition:
    /// DeepONet represents an operator G as:
    /// G(u)(y) = Σᵢ bᵢ(u) * tᵢ(y)
    ///
    /// Where:
    /// - u is the input function
    /// - y is the query location
    /// - bᵢ(u) are "basis functions" of the input (learned by Branch Net)
    /// - tᵢ(y) are "basis functions" of the location (learned by Trunk Net)
    ///
    /// Architecture:
    /// DeepONet has TWO networks:
    ///
    /// 1. Branch Network:
    ///    - Input: The entire input function u(x) (sampled at sensors)
    ///    - Output: Coefficients b₁, b₂, ..., bₚ
    ///    - Role: Encodes information about the input function
    ///
    /// 2. Trunk Network:
    ///    - Input: Query location y (where we want to evaluate output)
    ///    - Output: Basis functions t₁(y), t₂(y), ..., tₚ(y)
    ///    - Role: Encodes spatial/temporal patterns
    ///
    /// 3. Combination:
    ///    - Output: G(u)(y) = b · t = Σᵢ bᵢ * tᵢ(y)
    ///    - Simple dot product of the two network outputs
    ///
    /// Analogy:
    /// Think of it like a bilinear form or low-rank factorization:
    /// - Branch net learns "what" information matters in the input
    /// - Trunk net learns "where" patterns occur spatially
    /// - Their interaction gives the output
    ///
    /// Example - Heat Equation:
    /// Problem: Given initial temperature u(x,0), find temperature u(x,t)
    ///
    /// Branch Net:
    /// - Input: u(x,0) sampled at many points → [u(x₁,0), u(x₂,0), ..., u(xₙ,0)]
    /// - Learns: "This initial condition is smooth/peaked/oscillatory"
    /// - Output: Coefficients [b₁, b₂, ..., bₚ]
    ///
    /// Trunk Net:
    /// - Input: (x, t) where we want to know the temperature
    /// - Learns: Spatial-temporal basis functions
    /// - Output: Basis values [t₁(x,t), t₂(x,t), ..., tₚ(x,t)]
    ///
    /// Result:
    /// u(x,t) = Σᵢ bᵢ * tᵢ(x,t)
    ///
    /// Key Advantages:
    /// 1. Sensor flexibility: Can use different sensor locations at test time
    /// 2. Query flexibility: Can evaluate at any location y
    /// 3. Theoretical foundation: Universal approximation theorem
    /// 4. Efficient: Once trained, very fast evaluation
    /// 5. Interpretable: Decomposition into branch/trunk has clear meaning
    ///
    /// Comparison with FNO:
    /// DeepONet:
    /// - Works on unstructured data (any sensor locations)
    /// - More flexible for irregular domains
    /// - Requires specifying sensor locations
    /// - Good for problems with sparse/irregular data
    ///
    /// FNO:
    /// - Works on structured grids
    /// - Uses FFT (very efficient)
    /// - Resolution-invariant
    /// - Good for periodic/regular problems
    ///
    /// Both are powerful, choice depends on your problem!
    ///
    /// Applications:
    /// - Same as FNO: PDEs, climate, fluids, etc.
    /// - Particularly good for:
    ///   * Inverse problems (finding unknown parameters)
    ///   * Problems with sparse measurements
    ///   * Irregular geometries
    ///   * Multi-scale phenomena
    ///
    /// Historical Note:
    /// DeepONet was introduced by Lu et al. (2021) and has been highly successful
    /// in learning solution operators for PDEs with theoretical guarantees.
    /// </remarks>
    public class DeepOperatorNetwork<T> : NeuralNetworkBase<T>
    {
        private readonly NeuralNetworkBase<T> _branchNet;
        private readonly NeuralNetworkBase<T> _trunkNet;
        private readonly int _p; // Dimension of the latent space (number of basis functions)
        private readonly int _numSensors;
        private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
        private readonly bool _usesDefaultOptimizer;

        /// <summary>
        /// Initializes a new instance of DeepONet.
        /// </summary>
        /// <param name="architecture">The overall architecture (mainly for metadata).</param>
        /// <param name="branchArchitecture">Architecture for the branch network.</param>
        /// <param name="trunkArchitecture">Architecture for the trunk network.</param>
        /// <param name="latentDimension">Dimension p of the latent space (number of basis functions).</param>
        /// <param name="numSensors">Number of sensor locations where input function is sampled.</param>
        /// <remarks>
        /// For Beginners:
        ///
        /// Parameters:
        ///
        /// latentDimension (p): Number of basis functions
        /// - Controls the expressiveness of the operator
        /// - Higher p = more expressive but more parameters
        /// - Typical: 100-400
        /// - Like the rank in low-rank matrix factorization
        ///
        /// numSensors: How many points to sample the input function
        /// - More sensors = more information about input
        /// - Must be enough to capture important features
        /// - Typical: 50-200
        /// - Can use different sensor locations at train vs. test time!
        ///
        /// Branch Network:
        /// - Input size: numSensors (values of input function at sensors)
        /// - Output size: latentDimension (p)
        /// - Architecture: Deep feedforward network
        /// - Typical: 3-5 layers, 100-200 neurons per layer
        ///
        /// Trunk Network:
        /// - Input size: dimension of query location (e.g., 2 for (x,y), 3 for (x,y,t))
        /// - Output size: latentDimension (p)
        /// - Architecture: Deep feedforward network
        /// - Typical: 3-5 layers, 100-200 neurons per layer
        /// </remarks>
        public DeepOperatorNetwork(
            NeuralNetworkArchitecture<T> architecture,
            NeuralNetworkArchitecture<T> branchArchitecture,
            NeuralNetworkArchitecture<T> trunkArchitecture,
            int latentDimension = 128,
            int numSensors = 100,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
            : base(architecture ?? throw new ArgumentNullException(nameof(architecture)),
                NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), 1.0)
        {
            if (branchArchitecture == null)
            {
                throw new ArgumentNullException(nameof(branchArchitecture));
            }

            if (trunkArchitecture == null)
            {
                throw new ArgumentNullException(nameof(trunkArchitecture));
            }

            if (latentDimension <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(latentDimension), "Latent dimension must be positive.");
            }

            if (numSensors <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(numSensors), "Number of sensors must be positive.");
            }

            _p = latentDimension;
            _numSensors = numSensors;
            _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
            _usesDefaultOptimizer = optimizer == null;

            // Create branch network
            var branchNetArchitecture = EnsureOutputSize(branchArchitecture, latentDimension, "Branch");
            _branchNet = new FeedForwardNeuralNetwork<T>(branchNetArchitecture);

            // Create trunk network
            var trunkNetArchitecture = EnsureOutputSize(trunkArchitecture, latentDimension, "Trunk");
            _trunkNet = new FeedForwardNeuralNetwork<T>(trunkNetArchitecture);

            InitializeLayers();
        }

        protected override void InitializeLayers()
        {
            // DeepONet doesn't use traditional layers
            // Its computation is done via branch and trunk networks
        }

        private static NeuralNetworkArchitecture<T> EnsureOutputSize(
            NeuralNetworkArchitecture<T> architecture,
            int outputSize,
            string name)
        {
            if (architecture == null)
            {
                throw new ArgumentNullException(nameof(architecture));
            }

            if (architecture.OutputSize == outputSize)
            {
                return architecture;
            }

            if (architecture.Layers != null && architecture.Layers.Count > 0)
            {
                throw new ArgumentException($"{name} architecture output size must be {outputSize} when custom layers are provided.");
            }

            return new NeuralNetworkArchitecture<T>(
                architecture.InputType,
                architecture.TaskType,
                architecture.Complexity,
                architecture.InputSize,
                architecture.InputHeight,
                architecture.InputWidth,
                architecture.InputDepth,
                outputSize,
                null,
                architecture.ShouldReturnFullSequence);
        }

        /// <summary>
        /// Evaluates the operator: G(u)(y) = branch(u) · trunk(y).
        /// </summary>
        /// <param name="inputFunction">Values of input function at sensor locations [numSensors].</param>
        /// <param name="queryLocation">Location where to evaluate output [spatialDim].</param>
        /// <returns>Output value at the query location.</returns>
        /// <remarks>
        /// For Beginners:
        /// This is the forward pass of DeepONet.
        ///
        /// Steps:
        /// 1. Pass input function values through branch network → get coefficients b
        /// 2. Pass query location through trunk network → get basis functions t
        /// 3. Compute dot product: output = b · t
        ///
        /// Example:
        /// - inputFunction = [0.5, 0.7, 0.3, ...] (100 values)
        ///   → branch net → [b₁, b₂, ..., b₁₂₈] (128 coefficients)
        ///
        /// - queryLocation = [0.3, 0.5] (x=0.3, y=0.5)
        ///   → trunk net → [t₁, t₂, ..., t₁₂₈] (128 basis values)
        ///
        /// - output = b₁*t₁ + b₂*t₂ + ... + b₁₂₈*t₁₂₈ (single number)
        ///
        /// To get output at multiple locations, call this function multiple times
        /// with different queryLocation values (branch only computed once!).
        /// </remarks>
        public T Evaluate(T[] inputFunction, T[] queryLocation)
        {
            if (inputFunction == null)
            {
                throw new ArgumentNullException(nameof(inputFunction));
            }

            if (queryLocation == null)
            {
                throw new ArgumentNullException(nameof(queryLocation));
            }

            if (inputFunction.Length != _numSensors)
            {
                throw new ArgumentException($"Input function must have {_numSensors} sensor values.");
            }

            int trunkInputSize = _trunkNet.Architecture.CalculatedInputSize;
            if (trunkInputSize <= 0)
            {
                throw new InvalidOperationException("Trunk network input size must be positive.");
            }

            if (queryLocation.Length != trunkInputSize)
            {
                throw new ArgumentException($"Query location must have {trunkInputSize} values.", nameof(queryLocation));
            }

            // Branch network: input function → coefficients
            var inputTensor = new Tensor<T>(new int[] { 1, inputFunction.Length });
            for (int i = 0; i < inputFunction.Length; i++)
            {
                inputTensor[0, i] = inputFunction[i];
            }

            var branchOutput = _branchNet.Predict(inputTensor);
            // Trunk network: query location → basis functions
            var queryTensor = new Tensor<T>(new int[] { 1, queryLocation.Length });
            for (int i = 0; i < queryLocation.Length; i++)
            {
                queryTensor[0, i] = queryLocation[i];
            }

            var trunkOutput = _trunkNet.Predict(queryTensor);

            var branchOutput2D = branchOutput.Rank == 2 ? branchOutput : branchOutput.Reshape(1, _p);
            var trunkOutput2D = trunkOutput.Rank == 2 ? trunkOutput : trunkOutput.Reshape(1, _p);
            var product = Engine.TensorMultiply(branchOutput2D, trunkOutput2D);
            var summed = Engine.ReduceSum(product, new[] { 1 }, keepDims: true);

            return summed[0, 0];
        }

        /// <summary>
        /// Evaluates the operator at multiple query locations efficiently.
        /// </summary>
        /// <param name="inputFunction">Input function values at sensors.</param>
        /// <param name="queryLocations">Multiple query locations [numQueries, spatialDim].</param>
        /// <returns>Output values at all query locations [numQueries].</returns>
        /// <remarks>
        /// For Beginners:
        /// This is more efficient than calling Evaluate() multiple times because:
        /// - Branch network is evaluated only once (not per query point)
        /// - Only trunk network is evaluated for each query location
        ///
        /// This is a key advantage of DeepONet: once you encode the input function
        /// via the branch network, you can query the solution at many locations
        /// very cheaply (just trunk network evaluations).
        /// </remarks>
        public T[] EvaluateMultiple(T[] inputFunction, T[,] queryLocations)
        {
            if (inputFunction == null)
            {
                throw new ArgumentNullException(nameof(inputFunction));
            }

            if (queryLocations == null)
            {
                throw new ArgumentNullException(nameof(queryLocations));
            }

            if (inputFunction.Length != _numSensors)
            {
                throw new ArgumentException($"Input function must have {_numSensors} sensor values.");
            }

            int trunkInputSize = _trunkNet.Architecture.CalculatedInputSize;
            if (trunkInputSize <= 0)
            {
                throw new InvalidOperationException("Trunk network input size must be positive.");
            }

            int numQueries = queryLocations.GetLength(0);
            if (numQueries == 0)
            {
                return Array.Empty<T>();
            }

            if (queryLocations.GetLength(1) != trunkInputSize)
            {
                throw new ArgumentException($"Query locations must have {trunkInputSize} values per entry.", nameof(queryLocations));
            }

            // Branch network (computed once)
            var inputTensor = new Tensor<T>(new int[] { 1, inputFunction.Length });
            for (int i = 0; i < inputFunction.Length; i++)
            {
                inputTensor[0, i] = inputFunction[i];
            }

            var branchOutput = _branchNet.Predict(inputTensor);
            var branchOutput2D = branchOutput.Rank == 2 ? branchOutput : branchOutput.Reshape(1, _p);

            // Trunk network (batched queries)
            var queryTensor = new Tensor<T>(new int[] { numQueries, trunkInputSize });
            for (int q = 0; q < numQueries; q++)
            {
                for (int j = 0; j < trunkInputSize; j++)
                {
                    queryTensor[q, j] = queryLocations[q, j];
                }
            }

            var trunkOutput = _trunkNet.Predict(queryTensor);
            var branchOutputT = Engine.TensorTranspose(branchOutput2D);
            var predictions = Engine.TensorMatMul(trunkOutput, branchOutputT);

            var outputs = new T[numQueries];
            for (int q = 0; q < numQueries; q++)
            {
                outputs[q] = predictions[q, 0];
            }

            return outputs;
        }

        /// <summary>
        /// Trains DeepONet on input-output function pairs.
        /// </summary>
        /// <param name="inputFunctions">Training input functions [numSamples, numSensors].</param>
        /// <param name="queryLocations">Query locations for each sample [numSamples, numQueries, spatialDim].</param>
        /// <param name="targetValues">Target output values [numSamples, numQueries].</param>
        /// <param name="epochs">Number of training epochs.</param>
        /// <returns>Training history.</returns>
        /// <remarks>
        /// For Beginners:
        /// Training DeepONet involves:
        /// 1. For each training example (input function, query locations, target outputs):
        ///    a) Evaluate DeepONet at query locations
        ///    b) Compute loss (MSE between predictions and targets)
        ///    c) Backpropagate through both branch and trunk networks
        ///    d) Update all parameters
        ///
        /// The beauty is that both networks learn together:
        /// - Branch learns what features of the input matter
        /// - Trunk learns what spatial patterns exist
        /// - They coordinate through the shared latent space
        ///
        /// Training Data Format:
        /// - inputFunctions[i]: Values at sensor locations for sample i
        /// - queryLocations[i]: Where to evaluate output for sample i
        /// - targetValues[i]: Ground truth outputs at those locations
        ///
        /// You can use different query locations for each training sample!
        /// This flexibility is a key advantage of DeepONet.
        /// </remarks>
        public TrainingHistory<T> Train(
            T[,] inputFunctions,
            T[,,] queryLocations,
            T[,] targetValues,
            int epochs = 100,
            double learningRate = 0.001,
            bool verbose = true)
        {
            var history = new TrainingHistory<T>();
            var lossFunction = LossFunction ?? new MeanSquaredErrorLoss<T>();

            if (inputFunctions == null)
            {
                throw new ArgumentNullException(nameof(inputFunctions));
            }

            if (queryLocations == null)
            {
                throw new ArgumentNullException(nameof(queryLocations));
            }

            if (targetValues == null)
            {
                throw new ArgumentNullException(nameof(targetValues));
            }

            if (inputFunctions.GetLength(1) != _numSensors)
            {
                throw new ArgumentException($"Input functions must have {_numSensors} sensor values per sample.", nameof(inputFunctions));
            }

            int numSamples = inputFunctions.GetLength(0);
            if (queryLocations.GetLength(0) != numSamples)
            {
                throw new ArgumentException("Query locations sample count must match input functions.", nameof(queryLocations));
            }

            if (targetValues.GetLength(0) != numSamples)
            {
                throw new ArgumentException("Target values sample count must match input functions.", nameof(targetValues));
            }

            int numQueries = queryLocations.GetLength(1);
            if (numSamples == 0)
            {
                return history;
            }

            if (numQueries == 0)
            {
                throw new ArgumentException("Query locations must include at least one query per sample.", nameof(queryLocations));
            }

            if (targetValues.GetLength(1) != numQueries)
            {
                throw new ArgumentException("Target values query count must match query locations.", nameof(targetValues));
            }

            int trunkInputSize = _trunkNet.Architecture.CalculatedInputSize;
            if (trunkInputSize <= 0)
            {
                throw new InvalidOperationException("Trunk network input size must be positive.");
            }

            if (queryLocations.GetLength(2) != trunkInputSize)
            {
                throw new ArgumentException($"Query locations must have {trunkInputSize} values per query.", nameof(queryLocations));
            }

            if (_usesDefaultOptimizer)
            {
                var options = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = learningRate
                };
                _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, options);
            }

            SetTrainingMode(true);
            _branchNet.SetTrainingMode(true);
            _trunkNet.SetTrainingMode(true);
            foreach (var layer in _branchNet.Layers)
            {
                layer.SetTrainingMode(true);
            }
            foreach (var layer in _trunkNet.Layers)
            {
                layer.SetTrainingMode(true);
            }

            try
            {
                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    T totalLoss = NumOps.Zero;

                    for (int i = 0; i < numSamples; i++)
                    {
                        var branchInput = new Tensor<T>(new int[] { 1, _numSensors });
                        var inputFunction = new T[_numSensors];
                        for (int j = 0; j < _numSensors; j++)
                        {
                            inputFunction[j] = inputFunctions[i, j];
                            branchInput[0, j] = inputFunctions[i, j];
                        }

                        var trunkInput = new Tensor<T>(new int[] { numQueries, trunkInputSize });
                        for (int q = 0; q < numQueries; q++)
                        {
                            for (int d = 0; d < trunkInputSize; d++)
                            {
                                trunkInput[q, d] = queryLocations[i, q, d];
                            }
                        }

                        var branchOutput = _branchNet.ForwardWithMemory(branchInput);
                        var trunkOutput = _trunkNet.ForwardWithMemory(trunkInput);

                        var branchOutput2D = branchOutput.Rank == 2
                            ? branchOutput
                            : branchOutput.Reshape(1, _p);
                        var trunkOutput2D = trunkOutput.Rank == 2
                            ? trunkOutput
                            : trunkOutput.Reshape(numQueries, _p);
                        var branchOutputT = Engine.TensorTranspose(branchOutput2D);
                        var predictions = Engine.TensorMatMul(trunkOutput2D, branchOutputT);

                        var targets = new Tensor<T>(new int[] { numQueries, 1 });
                        var targetValuesSample = new T[numQueries];
                        for (int q = 0; q < numQueries; q++)
                        {
                            targets[q, 0] = targetValues[i, q];
                            targetValuesSample[q] = targetValues[i, q];
                        }

                        var loss = lossFunction.CalculateLoss(predictions.ToVector(), targets.ToVector());
                        totalLoss = NumOps.Add(totalLoss, loss);

                        // Backpropagation: compute gradients for both branch and trunk networks
                        var outputGradientVector = lossFunction.CalculateDerivative(predictions.ToVector(), targets.ToVector());
                        var outputGradient = new Tensor<T>(predictions.Shape, outputGradientVector);

                        // Gradient for branch network: grad_branch = (grad_output)^T * trunk_output
                        var branchGradient = Engine.TensorMatMul(Engine.TensorTranspose(outputGradient), trunkOutput2D);
                        // Gradient for trunk network: grad_trunk = grad_output * branch_output
                        var trunkGradient = Engine.TensorMatMul(outputGradient, branchOutput2D);

                        // Backpropagate through both networks
                        _branchNet.Backpropagate(branchGradient);
                        _trunkNet.Backpropagate(trunkGradient);

                        // Update parameters using optimizer
                        _optimizer.UpdateParameters(_branchNet.Layers);
                        _optimizer.UpdateParameters(_trunkNet.Layers);

                        ClearNetworkGradients(_branchNet);
                        ClearNetworkGradients(_trunkNet);
                    }

                    T avgLoss = numSamples > 0
                        ? NumOps.Divide(totalLoss, NumOps.FromDouble(numSamples))
                        : NumOps.Zero;

                    history.AddEpoch(avgLoss);

                    if (verbose && epoch % 10 == 0)
                    {
                        Console.WriteLine($"Epoch {epoch}/{epochs}, Loss: {avgLoss}");
                    }
                }
            }
            finally
            {
                ClearNetworkGradients(_branchNet);
                ClearNetworkGradients(_trunkNet);
                foreach (var layer in _trunkNet.Layers)
                {
                    layer.SetTrainingMode(false);
                }
                foreach (var layer in _branchNet.Layers)
                {
                    layer.SetTrainingMode(false);
                }

                _trunkNet.SetTrainingMode(false);
                _branchNet.SetTrainingMode(false);
                SetTrainingMode(false);
            }

            return history;
        }

        private T ComputeAverageLoss(T[,] inputFunctions, T[,,] queryLocations, T[,] targetValues)
        {
            int numSamples = inputFunctions.GetLength(0);
            int numQueries = queryLocations.GetLength(1);
            T totalLoss = NumOps.Zero;

            for (int i = 0; i < numSamples; i++)
            {
                T[] inputFunc = new T[_numSensors];
                for (int j = 0; j < _numSensors; j++)
                {
                    inputFunc[j] = inputFunctions[i, j];
                }

                T[,] queries = new T[numQueries, queryLocations.GetLength(2)];
                for (int q = 0; q < numQueries; q++)
                {
                    for (int d = 0; d < queries.GetLength(1); d++)
                    {
                        queries[q, d] = queryLocations[i, q, d];
                    }
                }

                T[] predictions = EvaluateMultiple(inputFunc, queries);

                T loss = NumOps.Zero;
                for (int q = 0; q < numQueries; q++)
                {
                    T error = NumOps.Subtract(predictions[q], targetValues[i, q]);
                    loss = NumOps.Add(loss, NumOps.Multiply(error, error));
                }

                loss = NumOps.Divide(loss, NumOps.FromDouble(numQueries));
                totalLoss = NumOps.Add(totalLoss, loss);
            }

            return numSamples > 0
                ? NumOps.Divide(totalLoss, NumOps.FromDouble(numSamples))
                : NumOps.Zero;
        }

        /// <summary>
        /// Makes a prediction using the DeepONet for a batch of input/query pairs.
        /// </summary>
        /// <param name="input">Tensor containing input function values followed by query locations.</param>
        /// <returns>Predicted output tensor.</returns>
        public override Tensor<T> Predict(Tensor<T> input)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            if (input.Rank != 2)
            {
                throw new ArgumentException("DeepONet expects a 2D input tensor.");
            }

            int trunkInputSize = _trunkNet.Architecture.CalculatedInputSize;
            if (trunkInputSize <= 0)
            {
                throw new InvalidOperationException("Trunk network input size must be positive.");
            }

            int expectedWidth = _numSensors + trunkInputSize;
            if (input.Shape[1] != expectedWidth)
            {
                throw new ArgumentException($"Expected input width {expectedWidth} (sensors + query), got {input.Shape[1]}.");
            }

            var branchInput = new Tensor<T>(new int[] { input.Shape[0], _numSensors });
            var trunkInput = new Tensor<T>(new int[] { input.Shape[0], trunkInputSize });
            for (int i = 0; i < input.Shape[0]; i++)
            {
                for (int j = 0; j < _numSensors; j++)
                {
                    branchInput[i, j] = input[i, j];
                }
                for (int j = 0; j < trunkInputSize; j++)
                {
                    trunkInput[i, j] = input[i, _numSensors + j];
                }
            }

            var branchOutput = _branchNet.Predict(branchInput);
            var trunkOutput = _trunkNet.Predict(trunkInput);

            var branchOutput2D = branchOutput.Rank == 2 ? branchOutput : branchOutput.Reshape(input.Shape[0], _p);
            var trunkOutput2D = trunkOutput.Rank == 2 ? trunkOutput : trunkOutput.Reshape(input.Shape[0], _p);
            var product = Engine.TensorMultiply(branchOutput2D, trunkOutput2D);
            var summed = Engine.ReduceSum(product, new[] { 1 }, keepDims: true);

            return summed;
        }

        /// <summary>
        /// Updates the branch and trunk network parameters from a flattened vector.
        /// </summary>
        /// <param name="parameters">Parameter vector.</param>
        public override void UpdateParameters(Vector<T> parameters)
        {
            int branchParameterCount = _branchNet.GetParameterCount();
            int trunkParameterCount = _trunkNet.GetParameterCount();

            if (parameters.Length != branchParameterCount + trunkParameterCount)
            {
                throw new ArgumentException($"Expected {branchParameterCount + trunkParameterCount} parameters, got {parameters.Length}.");
            }

            Vector<T> branchParameters = parameters.GetSubVector(0, branchParameterCount);
            Vector<T> trunkParameters = parameters.GetSubVector(branchParameterCount, trunkParameterCount);

            _branchNet.UpdateParameters(branchParameters);
            _trunkNet.UpdateParameters(trunkParameters);
        }

        /// <summary>
        /// Gets the trainable parameters as a flattened vector.
        /// </summary>
        public override Vector<T> GetParameters()
        {
            return Vector<T>.Concatenate(_branchNet.GetParameters(), _trunkNet.GetParameters());
        }

        public override Vector<T> GetGradients()
        {
            return Vector<T>.Concatenate(_branchNet.GetGradients(), _trunkNet.GetGradients());
        }

        /// <summary>
        /// Performs a basic supervised training step using MSE loss.
        /// </summary>
        /// <param name="input">Training input tensor.</param>
        /// <param name="expectedOutput">Expected output tensor.</param>
        /// <remarks>
        /// Uses standard backpropagation through both branch and trunk networks.
        /// </remarks>
        public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            if (expectedOutput == null)
            {
                throw new ArgumentNullException(nameof(expectedOutput));
            }

            if (input.Rank != 2)
            {
                throw new ArgumentException("DeepONet expects a 2D input tensor.", nameof(input));
            }

            int trunkInputSize = _trunkNet.Architecture.CalculatedInputSize;
            if (trunkInputSize <= 0)
            {
                throw new InvalidOperationException("Trunk network input size must be positive.");
            }

            int expectedWidth = _numSensors + trunkInputSize;
            if (input.Shape[1] != expectedWidth)
            {
                throw new ArgumentException($"Expected input width {expectedWidth} (sensors + query), got {input.Shape[1]}.", nameof(input));
            }

            int batchSize = input.Shape[0];
            Tensor<T> expectedOutputTensor = expectedOutput;
            if (expectedOutput.Rank == 1)
            {
                expectedOutputTensor = expectedOutput.Reshape(batchSize, 1);
            }
            else if (expectedOutput.Rank != 2 || expectedOutput.Shape[0] != batchSize || expectedOutput.Shape[1] != 1)
            {
                throw new ArgumentException("Expected output shape [batch, 1].", nameof(expectedOutput));
            }

            SetTrainingMode(true);

            // Step 1: Forward pass - prepare inputs and compute predictions
            var branchInput = new Tensor<T>(new int[] { batchSize, _numSensors });
            var trunkInput = new Tensor<T>(new int[] { batchSize, trunkInputSize });
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < _numSensors; j++)
                {
                    branchInput[i, j] = input[i, j];
                }
                for (int j = 0; j < trunkInputSize; j++)
                {
                    trunkInput[i, j] = input[i, _numSensors + j];
                }
            }

            var branchOutput = _branchNet.ForwardWithMemory(branchInput);
            var trunkOutput = _trunkNet.ForwardWithMemory(trunkInput);

            var branchOutput2D = branchOutput.Rank == 2 ? branchOutput : branchOutput.Reshape(batchSize, _p);
            var trunkOutput2D = trunkOutput.Rank == 2 ? trunkOutput : trunkOutput.Reshape(batchSize, _p);
            var product = Engine.TensorMultiply(branchOutput2D, trunkOutput2D);
            var predictions = Engine.ReduceSum(product, new[] { 1 }, keepDims: true);

            // Step 2: Calculate loss
            var lossFunction = LossFunction ?? new MeanSquaredErrorLoss<T>();
            LastLoss = lossFunction.CalculateLoss(predictions.ToVector(), expectedOutputTensor.ToVector());

            // Step 3: Backward pass - compute gradients for both networks
            var outputGradientVector = lossFunction.CalculateDerivative(predictions.ToVector(), expectedOutputTensor.ToVector());
            var outputGradient = new Tensor<T>(predictions.Shape, outputGradientVector);
            var outputGradientExpanded = Engine.TensorRepeatElements(outputGradient, _p, axis: 1);

            var branchGradient = Engine.TensorMultiply(outputGradientExpanded, trunkOutput2D);
            var trunkGradient = Engine.TensorMultiply(outputGradientExpanded, branchOutput2D);

            _branchNet.Backpropagate(branchGradient);
            _trunkNet.Backpropagate(trunkGradient);

            // Step 4: Update parameters
            _optimizer.UpdateParameters(_branchNet.Layers);
            _optimizer.UpdateParameters(_trunkNet.Layers);

            SetTrainingMode(false);
        }

        /// <summary>
        /// Gets metadata about the DeepONet model.
        /// </summary>
        /// <returns>Model metadata.</returns>
        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                ModelType = ModelType.NeuralNetwork,
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "LatentDimension", _p },
                    { "NumSensors", _numSensors },
                    { "BranchModel", _branchNet.GetType().Name },
                    { "TrunkModel", _trunkNet.GetType().Name },
                    { "ParameterCount", GetParameterCount() }
                },
                ModelData = Serialize()
            };
        }

        /// <summary>
        /// Serializes DeepONet-specific data.
        /// </summary>
        /// <param name="writer">Binary writer.</param>
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            writer.Write(_p);
            writer.Write(_numSensors);

            var branchBytes = _branchNet.Serialize();
            writer.Write(branchBytes.Length);
            writer.Write(branchBytes);

            var trunkBytes = _trunkNet.Serialize();
            writer.Write(trunkBytes.Length);
            writer.Write(trunkBytes);
        }

        /// <summary>
        /// Deserializes DeepONet-specific data.
        /// </summary>
        /// <param name="reader">Binary reader.</param>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            int storedP = reader.ReadInt32();
            int storedSensors = reader.ReadInt32();

            if (storedP != _p || storedSensors != _numSensors)
            {
                throw new InvalidOperationException("Serialized DeepONet configuration does not match the current instance.");
            }

            int branchLength = reader.ReadInt32();
            _branchNet.Deserialize(reader.ReadBytes(branchLength));

            int trunkLength = reader.ReadInt32();
            _trunkNet.Deserialize(reader.ReadBytes(trunkLength));
        }

        /// <summary>
        /// Creates a new instance with the same configuration.
        /// </summary>
        /// <returns>New DeepONet instance.</returns>
        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new DeepOperatorNetwork<T>(
                Architecture,
                _branchNet.Architecture,
                _trunkNet.Architecture,
                _p,
                _numSensors,
                _optimizer);
        }

        private static void ClearNetworkGradients(NeuralNetworkBase<T> network)
        {
            foreach (var layer in network.Layers)
            {
                layer.ClearGradients();
            }
        }

        /// <summary>
        /// Gets the total number of parameters across branch and trunk networks.
        /// </summary>
        public override int ParameterCount => _branchNet.GetParameterCount() + _trunkNet.GetParameterCount();

        public override bool SupportsTraining => true;
        public override bool SupportsJitCompilation => false;
    }
}

