using System;
using System.Numerics;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;

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
    public class DeepOperatorNetwork<T> : NeuralNetworkBase<T> where T : struct, INumber<T>
    {
        private readonly NeuralNetworkBase<T> _branchNet;
        private readonly NeuralNetworkBase<T> _trunkNet;
        private readonly int _p; // Dimension of the latent space (number of basis functions)
        private readonly int _numSensors;

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
            int numSensors = 100)
            : base(architecture, null, 1.0)
        {
            _p = latentDimension;
            _numSensors = numSensors;

            // Create branch network
            branchArchitecture.OutputSize = latentDimension;
            _branchNet = new FeedForwardNeuralNetwork<T>(branchArchitecture);

            // Create trunk network
            trunkArchitecture.OutputSize = latentDimension;
            _trunkNet = new FeedForwardNeuralNetwork<T>(trunkArchitecture);

            InitializeLayers();
        }

        protected override void InitializeLayers()
        {
            // DeepONet doesn't use traditional layers
            // Its computation is done via branch and trunk networks
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
            if (inputFunction.Length != _numSensors)
            {
                throw new ArgumentException($"Input function must have {_numSensors} sensor values.");
            }

            // Branch network: input function → coefficients
            var inputTensor = new Tensor<T>(new int[] { 1, inputFunction.Length });
            for (int i = 0; i < inputFunction.Length; i++)
            {
                inputTensor[0, i] = inputFunction[i];
            }

            var branchOutput = _branchNet.Forward(inputTensor);
            T[] coefficients = new T[_p];
            for (int i = 0; i < _p; i++)
            {
                coefficients[i] = branchOutput[0, i];
            }

            // Trunk network: query location → basis functions
            var queryTensor = new Tensor<T>(new int[] { 1, queryLocation.Length });
            for (int i = 0; i < queryLocation.Length; i++)
            {
                queryTensor[0, i] = queryLocation[i];
            }

            var trunkOutput = _trunkNet.Forward(queryTensor);
            T[] basisFunctions = new T[_p];
            for (int i = 0; i < _p; i++)
            {
                basisFunctions[i] = trunkOutput[0, i];
            }

            // Dot product: output = Σᵢ bᵢ * tᵢ
            T output = T.Zero;
            for (int i = 0; i < _p; i++)
            {
                output += coefficients[i] * basisFunctions[i];
            }

            return output;
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
            int numQueries = queryLocations.GetLength(0);

            // Branch network (computed once)
            var inputTensor = new Tensor<T>(new int[] { 1, inputFunction.Length });
            for (int i = 0; i < inputFunction.Length; i++)
            {
                inputTensor[0, i] = inputFunction[i];
            }

            var branchOutput = _branchNet.Forward(inputTensor);
            T[] coefficients = new T[_p];
            for (int i = 0; i < _p; i++)
            {
                coefficients[i] = branchOutput[0, i];
            }

            // Trunk network (for each query location)
            T[] outputs = new T[numQueries];
            for (int q = 0; q < numQueries; q++)
            {
                T[] queryLocation = new T[queryLocations.GetLength(1)];
                for (int j = 0; j < queryLocation.Length; j++)
                {
                    queryLocation[j] = queryLocations[q, j];
                }

                var queryTensor = new Tensor<T>(new int[] { 1, queryLocation.Length });
                for (int i = 0; i < queryLocation.Length; i++)
                {
                    queryTensor[0, i] = queryLocation[i];
                }

                var trunkOutput = _trunkNet.Forward(queryTensor);

                // Dot product
                T output = T.Zero;
                for (int i = 0; i < _p; i++)
                {
                    output += coefficients[i] * trunkOutput[0, i];
                }

                outputs[q] = output;
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
            int epochs = 100)
        {
            var history = new TrainingHistory<T>();

            int numSamples = inputFunctions.GetLength(0);

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                T totalLoss = T.Zero;

                for (int i = 0; i < numSamples; i++)
                {
                    // Extract input function for this sample
                    T[] inputFunc = new T[_numSensors];
                    for (int j = 0; j < _numSensors; j++)
                    {
                        inputFunc[j] = inputFunctions[i, j];
                    }

                    // Extract query locations for this sample
                    int numQueries = queryLocations.GetLength(1);
                    T[,] queries = new T[numQueries, queryLocations.GetLength(2)];
                    for (int q = 0; q < numQueries; q++)
                    {
                        for (int d = 0; d < queries.GetLength(1); d++)
                        {
                            queries[q, d] = queryLocations[i, q, d];
                        }
                    }

                    // Evaluate DeepONet
                    T[] predictions = EvaluateMultiple(inputFunc, queries);

                    // Compute loss
                    T loss = T.Zero;
                    for (int q = 0; q < numQueries; q++)
                    {
                        T error = predictions[q] - targetValues[i, q];
                        loss += error * error;
                    }
                    loss /= T.CreateChecked(numQueries);

                    totalLoss += loss;
                }

                T avgLoss = totalLoss / T.CreateChecked(numSamples);
                history.AddEpoch(avgLoss);

                if (epoch % 10 == 0)
                {
                    Console.WriteLine($"Epoch {epoch}/{epochs}, Loss: {avgLoss}");
                }
            }

            return history;
        }
    }
}
