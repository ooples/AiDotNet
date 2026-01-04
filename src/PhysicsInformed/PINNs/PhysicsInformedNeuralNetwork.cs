using System;
using System.Collections.Generic;
using System.IO;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.PhysicsInformed.PINNs
{
    /// <summary>
    /// Represents a Physics-Informed Neural Network (PINN) for solving PDEs.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// A Physics-Informed Neural Network (PINN) is a neural network that learns to solve
    /// Partial Differential Equations (PDEs) by incorporating physical laws directly into
    /// the training process.
    ///
    /// Traditional Approach (Finite Elements/Differences):
    /// - Discretize the domain into a grid
    /// - Approximate derivatives using neighboring points
    /// - Solve a large system of equations
    /// - Works well but can be slow for complex geometries
    ///
    /// PINN Approach:
    /// - Neural network represents the solution u(x,t)
    /// - Use automatic differentiation to compute ∂u/∂x, ∂²u/∂x², etc.
    /// - Train the network to minimize:
    ///   * PDE residual (how much the PDE is violated)
    ///   * Boundary condition errors
    ///   * Initial condition errors
    ///   * Data fitting errors (if measurements are available)
    ///
    /// Key Advantages:
    /// 1. Meshless: No need to discretize the domain
    /// 2. Data-efficient: Can work with sparse or noisy data
    /// 3. Flexible: Easy to handle complex geometries and boundary conditions
    /// 4. Interpolation: Get solution at any point by evaluating the network
    /// 5. Inverse problems: Can discover unknown parameters in the PDE
    ///
    /// Key Challenges:
    /// 1. Training can be difficult (multiple objectives to balance)
    /// 2. May require careful tuning of loss weights
    /// 3. Network architecture affects accuracy
    /// 4. Computational cost during training (many derivative evaluations)
    ///
    /// Applications:
    /// - Fluid dynamics (Navier-Stokes equations)
    /// - Heat transfer
    /// - Structural mechanics
    /// - Quantum mechanics
    /// - Financial modeling (Black-Scholes PDE)
    /// - Climate and weather modeling
    ///
    /// Historical Context:
    /// PINNs were introduced by Raissi, Perdikaris, and Karniadakis in 2019.
    /// They've revolutionized scientific machine learning by showing that deep learning
    /// can be guided by physics rather than just data.
    /// </remarks>
    public class PhysicsInformedNeuralNetwork<T> : NeuralNetworkBase<T>
    {
        /// <summary>
        /// The PDE specification that defines the physics constraints.
        /// Protected to allow derived classes (e.g., MultiFidelityPINN) to evaluate residuals on custom solutions.
        /// </summary>
        protected readonly IPDESpecification<T> _pdeSpecification;
        private readonly IBoundaryCondition<T>[] _boundaryConditions;
        private readonly IInitialCondition<T>? _initialCondition;
        private readonly PhysicsInformedLoss<T> _physicsLoss;
        private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
        private readonly bool _usesDefaultOptimizer;

        // Collocation points for PDE residual evaluation
        private T[,]? _collocationPoints;
        private readonly int _numCollocationPoints;

        /// <summary>
        /// Initializes a new instance of the PINN class.
        /// </summary>
        /// <param name="architecture">The neural network architecture (typically a deep feedforward network).</param>
        /// <param name="pdeSpecification">The PDE that the solution must satisfy.</param>
        /// <param name="boundaryConditions">Boundary conditions for the problem.</param>
        /// <param name="initialCondition">Initial condition for time-dependent problems (optional).</param>
        /// <param name="numCollocationPoints">Number of points in the domain where to enforce the PDE.</param>
        /// <param name="optimizer">Optimization algorithm (Adam is recommended for PINNs).</param>
        /// <param name="dataWeight">Weight for data loss component.</param>
        /// <param name="pdeWeight">Weight for PDE residual loss (often needs tuning).</param>
        /// <param name="boundaryWeight">Weight for boundary condition loss.</param>
        /// <param name="initialWeight">Weight for initial condition loss.</param>
        /// <remarks>
        /// For Beginners:
        /// When creating a PINN, you need to specify:
        /// 1. Network architecture: Usually a deep network (5-10 hidden layers, 20-50 neurons each)
        ///    - Activation: tanh or sin often work well for smooth solutions
        ///    - Input: spatial coordinates (x, y, z) and possibly time (t)
        ///    - Output: solution values u(x,t)
        ///
        /// 2. PDE specification: Defines the physics (e.g., Heat Equation, Navier-Stokes)
        ///
        /// 3. Boundary conditions: What happens at the edges of your domain
        ///
        /// 4. Collocation points: Where to enforce the PDE
        ///    - More points = better accuracy but slower training
        ///    - Typically 10,000-100,000 points
        ///    - Can use random sampling or quasi-random (Sobol, Latin hypercube)
        ///
        /// 5. Loss weights: Balance between different objectives
        ///    - Start with all weights = 1.0
        ///    - If PDE residual is large, increase pdeWeight
        ///    - If boundary conditions are violated, increase boundaryWeight
        ///    - This is often the trickiest part of PINN training!
        /// </remarks>
        public PhysicsInformedNeuralNetwork(
            NeuralNetworkArchitecture<T> architecture,
            IPDESpecification<T> pdeSpecification,
            IBoundaryCondition<T>[] boundaryConditions,
            IInitialCondition<T>? initialCondition = null,
            int numCollocationPoints = 10000,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
            double? dataWeight = null,
            double? pdeWeight = null,
            double? boundaryWeight = null,
            double? initialWeight = null)
            : base(architecture, NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), 1.0)
        {
            _pdeSpecification = pdeSpecification ?? throw new ArgumentNullException(nameof(pdeSpecification));
            _boundaryConditions = boundaryConditions ?? throw new ArgumentNullException(nameof(boundaryConditions));
            _initialCondition = initialCondition;
            _numCollocationPoints = numCollocationPoints;

            // Create the physics-informed loss function
            _physicsLoss = new PhysicsInformedLoss<T>(
                _pdeSpecification,
                _boundaryConditions,
                _initialCondition,
                dataWeight,
                pdeWeight,
                boundaryWeight,
                initialWeight);

            if (optimizer == null)
            {
                _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
                _usesDefaultOptimizer = true;
            }
            else
            {
                _optimizer = optimizer;
                _usesDefaultOptimizer = false;
            }

            InitializeLayers();
            GenerateCollocationPoints();
        }

        /// <summary>
        /// Initializes the neural network layers.
        /// </summary>
        protected override void InitializeLayers()
        {
            // Use custom layers if provided in architecture
            if (Architecture.Layers != null && Architecture.Layers.Count > 0)
            {
                Layers.AddRange(Architecture.Layers);
                ValidateCustomLayers(Layers);
            }
            else
            {
                // Create default deep feedforward architecture
                // For PINNs, we typically want deeper networks with moderate width
                Layers.AddRange(LayerHelper<T>.CreateDefaultPINNLayers(Architecture));
            }
        }

        /// <summary>
        /// Generates collocation points for enforcing the PDE in the domain.
        /// </summary>
        /// <remarks>
        /// For Beginners:
        /// Collocation points are locations in the domain where we enforce the PDE.
        /// Think of them as "checkpoints" where the neural network must satisfy the physics.
        ///
        /// Sampling Strategies:
        /// 1. Uniform grid: Simple but can miss important regions
        /// 2. Random sampling: Used here - good coverage, easy to implement
        /// 3. Latin Hypercube: Better space-filling properties
        /// 4. Adaptive sampling: Add more points where error is high
        ///
        /// For this implementation, we use random sampling in the unit hypercube [0,1]^d.
        /// You should scale these to your actual domain (e.g., x ∈ [-1, 1], t ∈ [0, 10]).
        /// </remarks>
        private void GenerateCollocationPoints()
        {
            int inputDim = _pdeSpecification.InputDimension;
            _collocationPoints = new T[_numCollocationPoints, inputDim];

            var random = RandomHelper.CreateSeededRandom(42); // Fixed seed for reproducibility

            for (int i = 0; i < _numCollocationPoints; i++)
            {
                for (int j = 0; j < inputDim; j++)
                {
                    // Random points in [0, 1]
                    // In practice, you'd scale this to your actual domain
                    _collocationPoints[i, j] = NumOps.FromDouble(random.NextDouble());
                }
            }
        }

        /// <summary>
        /// Sets custom collocation points (for advanced users who want specific sampling).
        /// </summary>
        /// <param name="points">Collocation points [numPoints, inputDim].</param>
        public void SetCollocationPoints(T[,] points)
        {
            if (points.GetLength(1) != _pdeSpecification.InputDimension)
            {
                throw new ArgumentException(
                    $"Collocation points must have {_pdeSpecification.InputDimension} dimensions.");
            }
            _collocationPoints = points;
        }

        /// <summary>
        /// Solves the PDE by training the PINN using automatic differentiation.
        /// </summary>
        /// <param name="dataInputs">Optional measured input data.</param>
        /// <param name="dataOutputs">Optional measured output data.</param>
        /// <param name="epochs">Number of training epochs.</param>
        /// <param name="learningRate">Learning rate for optimization.</param>
        /// <param name="verbose">Whether to print progress.</param>
        /// <param name="batchSize">Number of points per batch.</param>
        /// <returns>Training history (losses over epochs).</returns>
        /// <remarks>
        /// For Beginners:
        /// Training a PINN is like training a regular neural network, but with a special loss function.
        ///
        /// Training Process:
        /// 1. Sample collocation points
        /// 2. For each point:
        ///    a) Evaluate network: u = NN(x)
        ///    b) Compute derivatives using automatic differentiation: ∂u/∂x, ∂²u/∂x², etc.
        ///    c) Evaluate PDE residual: PDE(u, ∂u/∂x, ...)
        /// 3. Evaluate boundary and initial conditions
        /// 4. Compute total loss
        /// 5. Backpropagate and update network weights
        /// 6. Repeat
        ///
        /// This implementation uses GradientTape-based automatic differentiation for computing
        /// spatial derivatives (∂u/∂x), which is more accurate than finite differences.
        ///
        /// Tips for Success:
        /// - Start with simpler PDEs (heat, Poisson) before trying complex ones
        /// - Monitor individual loss components (data, PDE, BC, IC)
        /// - If one component dominates, adjust the weights
        /// - Learning rate scheduling can help
        /// - Sometimes training is unstable - try different architectures or optimizers
        /// </remarks>
        public TrainingHistory<T> Solve(
            T[,]? dataInputs = null,
            T[,]? dataOutputs = null,
            int epochs = 10000,
            double learningRate = 0.001,
            bool verbose = true,
            int batchSize = 256)
        {
            if (batchSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be positive.");
            }

            var history = new TrainingHistory<T>();
            int inputDim = _pdeSpecification.InputDimension;
            int outputDim = _pdeSpecification.OutputDimension;

            if ((dataInputs == null) != (dataOutputs == null))
            {
                throw new ArgumentException("Data inputs and outputs must both be provided or both be null.");
            }

            if (dataInputs != null && dataOutputs != null)
            {
                if (dataInputs.GetLength(0) != dataOutputs.GetLength(0))
                {
                    throw new ArgumentException("Data input and output sample counts must match.");
                }

                if (dataInputs.GetLength(1) != inputDim)
                {
                    throw new ArgumentException($"Data inputs must have {inputDim} dimensions.");
                }

                if (dataOutputs.GetLength(1) != outputDim)
                {
                    throw new ArgumentException($"Data outputs must have {outputDim} dimensions.");
                }
            }

            int collocationCount = _collocationPoints?.GetLength(0) ?? 0;
            int dataCount = dataInputs?.GetLength(0) ?? 0;
            int totalCount = collocationCount + dataCount;

            if (totalCount == 0)
            {
                throw new InvalidOperationException("No collocation points or data inputs are available for training.");
            }

            // Prepare all training points
            var allPoints = new T[totalCount, inputDim];
            var allTargets = new T[totalCount, outputDim];
            var hasTargets = new bool[totalCount];

            if (_collocationPoints != null)
            {
                for (int i = 0; i < collocationCount; i++)
                {
                    for (int j = 0; j < inputDim; j++)
                    {
                        allPoints[i, j] = _collocationPoints[i, j];
                    }
                }
            }

            if (dataInputs != null && dataOutputs != null)
            {
                int offset = collocationCount;
                for (int i = 0; i < dataCount; i++)
                {
                    for (int j = 0; j < inputDim; j++)
                    {
                        allPoints[offset + i, j] = dataInputs[i, j];
                    }

                    for (int j = 0; j < outputDim; j++)
                    {
                        allTargets[offset + i, j] = dataOutputs[i, j];
                    }

                    hasTargets[offset + i] = true;
                }
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
            foreach (var layer in Layers)
            {
                layer.SetTrainingMode(true);
            }

            try
            {
                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    T epochLossSum = NumOps.Zero;
                    int epochCount = 0;

                    for (int batchStart = 0; batchStart < totalCount; batchStart += batchSize)
                    {
                        int batchCount = Math.Min(batchSize, totalCount - batchStart);
                        T batchScale = NumOps.Divide(NumOps.One, NumOps.FromDouble(batchCount));

                        // Build batch input tensor for forward pass
                        var batchInput = new Tensor<T>([batchCount, inputDim]);
                        for (int b = 0; b < batchCount; b++)
                        {
                            int row = batchStart + b;
                            for (int j = 0; j < inputDim; j++)
                            {
                                batchInput[b, j] = allPoints[row, j];
                            }
                        }

                        // Forward pass
                        var outputs = ForwardWithMemory(batchInput);

                        if (outputs.Shape[1] != outputDim)
                        {
                            throw new InvalidOperationException(
                                $"Expected {outputDim} outputs from the network, got {outputs.Shape[1]}.");
                        }

                        var outputGradients = new Tensor<T>(outputs.Shape);
                        var point = new T[inputDim];
                        var output = new T[outputDim];
                        var targetBuffer = new T[outputDim];

                        // Process each point in the batch using autodiff derivatives
                        for (int b = 0; b < batchCount; b++)
                        {
                            int row = batchStart + b;

                            // Extract point coordinates
                            for (int j = 0; j < inputDim; j++)
                            {
                                point[j] = allPoints[row, j];
                            }

                            // Extract network output for this point
                            for (int j = 0; j < outputDim; j++)
                            {
                                output[j] = outputs[b, j];
                            }

                            // Compute derivatives using automatic differentiation
                            // This uses GradientTape with analytic fallback, avoiding finite differences
                            var derivatives = NeuralNetworkDerivatives<T>.ComputeDerivatives(
                                this,
                                point,
                                outputDim);

                            // Prepare targets if available
                            T[]? targets = null;
                            if (hasTargets[row])
                            {
                                for (int j = 0; j < outputDim; j++)
                                {
                                    targetBuffer[j] = allTargets[row, j];
                                }
                                targets = targetBuffer;
                            }

                            // Compute physics loss and gradients
                            var lossGradients = _physicsLoss.ComputePhysicsLossGradients(
                                output,
                                targets,
                                derivatives,
                                point);

                            epochLossSum = NumOps.Add(epochLossSum, lossGradients.Loss);
                            epochCount++;

                            // Accumulate output gradients (scaled by batch size for averaging)
                            for (int outIdx = 0; outIdx < outputDim; outIdx++)
                            {
                                T grad = NumOps.Multiply(batchScale, lossGradients.OutputGradients[outIdx]);
                                outputGradients[b, outIdx] = NumOps.Add(outputGradients[b, outIdx], grad);
                            }
                        }

                        // Backpropagate through network and update parameters
                        Backpropagate(outputGradients);
                        _optimizer.UpdateParameters(Layers);
                    }

                    T avgLoss = epochCount > 0
                        ? NumOps.Divide(epochLossSum, NumOps.FromDouble(epochCount))
                        : NumOps.Zero;

                    LastLoss = avgLoss;
                    history.AddEpoch(avgLoss);

                    if (verbose && epoch % 100 == 0)
                    {
                        Console.WriteLine($"Epoch {epoch}/{epochs}, Loss: {avgLoss}");
                    }
                }
            }
            finally
            {
                foreach (var layer in Layers)
                {
                    layer.SetTrainingMode(false);
                }

                SetTrainingMode(false);
            }

            return history;
        }

        private T ComputeAverageLoss(T[,]? dataInputs, T[,]? dataOutputs)
        {
            T totalLoss = NumOps.Zero;
            int sampleCount = 0;

            if ((dataInputs == null) != (dataOutputs == null))
            {
                throw new ArgumentException("Data inputs and outputs must both be provided or both be null.");
            }

            if (dataInputs != null && dataOutputs != null)
            {
                if (dataInputs.GetLength(0) != dataOutputs.GetLength(0))
                {
                    throw new ArgumentException("Data input and output sample counts must match.");
                }

                if (dataInputs.GetLength(1) != _pdeSpecification.InputDimension)
                {
                    throw new ArgumentException(
                        $"Data inputs must have {_pdeSpecification.InputDimension} dimensions.");
                }

                if (dataOutputs.GetLength(1) != _pdeSpecification.OutputDimension)
                {
                    throw new ArgumentException(
                        $"Data outputs must have {_pdeSpecification.OutputDimension} dimensions.");
                }
            }

            if (_collocationPoints != null)
            {
                var collocationTensor = CreateBatchTensor(_collocationPoints);
                var collocationOutputs = Forward(collocationTensor);
                int outputDim = _pdeSpecification.OutputDimension;

                if (collocationOutputs.Shape[1] != outputDim)
                {
                    throw new InvalidOperationException(
                        $"Expected {outputDim} outputs from the network, got {collocationOutputs.Shape[1]}.");
                }

                int inputDim = _pdeSpecification.InputDimension;
                var point = new T[inputDim];
                var output = new T[outputDim];

                for (int i = 0; i < _collocationPoints.GetLength(0); i++)
                {
                    for (int j = 0; j < inputDim; j++)
                    {
                        point[j] = _collocationPoints[i, j];
                    }

                    for (int j = 0; j < outputDim; j++)
                    {
                        output[j] = collocationOutputs[i, j];
                    }

                    var derivatives = NeuralNetworkDerivatives<T>.ComputeDerivatives(
                        this,
                        point,
                        _pdeSpecification.OutputDimension);

                    T loss = _physicsLoss.ComputePhysicsLoss(output, null, derivatives, point);
                    totalLoss = NumOps.Add(totalLoss, loss);
                    sampleCount++;
                }
            }

            if (dataInputs != null && dataOutputs != null)
            {
                var dataTensor = CreateBatchTensor(dataInputs);
                var dataPredictions = Forward(dataTensor);
                int outputDim = _pdeSpecification.OutputDimension;

                if (dataPredictions.Shape[1] != outputDim)
                {
                    throw new InvalidOperationException(
                        $"Expected {outputDim} outputs from the network, got {dataPredictions.Shape[1]}.");
                }

                var point = new T[dataInputs.GetLength(1)];
                var target = new T[dataOutputs.GetLength(1)];
                var output = new T[outputDim];

                for (int i = 0; i < dataInputs.GetLength(0); i++)
                {
                    for (int j = 0; j < point.Length; j++)
                    {
                        point[j] = dataInputs[i, j];
                    }

                    for (int j = 0; j < target.Length; j++)
                    {
                        target[j] = dataOutputs[i, j];
                    }

                    for (int j = 0; j < outputDim; j++)
                    {
                        output[j] = dataPredictions[i, j];
                    }

                    var derivatives = NeuralNetworkDerivatives<T>.ComputeDerivatives(
                        this,
                        point,
                        _pdeSpecification.OutputDimension);

                    T loss = _physicsLoss.ComputePhysicsLoss(output, target, derivatives, point);
                    totalLoss = NumOps.Add(totalLoss, loss);
                    sampleCount++;
                }
            }

            return sampleCount > 0
                ? NumOps.Divide(totalLoss, NumOps.FromDouble(sampleCount))
                : NumOps.Zero;
        }

        /// <summary>
        /// Performs a forward pass through the network.
        /// </summary>
        /// <param name="input">Input tensor for evaluation.</param>
        /// <returns>Network output tensor.</returns>
        public Tensor<T> Forward(Tensor<T> input)
        {
            Tensor<T> output = input;
            foreach (var layer in Layers)
            {
                output = layer.Forward(output);
            }

            return output;
        }

        private Tensor<T> CreateBatchTensor(T[,] inputs)
        {
            int rows = inputs.GetLength(0);
            int cols = inputs.GetLength(1);
            var tensor = new Tensor<T>(new int[] { rows, cols });

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    tensor[i, j] = inputs[i, j];
                }
            }

            return tensor;
        }

        /// <summary>
        /// Evaluates the network at a single point.
        /// </summary>
        private T[] EvaluateAtPoint(T[] inputs)
        {
            var inputTensor = new Tensor<T>(new int[] { 1, inputs.Length });
            for (int i = 0; i < inputs.Length; i++)
            {
                inputTensor[0, i] = inputs[i];
            }

            var outputTensor = Forward(inputTensor);

            T[] result = new T[outputTensor.Shape[1]];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = outputTensor[0, i];
            }

            return result;
        }

        /// <summary>
        /// Gets the solution at a specific point in the domain.
        /// </summary>
        /// <param name="point">The point coordinates (x, y, t, etc.).</param>
        /// <returns>The solution value(s) at that point.</returns>
        public T[] GetSolution(T[] point)
        {
            return EvaluateAtPoint(point);
        }

        /// <summary>
        /// Evaluates the PDE residual at a point (for validation).
        /// </summary>
        /// <param name="point">The point coordinates.</param>
        /// <returns>The PDE residual (should be close to zero for a good solution).</returns>
        public T EvaluatePDEResidual(T[] point)
        {
            T[] output = EvaluateAtPoint(point);
            var derivatives = NeuralNetworkDerivatives<T>.ComputeDerivatives(
                this,
                point,
                _pdeSpecification.OutputDimension);

            return _pdeSpecification.ComputeResidual(point, output, derivatives);
        }

        /// <summary>
        /// Sums two sets of PDE derivatives element-wise.
        /// Used by derived classes (e.g., MultiFidelityPINN) to compute derivatives of combined solutions.
        /// </summary>
        /// <param name="a">First set of derivatives.</param>
        /// <param name="b">Second set of derivatives.</param>
        /// <returns>Combined derivatives where each element is the sum of corresponding elements.</returns>
        protected PDEDerivatives<T> SumDerivatives(PDEDerivatives<T> a, PDEDerivatives<T> b)
        {
            var result = new PDEDerivatives<T>();

            // Sum first derivatives
            if (a.FirstDerivatives is not null && b.FirstDerivatives is not null)
            {
                int dim0 = a.FirstDerivatives.GetLength(0);
                int dim1 = a.FirstDerivatives.GetLength(1);

                if (b.FirstDerivatives.GetLength(0) != dim0 || b.FirstDerivatives.GetLength(1) != dim1)
                {
                    throw new ArgumentException(
                        $"First derivatives dimension mismatch: a has [{dim0}, {dim1}], b has [{b.FirstDerivatives.GetLength(0)}, {b.FirstDerivatives.GetLength(1)}].");
                }

                result.FirstDerivatives = new T[dim0, dim1];
                for (int i = 0; i < dim0; i++)
                {
                    for (int j = 0; j < dim1; j++)
                    {
                        result.FirstDerivatives[i, j] = NumOps.Add(a.FirstDerivatives[i, j], b.FirstDerivatives[i, j]);
                    }
                }
            }
            else
            {
                result.FirstDerivatives = a.FirstDerivatives ?? b.FirstDerivatives;
            }

            // Sum second derivatives
            if (a.SecondDerivatives is not null && b.SecondDerivatives is not null)
            {
                int dim0 = a.SecondDerivatives.GetLength(0);
                int dim1 = a.SecondDerivatives.GetLength(1);
                int dim2 = a.SecondDerivatives.GetLength(2);

                if (b.SecondDerivatives.GetLength(0) != dim0 || b.SecondDerivatives.GetLength(1) != dim1 || b.SecondDerivatives.GetLength(2) != dim2)
                {
                    throw new ArgumentException(
                        $"Second derivatives dimension mismatch: a has [{dim0}, {dim1}, {dim2}], b has [{b.SecondDerivatives.GetLength(0)}, {b.SecondDerivatives.GetLength(1)}, {b.SecondDerivatives.GetLength(2)}].");
                }

                result.SecondDerivatives = new T[dim0, dim1, dim2];
                for (int i = 0; i < dim0; i++)
                {
                    for (int j = 0; j < dim1; j++)
                    {
                        for (int k = 0; k < dim2; k++)
                        {
                            result.SecondDerivatives[i, j, k] = NumOps.Add(a.SecondDerivatives[i, j, k], b.SecondDerivatives[i, j, k]);
                        }
                    }
                }
            }
            else
            {
                result.SecondDerivatives = a.SecondDerivatives ?? b.SecondDerivatives;
            }

            // Sum third derivatives if present
            if (a.ThirdDerivatives is not null && b.ThirdDerivatives is not null)
            {
                int dim0 = a.ThirdDerivatives.GetLength(0);
                int dim1 = a.ThirdDerivatives.GetLength(1);
                int dim2 = a.ThirdDerivatives.GetLength(2);
                int dim3 = a.ThirdDerivatives.GetLength(3);

                if (b.ThirdDerivatives.GetLength(0) != dim0 || b.ThirdDerivatives.GetLength(1) != dim1 ||
                    b.ThirdDerivatives.GetLength(2) != dim2 || b.ThirdDerivatives.GetLength(3) != dim3)
                {
                    throw new ArgumentException(
                        $"Third derivatives dimension mismatch: a has [{dim0}, {dim1}, {dim2}, {dim3}], b has [{b.ThirdDerivatives.GetLength(0)}, {b.ThirdDerivatives.GetLength(1)}, {b.ThirdDerivatives.GetLength(2)}, {b.ThirdDerivatives.GetLength(3)}].");
                }

                result.ThirdDerivatives = new T[dim0, dim1, dim2, dim3];
                for (int i = 0; i < dim0; i++)
                {
                    for (int j = 0; j < dim1; j++)
                    {
                        for (int k = 0; k < dim2; k++)
                        {
                            for (int l = 0; l < dim3; l++)
                            {
                                result.ThirdDerivatives[i, j, k, l] = NumOps.Add(a.ThirdDerivatives[i, j, k, l], b.ThirdDerivatives[i, j, k, l]);
                            }
                        }
                    }
                }
            }
            else
            {
                result.ThirdDerivatives = a.ThirdDerivatives ?? b.ThirdDerivatives;
            }

            return result;
        }

        /// <summary>
        /// Makes a prediction using the PINN.
        /// </summary>
        /// <param name="input">Input tensor.</param>
        /// <returns>Predicted output tensor.</returns>
        public override Tensor<T> Predict(Tensor<T> input)
        {
            bool wasTraining = IsTrainingMode;
            SetTrainingMode(false);

            try
            {
                return Forward(input);
            }
            finally
            {
                SetTrainingMode(wasTraining);
            }
        }

        /// <summary>
        /// Updates the network parameters from a flattened vector.
        /// </summary>
        /// <param name="parameters">Parameter vector.</param>
        public override void UpdateParameters(Vector<T> parameters)
        {
            int index = 0;
            foreach (var layer in Layers)
            {
                int layerParameterCount = layer.ParameterCount;
                if (layerParameterCount > 0)
                {
                    Vector<T> layerParameters = parameters.GetSubVector(index, layerParameterCount);
                    layer.UpdateParameters(layerParameters);
                    index += layerParameterCount;
                }
            }
        }

        /// <summary>
        /// Performs a basic supervised training step using MSE loss.
        /// </summary>
        /// <param name="input">Training input tensor.</param>
        /// <param name="expectedOutput">Expected output tensor.</param>
        public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            SetTrainingMode(true);

            var prediction = Forward(input);
            var lossFunction = LossFunction ?? new MeanSquaredErrorLoss<T>();
            LastLoss = lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

            var outputGradient = lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
            var outputGradientTensor = Tensor<T>.FromVector(outputGradient).Reshape(prediction.Shape);

            Backpropagate(outputGradientTensor);
            _optimizer.UpdateParameters(Layers);

            SetTrainingMode(false);
        }

        /// <summary>
        /// Gets metadata about the PINN model.
        /// </summary>
        /// <returns>Model metadata.</returns>
        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                ModelType = ModelType.NeuralNetwork,
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "PDE", _pdeSpecification.Name },
                    { "InputDimension", _pdeSpecification.InputDimension },
                    { "OutputDimension", _pdeSpecification.OutputDimension },
                    { "BoundaryConditionCount", _boundaryConditions.Length },
                    { "HasInitialCondition", _initialCondition != null },
                    { "ParameterCount", GetParameterCount() }
                },
                ModelData = Serialize()
            };
        }

        /// <summary>
        /// Serializes PINN-specific data.
        /// </summary>
        /// <param name="writer">Binary writer.</param>
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            writer.Write(_numCollocationPoints);
            writer.Write(_pdeSpecification.InputDimension);
            writer.Write(_pdeSpecification.OutputDimension);

            if (_collocationPoints == null)
            {
                writer.Write(false);
                return;
            }

            writer.Write(true);
            writer.Write(_collocationPoints.GetLength(0));
            writer.Write(_collocationPoints.GetLength(1));

            for (int i = 0; i < _collocationPoints.GetLength(0); i++)
            {
                for (int j = 0; j < _collocationPoints.GetLength(1); j++)
                {
                    SerializationHelper<T>.WriteValue(writer, _collocationPoints[i, j]);
                }
            }
        }

        /// <summary>
        /// Deserializes PINN-specific data.
        /// </summary>
        /// <param name="reader">Binary reader.</param>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            int storedCollocationCount = reader.ReadInt32();
            int storedInputDim = reader.ReadInt32();
            int storedOutputDim = reader.ReadInt32();

            if (storedCollocationCount != _numCollocationPoints ||
                storedInputDim != _pdeSpecification.InputDimension ||
                storedOutputDim != _pdeSpecification.OutputDimension)
            {
                throw new InvalidOperationException("Serialized PINN configuration does not match the current instance.");
            }

            bool hasPoints = reader.ReadBoolean();
            if (!hasPoints)
            {
                _collocationPoints = null;
                return;
            }

            int rows = reader.ReadInt32();
            int cols = reader.ReadInt32();
            _collocationPoints = new T[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    _collocationPoints[i, j] = SerializationHelper<T>.ReadValue(reader);
                }
            }
        }

        /// <summary>
        /// Creates a new instance with the same configuration.
        /// </summary>
        /// <returns>New PINN instance.</returns>
        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new PhysicsInformedNeuralNetwork<T>(
                Architecture,
                _pdeSpecification,
                _boundaryConditions,
                _initialCondition,
                _numCollocationPoints,
                _optimizer);
        }

        /// <summary>
        /// Indicates whether this PINN supports training.
        /// </summary>
        public override bool SupportsTraining => true;
    }

}


