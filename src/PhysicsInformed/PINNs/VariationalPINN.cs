using System;
using System.IO;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.PhysicsInformed.PINNs
{
    /// <summary>
    /// Implements Variational Physics-Informed Neural Networks (VPINNs).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// Variational PINNs (VPINNs) use the weak (variational) formulation of PDEs instead of
    /// the strong form. This is similar to finite element methods (FEM).
    ///
    /// Strong vs. Weak Formulation:
    ///
    /// Strong Form (standard PINN):
    /// - PDE must hold pointwise: PDE(u) = 0 at every point
    /// - Example: -∇²u = f everywhere
    /// - Requires computing second derivatives
    /// - Solution must be twice differentiable
    ///
    /// Weak Form (VPINN):
    /// - PDE holds "on average" against test functions
    /// - ∫∇u·∇v dx = ∫fv dx for all test functions v
    /// - Integration by parts reduces derivative order
    /// - Solution only needs to be once differentiable
    /// - More stable numerically
    ///
    /// Key Advantages:
    /// 1. Lower derivative requirements (better numerical stability)
    /// 2. Natural incorporation of boundary conditions (through integration by parts)
    /// 3. Can handle discontinuities and rough solutions better
    /// 4. Closer to FEM (well-understood mathematical theory)
    /// 5. Often better convergence properties
    ///
    /// How VPINNs Work:
    /// 1. Choose test functions (often neural networks themselves)
    /// 2. Multiply PDE by test function and integrate
    /// 3. Use integration by parts to reduce derivative order
    /// 4. Minimize the residual in the weak sense
    ///
    /// Example - Poisson Equation:
    /// Strong: -∇²u = f
    /// Weak: ∫∇u·∇v dx = ∫fv dx (after integration by parts)
    ///
    /// VPINNs train the network u(x) to satisfy the weak form for all test functions v.
    ///
    /// Applications:
    /// - Same as PINNs, but particularly useful for:
    ///   * Problems with rough solutions
    ///   * Conservation laws
    ///   * Problems where weak solutions are more natural
    ///   * High-order PDEs (where reducing derivative order helps)
    ///
    /// Comparison with Standard PINNs:
    /// - VPINN: More stable, lower derivative requirements, closer to FEM
    /// - Standard PINN: Simpler to implement, direct enforcement of PDE
    ///
    /// The variational formulation often provides better training dynamics and accuracy.
    /// </remarks>
    public class VariationalPINN<T> : NeuralNetworkBase<T>
    {
        private readonly Func<T[], T[], T[,], T[], T[,], T> _weakFormResidual;
        private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
        private readonly bool _usesDefaultOptimizer;
        private T[,]? _quadraturePoints;
        private T[]? _quadratureWeights;
        private readonly int _numTestFunctions;
        private readonly int _numQuadraturePoints;

        /// <summary>
        /// Initializes a new instance of the Variational PINN.
        /// </summary>
        /// <param name="architecture">The neural network architecture for the solution.</param>
        /// <param name="weakFormResidual">The weak form residual: R(x, u, ∇u, v, ∇v).</param>
        /// <param name="numQuadraturePoints">Number of quadrature points for integration.</param>
        /// <param name="numTestFunctions">Number of test functions to use.</param>
        /// <remarks>
        /// For Beginners:
        /// The weak form residual should encode the variational formulation of your PDE.
        ///
        /// Example - Poisson Equation (-∇²u = f):
        /// Weak form: ∫∇u·∇v dx - ∫fv dx = 0
        /// weakFormResidual = (x, u, grad_u, v, grad_v) => {
        ///     T term1 = DotProduct(grad_u, grad_v);  // ∇u·∇v
        ///     T term2 = f(x) * v;                     // fv
        ///     return term1 - term2;
        /// }
        ///
        /// The method integrates this over the domain using numerical quadrature.
        /// </remarks>
        public VariationalPINN(
            NeuralNetworkArchitecture<T> architecture,
            Func<T[], T[], T[,], T[], T[,], T> weakFormResidual,
            int numQuadraturePoints = 10000,
            int numTestFunctions = 10)
            : base(architecture, NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), 1.0)
        {
            _weakFormResidual = weakFormResidual ?? throw new ArgumentNullException(nameof(weakFormResidual));
            _numTestFunctions = numTestFunctions;
            _numQuadraturePoints = numQuadraturePoints;
            _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
            _usesDefaultOptimizer = true;

            InitializeLayers();
            GenerateQuadraturePoints(_numQuadraturePoints, architecture.InputSize);
        }

        protected override void InitializeLayers()
        {
            if (Architecture.Layers != null && Architecture.Layers.Count > 0)
            {
                Layers.AddRange(Architecture.Layers);
                ValidateCustomLayers(Layers);
            }
            else
            {
                Layers.AddRange(LayerHelper<T>.CreateDefaultVariationalPINNLayers(Architecture));
            }
        }

        private void GenerateQuadraturePoints(int numPoints, int dimension)
        {
            _quadraturePoints = new T[numPoints, dimension];
            _quadratureWeights = new T[numPoints];

            var random = RandomHelper.CreateSeededRandom(42);
            T weight = NumOps.Divide(NumOps.One, NumOps.FromDouble(numPoints));

            for (int i = 0; i < numPoints; i++)
            {
                for (int j = 0; j < dimension; j++)
                {
                    _quadraturePoints[i, j] = NumOps.FromDouble(random.NextDouble());
                }
                _quadratureWeights[i] = weight;
            }
        }

        /// <summary>
        /// Computes the weak form residual by integrating over the domain.
        /// </summary>
        /// <param name="testFunctionIndex">Index of the test function to use.</param>
        /// <returns>The weak residual (should be zero for a perfect solution).</returns>
        /// <remarks>
        /// For Beginners:
        /// This computes ∫R(u, v)dx where:
        /// - u is the neural network solution
        /// - v is a test function
        /// - R is the weak form residual
        ///
        /// For a true solution, this integral should be zero for ALL test functions.
        /// We approximate "all" by using a finite set of test functions.
        ///
        /// Test Function Choices:
        /// 1. Polynomial basis (Legendre, Chebyshev)
        /// 2. Trigonometric functions (Fourier)
        /// 3. Another neural network
        /// 4. Random functions
        ///
        /// This implementation uses simple polynomial test functions.
        /// </remarks>
        public T ComputeWeakResidual(int testFunctionIndex)
        {
            if (_quadraturePoints == null || _quadratureWeights == null)
            {
                throw new InvalidOperationException("Quadrature points not initialized.");
            }

            T residual = NumOps.Zero;

            for (int i = 0; i < _quadraturePoints.GetLength(0); i++)
            {
                T[] point = new T[_quadraturePoints.GetLength(1)];
                for (int j = 0; j < point.Length; j++)
                {
                    point[j] = _quadraturePoints[i, j];
                }

                // Evaluate solution
                T[] u = EvaluateAtPoint(point);
                T[,] gradU = NeuralNetworkDerivatives<T>.ComputeGradient(
                    this,
                    point,
                    Architecture.OutputSize);

                // Evaluate test function
                T[] v = EvaluateTestFunction(point, testFunctionIndex);
                T[,] gradV = ComputeTestFunctionGradient(point, testFunctionIndex);

                // Compute weak form residual at this point
                T localResidual = _weakFormResidual(point, u, gradU, v, gradV);

                // Integrate
                residual = NumOps.Add(residual, NumOps.Multiply(_quadratureWeights[i], localResidual));
            }

            return residual;
        }

        /// <summary>
        /// Evaluates a test function (simple polynomial basis for demonstration).
        /// </summary>
        private T[] EvaluateTestFunction(T[] point, int index)
        {
            // Simple polynomial test functions: v_i(x) = x₁^i₁ * x₂^i₂ * ...
            // This is a basic implementation; in practice, you'd use better basis functions

            T[] result = new T[Architecture.OutputSize];

            // Generate multi-index based on index
            int[] multiIndex = GenerateMultiIndex(index, point.Length);

            for (int k = 0; k < result.Length; k++)
            {
                T value = NumOps.One;
                for (int j = 0; j < point.Length; j++)
                {
                    // Compute point[j]^multiIndex[j]
                    for (int p = 0; p < multiIndex[j]; p++)
                    {
                        value = NumOps.Multiply(value, point[j]);
                    }
                }
                result[k] = value;
            }

            return result;
        }

        /// <summary>
        /// Computes the gradient of a test function.
        /// </summary>
        private T[,] ComputeTestFunctionGradient(T[] point, int index)
        {
            int outputSize = Architecture.OutputSize;
            int inputDim = point.Length;
            var gradient = new T[outputSize, inputDim];

            int[] multiIndex = GenerateMultiIndex(index, inputDim);

            for (int outIdx = 0; outIdx < outputSize; outIdx++)
            {
                for (int d = 0; d < inputDim; d++)
                {
                    if (multiIndex[d] == 0)
                    {
                        gradient[outIdx, d] = NumOps.Zero;
                        continue;
                    }

                    T value = NumOps.FromDouble(multiIndex[d]);
                    for (int j = 0; j < inputDim; j++)
                    {
                        int power = multiIndex[j] - (j == d ? 1 : 0);
                        for (int p = 0; p < power; p++)
                        {
                            value = NumOps.Multiply(value, point[j]);
                        }
                    }

                    gradient[outIdx, d] = value;
                }
            }

            return gradient;
        }

        /// <summary>
        /// Generates a multi-index for polynomial basis.
        /// </summary>
        private int[] GenerateMultiIndex(int index, int dimension)
        {
            int[] multiIndex = new int[dimension];
            int remaining = index;

            for (int i = 0; i < dimension; i++)
            {
                multiIndex[i] = remaining % 3; // Use powers 0, 1, 2
                remaining /= 3;
            }

            return multiIndex;
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

        private Tensor<T> BuildEvaluationTensor(T[,] basePoints, int start, int count, T step)
        {
            int inputDim = basePoints.GetLength(1);
            int evalPerBase = 1 + (2 * inputDim);
            var tensor = new Tensor<T>(new int[] { count * evalPerBase, inputDim });

            int row = 0;
            for (int b = 0; b < count; b++)
            {
                int baseRow = start + b;
                CopyPoint(basePoints, baseRow, tensor, row++);

                for (int dim = 0; dim < inputDim; dim++)
                {
                    CopyPoint(basePoints, baseRow, tensor, row);
                    tensor[row, dim] = NumOps.Add(tensor[row, dim], step);
                    row++;

                    CopyPoint(basePoints, baseRow, tensor, row);
                    tensor[row, dim] = NumOps.Subtract(tensor[row, dim], step);
                    row++;
                }
            }

            return tensor;
        }

        private static void CopyPoint(T[,] source, int sourceRow, Tensor<T> target, int targetRow)
        {
            int inputDim = source.GetLength(1);
            for (int dim = 0; dim < inputDim; dim++)
            {
                target[targetRow, dim] = source[sourceRow, dim];
            }
        }

        private (T Residual, T[] OutputGradients, T[,] GradientGradients) ComputeWeakResidualGradient(
            T[] point,
            T[] outputs,
            T[,] gradients,
            T[] testFunction,
            T[,] testGradient)
        {
            int outputDim = outputs.Length;
            int inputDim = gradients.GetLength(1);

            var outputGradients = new T[outputDim];
            var gradientGradients = new T[outputDim, inputDim];

            T residual = _weakFormResidual(point, outputs, gradients, testFunction, testGradient);
            T eps = NumOps.FromDouble(1e-4);
            T invTwoEps = NumOps.Divide(NumOps.One, NumOps.Multiply(NumOps.FromDouble(2.0), eps));

            var outputsCopy = new T[outputDim];
            Array.Copy(outputs, outputsCopy, outputDim);

            var gradientsCopy = new T[outputDim, inputDim];
            Array.Copy(gradients, gradientsCopy, gradients.Length);

            for (int outIdx = 0; outIdx < outputDim; outIdx++)
            {
                T original = outputsCopy[outIdx];
                outputsCopy[outIdx] = NumOps.Add(original, eps);
                T plus = _weakFormResidual(point, outputsCopy, gradientsCopy, testFunction, testGradient);
                outputsCopy[outIdx] = NumOps.Subtract(original, eps);
                T minus = _weakFormResidual(point, outputsCopy, gradientsCopy, testFunction, testGradient);
                outputsCopy[outIdx] = original;

                outputGradients[outIdx] = NumOps.Multiply(NumOps.Subtract(plus, minus), invTwoEps);
            }

            for (int outIdx = 0; outIdx < outputDim; outIdx++)
            {
                for (int dim = 0; dim < inputDim; dim++)
                {
                    T original = gradientsCopy[outIdx, dim];
                    gradientsCopy[outIdx, dim] = NumOps.Add(original, eps);
                    T plus = _weakFormResidual(point, outputsCopy, gradientsCopy, testFunction, testGradient);
                    gradientsCopy[outIdx, dim] = NumOps.Subtract(original, eps);
                    T minus = _weakFormResidual(point, outputsCopy, gradientsCopy, testFunction, testGradient);
                    gradientsCopy[outIdx, dim] = original;

                    gradientGradients[outIdx, dim] = NumOps.Multiply(NumOps.Subtract(plus, minus), invTwoEps);
                }
            }

            return (residual, outputGradients, gradientGradients);
        }

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
        /// Trains the network to minimize the weak residual.
        /// </summary>
        /// <param name="epochs">Number of training epochs.</param>
        /// <param name="learningRate">Learning rate for optimization.</param>
        /// <param name="verbose">Whether to print progress.</param>
        /// <param name="batchSize">Number of quadrature points per batch.</param>
        /// <param name="derivativeStep">Finite-difference step size for input derivatives.</param>
        public TrainingHistory<T> Solve(
            int epochs = 1000,
            double learningRate = 0.001,
            bool verbose = true,
            int batchSize = 256,
            double derivativeStep = 1e-4)
        {
            if (_quadraturePoints == null || _quadratureWeights == null)
            {
                throw new InvalidOperationException("Quadrature points not initialized.");
            }

            if (batchSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be positive.");
            }

            if (derivativeStep <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(derivativeStep), "Derivative step must be positive.");
            }

            int inputDim = Architecture.InputSize;
            int outputDim = Architecture.OutputSize;
            int totalCount = _quadraturePoints.GetLength(0);

            if (totalCount == 0)
            {
                throw new InvalidOperationException("Quadrature points are empty.");
            }

            var history = new TrainingHistory<T>();
            T step = NumOps.FromDouble(derivativeStep);
            int evalPerBase = 1 + (2 * inputDim);

            if (_usesDefaultOptimizer)
            {
                var options = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    LearningRate = learningRate
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
                    int batchCounter = 0;

                    for (int batchStart = 0; batchStart < totalCount; batchStart += batchSize)
                    {
                        int batchCount = Math.Min(batchSize, totalCount - batchStart);
                        var evaluationInputs = BuildEvaluationTensor(_quadraturePoints, batchStart, batchCount, step);
                        var outputs = ForwardWithMemory(evaluationInputs);

                        if (outputs.Shape[1] != outputDim)
                        {
                            throw new InvalidOperationException(
                                $"Expected {outputDim} outputs from the network, got {outputs.Shape[1]}.");
                        }
                        var outputGradients = new Tensor<T>(outputs.Shape);

                        T invTwoStep = NumOps.Divide(NumOps.One, NumOps.Multiply(NumOps.FromDouble(2.0), step));
                        T batchWeightSum = NumOps.Zero;

                        for (int i = 0; i < batchCount; i++)
                        {
                            batchWeightSum = NumOps.Add(batchWeightSum, _quadratureWeights[batchStart + i]);
                        }

                        T weightNormalizer = NumOps.Divide(NumOps.One, batchWeightSum);

                        var baseOutputs = new T[batchCount, outputDim];
                        var gradUs = new T[batchCount, outputDim, inputDim];

                        for (int b = 0; b < batchCount; b++)
                        {
                            int baseOffset = b * evalPerBase;

                            for (int outIdx = 0; outIdx < outputDim; outIdx++)
                            {
                                baseOutputs[b, outIdx] = outputs[baseOffset, outIdx];
                                for (int dim = 0; dim < inputDim; dim++)
                                {
                                    int plusIndex = baseOffset + 1 + (2 * dim);
                                    int minusIndex = plusIndex + 1;
                                    T plus = outputs[plusIndex, outIdx];
                                    T minus = outputs[minusIndex, outIdx];
                                    gradUs[b, outIdx, dim] = NumOps.Multiply(NumOps.Subtract(plus, minus), invTwoStep);
                                }
                            }
                        }

                        T batchLoss = NumOps.Zero;
                        for (int testIdx = 0; testIdx < _numTestFunctions; testIdx++)
                        {
                            T residualSum = NumOps.Zero;
                            var outputResidualGrads = new T[batchCount, outputDim];
                            var gradResidualGrads = new T[batchCount, outputDim, inputDim];

                            for (int b = 0; b < batchCount; b++)
                            {
                                int baseRow = batchStart + b;
                                int baseOffset = b * evalPerBase;
                                var point = new T[inputDim];

                                for (int j = 0; j < inputDim; j++)
                                {
                                    point[j] = _quadraturePoints[baseRow, j];
                                }

                                var u = new T[outputDim];
                                var gradU = new T[outputDim, inputDim];
                                for (int outIdx = 0; outIdx < outputDim; outIdx++)
                                {
                                    u[outIdx] = baseOutputs[b, outIdx];
                                    for (int dim = 0; dim < inputDim; dim++)
                                    {
                                        gradU[outIdx, dim] = gradUs[b, outIdx, dim];
                                    }
                                }

                                T[] v = EvaluateTestFunction(point, testIdx);
                                T[,] gradV = ComputeTestFunctionGradient(point, testIdx);
                                var (localResidual, outputGrad, gradGrad) = ComputeWeakResidualGradient(point, u, gradU, v, gradV);

                                T weight = NumOps.Multiply(_quadratureWeights[baseRow], weightNormalizer);
                                residualSum = NumOps.Add(residualSum, NumOps.Multiply(weight, localResidual));

                                for (int outIdx = 0; outIdx < outputDim; outIdx++)
                                {
                                    outputResidualGrads[b, outIdx] = NumOps.Multiply(weight, outputGrad[outIdx]);
                                    for (int dim = 0; dim < inputDim; dim++)
                                    {
                                        gradResidualGrads[b, outIdx, dim] = NumOps.Multiply(weight, gradGrad[outIdx, dim]);
                                    }
                                }
                            }

                            T lossScale = NumOps.Multiply(
                                NumOps.Divide(NumOps.FromDouble(2.0), NumOps.FromDouble(_numTestFunctions)),
                                residualSum);

                            batchLoss = NumOps.Add(
                                batchLoss,
                                NumOps.Multiply(
                                    NumOps.Divide(NumOps.One, NumOps.FromDouble(_numTestFunctions)),
                                    NumOps.Multiply(residualSum, residualSum)));

                            for (int b = 0; b < batchCount; b++)
                            {
                                int baseOffset = b * evalPerBase;

                                for (int outIdx = 0; outIdx < outputDim; outIdx++)
                                {
                                    T outputContribution = NumOps.Multiply(lossScale, outputResidualGrads[b, outIdx]);
                                    outputGradients[baseOffset, outIdx] = NumOps.Add(outputGradients[baseOffset, outIdx], outputContribution);

                                    for (int dim = 0; dim < inputDim; dim++)
                                    {
                                        int plusIndex = baseOffset + 1 + (2 * dim);
                                        int minusIndex = plusIndex + 1;

                                        T gradContribution = NumOps.Multiply(lossScale, gradResidualGrads[b, outIdx, dim]);
                                        T delta = NumOps.Multiply(gradContribution, invTwoStep);
                                        outputGradients[plusIndex, outIdx] = NumOps.Add(outputGradients[plusIndex, outIdx], delta);
                                        outputGradients[minusIndex, outIdx] = NumOps.Subtract(outputGradients[minusIndex, outIdx], delta);
                                    }
                                }
                            }
                        }

                        Backpropagate(outputGradients);
                        _optimizer.UpdateParameters(Layers);

                        epochLossSum = NumOps.Add(epochLossSum, batchLoss);
                        batchCounter++;
                    }

                    T avgLoss = batchCounter > 0
                        ? NumOps.Divide(epochLossSum, NumOps.FromDouble(batchCounter))
                        : NumOps.Zero;

                    LastLoss = avgLoss;
                    history.AddEpoch(avgLoss);

                    if (verbose && epoch % 100 == 0)
                    {
                        Console.WriteLine($"Epoch {epoch}/{epochs}, Weak Residual: {avgLoss}");
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

        private T ComputeAverageWeakLoss()
        {
            T totalLoss = NumOps.Zero;

            for (int i = 0; i < _numTestFunctions; i++)
            {
                T residual = ComputeWeakResidual(i);
                totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(residual, residual));
            }

            return _numTestFunctions > 0
                ? NumOps.Divide(totalLoss, NumOps.FromDouble(_numTestFunctions))
                : NumOps.Zero;
        }

        /// <summary>
        /// Gets the solution at a specific point.
        /// </summary>
        public T[] GetSolution(T[] point)
        {
            return EvaluateAtPoint(point);
        }

        /// <summary>
        /// Makes a prediction using the VPINN.
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
        /// Gets metadata about the VPINN model.
        /// </summary>
        /// <returns>Model metadata.</returns>
        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                ModelType = ModelType.NeuralNetwork,
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "InputSize", Architecture.InputSize },
                    { "OutputSize", Architecture.OutputSize },
                    { "QuadraturePoints", _numQuadraturePoints },
                    { "TestFunctions", _numTestFunctions },
                    { "ParameterCount", GetParameterCount() }
                },
                ModelData = Serialize()
            };
        }

        /// <summary>
        /// Serializes VPINN-specific data.
        /// </summary>
        /// <param name="writer">Binary writer.</param>
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            writer.Write(_numQuadraturePoints);
            writer.Write(Architecture.InputSize);
            writer.Write(_numTestFunctions);
        }

        /// <summary>
        /// Deserializes VPINN-specific data.
        /// </summary>
        /// <param name="reader">Binary reader.</param>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            int storedNumPoints = reader.ReadInt32();
            int storedDimension = reader.ReadInt32();
            int storedTestFunctions = reader.ReadInt32();

            if (storedNumPoints != _numQuadraturePoints ||
                storedDimension != Architecture.InputSize ||
                storedTestFunctions != _numTestFunctions)
            {
                throw new InvalidOperationException("Serialized VPINN configuration does not match the current instance.");
            }

            GenerateQuadraturePoints(storedNumPoints, storedDimension);
        }

        /// <summary>
        /// Creates a new instance with the same configuration.
        /// </summary>
        /// <returns>New VPINN instance.</returns>
        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new VariationalPINN<T>(
                Architecture,
                _weakFormResidual,
                _numQuadraturePoints,
                _numTestFunctions);
        }

        /// <summary>
        /// Indicates whether this model supports training.
        /// </summary>
        public override bool SupportsTraining => true;
    }
}


