using System;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Autodiff
{
    /// <summary>
    /// Provides first- and second-order derivatives for neural networks with safe fallbacks.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public static class NeuralNetworkDerivatives<T>
    {
        /// <summary>
        /// Computes first and second derivatives for a feedforward network at a single input point.
        /// Falls back to finite differences when analytic derivatives are unavailable.
        /// </summary>
        public static PDEDerivatives<T> ComputeDerivatives(
            NeuralNetworkBase<T> network,
            T[] inputs,
            int outputDim)
        {
            if (network == null)
            {
                throw new ArgumentNullException(nameof(network));
            }

            if (inputs == null)
            {
                throw new ArgumentNullException(nameof(inputs));
            }

            if (outputDim <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(outputDim), "Output dimension must be positive.");
            }

            try
            {
                var result = EvaluateNetworkAnalytic(network, inputs, outputDim);
                return new PDEDerivatives<T>
                {
                    FirstDerivatives = result.Gradients,
                    SecondDerivatives = result.Hessians
                };
            }
            catch (InvalidOperationException)
            {
                return ComputeDerivativesFiniteDifference(network, inputs, outputDim);
            }
            catch (NotSupportedException)
            {
                return ComputeDerivativesFiniteDifference(network, inputs, outputDim);
            }
        }

        /// <summary>
        /// Computes first derivatives (Jacobian) for a network output with autodiff-first fallback.
        /// </summary>
        public static T[,] ComputeGradient(
            NeuralNetworkBase<T> network,
            T[] inputs,
            int outputDim)
        {
            if (TryComputeGradientViaAutodiff(network, inputs, outputDim, out var gradients))
            {
                return gradients;
            }

            var derivatives = ComputeDerivatives(network, inputs, outputDim);
            if (derivatives.FirstDerivatives == null)
            {
                throw new InvalidOperationException("First derivatives were not computed.");
            }

            return derivatives.FirstDerivatives;
        }

        /// <summary>
        /// Computes the Hessian for a scalar output index.
        /// </summary>
        public static T[,] ComputeHessian(
            NeuralNetworkBase<T> network,
            T[] inputs,
            int outputIndex = 0)
        {
            if (network == null)
            {
                throw new ArgumentNullException(nameof(network));
            }

            if (inputs == null)
            {
                throw new ArgumentNullException(nameof(inputs));
            }

            var derivatives = ComputeDerivatives(network, inputs, outputIndex + 1);
            if (derivatives.SecondDerivatives == null)
            {
                throw new InvalidOperationException("Second derivatives were not computed.");
            }

            int inputDim = inputs.Length;
            var hessian = new T[inputDim, inputDim];
            for (int r = 0; r < inputDim; r++)
            {
                for (int c = 0; c < inputDim; c++)
                {
                    hessian[r, c] = derivatives.SecondDerivatives[outputIndex, r, c];
                }
            }

            return hessian;
        }

        private static AutoDiffResult EvaluateNetworkAnalytic(
            NeuralNetworkBase<T> network,
            T[] inputs,
            int outputDim)
        {
            if (network.Layers == null || network.Layers.Count == 0)
            {
                throw new InvalidOperationException("Automatic differentiation requires a network with layers.");
            }

            var numOps = MathHelper.GetNumericOperations<T>();
            int inputDim = inputs.Length;

            var activations = new T[inputDim];
            for (int i = 0; i < inputDim; i++)
            {
                activations[i] = inputs[i];
            }

            var gradients = new T[inputDim][];
            var hessians = new T[inputDim][,];

            for (int i = 0; i < inputDim; i++)
            {
                gradients[i] = CreateZeroVector(inputDim, numOps);
                gradients[i][i] = numOps.One;
                hessians[i] = CreateZeroMatrix(inputDim, inputDim, numOps);
            }

            foreach (var layer in network.Layers)
            {
                switch (layer)
                {
                    case DenseLayer<T> dense:
                        {
                            var weights = dense.GetWeights();
                            var biases = dense.GetBiases();
                            if (weights == null || biases == null)
                            {
                                throw new InvalidOperationException("DenseLayer weights or biases are not initialized.");
                            }

                            ProcessDenseLayer(
                                weights,
                                biases,
                                dense.ScalarActivation,
                                dense.VectorActivation,
                                numOps,
                                activations,
                                gradients,
                                hessians,
                                out activations,
                                out gradients,
                                out hessians);
                        }
                        break;
                    case FullyConnectedLayer<T> fullyConnected:
                        {
                            var weights = fullyConnected.GetWeights();
                            var biases = fullyConnected.GetBiases();
                            if (weights == null || biases == null)
                            {
                                throw new InvalidOperationException("FullyConnectedLayer weights or biases are not initialized.");
                            }

                            ProcessDenseLayer(
                                weights,
                                biases,
                                fullyConnected.ScalarActivation,
                                fullyConnected.VectorActivation,
                                numOps,
                                activations,
                                gradients,
                                hessians,
                                out activations,
                                out gradients,
                                out hessians);
                        }
                        break;
                    case ActivationLayer<T> activation:
                        ProcessActivationLayer(
                            activation.ScalarActivation,
                            activation.VectorActivation,
                            numOps,
                            activations,
                            gradients,
                            hessians,
                            out activations,
                            out gradients,
                            out hessians);
                        break;
                    default:
                        throw new NotSupportedException(
                            $"Automatic differentiation supports DenseLayer, FullyConnectedLayer, and ActivationLayer only. " +
                            $"Unsupported layer: {layer.GetType().Name}.");
                }
            }

            if (activations.Length != outputDim)
            {
                throw new ArgumentException(
                    $"Output dimension mismatch. Expected {outputDim}, got {activations.Length}.",
                    nameof(outputDim));
            }

            var first = new T[outputDim, inputDim];
            var second = new T[outputDim, inputDim, inputDim];

            for (int i = 0; i < outputDim; i++)
            {
                for (int j = 0; j < inputDim; j++)
                {
                    first[i, j] = gradients[i][j];
                }

                for (int r = 0; r < inputDim; r++)
                {
                    for (int c = 0; c < inputDim; c++)
                    {
                        second[i, r, c] = hessians[i][r, c];
                    }
                }
            }

            return new AutoDiffResult(activations, first, second);
        }

        private static bool TryComputeGradientViaAutodiff(
            NeuralNetworkBase<T> network,
            T[] inputs,
            int outputDim,
            out T[,] gradients)
        {
            gradients = new T[outputDim, inputs.Length];

            if (!SupportsAutodiffGraph(network))
            {
                return false;
            }

            try
            {
                var inputTensor = CreateInputTensor(inputs);
                var inputNode = TensorOperations<T>.Variable(inputTensor, "input", requiresGradient: true);
                var outputNode = BuildGraph(network, inputNode);
                var outputShape = outputNode.Value.Shape;

                if (outputShape.Length != 2 || outputShape[0] != 1 || outputShape[1] != outputDim)
                {
                    return false;
                }

                var numOps = MathHelper.GetNumericOperations<T>();

                for (int outIdx = 0; outIdx < outputDim; outIdx++)
                {
                    var maskTensor = new Tensor<T>(outputShape);
                    maskTensor.Fill(numOps.Zero);
                    maskTensor[0, outIdx] = numOps.One;
                    var maskNode = TensorOperations<T>.Constant(maskTensor, $"mask_{outIdx}");
                    var masked = TensorOperations<T>.ElementwiseMultiply(outputNode, maskNode);
                    var scalar = TensorOperations<T>.Sum(masked);

                    scalar.Backward();

                    if (inputNode.Gradient == null)
                    {
                        return false;
                    }

                    for (int i = 0; i < inputs.Length; i++)
                    {
                        gradients[outIdx, i] = inputNode.Gradient[0, i];
                    }
                }

                return true;
            }
            catch (InvalidOperationException)
            {
                return false;
            }
            catch (NotSupportedException)
            {
                return false;
            }
        }

        private static bool SupportsAutodiffGraph(NeuralNetworkBase<T> network)
        {
            foreach (var layer in network.Layers)
            {
                if (layer is LayerBase<T> layerBase && !layerBase.SupportsJitCompilation)
                {
                    return false;
                }
            }

            return true;
        }

        private static ComputationNode<T> BuildGraph(
            NeuralNetworkBase<T> network,
            ComputationNode<T> inputNode)
        {
            var current = inputNode;
            foreach (var layer in network.Layers)
            {
                var inputs = new System.Collections.Generic.List<ComputationNode<T>> { current };
                current = layer.ExportComputationGraph(inputs);
            }

            return current;
        }

        private static PDEDerivatives<T> ComputeDerivativesFiniteDifference(
            NeuralNetworkBase<T> network,
            T[] inputs,
            int outputDim,
            double step = 1e-5)
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            int inputDim = inputs.Length;

            var derivatives = new PDEDerivatives<T>
            {
                FirstDerivatives = new T[outputDim, inputDim],
                SecondDerivatives = new T[outputDim, inputDim, inputDim]
            };

            var baseOutput = EvaluateOutput(network, inputs, outputDim);
            var epsilon = numOps.FromDouble(step);
            var twoEpsilon = numOps.FromDouble(2.0 * step);
            var epsSquared = numOps.Multiply(epsilon, epsilon);
            var fourEpsSquared = numOps.Multiply(numOps.FromDouble(4.0), epsSquared);

            var perturbed = new T[inputDim];
            Array.Copy(inputs, perturbed, inputDim);

            for (int i = 0; i < inputDim; i++)
            {
                T original = perturbed[i];
                perturbed[i] = numOps.Add(original, epsilon);
                var plus = EvaluateOutput(network, perturbed, outputDim);
                perturbed[i] = numOps.Subtract(original, epsilon);
                var minus = EvaluateOutput(network, perturbed, outputDim);

                for (int outIdx = 0; outIdx < outputDim; outIdx++)
                {
                    derivatives.FirstDerivatives[outIdx, i] = numOps.Divide(
                        numOps.Subtract(plus[outIdx], minus[outIdx]),
                        twoEpsilon);

                    var second = numOps.Divide(
                        numOps.Add(
                            numOps.Subtract(plus[outIdx], numOps.Multiply(numOps.FromDouble(2.0), baseOutput[outIdx])),
                            minus[outIdx]),
                        epsSquared);
                    derivatives.SecondDerivatives[outIdx, i, i] = second;
                }

                perturbed[i] = original;
            }

            for (int i = 0; i < inputDim; i++)
            {
                for (int j = i + 1; j < inputDim; j++)
                {
                    T originalI = perturbed[i];
                    T originalJ = perturbed[j];

                    perturbed[i] = numOps.Add(originalI, epsilon);
                    perturbed[j] = numOps.Add(originalJ, epsilon);
                    var plusPlus = EvaluateOutput(network, perturbed, outputDim);

                    perturbed[j] = numOps.Subtract(originalJ, epsilon);
                    var plusMinus = EvaluateOutput(network, perturbed, outputDim);

                    perturbed[i] = numOps.Subtract(originalI, epsilon);
                    perturbed[j] = numOps.Add(originalJ, epsilon);
                    var minusPlus = EvaluateOutput(network, perturbed, outputDim);

                    perturbed[j] = numOps.Subtract(originalJ, epsilon);
                    var minusMinus = EvaluateOutput(network, perturbed, outputDim);

                    for (int outIdx = 0; outIdx < outputDim; outIdx++)
                    {
                        var cross = numOps.Divide(
                            numOps.Add(
                                numOps.Subtract(plusPlus[outIdx], plusMinus[outIdx]),
                                numOps.Subtract(minusMinus[outIdx], minusPlus[outIdx])),
                            fourEpsSquared);

                        derivatives.SecondDerivatives[outIdx, i, j] = cross;
                        derivatives.SecondDerivatives[outIdx, j, i] = cross;
                    }

                    perturbed[i] = originalI;
                    perturbed[j] = originalJ;
                }
            }

            return derivatives;
        }

        private static T[] EvaluateOutput(
            NeuralNetworkBase<T> network,
            T[] inputs,
            int outputDim)
        {
            var inputTensor = CreateInputTensor(inputs);
            var outputTensor = network.Predict(inputTensor);

            if (outputTensor.Rank == 2 && outputTensor.Shape[0] == 1)
            {
                if (outputTensor.Shape[1] != outputDim)
                {
                    throw new ArgumentException(
                        $"Output dimension mismatch. Expected {outputDim}, got {outputTensor.Shape[1]}.");
                }

                var output = new T[outputDim];
                for (int i = 0; i < outputDim; i++)
                {
                    output[i] = outputTensor[0, i];
                }
                return output;
            }

            if (outputTensor.Rank == 1)
            {
                if (outputTensor.Shape[0] != outputDim)
                {
                    throw new ArgumentException(
                        $"Output dimension mismatch. Expected {outputDim}, got {outputTensor.Shape[0]}.");
                }

                var output = new T[outputDim];
                for (int i = 0; i < outputDim; i++)
                {
                    output[i] = outputTensor[i];
                }
                return output;
            }

            throw new InvalidOperationException("Unexpected output tensor shape for derivative evaluation.");
        }

        private static Tensor<T> CreateInputTensor(T[] inputs)
        {
            var inputTensor = new Tensor<T>(new int[] { 1, inputs.Length });
            for (int i = 0; i < inputs.Length; i++)
            {
                inputTensor[0, i] = inputs[i];
            }

            return inputTensor;
        }

        private static void ProcessDenseLayer(
            Tensor<T> weights,
            Tensor<T> biases,
            IActivationFunction<T>? activation,
            IVectorActivationFunction<T>? vectorActivation,
            INumericOperations<T> numOps,
            T[] prevActivations,
            T[][] prevGradients,
            T[][,] prevHessians,
            out T[] activations,
            out T[][] gradients,
            out T[][,] hessians)
        {
            if (vectorActivation != null)
            {
                throw new NotSupportedException("Vector activations are not supported for PDE derivatives.");
            }

            int outputSize = weights.Shape[0];
            int inputSize = weights.Shape[1];

            if (prevActivations.Length != inputSize)
            {
                throw new ArgumentException(
                    $"Layer input size mismatch. Expected {inputSize}, got {prevActivations.Length}.");
            }

            if (biases.Shape[0] != outputSize)
            {
                throw new ArgumentException(
                    $"Bias vector size mismatch. Expected {outputSize}, got {biases.Shape[0]}.");
            }

            int inputDim = prevGradients[0].Length;
            activations = new T[outputSize];
            gradients = new T[outputSize][];
            hessians = new T[outputSize][,];

            for (int i = 0; i < outputSize; i++)
            {
                T z = biases[i];
                for (int j = 0; j < inputSize; j++)
                {
                    z = numOps.Add(z, numOps.Multiply(weights[i, j], prevActivations[j]));
                }

                EvaluateActivation(activation, z, numOps, out T value, out T first, out T second);
                activations[i] = value;

                var v = CreateZeroVector(inputDim, numOps);
                var sumH = CreateZeroMatrix(inputDim, inputDim, numOps);

                for (int j = 0; j < inputSize; j++)
                {
                    var weight = weights[i, j];
                    var prevGradient = prevGradients[j];
                    var prevHessian = prevHessians[j];

                    for (int k = 0; k < inputDim; k++)
                    {
                        v[k] = numOps.Add(v[k], numOps.Multiply(weight, prevGradient[k]));
                    }

                    for (int r = 0; r < inputDim; r++)
                    {
                        for (int c = 0; c < inputDim; c++)
                        {
                            sumH[r, c] = numOps.Add(sumH[r, c], numOps.Multiply(weight, prevHessian[r, c]));
                        }
                    }
                }

                var gradient = CreateZeroVector(inputDim, numOps);
                for (int k = 0; k < inputDim; k++)
                {
                    gradient[k] = numOps.Multiply(first, v[k]);
                }
                gradients[i] = gradient;

                var hessian = CreateZeroMatrix(inputDim, inputDim, numOps);
                for (int r = 0; r < inputDim; r++)
                {
                    for (int c = 0; c < inputDim; c++)
                    {
                        var outer = numOps.Multiply(v[r], v[c]);
                        var term1 = numOps.Multiply(second, outer);
                        var term2 = numOps.Multiply(first, sumH[r, c]);
                        hessian[r, c] = numOps.Add(term1, term2);
                    }
                }
                hessians[i] = hessian;
            }
        }

        private static void ProcessActivationLayer(
            IActivationFunction<T>? activation,
            IVectorActivationFunction<T>? vectorActivation,
            INumericOperations<T> numOps,
            T[] prevActivations,
            T[][] prevGradients,
            T[][,] prevHessians,
            out T[] activations,
            out T[][] gradients,
            out T[][,] hessians)
        {
            if (vectorActivation != null)
            {
                throw new NotSupportedException("Vector activations are not supported for PDE derivatives.");
            }

            int count = prevActivations.Length;
            int inputDim = prevGradients[0].Length;

            activations = new T[count];
            gradients = new T[count][];
            hessians = new T[count][,];

            for (int i = 0; i < count; i++)
            {
                EvaluateActivation(activation, prevActivations[i], numOps, out T value, out T first, out T second);
                activations[i] = value;

                var gradient = CreateZeroVector(inputDim, numOps);
                var prevGradient = prevGradients[i];
                for (int k = 0; k < inputDim; k++)
                {
                    gradient[k] = numOps.Multiply(first, prevGradient[k]);
                }
                gradients[i] = gradient;

                var hessian = CreateZeroMatrix(inputDim, inputDim, numOps);
                var prevHessian = prevHessians[i];
                for (int r = 0; r < inputDim; r++)
                {
                    for (int c = 0; c < inputDim; c++)
                    {
                        var outer = numOps.Multiply(prevGradient[r], prevGradient[c]);
                        var term1 = numOps.Multiply(second, outer);
                        var term2 = numOps.Multiply(first, prevHessian[r, c]);
                        hessian[r, c] = numOps.Add(term1, term2);
                    }
                }
                hessians[i] = hessian;
            }
        }

        private static void EvaluateActivation(
            IActivationFunction<T>? activation,
            T input,
            INumericOperations<T> numOps,
            out T value,
            out T firstDerivative,
            out T secondDerivative)
        {
            if (activation == null || activation is IdentityActivation<T>)
            {
                value = input;
                firstDerivative = numOps.One;
                secondDerivative = numOps.Zero;
                return;
            }

            if (activation is TanhActivation<T>)
            {
                value = MathHelper.Tanh(input);
                var tanhSquared = numOps.Multiply(value, value);
                var oneMinus = numOps.Subtract(numOps.One, tanhSquared);
                firstDerivative = oneMinus;
                secondDerivative = numOps.Multiply(
                    numOps.FromDouble(-2.0),
                    numOps.Multiply(value, oneMinus));
                return;
            }

            if (activation is SigmoidActivation<T>)
            {
                value = MathHelper.Sigmoid(input);
                var oneMinus = numOps.Subtract(numOps.One, value);
                firstDerivative = numOps.Multiply(value, oneMinus);
                var two = numOps.FromDouble(2.0);
                secondDerivative = numOps.Multiply(firstDerivative, numOps.Subtract(numOps.One, numOps.Multiply(two, value)));
                return;
            }

            if (activation is ReLUActivation<T>)
            {
                if (numOps.GreaterThan(input, numOps.Zero))
                {
                    value = input;
                    firstDerivative = numOps.One;
                }
                else
                {
                    value = numOps.Zero;
                    firstDerivative = numOps.Zero;
                }
                secondDerivative = numOps.Zero;
                return;
            }

            if (activation is LeakyReLUActivation<T> leaky)
            {
                if (numOps.GreaterThan(input, numOps.Zero))
                {
                    value = input;
                    firstDerivative = numOps.One;
                }
                else
                {
                    value = numOps.Multiply(leaky.Alpha, input);
                    firstDerivative = leaky.Alpha;
                }
                secondDerivative = numOps.Zero;
                return;
            }

            if (activation is GELUActivation<T>)
            {
                var sqrt2OverPi = numOps.FromDouble(Math.Sqrt(2.0 / Math.PI));
                var c = numOps.FromDouble(0.044715);
                var half = numOps.FromDouble(0.5);
                var one = numOps.One;
                var three = numOps.FromDouble(3.0);
                var six = numOps.FromDouble(6.0);

                var x2 = numOps.Multiply(input, input);
                var x3 = numOps.Multiply(x2, input);
                var inner = numOps.Multiply(sqrt2OverPi, numOps.Add(input, numOps.Multiply(c, x3)));
                var tanhInner = MathHelper.Tanh(inner);
                var sech2 = numOps.Subtract(one, numOps.Multiply(tanhInner, tanhInner));

                value = numOps.Multiply(half, numOps.Multiply(input, numOps.Add(one, tanhInner)));

                var innerDeriv = numOps.Add(one, numOps.Multiply(numOps.Multiply(three, c), x2));
                var firstTerm = numOps.Multiply(half, numOps.Add(one, tanhInner));
                var secondTerm = numOps.Multiply(
                    numOps.Multiply(numOps.Multiply(half, input), sech2),
                    numOps.Multiply(sqrt2OverPi, innerDeriv));
                firstDerivative = numOps.Add(firstTerm, secondTerm);

                var uPrime = numOps.Multiply(sqrt2OverPi, innerDeriv);
                var uSecond = numOps.Multiply(sqrt2OverPi, numOps.Multiply(six, numOps.Multiply(c, input)));
                var term1 = numOps.Multiply(sech2, uPrime);
                var uPrimeSquared = numOps.Multiply(uPrime, uPrime);
                var innerSecondTerm = numOps.Subtract(numOps.Multiply(half, uSecond), numOps.Multiply(tanhInner, uPrimeSquared));
                var term2 = numOps.Multiply(numOps.Multiply(input, sech2), innerSecondTerm);
                secondDerivative = numOps.Add(term1, term2);
                return;
            }

            throw new NotSupportedException(
                $"Activation {activation.GetType().Name} is not supported for second-order derivatives.");
        }

        private static T[] CreateZeroVector(int length, INumericOperations<T> numOps)
        {
            var vector = new T[length];
            for (int i = 0; i < length; i++)
            {
                vector[i] = numOps.Zero;
            }
            return vector;
        }

        private static T[,] CreateZeroMatrix(int rows, int cols, INumericOperations<T> numOps)
        {
            var matrix = new T[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = numOps.Zero;
                }
            }
            return matrix;
        }

        private sealed class AutoDiffResult
        {
            public AutoDiffResult(T[] outputs, T[,] gradients, T[,,] hessians)
            {
                Outputs = outputs;
                Gradients = gradients;
                Hessians = hessians;
            }

            public T[] Outputs { get; }
            public T[,] Gradients { get; }
            public T[,,] Hessians { get; }
        }
    }
}
