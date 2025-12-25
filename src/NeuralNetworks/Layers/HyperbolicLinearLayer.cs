using AiDotNet.ActivationFunctions;
using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a fully connected layer operating in hyperbolic (Poincare ball) space.
/// </summary>
/// <remarks>
/// <para>
/// A hyperbolic linear layer performs linear transformations in hyperbolic space using
/// Mobius operations. This is particularly useful for learning hierarchical representations
/// where tree-like structures need to be embedded.
/// </para>
/// <para><b>For Beginners:</b> This layer works in hyperbolic space instead of flat Euclidean space.
///
/// Benefits of hyperbolic layers:
/// - Naturally represents hierarchical data (trees, graphs, taxonomies)
/// - Can embed large hierarchies with low distortion
/// - Fewer dimensions needed for complex hierarchical structures
///
/// The layer uses the Poincare ball model where all points are inside a unit ball.
/// Points near the center are "higher" in the hierarchy, points near the edge are "lower".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (float or double).</typeparam>
public class HyperbolicLinearLayer<T> : LayerBase<T>
{
    private readonly IHyperbolicManifoldEngine _engine;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Weight matrix stored in tangent space at the origin.
    /// Shape: [OutputFeatures, InputFeatures]
    /// </summary>
    private Matrix<T> _weights;

    /// <summary>
    /// Bias values as points on the Poincare ball.
    /// Shape: [OutputFeatures, InputFeatures] - each row is a bias point.
    /// </summary>
    private Matrix<T> _biases;

    /// <summary>
    /// The curvature of the hyperbolic space (negative for hyperbolic).
    /// </summary>
    private readonly T _curvature;

    /// <summary>
    /// Stored input from forward pass for backpropagation.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stored pre-activation output for gradient computation.
    /// </summary>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Gradient for weights, stored during backward pass.
    /// </summary>
    private Matrix<T>? _weightsGradient;

    /// <summary>
    /// Gradient for biases, stored during backward pass.
    /// </summary>
    private Matrix<T>? _biasesGradient;

    /// <summary>
    /// Gets the number of input features.
    /// </summary>
    public int InputFeatures { get; }

    /// <summary>
    /// Gets the number of output features.
    /// </summary>
    public int OutputFeatures { get; }

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        (OutputFeatures * InputFeatures) + (OutputFeatures * InputFeatures);

    /// <summary>
    /// Gets whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets whether this layer supports JIT compilation.
    /// Hyperbolic operations are not yet supported for JIT.
    /// </summary>
    public override bool SupportsJitCompilation => false;

    /// <summary>
    /// Initializes a new instance of the HyperbolicLinearLayer.
    /// </summary>
    /// <param name="inputFeatures">Number of input features.</param>
    /// <param name="outputFeatures">Number of output features.</param>
    /// <param name="curvature">Curvature of hyperbolic space (default -1).</param>
    /// <param name="activationFunction">Optional activation function.</param>
    public HyperbolicLinearLayer(
        int inputFeatures,
        int outputFeatures,
        double curvature = -1.0,
        IActivationFunction<T>? activationFunction = null)
        : base(
            [inputFeatures],
            [outputFeatures],
            activationFunction ?? new IdentityActivation<T>())
    {
        if (curvature >= 0)
        {
            throw new ArgumentException("Curvature must be negative for hyperbolic space.", nameof(curvature));
        }

        InputFeatures = inputFeatures;
        OutputFeatures = outputFeatures;

        _engine = CpuHyperbolicManifoldEngine.Instance;
        _numOps = MathHelper.GetNumericOperations<T>();
        _curvature = _numOps.FromDouble(curvature);

        _weights = new Matrix<T>(outputFeatures, inputFeatures);
        _biases = new Matrix<T>(outputFeatures, inputFeatures);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes weights using Xavier/Glorot initialization adapted for hyperbolic space.
    /// Weights are initialized in tangent space at the origin.
    /// </summary>
    private void InitializeParameters()
    {
        var scale = Math.Sqrt(2.0 / (InputFeatures + OutputFeatures));
        var random = RandomHelper.CreateSeededRandom(42);

        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                // Initialize weights in tangent space (small values)
                _weights[o, i] = _numOps.FromDouble((random.NextDouble() - 0.5) * 2 * scale * 0.1);
                // Initialize biases close to origin (small values for Poincare ball)
                _biases[o, i] = _numOps.FromDouble((random.NextDouble() - 0.5) * 2 * 0.01);
            }
        }
    }

    /// <summary>
    /// Performs the forward pass through the layer.
    /// </summary>
    /// <param name="input">Input tensor with shape [inputFeatures] or [batch, inputFeatures].</param>
    /// <returns>Output tensor with shape [outputFeatures] or [batch, outputFeatures].</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        bool wasSingleSample = input.Rank == 1;
        int batchSize;
        int inputLen;
        Tensor<T> inputTensor;

        if (wasSingleSample)
        {
            batchSize = 1;
            inputLen = input.Shape[0];
            // Convert 1D to 2D for processing
            inputTensor = new Tensor<T>([1, inputLen]);
            for (int i = 0; i < inputLen; i++)
            {
                inputTensor[0, i] = input[i];
            }
        }
        else
        {
            batchSize = input.Shape[0];
            inputLen = input.Shape[1];
            inputTensor = input;
        }

        // Validate input shape
        if (inputLen != InputFeatures)
        {
            throw new ArgumentException(
                $"Input size {inputLen} does not match expected {InputFeatures}.");
        }

        _lastInput = input;

        // Output tensor: [batchSize, OutputFeatures]
        var output = new Tensor<T>([batchSize, OutputFeatures]);

        // For each sample in batch
        for (int b = 0; b < batchSize; b++)
        {
            // Extract input vector for this sample
            var inputVec = new Vector<T>(InputFeatures);
            for (int i = 0; i < InputFeatures; i++)
            {
                inputVec[i] = inputTensor[b, i];
            }

            // Project input to Poincare ball (ensure valid point)
            var epsilon = _numOps.FromDouble(1e-5);
            var projectedInput = _engine.PoincareProject(inputVec, _curvature, epsilon);

            // For each output feature
            for (int o = 0; o < OutputFeatures; o++)
            {
                // Get weight vector for this output
                var weightVec = new Vector<T>(InputFeatures);
                for (int i = 0; i < InputFeatures; i++)
                {
                    weightVec[i] = _weights[o, i];
                }

                // Get bias vector for this output
                var biasVec = new Vector<T>(InputFeatures);
                for (int i = 0; i < InputFeatures; i++)
                {
                    biasVec[i] = _biases[o, i];
                }

                // Compute hyperbolic linear transformation:
                // 1. Apply exponential map from origin with weight as tangent vector
                var origin = CreateOriginVector(InputFeatures);
                var weightPoint = _engine.PoincareExpMap(origin, weightVec, _curvature);

                // 2. Mobius addition of input with weight point
                var transformed = _engine.MobiusAdd(projectedInput, weightPoint, _curvature);

                // 3. Mobius addition with bias
                var biasProjected = _engine.PoincareProject(biasVec, _curvature, epsilon);
                var withBias = _engine.MobiusAdd(transformed, biasProjected, _curvature);

                // 4. Compute output as distance from origin (scalar output)
                // This gives a scalar representing "how far down the hierarchy"
                var distance = _engine.PoincareDistance(origin, withBias, _curvature);
                output[b, o] = distance;
            }
        }

        _lastOutput = output;

        // Apply activation function
        var activated = ApplyActivation(output);

        // Return 1D tensor for single sample input to match input rank
        if (wasSingleSample)
        {
            var result = new Tensor<T>([OutputFeatures]);
            for (int o = 0; o < OutputFeatures; o++)
            {
                result[o] = activated[0, o];
            }
            return result;
        }

        return activated;
    }

    /// <summary>
    /// Performs the backward pass through the layer.
    /// </summary>
    /// <param name="outputGradient">Gradient from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
        {
            throw new InvalidOperationException("Backward called before Forward.");
        }

        bool wasSingleSample = outputGradient.Rank == 1;
        int batchSize;
        Tensor<T> gradTensor;

        if (wasSingleSample)
        {
            batchSize = 1;
            // Convert 1D to 2D for processing
            gradTensor = new Tensor<T>([1, OutputFeatures]);
            for (int o = 0; o < OutputFeatures; o++)
            {
                gradTensor[0, o] = outputGradient[o];
            }
        }
        else
        {
            batchSize = outputGradient.Shape[0];
            gradTensor = outputGradient;
        }

        // Initialize gradients
        _weightsGradient = new Matrix<T>(OutputFeatures, InputFeatures);
        _biasesGradient = new Matrix<T>(OutputFeatures, InputFeatures);
        var inputGradient = new Tensor<T>(_lastInput.Shape);

        // Compute gradients using proper Riemannian gradient descent for Poincaré ball geometry.
        // For the Poincaré ball with curvature c, the conformal factor is:
        //   λ(x) = 2 / (1 - c||x||²)  where c = |curvature|
        // The Riemannian gradient is related to the Euclidean gradient by:
        //   grad_R = (1/λ(x)²) * grad_E = ((1 - c||x||²)² / 4) * grad_E
        var epsilon = _numOps.FromDouble(1e-5);
        var absCurvature = _numOps.Abs(_curvature);

        for (int b = 0; b < batchSize; b++)
        {
            // Extract input vector
            var inputVec = new Vector<T>(InputFeatures);
            for (int i = 0; i < InputFeatures; i++)
            {
                inputVec[i] = _lastInput.Rank > 1 ? _lastInput[b, i] : _lastInput[i];
            }
            var projectedInput = _engine.PoincareProject(inputVec, _curvature, epsilon);

            // Compute squared norm of projected input: ||x||²
            T squaredNorm = _numOps.Zero;
            for (int i = 0; i < InputFeatures; i++)
            {
                squaredNorm = _numOps.Add(squaredNorm, _numOps.Multiply(projectedInput[i], projectedInput[i]));
            }

            // Compute conformal factor for Riemannian gradient:
            // conformalFactor = (1 - c||x||²)² / 4 = 1/λ(x)²
            // This converts Euclidean gradients to Riemannian gradients on the Poincaré ball
            var cNormSquared = _numOps.Multiply(absCurvature, squaredNorm);
            var oneMinusCNorm = _numOps.Subtract(_numOps.One, cNormSquared);
            var conformalFactor = _numOps.Divide(
                _numOps.Multiply(oneMinusCNorm, oneMinusCNorm),
                _numOps.FromDouble(4.0));

            for (int o = 0; o < OutputFeatures; o++)
            {
                T gradOutput = gradTensor[b, o];

                // Scale the output gradient by the conformal factor for Riemannian geometry
                var riemannianGrad = _numOps.Multiply(gradOutput, conformalFactor);

                for (int i = 0; i < InputFeatures; i++)
                {
                    // Weight gradient: Riemannian gradient scaled by input direction
                    var existingWGrad = _weightsGradient[o, i];
                    var inputContrib = _numOps.Multiply(riemannianGrad, projectedInput[i]);
                    _weightsGradient[o, i] = _numOps.Add(existingWGrad, inputContrib);

                    // Bias gradient: Riemannian gradient (conformal factor already applied)
                    // Distributed across input features for the bias point
                    var existingBGrad = _biasesGradient[o, i];
                    var biasContrib = _numOps.Divide(riemannianGrad, _numOps.FromDouble(InputFeatures));
                    _biasesGradient[o, i] = _numOps.Add(existingBGrad, biasContrib);

                    // Input gradient: Riemannian gradient scaled by weight direction
                    var existingIGrad = inputGradient.Rank > 1 ? inputGradient[b, i] : inputGradient[i];
                    var weightContrib = _numOps.Multiply(riemannianGrad, _weights[o, i]);
                    if (inputGradient.Rank > 1)
                    {
                        inputGradient[b, i] = _numOps.Add(existingIGrad, weightContrib);
                    }
                    else
                    {
                        inputGradient[i] = _numOps.Add(existingIGrad, weightContrib);
                    }
                }
            }
        }

        return inputGradient;
    }

    /// <summary>
    /// Updates the layer's parameters using the calculated gradients.
    /// Uses Riemannian gradient descent (exponential map of negative gradient).
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the update.</param>
    public override void UpdateParameters(T learningRate)
    {
        if (_weightsGradient == null || _biasesGradient == null)
        {
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        }

        var epsilon = _numOps.FromDouble(1e-5);

        for (int o = 0; o < OutputFeatures; o++)
        {
            // Update weights in tangent space (per-element update)
            for (int i = 0; i < InputFeatures; i++)
            {
                var grad = _weightsGradient[o, i];
                var scaledGrad = _numOps.Multiply(learningRate, grad);
                _weights[o, i] = _numOps.Subtract(_weights[o, i], scaledGrad);
            }

            // Update biases using Riemannian gradient descent
            // Build the full tangent vector from all bias gradients at once
            var biasPoint = new Vector<T>(InputFeatures);
            var tangentVec = new Vector<T>(InputFeatures);
            for (int j = 0; j < InputFeatures; j++)
            {
                biasPoint[j] = _biases[o, j];
                tangentVec[j] = _numOps.Negate(_numOps.Multiply(learningRate, _biasesGradient[o, j]));
            }

            // Project bias to valid region and apply exponential map update once per output
            var projectedBias = _engine.PoincareProject(biasPoint, _curvature, epsilon);
            var updatedBias = _engine.PoincareExpMap(projectedBias, tangentVec, _curvature);
            updatedBias = _engine.PoincareProject(updatedBias, _curvature, epsilon);

            for (int j = 0; j < InputFeatures; j++)
            {
                _biases[o, j] = updatedBias[j];
            }
        }
    }

    /// <summary>
    /// Gets all trainable parameters as a single vector.
    /// </summary>
    /// <returns>A vector containing all weights and biases.</returns>
    public override Vector<T> GetParameters()
    {
        var paramArray = new T[ParameterCount];
        int idx = 0;

        // Flatten weights
        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                paramArray[idx++] = _weights[o, i];
            }
        }

        // Flatten biases
        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                paramArray[idx++] = _biases[o, i];
            }
        }

        return new Vector<T>(paramArray);
    }

    /// <summary>
    /// Sets all trainable parameters from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException($"Expected {ParameterCount} parameters, but got {parameters.Length}");
        }

        int idx = 0;

        // Restore weights
        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                _weights[o, i] = parameters[idx++];
            }
        }

        // Restore biases
        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                _biases[o, i] = parameters[idx++];
            }
        }
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _weightsGradient = null;
        _biasesGradient = null;
    }

    /// <summary>
    /// Exports the layer's forward pass as a JIT-compilable computation graph.
    /// Currently not supported for hyperbolic layers.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node.</returns>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException(
            "HyperbolicLinearLayer does not yet support JIT compilation. " +
            "Hyperbolic operations require specialized IR operations.");
    }

    /// <summary>
    /// Creates a vector at the origin of hyperbolic space.
    /// </summary>
    private Vector<T> CreateOriginVector(int dimension)
    {
        var origin = new Vector<T>(dimension);
        for (int i = 0; i < dimension; i++)
        {
            origin[i] = _numOps.Zero;
        }
        return origin;
    }
}
