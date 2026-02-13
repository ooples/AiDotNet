using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Interpretability.Helpers;

/// <summary>
/// Provides gradient computation using the GradientTape automatic differentiation system.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This helper uses automatic differentiation (autodiff) to compute
/// gradients. Autodiff is a technique that records mathematical operations as they happen
/// and then plays them backward to compute gradients.
///
/// How it works:
/// 1. Create a GradientTape and start recording
/// 2. Mark the input as a variable we want gradients for (Watch)
/// 3. Run the model's forward pass - all operations are recorded
/// 4. Call Gradient to compute how the output changes with respect to the input
///
/// This is the same technology used by TensorFlow and PyTorch for training neural networks.
///
/// When to use this helper:
/// - When your model is built using TensorOperations (autodiff-aware operations)
/// - When you need gradients with respect to multiple variables
/// - When you want to compute higher-order gradients (gradient of gradient)
///
/// This helper integrates with the existing GradientTape system in AiDotNet.Autodiff.
/// </para>
/// </remarks>
public class AutodiffGradientHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<ComputationNode<T>, ComputationNode<T>> _modelFunction;

    /// <summary>
    /// Gets the method name for this gradient computation approach.
    /// </summary>
    public string GradientMethod => "GradientTape";

    /// <summary>
    /// Gets whether this helper supports exact gradients.
    /// </summary>
    public bool SupportsExactGradients => true;

    /// <summary>
    /// Creates an autodiff gradient helper from a model function that uses ComputationNodes.
    /// </summary>
    /// <param name="modelFunction">
    /// A function that takes an input ComputationNode and returns an output ComputationNode.
    /// This function should use TensorOperations to perform calculations.
    /// </param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The model function is the forward pass of your model expressed
    /// using autodiff operations. Instead of regular tensor operations, you use methods
    /// from TensorOperations&lt;T&gt; which automatically record the computation graph.
    ///
    /// Example:
    /// <code>
    /// // Create a simple linear model using autodiff operations
    /// ComputationNode&lt;double&gt; ModelFunction(ComputationNode&lt;double&gt; input)
    /// {
    ///     var weights = TensorOperations&lt;double&gt;.Variable(weightsTensor, "weights");
    ///     var bias = TensorOperations&lt;double&gt;.Constant(biasTensor, "bias");
    ///     var linear = TensorOperations&lt;double&gt;.MatrixMultiply(input, weights);
    ///     var output = TensorOperations&lt;double&gt;.Add(linear, bias);
    ///     return output;
    /// }
    ///
    /// var helper = new AutodiffGradientHelper&lt;double&gt;(ModelFunction);
    /// </code>
    /// </para>
    /// </remarks>
    public AutodiffGradientHelper(Func<ComputationNode<T>, ComputationNode<T>> modelFunction)
    {
        Guard.NotNull(modelFunction);
        _modelFunction = modelFunction;
    }

    /// <summary>
    /// Computes the gradient of the model output with respect to the input.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="outputIndex">Index of the output to compute gradients for.</param>
    /// <returns>Gradient tensor with the same shape as input.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method:
    /// 1. Creates a GradientTape to record operations
    /// 2. Wraps your input in a ComputationNode and tells the tape to watch it
    /// 3. Runs your model (recorded to the tape)
    /// 4. Computes gradients using reverse-mode autodiff
    ///
    /// The result shows how each input element affects the specified output.
    ///
    /// Under the hood, this uses the chain rule of calculus applied automatically
    /// through all the recorded operations.
    /// </para>
    /// </remarks>
    public Tensor<T> ComputeGradient(Tensor<T> input, int outputIndex = 0)
    {
        using (var tape = new GradientTape<T>())
        {
            // Create input node and watch it
            var inputNode = TensorOperations<T>.Variable(input, "input", requiresGradient: true);
            tape.Watch(inputNode);

            // Run model forward pass
            var outputNode = _modelFunction(inputNode);

            // Create one-hot gradient for target output
            var outputGrad = new Tensor<T>(outputNode.Value.Shape);
            if (outputIndex < outputNode.Value.Length)
            {
                outputGrad[outputIndex] = NumOps.One;
            }

            // Set the output gradient and compute backward
            outputNode.Gradient = outputGrad;
            outputNode.Backward();

            // Return input gradients
            if (inputNode.Gradient != null)
            {
                return inputNode.Gradient;
            }

            return new Tensor<T>(input.Shape);
        }
    }

    /// <summary>
    /// Computes gradients for a vector input.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <param name="outputIndex">Index of the output to compute gradients for.</param>
    /// <returns>Gradient vector with the same length as input.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a convenience method for vector inputs.
    /// It converts the vector to a tensor, computes gradients, and converts back.
    /// </para>
    /// </remarks>
    public Vector<T> ComputeGradient(Vector<T> input, int outputIndex = 0)
    {
        // Convert vector to 2D tensor [1, n]
        var inputTensor = new Tensor<T>(new[] { 1, input.Length });
        for (int i = 0; i < input.Length; i++)
        {
            inputTensor[0, i] = input[i];
        }

        var gradTensor = ComputeGradient(inputTensor, outputIndex);

        // Convert back to vector
        var gradient = new T[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            gradient[i] = gradTensor[0, i];
        }

        return new Vector<T>(gradient);
    }

    /// <summary>
    /// Computes gradients with respect to multiple watched nodes.
    /// </summary>
    /// <param name="inputNodes">Dictionary of named input nodes to watch.</param>
    /// <param name="modelOutput">Function that produces output given input nodes.</param>
    /// <param name="outputIndex">Index of the output to compute gradients for.</param>
    /// <returns>Dictionary mapping node names to their gradients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Sometimes you want gradients with respect to multiple
    /// things at once (e.g., both input features and model parameters). This method
    /// lets you watch multiple nodes and get all their gradients in one backward pass.
    ///
    /// This is more efficient than computing gradients separately because we only
    /// need one forward pass and one backward pass for all gradients.
    ///
    /// Example use case: Computing gradients for both input and layer activations
    /// to understand what the network is doing at different levels.
    /// </para>
    /// </remarks>
    public Dictionary<string, Tensor<T>> ComputeMultipleGradients(
        Dictionary<string, ComputationNode<T>> inputNodes,
        Func<Dictionary<string, ComputationNode<T>>, ComputationNode<T>> modelOutput,
        int outputIndex = 0)
    {
        using (var tape = new GradientTape<T>())
        {
            // Watch all input nodes
            foreach (var node in inputNodes.Values)
            {
                tape.Watch(node);
            }

            // Run forward pass
            var outputNode = modelOutput(inputNodes);

            // Compute gradients
            var gradientMap = tape.Gradient(outputNode, inputNodes.Values);

            // Convert to named dictionary
            var result = new Dictionary<string, Tensor<T>>();
            foreach (var kvp in inputNodes)
            {
                if (gradientMap.TryGetValue(kvp.Value, out var grad))
                {
                    result[kvp.Key] = grad;
                }
            }

            return result;
        }
    }

    /// <summary>
    /// Computes second-order gradients (Hessian-vector product).
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="vector">The vector to multiply with the Hessian.</param>
    /// <param name="outputIndex">Index of the output for first gradient.</param>
    /// <returns>Hessian-vector product.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Hessian is the matrix of second derivatives - it tells
    /// you about the curvature of the function. Computing the full Hessian is expensive
    /// (O(n²) memory), but we can efficiently compute the Hessian times a vector.
    ///
    /// This is useful for:
    /// - Second-order optimization methods (Newton's method)
    /// - Understanding model sensitivity
    /// - Computing influence functions
    ///
    /// The method uses nested GradientTapes to compute gradient-of-gradient.
    /// </para>
    /// </remarks>
    public Tensor<T> ComputeHessianVectorProduct(Tensor<T> input, Tensor<T> vector, int outputIndex = 0)
    {
        using (var outerTape = new GradientTape<T>())
        {
            var inputNode = TensorOperations<T>.Variable(input, "input", requiresGradient: true);
            outerTape.Watch(inputNode);

            using (var innerTape = new GradientTape<T>())
            {
                innerTape.Watch(inputNode);

                // Forward pass
                var outputNode = _modelFunction(inputNode);

                // First gradient
                var outputGrad = new Tensor<T>(outputNode.Value.Shape);
                if (outputIndex < outputNode.Value.Length)
                {
                    outputGrad[outputIndex] = NumOps.One;
                }

                // Compute first gradient with createGraph=true to enable second gradient
                outputNode.Gradient = outputGrad;
                var firstGradients = innerTape.Gradient(outputNode, new[] { inputNode }, createGraph: true);

                if (!firstGradients.TryGetValue(inputNode, out var firstGrad))
                {
                    return new Tensor<T>(input.Shape);
                }

                // Compute dot product of first gradient and vector
                var gradNode = TensorOperations<T>.Variable(firstGrad, "firstGrad", requiresGradient: true);
                var vectorNode = TensorOperations<T>.Constant(vector, "vector");
                var dotProduct = TensorOperations<T>.Sum(
                    TensorOperations<T>.ElementwiseMultiply(gradNode, vectorNode));

                // Second gradient (Hessian-vector product)
                dotProduct.Gradient = new Tensor<T>(dotProduct.Value.Shape);
                dotProduct.Gradient[0] = NumOps.One;
                var hvp = outerTape.Gradient(dotProduct, new[] { inputNode });

                if (hvp.TryGetValue(inputNode, out var result))
                {
                    return result;
                }
            }
        }

        return new Tensor<T>(input.Shape);
    }

    /// <summary>
    /// Creates a gradient function suitable for explainers.
    /// </summary>
    /// <returns>A function that computes gradients given input and output index.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This returns a function that explainers can use to compute
    /// gradients. It's a bridge between the autodiff system and explainer APIs.
    /// </para>
    /// </remarks>
    public Func<Vector<T>, int, Vector<T>> CreateGradientFunction()
    {
        return ComputeGradient;
    }

    /// <summary>
    /// Creates a tensor gradient function for image-based explainers.
    /// </summary>
    /// <returns>A function that computes tensor gradients.</returns>
    public Func<Tensor<T>, int, Tensor<T>> CreateTensorGradientFunction()
    {
        return ComputeGradient;
    }
}

/// <summary>
/// Factory methods for creating gradient helpers from various model types.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This factory makes it easy to get gradient computation for
/// any type of model. Just pass your model and it figures out the best way to
/// compute gradients.
///
/// The factory supports:
/// - Neural networks (uses efficient backpropagation)
/// - Autodiff models (uses GradientTape)
/// - Any prediction function (uses numerical gradients)
/// </para>
/// </remarks>
public static class GradientHelperFactory<T>
{
    /// <summary>
    /// Creates a gradient helper for a neural network.
    /// </summary>
    /// <param name="network">The neural network model.</param>
    /// <returns>A gradient helper that uses backpropagation.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the most efficient option for neural networks.
    /// It uses the network's built-in backpropagation which is O(1) forward/backward
    /// passes regardless of the number of input features.
    /// </para>
    /// </remarks>
    public static InputGradientHelper<T> FromNeuralNetwork(INeuralNetwork<T> network)
    {
        return new InputGradientHelper<T>(network);
    }

    /// <summary>
    /// Creates a gradient helper from a prediction function.
    /// </summary>
    /// <param name="predictFunction">Function that maps input to output.</param>
    /// <param name="epsilon">Epsilon for numerical gradients.</param>
    /// <returns>A gradient helper using numerical differentiation.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this for black-box models where you only have
    /// access to a prediction function. Gradients are computed numerically,
    /// which is slower but works with any model.
    /// </para>
    /// </remarks>
    public static InputGradientHelper<T> FromPredictFunction(
        Func<Vector<T>, Vector<T>> predictFunction,
        double epsilon = 1e-4)
    {
        return new InputGradientHelper<T>(predictFunction, epsilon);
    }

    /// <summary>
    /// Creates a gradient helper from an autodiff model function.
    /// </summary>
    /// <param name="modelFunction">Function using TensorOperations for forward pass.</param>
    /// <returns>A gradient helper using GradientTape autodiff.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this for models built with the autodiff system.
    /// It uses reverse-mode automatic differentiation which is efficient and exact.
    /// </para>
    /// </remarks>
    public static AutodiffGradientHelper<T> FromAutodiffModel(
        Func<ComputationNode<T>, ComputationNode<T>> modelFunction)
    {
        return new AutodiffGradientHelper<T>(modelFunction);
    }

    /// <summary>
    /// Creates a gradient function for any IFullModel.
    /// </summary>
    /// <param name="model">The model to compute gradients for.</param>
    /// <returns>A function that computes input gradients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method attempts to use the most efficient gradient
    /// computation available for the given model:
    /// 1. If it's a neural network, use backpropagation
    /// 2. Otherwise, use numerical gradients
    /// </para>
    /// </remarks>
    public static Func<Vector<T>, int, Vector<T>> CreateGradientFunction(
        IFullModel<T, Tensor<T>, Tensor<T>> model)
    {
        if (model is INeuralNetwork<T> network)
        {
            return new InputGradientHelper<T>(network).CreateGradientFunction();
        }

        // Fall back to numerical gradients using model's Predict
        Func<Vector<T>, Vector<T>> predict = input =>
        {
            var tensor = new Tensor<T>(new[] { 1, input.Length });
            for (int i = 0; i < input.Length; i++)
            {
                tensor[0, i] = input[i];
            }
            var output = model.Predict(tensor);
            var result = new T[output.Length];
            for (int i = 0; i < output.Length; i++)
            {
                result[i] = output.ToArray()[i];
            }
            return new Vector<T>(result);
        };

        return new InputGradientHelper<T>(predict).CreateGradientFunction();
    }
}
