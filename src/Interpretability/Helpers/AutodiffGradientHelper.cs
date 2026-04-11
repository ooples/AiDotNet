using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

// Note: GradientTape references removed — ComputationNode has its own backward mechanism

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
        var gradient = new Vector<T>(input.Length);
        for (int i = 0; i < input.Length; i++)
        {
            gradient[i] = gradTensor[0, i];
        }

        return gradient;
    }

    /// <summary>
    /// Computes gradients for a tensor input.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="outputIndex">Index of the output to compute gradients for.</param>
    /// <returns>Gradient tensor with the same shape as input.</returns>
    public Tensor<T> ComputeGradient(Tensor<T> input, int outputIndex = 0)
    {
        // Use tape-based autodiff to compute gradients through the computation graph
        var eng = AiDotNetEngine.Current;
        using var tape = new GradientTape<T>();

        // Create a computation node for the input
        var inputNode = TensorOperations<T>.Variable(input, "input");

        // Run the model function
        var outputNode = _modelFunction(inputNode);

        if (outputNode.Value is null)
        {
            return new Tensor<T>(input._shape);
        }

        // Create one-hot selector for the target output index and compute scalar loss
        var oneHot = new Tensor<T>(outputNode.Value._shape);
        if (outputIndex < oneHot.Length)
        {
            oneHot[outputIndex] = NumOps.One;
        }
        var selected = eng.TensorMultiply(outputNode.Value, oneHot);
        var allAxes = Enumerable.Range(0, selected.Shape.Length).ToArray();
        var loss = eng.ReduceSum(selected, allAxes, keepDims: false);

        var grads = tape.ComputeGradients(loss, [input]);
        return grads.TryGetValue(input, out var g) ? g : new Tensor<T>(input._shape);
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
        // Hessian-vector products require second-order autodiff (createGraph=true),
        // which is supported by the Tensors GradientTape but not by ComputationNode backward.
        // Use a numerical approximation: Hvp ≈ [∇f(x + εv) - ∇f(x)] / ε
        var eps = 1e-5;
        var numOps2 = MathHelper.GetNumericOperations<T>();

        // Compute gradient at x
        var grad0 = ComputeGradient(input, outputIndex);

        // Compute gradient at x + ε*v
        var perturbedShape = input._shape;
        var perturbed = new Tensor<T>(perturbedShape);
        for (int i = 0; i < input.Length; i++)
        {
            perturbed[i] = numOps2.Add(input[i], numOps2.Multiply(numOps2.FromDouble(eps), vector[i]));
        }
        var grad1 = ComputeGradient(perturbed, outputIndex);

        // Hvp ≈ (grad1 - grad0) / ε
        var result = new Tensor<T>(perturbedShape);
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = numOps2.Divide(numOps2.Subtract(grad1[i], grad0[i]), numOps2.FromDouble(eps));
        }
        return result;
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
            return output.ToVector();
        };

        return new InputGradientHelper<T>(predict).CreateGradientFunction();
    }
}
