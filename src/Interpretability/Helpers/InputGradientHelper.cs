using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interpretability.Helpers;

/// <summary>
/// Provides unified input gradient computation for interpretability explainers.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This helper computes how the model's output changes when you
/// slightly change each input feature. These "input gradients" are essential for many
/// interpretability methods like Integrated Gradients, GradCAM, and DeepLIFT.
///
/// The helper automatically chooses the best available method:
/// 1. <b>Native Backpropagation</b>: If the model is a neural network with backprop support,
///    uses the efficient built-in gradient computation.
/// 2. <b>GradientTape</b>: If the model uses the autodiff system with TensorOperations,
///    uses tape-based automatic differentiation.
/// 3. <b>Numerical Gradients</b>: As a fallback, computes approximate gradients by
///    slightly perturbing each input (slower but works with any model).
///
/// Why input gradients matter:
/// - They show the sensitivity of the output to each input feature
/// - Positive gradient: increasing this feature increases the output
/// - Negative gradient: increasing this feature decreases the output
/// - Large magnitude: the feature has strong influence on the output
/// </para>
/// </remarks>
public class InputGradientHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly INeuralNetwork<T>? _neuralNetwork;
    private readonly Func<Vector<T>, Vector<T>>? _predictFunction;
    private readonly Func<Tensor<T>, Tensor<T>>? _tensorPredictFunction;
    private readonly double _epsilon;

    /// <summary>
    /// Gets whether this helper can compute exact gradients (via backprop or autodiff).
    /// </summary>
    /// <value>True if exact gradients are available; false if numerical approximation will be used.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Exact gradients are computed mathematically using the chain rule,
    /// which is fast and accurate. Numerical gradients are approximations that require many
    /// forward passes and can have numerical precision issues.
    /// </para>
    /// </remarks>
    public bool SupportsExactGradients => _neuralNetwork != null;

    /// <summary>
    /// Gets the method name used for gradient computation.
    /// </summary>
    public string GradientMethod
    {
        get
        {
            if (_neuralNetwork != null) return "Backpropagation";
            return "NumericalApproximation";
        }
    }

    /// <summary>
    /// Creates a gradient helper from a neural network model.
    /// </summary>
    /// <param name="neuralNetwork">The neural network model that supports backpropagation.</param>
    /// <param name="epsilon">Epsilon for numerical gradients (fallback only).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the preferred way to create a gradient helper if you have
    /// a neural network model. It will use the efficient built-in backpropagation to compute
    /// exact gradients, which is much faster than numerical approximation.
    /// </para>
    /// </remarks>
    public InputGradientHelper(INeuralNetwork<T> neuralNetwork, double epsilon = 1e-4)
    {
        _neuralNetwork = neuralNetwork ?? throw new ArgumentNullException(nameof(neuralNetwork));
        _epsilon = epsilon;
    }

    /// <summary>
    /// Creates a gradient helper from a prediction function (uses numerical gradients).
    /// </summary>
    /// <param name="predictFunction">Function that maps input vector to output vector.</param>
    /// <param name="epsilon">Epsilon for numerical gradient approximation.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you only have access to a prediction function
    /// (like an external API or black-box model). The helper will compute numerical gradients
    /// by slightly perturbing each input and measuring how the output changes.
    ///
    /// This is slower than backpropagation but works with any model.
    /// </para>
    /// </remarks>
    public InputGradientHelper(Func<Vector<T>, Vector<T>> predictFunction, double epsilon = 1e-4)
    {
        _predictFunction = predictFunction ?? throw new ArgumentNullException(nameof(predictFunction));
        _epsilon = epsilon;
    }

    /// <summary>
    /// Creates a gradient helper from a tensor prediction function.
    /// </summary>
    /// <param name="tensorPredictFunction">Function that maps input tensor to output tensor.</param>
    /// <param name="epsilon">Epsilon for numerical gradient approximation.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this for models that work with tensors directly (like image
    /// models or models with batch processing). Similar to the vector version, this uses
    /// numerical gradients.
    /// </para>
    /// </remarks>
    public InputGradientHelper(Func<Tensor<T>, Tensor<T>> tensorPredictFunction, double epsilon = 1e-4)
    {
        _tensorPredictFunction = tensorPredictFunction ?? throw new ArgumentNullException(nameof(tensorPredictFunction));
        _epsilon = epsilon;
    }

    /// <summary>
    /// Computes gradients of the output with respect to the input.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <param name="outputIndex">Index of the output to compute gradients for.</param>
    /// <returns>Gradient vector with the same shape as input.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method computes the partial derivative of the specified
    /// output with respect to each input feature. The result tells you how sensitive
    /// the output is to changes in each input.
    ///
    /// Example: If gradient[i] = 0.5, then increasing input[i] by 1 would increase
    /// the output by approximately 0.5 (for small changes).
    /// </para>
    /// </remarks>
    public Vector<T> ComputeGradient(Vector<T> input, int outputIndex = 0)
    {
        if (_neuralNetwork != null)
        {
            return ComputeGradientViaBackprop(input, outputIndex);
        }
        else if (_predictFunction != null)
        {
            return ComputeNumericalGradient(input, outputIndex);
        }
        else if (_tensorPredictFunction != null)
        {
            // Convert vector to tensor and back
            var tensor = new Tensor<T>(new[] { 1, input.Length });
            for (int i = 0; i < input.Length; i++)
            {
                tensor[0, i] = input[i];
            }
            var gradTensor = ComputeGradientTensor(tensor, outputIndex);
            var grad = new T[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                grad[i] = gradTensor[0, i];
            }
            return new Vector<T>(grad);
        }

        throw new InvalidOperationException("No prediction method configured.");
    }

    /// <summary>
    /// Computes gradients for a tensor input.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="outputIndex">Index of the output to compute gradients for.</param>
    /// <returns>Gradient tensor with the same shape as input.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the tensor version of gradient computation,
    /// useful for image inputs or batched data. The gradient tensor shows how
    /// each element of the input affects the output.
    /// </para>
    /// </remarks>
    public Tensor<T> ComputeGradientTensor(Tensor<T> input, int outputIndex = 0)
    {
        if (_neuralNetwork != null)
        {
            return ComputeGradientTensorViaBackprop(input, outputIndex);
        }
        else if (_tensorPredictFunction != null)
        {
            return ComputeNumericalGradientTensor(input, outputIndex);
        }
        else if (_predictFunction != null)
        {
            // Flatten tensor, compute vector gradient, reshape back
            int total = input.Length;
            var flatInput = new T[total];
            int idx = 0;
            foreach (var val in input.ToArray())
            {
                flatInput[idx++] = val;
            }
            var grad = ComputeNumericalGradient(new Vector<T>(flatInput), outputIndex);
            var gradTensor = new Tensor<T>(input.Shape);
            for (int i = 0; i < total; i++)
            {
                gradTensor[i] = grad[i];
            }
            return gradTensor;
        }

        throw new InvalidOperationException("No prediction method configured.");
    }

    /// <summary>
    /// Computes gradients via neural network backpropagation.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <param name="outputIndex">Index of the output to differentiate.</param>
    /// <returns>Input gradients computed via backprop.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This uses the neural network's built-in backpropagation.
    /// The process is:
    /// 1. Forward pass: Run input through network, saving intermediate values
    /// 2. Create output gradient: Set gradient to 1 for the target output, 0 for others
    /// 3. Backward pass: Propagate gradients back to input
    ///
    /// This is the most efficient way to compute input gradients for neural networks.
    /// </para>
    /// </remarks>
    private Vector<T> ComputeGradientViaBackprop(Vector<T> input, int outputIndex)
    {
        // Convert vector to tensor for neural network
        var inputTensor = new Tensor<T>(new[] { 1, input.Length });
        for (int i = 0; i < input.Length; i++)
        {
            inputTensor[0, i] = input[i];
        }

        var gradTensor = ComputeGradientTensorViaBackprop(inputTensor, outputIndex);

        // Extract gradient vector
        var gradient = new T[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            gradient[i] = gradTensor[0, i];
        }

        return new Vector<T>(gradient);
    }

    /// <summary>
    /// Computes tensor gradients via neural network backpropagation.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="outputIndex">Index of the output to differentiate.</param>
    /// <returns>Input gradients as tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the tensor version of backprop gradient computation.
    /// It's used internally for image inputs or when working with multi-dimensional data.
    ///
    /// The neural network's ForwardWithMemory saves intermediate activations needed
    /// for the backward pass. The Backpropagate method then computes how the output
    /// changes with respect to the input by applying the chain rule through all layers.
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeGradientTensorViaBackprop(Tensor<T> input, int outputIndex)
    {
        if (_neuralNetwork == null)
            throw new InvalidOperationException("Neural network not configured.");

        // Set to inference mode for gradient computation
        // (we want deterministic behavior, no dropout)
        _neuralNetwork.SetTrainingMode(false);

        // Forward pass with memory (stores activations)
        var output = _neuralNetwork.ForwardWithMemory(input);

        // Create one-hot gradient for the target output
        var outputGradient = new Tensor<T>(output.Shape);
        if (outputIndex < output.Length)
        {
            outputGradient[outputIndex] = NumOps.One;
        }

        // Backward pass returns gradient with respect to input
        var inputGradient = _neuralNetwork.Backpropagate(outputGradient);

        return inputGradient;
    }

    /// <summary>
    /// Computes numerical gradient approximation for vector input.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <param name="outputIndex">Index of the output to differentiate.</param>
    /// <returns>Approximated gradient vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Numerical gradients approximate the derivative by computing:
    /// gradient[i] ≈ (f(x + ε*eᵢ) - f(x - ε*eᵢ)) / (2ε)
    ///
    /// Where eᵢ is a unit vector with 1 in position i and 0 elsewhere.
    ///
    /// This "central difference" formula is more accurate than forward/backward difference.
    /// However, it requires 2 forward passes per feature, so it's O(n) times slower
    /// than backpropagation where n is the number of features.
    /// </para>
    /// </remarks>
    private Vector<T> ComputeNumericalGradient(Vector<T> input, int outputIndex)
    {
        if (_predictFunction == null)
            throw new InvalidOperationException("Predict function not configured.");

        int n = input.Length;
        var gradient = new T[n];

        for (int i = 0; i < n; i++)
        {
            // Create perturbed inputs
            var inputPlus = new T[n];
            var inputMinus = new T[n];

            for (int j = 0; j < n; j++)
            {
                inputPlus[j] = input[j];
                inputMinus[j] = input[j];
            }

            inputPlus[i] = NumOps.FromDouble(NumOps.ToDouble(input[i]) + _epsilon);
            inputMinus[i] = NumOps.FromDouble(NumOps.ToDouble(input[i]) - _epsilon);

            // Compute predictions
            var predPlus = _predictFunction(new Vector<T>(inputPlus));
            var predMinus = _predictFunction(new Vector<T>(inputMinus));

            // Extract target output values
            double valPlus = outputIndex < predPlus.Length ? NumOps.ToDouble(predPlus[outputIndex]) : 0;
            double valMinus = outputIndex < predMinus.Length ? NumOps.ToDouble(predMinus[outputIndex]) : 0;

            // Central difference
            gradient[i] = NumOps.FromDouble((valPlus - valMinus) / (2 * _epsilon));
        }

        return new Vector<T>(gradient);
    }

    /// <summary>
    /// Computes numerical gradient approximation for tensor input.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="outputIndex">Index of the output to differentiate.</param>
    /// <returns>Approximated gradient tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Same as vector numerical gradient, but works with
    /// multi-dimensional tensors. Each element of the tensor is perturbed independently
    /// to measure its effect on the output.
    ///
    /// For a 28x28 image, this requires 784 * 2 = 1568 forward passes!
    /// This is why backpropagation (1 forward + 1 backward) is so much faster.
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeNumericalGradientTensor(Tensor<T> input, int outputIndex)
    {
        if (_tensorPredictFunction == null)
            throw new InvalidOperationException("Tensor predict function not configured.");

        var gradient = new Tensor<T>(input.Shape);
        var inputArray = input.ToArray();
        int total = input.Length;

        for (int i = 0; i < total; i++)
        {
            // Create perturbed inputs
            var inputPlus = new Tensor<T>(input.Shape);
            var inputMinus = new Tensor<T>(input.Shape);

            for (int j = 0; j < total; j++)
            {
                inputPlus[j] = inputArray[j];
                inputMinus[j] = inputArray[j];
            }

            inputPlus[i] = NumOps.FromDouble(NumOps.ToDouble(inputArray[i]) + _epsilon);
            inputMinus[i] = NumOps.FromDouble(NumOps.ToDouble(inputArray[i]) - _epsilon);

            // Compute predictions
            var predPlus = _tensorPredictFunction(inputPlus);
            var predMinus = _tensorPredictFunction(inputMinus);

            // Extract target output values
            double valPlus = outputIndex < predPlus.Length ? NumOps.ToDouble(predPlus.ToArray()[outputIndex]) : 0;
            double valMinus = outputIndex < predMinus.Length ? NumOps.ToDouble(predMinus.ToArray()[outputIndex]) : 0;

            // Central difference
            gradient[i] = NumOps.FromDouble((valPlus - valMinus) / (2 * _epsilon));
        }

        return gradient;
    }

    /// <summary>
    /// Creates a gradient function suitable for Integrated Gradients or similar methods.
    /// </summary>
    /// <returns>A function that computes gradients given input and output index.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This returns a delegate that explainers can call to compute
    /// gradients. It encapsulates the gradient computation logic so explainers don't
    /// need to know how gradients are computed internally.
    ///
    /// Usage:
    /// <code>
    /// var helper = new InputGradientHelper&lt;double&gt;(myNetwork);
    /// var gradientFunc = helper.CreateGradientFunction();
    /// var gradients = gradientFunc(input, outputIndex);
    /// </code>
    /// </para>
    /// </remarks>
    public Func<Vector<T>, int, Vector<T>> CreateGradientFunction()
    {
        return ComputeGradient;
    }

    /// <summary>
    /// Creates a tensor gradient function for image-based explainers.
    /// </summary>
    /// <returns>A function that computes tensor gradients given input and output index.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Similar to CreateGradientFunction but for tensor inputs.
    /// Useful for GradCAM and other visual explanation methods that work with images.
    /// </para>
    /// </remarks>
    public Func<Tensor<T>, int, Tensor<T>> CreateTensorGradientFunction()
    {
        return ComputeGradientTensor;
    }

    /// <summary>
    /// Computes gradients along a path from baseline to input (for Integrated Gradients).
    /// </summary>
    /// <param name="input">The input to explain.</param>
    /// <param name="baseline">The baseline (reference) input.</param>
    /// <param name="numSteps">Number of interpolation steps.</param>
    /// <param name="outputIndex">Index of the output to explain.</param>
    /// <returns>Array of gradients at each step along the path.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Integrated Gradients needs gradients at multiple points
    /// along a path from baseline to input. This method efficiently computes all those
    /// gradients in one call.
    ///
    /// The path is: baseline + α * (input - baseline) for α from 0 to 1.
    /// The method returns gradients at numSteps evenly-spaced points along this path.
    ///
    /// For neural networks, this is much more efficient than computing each gradient
    /// separately because we can batch the forward passes.
    /// </para>
    /// </remarks>
    public Vector<T>[] ComputePathGradients(
        Vector<T> input,
        Vector<T> baseline,
        int numSteps,
        int outputIndex = 0)
    {
        var gradients = new Vector<T>[numSteps];
        int n = input.Length;

        for (int step = 0; step < numSteps; step++)
        {
            double alpha = (double)step / (numSteps - 1);

            // Interpolate between baseline and input
            var interpolated = new T[n];
            for (int i = 0; i < n; i++)
            {
                double baseVal = NumOps.ToDouble(baseline[i]);
                double inputVal = NumOps.ToDouble(input[i]);
                interpolated[i] = NumOps.FromDouble(baseVal + alpha * (inputVal - baseVal));
            }

            gradients[step] = ComputeGradient(new Vector<T>(interpolated), outputIndex);
        }

        return gradients;
    }

    /// <summary>
    /// Computes Integrated Gradients attributions directly.
    /// </summary>
    /// <param name="input">The input to explain.</param>
    /// <param name="baseline">The baseline (reference) input.</param>
    /// <param name="numSteps">Number of integration steps.</param>
    /// <param name="outputIndex">Index of the output to explain.</param>
    /// <returns>Integrated Gradients attributions for each input feature.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method computes Integrated Gradients end-to-end:
    /// 1. Creates interpolation path from baseline to input
    /// 2. Computes gradients at each point on the path
    /// 3. Integrates (averages) the gradients using the trapezoidal rule
    /// 4. Multiplies by (input - baseline) to get final attributions
    ///
    /// The result satisfies the completeness axiom: attributions sum to
    /// (prediction(input) - prediction(baseline)).
    /// </para>
    /// </remarks>
    public Vector<T> ComputeIntegratedGradients(
        Vector<T> input,
        Vector<T> baseline,
        int numSteps = 50,
        int outputIndex = 0)
    {
        int n = input.Length;
        var attributions = new double[n];

        // Compute path gradients
        var pathGradients = ComputePathGradients(input, baseline, numSteps + 1, outputIndex);

        // Integrate using trapezoidal rule
        for (int step = 0; step <= numSteps; step++)
        {
            double weight = (step == 0 || step == numSteps) ? 0.5 : 1.0;

            for (int i = 0; i < n; i++)
            {
                attributions[i] += weight * NumOps.ToDouble(pathGradients[step][i]) / numSteps;
            }
        }

        // Multiply by (input - baseline)
        var result = new T[n];
        for (int i = 0; i < n; i++)
        {
            double diff = NumOps.ToDouble(input[i]) - NumOps.ToDouble(baseline[i]);
            result[i] = NumOps.FromDouble(attributions[i] * diff);
        }

        return new Vector<T>(result);
    }
}
