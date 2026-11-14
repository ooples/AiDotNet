using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

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
        Guard.NotNull(neuralNetwork);
        _neuralNetwork = neuralNetwork;
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
        Guard.NotNull(predictFunction);
        _predictFunction = predictFunction;
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
        Guard.NotNull(tensorPredictFunction);
        _tensorPredictFunction = tensorPredictFunction;
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
            var tensor = new Tensor<T>(new[] { 1, input.Length }, input.Clone());
            var gradTensor = ComputeGradientTensor(tensor, outputIndex);
            return FirstRow(gradTensor, input.Length);
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
        }
        else if (_tensorPredictFunction != null)
        {
            return ComputeNumericalGradientTensor(input, outputIndex);
        }
        else if (_predictFunction != null)
        {
            // Flatten tensor, compute vector gradient, reshape back
            var flatInput = input.ToVector();
            var grad = ComputeNumericalGradient(flatInput, outputIndex);
            return new Tensor<T>(input._shape, grad);
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
        var inputTensor = new Tensor<T>(new[] { 1, input.Length }, input.Clone());

        // Compute gradient of specific output w.r.t. input using tape
        var eng = AiDotNetEngine.Current;
        using var tape = new GradientTape<T>();
        var output = _neuralNetwork?.Predict(inputTensor) ?? inputTensor;

        // Create a selector that extracts the target output index
        // Loss = output[outputIndex] (we want gradient of this single output w.r.t. input)
        var oneHot = new Tensor<T>(output._shape);
        oneHot[outputIndex] = NumOps.One;
        var selected = eng.TensorMultiply(output, oneHot);
        var allAxes = Enumerable.Range(0, selected.Shape.Length).ToArray();
        var loss = eng.ReduceSum(selected, allAxes, keepDims: false);

        var grads = tape.ComputeGradients(loss, [inputTensor]);
        var gradTensor = grads.TryGetValue(inputTensor, out var g) ? g : new Tensor<T>(inputTensor._shape);

        // Extract gradient vector from tensor
        return FirstRow(gradTensor, input.Length);
    }

    /// <summary>
    /// The first <paramref name="length"/> entries of row 0 of a [1, n] gradient tensor.
    /// </summary>
    private static Vector<T> FirstRow(Tensor<T> gradTensor, int length)
    {
        var row = AiDotNetEngine.Current.TensorSliceAxis(gradTensor, 0, 0).ToVector();
        return row.Length == length ? row : row.GetSubVector(0, length);
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
        if (_predictFunction is not { } predict)
            throw new InvalidOperationException("Predict function not configured.");

        var engine = AiDotNetEngine.Current;
        T epsilon = NumOps.FromDouble(_epsilon);
        var inputTensor = input.AsTensor();
        var plusAll = engine.TensorAddScalar(inputTensor, epsilon);
        var minusAll = engine.TensorSubtractScalar(inputTensor, epsilon);
        var positions = engine.TensorArange<int>(0, input.Length, 1);

        // Feature i is perturbed on its own: one-hot select of the shifted values.
        return CentralDifferences(
            input.Length,
            i => predict(engine.TensorWhere(engine.TensorEqScalar(positions, i), plusAll, inputTensor).ToVector()),
            i => predict(engine.TensorWhere(engine.TensorEqScalar(positions, i), minusAll, inputTensor).ToVector()),
            outputIndex);
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
        if (_tensorPredictFunction is not { } predict)
            throw new InvalidOperationException("Tensor predict function not configured.");

        var engine = AiDotNetEngine.Current;
        T epsilon = NumOps.FromDouble(_epsilon);
        var plusAll = engine.TensorAddScalar(input, epsilon);
        var minusAll = engine.TensorSubtractScalar(input, epsilon);
        var positions = engine.TensorArange<int>(0, input.Length, 1).Reshape(input._shape);

        // Element i is perturbed on its own: one-hot select of the shifted values.
        var gradient = CentralDifferences(
            input.Length,
            i => predict(engine.TensorWhere(engine.TensorEqScalar(positions, i), plusAll, input)).ToVector(),
            i => predict(engine.TensorWhere(engine.TensorEqScalar(positions, i), minusAll, input)).ToVector(),
            outputIndex);
        return new Tensor<T>(input._shape, gradient);
    }

    /// <summary>
    /// Central differences (f(x + eps e_i) - f(x - eps e_i)) / (2 eps) of output <paramref name="outputIndex"/> for
    /// every element i, evaluating plus then minus for each element in turn. An output index past the end of the
    /// prediction contributes 0.
    /// </summary>
    private Vector<T> CentralDifferences(int count, Func<int, Vector<T>> plus, Func<int, Vector<T>> minus, int outputIndex)
    {
        var plusOutputs = new List<Vector<T>>(count);
        var minusOutputs = new List<Vector<T>>(count);
        for (int i = 0; i < count; i++)
        {
            plusOutputs.Add(plus(i));
            minusOutputs.Add(minus(i));
        }

        var engine = AiDotNetEngine.Current;
        var difference = engine.Subtract(SelectOutput(plusOutputs, outputIndex), SelectOutput(minusOutputs, outputIndex));
        return engine.Divide(difference, NumOps.FromDouble(2 * _epsilon));
    }

    /// <summary>
    /// Output <paramref name="outputIndex"/> of each prediction (0 where the prediction is shorter).
    /// </summary>
    private static Vector<T> SelectOutput(List<Vector<T>> outputs, int outputIndex)
    {
        if (outputs.Count == 0 || outputIndex >= outputs[0].Length)
            return new Vector<T>(outputs.Count);

        var engine = AiDotNetEngine.Current;
        var stacked = engine.TensorStack(outputs.Select(o => o.AsTensor()).ToArray(), 0);
        return engine.TensorSliceAxis(stacked, 1, outputIndex).ToVector();
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
        if (numSteps == 0)
            return gradients;

        // Interpolate between baseline and input: row s = baseline + alpha_s * (input - baseline),
        // alpha_s = s / (numSteps - 1)
        var engine = AiDotNetEngine.Current;
        var alphas = engine.TensorDivideScalar(engine.TensorArange<T>(0, numSteps, 1), NumOps.FromDouble(numSteps - 1));
        var offsets = engine.TensorOuter(alphas, engine.Subtract(input, baseline).AsTensor());
        var path = engine.TensorAdd(offsets, engine.TensorBroadcastTo(baseline.AsTensor().Reshape(new[] { 1, input.Length }), offsets.Shape.ToArray()));

        for (int step = 0; step < numSteps; step++)
        {
            gradients[step] = ComputeGradient(path.GetSlice(step).ToVector(), outputIndex);
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
        var engine = AiDotNetEngine.Current;

        // Compute path gradients
        var pathGradients = ComputePathGradients(input, baseline, numSteps + 1, outputIndex);

        // Integrate using trapezoidal rule: end points weigh 0.5, interior points 1, all over numSteps
        var weights = engine.Fill(numSteps + 1, NumOps.Divide(NumOps.One, NumOps.FromDouble(numSteps)));
        T halfWeight = NumOps.Divide(NumOps.FromDouble(0.5), NumOps.FromDouble(numSteps));
        weights[0] = halfWeight;
        weights[numSteps] = halfWeight;
        var stacked = engine.TensorStack(pathGradients.Select(g => g.AsTensor()).ToArray(), 0);
        var attributions = engine.TensorMatMul(weights.AsTensor().Reshape(new[] { 1, numSteps + 1 }), stacked).ToVector();

        // Multiply by (input - baseline)
        return engine.Multiply(attributions, engine.Subtract(input, baseline));
    }
}
