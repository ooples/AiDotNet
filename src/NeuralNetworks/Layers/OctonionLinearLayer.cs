#pragma warning disable CS0649, CS0414, CS0169
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a fully connected layer using octonion-valued weights and inputs.
/// </summary>
/// <remarks>
/// <para>
/// An octonion linear layer performs matrix-vector multiplication in the 8-dimensional
/// octonion algebra. Each weight and input is an octonion (8 real components), enabling
/// the layer to capture more complex relationships than real-valued layers.
/// </para>
/// <para><b>For Beginners:</b> This layer is like a regular dense layer, but it uses
/// 8-dimensional numbers (octonions) instead of regular numbers.
///
/// Benefits of octonion layers:
/// - Can model more complex relationships with fewer parameters
/// - Useful for certain types of image and signal processing
/// - Better at capturing rotational relationships in data
///
/// The tradeoff is that computations are more expensive per parameter.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (float or double).</typeparam>
[LayerCategory(LayerCategory.Dense)]
[LayerTask(LayerTask.Projection)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "1, 128", TestConstructorArgs = "16, 8")]
public class OctonionLinearLayer<T> : LayerBase<T>
{

    /// <summary>
    /// The octonion weight tensor with shape [OutputFeatures, InputFeatures, 8].
    /// Each [o, i, 0..7] slice stores one octonion's 8 components (Scalar, E1-E7).
    /// </summary>
    private Tensor<T> _weights;

    /// <summary>
    /// The octonion bias tensor with shape [OutputFeatures, 8].
    /// Each [o, 0..7] slice stores one octonion's 8 components.
    /// </summary>
    private Tensor<T> _biases;

    /// <summary>
    /// Stored input from forward pass for backpropagation.
    /// Shape: [batch, inputFeatures, 8]
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// Stored pre-activation output for gradient computation.
    /// Shape: [batch, outputFeatures, 8]
    /// </summary>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Gradient for weights. Shape: [OutputFeatures, InputFeatures, 8]
    /// </summary>
    private Tensor<T>? _weightsGradient;

    /// <summary>
    /// Gradient for biases. Shape: [OutputFeatures, 8]
    /// </summary>
    private Tensor<T>? _biasesGradient;

    /// <summary>
    /// Gets the number of input features (octonion-valued).
    /// </summary>
    public int InputFeatures { get; }

    /// <summary>
    /// Gets the number of output features (octonion-valued).
    /// </summary>
    public int OutputFeatures { get; }

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    /// <remarks>
    /// Each octonion has 8 real components, so the parameter count is:
    /// (InputFeatures * OutputFeatures + OutputFeatures) * 8
    /// </remarks>
    public override int ParameterCount =>
        (InputFeatures * OutputFeatures + OutputFeatures) * 8;

    /// <summary>
    /// Gets whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Initializes a new instance of the OctonionLinearLayer.
    /// </summary>
    /// <param name="inputFeatures">Number of input features (octonion-valued).</param>
    /// <param name="outputFeatures">Number of output features (octonion-valued).</param>
    /// <param name="activationFunction">Optional activation function.</param>
    public OctonionLinearLayer(
        int inputFeatures,
        int outputFeatures,
        IActivationFunction<T>? activationFunction = null)
        : base(
            [inputFeatures * 8], // Input shape: inputFeatures octonions = inputFeatures * 8 reals
            [outputFeatures * 8], // Output shape: outputFeatures octonions = outputFeatures * 8 reals
            activationFunction ?? new IdentityActivation<T>())
    {
        InputFeatures = inputFeatures;
        OutputFeatures = outputFeatures;


        _weights = new Tensor<T>([outputFeatures, inputFeatures, 8]);
        _biases = new Tensor<T>([outputFeatures, 8]);

        InitializeParameters();

        RegisterTrainableParameter(_weights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Initializes weights using Xavier/Glorot initialization adapted for octonions.
    /// Writes directly to the Tensor<T> components — no Octonion<T> object allocation.
    /// </summary>
    private void InitializeParameters()
    {
        var scale = Math.Sqrt(2.0 / (InputFeatures + OutputFeatures));
        var random = RandomHelper.CreateSeededRandom(42);

        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                for (int c = 0; c < 8; c++)
                    _weights[o, i, c] = NumOps.FromDouble((random.NextDouble() - 0.5) * 2 * scale);
            }
            // Initialize biases to zero (Tensor<T> is zero-initialized by default)
        }
    }

    /// <summary>
    /// Performs the forward pass through the layer.
    /// </summary>
    /// <param name="input">Input tensor with shape [inputFeatures * 8] or [batch, inputFeatures * 8].</param>
    /// <returns>Output tensor with shape [outputFeatures * 8] or [batch, outputFeatures * 8].</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input.Shape.ToArray();

        int batchSize;
        int inputLen;
        Tensor<T> inputTensor;

        if (input.Rank == 1)
        {
            batchSize = 1;
            inputLen = input.Shape[0];
            inputTensor = input.Reshape([1, inputLen]);
        }
        else if (input.Rank == 2)
        {
            batchSize = input.Shape[0];
            inputLen = input.Shape[1];
            inputTensor = input;
        }
        else
        {
            int flatBatch = 1;
            for (int d = 0; d < input.Rank - 1; d++)
            {
                flatBatch *= input.Shape[d];
            }
            batchSize = flatBatch;
            inputLen = input.Shape[input.Rank - 1];
            inputTensor = input.Reshape([batchSize, inputLen]);
        }

        // Validate input shape
        if (inputLen != InputFeatures * 8)
        {
            throw new ArgumentException(
                $"Input size {inputLen} does not match expected {InputFeatures * 8} " +
                $"({InputFeatures} octonions * 8 components).");
        }

        // Reshape flat input [batch, inputFeatures*8] → [batch, inputFeatures, 8]
        var input3D = inputTensor.Reshape([batchSize, InputFeatures, 8]);
        _lastInput = input3D;

        // Tensor-based octonion matrix multiplication — no Octonion<T> allocation
        var output3D = Engine.OctonionMatMulTensor(input3D, _weights);

        // Add biases: broadcast [OutputFeatures, 8] across batch dimension
        for (int b = 0; b < batchSize; b++)
        {
            for (int o = 0; o < OutputFeatures; o++)
            {
                for (int c = 0; c < 8; c++)
                    output3D[b, o, c] = NumOps.Add(output3D[b, o, c], _biases[o, c]);
            }
        }

        _lastOutput = output3D;

        // Flatten back to [batch, outputFeatures*8] for activation and output
        var outputTensor = output3D.Reshape([batchSize, OutputFeatures * 8]);

        // Apply activation function
        var activated = ApplyActivation(outputTensor);

        if (_originalInputShape == null || _originalInputShape.Length == 2)
        {
            return activated;
        }

        if (_originalInputShape.Length == 1)
        {
            return activated.Reshape([OutputFeatures * 8]);
        }

        var outputShape = new int[_originalInputShape.Length];
        for (int d = 0; d < _originalInputShape.Length - 1; d++)
        {
            outputShape[d] = _originalInputShape[d];
        }
        outputShape[_originalInputShape.Length - 1] = OutputFeatures * 8;
        return activated.Reshape(outputShape);
    }

    /// <summary>
    /// Updates the parameters of the layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the update.</param>
    public override void UpdateParameters(T learningRate)
    {
        if (_weightsGradient == null || _biasesGradient == null)
        {
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        }

        // Update weights: w = w - lr * grad (tensor ops, no Octonion<T> objects)
        var scaledWeightGrad = Engine.TensorMultiplyScalar(_weightsGradient, learningRate);
        Engine.TensorSubtractInPlace(_weights, scaledWeightGrad);

        // Update biases
        var scaledBiasGrad = Engine.TensorMultiplyScalar(_biasesGradient, learningRate);
        Engine.TensorSubtractInPlace(_biases, scaledBiasGrad);
    }

    /// <summary>
    /// Gets all trainable parameters as a single vector.
    /// </summary>
    /// <returns>A vector containing all weights and biases.</returns>
    public override Vector<T> GetParameters()
    {
        // Weights and biases are Tensor<T>, flatten directly
        var weightsFlat = _weights.Reshape([OutputFeatures * InputFeatures * 8]);
        var biasesFlat = _biases.Reshape([OutputFeatures * 8]);

        var paramArray = new T[ParameterCount];
        for (int i = 0; i < weightsFlat.Length; i++)
            paramArray[i] = weightsFlat[i];
        for (int i = 0; i < biasesFlat.Length; i++)
            paramArray[weightsFlat.Length + i] = biasesFlat[i];

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

        int weightSize = OutputFeatures * InputFeatures * 8;
        int biasSize = OutputFeatures * 8;

        // Copy weights from flat vector into tensor
        for (int i = 0; i < weightSize; i++)
            _weights.SetFlat(i, parameters[i]);
        for (int i = 0; i < biasSize; i++)
            _biases.SetFlat(i, parameters[weightSize + i]);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        var gradients = new Vector<T>(ParameterCount);
        int weightSize = OutputFeatures * InputFeatures * 8;

        if (_weightsGradient is not null)
        {
            for (int i = 0; i < weightSize; i++)
                gradients[i] = _weightsGradient.GetFlat(i);
        }

        if (_biasesGradient is not null)
        {
            int biasSize = OutputFeatures * 8;
            for (int i = 0; i < biasSize; i++)
                gradients[weightSize + i] = _biasesGradient.GetFlat(i);
        }

        return gradients;
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    public override void ClearGradients()
    {
        base.ClearGradients();
        _weightsGradient = null;
        _biasesGradient = null;
    }

    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _originalInputShape = null;
        _weightsGradient = null;
        _biasesGradient = null;
    }

}
