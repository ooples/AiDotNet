using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Base class for all activation functions used in neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Activation functions are mathematical operations that determine the output
/// of a neural network node. They introduce non-linearity into the network, allowing it to
/// learn complex patterns. Think of them as decision-makers that determine how strongly a
/// neuron "fires" based on its inputs.
/// 
/// Common activation functions include:
/// - Sigmoid: Outputs values between 0 and 1 (like probabilities)
/// - ReLU: Returns 0 for negative inputs, or the input value for positive inputs
/// - Tanh: Similar to sigmoid but outputs values between -1 and 1
/// 
/// The "derivative" methods are used during training to determine how to adjust the network's
/// weights to improve its accuracy.
/// </para>
/// </remarks>
public abstract class ActivationFunctionBase<T> : IActivationFunction<T>, IVectorActivationFunction<T>
{
    /// <summary>
    /// Provides mathematical operations for the numeric type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the global execution engine for vector operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Determines if the activation function supports operations on individual scalar values.
    /// </summary>
    /// <returns>True if scalar operations are supported; otherwise, false.</returns>
    protected abstract bool SupportsScalarOperations();

    /// <summary>
    /// Applies the activation function to a single input value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The activated output value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method transforms a single number using the activation function.
    /// The default implementation is the identity function (returns the input unchanged).
    /// Derived classes will override this with specific activation functions like sigmoid or ReLU.
    /// </para>
    /// </remarks>
    public virtual T Activate(T input)
    {
        return input; // Default to identity function
    }

    /// <summary>
    /// Calculates the derivative of the activation function for a single input value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The derivative value at the input point.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The derivative measures how much the activation function's output changes
    /// when its input changes slightly. This is essential for training neural networks through
    /// backpropagation. The default implementation returns 1, meaning the output changes at the
    /// same rate as the input.
    /// </para>
    /// </remarks>
    public virtual T Derivative(T input)
    {
        return NumOps.One; // Default to constant derivative of 1
    }

    /// <summary>
    /// Applies the activation function to each element in a vector.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>A new vector with the activation function applied to each element.</returns>
    public virtual Vector<T> Activate(Vector<T> input)
    {
        return input.Transform(Activate);
    }

    /// <summary>
    /// Calculates the derivative matrix for a vector input.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>A diagonal matrix containing derivatives for each input element.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a special matrix where the diagonal contains the derivatives
    /// for each input value. This matrix is used during backpropagation to efficiently calculate
    /// how errors propagate through the network.
    /// </para>
    /// </remarks>
    public virtual Matrix<T> Derivative(Vector<T> input)
    {
        return Matrix<T>.CreateDiagonal(input.Transform(Derivative));
    }

    /// <summary>
    /// Applies the activation function to each element in a tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A new tensor with the activation function applied to each element.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A tensor is a multi-dimensional array that can represent complex data
    /// structures like images (3D tensors) or video (4D tensors). This method applies the
    /// activation function to every single value in the tensor.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> Activate(Tensor<T> input)
    {
        // Use TensorAllocator.RentUninitialized — all elements are immediately overwritten below
        Tensor<T> output = TensorAllocator.RentUninitialized<T>(input._shape);
        try
        {
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = Activate(input[i]);
            }
        }
        catch
        {
            TensorAllocator.Return(output);
            throw;
        }

        return output;
    }

    /// <summary>
    /// Calculates the derivative for each element in a tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A new tensor containing derivatives for each input element.</returns>
    public virtual Tensor<T> Derivative(Tensor<T> input)
    {
        Tensor<T> output = new Tensor<T>(input._shape);
        for (int i = 0; i < input.Length; i++)
        {
            output[i] = Derivative(input[i]);
        }

        return output;
    }

    /// <summary>
    /// Calculates the backward pass gradient for this activation function.
    /// </summary>
    /// <param name="input">The input tensor from the forward pass.</param>
    /// <param name="outputGradient">The gradient flowing back from the next layer.</param>
    /// <returns>The gradient with respect to the input.</returns>
    public virtual Tensor<T> Backward(Tensor<T> input, Tensor<T> outputGradient)
    {
        // Default: element-wise product of derivative and output gradient (Hadamard product)
        var derivative = Derivative(input);
        return derivative.PointwiseMultiply(outputGradient);
    }

    /// <summary>
    /// Builds the Jacobian of an element-wise activation for a vector input: a diagonal matrix whose
    /// diagonal is the tensor derivative of <paramref name="input"/>.
    /// </summary>
    /// <param name="input">The vector at which the derivative is evaluated.</param>
    /// <returns>The [n, n] diagonal Jacobian.</returns>
    /// <remarks>
    /// <para>
    /// Only valid for activations whose <see cref="Derivative(Tensor{T})"/> returns one value per input
    /// element. The derivative is one whole-tensor engine evaluation; its values are then placed straight
    /// into the result, so the only [n, n] allocation is the matrix that is returned.
    /// </para>
    /// <para>
    /// The input and the result both live in host memory, and the work is n derivative values. Below
    /// <see cref="ScalarJacobianMaxLength"/> elements, and at any length on an engine that dispatches to a device,
    /// the fixed cost of the engine evaluation (wrapping the vector, one result tensor per operation, and on a
    /// device a dispatch and a read-back per operation) is larger than n scalar derivatives, so those cases take
    /// <see cref="ScalarDiagonalJacobian"/>. The measurements are in the pull request that introduced this.
    /// </para>
    /// </remarks>
    private protected Matrix<T> ElementwiseJacobian(Vector<T> input)
    {
        int n = input.Length;
        if (UseScalarJacobian(n) || !TensorDerivativeHonorsScalarDerivative())
        {
            return ScalarDiagonalJacobian(input);
        }

        var jacobian = new Matrix<T>(n, n);
        WriteDiagonal(jacobian, Derivative(Tensor<T>.FromVector(input)));
        return jacobian;
    }

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<Type, bool> TensorDerivativeHonorsScalar = new();

    /// <summary>
    /// Whether this runtime type's tensor derivative computes the same function as its scalar derivative: false when
    /// a subclass overrides <see cref="Derivative(T)"/> below the class that implements
    /// <see cref="Derivative(Tensor{T})"/>. Such a subclass (an ELU with its own scalar formula, say) inherits a
    /// tensor derivative that ignores its override, so its vector Jacobian must come from the scalar derivative to
    /// agree with its vector forward pass, which applies the scalar <see cref="Activate(T)"/>.
    /// </summary>
    private bool TensorDerivativeHonorsScalarDerivative()
        => TensorDerivativeHonorsScalar.GetOrAdd(GetType(), static type =>
        {
            var scalar = type.GetMethod(nameof(Derivative), new[] { typeof(T) });
            var tensor = type.GetMethod(nameof(Derivative), new[] { typeof(Tensor<T>) });
            return scalar?.DeclaringType is { } scalarOwner
                && tensor?.DeclaringType is { } tensorOwner
                && !scalarOwner.IsSubclassOf(tensorOwner);
        });

    /// <summary>
    /// The longest vector whose diagonal Jacobian is filled with scalar derivatives on the CPU engine: measured
    /// on the CPU engine, the engine evaluation is slower than the scalar loop up to this length and faster from
    /// twice this length.
    /// </summary>
    private protected const int ScalarJacobianMaxLength = 32;

    /// <summary>
    /// Whether a vector Jacobian of this length is cheaper in scalar arithmetic than through the engine: short
    /// vectors on any engine, and every length on an engine that dispatches to a device (one dispatch and read-back
    /// costs more than the whole scalar Jacobian of a vector that fits in memory as an [n, n] matrix).
    /// </summary>
    private protected bool UseScalarJacobian(int n) => n <= ScalarJacobianMaxLength || Engine.SupportsGpu;

    /// <summary>
    /// <c>factor * (diag(p) - p p^T)</c> in scalar arithmetic: the softmax Jacobian for factor 1, the softmin
    /// Jacobian for factor -1, a tempered softmax for factor 1/temperature.
    /// </summary>
    [AiDotNet.Attributes.ScalarKernel("host vector in, host matrix out: below the measured break-even length, and on a device engine at any "
        + "length, the fixed cost of the engine operations exceeds the scalar arithmetic")]
    private protected Matrix<T> ScalarSoftmaxJacobian(Vector<T> probabilities, T factor)
    {
        int n = probabilities.Length;
        var jacobian = new Matrix<T>(n, n);
        var p = probabilities.AsSpan();
        var cells = jacobian.AsWritableSpan();
        for (int i = 0; i < n; i++)
        {
            T pi = p[i];
            int row = i * n;
            for (int j = 0; j < n; j++)
            {
                cells[row + j] = NumOps.Multiply(factor, NumOps.Negate(NumOps.Multiply(pi, p[j])));
            }

            cells[row + i] = NumOps.Multiply(factor, NumOps.Multiply(pi, NumOps.Subtract(NumOps.One, pi)));
        }

        return jacobian;
    }

    /// <summary>
    /// <c>factor * (I - 1 p^T)</c> in scalar arithmetic: the log-softmax Jacobian for factor 1 and the log-softmin
    /// Jacobian for factor -1.
    /// </summary>
    [AiDotNet.Attributes.ScalarKernel("host vector in, host matrix out: below the measured break-even length, and on a device engine at any "
        + "length, the fixed cost of the engine operations exceeds the scalar arithmetic")]
    private protected Matrix<T> ScalarLogSoftmaxJacobian(Vector<T> probabilities, T factor)
    {
        int n = probabilities.Length;
        var jacobian = new Matrix<T>(n, n);
        var p = probabilities.AsSpan();
        var cells = jacobian.AsWritableSpan();
        for (int i = 0; i < n; i++)
        {
            int row = i * n;
            for (int j = 0; j < n; j++)
            {
                cells[row + j] = NumOps.Multiply(factor, NumOps.Negate(p[j]));
            }

            cells[row + i] = NumOps.Multiply(factor, NumOps.Subtract(NumOps.One, p[i]));
        }

        return jacobian;
    }

    /// <summary>
    /// The diagonal Jacobian from the scalar derivative, one value per element.
    /// </summary>
    [AiDotNet.Attributes.ScalarKernel("n scalar derivatives for a host vector: cheaper than one engine evaluation for short vectors, and "
        + "than a device dispatch and read-back at any length whose Jacobian fits in memory")]
    private Matrix<T> ScalarDiagonalJacobian(Vector<T> input)
    {
        int n = input.Length;
        var jacobian = new Matrix<T>(n, n);
        var values = input.AsSpan();
        var cells = jacobian.AsWritableSpan();
        int stride = n + 1;
        for (int i = 0; i < n; i++)
        {
            cells[i * stride] = Derivative(values[i]);
        }

        return jacobian;
    }

    /// <summary>
    /// Overwrites the main diagonal of a square matrix with <paramref name="diagonal"/>: a strided copy
    /// between the two buffers, with no arithmetic on the values.
    /// </summary>
    /// <param name="matrix">The [n, n] matrix to write into.</param>
    /// <param name="diagonal">The [n] values for the diagonal.</param>
    private protected static void WriteDiagonal(Matrix<T> matrix, Tensor<T> diagonal)
    {
        var values = diagonal.AsSpan();
        var cells = matrix.AsWritableSpan();
        int stride = matrix.Columns + 1;
        for (int i = 0; i < values.Length; i++)
        {
            cells[i * stride] = values[i];
        }
    }

    /// <summary>
    /// The matrix <c>left right^T</c> with its main diagonal replaced by <paramref name="diagonal"/>: one engine
    /// outer product written straight into the result, which is the only [n, n] allocation.
    /// </summary>
    /// <param name="left">The [n] column factor.</param>
    /// <param name="right">The [n] row factor.</param>
    /// <param name="diagonal">The [n] tensor supplying the diagonal entries.</param>
    private protected Matrix<T> OuterWithDiagonal(Tensor<T> left, Tensor<T> right, Tensor<T> diagonal)
    {
        var matrix = Engine.OuterProduct(left.ToVector(), right.ToVector());
        WriteDiagonal(matrix, diagonal);
        return matrix;
    }

    /// <summary>
    /// Returns the [n, n] tensor <paramref name="square"/> as a matrix with its main diagonal replaced by
    /// <paramref name="diagonal"/>.
    /// </summary>
    /// <param name="square">The [n, n] tensor supplying every off-diagonal entry.</param>
    /// <param name="diagonal">The [n] tensor supplying the diagonal entries.</param>
    private protected static Matrix<T> SquareWithDiagonal(Tensor<T> square, Tensor<T> diagonal)
    {
        int n = diagonal.Length;
        var matrix = Matrix<T>.FromMemory(square.ToArray(), n, n);
        WriteDiagonal(matrix, diagonal);
        return matrix;
    }

    /// <summary>
    /// Builds the Jacobian shared by the softmax family from the output probabilities:
    /// J[i, i] = p[i] * (1 - p[i]) and J[i, j] = -(p[i] * p[j]) for i != j.
    /// </summary>
    /// <param name="probabilities">The [n] output of the activation.</param>
    /// <returns>The [n, n] Jacobian; 0 x 0 for an empty input.</returns>
    private protected Matrix<T> SoftmaxJacobian(Tensor<T> probabilities)
    {
        if (probabilities.Length == 0)
        {
            return new Matrix<T>(0, 0);
        }

        var engine = Engine;
        var negated = engine.TensorNegate(probabilities);
        var oneMinus = engine.TensorAddScalar(negated, NumOps.One);
        return OuterWithDiagonal(negated, probabilities, engine.TensorMultiply(probabilities, oneMinus));
    }

    /// <summary>
    /// A tensor that is NaN wherever <paramref name="values"/> is NaN and zero everywhere else. Adding it to a
    /// result carries a NaN input through an engine reduction that would otherwise skip it.
    /// </summary>
    private protected Tensor<T> NaNOrZero(Tensor<T> values)
    {
        var engine = Engine;
        // A value that is neither above nor below zero is a zero or a NaN; keep those and zero the rest.
        var ordered = engine.TensorAdd(
            engine.TensorGreaterThan(values, NumOps.Zero), engine.TensorLessThan(values, NumOps.Zero));
        return engine.TensorWhere(ordered, engine.TensorZerosLike(values), values);
    }

    #region GPU Training Support

    /// <summary>
    /// Gets whether this activation function supports GPU-resident training.
    /// </summary>
    /// <value>
    /// True if the activation can perform forward and backward passes entirely on GPU.
    /// Default is false; derived classes override this to return true if GPU kernels are available.
    /// </value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> GPU-resident training keeps all data on the graphics card during
    /// training, avoiding slow data transfers between CPU and GPU memory. This property indicates
    /// whether this activation function can participate in GPU-resident training.
    /// </para>
    /// </remarks>
    public virtual bool SupportsGpuTraining => false;

    /// <summary>
    /// Applies the activation function on GPU.
    /// </summary>
    /// <param name="backend">The GPU backend to use for execution.</param>
    /// <param name="input">The input GPU buffer.</param>
    /// <param name="output">The output GPU buffer to store the activated values.</param>
    /// <param name="size">The number of elements to process.</param>
    /// <exception cref="NotSupportedException">
    /// Thrown because this activation function does not support GPU execution.
    /// Override this method in derived classes to provide GPU support.
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the GPU-accelerated version of the Activate method.
    /// The default implementation throws an exception because most activation functions
    /// need to provide their own GPU kernel implementation.
    /// </para>
    /// </remarks>
    public virtual void ForwardGpu(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer output, int size)
    {
        throw new NotSupportedException(
            $"{GetType().Name} does not support GPU-resident training. " +
            $"SupportsGpuTraining = {SupportsGpuTraining}. " +
            "Use the CPU-based Activate() method instead or implement GPU support in the derived class.");
    }

    #endregion
}
