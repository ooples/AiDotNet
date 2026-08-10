using AiDotNet.Helpers;
using AiDotNet.Autodiff;
using AiDotNet.Extensions;

// File-level, deliberately: two Tensors namespaces in the project's global usings also define a
// TensorLayout, so [TensorLayout(...)] only binds when this import shadows them from a nearer scope.
using AiDotNet.Attributes;

namespace AiDotNet.LoRA;

/// <summary>
/// Implements Low-Rank Adaptation (LoRA) layer for parameter-efficient fine-tuning of neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// LoRA works by decomposing weight updates into two low-rank matrices A and B, where the actual update
/// is computed as B * A. This dramatically reduces the number of trainable parameters compared to
/// fine-tuning all weights directly.
/// </para>
/// <para><b>For Beginners:</b> LoRA is a technique that makes it much cheaper to adapt large neural networks
/// to new tasks. Instead of updating all the weights in a layer (which can be millions of parameters),
/// LoRA adds two small matrices that work together to approximate the needed changes.
///
/// Think of it like this:
/// - Traditional fine-tuning: Adjusting every single knob on a massive control panel
/// - LoRA: Using just a few master controls that influence many knobs at once
///
/// The key insight is that the changes needed for fine-tuning often lie in a "low-rank" space,
/// meaning we don't need full freedom to adjust every parameter independently.
///
/// Key parameters:
/// - Rank (r): Controls how many "master controls" you have. Higher rank = more flexibility but more parameters
/// - Alpha: A scaling factor that controls how much influence the LoRA adaptation has
///
/// For example, adapting a layer with 1000x1000 weights (1M parameters) using LoRA with rank=8 only
/// requires 8x1000 + 8x1000 = 16,000 parameters (98.4% reduction!).
/// </para>
/// </remarks>
// The same two forms LoRAAdapterBase declares, and for the same reason: this is a linear projection, so
// it sees [Batch, Features] for the classic dense case and [Batch, Time, Features] when it wraps an
// attention or FFN sublayer. ForwardTraced states this itself - "Features live on the LAST axis; every
// leading axis is batch-like" - and its tail explicitly restores the leading axes so a rank-3 input comes
// back as [batch, seq, out].
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class LoRALayer<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Feature-last, leading axes untouched - the same rule as <c>DenseLayer</c>, which is what LoRA is a
    /// low-rank stand-in for. The output width is <c>_loraB.Shape[1]</c>, the <c>outputSize</c> constructor
    /// argument: <c>ForwardTraced</c> computes <c>(input @ A) @ B</c>, so the trailing dim of the result
    /// is B's column count by construction, not by convention.
    /// </para>
    /// <para>
    /// The rank (r) never appears here. It is an INTERNAL bottleneck - <c>input @ A</c> is
    /// <c>[.., r]</c> - and B expands straight back out, so no externally visible axis is ever sized by
    /// it. Declaring the sequence length would be the other mistake: the matrices are sized by the
    /// feature width alone, so any number of time steps is valid and pinning one would make a correct
    /// layer look like it rejects valid input.
    /// </para>
    /// <para>
    /// RANK 1 IS DELIBERATELY NOT DECLARED, and that is a defect of the layer rather than of the
    /// contract. A rank-1 <c>[features]</c> input is reshaped to <c>[1, features]</c> for the matmul, but
    /// the restore step is guarded by <c>if (input.Shape.Length &gt; 2)</c> - so it returns rank-2
    /// <c>[1, out]</c>, changing the rank. Declaring rank 1 here would have to claim either the rank it
    /// takes or the rank it returns; neither is true of both ends.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        // _loraB is built as [rank, outputSize], so the trailing axis IS the output width. This reads
        // Shape[1] rather than a Matrix-style .Columns because the parameter-automation refactor on
        // this branch moved the LoRA factors from Matrix<T> to Tensor<T>.
        int outputSize = _loraB.Shape[1];
        if (outputSize <= 0) return null;

        var features = new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(outputSize));

        return inputRank switch
        {
            2 => new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                features,
            },
            3 => new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
                features,
            },
            _ => null,
        };
    }

    /// <summary>
    /// Low-rank matrix A with dimensions (inputSize × rank).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Matrix A is the first part of the low-rank decomposition. It projects the input from
    /// inputSize dimensions down to rank dimensions. This matrix is initialized with random values
    /// and trained during fine-tuning.
    /// </para>
    /// <para><b>For Beginners:</b> This is the first of two small matrices that work together.
    /// Think of it as compressing the input data into a smaller representation before expanding it again.
    /// </para>
    /// </remarks>
    private Tensor<T> _loraA;

    /// <summary>
    /// Low-rank matrix B with dimensions (rank × outputSize).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Matrix B is the second part of the low-rank decomposition. It projects from the rank dimensions
    /// back up to outputSize dimensions. This matrix is initialized to zero so that at the start of
    /// training, the LoRA layer has no effect on the base model's behavior.
    /// </para>
    /// <para><b>For Beginners:</b> This is the second matrix that expands the compressed data back
    /// to full size. It starts at zero so the adapted model initially behaves exactly like the original.
    /// </para>
    /// </remarks>
    private Tensor<T> _loraB;

    /// <summary>
    /// The rank of the low-rank decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The rank determines the dimensionality of the intermediate representation. Lower ranks mean
    /// fewer parameters but less expressiveness. Typical values range from 1 to 64, with 8 being
    /// a common choice.
    /// </para>
    /// <para><b>For Beginners:</b> The rank is like the number of "compression channels" you use.
    /// Higher rank = more flexibility but more parameters to train. It's a trade-off between
    /// efficiency and capability.
    /// </para>
    /// </remarks>
    private readonly int _rank;

    /// <summary>
    /// Scaling factor for the LoRA contribution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Alpha controls how much the LoRA adaptation influences the final output. The actual scaling
    /// applied is alpha/rank, which helps normalize the contribution across different rank values.
    /// Typical values for alpha are in the range of the rank (e.g., alpha = 16 with rank = 8).
    /// </para>
    /// <para><b>For Beginners:</b> This controls how strongly the LoRA adaptation affects the output.
    /// It's like a volume knob for the adaptations. The formula alpha/rank automatically adjusts
    /// so that different rank values produce similar strength adaptations.
    /// </para>
    /// </remarks>
    private readonly T _alpha;

    /// <summary>
    /// Computed scaling factor (alpha / rank) used during forward pass.
    /// </summary>
    private readonly T _scaling;

    /// <summary>
    /// Gradients for matrix A computed during backpropagation.
    /// </summary>
    private Tensor<T>? _loraAGradient;

    /// <summary>
    /// Gradients for matrix B computed during backpropagation.
    /// </summary>
    private Tensor<T>? _loraBGradient;

    /// <summary>
    /// Stored input from the forward pass, needed for gradient computation.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stored pre-activation output from the forward pass, needed for activation derivative computation.
    /// </summary>
    private Tensor<T>? _lastPreActivation;

    /// <summary>
    /// Gets whether this layer supports training (always true for LoRA).
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>Construction state: the 'inputSize' the layer was built with.</summary>
    private readonly int _inputSize;

    /// <summary>Construction state: the 'outputSize' the layer was built with.</summary>
    private readonly int _outputSize;

    /// <summary>
    /// Initializes a new LoRA layer with the specified dimensions and hyperparameters.
    /// </summary>
    /// <param name="inputSize">The number of input features.</param>
    /// <param name="outputSize">The number of output features.</param>
    /// <param name="rank">The rank of the low-rank decomposition (must be positive and less than min(inputSize, outputSize)).</param>
    /// <param name="alpha">The scaling factor for LoRA contributions (typically similar to rank value).</param>
    /// <param name="activationFunction">Optional activation function to apply after the LoRA transformation.</param>
    /// <exception cref="ArgumentException">Thrown when rank is not positive or exceeds min(inputSize, outputSize).</exception>
    /// <remarks>
    /// <para>
    /// The LoRA matrices are initialized as follows:
    /// - Matrix A: Random values from a Gaussian distribution (similar to Kaiming initialization)
    /// - Matrix B: Zero initialization (so LoRA starts with no effect)
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new LoRA layer. You specify the input and output sizes
    /// (which should match the layer you're adapting), the rank (how much compression), and alpha
    /// (how strong the adaptation is).
    ///
    /// The initialization is carefully chosen:
    /// - Matrix A gets random values (so training can start moving in useful directions)
    /// - Matrix B starts at zero (so initially, LoRA doesn't change anything)
    /// </para>
    /// </remarks>
    public LoRALayer(int inputSize, int outputSize, int rank, double alpha = -1, IActivationFunction<T>? activationFunction = null)
        : base(new[] { inputSize }, new[] { outputSize }, activationFunction ?? new IdentityActivation<T>())
    {
        _outputSize = outputSize;
        _inputSize = inputSize;
        if (inputSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(inputSize), "Input size must be positive");
        }

        if (outputSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(outputSize), "Output size must be positive");
        }

        if (rank <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(rank), "Rank must be positive");
        }

        if (rank > Math.Min(inputSize, outputSize))
        {
            throw new ArgumentOutOfRangeException(nameof(rank), $"Rank ({rank}) cannot exceed min(inputSize, outputSize) = {Math.Min(inputSize, outputSize)}");
        }

        _rank = rank;

        // Default alpha to rank if not specified
        _alpha = alpha > 0 ? NumOps.FromDouble(alpha) : NumOps.FromDouble(rank);
        _scaling = NumOps.Divide(_alpha, NumOps.FromDouble(rank));

        // Initialize LoRA matrices
        // Matrix A: Random initialization (Gaussian with std = 1/sqrt(rank))
        _loraA = new Tensor<T>([inputSize, rank]);
        T stddev = NumOps.Sqrt(NumOps.Divide(NumOps.One, NumOps.FromDouble(rank)));
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < rank; j++)
            {
                _loraA[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextGaussian()), stddev);
            }
        }

        // Matrix B: Zero initialization (so LoRA has no effect initially). A fresh
        // Tensor<T> is already zero-filled, so there is nothing left to write here.
        _loraB = new Tensor<T>([rank, outputSize]);
    }

    /// <summary>
    /// Performs the forward pass through the LoRA layer.
    /// </summary>
    /// <param name="input">Input tensor of shape [batchSize, inputSize].</param>
    /// <returns>Output tensor of shape [batchSize, outputSize].</returns>
    /// <remarks>
    /// <para>
    /// The forward pass computes: output = input * A * B * scaling
    /// where scaling = alpha / rank.
    /// </para>
    /// <para><b>For Beginners:</b> This processes data through the LoRA layer. The input is:
    /// 1. Multiplied by matrix A (compressing to rank dimensions)
    /// 2. Multiplied by matrix B (expanding back to output dimensions)
    /// 3. Scaled by alpha/rank (controlling the strength)
    ///
    /// The result represents the adaptation that gets added to the base layer's output.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        _lastInput = input.Clone();

        // Features live on the LAST axis; every leading axis is batch-like.
        // Rank-2 [batch, features] is the classic dense case; rank-3
        // [batch, seq, features] is the sequence case (LoRA wrapping
        // attention / FFN sublayers inside transformer blocks). Reading
        // Shape[1] here would misinterpret the SEQUENCE dim of a rank-3
        // input as the feature dim and reject valid sequence inputs.
        int inputSize = input.Shape.Length > 1 ? input.Shape[input.Shape.Length - 1] : input.Length;
        int batchSize = inputSize > 0 ? input.Length / inputSize : input.Shape[0];

        if (inputSize != _loraA.Shape[0])
        {
            throw new ArgumentException($"Input size {inputSize} does not match expected input size {_loraA.Shape[0]}");
        }

        // Reshape input to [batch, in] without per-element copy. Engine.Reshape
        // is a zero-cost view when the tensor is contiguous; the previous
        // per-element copy loops at lines 247–278 / 290–298 dominated the
        // forward path on large LoRA ranks (e.g. ChainLoRAAdapter at
        // visionEmbeddingDim=1024 would copy 32 MB in two directions every
        // forward).
        Tensor<T> input2D = input.Shape.Length == 2 ? input : Engine.Reshape(input, [batchSize, inputSize]);

        // _loraA / _loraB ARE tensors, so Engine.TensorMatMul dispatches straight to
        // the SIMD / BLAS-backed matmul that Dense / FCL / FusedLinear share. They
        // used to be Matrix<T> mirrored into cached Tensor wrappers that had to be
        // invalidated whenever the weights changed; holding ONE representation drops
        // both the copy and the chance of the mirror going stale.
        Tensor<T> aTensor = _loraA;
        Tensor<T> bTensor = _loraB;

        // (input @ A) @ B * scaling — chained matmuls dispatched through the
        // Engine. For T=float with a base-output tensor available, the
        // Tensors-side LoRAFusionPattern auto-fuses (input @ A @ B + base)
        // into CpuFusedOperations.FusedLoRAForward (ooples/AiDotNet.Tensors#301)
        // when the compiled-plan path observes the chain — no manual
        // dispatch needed here, the fusion runs inside Engine.TensorMatMul's
        // pattern matcher when the LoRA branch is part of a larger graph.
        // The standalone delta path still benefits from the matmul SIMD
        // + amortizes the original per-element copy cost.
        Tensor<T> intermediate = Engine.TensorMatMul(input2D, aTensor);
        Tensor<T> deltaOutput = Engine.TensorMatMul(intermediate, bTensor);
        deltaOutput = Engine.TensorMultiplyScalar(deltaOutput, _scaling);

        // Store pre-activation for gradient computation. Reshape preserves
        // tensor identity; the [.., outputSize] -> [batch, outputSize]
        // assertion lets downstream consumers see a stable rank-2 surface.
        _lastPreActivation = deltaOutput.Clone();

        // Apply activation if specified
        Tensor<T> result = ScalarActivation != null ? ApplyActivation(deltaOutput) : deltaOutput;

        // Restore the original leading (batch-like) axes so the delta can be
        // added elementwise to the base layer's output: a rank-3
        // [batch, seq, in] input must come back as [batch, seq, out], not the
        // flattened [batch*seq, out] the matmul produced.
        if (input.Shape.Length > 2)
        {
            var outShape = new int[input.Shape.Length];
            for (int i = 0; i < input.Shape.Length - 1; i++)
                outShape[i] = input.Shape[i];
            outShape[input.Shape.Length - 1] = result.Shape[result.Shape.Length - 1];
            result = Engine.Reshape(result, outShape);
        }

        return result;
    }

    /// <summary>
    /// Updates the layer's parameters using the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    public override void UpdateParameters(T learningRate)
    {
        if (_loraAGradient == null || _loraBGradient == null)
        {
            return;
        }

        // Update matrix A
        for (int i = 0; i < _loraA.Shape[0]; i++)
        {
            for (int j = 0; j < _loraA.Shape[1]; j++)
            {
                T update = NumOps.Multiply(_loraAGradient[i, j], learningRate);
                _loraA[i, j] = NumOps.Subtract(_loraA[i, j], update);
            }
        }

        // Update matrix B
        for (int i = 0; i < _loraB.Shape[0]; i++)
        {
            for (int j = 0; j < _loraB.Shape[1]; j++)
            {
                T update = NumOps.Multiply(_loraBGradient[i, j], learningRate);
                _loraB[i, j] = NumOps.Subtract(_loraB[i, j], update);
            }
        }
    }

    /// <summary>
    /// Updates the parameter gradients vector from the matrix gradients.
    /// </summary>
    private void UpdateParameterGradients()
    {
        if (_loraAGradient == null || _loraBGradient == null)
        {
            return;
        }

        ParameterGradients = new Vector<T>(ParameterCountHelper.ToFlatVectorSize(ParameterCount));
        int idx = 0;

        // Pack matrix A gradients
        for (int i = 0; i < _loraAGradient.Shape[0]; i++)
        {
            for (int j = 0; j < _loraAGradient.Shape[1]; j++)
            {
                ParameterGradients[idx++] = _loraAGradient[i, j];
            }
        }

        // Pack matrix B gradients
        for (int i = 0; i < _loraBGradient.Shape[0]; i++)
        {
            for (int j = 0; j < _loraBGradient.Shape[1]; j++)
            {
                ParameterGradients[idx++] = _loraBGradient[i, j];
            }
        }
    }

    /// <summary>
    /// Merges the LoRA weights into a dense weight matrix that can be added to a base layer.
    /// </summary>
    /// <returns>The merged weight matrix (inputSize × outputSize) representing the full LoRA contribution.</returns>
    /// <remarks>
    /// <para>
    /// This computes the full weight matrix W_lora = A * B * scaling, which can then be added to the
    /// base layer's weights. This is useful for deployment when you want to merge the adaptation
    /// back into the base model for inference efficiency.
    /// </para>
    /// <para><b>For Beginners:</b> This "bakes in" the LoRA adaptation into a regular weight matrix.
    /// Instead of storing two small matrices (A and B) and computing them during inference,
    /// you can merge them into one larger matrix and add it to the original weights.
    ///
    /// This is like converting assembly instructions back into a final product - once you're done
    /// training, you can simplify the model for faster inference.
    /// </para>
    /// </remarks>
    public Matrix<T> MergeWeights()
    {
        // Compute W_lora = A * B * scaling
        // A: [inputSize, rank], B: [rank, outputSize]
        // Result: [inputSize, outputSize] - matches DenseLayer's industry standard convention
        return Engine.TensorMultiplyScalar(Engine.TensorMatMul(_loraA, _loraB), _scaling).ToMatrix();
    }

    /// <summary>
    /// Gets the rank of this LoRA layer.
    /// </summary>
    public int Rank => _rank;

    /// <summary>
    /// Gets the alpha scaling factor.
    /// </summary>
    public T Alpha => _alpha;

    /// <summary>
    /// Gets the computed scaling factor (alpha / rank).
    /// </summary>
    public T Scaling => _scaling;

    /// <summary>
    /// Gets matrix A (for inspection or advanced use cases).
    /// </summary>
    public Matrix<T> GetMatrixA() => _loraA.ToMatrix();

    /// <summary>
    /// Gets matrix B (for inspection or advanced use cases).
    /// </summary>
    public Matrix<T> GetMatrixB() => _loraB.ToMatrix();

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For LoRA layers, this clears the stored input from the last forward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This clears the layer's memory of the last input it processed.
    /// It's like hitting a reset button before processing a new, unrelated batch of data.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        _lastInput = null;
        _lastPreActivation = null;
        _loraAGradient = null;
        _loraBGradient = null;
    }


}
