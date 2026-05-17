using AiDotNet.Autodiff;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using SimdVector = System.Numerics.Vector<float>;
using SimdVectorOps = System.Numerics.Vector;

namespace AiDotNet.Inference.Quantization;

/// <summary>
/// Inference-only dense layer that uses weight-only INT8 quantization (per-output scaling).
/// </summary>
internal sealed class QuantizedDenseLayer : LayerBase<float>
{
    private readonly int _inputSize;
    private readonly int _outputSize;
    private readonly sbyte[] _weightsInt8; // row-major [out, in]
    private readonly float[] _rowScales; // per out
    private readonly float[] _biases;

    public QuantizedDenseLayer(DenseLayer<float> source)
        : base(
            inputShape: source.GetInputShape(),
            outputShape: source.GetOutputShape(),
            scalarActivation: source.ScalarActivation ?? new AiDotNet.ActivationFunctions.IdentityActivation<float>())
    {
        _inputSize = source.GetInputShape()[0];
        _outputSize = source.GetOutputShape()[0];

        if (source.VectorActivation != null)
            throw new InvalidOperationException("QuantizedDenseLayer scalar-activation ctor called for a vector-activation layer.");

        var weights = source.GetWeights();
        var biases = source.GetBiases();
        if (weights == null || biases == null)
            throw new ArgumentException("Dense layer must expose weights and biases.", nameof(source));

        // DenseLayer uses [inputSize, outputSize] but we need [outputSize, inputSize] for row-major quantization
        var transposedWeights = TransposeWeights(weights, _inputSize, _outputSize);
        var q = Int8WeightOnlyQuantization.QuantizePerRow(transposedWeights, rows: _outputSize, cols: _inputSize);
        _weightsInt8 = q.Weights;
        _rowScales = q.Scales;

        _biases = biases.ToArray();
    }

    public QuantizedDenseLayer(DenseLayer<float> source, IVectorActivationFunction<float> vectorActivation)
        : base(
            inputShape: source.GetInputShape(),
            outputShape: source.GetOutputShape(),
            vectorActivation: vectorActivation)
    {
        _inputSize = source.GetInputShape()[0];
        _outputSize = source.GetOutputShape()[0];

        var weights = source.GetWeights();
        var biases = source.GetBiases();
        if (weights == null || biases == null)
            throw new ArgumentException("Dense layer must expose weights and biases.", nameof(source));

        // DenseLayer uses [inputSize, outputSize] but we need [outputSize, inputSize] for row-major quantization
        var transposedWeights = TransposeWeights(weights, _inputSize, _outputSize);
        var q = Int8WeightOnlyQuantization.QuantizePerRow(transposedWeights, rows: _outputSize, cols: _inputSize);
        _weightsInt8 = q.Weights;
        _rowScales = q.Scales;

        _biases = biases.ToArray();
    }

    private static float[] TransposeWeights(Tensor<float> weights, int inputSize, int outputSize)
    {
        // weights is [inputSize, outputSize], we need [outputSize, inputSize]
        var transposed = new float[outputSize * inputSize];
        for (int o = 0; o < outputSize; o++)
        {
            for (int i = 0; i < inputSize; i++)
            {
                transposed[o * inputSize + i] = weights[i, o];
            }
        }
        return transposed;
    }

    public override bool SupportsTraining => false;

    public override long ParameterCount => 0;

    public override Tensor<float>? GetWeights() => null;

    public override Tensor<float>? GetBiases() => null;

    public override Tensor<float> Forward(Tensor<float> input)
    {
        bool inputWas1D = false;
        Tensor<float> flat;
        if (input.Rank == 1)
        {
            inputWas1D = true;
            flat = input.Reshape(1, input.Shape[0]);
        }
        else if (input.Rank == 2)
        {
            flat = input;
        }
        else
        {
            int batch = input.Shape[0];
            int features = input.Length / batch;
            flat = input.Reshape(batch, features);
        }

        int batchSize = flat.Shape[0];
        int featuresIn = flat.Shape[1];
        if (featuresIn != _inputSize)
            throw new ArgumentException($"QuantizedDenseLayer input size mismatch. Expected {_inputSize}, got {featuresIn}.");

        var output = new Tensor<float>(new[] { batchSize, _outputSize });
        var inputSpan = flat.AsSpan();

        // SIMD-vectorize the inner dequant-on-fly matmul over input dimension.
        // Closes part of AiDotNet#1349 (consumer-side wiring of int8 weight-only
        // matmul to a vectorized hot path). Previously this triple loop did one
        // scalar load + one sbyte→float widen + one fma per inner iteration; the
        // measured wall-clock was ~20× slower than FP32 matmul (PR #1348 follow-
        // up benchmark). The vectorized form processes Vector<float>.Count
        // elements per iteration (8 on AVX2, 16 on AVX-512), with a per-chunk
        // sbyte→float widen via a small stackalloc buffer + a portable
        // Vector<float> accumulator.
        //
        // Numerical equivalence: SIMD changes the order of accumulation
        // (associativity is approximate for FP32), so the result differs from
        // the scalar path by ~1 ULP per K elements summed. For the inference-
        // only use case this is well below the INT8 quantization noise floor
        // (~35-40 dB SNR on typical transformer weights, ~1e-2 relative error).
        int vecSize = SimdVector.Count;
        Span<float> weightChunk = stackalloc float[vecSize];
        for (int b = 0; b < batchSize; b++)
        {
            int inputBase = b * featuresIn;
            int outputBase = b * _outputSize;
            for (int o = 0; o < _outputSize; o++)
            {
                float scale = _rowScales[o];
                int wBase = o * _inputSize;

                var sumVec = SimdVector.Zero;
                var scaleVec = new SimdVector(scale);
                int vectorLimit = _inputSize - (_inputSize % vecSize);
                int i = 0;
                for (; i < vectorLimit; i += vecSize)
                {
                    // Widen Vector<float>.Count contiguous sbyte weights to FP32.
                    // (Manual widen is the portable path; intrinsics-specific
                    // ConvertToVector*Int32 + ConvertToVector*Single paths are
                    // CPU-feature-gated and not portable to net471 without
                    // additional version branches. The scalar widen is the
                    // only non-vectorized op in the inner loop; the dominant
                    // cost — the fma — is fully vectorized.)
                    for (int j = 0; j < vecSize; j++)
                    {
                        weightChunk[j] = _weightsInt8[wBase + i + j];
                    }
                    var inputVec = new SimdVector(inputSpan.Slice(inputBase + i, vecSize));
                    var weightVec = new SimdVector(weightChunk);
                    sumVec += inputVec * weightVec * scaleVec;
                }

                // Reduce the SIMD accumulator to scalar. Vector.Sum is .NET 7+;
                // since this assembly targets net471 as well we sum lanes
                // explicitly.
                float laneSum = 0;
                for (int j = 0; j < vecSize; j++)
                {
                    laneSum += sumVec[j];
                }
                float sum = _biases[o] + laneSum;

                // Tail: remaining (_inputSize % vecSize) elements via scalar.
                for (; i < _inputSize; i++)
                {
                    sum += inputSpan[inputBase + i] * (_weightsInt8[wBase + i] * scale);
                }

                output.SetFlat(outputBase + o, sum);
            }
        }

        var activated = ApplyActivation(output);
        if (inputWas1D)
        {
            return activated.Reshape(_outputSize);
        }

        return activated;
    }

    public override void UpdateParameters(float learningRate)
        => throw new NotSupportedException("QuantizedDenseLayer is inference-only.");

    public override void UpdateParameters(Vector<float> parameters)
        => throw new NotSupportedException("QuantizedDenseLayer is inference-only.");

    public override Vector<float> GetParameters()
        => Vector<float>.Empty();

    public override void ResetState()
    {
        // Inference-only; no recurrent state to clear.
    }
}
