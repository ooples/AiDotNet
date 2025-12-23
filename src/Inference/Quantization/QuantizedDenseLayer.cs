using AiDotNet.Autodiff;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

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

        var q = Int8WeightOnlyQuantization.QuantizePerRow(weights);
        _weightsInt8 = q.Weights;
        _rowScales = q.Scales;

        _biases = new float[biases.Length];
        for (int i = 0; i < _biases.Length; i++)
        {
            _biases[i] = biases[i];
        }
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

        var q = Int8WeightOnlyQuantization.QuantizePerRow(weights);
        _weightsInt8 = q.Weights;
        _rowScales = q.Scales;

        _biases = new float[biases.Length];
        for (int i = 0; i < _biases.Length; i++)
        {
            _biases[i] = biases[i];
        }
    }

    public override bool SupportsTraining=> false;

    public override bool SupportsJitCompilation => false;

    public override int ParameterCount => 0;

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

        for (int b = 0; b < batchSize; b++)
        {
            for (int o = 0; o < _outputSize; o++)
            {
                float sum = _biases[o];
                float scale = _rowScales[o];
                int wBase = o * _inputSize;
                for (int i = 0; i < _inputSize; i++)
                {
                    sum += flat[b, i] * (_weightsInt8[wBase + i] * scale);
                }
                output[b, o] = sum;
            }
        }

        var activated = ApplyActivation(output);
        if (inputWas1D)
        {
            return activated.Reshape(_outputSize);
        }

        return activated;
    }

    public override Tensor<float> Backward(Tensor<float> outputGradient)
        => throw new NotSupportedException("QuantizedDenseLayer is inference-only.");

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

    public override ComputationNode<float> ExportComputationGraph(List<ComputationNode<float>> inputNodes)
    {
        // WOQ is a runtime inference rewrite; we intentionally don't support JIT graph export here.
        throw new NotSupportedException("QuantizedDenseLayer does not support JIT compilation.");
    }
}
