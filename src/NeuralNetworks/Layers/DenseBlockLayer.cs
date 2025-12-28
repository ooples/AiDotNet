using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A single layer within a DenseBlock: BN-ReLU-Conv1x1-BN-ReLU-Conv3x3.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class DenseBlockLayer<T> : LayerBase<T>
{
    private readonly BatchNormalizationLayer<T> _bn1;
    private readonly ConvolutionalLayer<T> _conv1x1;
    private readonly BatchNormalizationLayer<T> _bn2;
    private readonly ConvolutionalLayer<T> _conv3x3;
    private readonly IActivationFunction<T> _relu;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _bn1Out;
    private Tensor<T>? _relu1Out;
    private Tensor<T>? _conv1Out;
    private Tensor<T>? _bn2Out;
    private Tensor<T>? _relu2Out;

    public override bool SupportsTraining => true;

    public DenseBlockLayer(int inputChannels, int growthRate, int height, int width, double bnMomentum = 0.1)
        : base([inputChannels, height, width], [growthRate, height, width])
    {
        _relu = new ReLUActivation<T>();

        // Bottleneck layer: 1x1 conv to reduce channels (4 * growthRate is standard)
        int bottleneckChannels = 4 * growthRate;

        _bn1 = new BatchNormalizationLayer<T>(inputChannels);
        _conv1x1 = new ConvolutionalLayer<T>(
            inputDepth: inputChannels,
            outputDepth: bottleneckChannels,
            kernelSize: 1,
            inputHeight: height,
            inputWidth: width,
            stride: 1,
            padding: 0,
            activationFunction: new IdentityActivation<T>());

        _bn2 = new BatchNormalizationLayer<T>(bottleneckChannels);
        _conv3x3 = new ConvolutionalLayer<T>(
            inputDepth: bottleneckChannels,
            outputDepth: growthRate,
            kernelSize: 3,
            inputHeight: height,
            inputWidth: width,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>());
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // BN-ReLU-Conv1x1
        _bn1Out = _bn1.Forward(input);
        _relu1Out = _relu.Activate(_bn1Out);
        _conv1Out = _conv1x1.Forward(_relu1Out);

        // BN-ReLU-Conv3x3
        _bn2Out = _bn2.Forward(_conv1Out);
        _relu2Out = _relu.Activate(_bn2Out);
        var output = _conv3x3.Forward(_relu2Out);

        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput is null || _bn1Out is null || _relu1Out is null ||
            _conv1Out is null || _bn2Out is null || _relu2Out is null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Backward through conv3x3
        var grad = _conv3x3.Backward(outputGradient);

        // Backward through ReLU2
        grad = ApplyReluDerivative(_bn2Out, grad);

        // Backward through BN2
        grad = _bn2.Backward(grad);

        // Backward through conv1x1
        grad = _conv1x1.Backward(grad);

        // Backward through ReLU1
        grad = ApplyReluDerivative(_bn1Out, grad);

        // Backward through BN1
        grad = _bn1.Backward(grad);

        return grad;
    }

    private Tensor<T> ApplyReluDerivative(Tensor<T> input, Tensor<T> grad)
    {
        var result = new T[grad.Data.Length];
        for (int i = 0; i < grad.Data.Length; i++)
        {
            // ReLU derivative: 1 if x > 0, else 0
            result[i] = NumOps.GreaterThan(input.Data[i], NumOps.Zero)
                ? grad.Data[i]
                : NumOps.Zero;
        }
        return new Tensor<T>(grad.Shape, new Vector<T>(result));
    }

    public override void UpdateParameters(T learningRate)
    {
        _bn1.UpdateParameters(learningRate);
        _conv1x1.UpdateParameters(learningRate);
        _bn2.UpdateParameters(learningRate);
        _conv3x3.UpdateParameters(learningRate);
    }

    public override Vector<T> GetParameters()
    {
        var parameters = new List<T>();
        parameters.AddRange(_bn1.GetParameters().ToArray());
        parameters.AddRange(_conv1x1.GetParameters().ToArray());
        parameters.AddRange(_bn2.GetParameters().ToArray());
        parameters.AddRange(_conv3x3.GetParameters().ToArray());
        return new Vector<T>(parameters.ToArray());
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;

        int count = _bn1.GetParameters().Length;
        _bn1.SetParameters(parameters.SubVector(offset, count));
        offset += count;

        count = _conv1x1.GetParameters().Length;
        _conv1x1.SetParameters(parameters.SubVector(offset, count));
        offset += count;

        count = _bn2.GetParameters().Length;
        _bn2.SetParameters(parameters.SubVector(offset, count));
        offset += count;

        count = _conv3x3.GetParameters().Length;
        _conv3x3.SetParameters(parameters.SubVector(offset, count));
    }

    public override void ResetState()
    {
        _lastInput = null;
        _bn1Out = null;
        _relu1Out = null;
        _conv1Out = null;
        _bn2Out = null;
        _relu2Out = null;

        _bn1.ResetState();
        _conv1x1.ResetState();
        _bn2.ResetState();
        _conv3x3.ResetState();
    }

    public override bool SupportsJitCompilation => false;

    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException("DenseBlockLayer does not support JIT compilation.");
    }
}
