namespace AiDotNet.NeuralNetworks.Layers;

public class ConvolutionalLayer<T> : LayerBase<T>
{
    public int InputDepth { get; }
    public int OutputDepth { get; }
    public int KernelSize { get; }
    public int Stride { get; }
    public int Padding { get; }
    
    private Tensor<T> Kernels { get; }
    private Vector<T> Biases { get; }
    private Tensor<T> LastInput { get; set; }
    private Tensor<T> LastOutput { get; set; }
    private readonly Random _random;

    public ConvolutionalLayer(int inputDepth, int outputDepth, int kernelSize, int inputHeight, int inputWidth, int stride = 1, int padding = 0, 
                              IActivationFunction<T>? activation = null)
        : base(CalculateInputShape(inputDepth, inputHeight, inputWidth), 
               CalculateOutputShape(outputDepth, CalculateOutputDimension(inputHeight, kernelSize, stride, padding), 
                   CalculateOutputDimension(inputWidth, kernelSize, stride, padding)), activation ?? new ReLUActivation<T>())
    {
        InputDepth = inputDepth;
        OutputDepth = outputDepth;
        KernelSize = kernelSize;
        Stride = stride;
        Padding = padding;

        Kernels = new Tensor<T>([OutputDepth, InputDepth, KernelSize, KernelSize]);
        Biases = new Vector<T>(OutputDepth);
        LastInput = new Tensor<T>([OutputDepth, InputDepth, KernelSize, KernelSize]);
        LastOutput = new Tensor<T>([OutputDepth, InputDepth, KernelSize, KernelSize]);
        _random = new Random();

        InitializeWeights();
    }

    public ConvolutionalLayer(int inputDepth, int outputDepth, int kernelSize, int inputHeight, int inputWidth, int stride = 1, int padding = 0, 
                              IVectorActivationFunction<T>? vectorActivation = null)
        : base(CalculateInputShape(inputDepth, inputHeight, inputWidth), 
               CalculateOutputShape(outputDepth, CalculateOutputDimension(inputHeight, kernelSize, stride, padding), 
                   CalculateOutputDimension(inputWidth, kernelSize, stride, padding)), vectorActivation ?? new SoftmaxActivation<T>())
    {
        InputDepth = inputDepth;
        OutputDepth = outputDepth;
        KernelSize = kernelSize;
        Stride = stride;
        Padding = padding;

        Kernels = new Tensor<T>([OutputDepth, InputDepth, KernelSize, KernelSize]);
        Biases = new Vector<T>(OutputDepth);
        LastInput = new Tensor<T>([OutputDepth, InputDepth, KernelSize, KernelSize]);
        LastOutput = new Tensor<T>([OutputDepth, InputDepth, KernelSize, KernelSize]);
        _random = new Random();

        InitializeWeights();
    }

    private static int CalculateOutputDimension(int inputDim, int kernelSize, int stride, int padding)
    {
        return (inputDim - kernelSize + 2 * padding) / stride + 1;
    }

    private void InitializeWeights()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (InputDepth * KernelSize * KernelSize + OutputDepth)));
    
        for (int i = 0; i < OutputDepth; i++)
        {
            for (int j = 0; j < InputDepth; j++)
            {
                for (int k = 0; k < KernelSize; k++)
                {
                    for (int l = 0; l < KernelSize; l++)
                        {
                        Kernels[i, j, k, l] = NumOps.Multiply(scale, NumOps.FromDouble(_random.NextDouble() * 2 - 1));
                    }
                }
            }

            Biases[i] = NumOps.Zero;
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        LastInput = input;
        int batchSize = input.Shape[0];
        int inputHeight = input.Shape[2];
        int inputWidth = input.Shape[3];
        int outputHeight = (inputHeight - KernelSize + 2 * Padding) / Stride + 1;
        int outputWidth = (inputWidth - KernelSize + 2 * Padding) / Stride + 1;

        Tensor<T> output = new Tensor<T>([batchSize, OutputDepth, outputHeight, outputWidth]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int o = 0; o < OutputDepth; o++)
            {
                for (int y = 0; y < outputHeight; y++)
                {
                    for (int x = 0; x < outputWidth; x++)
                    {
                        T sum = Biases[o];
                        for (int i = 0; i < InputDepth; i++)
                        {
                            for (int ky = 0; ky < KernelSize; ky++)
                            {
                                for (int kx = 0; kx < KernelSize; kx++)
                                {
                                    int inputY = y * Stride + ky - Padding;
                                    int inputX = x * Stride + kx - Padding;
                                    if (inputY >= 0 && inputY < inputHeight && inputX >= 0 && inputX < inputWidth)
                                    {
                                        sum = NumOps.Add(sum, NumOps.Multiply(input[b, i, inputY, inputX], Kernels[o, i, ky, kx]));
                                    }
                                }
                            }
                        }

                        output[b, o, y, x] = sum;
                    }
                }
            }
        }

        LastOutput = ApplyActivation(output);
        return LastOutput;
    }

    

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        Tensor<T> activationGradient = ApplyActivationDerivative(LastOutput, outputGradient);
        outputGradient = Tensor<T>.ElementwiseMultiply(outputGradient, activationGradient);

        int batchSize = LastInput.Shape[0];
        int inputHeight = LastInput.Shape[2];
        int inputWidth = LastInput.Shape[3];
        int outputHeight = outputGradient.Shape[2];
        int outputWidth = outputGradient.Shape[3];

        Tensor<T> inputGradient = new Tensor<T>(LastInput.Shape);
        Tensor<T> kernelGradients = new Tensor<T>(Kernels.Shape);
        Vector<T> biasGradients = new Vector<T>(OutputDepth);

        for (int b = 0; b < batchSize; b++)
        {
            for (int o = 0; o < OutputDepth; o++)
            {
                for (int y = 0; y < outputHeight; y++)
                {
                    for (int x = 0; x < outputWidth; x++)
                    {
                        T outputGrad = outputGradient[b, o, y, x];
                        biasGradients[o] = NumOps.Add(biasGradients[o], outputGrad);

                        for (int i = 0; i < InputDepth; i++)
                        {
                            for (int ky = 0; ky < KernelSize; ky++)
                            {
                                for (int kx = 0; kx < KernelSize; kx++)
                                {
                                    int inputY = y * Stride + ky - Padding;
                                    int inputX = x * Stride + kx - Padding;
                                    if (inputY >= 0 && inputY < inputHeight && inputX >= 0 && inputX < inputWidth)
                                    {
                                        T inputValue = LastInput[b, i, inputY, inputX];
                                        kernelGradients[o, i, ky, kx] = NumOps.Add(kernelGradients[o, i, ky, kx], NumOps.Multiply(outputGrad, inputValue));
                                        inputGradient[b, i, inputY, inputX] = NumOps.Add(inputGradient[b, i, inputY, inputX], NumOps.Multiply(outputGrad, Kernels[o, i, ky, kx]));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Update kernels and biases
        for (int i = 0; i < Kernels.Length; i++)
        {
            Kernels[i] = NumOps.Subtract(Kernels[i], NumOps.Multiply(NumOps.FromDouble(0.01), kernelGradients[i])); // Learning rate of 0.01
        }

        for (int i = 0; i < Biases.Length; i++)
        {
            Biases[i] = NumOps.Subtract(Biases[i], NumOps.Multiply(NumOps.FromDouble(0.01), biasGradients[i])); // Learning rate of 0.01
        }

        return inputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        // Update kernels
        for (int o = 0; o < OutputDepth; o++)
        {
            for (int i = 0; i < InputDepth; i++)
            {
                for (int ky = 0; ky < KernelSize; ky++)
                {
                    for (int kx = 0; kx < KernelSize; kx++)
                    {
                        T update = NumOps.Multiply(learningRate, Kernels[o, i, ky, kx]);
                        Kernels[o, i, ky, kx] = NumOps.Subtract(Kernels[o, i, ky, kx], update);
                    }
                }
            }
        }

        // Update biases
        for (int o = 0; o < OutputDepth; o++)
        {
            T update = NumOps.Multiply(learningRate, Biases[o]);
            Biases[o] = NumOps.Subtract(Biases[o], update);
        }
    }
}