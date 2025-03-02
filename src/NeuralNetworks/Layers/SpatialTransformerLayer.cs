namespace AiDotNet.NeuralNetworks.Layers;

public class SpatialTransformerLayer<T> : LayerBase<T>
{
    private Matrix<T> _localizationWeights1;
    private Vector<T> _localizationBias1;
    private Matrix<T> _localizationWeights2;
    private Vector<T> _localizationBias2;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Matrix<T>? _lastTransformationMatrix;

    private Matrix<T>? _localizationWeights1Gradient;
    private Vector<T>? _localizationBias1Gradient;
    private Matrix<T>? _localizationWeights2Gradient;
    private Vector<T>? _localizationBias2Gradient;

    private readonly int _inputHeight;
    private readonly int _inputWidth;
    private readonly int _outputHeight;
    private readonly int _outputWidth;

    public override bool SupportsTraining => true;

    public SpatialTransformerLayer(int inputHeight, int inputWidth, int outputHeight, int outputWidth, IActivationFunction<T>? activationFunction = null)
        : base([inputHeight, inputWidth], [outputHeight, outputWidth], activationFunction ?? new TanhActivation<T>())
    {
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        _outputHeight = outputHeight;
        _outputWidth = outputWidth;

        // Initialize localization network weights and biases
        _localizationWeights1 = new Matrix<T>(inputHeight * inputWidth, 32);
        _localizationBias1 = new Vector<T>(32);
        _localizationWeights2 = new Matrix<T>(32, 6);
        _localizationBias2 = new Vector<T>(6);

        InitializeParameters();
    }

    public SpatialTransformerLayer(int inputHeight, int inputWidth, int outputHeight, int outputWidth, IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base([inputHeight, inputWidth], [outputHeight, outputWidth], vectorActivationFunction ?? new TanhActivation<T>())
    {
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        _outputHeight = outputHeight;
        _outputWidth = outputWidth;

        // Initialize localization network weights and biases
        _localizationWeights1 = new Matrix<T>(inputHeight * inputWidth, 32);
        _localizationBias1 = new Vector<T>(32);
        _localizationWeights2 = new Matrix<T>(32, 6);
        _localizationBias2 = new Vector<T>(6);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_localizationWeights1.Rows + _localizationWeights1.Columns)));
        InitializeMatrix(_localizationWeights1, scale);
        InitializeMatrix(_localizationWeights2, scale);

        for (int i = 0; i < _localizationBias1.Length; i++)
        {
            _localizationBias1[i] = NumOps.Zero;
        }

        // Initialize the localization bias2 to represent identity transformation
        _localizationBias2[0] = NumOps.One;
        _localizationBias2[4] = NumOps.One;
    }

    private void InitializeMatrix(Matrix<T> matrix, T scale)
    {
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                matrix[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
            }
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];

        // Localization network
        var flattenedInput = input.Reshape(batchSize, _inputHeight * _inputWidth);
        var localization1 = flattenedInput.Multiply(_localizationWeights1).Add(_localizationBias1);
        localization1 = ApplyActivation(localization1);
        var transformationParams = localization1.Multiply(_localizationWeights2).Add(_localizationBias2);

        // Convert transformation parameters to 2x3 transformation matrices
        _lastTransformationMatrix = ConvertToTransformationMatrix(transformationParams);

        // Grid generator
        var outputGrid = GenerateOutputGrid();

        // Sampler
        var output = SampleInputImage(input, outputGrid, _lastTransformationMatrix);

        _lastOutput = output;
        return _lastOutput;
    }

    private Matrix<T> ConvertToTransformationMatrix(Tensor<T> transformationParams)
    {
        if (transformationParams.Shape[0] != 1 || transformationParams.Shape[1] != 6)
        {
            throw new ArgumentException("Transformation parameters should be a 1x6 tensor.");
        }

        var matrix = new Matrix<T>(2, 3);

        // Extract the parameters
        T theta11 = transformationParams[0, 0];
        T theta12 = transformationParams[0, 1];
        T theta13 = transformationParams[0, 2];
        T theta21 = transformationParams[0, 3];
        T theta22 = transformationParams[0, 4];
        T theta23 = transformationParams[0, 5];

        // Apply constraints to prevent extreme transformations
        T scale = NumOps.FromDouble(0.1); // Adjust this value to control the scale of transformations
    
        // Limit scaling and shearing
        theta11 = MathHelper.Tanh(NumOps.Multiply(theta11, scale));
        theta12 = MathHelper.Tanh(NumOps.Multiply(theta12, scale));
        theta21 = MathHelper.Tanh(NumOps.Multiply(theta21, scale));
        theta22 = MathHelper.Tanh(NumOps.Multiply(theta22, scale));

        // Limit translation
        theta13 = MathHelper.Tanh(NumOps.Multiply(theta13, scale));
        theta23 = MathHelper.Tanh(NumOps.Multiply(theta23, scale));

        // Ensure the transformation is close to identity if parameters are small
        T epsilon = NumOps.FromDouble(1e-5);
        theta11 = NumOps.Add(theta11, NumOps.One);
        theta22 = NumOps.Add(theta22, NumOps.One);

        // Construct the transformation matrix
        matrix[0, 0] = theta11;
        matrix[0, 1] = theta12;
        matrix[0, 2] = theta13;
        matrix[1, 0] = theta21;
        matrix[1, 1] = theta22;
        matrix[1, 2] = theta23;

        return matrix;
    }

    private Tensor<T> GenerateOutputGrid()
    {
        // Generate a grid of (x, y) coordinates for the output
        var grid = new Tensor<T>([_outputHeight, _outputWidth, 2]);
        for (int i = 0; i < _outputHeight; i++)
        {
            for (int j = 0; j < _outputWidth; j++)
            {
                grid[i, j, 0] = NumOps.FromDouble((double)j / (_outputWidth - 1) * 2 - 1);
                grid[i, j, 1] = NumOps.FromDouble((double)i / (_outputHeight - 1) * 2 - 1);
            }
        }

        return grid;
    }

    private Tensor<T> SampleInputImage(Tensor<T> input, Tensor<T> outputGrid, Matrix<T> transformationMatrix)
    {
        int batchSize = input.Shape[0];
        int channels = input.Shape[3];
        var output = new Tensor<T>([batchSize, _outputHeight, _outputWidth, channels]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int y = 0; y < _outputHeight; y++)
            {
                for (int x = 0; x < _outputWidth; x++)
                {
                    // Apply transformation to output coordinates
                    T srcX = NumOps.Add(
                        NumOps.Add(
                            NumOps.Multiply(transformationMatrix[0, 0], outputGrid[y, x, 0]),
                            NumOps.Multiply(transformationMatrix[0, 1], outputGrid[y, x, 1])
                        ),
                        transformationMatrix[0, 2]
                    );
                    T srcY = NumOps.Add(
                        NumOps.Add(
                            NumOps.Multiply(transformationMatrix[1, 0], outputGrid[y, x, 0]),
                            NumOps.Multiply(transformationMatrix[1, 1], outputGrid[y, x, 1])
                        ),
                        transformationMatrix[1, 2]
                    );

                    // Convert to input image coordinates
                    srcX = NumOps.Multiply(NumOps.Add(srcX, NumOps.One), NumOps.Divide(NumOps.FromDouble(_inputWidth - 1), NumOps.FromDouble(2)));
                    srcY = NumOps.Multiply(NumOps.Add(srcY, NumOps.One), NumOps.Divide(NumOps.FromDouble(_inputHeight - 1), NumOps.FromDouble(2)));

                    // Compute the four nearest neighbor coordinates
                    int x0 = (int)Math.Floor(Convert.ToDouble(srcX));
                    int x1 = Math.Min(x0 + 1, _inputWidth - 1);
                    int y0 = (int)Math.Floor(Convert.ToDouble(srcY));
                    int y1 = Math.Min(y0 + 1, _inputHeight - 1);

                    // Compute interpolation weights
                    T wx1 = NumOps.Subtract(srcX, NumOps.FromDouble(x0));
                    T wx0 = NumOps.Subtract(NumOps.One, wx1);
                    T wy1 = NumOps.Subtract(srcY, NumOps.FromDouble(y0));
                    T wy0 = NumOps.Subtract(NumOps.One, wy1);

                    // Perform bilinear interpolation for each channel
                    for (int c = 0; c < channels; c++)
                    {
                        T v00 = input[b, y0, x0, c];
                        T v01 = input[b, y0, x1, c];
                        T v10 = input[b, y1, x0, c];
                        T v11 = input[b, y1, x1, c];

                        T interpolated = NumOps.Add(
                            NumOps.Add(
                                NumOps.Multiply(NumOps.Multiply(v00, wx0), wy0),
                                NumOps.Multiply(NumOps.Multiply(v01, wx1), wy0)
                            ),
                            NumOps.Add(
                                NumOps.Multiply(NumOps.Multiply(v10, wx0), wy1),
                                NumOps.Multiply(NumOps.Multiply(v11, wx1), wy1)
                            )
                        );

                        output[b, y, x, c] = interpolated;
                    }
                }
            }
        }

        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastTransformationMatrix == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int channels = _lastInput.Shape[3];

        // Initialize gradients
        var inputGradient = new Tensor<T>(_lastInput.Shape);
        _localizationWeights1Gradient = new Matrix<T>(_localizationWeights1.Rows, _localizationWeights1.Columns);
        _localizationBias1Gradient = new Vector<T>(_localizationBias1.Length);
        _localizationWeights2Gradient = new Matrix<T>(_localizationWeights2.Rows, _localizationWeights2.Columns);
        _localizationBias2Gradient = new Vector<T>(_localizationBias2.Length);

        // Generate output grid
        var outputGrid = GenerateOutputGrid();

        for (int b = 0; b < batchSize; b++)
        {
            // Backward pass through the sampler
            var samplerGradient = BackwardSampler(outputGradient.GetSlice(b), _lastInput.GetSlice(b), outputGrid, _lastTransformationMatrix);

            // Backward pass through the grid generator
            var gridGeneratorGradient = BackwardGridGenerator(samplerGradient, _lastTransformationMatrix);

            // Backward pass through the localization network
            BackwardLocalizationNetwork(gridGeneratorGradient, _lastInput.GetSlice(b));

            // Accumulate input gradient
            for (int y = 0; y < _inputHeight; y++)
            {
                for (int x = 0; x < _inputWidth; x++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        inputGradient[b, y, x, c] = samplerGradient[y, x, c];
                    }
                }
            }
        }

        return inputGradient;
    }

    private Tensor<T> BackwardSampler(Tensor<T> outputGradient, Tensor<T> input, Tensor<T> outputGrid, Matrix<T> transformationMatrix)
    {
        var samplerGradient = new Tensor<T>(input.Shape);

        for (int y = 0; y < _outputHeight; y++)
        {
            for (int x = 0; x < _outputWidth; x++)
            {
                // Apply transformation to output coordinates
                T srcX = NumOps.Add(
                    NumOps.Add(
                        NumOps.Multiply(transformationMatrix[0, 0], outputGrid[y, x, 0]),
                        NumOps.Multiply(transformationMatrix[0, 1], outputGrid[y, x, 1])
                    ),
                    transformationMatrix[0, 2]
                );
                T srcY = NumOps.Add(
                    NumOps.Add(
                        NumOps.Multiply(transformationMatrix[1, 0], outputGrid[y, x, 0]),
                        NumOps.Multiply(transformationMatrix[1, 1], outputGrid[y, x, 1])
                    ),
                    transformationMatrix[1, 2]
                );

                // Convert to input image coordinates
                srcX = NumOps.Multiply(NumOps.Add(srcX, NumOps.One), NumOps.Divide(NumOps.FromDouble(_inputWidth - 1), NumOps.FromDouble(2)));
                srcY = NumOps.Multiply(NumOps.Add(srcY, NumOps.One), NumOps.Divide(NumOps.FromDouble(_inputHeight - 1), NumOps.FromDouble(2)));

                // Compute the four nearest neighbor coordinates
                int x0 = (int)Math.Floor(Convert.ToDouble(srcX));
                int x1 = Math.Min(x0 + 1, _inputWidth - 1);
                int y0 = (int)Math.Floor(Convert.ToDouble(srcY));
                int y1 = Math.Min(y0 + 1, _inputHeight - 1);

                // Compute interpolation weights
                T wx1 = NumOps.Subtract(srcX, NumOps.FromDouble(x0));
                T wx0 = NumOps.Subtract(NumOps.One, wx1);
                T wy1 = NumOps.Subtract(srcY, NumOps.FromDouble(y0));
                T wy0 = NumOps.Subtract(NumOps.One, wy1);

                // Distribute gradients to the four nearest neighbors
                for (int c = 0; c < input.Shape[3]; c++)
                {
                    T gradValue = outputGradient[y, x, c];
                    samplerGradient[y0, x0, c] = NumOps.Add(samplerGradient[y0, x0, c], NumOps.Multiply(NumOps.Multiply(gradValue, wx0), wy0));
                    samplerGradient[y0, x1, c] = NumOps.Add(samplerGradient[y0, x1, c], NumOps.Multiply(NumOps.Multiply(gradValue, wx1), wy0));
                    samplerGradient[y1, x0, c] = NumOps.Add(samplerGradient[y1, x0, c], NumOps.Multiply(NumOps.Multiply(gradValue, wx0), wy1));
                    samplerGradient[y1, x1, c] = NumOps.Add(samplerGradient[y1, x1, c], NumOps.Multiply(NumOps.Multiply(gradValue, wx1), wy1));
                }
            }
        }

        return samplerGradient;
    }

    private Matrix<T> BackwardGridGenerator(Tensor<T> samplerGradient, Matrix<T> transformationMatrix)
    {
        var gridGeneratorGradient = new Matrix<T>(2, 3);

        for (int y = 0; y < _outputHeight; y++)
        {
            for (int x = 0; x < _outputWidth; x++)
            {
                T gradX = samplerGradient[y, x, 0];
                T gradY = samplerGradient[y, x, 1];

                gridGeneratorGradient[0, 0] = NumOps.Add(gridGeneratorGradient[0, 0], NumOps.Multiply(gradX, NumOps.FromDouble((double)x / (_outputWidth - 1) * 2 - 1)));
                gridGeneratorGradient[0, 1] = NumOps.Add(gridGeneratorGradient[0, 1], NumOps.Multiply(gradX, NumOps.FromDouble((double)y / (_outputHeight - 1) * 2 - 1)));
                gridGeneratorGradient[0, 2] = NumOps.Add(gridGeneratorGradient[0, 2], gradX);
                gridGeneratorGradient[1, 0] = NumOps.Add(gridGeneratorGradient[1, 0], NumOps.Multiply(gradY, NumOps.FromDouble((double)x / (_outputWidth - 1) * 2 - 1)));
                gridGeneratorGradient[1, 1] = NumOps.Add(gridGeneratorGradient[1, 1], NumOps.Multiply(gradY, NumOps.FromDouble((double)y / (_outputHeight - 1) * 2 - 1)));
                gridGeneratorGradient[1, 2] = NumOps.Add(gridGeneratorGradient[1, 2], gradY);
            }
        }

        return gridGeneratorGradient;
    }

    private void BackwardLocalizationNetwork(Matrix<T> gridGeneratorGradient, Tensor<T> input)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
    
        // Flatten input
        var flattenedInput = input.Reshape([input.Shape[0], _inputHeight * _inputWidth * input.Shape[3]]);

        // Backward pass through the second layer of localization network
        var dL_dTheta = gridGeneratorGradient.Reshape(gridGeneratorGradient.Rows, 6);
        var dL_dL1 = dL_dTheta.Multiply(_localizationWeights2.Transpose());

        _localizationWeights2Gradient ??= new Matrix<T>(_localizationWeights2.Rows, _localizationWeights2.Columns);
        _localizationWeights2Gradient = _localizationWeights2Gradient.Add(
            flattenedInput.ToMatrix().Transpose().Multiply(dL_dL1));

        _localizationBias2Gradient ??= new Vector<T>(_localizationBias2.Length);
        for (int i = 0; i < dL_dTheta.Rows; i++)
        {
            _localizationBias2Gradient = _localizationBias2Gradient.Add(dL_dTheta.GetRow(i));
        }

        // Backward pass through the activation function
        var z1 = flattenedInput.ToMatrix().Multiply(_localizationWeights1).AddColumn(_localizationBias1);
        var dL_dZ1 = ApplyActivationGradient(dL_dL1, z1);

        // Backward pass through the first layer of localization network
        _localizationWeights1Gradient ??= new Matrix<T>(_localizationWeights1.Rows, _localizationWeights1.Columns);
        _localizationWeights1Gradient = _localizationWeights1Gradient.Add(
            flattenedInput.Transpose([1, 0]).ToMatrix().Multiply(dL_dZ1));

        _localizationBias1Gradient ??= new Vector<T>(_localizationBias1.Length);
        for (int i = 0; i < dL_dZ1.Rows; i++)
        {
            _localizationBias1Gradient = _localizationBias1Gradient.Add(dL_dZ1.GetRow(i));
        }
    }

   private Matrix<T> ApplyActivationGradient(Matrix<T> upstream, Matrix<T> z)
    {
        var gradient = new Matrix<T>(z.Rows, z.Columns);

        for (int i = 0; i < z.Rows; i++)
        {
            Vector<T> rowZ = z.GetRow(i);
            Vector<T> rowUpstream = upstream.GetRow(i);
            Vector<T> rowGradient = ApplyActivationDerivative(rowZ, rowUpstream);
            gradient.SetRow(i, rowGradient);
        }

        return gradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_localizationWeights1Gradient == null || _localizationBias1Gradient == null ||
            _localizationWeights2Gradient == null || _localizationBias2Gradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _localizationWeights1 = _localizationWeights1.Subtract(_localizationWeights1Gradient.Multiply(learningRate));
        _localizationBias1 = _localizationBias1.Subtract(_localizationBias1Gradient.Multiply(learningRate));
        _localizationWeights2 = _localizationWeights2.Subtract(_localizationWeights2Gradient.Multiply(learningRate));
        _localizationBias2 = _localizationBias2.Subtract(_localizationBias2Gradient.Multiply(learningRate));
    }

    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _localizationWeights1.Rows * _localizationWeights1.Columns +
                          _localizationBias1.Length +
                          _localizationWeights2.Rows * _localizationWeights2.Columns +
                          _localizationBias2.Length;
    
        var parameters = new Vector<T>(totalParams);
        int index = 0;
    
        // Copy localization weights1
        for (int i = 0; i < _localizationWeights1.Rows; i++)
        {
            for (int j = 0; j < _localizationWeights1.Columns; j++)
            {
                parameters[index++] = _localizationWeights1[i, j];
            }
        }
    
        // Copy localization bias1
        for (int i = 0; i < _localizationBias1.Length; i++)
        {
            parameters[index++] = _localizationBias1[i];
        }
    
        // Copy localization weights2
        for (int i = 0; i < _localizationWeights2.Rows; i++)
        {
            for (int j = 0; j < _localizationWeights2.Columns; j++)
            {
                parameters[index++] = _localizationWeights2[i, j];
            }
        }
    
        // Copy localization bias2
        for (int i = 0; i < _localizationBias2.Length; i++)
        {
            parameters[index++] = _localizationBias2[i];
        }
    
        return parameters;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int totalParams = _localizationWeights1.Rows * _localizationWeights1.Columns +
                          _localizationBias1.Length +
                          _localizationWeights2.Rows * _localizationWeights2.Columns +
                          _localizationBias2.Length;
    
        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }
    
        int index = 0;
    
        // Set localization weights1
        for (int i = 0; i < _localizationWeights1.Rows; i++)
        {
            for (int j = 0; j < _localizationWeights1.Columns; j++)
            {
                _localizationWeights1[i, j] = parameters[index++];
            }
        }
    
        // Set localization bias1
        for (int i = 0; i < _localizationBias1.Length; i++)
        {
            _localizationBias1[i] = parameters[index++];
        }
    
        // Set localization weights2
        for (int i = 0; i < _localizationWeights2.Rows; i++)
        {
            for (int j = 0; j < _localizationWeights2.Columns; j++)
            {
                _localizationWeights2[i, j] = parameters[index++];
            }
        }
    
        // Set localization bias2
        for (int i = 0; i < _localizationBias2.Length; i++)
        {
            _localizationBias2[i] = parameters[index++];
        }
    }

    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastOutput = null;
        _lastTransformationMatrix = null;
        _localizationWeights1Gradient = null;
        _localizationBias1Gradient = null;
        _localizationWeights2Gradient = null;
        _localizationBias2Gradient = null;
    }
}