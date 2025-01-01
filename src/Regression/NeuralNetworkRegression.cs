namespace AiDotNet.Regression;

public class NeuralNetworkRegression<T> : NonLinearRegressionBase<T>
{
    private readonly NeuralNetworkRegressionOptions<T> _options;
    private readonly List<Matrix<T>> _weights;
    private readonly List<Vector<T>> _biases;
    private readonly IOptimizer<T> _optimizer;

    public NeuralNetworkRegression(NeuralNetworkRegressionOptions<T>? options = null, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new NeuralNetworkRegressionOptions<T>();
        _optimizer = _options.Optimizer ?? new AdamOptimizer<T>();
        _weights = [];
        _biases = [];
        InitializeNetwork();
    }

    private void InitializeNetwork()
    {
        for (int i = 0; i < _options.LayerSizes.Count - 1; i++)
        {
            int inputSize = _options.LayerSizes[i];
            int outputSize = _options.LayerSizes[i + 1];

            Matrix<T> weight = Matrix<T>.CreateRandom(outputSize, inputSize, NumOps);
            Vector<T> bias = Vector<T>.CreateRandom(outputSize);

            // Xavier/Glorot initialization
            T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (inputSize + outputSize)));
            weight = weight.Transform((w, row, col) => NumOps.Multiply(w, scale));

            _weights.Add(weight);
            _biases.Add(bias);
        }
    }

    public override void Train(Matrix<T> X, Vector<T> y)
    {
        int batchSize = _options.BatchSize;
        int numBatches = (X.Rows + batchSize - 1) / batchSize;

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            // Shuffle the data
            int[] indices = Enumerable.Range(0, X.Rows).ToArray();
            ShuffleArray(indices);

            T totalLoss = NumOps.Zero;

            for (int batch = 0; batch < numBatches; batch++)
            {
                int startIdx = batch * batchSize;
                int endIdx = Math.Min(startIdx + batchSize, X.Rows);

                Matrix<T> batchX = GetBatchRows(X, indices, startIdx, endIdx);
                Vector<T> batchY = GetBatchElements(y, indices, startIdx, endIdx);

                T batchLoss = TrainOnBatch(batchX, batchY);
                totalLoss = NumOps.Add(totalLoss, batchLoss);
            }

            if (epoch % 100 == 0)
            {
                Console.WriteLine($"Epoch {epoch}, Loss: {totalLoss}");
            }
        }
    }

    private void ShuffleArray(int[] array)
    {
        Random random = new();
        int n = array.Length;
        for (int i = n - 1; i > 0; i--)
        {
            int j = random.Next(0, i + 1);
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }

    private Matrix<T> GetBatchRows(Matrix<T> matrix, int[] indices, int startIdx, int endIdx)
    {
        int batchSize = endIdx - startIdx;
        Matrix<T> result = new Matrix<T>(batchSize, matrix.Columns, NumOps);
        for (int i = 0; i < batchSize; i++)
        {
            result.SetRow(i, matrix.GetRow(indices[startIdx + i]));
        }

        return result;
    }

    private Vector<T> GetBatchElements(Vector<T> vector, int[] indices, int startIdx, int endIdx)
    {
        int batchSize = endIdx - startIdx;
        Vector<T> result = new(batchSize, NumOps);
        for (int i = 0; i < batchSize; i++)
        {
            result[i] = vector[indices[startIdx + i]];
        }

        return result;
    }

    private T TrainOnBatch(Matrix<T> X, Vector<T> y)
    {
        List<Matrix<T>> weightGradients = [];
        List<Vector<T>> biasGradients = [];

        T batchLoss = NumOps.Zero;

        for (int i = 0; i < X.Rows; i++)
        {
            Vector<T> input = X.GetRow(i);
            Vector<T> target = new([y[i]], NumOps);

            // Forward pass
            List<Vector<T>> activations = ForwardPass(input);

            // Compute loss
            T loss = _options.LossFunction(activations[activations.Count - 1], target);
            batchLoss = NumOps.Add(batchLoss, loss);

            // Backward pass
            List<Vector<T>> deltas = BackwardPass(activations, target);

            // Accumulate gradients
            AccumulateGradients(activations, deltas, weightGradients, biasGradients);
        }

        // Update parameters
        UpdateParameters(weightGradients, biasGradients, X.Rows);

        return NumOps.Divide(batchLoss, NumOps.FromDouble(X.Rows));
    }

    private List<Vector<T>> ForwardPass(Vector<T> input)
    {
        List<Vector<T>> activations = new List<Vector<T>> { input };

        for (int i = 0; i < _weights.Count; i++)
        {
            Vector<T> z = _weights[i].Multiply(activations[i]).Add(_biases[i]);
            Vector<T> a = ApplyActivation(z, i == _weights.Count - 1);
            activations.Add(a);
        }

        return activations;
    }

    private List<Vector<T>> BackwardPass(List<Vector<T>> activations, Vector<T> target)
    {
        List<Vector<T>> deltas = new List<Vector<T>>();

        // Output layer error
        int lastIndex = activations.Count - 1;
        Vector<T> error = _options.LossFunctionDerivative(activations[lastIndex], target);
        Vector<T> delta = error.PointwiseMultiply(ApplyActivationDerivative(activations[lastIndex], true));
        deltas.Add(delta);

        // Hidden layers
        for (int i = _weights.Count - 1; i > 0; i--)
        {
            delta = _weights[i].Transpose().Multiply(delta).PointwiseMultiply(ApplyActivationDerivative(activations[i], false));
            deltas.Insert(0, delta);
        }

        return deltas;
    }

    private void AccumulateGradients(List<Vector<T>> activations, List<Vector<T>> deltas, 
                                     List<Matrix<T>> weightGradients, List<Vector<T>> biasGradients)
    {
        for (int i = 0; i < _weights.Count; i++)
        {
            Matrix<T> weightGradient = deltas[i].OuterProduct(activations[i]);
            Vector<T> biasGradient = deltas[i];

            if (weightGradients.Count <= i)
            {
                weightGradients.Add(weightGradient);
                biasGradients.Add(biasGradient);
            }
            else
            {
                weightGradients[i] = weightGradients[i].Add(weightGradient);
                biasGradients[i] = biasGradients[i].Add(biasGradient);
            }
        }
    }

    private void UpdateParameters(List<Matrix<T>> weightGradients, List<Vector<T>> biasGradients, int batchSize)
    {
        T scaleFactor = NumOps.FromDouble(1.0 / batchSize);

        for (int i = 0; i < _weights.Count; i++)
        {
            Matrix<T> avgWeightGradient = weightGradients[i].Transform((g, _, _) => NumOps.Multiply(g, scaleFactor));
            Vector<T> avgBiasGradient = biasGradients[i].Transform(g => NumOps.Multiply(g, scaleFactor));

            if (_optimizer is IGradientBasedOptimizer<T> gradientOptimizer)
            {
                _weights[i] = gradientOptimizer.UpdateMatrix(_weights[i], avgWeightGradient);
                _biases[i] = gradientOptimizer.UpdateVector(_biases[i], avgBiasGradient);
            }
            else
            {
                // For non-gradient-based optimizers, we'll use a simple update rule
                _weights[i] = _weights[i].Subtract(avgWeightGradient.Multiply(NumOps.FromDouble(_options.LearningRate)));
                _biases[i] = _biases[i].Subtract(avgBiasGradient.Multiply(NumOps.FromDouble(_options.LearningRate)));
            }

            // Apply regularization
            _weights[i] = Regularization.RegularizeMatrix(_weights[i]);
            _biases[i] = Regularization.RegularizeCoefficients(_biases[i]);
        }
    }

    public override Vector<T> Predict(Matrix<T> X)
    {
        Vector<T> predictions = new(X.Rows, NumOps);

        for (int i = 0; i < X.Rows; i++)
        {
            Vector<T> input = X.GetRow(i);
            List<Vector<T>> activations = ForwardPass(input);
            predictions[i] = activations[activations.Count - 1][0];
        }

        return predictions;
    }

    private Vector<T> ApplyActivation(Vector<T> input, bool isOutputLayer)
    {
        var activationFunc = isOutputLayer ? _options.OutputActivationFunction : _options.HiddenActivationFunction;
        return input.Transform(x => activationFunc(x));
    }

    private Vector<T> ApplyActivationDerivative(Vector<T> input, bool isOutputLayer)
    {
        var activationFuncDerivative = isOutputLayer ? _options.OutputActivationFunctionDerivative : _options.HiddenActivationFunctionDerivative;
        return input.Transform(x => activationFuncDerivative(x));
    }

    protected override ModelType GetModelType() => ModelType.NeuralNetworkRegression;

    protected override void OptimizeModel(Matrix<T> x, Vector<T> y)
    {
        Train(x, y);
    }

    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize NeuralNetworkRegression specific data
        writer.Write(_options.LayerSizes.Count);
        foreach (var size in _options.LayerSizes)
        {
            writer.Write(size);
        }

        writer.Write(_weights.Count);
        foreach (var weight in _weights)
        {
            writer.Write(weight.Rows);
            writer.Write(weight.Columns);
            foreach (var value in weight.Flatten())
            {
                writer.Write(Convert.ToDouble(value));
            }
        }

        writer.Write(_biases.Count);
        foreach (var bias in _biases)
        {
            writer.Write(bias.Length);
            foreach (var value in bias)
            {
                writer.Write(Convert.ToDouble(value));
            }
        }

        return ms.ToArray();
    }

    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize NeuralNetworkRegression specific data
        int layerCount = reader.ReadInt32();
        _options.LayerSizes = new List<int>();
        for (int i = 0; i < layerCount; i++)
        {
            _options.LayerSizes.Add(reader.ReadInt32());
        }

        int weightCount = reader.ReadInt32();
        _weights.Clear();
        for (int i = 0; i < weightCount; i++)
        {
            int rows = reader.ReadInt32();
            int cols = reader.ReadInt32();
            var weightData = new T[rows, cols];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    weightData[r, c] = NumOps.FromDouble(reader.ReadDouble());
                }
            }
            _weights.Add(new Matrix<T>(weightData, NumOps));
        }

        int biasCount = reader.ReadInt32();
        _biases.Clear();
        for (int i = 0; i < biasCount; i++)
        {
            int length = reader.ReadInt32();
            var biasData = new T[length];
            for (int j = 0; j < length; j++)
            {
                biasData[j] = NumOps.FromDouble(reader.ReadDouble());
            }
            _biases.Add(new Vector<T>(biasData, NumOps));
        }

        InitializeNetwork();
    }
}