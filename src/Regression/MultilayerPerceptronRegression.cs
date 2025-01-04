namespace AiDotNet.Regression;

public class MultilayerPerceptronRegression<T> : NonLinearRegressionBase<T>
{
    private readonly MultilayerPerceptronOptions<T> _options;
    private readonly List<Matrix<T>> _weights;
    private readonly List<Vector<T>> _biases;
    private IOptimizer<T> _optimizer;

    public MultilayerPerceptronRegression(MultilayerPerceptronOptions<T>? options = null, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new MultilayerPerceptronOptions<T>();
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
            T scaleFactor = NumOps.Sqrt(NumOps.FromDouble(2.0 / (inputSize + outputSize)));
            weight = weight.Transform((w, _, _) => NumOps.Multiply(w, scaleFactor));

            _weights.Add(weight);
            _biases.Add(bias);
        }
    }

    public override void Train(Matrix<T> X, Vector<T> y)
    {
        int numSamples = X.Rows;
        int numFeatures = X.Columns;

        if (_options.LayerSizes[0] != numFeatures)
        {
            throw new ArgumentException("Input feature size does not match the first layer size.");
        }

        for (int epoch = 0; epoch < _options.MaxEpochs; epoch++)
        {
            T totalLoss = NumOps.Zero;

            // Mini-batch gradient descent
            for (int i = 0; i < numSamples; i += _options.BatchSize)
            {
                int batchSize = Math.Min(_options.BatchSize, numSamples - i);
                Matrix<T> batchX = X.GetRowRange(i, batchSize);
                Vector<T> batchY = y.GetSubVector(i, batchSize);

                (T batchLoss, List<Matrix<T>> weightGradients, List<Vector<T>> biasGradients) = ComputeGradients(batchX, batchY);
                UpdateParameters(weightGradients, biasGradients, batchSize);

                totalLoss = NumOps.Add(totalLoss, batchLoss);
            }

            T averageLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(numSamples));

            if (_options.Verbose && epoch % 100 == 0)
            {
                Console.WriteLine($"Epoch {epoch}, Average Loss: {averageLoss}");
            }

            if (NumOps.LessThan(averageLoss, NumOps.FromDouble(_options.Tolerance)))
            {
                if (_options.Verbose)
                {
                    Console.WriteLine($"Converged at epoch {epoch}");
                }
                break;
            }
        }
    }

    private (T loss, List<Matrix<T>> weightGradients, List<Vector<T>> biasGradients) ComputeGradients(Matrix<T> X, Vector<T> y)
    {
        int numLayers = _weights.Count;
        List<Vector<T>> activations = new List<Vector<T>>(numLayers + 1);
        List<Vector<T>> zs = new List<Vector<T>>(numLayers);

        // Forward pass
        activations.Add(X.Transpose().GetColumn(0));  // Input layer
        for (int i = 0; i < numLayers; i++)
        {
            Vector<T> z = _weights[i].Multiply(activations[i]).Add(_biases[i]);
            zs.Add(z);
            activations.Add(ApplyActivation(z, i == numLayers - 1));
        }

        // Compute loss
        T loss = ComputeLoss(activations[activations.Count - 1], y);

        // Backward pass
        List<Matrix<T>> weightGradients = new(numLayers);
        List<Vector<T>> biasGradients = new(numLayers);

        Vector<T> delta = ComputeOutputLayerDelta(activations[activations.Count - 1], y, zs[zs.Count - 1]);

        for (int i = numLayers - 1; i >= 0; i--)
        {
            Matrix<T> weightGradient = delta.OuterProduct(activations[i]);
            weightGradients.Insert(0, weightGradient);
            biasGradients.Insert(0, delta);

            if (i > 0)
            {
                delta = _weights[i].Transpose().Multiply(delta).Transform((d, index) => 
                    NumOps.Multiply(d, ApplyActivationDerivative(zs[i - 1], false)[index]));
            }
        }

        return (loss, weightGradients, biasGradients);
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
        Vector<T> predictions = new Vector<T>(X.Rows, NumOps);

        for (int i = 0; i < X.Rows; i++)
        {
            Vector<T> input = X.GetRow(i);
            Vector<T> output = ForwardPass(input);
            predictions[i] = output[0];
        }

        return predictions;
    }

    private Vector<T> ForwardPass(Vector<T> input)
    {
        Vector<T> activation = input;
        for (int i = 0; i < _weights.Count; i++)
        {
            activation = ApplyActivation(_weights[i].Multiply(activation).Add(_biases[i]), i == _weights.Count - 1);
        }

        return activation;
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

    private T ComputeLoss(Vector<T> predictions, Vector<T> targets)
    {
        T sumSquaredErrors = predictions.Subtract(targets).Transform(x => NumOps.Multiply(x, x)).Sum();
        return NumOps.Divide(sumSquaredErrors, NumOps.FromDouble(predictions.Length));
    }

    private Vector<T> ComputeOutputLayerDelta(Vector<T> predictions, Vector<T> targets, Vector<T> outputLayerZ)
    {
        Vector<T> error = predictions.Subtract(targets);
        Vector<T> activationDerivative = ApplyActivationDerivative(outputLayerZ, true);

        return error.Transform((e, i) => NumOps.Multiply(e, activationDerivative[i]));
    }

    protected override ModelType GetModelType() => ModelType.MultilayerPerceptronRegression;

    protected override void OptimizeModel(Matrix<T> x, Vector<T> y)
    {
        Train(x, y);
    }

    public override byte[] Serialize()
    {
        using MemoryStream ms = new();
        using BinaryWriter writer = new(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize MultilayerPerceptronRegression specific data
        writer.Write(_options.LayerSizes.Count);
        foreach (var size in _options.LayerSizes)
        {
            writer.Write(size);
        }
        writer.Write(_options.MaxEpochs);
        writer.Write(_options.BatchSize);
        writer.Write(Convert.ToDouble(_options.LearningRate));
        writer.Write(Convert.ToDouble(_options.Tolerance));
        writer.Write(_options.Verbose);

        // Serialize weights and biases
        writer.Write(_weights.Count);
        foreach (var weight in _weights)
        {
            byte[] weightData = weight.Serialize();
            writer.Write(weightData.Length);
            writer.Write(weightData);
        }

        writer.Write(_biases.Count);
        foreach (var bias in _biases)
        {
            byte[] biasData = bias.Serialize();
            writer.Write(biasData.Length);
            writer.Write(biasData);
        }

        // Serialize optimizer
        writer.Write((int)OptimizerFactory.GetOptimizerType(_optimizer));
        byte[] optimizerData = _optimizer.Serialize();
        writer.Write(optimizerData.Length);
        writer.Write(optimizerData);

        // Serialize optimizer options
        string optionsJson = JsonConvert.SerializeObject(_optimizer.GetOptions());
        writer.Write(optionsJson);

        return ms.ToArray();
    }

    public override void Deserialize(byte[] data)
    {
        using MemoryStream ms = new(data);
        using BinaryReader reader = new(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize MultilayerPerceptronRegression specific data
        int layerCount = reader.ReadInt32();
        _options.LayerSizes = [];
        for (int i = 0; i < layerCount; i++)
        {
            _options.LayerSizes.Add(reader.ReadInt32());
        }
        _options.MaxEpochs = reader.ReadInt32();
        _options.BatchSize = reader.ReadInt32();
        _options.LearningRate = reader.ReadDouble();
        _options.Tolerance = reader.ReadDouble();
        _options.Verbose = reader.ReadBoolean();

        // Deserialize weights and biases
        int weightCount = reader.ReadInt32();
        _weights.Clear();
        for (int i = 0; i < weightCount; i++)
        {
            int weightDataLength = reader.ReadInt32();
            byte[] weightData = reader.ReadBytes(weightDataLength);
            _weights.Add(Matrix<T>.Deserialize(weightData));
        }

        int biasCount = reader.ReadInt32();
        _biases.Clear();
        for (int i = 0; i < biasCount; i++)
        {
            int biasDataLength = reader.ReadInt32();
            byte[] biasData = reader.ReadBytes(biasDataLength);
            _biases.Add(Vector<T>.Deserialize(biasData));
        }

        // Deserialize optimizer
        OptimizerType optimizerType = (OptimizerType)reader.ReadInt32();
        int optimizerDataLength = reader.ReadInt32();
        byte[] optimizerData = reader.ReadBytes(optimizerDataLength);

        // Deserialize optimizer options
        string optionsJson = reader.ReadString();
        var options = JsonConvert.DeserializeObject<OptimizationAlgorithmOptions>(optionsJson);

        if (options == null)
        {
            throw new InvalidOperationException("Failed to deserialize optimizer options.");
        }

        // Create optimizer using factory
        _optimizer = OptimizerFactory.CreateOptimizer<T>(optimizerType, options);
        _optimizer.Deserialize(optimizerData);
    }
}