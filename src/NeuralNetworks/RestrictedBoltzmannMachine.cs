namespace AiDotNet.NeuralNetworks;

public class RestrictedBoltzmannMachine<T> : NeuralNetworkBase<T>
{
    private Vector<T> VisibleBiases { get; set; }
    private Vector<T> HiddenBiases { get; set; }
    private Matrix<T> Weights { get; set; }
    public int VisibleSize { get; private set; }
    public int HiddenSize { get; private set; }

    private IActivationFunction<T>? ScalarActivation { get; }
    private IVectorActivationFunction<T>? VectorActivation { get; }

    public RestrictedBoltzmannMachine(NeuralNetworkArchitecture<T> architecture, int visibleSize, int hiddenSize, IActivationFunction<T>? scalarActivation = null) : 
        base(architecture)
    {
        VisibleSize = visibleSize;
        HiddenSize = hiddenSize;
        Weights = Matrix<T>.CreateRandom(hiddenSize, visibleSize);
        VisibleBiases = Vector<T>.CreateDefault(visibleSize, NumOps.Zero);
        HiddenBiases = Vector<T>.CreateDefault(hiddenSize, NumOps.Zero);
        ScalarActivation = scalarActivation;
        VectorActivation = null;
    }

    public RestrictedBoltzmannMachine(NeuralNetworkArchitecture<T> architecture, int visibleSize, int hiddenSize, IVectorActivationFunction<T>? vectorActivation = null) : 
        base(architecture)
    {
        VisibleSize = visibleSize;
        HiddenSize = hiddenSize;
        Weights = Matrix<T>.CreateRandom(hiddenSize, visibleSize);
        VisibleBiases = Vector<T>.CreateDefault(visibleSize, NumOps.Zero);
        HiddenBiases = Vector<T>.CreateDefault(hiddenSize, NumOps.Zero);
        ScalarActivation = null;
        VectorActivation = vectorActivation;
    }

    public RestrictedBoltzmannMachine(NeuralNetworkArchitecture<T> architecture) : base(architecture)
    {
        // Get the layer sizes
        int[] layerSizes = architecture.GetLayerSizes();

        // Check if we have exactly two layers (visible and hidden)
        if (layerSizes.Length != 2)
        {
            throw new ArgumentException("RBM requires exactly two layers (visible and hidden units).");
        }

        VisibleSize = layerSizes[0];
        HiddenSize = layerSizes[1];

        if (VisibleSize <= 0 || HiddenSize <= 0)
        {
            throw new ArgumentException("Both visible and hidden unit counts must be positive for RBM.");
        }

        VisibleBiases = new Vector<T>(VisibleSize);
        HiddenBiases = new Vector<T>(HiddenSize);
        Weights = new Matrix<T>(HiddenSize, VisibleSize);

        InitializeParameters();
    }

    protected override void InitializeLayers()
    {
        // RBM doesn't use layers in the same way as feedforward networks
        // Instead, we'll initialize the weights and biases directly
    }

    private void InitializeParameters()
    {
        // Initialize biases to zero and weights to small random values
        for (int i = 0; i < VisibleSize; i++)
        {
            VisibleBiases[i] = NumOps.Zero;
            for (int j = 0; j < HiddenSize; j++)
            {
                Weights[i, j] = NumOps.FromDouble(Random.NextDouble() * 0.1 - 0.05);
            }
        }

        for (int j = 0; j < HiddenSize; j++)
        {
            HiddenBiases[j] = NumOps.Zero;
        }
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        // In an RBM, prediction is typically done by reconstructing the visible layer
        Vector<T> hiddenProbs = SampleHiddenGivenVisible(input);
        return SampleVisibleGivenHidden(hiddenProbs);
    }

    public Tensor<T> GetHiddenLayerActivation(Tensor<T> visibleLayer)
    {
        var hiddenActivations = Weights.Multiply(visibleLayer.ToMatrix()).Add(HiddenBiases.ToColumnMatrix());
            
        if (VectorActivation != null)
        {
            return VectorActivation.Activate(Tensor<T>.FromMatrix(hiddenActivations));
        }
        else if (ScalarActivation != null)
        {
            return Tensor<T>.FromMatrix(hiddenActivations.Transform((x, _, _) => ScalarActivation.Activate(x)));
        }
        else
        {
            throw new InvalidOperationException("No activation function specified.");
        }
    }

    private Vector<T> SampleHiddenGivenVisible(Vector<T> visible)
    {
        Vector<T> hiddenProbs = new Vector<T>(HiddenSize);
        for (int j = 0; j < HiddenSize; j++)
        {
            T activation = HiddenBiases[j];
            for (int i = 0; i < VisibleSize; i++)
            {
                activation = NumOps.Add(activation, NumOps.Multiply(Weights[i, j], visible[i]));
            }

            hiddenProbs[j] = new SigmoidActivation<T>().Activate(activation);
        }

        return hiddenProbs;
    }

    private Tensor<T> GetVisibleLayerReconstruction(Tensor<T> hiddenLayer)
    {
        var visibleActivations = Weights.Transpose().Multiply(hiddenLayer.ToMatrix()).Add(VisibleBiases.ToColumnMatrix());
            
        if (VectorActivation != null)
        {
            return VectorActivation.Activate(Tensor<T>.FromMatrix(visibleActivations));
        }
        else if (ScalarActivation != null)
        {
            return Tensor<T>.FromMatrix(visibleActivations.Transform((x, _, _) => ScalarActivation.Activate(x)));
        }
        else
        {
            throw new InvalidOperationException("No activation function specified.");
        }
    }

    private Vector<T> SampleVisibleGivenHidden(Vector<T> hidden)
    {
        Vector<T> visibleProbs = new Vector<T>(VisibleSize);
        for (int i = 0; i < VisibleSize; i++)
        {
            T activation = VisibleBiases[i];
            for (int j = 0; j < HiddenSize; j++)
            {
                activation = NumOps.Add(activation, NumOps.Multiply(Weights[i, j], hidden[j]));
            }

            visibleProbs[i] = new SigmoidActivation<T>().Activate(activation);
        }

        return visibleProbs;
    }

    public void Train(Tensor<T> data, int epochs, T learningRate)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // Positive phase
            Tensor<T> visibleLayer = data;
            Tensor<T> hiddenLayer = GetHiddenLayerActivation(visibleLayer);
            Matrix<T> posGradient = TensorOuterProduct(visibleLayer, hiddenLayer);

            // Negative phase
            Tensor<T> visibleReconstruction = GetVisibleLayerReconstruction(hiddenLayer);
            Tensor<T> hiddenReconstruction = GetHiddenLayerActivation(visibleReconstruction);
            Matrix<T> negGradient = TensorOuterProduct(visibleReconstruction, hiddenReconstruction);

            // Update weights and biases
            Weights = Weights.Add(posGradient.Subtract(negGradient).Multiply(learningRate));
    
            // Update visible biases
            Vector<T> visibleBiasGradient = TensorToVector(visibleLayer.Subtract(visibleReconstruction));
            T visibleBiasMean = NumOps.Divide(visibleBiasGradient.Sum(), NumOps.FromDouble(visibleBiasGradient.Length));
            VisibleBiases = VisibleBiases.Add(Vector<T>.CreateDefault(VisibleSize, NumOps.Multiply(visibleBiasMean, learningRate)));

            // Update hidden biases
            Vector<T> hiddenBiasGradient = TensorToVector(hiddenLayer.Subtract(hiddenReconstruction));
            T hiddenBiasMean = NumOps.Divide(hiddenBiasGradient.Sum(), NumOps.FromDouble(hiddenBiasGradient.Length));
            HiddenBiases = HiddenBiases.Add(Vector<T>.CreateDefault(HiddenSize, NumOps.Multiply(hiddenBiasMean, learningRate)));
        }
    }

    private Matrix<T> TensorOuterProduct(Tensor<T> t1, Tensor<T> t2)
    {
        Vector<T> v1 = TensorToVector(t1);
        Vector<T> v2 = TensorToVector(t2);

        return OuterProduct(v1, v2);
    }

    private Vector<T> TensorToVector(Tensor<T> tensor)
    {
        return tensor.ToMatrix().Flatten();
    }

    private Matrix<T> OuterProduct(Vector<T> v1, Vector<T> v2)
    {
        Matrix<T> result = new Matrix<T>(v1.Length, v2.Length);
        for (int i = 0; i < v1.Length; i++)
        {
            for (int j = 0; j < v2.Length; j++)
            {
                result[i, j] = NumOps.Multiply(v1[i], v2[j]);
            }
        }

        return result;
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        // This method is not typically used in RBMs
        throw new NotImplementedException("UpdateParameters is not implemented for Restricted Boltzmann Machines.");
    }

    public override void Serialize(BinaryWriter writer)
    {
        if (writer == null)
            throw new ArgumentNullException(nameof(writer));

        writer.Write(VisibleSize);
        writer.Write(HiddenSize);

        for (int i = 0; i < VisibleSize; i++)
        {
            writer.Write(Convert.ToDouble(VisibleBiases[i]));
        }

        for (int j = 0; j < HiddenSize; j++)
        {
            writer.Write(Convert.ToDouble(HiddenBiases[j]));
        }

        for (int i = 0; i < VisibleSize; i++)
        {
            for (int j = 0; j < HiddenSize; j++)
            {
                writer.Write(Convert.ToDouble(Weights[i, j]));
            }
        }
    }

    public override void Deserialize(BinaryReader reader)
    {
        if (reader == null)
            throw new ArgumentNullException(nameof(reader));

        VisibleSize = reader.ReadInt32();
        HiddenSize = reader.ReadInt32();

        VisibleBiases = new Vector<T>(VisibleSize);
        HiddenBiases = new Vector<T>(HiddenSize);
        Weights = new Matrix<T>(VisibleSize, HiddenSize);

        for (int i = 0; i < VisibleSize; i++)
        {
            VisibleBiases[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        for (int j = 0; j < HiddenSize; j++)
        {
            HiddenBiases[j] = NumOps.FromDouble(reader.ReadDouble());
        }

        for (int i = 0; i < VisibleSize; i++)
        {
            for (int j = 0; j < HiddenSize; j++)
            {
                Weights[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }
    }
}