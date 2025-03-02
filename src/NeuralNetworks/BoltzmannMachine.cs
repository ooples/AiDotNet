namespace AiDotNet.NeuralNetworks;

public class BoltzmannMachine<T> : NeuralNetworkBase<T>
{
    private Vector<T> Biases { get; set; }
    private Matrix<T> Weights { get; set; }

    public BoltzmannMachine(NeuralNetworkArchitecture<T> architecture) : base(architecture)
    {
        if (architecture.InputType != InputType.OneDimensional)
        {
            throw new ArgumentException("Boltzmann Machine requires one-dimensional input.");
        }

        int size = architecture.CalculatedInputSize;
        if (size <= 0)
        {
            throw new ArgumentException("Invalid input size for Boltzmann Machine.");
        }

        Biases = new Vector<T>(size);
        Weights = new Matrix<T>(size, size);

        // Check if custom layers are provided (which is not typical for Boltzmann Machines)
        if (architecture.Layers != null && architecture.Layers.Count > 0)
        {
            throw new ArgumentException("Boltzmann Machine does not support custom layers.");
        }

        InitializeParameters(size);
    }

    protected override void InitializeLayers()
    {
        // Boltzmann Machine doesn't use layers in the same way as feedforward networks
        // Instead, we'll initialize the weights and biases directly
    }

    private void InitializeParameters(int size)
    {
        // Initialize biases to zero and weights to small random values
        for (int i = 0; i < size; i++)
        {
            Biases[i] = NumOps.Zero;
            for (int j = 0; j < size; j++)
            {
                Weights[i, j] = NumOps.FromDouble(Random.NextDouble() * 0.1 - 0.05);
            }
        }
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        // In a Boltzmann Machine, prediction is typically done through sampling
        return Sample(input);
    }

    private Vector<T> Sample(Vector<T> state)
    {
        Vector<T> newState = new Vector<T>(state.Length);
        for (int i = 0; i < state.Length; i++)
        {
            T activation = Biases[i];
            for (int j = 0; j < state.Length; j++)
            {
                activation = NumOps.Add(activation, NumOps.Multiply(Weights[i, j], state[j]));
            }

            T probability = new SigmoidActivation<T>().Activate(activation);
            newState[i] = NumOps.FromDouble(Random.NextDouble() < Convert.ToDouble(probability) ? 1 : 0);
        }

        return newState;
    }

    public void Train(Vector<T> data, int epochs, T learningRate)
    {
        int size = Architecture.CalculatedInputSize;
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            Vector<T> visibleProbs = data;
            Vector<T> hiddenProbs = Sample(visibleProbs);
            Matrix<T> posGradient = OuterProduct(visibleProbs, hiddenProbs);

            Vector<T> visibleReconstruction = Sample(hiddenProbs);
            Vector<T> hiddenReconstruction = Sample(visibleReconstruction);
            Matrix<T> negGradient = OuterProduct(visibleReconstruction, hiddenReconstruction);

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    T weightUpdate = NumOps.Multiply(learningRate, NumOps.Subtract(posGradient[i, j], negGradient[i, j]));
                    Weights[i, j] = NumOps.Add(Weights[i, j], weightUpdate);
                }

                T biasUpdate = NumOps.Multiply(learningRate, NumOps.Subtract(visibleProbs[i], visibleReconstruction[i]));
                Biases[i] = NumOps.Add(Biases[i], biasUpdate);
            }
        }
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
        // This method is not typically used in Boltzmann Machines
        throw new NotImplementedException("UpdateParameters is not implemented for Boltzmann Machines.");
    }

    public override void Serialize(BinaryWriter writer)
    {
        if (writer == null)
            throw new ArgumentNullException(nameof(writer));

        writer.Write(Biases.Length);
        for (int i = 0; i < Biases.Length; i++)
        {
            writer.Write(Convert.ToDouble(Biases[i]));
        }

        for (int i = 0; i < Weights.Rows; i++)
        {
            for (int j = 0; j < Weights.Columns; j++)
            {
                writer.Write(Convert.ToDouble(Weights[i, j]));
            }
        }
    }

    public override void Deserialize(BinaryReader reader)
    {
        if (reader == null)
            throw new ArgumentNullException(nameof(reader));

        int size = reader.ReadInt32();
        Biases = new Vector<T>(size);
        Weights = new Matrix<T>(size, size);

        for (int i = 0; i < size; i++)
        {
            Biases[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                Weights[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }
    }
}