namespace AiDotNet.NeuralNetworks;

public class HopfieldNetwork<T> : NeuralNetworkBase<T>
{
    private Matrix<T> _weights;
    private int _size;
    private readonly IActivationFunction<T> _activationFunction;

    public HopfieldNetwork(NeuralNetworkArchitecture<T> architecture, int size) : base(new NeuralNetworkArchitecture<T>(
        architecture.InputType,
        taskType: architecture.TaskType,
        complexity: architecture.Complexity,
        inputSize: size,
        outputSize: size))
    {
        _size = size;
        _weights = new Matrix<T>(size, size);
        _activationFunction = new SignActivation<T>();

        InitializeWeights();
        InitializeLayers();
    }

    private void InitializeWeights()
    {
        for (int i = 0; i < _size; i++)
        {
            for (int j = 0; j < _size; j++)
            {
                if (i != j)
                {
                    _weights[i, j] = NumOps.Zero;
                }
                else
                {
                    _weights[i, j] = NumOps.Zero;
                }
            }
        }
    }

    public void Train(List<Vector<T>> patterns)
    {
        foreach (var pattern in patterns)
        {
            for (int i = 0; i < _size; i++)
            {
                for (int j = 0; j < _size; j++)
                {
                    if (i != j)
                    {
                        _weights[i, j] = NumOps.Add(_weights[i, j], NumOps.Multiply(pattern[i], pattern[j]));
                    }
                }
            }
        }

        // Normalize weights
        for (int i = 0; i < _size; i++)
        {
            for (int j = 0; j < _size; j++)
            {
                _weights[i, j] = NumOps.Divide(_weights[i, j], NumOps.FromDouble(patterns.Count));
            }
        }
    }

    public Vector<T> Recall(Vector<T> input, int maxIterations = 100)
    {
        var current = input;
        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            var next = new Vector<T>(_size);
            for (int i = 0; i < _size; i++)
            {
                T sum = NumOps.Zero;
                for (int j = 0; j < _size; j++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(_weights[i, j], current[j]));
                }
                next[i] = _activationFunction.Activate(sum);
            }

            if (current.Equals(next))
            {
                break;
            }
            current = next;
        }

        return current;
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        return Recall(input);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        // Hopfield networks typically don't use gradient-based updates
        throw new NotImplementedException("Hopfield networks do not support gradient-based parameter updates.");
    }

    public override void Serialize(BinaryWriter writer)
    {
        if (writer == null)
            throw new ArgumentNullException(nameof(writer));

        writer.Write(_size);
        for (int i = 0; i < _size; i++)
        {
            for (int j = 0; j < _size; j++)
            {
                writer.Write(Convert.ToDouble(_weights[i, j]));
            }
        }
    }

    public override void Deserialize(BinaryReader reader)
    {
        if (reader == null)
            throw new ArgumentNullException(nameof(reader));

        _size = reader.ReadInt32();
        _weights = new Matrix<T>(_size, _size);
        for (int i = 0; i < _size; i++)
        {
            for (int j = 0; j < _size; j++)
            {
                _weights[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }
    }

    protected override void InitializeLayers()
    {
        // Hopfield networks don't use layers in the same way as feedforward networks
    }
}