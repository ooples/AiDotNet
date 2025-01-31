namespace AiDotNet.NeuralNetworks;

public class Layer<T>
{
    private Matrix<T> _weights;
    private Vector<T> _biases;
    private IActivationFunction<T>? _activationFunction;
    private IVectorActivationFunction<T>? _vectorActivationFunction;
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public int ParameterCount => _weights.Rows * _weights.Columns + _biases.Length;
    public int InputSize => _weights.Columns;
    public int OutputSize => _weights.Rows;
    public bool IsVectorActivation => _vectorActivationFunction != null;

    public Layer(int inputSize, int outputSize, IActivationFunction<T>? activationFunction = null)
    {
        _weights = new Matrix<T>(outputSize, inputSize);
        _biases = new Vector<T>(outputSize);
        _activationFunction = activationFunction ?? ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU);
        _vectorActivationFunction = null;
        InitializeParameters();
    }

    public Layer(int inputSize, int outputSize, IVectorActivationFunction<T>? vectorActivationFunction = null)
    {
        _weights = new Matrix<T>(outputSize, inputSize);
        _biases = new Vector<T>(outputSize);
        _activationFunction = null;
        _vectorActivationFunction = vectorActivationFunction ?? ActivationFunctionFactory<T>.CreateVectorActivationFunction(ActivationFunction.Softmax);
        InitializeParameters();
    }

    private void InitializeParameters()
    {
        // Xavier/Glorot initialization
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_weights.Columns + _weights.Rows)));
        Random rand = new Random();
        for (int i = 0; i < _weights.Rows; i++)
        {
            for (int j = 0; j < _weights.Columns; j++)
            {
                _weights[i, j] = NumOps.Multiply(NumOps.FromDouble(rand.NextDouble() - 0.5), scale);
            }
            _biases[i] = NumOps.Zero;
        }
    }

    public Vector<T> Forward(Vector<T> input)
    {
        Vector<T> output = _weights * input + _biases;
        return ApplyActivation(output);
    }

    private Vector<T> ApplyActivation(Vector<T> input)
    {
        if (_vectorActivationFunction != null)
        {
            return _vectorActivationFunction.Activate(input);
        }
        else if (_activationFunction != null)
        {
            return input.Transform(_activationFunction.Activate);
        }
        else
        {
            return Vector<T>.Empty();
        }
    }

    public void UpdateParameters(Vector<T> parameters)
    {
        int weightCount = _weights.Rows * _weights.Columns;
        for (int i = 0; i < _weights.Rows; i++)
        {
            for (int j = 0; j < _weights.Columns; j++)
            {
                _weights[i, j] = parameters[i * _weights.Columns + j];
            }
        }

        for (int i = 0; i < _biases.Length; i++)
        {
            _biases[i] = parameters[weightCount + i];
        }
    }

    public void Serialize(BinaryWriter writer)
    {
        writer.Write(_weights.Rows);
        writer.Write(_weights.Columns);
        for (int i = 0; i < _weights.Rows; i++)
        {
            for (int j = 0; j < _weights.Columns; j++)
            {
                writer.Write(Convert.ToDouble(_weights[i, j]));
            }
        }

        for (int i = 0; i < _biases.Length; i++)
        {
            writer.Write(Convert.ToDouble(_biases[i]));
        }

        // Serialize the type of activation function used
        if (_vectorActivationFunction != null)
        {
            writer.Write(_vectorActivationFunction.GetType().AssemblyQualifiedName ?? string.Empty);
            writer.Write(true); // Flag to indicate it's a vector activation function
        }
        else if (_activationFunction != null)
        {
            writer.Write(_activationFunction.GetType().AssemblyQualifiedName ?? string.Empty);
            writer.Write(false); // Flag to indicate it's not a vector activation function
        }
        else
        {
            writer.Write(string.Empty);
            writer.Write(false);
        }
    }

    public void Deserialize(BinaryReader reader)
    {
        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        _weights = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                _weights[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        _biases = new Vector<T>(rows);
        for (int i = 0; i < rows; i++)
        {
            _biases[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        string activationFunctionTypeName = reader.ReadString();
        bool isVectorActivationFunction = reader.ReadBoolean();

        if (!string.IsNullOrEmpty(activationFunctionTypeName))
        {
            Type? activationFunctionType = Type.GetType(activationFunctionTypeName);
            if (activationFunctionType != null)
            {
                if (isVectorActivationFunction)
                {
                    _vectorActivationFunction = (IVectorActivationFunction<T>?)Activator.CreateInstance(activationFunctionType);
                    _activationFunction = null;
                }
                else
                {
                    _activationFunction = (IActivationFunction<T>?)Activator.CreateInstance(activationFunctionType);
                    _vectorActivationFunction = null;
                }
            }
        }
    }
}
