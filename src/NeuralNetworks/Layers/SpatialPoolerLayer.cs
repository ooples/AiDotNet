namespace AiDotNet.NeuralNetworks.Layers;

public class SpatialPoolerLayer<T> : LayerBase<T>
{
    private readonly int InputSize;
    private readonly int ColumnCount;
    private readonly double SparsityThreshold;
    private Matrix<T> Connections;
    
    private Vector<T>? LastInput;
    private Vector<T>? LastOutput;
    private readonly double LearningRate = 0.01;
    private readonly double BoostFactor = 0.005;

    public override bool SupportsTraining => true;

    public SpatialPoolerLayer(int inputSize, int columnCount, double sparsityThreshold)
        : base([inputSize], [columnCount])
    {
        InputSize = inputSize;
        ColumnCount = columnCount;
        SparsityThreshold = sparsityThreshold;
        Connections = new Matrix<T>(inputSize, columnCount);

        InitializeConnections();
    }

    private void InitializeConnections()
    {
        // Initialize connections with random values
        for (int i = 0; i < InputSize; i++)
        {
            for (int j = 0; j < ColumnCount; j++)
            {
                Connections[i, j] = NumOps.FromDouble(Random.NextDouble());
            }
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        var inputVector = input.ToVector();
        var output = new Vector<T>(ColumnCount);

        for (int i = 0; i < ColumnCount; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < InputSize; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(inputVector[j], Connections[j, i]));
            }
            output[i] = NumOps.GreaterThan(sum, NumOps.FromDouble(SparsityThreshold)) ? NumOps.One : NumOps.Zero;
        }

        return Tensor<T>.FromVector(output);
    }

    public void Learn(Vector<T> input)
    {
        LastInput = input;
        LastOutput = Forward(Tensor<T>.FromVector(input)).ToVector();

        for (int i = 0; i < ColumnCount; i++)
        {
            if (NumOps.Equals(LastOutput[i], NumOps.One))
            {
                for (int j = 0; j < InputSize; j++)
                {
                    T delta = NumOps.Multiply(NumOps.FromDouble(LearningRate), 
                        NumOps.Subtract(input[j], Connections[j, i]));
                    Connections[j, i] = NumOps.Add(Connections[j, i], delta);
                }
            }
        }

        // Boost inactive columns
        for (int i = 0; i < ColumnCount; i++)
        {
            if (NumOps.Equals(LastOutput[i], NumOps.Zero))
            {
                for (int j = 0; j < InputSize; j++)
                {
                    T boost = NumOps.Multiply(NumOps.FromDouble(BoostFactor), input[j]);
                    Connections[j, i] = NumOps.Add(Connections[j, i], boost);
                }
            }
        }

        NormalizeConnections();
    }

    private void NormalizeConnections()
    {
        for (int i = 0; i < ColumnCount; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < InputSize; j++)
            {
                sum = NumOps.Add(sum, Connections[j, i]);
            }
            if (!NumOps.Equals(sum, NumOps.Zero))
            {
                for (int j = 0; j < InputSize; j++)
                {
                    Connections[j, i] = NumOps.Divide(Connections[j, i], sum);
                }
            }
        }
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        var inputGradient = new Vector<T>(InputSize);
        var flatGradient = outputGradient.ToVector();

        for (int i = 0; i < InputSize; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < ColumnCount; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(flatGradient[j], Connections[i, j]));
            }

            inputGradient[i] = sum;
        }

        return Tensor<T>.FromVector(inputGradient);
    }

    public override void UpdateParameters(T learningRate)
    {
        if (LastOutput == null || LastInput == null)
        {
            // Skip parameter update if we don't have previous input/output data
            return;
        }

        for (int i = 0; i < InputSize; i++)
        {
            for (int j = 0; j < ColumnCount; j++)
            {
                T delta = NumOps.Multiply(learningRate, 
                    NumOps.Multiply(LastOutput[j], LastInput[i]));
                Connections[i, j] = NumOps.Add(Connections[i, j], delta);
            }
        }

        NormalizeConnections();
    }

    public override Vector<T> GetParameters()
    {
        // Convert the connection matrix to a vector
        var parameters = new Vector<T>(InputSize * ColumnCount);
        int index = 0;
    
        for (int i = 0; i < InputSize; i++)
        {
            for (int j = 0; j < ColumnCount; j++)
            {
                parameters[index++] = Connections[i, j];
            }
        }
    
        return parameters;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != InputSize * ColumnCount)
        {
            throw new ArgumentException($"Expected {InputSize * ColumnCount} parameters, but got {parameters.Length}");
        }
    
        int index = 0;
    
        for (int i = 0; i < InputSize; i++)
        {
            for (int j = 0; j < ColumnCount; j++)
            {
                Connections[i, j] = parameters[index++];
            }
        }
    }

    public override void ResetState()
    {
        // Clear cached values
        LastInput = null;
        LastOutput = null;
    }
}