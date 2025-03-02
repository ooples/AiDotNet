namespace AiDotNet.NeuralNetworks.Layers;

public class TemporalMemoryLayer<T> : LayerBase<T>
{
    private readonly int ColumnCount;
    private readonly int CellsPerColumn;
    private Matrix<T> CellStates;

    public Vector<T> PreviousState { get; set; }

    public override bool SupportsTraining => true;

    public TemporalMemoryLayer(int columnCount, int cellsPerColumn)
        : base([columnCount], [columnCount * cellsPerColumn])
    {
        ColumnCount = columnCount;
        CellsPerColumn = cellsPerColumn;
        CellStates = new Matrix<T>(columnCount, cellsPerColumn);
        PreviousState = Vector<T>.Empty();

        InitializeCellStates();
    }

    private void InitializeCellStates()
    {
        for (int i = 0; i < ColumnCount; i++)
        {
            for (int j = 0; j < CellsPerColumn; j++)
            {
                CellStates[i, j] = NumOps.Zero;
            }
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        var inputVector = input.ToVector();
        var output = new Vector<T>(ColumnCount * CellsPerColumn);

        for (int i = 0; i < ColumnCount; i++)
        {
            if (NumOps.Equals(inputVector[i], NumOps.One))
            {
                for (int j = 0; j < CellsPerColumn; j++)
                {
                    output[i * CellsPerColumn + j] = CellStates[i, j];
                }
            }
        }

        return Tensor<T>.FromVector(output);
    }

    public void Learn(Vector<T> input, Vector<T> previousState)
    {
        for (int i = 0; i < ColumnCount; i++)
        {
            if (NumOps.Equals(input[i], NumOps.One))
            {
                // Strengthen connections for active cells
                for (int j = 0; j < CellsPerColumn; j++)
                {
                    if (NumOps.Equals(CellStates[i, j], NumOps.One))
                    {
                        CellStates[i, j] = NumOps.Add(CellStates[i, j], NumOps.Multiply(NumOps.FromDouble(0.1), NumOps.One));
                    }
                }
            }
            else
            {
                // Weaken connections for inactive cells
                for (int j = 0; j < CellsPerColumn; j++)
                {
                    CellStates[i, j] = NumOps.Subtract(CellStates[i, j], NumOps.Multiply(NumOps.FromDouble(0.05), NumOps.One));
                }
            }
        }

        // Normalize cell states
        for (int i = 0; i < ColumnCount; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < CellsPerColumn; j++)
            {
                sum = NumOps.Add(sum, CellStates[i, j]);
            }
            if (!NumOps.Equals(sum, NumOps.Zero))
            {
                for (int j = 0; j < CellsPerColumn; j++)
                {
                    CellStates[i, j] = NumOps.Divide(CellStates[i, j], sum);
                }
            }
        }
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        var inputGradient = new Vector<T>(ColumnCount);
        var flatGradient = outputGradient.ToVector();

        for (int i = 0; i < ColumnCount; i++)
        {
            T columnGradient = NumOps.Zero;
            for (int j = 0; j < CellsPerColumn; j++)
            {
                columnGradient = NumOps.Add(columnGradient, flatGradient[i * CellsPerColumn + j]);
            }

            inputGradient[i] = columnGradient;
        }

        return Tensor<T>.FromVector(inputGradient);
    }

    public override void UpdateParameters(T learningRate)
    {
        // In this implementation, we don't have trainable parameters to update
        // The learning is done in the Learn method by adjusting cell states
    }

    public override Vector<T> GetParameters()
    {
        // Flatten the cell states matrix into a vector
        var parameters = new Vector<T>(ColumnCount * CellsPerColumn);
    
        int index = 0;
        for (int i = 0; i < ColumnCount; i++)
        {
            for (int j = 0; j < CellsPerColumn; j++)
            {
                parameters[index++] = CellStates[i, j];
            }
        }
    
        return parameters;
    }

    public override void ResetState()
    {
        // Reset all cell states to zero
        InitializeCellStates();
    
        // Clear previous state
        if (PreviousState.Length > 0)
        {
            for (int i = 0; i < PreviousState.Length; i++)
            {
                PreviousState[i] = NumOps.Zero;
            }
        }
        else if (ColumnCount > 0)
        {
            // Initialize previous state if it's empty
            PreviousState = new Vector<T>(ColumnCount);
            for (int i = 0; i < ColumnCount; i++)
            {
                PreviousState[i] = NumOps.Zero;
            }
        }
    }
}