namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a temporal memory layer that models sequence learning through hierarchical temporal memory concepts.
/// </summary>
/// <remarks>
/// <para>
/// A temporal memory layer implements a simplified version of Hierarchical Temporal Memory (HTM) concepts to learn
/// sequential patterns in data. It organizes cells into columns, where cells within the same column represent
/// alternative contexts for the same input pattern. This allows the layer to maintain multiple predictions
/// simultaneously and learn temporal patterns in the input data.
/// </para>
/// <para><b>For Beginners:</b> This layer helps the network remember and predict sequences of patterns.
/// 
/// Think of it like learning to anticipate what comes next in a song:
/// - The layer organizes memory cells into columns (like musical notes)
/// - Each column can have multiple cells (representing different contexts for the same note)
/// - When a note plays, the layer activates specific cells based on what came before
/// - Over time, it learns which notes typically follow others in different contexts
/// 
/// For example, in a melody, the note "C" might be followed by "D" in the verse but by "G" in the chorus.
/// This layer helps the network learn such context-dependent sequences by remembering not just what
/// happened, but the context in which it happened.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class TemporalMemoryLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The number of columns in the temporal memory layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the number of columns in the temporal memory layer. Each column represents a distinct
    /// spatial pattern or feature that can be recognized in the input.
    /// </para>
    /// <para><b>For Beginners:</b> This represents how many different patterns the layer can recognize.
    /// 
    /// Columns work like this:
    /// - Each column responds to a specific input pattern
    /// - When an input matches that pattern, the column becomes active
    /// - More columns means the layer can recognize more distinct patterns
    /// 
    /// For example, if processing text, different columns might recognize different letters,
    /// words, or phrases depending on the level of abstraction.
    /// </para>
    /// </remarks>
    private readonly int ColumnCount;

    /// <summary>
    /// The number of cells per column in the temporal memory layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the number of cells in each column of the temporal memory layer. Multiple cells per column
    /// allow the layer to represent the same input pattern in different temporal contexts.
    /// </para>
    /// <para><b>For Beginners:</b> This represents how many different contexts the layer can distinguish for each pattern.
    /// 
    /// Cells per column work like this:
    /// - Each cell in a column represents the same pattern in a different context
    /// - More cells per column means more temporal contexts can be distinguished
    /// - This allows the layer to make different predictions based on what came before
    /// 
    /// For example, in language processing, the word "bank" might activate different cells
    /// depending on whether the previous words suggested a financial context or a river context.
    /// </para>
    /// </remarks>
    private readonly int CellsPerColumn;

    /// <summary>
    /// The states of all cells in the temporal memory layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This matrix stores the state of each cell in the temporal memory layer. Each value represents the activation
    /// level of a cell, with higher values indicating stronger activation.
    /// </para>
    /// <para><b>For Beginners:</b> This stores how active each memory cell currently is.
    /// 
    /// Cell states work like this:
    /// - Each cell has an activation value between 0 and 1
    /// - Higher values mean the cell is more strongly activated
    /// - These states change as the layer processes sequences of inputs
    /// - The pattern of active cells represents both the current input and its context
    /// 
    /// Think of it as a grid where each position holds a number indicating how "excited"
    /// that particular memory cell is at the current moment.
    /// </para>
    /// </remarks>
    private Matrix<T> CellStates;

    /// <summary>
    /// Gets or sets the previous input state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property stores the input vector from the previous time step, which is used to establish temporal context
    /// for learning sequential patterns.
    /// </para>
    /// <para><b>For Beginners:</b> This remembers what patterns were active in the previous step.
    /// 
    /// Previous state works like this:
    /// - It stores which columns were active in the last time step
    /// - This provides context for understanding the current input
    /// - It helps the layer learn which patterns typically follow others
    /// 
    /// For example, when learning a melody, this would remember which notes were just played,
    /// helping the layer understand the musical context of the current note.
    /// </para>
    /// </remarks>
    public Vector<T> PreviousState { get; set; }

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>true</c> for this layer, as it implements temporal learning mechanisms.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the temporal memory layer can be trained. Since this layer implements
    /// mechanisms for learning sequential patterns, it supports training.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer has internal states that can be adjusted during training
    /// - It will improve its performance as it sees more data
    /// - It participates in the learning process
    /// 
    /// For this layer, the value is always true because it needs to learn which sequences
    /// of patterns commonly occur in the input data.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="TemporalMemoryLayer{T}"/> class.
    /// </summary>
    /// <param name="columnCount">The number of columns in the layer.</param>
    /// <param name="cellsPerColumn">The number of cells per column.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a temporal memory layer with the specified number of columns and cells per column.
    /// It initializes all cell states to zero and sets up the layer for processing sequential data.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new temporal memory layer.
    /// 
    /// The parameters you provide determine:
    /// - columnCount: How many different patterns the layer can recognize
    /// - cellsPerColumn: How many different contexts the layer can distinguish for each pattern
    /// 
    /// Together, these parameters define the memory capacity of the layer:
    /// - More columns allow recognition of more distinct input patterns
    /// - More cells per column allow finer discrimination of temporal contexts
    /// 
    /// For example, a layer with 1000 columns and 4 cells per column could recognize 1000 different
    /// patterns, each in 4 different temporal contexts.
    /// </para>
    /// </remarks>
    public TemporalMemoryLayer(int columnCount, int cellsPerColumn)
        : base([columnCount], [columnCount * cellsPerColumn])
    {
        ColumnCount = columnCount;
        CellsPerColumn = cellsPerColumn;
        CellStates = new Matrix<T>(columnCount, cellsPerColumn);
        PreviousState = Vector<T>.Empty();

        InitializeCellStates();
    }

    /// <summary>
    /// Initializes all cell states to zero.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes all cell states in the temporal memory layer to zero, effectively resetting the
    /// layer's internal state to an inactive baseline.
    /// </para>
    /// <para><b>For Beginners:</b> This method resets all memory cells to an inactive state.
    /// 
    /// When initializing cell states:
    /// - All cells are set to 0 (completely inactive)
    /// - The layer starts with a "clean slate"
    /// - No patterns or contexts are recognized yet
    /// 
    /// This is like erasing a whiteboard before starting to write new information.
    /// The layer needs to learn patterns from scratch after this initialization.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Performs the forward pass of the temporal memory layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor representing cell activations.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the temporal memory layer. It takes an input tensor where each
    /// element represents the activation of a column, and outputs a tensor where each element represents the
    /// activation of an individual cell. Only cells in active columns (where the input is 1) can be active in the output.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes the input data through the temporal memory cells.
    /// 
    /// During the forward pass:
    /// - The layer receives input indicating which columns should be active
    /// - For each active column, it outputs the activation states of all cells in that column
    /// - Inactive columns produce no output (all zeros)
    /// 
    /// This process:
    /// - Transforms column-level input into cell-level output
    /// - Preserves the context information stored in cell states
    /// - Creates an output that represents both the input and its temporal context
    /// 
    /// For example, if the input activates the column for the letter "a", the output
    /// will reflect which specific "a" contexts are currently active based on past inputs.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Updates the cell states based on the current input and previous state.
    /// </summary>
    /// <param name="input">The current input vector.</param>
    /// <param name="previousState">The previous input state.</param>
    /// <remarks>
    /// <para>
    /// This method implements the learning mechanism of the temporal memory layer. It strengthens the activation
    /// of cells in active columns and weakens the activation of cells in inactive columns. After updating, the
    /// cell states are normalized so that they sum to 1 within each column.
    /// </para>
    /// <para><b>For Beginners:</b> This method is where the layer learns temporal patterns.
    /// 
    /// During learning:
    /// 
    /// 1. For active columns (input = 1):
    ///    - Active cells get strengthened (their activation increases)
    ///    - This reinforces the cells that correctly predicted the current input
    /// 
    /// 2. For inactive columns (input = 0):
    ///    - All cells get weakened (their activation decreases)
    ///    - This reduces false predictions
    /// 
    /// 3. Finally, cell states within each column are normalized:
    ///    - This ensures the total activation in each column remains constant
    ///    - It creates a competition among cells in the same column
    /// 
    /// Over time, this process helps the layer learn which cells should be active
    /// in which temporal contexts, allowing it to make better predictions.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Gets the predicted columns based on the current cell states.
    /// </summary>
    /// <returns>A vector containing the predicted column activations.</returns>
    /// <remarks>
    /// <para>
    /// This method predicts which columns are likely to be active in the next time step based on the current
    /// activation patterns of cells. A column is predicted to be active if any of its cells have a high activation state,
    /// indicating that the pattern represented by that column is expected to occur based on the current context.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you what patterns the network expects to see next.
    /// 
    /// The prediction process works like this:
    /// - The layer examines the current state of all cells
    /// - For each column, it determines if any cells are strongly activated
    /// - If yes, that column is predicted to be active in the next time step
    /// - The result is a vector where 1 means "predicted active" and 0 means "not predicted"
    /// 
    /// Think of it like predicting the next note in a melody based on what you've heard so far.
    /// If you've heard a sequence like "do-re-mi" many times, when you hear "do-re", 
    /// you would predict "mi" as the next note.
    /// </para>
    /// </remarks>
    public Vector<T> GetPredictions()
    {
        var predictions = new Vector<T>(ColumnCount);
    
        // Prediction threshold - cells with activation above this value
        // contribute to predicting their column will be active
        T predictionThreshold = NumOps.FromDouble(0.3);
    
        for (int i = 0; i < ColumnCount; i++)
        {
            // A column is predicted if any of its cells have a high activation state
            bool isPredicted = false;
        
            for (int j = 0; j < CellsPerColumn; j++)
            {
                if (NumOps.GreaterThan(CellStates[i, j], predictionThreshold))
                {
                    isPredicted = true;
                    break;
                }
            }
        
            // Set prediction for this column
            predictions[i] = isPredicted ? NumOps.One : NumOps.Zero;
        }
    
        return predictions;
    }

    /// <summary>
    /// Performs the backward pass of the temporal memory layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the temporal memory layer, which is used during training to propagate
    /// error gradients back through the network. It sums the gradients for all cells in each column to produce a
    /// gradient for each column in the input.
    /// </para>
    /// <para><b>For Beginners:</b> This method helps the layer understand how to adjust its behavior during training.
    /// 
    /// During the backward pass:
    /// - The layer receives information about how its output should change (outputGradient)
    /// - It calculates how each input column contributed to the output errors
    /// - It creates an inputGradient to pass back to previous layers
    /// 
    /// The process works by:
    /// - Summing up the gradients for all cells in each column
    /// - This tells the layer how each column's activation affected the final result
    /// 
    /// While the main learning in this layer happens through the Learn method,
    /// this backward pass allows it to participate in the neural network's
    /// overall gradient-based learning.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass implementation using optimized gradient calculations.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
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

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation to compute gradients. Specialized operations
    /// are not yet available in TensorOperations, so this falls back to the manual implementation.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        // TemporalMemoryLayer implements HTM temporal memory with complex state management
        // The manual implementation provides correct gradient computation through temporal sequences
        // No new TensorOperation needed as the logic is domain-specific to HTM.
        return BackwardManual(outputGradient);
    }


    /// <summary>
    /// Updates the parameters of the layer.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method is empty in the current implementation as the layer does not have traditional trainable parameters
    /// updated through gradient descent. Instead, learning occurs in the Learn method which directly updates cell states.
    /// </para>
    /// <para><b>For Beginners:</b> This method is included for compatibility but doesn't do anything in this layer.
    /// 
    /// The reason this method is empty:
    /// - This layer doesn't use traditional gradient-based parameter updates
    /// - Instead, it learns by directly modifying cell states in the Learn method
    /// - This method is included only to satisfy the requirements of the LayerBase class
    /// 
    /// Think of it like this: while standard neural network layers learn through small
    /// adjustments based on error gradients, this layer learns through its specialized
    /// temporal learning algorithm implemented in the Learn method.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // In this implementation, we don't have trainable parameters to update
        // The learning is done in the Learn method by adjusting cell states
    }

    /// <summary>
    /// Gets all cell states of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all cell states.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all cell states in the temporal memory layer and combines them into a single vector.
    /// This is useful for saving the layer's state or for visualization purposes.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the cell activation values into a single list.
    /// 
    /// The parameters:
    /// - Are the current activation values of all cells in the layer
    /// - Represent what the layer has learned about temporal sequences
    /// - Are flattened into a single long list (vector)
    /// 
    /// This is useful for:
    /// - Saving the layer's current state to disk
    /// - Visualizing what patterns the layer has learned
    /// - Transferring the learned state to another network
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the temporal memory layer by setting all cell states and the previous
    /// state vector to zero. This is useful when starting to process a new, unrelated sequence.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - All cell activations are set to zero
    /// - The record of previous input is cleared
    /// - The layer forgets all temporal context
    /// 
    /// This is important for:
    /// - Processing a new, unrelated sequence
    /// - Preventing information from one sequence affecting another
    /// - Testing the layer with fresh inputs
    /// 
    /// Think of it like clearing your mind before starting a completely new task,
    /// so that memories from the previous task don't interfere.
    /// </para>
    /// </remarks>
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

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        // TemporalMemoryLayer uses HTM sequence learning with complex cell state tracking
        throw new NotSupportedException(
            "TemporalMemoryLayer does not support JIT compilation because it implements Hierarchical Temporal Memory (HTM) " +
            "sequence learning with complex cell state tracking, predictive columns, and temporal context. The layer maintains " +
            "internal state across time steps and uses adaptive learning rules that cannot be represented in a static computation graph.");
    }

    public override bool SupportsJitCompilation => false; // Requires HTM temporal state tracking

}