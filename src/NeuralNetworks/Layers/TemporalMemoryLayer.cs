using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Helpers;

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
[LayerCategory(LayerCategory.Memory)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(NormalizesInput = true, IsTrainable = true, SupportsBackpropagation = false, IsStateful = true, TestInputShape = "1, 4", TestConstructorArgs = "4, 4")]
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
    private Tensor<T>? _lastInput;

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
    private Tensor<T> CellStates;

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
    public override int ParameterCount => CellStates.Length;
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    protected override bool SupportsGpuExecution => true;

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
        CellStates = new Tensor<T>([columnCount, cellsPerColumn]);
        PreviousState = Vector<T>.Empty();

        InitializeCellStates();

        RegisterBuffer(CellStates, nameof(CellStates), PersistentTensorRole.Weights);
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
        // Use Fill for efficient initialization
        CellStates.Fill(NumOps.Zero);
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
        _lastInput = input;
        // Reshape CellStates from [ColumnCount, CellsPerColumn] to [ColumnCount * CellsPerColumn]
        var flatCellStates = CellStates.Reshape([ColumnCount * CellsPerColumn]);

        // Create mask by repeating input values CellsPerColumn times each
        // input is [ColumnCount], we need mask of [ColumnCount * CellsPerColumn]
        // Use Engine.TensorRepeatElements to repeat each input element CellsPerColumn times
        var inputFlat = input.Reshape([ColumnCount]);
        var mask = Engine.TensorRepeatElements(inputFlat, CellsPerColumn, axis: 0);

        // output = mask * flatCellStates (element-wise)
        var output = Engine.TensorMultiply(mask, flatCellStates);

        return output;
    }

    /// <inheritdoc/>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine");

        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend unavailable");

        var input = inputs[0];
        int outputSize = ColumnCount * CellsPerColumn;

        // Convert cell states to float array and upload to GPU
        var cellStatesData = DirectGpuEngine.ToFloatArray<T>(CellStates.Data.ToArray());
        using var cellStatesBuffer = backend.AllocateBuffer(cellStatesData);
        var outputBuffer = backend.AllocateBuffer(outputSize);

        // Use BroadcastMultiplyFirstAxis: output[col, cell] = cellStates[col, cell] * input[col]
        // This broadcasts input[ColumnCount] across the CellsPerColumn dimension without CPU download
        backend.BroadcastMultiplyFirstAxis(cellStatesBuffer, input.Buffer, outputBuffer, ColumnCount, CellsPerColumn);

        return GpuTensorHelper.UploadToGpu<T>(backend, outputBuffer, new[] { outputSize }, GpuTensorRole.Activation, ownsBuffer: true);
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
        // Convert input to tensor and create mask for active/inactive columns
        var inputTensor = new Tensor<T>(new[] { ColumnCount }, input);

        // Create column mask repeated for each cell: [ColumnCount, CellsPerColumn]
        var columnMask = Engine.TensorRepeatElements(inputTensor, CellsPerColumn, axis: 0)
            .Reshape([ColumnCount, CellsPerColumn]);

        // Create ones tensor for comparison
        var onesTensor = new Tensor<T>([ColumnCount, CellsPerColumn]);
        onesTensor.Fill(NumOps.One);

        // Create delta tensors for strengthening and weakening
        var strengthenDelta = new Tensor<T>([ColumnCount, CellsPerColumn]);
        strengthenDelta.Fill(NumOps.FromDouble(0.1));

        var weakenDelta = new Tensor<T>([ColumnCount, CellsPerColumn]);
        weakenDelta.Fill(NumOps.FromDouble(0.05));

        var zeroTensor = new Tensor<T>([ColumnCount, CellsPerColumn]);
        zeroTensor.Fill(NumOps.Zero);

        // For active columns (mask == 1): strengthen active cells
        // activeCellMask = (CellStates == 1) to identify which cells are active
        var activeCellMask = Engine.TensorEquals(CellStates, NumOps.One);

        // strengthenAmount = activeCellMask * strengthenDelta (only apply to active cells)
        var strengthenAmount = Engine.TensorMultiply(activeCellMask, strengthenDelta);

        // For inactive columns (mask == 0): weaken all cells
        // inactiveMask = 1 - columnMask
        var inactiveMask = Engine.TensorSubtract(onesTensor, columnMask);

        // weakenAmount = inactiveMask * weakenDelta
        var weakenAmount = Engine.TensorMultiply(inactiveMask, weakenDelta);

        // Apply strengthening for active columns: CellStates += columnMask * strengthenAmount
        var activeContribution = Engine.TensorMultiply(columnMask, strengthenAmount);
        CellStates = Engine.TensorAdd(CellStates, activeContribution);

        // Apply weakening for inactive columns: CellStates -= weakenAmount
        CellStates = Engine.TensorSubtract(CellStates, weakenAmount);

        // Normalize cell states per column using ReduceSum along axis 1
        // Sum along cells (axis 1) to get [ColumnCount, 1] sums
        var columnSums = Engine.ReduceSum(CellStates, [1], keepDims: true);

        // Avoid division by zero: where sum is zero, keep original values
        var safeSums = Engine.TensorMax(columnSums, NumOps.FromDouble(1e-10));

        // Normalize: CellStates = CellStates / safeSums (broadcast division)
        // safeSums shape [numColumns, 1] broadcasts across [numColumns, cellsPerColumn]
        CellStates = Engine.TensorBroadcastDivide(CellStates, safeSums);
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
        // Prediction threshold - cells with activation above this value
        // contribute to predicting their column will be active
        T predictionThreshold = NumOps.FromDouble(0.3);

        // Check which cells exceed threshold: CellStates > threshold
        var exceedsThreshold = Engine.TensorGreaterThan(CellStates, predictionThreshold);

        // Reduce max along axis 1 (cells) to see if any cell in each column exceeds threshold
        // If any cell in a column exceeds threshold, the max will be > 0
        var columnMax = Engine.ReduceMax(exceedsThreshold, [1], keepDims: false, out _);

        // Convert to binary predictions (1 if max > 0, else 0)
        // Create a zero tensor for comparison
        var zeroTensor = new Tensor<T>(columnMax.Shape.ToArray());
        zeroTensor.Fill(NumOps.Zero);
        var predictions = Engine.TensorGreaterThan(columnMax, zeroTensor);

        return new Vector<T>(predictions.ToArray());
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
        // Use Tensor.ToArray() to efficiently convert to vector
        return new Vector<T>(CellStates.ToArray());
    }

    /// <summary>
    /// Sets the trainable parameters of the layer from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets the cell states from a flattened vector. This is useful for loading
    /// saved model states or for transferring learned patterns to another network.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int expectedParams = ColumnCount * CellsPerColumn;

        if (parameters.Length != expectedParams)
        {
            throw new ArgumentException($"Expected {expectedParams} parameters, but got {parameters.Length}");
        }

        // Use Tensor<T>.FromVector and reshape to restore cell states
        CellStates = new Tensor<T>(new[] { ColumnCount, CellsPerColumn }, parameters);
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
        // Reset all cell states to zero using Fill
        InitializeCellStates();

        // Clear or initialize previous state
        if (PreviousState.Length > 0)
        {
            // Convert to tensor, fill with zeros, convert back
            var prevStateTensor = new Tensor<T>([PreviousState.Length]);
            prevStateTensor.Fill(NumOps.Zero);
            PreviousState = new Vector<T>(prevStateTensor.ToArray());
        }
        else if (ColumnCount > 0)
        {
            // Initialize previous state if it's empty
            var prevStateTensor = new Tensor<T>([ColumnCount]);
            prevStateTensor.Fill(NumOps.Zero);
            PreviousState = new Vector<T>(prevStateTensor.ToArray());
        }
    }

}
