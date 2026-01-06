using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

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
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine");

        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend unavailable");

        var input = inputs[0];
        int inputSize = input.ElementCount;
        int outputSize = ColumnCount * CellsPerColumn;

        // Download input data from GPU
        var inputData = backend.DownloadBuffer(input.Buffer);

        // Get cell states as float array
        var cellStatesData = DirectGpuEngine.ToFloatArray<T>(CellStates.Data);

        // Create mask by repeating each input element CellsPerColumn times
        var maskData = new float[outputSize];
        for (int c = 0; c < ColumnCount; c++)
        {
            float inputVal = inputData[c];
            for (int cell = 0; cell < CellsPerColumn; cell++)
            {
                maskData[c * CellsPerColumn + cell] = inputVal;
            }
        }

        // Upload mask and cell states to GPU
        var maskBuffer = backend.AllocateBuffer(maskData);
        var cellStatesBuffer = backend.AllocateBuffer(cellStatesData);
        var outputBuffer = backend.AllocateBuffer(outputSize);

        // Element-wise multiply on GPU: output = mask * cellStates
        backend.Multiply(maskBuffer, cellStatesBuffer, outputBuffer, outputSize);

        // Clean up intermediate buffers
        maskBuffer.Dispose();
        cellStatesBuffer.Dispose();

        return new GpuTensor<T>(backend, outputBuffer, new[] { outputSize }, GpuTensorRole.Activation, ownsBuffer: true);
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
        CellStates = Engine.TensorDivide(CellStates, safeSums);
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
        var zeroTensor = new Tensor<T>(columnMax.Shape);
        zeroTensor.Fill(NumOps.Zero);
        var predictions = Engine.TensorGreaterThan(columnMax, zeroTensor);

        return new Vector<T>(predictions.ToArray());
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
        // Reshape output gradient from [ColumnCount * CellsPerColumn] to [ColumnCount, CellsPerColumn]
        var reshapedGrad = outputGradient.Reshape([ColumnCount, CellsPerColumn]);

        // Sum gradients along axis 1 (cells) to get gradient per column
        // This aggregates the gradients from all cells in each column
        var inputGradient = Engine.ReduceSum(reshapedGrad, [1], keepDims: false);

        return inputGradient;
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
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Build graph equivalent to repeating input across cells: matmul with ones then flatten
        var inputNode = Autodiff.TensorOperations<T>.Variable(_lastInput, "tm_input", requiresGradient: true);
        var inputReshaped = Autodiff.TensorOperations<T>.Reshape(inputNode, new[] { ColumnCount, 1 });

        var onesData = new T[CellsPerColumn];
        for (int i = 0; i < CellsPerColumn; i++) onesData[i] = NumOps.One;
        var onesTensor = new Tensor<T>(new[] { 1, CellsPerColumn }, new Vector<T>(onesData));
        var onesNode = Autodiff.TensorOperations<T>.Constant(onesTensor, "tm_ones");

        var repeated = Autodiff.TensorOperations<T>.MatrixMultiply(inputReshaped, onesNode);
        var flattened = Autodiff.TensorOperations<T>.Reshape(repeated, new[] { ColumnCount * CellsPerColumn });

        flattened.Gradient = outputGradient;

        // Inline topological sort
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var topoOrder = new List<Autodiff.ComputationNode<T>>();
        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((flattened, false));

        while (stack.Count > 0)
        {
            var (node, processed) = stack.Pop();
            if (visited.Contains(node)) continue;

            if (processed)
            {
                visited.Add(node);
                topoOrder.Add(node);
            }
            else
            {
                stack.Push((node, true));
                if (node.Parents != null)
                {
                    foreach (var parent in node.Parents)
                    {
                        if (!visited.Contains(parent))
                            stack.Push((parent, false));
                    }
                }
            }
        }

        for (int i = topoOrder.Count - 1; i >= 0; i--)
        {
            var node = topoOrder[i];
            if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
            {
                node.BackwardFunction(node.Gradient);
            }
        }

        if (inputNode.Gradient == null)
            throw new InvalidOperationException("Gradient computation failed in temporal memory autodiff.");

        return inputNode.Gradient;
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

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (inputNodes.Count == 0)
            throw new ArgumentException("At least one input node is required.", nameof(inputNodes));

        // TemporalMemoryLayer JIT uses a simplified differentiable approximation:
        // 1. Project input through cell states matrix
        // 2. Apply sigmoid for cell activation probabilities
        // 3. Apply straight-through threshold for binary output
        //
        // This approximates the HTM temporal memory behavior with differentiable operations.

        var input = inputNodes[0];

        // CellStates is [ColumnCount, CellsPerColumn], need to reshape for projection
        // Transpose CellStates to [CellsPerColumn, ColumnCount] then expand for output
        int outputSize = ColumnCount * CellsPerColumn;

        // Create expanded cell states tensor for projection [outputSize, ColumnCount]
        var cellStatesTensor = new Tensor<T>([outputSize, ColumnCount]);
        for (int col = 0; col < ColumnCount; col++)
        {
            for (int cell = 0; cell < CellsPerColumn; cell++)
            {
                int outputIdx = col * CellsPerColumn + cell;
                // Each output cell responds to its column's input
                cellStatesTensor[outputIdx, col] = CellStates[col, cell];
            }
        }

        var cellStatesNode = TensorOperations<T>.Constant(cellStatesTensor, "tm_cell_states");

        // Project input through cell states
        var inputReshaped = TensorOperations<T>.Reshape(input, ColumnCount, 1);
        var projection = TensorOperations<T>.MatrixMultiply(cellStatesNode, inputReshaped);
        var projectionFlat = TensorOperations<T>.Reshape(projection, outputSize);

        // Apply sigmoid for activation probabilities
        var activations = TensorOperations<T>.Sigmoid(projectionFlat);

        // Apply straight-through threshold for binary cell output
        var output = TensorOperations<T>.StraightThroughThreshold(activations, 0.5);

        // Apply layer activation
        output = ApplyActivationToGraph(output);

        return output;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <value>
    /// Always <c>true</c>. TemporalMemoryLayer uses a differentiable approximation for JIT.
    /// </value>
    /// <remarks>
    /// <para>
    /// JIT compilation for TemporalMemory uses a simplified differentiable approximation
    /// of the HTM algorithm. The complex cell state tracking and prediction mechanisms
    /// are approximated with matrix projections and sigmoid activations, enabling
    /// gradient-based optimization while maintaining similar sparse activation patterns.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

}
