using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Provides base implementation for distributed communication backends.
/// </summary>
/// <remarks>
/// <para>
/// This abstract class implements common functionality for all communication backends,
/// including state management, validation, and helper methods for collective operations.
/// Derived classes implement the specific communication mechanisms (MPI, NCCL, in-memory, etc.).
/// </para>
/// <para><b>For Beginners:</b> This is the foundation that all communication systems build upon.
///
/// Think of this as a template that defines how any communication system should work.
/// It handles common tasks like:
/// - Keeping track of whether the system is initialized
/// - Validating inputs (checking for null values, correct sizes, etc.)
/// - Providing helper methods for common operations
///
/// Specific communication backends (like MPI or in-memory) inherit from this and add
/// their own implementation details. This prevents code duplication and ensures
/// all backends work consistently.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for operations</typeparam>
public abstract class CommunicationBackendBase<T> : ICommunicationBackend<T>
{
    /// <summary>
    /// Provides numeric operations for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    private bool _isInitialized;

    /// <inheritdoc/>
    public abstract int Rank { get; }

    /// <inheritdoc/>
    public abstract int WorldSize { get; }

    /// <inheritdoc/>
    public bool IsInitialized => _isInitialized;

    /// <summary>
    /// Initializes a new instance of the CommunicationBackendBase class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor sets up the numeric operations provider that will be used
    /// for all mathematical operations on type T.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor is called when creating any communication backend.
    ///
    /// It sets up the math helper that allows the backend to perform operations
    /// like addition, multiplication, and comparison on any numeric type (double, float, etc.)
    /// without knowing in advance which type will be used.
    /// </para>
    /// </remarks>
    protected CommunicationBackendBase()
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        _isInitialized = false;
    }

    /// <inheritdoc/>
    public virtual void Initialize()
    {
        if (_isInitialized)
        {
            return;
        }

        OnInitialize();
        _isInitialized = true;
    }

    /// <inheritdoc/>
    public virtual void Shutdown()
    {
        if (!_isInitialized)
        {
            return;
        }

        OnShutdown();
        _isInitialized = false;
    }

    /// <summary>
    /// Called during initialization to perform backend-specific setup.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Derived classes override this method to implement their specific initialization logic,
    /// such as connecting to MPI or setting up shared memory structures.
    /// </para>
    /// <para><b>For Beginners:</b> This is where each specific backend does its setup work.
    ///
    /// For example:
    /// - An MPI backend would connect to the MPI environment
    /// - An in-memory backend would create shared data structures
    /// - An NCCL backend would initialize GPU communication channels
    /// </para>
    /// </remarks>
    protected virtual void OnInitialize()
    {
        // Base implementation does nothing - derived classes override
    }

    /// <summary>
    /// Called during shutdown to perform backend-specific cleanup.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Derived classes override this method to implement their specific cleanup logic,
    /// such as disconnecting from MPI or releasing shared memory.
    /// </para>
    /// <para><b>For Beginners:</b> This is where each backend cleans up its resources.
    ///
    /// It's like turning off equipment when you're done - releasing memory,
    /// closing connections, and ensuring everything shuts down cleanly.
    /// </para>
    /// </remarks>
    protected virtual void OnShutdown()
    {
        // Base implementation does nothing - derived classes override
    }

    /// <inheritdoc/>
    public abstract void Barrier();

    /// <inheritdoc/>
    public abstract void AllReduce(Vector<T> data, ReductionOperation operation);

    /// <inheritdoc/>
    public abstract Vector<T> AllGather(Vector<T> sendData);

    /// <inheritdoc/>
    public abstract Vector<T> Broadcast(Vector<T> data, int root = 0);

    /// <inheritdoc/>
    public abstract Vector<T> Scatter(Vector<T> sendData, int root = 0);

    /// <inheritdoc/>
    public abstract Vector<T> ReduceScatter(Vector<T> data, ReductionOperation operation);

    /// <summary>
    /// Ensures the backend is initialized before performing operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method throws an exception if the backend has not been initialized.
    /// All communication operations should call this before proceeding.
    /// </para>
    /// <para><b>For Beginners:</b> This is a safety check.
    ///
    /// Before doing any communication, we make sure the system is ready.
    /// If someone tries to use the backend without initializing it first,
    /// this method will throw an error with a helpful message.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown if the backend is not initialized</exception>
    protected void EnsureInitialized()
    {
        if (!_isInitialized)
        {
            throw new InvalidOperationException(
                "Communication backend is not initialized. Call Initialize() first.");
        }
    }

    /// <summary>
    /// Validates that a root rank is within valid bounds.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method ensures that the specified root rank is a valid process ID
    /// (between 0 and WorldSize - 1).
    /// </para>
    /// <para><b>For Beginners:</b> When one process acts as the "root" or "leader",
    /// we need to make sure that process actually exists.
    ///
    /// For example, if you have 4 processes (ranks 0-3), specifying rank 5 as the
    /// root would be an error. This method catches such mistakes.
    /// </para>
    /// </remarks>
    /// <param name="root">The root rank to validate</param>
    /// <exception cref="ArgumentException">Thrown if root is out of bounds</exception>
    protected void ValidateRoot(int root)
    {
        if (root < 0 || root >= WorldSize)
        {
            throw new ArgumentException(
                $"Invalid root {root}. Must be between 0 and {WorldSize - 1}.",
                nameof(root));
        }
    }

    /// <summary>
    /// Validates that data is not null.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method ensures that the data vector is not null before attempting
    /// communication operations.
    /// </para>
    /// <para><b>For Beginners:</b> This is a basic safety check to make sure
    /// we're not trying to send or receive null data, which would cause errors.
    /// </para>
    /// </remarks>
    /// <param name="data">The data to validate</param>
    /// <param name="paramName">The parameter name for error messages</param>
    /// <exception cref="ArgumentNullException">Thrown if data is null</exception>
    protected void ValidateData(Vector<T>? data, string paramName)
    {
        if (data == null)
        {
            throw new ArgumentNullException(paramName, "Data cannot be null.");
        }
    }

    /// <summary>
    /// Applies a reduction operation to two values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This helper method performs the specified reduction operation (Sum, Product, Min, Max)
    /// on two values. It's used internally by AllReduce and ReduceScatter implementations.
    /// </para>
    /// <para><b>For Beginners:</b> This is a helper that knows how to combine two numbers
    /// in different ways.
    ///
    /// For example:
    /// - Sum operation: 3 + 5 = 8
    /// - Product operation: 3 * 5 = 15
    /// - Min operation: Min(3, 5) = 3
    /// - Max operation: Max(3, 5) = 5
    ///
    /// We use this when combining values from multiple processes.
    /// </para>
    /// </remarks>
    /// <param name="a">The first value</param>
    /// <param name="b">The second value</param>
    /// <param name="operation">The reduction operation to apply</param>
    /// <returns>The result of applying the operation</returns>
    protected T ApplyReductionOperation(T a, T b, ReductionOperation operation)
    {
        return operation switch
        {
            ReductionOperation.Sum or ReductionOperation.Average => NumOps.Add(a, b),
            ReductionOperation.Product => NumOps.Multiply(a, b),
            ReductionOperation.Min => NumOps.LessThan(a, b) ? a : b,
            ReductionOperation.Max => NumOps.GreaterThan(a, b) ? a : b,
            _ => throw new NotSupportedException($"Operation {operation} is not supported.")
        };
    }
}
