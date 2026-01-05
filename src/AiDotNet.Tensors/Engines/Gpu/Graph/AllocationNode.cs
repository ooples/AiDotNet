using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.Gpu.Graph;

/// <summary>
/// Represents a buffer allocation node in the execution graph.
/// Used for allocating and deallocating GPU memory.
/// </summary>
public sealed class AllocationNode : ExecutionNode
{
    private readonly int _size;
    private readonly bool _isDeallocation;
    private IGpuBuffer? _allocatedBuffer;
    private readonly IGpuBuffer? _bufferToDeallocate;

    /// <inheritdoc/>
    public override ExecutionNodeType NodeType =>
        _isDeallocation ? ExecutionNodeType.Deallocate : ExecutionNodeType.Allocate;

    /// <summary>
    /// Gets the size of the allocation in elements.
    /// </summary>
    public int Size => _size;

    /// <summary>
    /// Gets the allocated buffer after execution (for allocation nodes).
    /// </summary>
    public IGpuBuffer? AllocatedBuffer => _allocatedBuffer;

    /// <summary>
    /// Gets whether this is a deallocation node.
    /// </summary>
    public bool IsDeallocation => _isDeallocation;

    /// <summary>
    /// Gets the tensor role for the allocated buffer.
    /// </summary>
    public GpuTensorRole Role { get; }

    /// <inheritdoc/>
    public override int EstimatedCost => _isDeallocation ? 1 : (_size / 4096) + 2;

    /// <inheritdoc/>
    public override string Name => _isDeallocation
        ? $"Deallocate_{NodeId}"
        : $"Allocate_{_size}_{Role}_{NodeId}";

    /// <summary>
    /// Creates an allocation node.
    /// </summary>
    /// <param name="size">Number of elements to allocate.</param>
    /// <param name="role">The role of the allocated buffer.</param>
    public static AllocationNode CreateAllocate(int size, GpuTensorRole role = GpuTensorRole.Intermediate)
    {
        return new AllocationNode(size, role);
    }

    /// <summary>
    /// Creates a deallocation node.
    /// </summary>
    /// <param name="buffer">The buffer to deallocate.</param>
    public static AllocationNode CreateDeallocate(IGpuBuffer buffer)
    {
        return new AllocationNode(buffer);
    }

    private AllocationNode(int size, GpuTensorRole role)
    {
        _size = size;
        Role = role;
        _isDeallocation = false;
    }

    private AllocationNode(IGpuBuffer buffer)
    {
        _bufferToDeallocate = buffer ?? throw new ArgumentNullException(nameof(buffer));
        _size = buffer.Size;
        _isDeallocation = true;
    }

    /// <inheritdoc/>
    public override void Execute(IDirectGpuBackend backend)
    {
        if (_isDeallocation)
        {
            _bufferToDeallocate?.Dispose();
        }
        else
        {
            _allocatedBuffer = backend.AllocateBuffer(_size);
        }
    }

    /// <inheritdoc/>
    public override void ExecuteAsync(IDirectGpuBackend backend, IGpuStream stream)
    {
        // Allocations are typically synchronous, but we still track stream assignment
        AssignedStream = stream;
        Execute(backend);
    }

    /// <inheritdoc/>
    public override ExecutionNode Clone()
    {
        if (_isDeallocation && _bufferToDeallocate != null)
        {
            return CreateDeallocate(_bufferToDeallocate);
        }
        return CreateAllocate(_size, Role);
    }
}
