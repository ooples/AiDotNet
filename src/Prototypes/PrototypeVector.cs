using AiDotNet.Engines;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Validation;

namespace AiDotNet.Prototypes;

/// <summary>
/// Prototype vector class that delegates operations to the execution engine.
/// Used to validate the Execution Engine pattern before production integration.
/// </summary>
/// <typeparam name="T">The numeric type of vector elements.</typeparam>
/// <remarks>
/// This is a PROTOTYPE for Phase A validation. The production version will integrate
/// directly with the existing Vector&lt;T&gt; class.
///
/// PrototypeVector demonstrates:
/// - Engine delegation pattern
/// - Vectorized operations (no element-wise for-loops)
/// - Transparent GPU acceleration
/// - Zero constraint cascade
/// </remarks>
public class PrototypeVector<T>
{
    private readonly Vector<T> _data;

    /// <summary>
    /// Gets the length of the vector.
    /// </summary>
    public int Length => _data.Length;

    /// <summary>
    /// Gets or sets the element at the specified index.
    /// </summary>
    public T this[int index]
    {
        get => _data[index];
        set => _data[index] = value;
    }

    /// <summary>
    /// Initializes a new instance of the PrototypeVector class with the specified length.
    /// </summary>
    public PrototypeVector(int length)
    {
        _data = new Vector<T>(length);
    }

    /// <summary>
    /// Initializes a new instance of the PrototypeVector class from an existing Vector.
    /// </summary>
    public PrototypeVector(Vector<T> data)
    {
        Guard.NotNull(data);
        _data = data;
    }

    /// <summary>
    /// Initializes a new instance of the PrototypeVector class from an array.
    /// </summary>
    public PrototypeVector(T[] data)
    {
        if (data == null) throw new ArgumentNullException(nameof(data));
        _data = new Vector<T>(data.Length);
        for (int i = 0; i < data.Length; i++)
        {
            _data[i] = data[i];
        }
    }

    /// <summary>
    /// Gets the underlying Vector&lt;T&gt; data.
    /// </summary>
    public Vector<T> ToVector() => _data;

    /// <summary>
    /// Converts to array.
    /// </summary>
    public T[] ToArray()
    {
        var result = new T[Length];
        for (int i = 0; i < Length; i++)
        {
            result[i] = _data[i];
        }
        return result;
    }

    #region Vectorized Operations (Engine-Delegated)

    /// <summary>
    /// Adds two vectors element-wise using the current execution engine.
    /// </summary>
    /// <remarks>
    /// This operation is delegated to AiDotNetEngine.Current, which routes it to
    /// either GPU (for float) or CPU (for other types) based on runtime type checking.
    /// </remarks>
    public PrototypeVector<T> Add(PrototypeVector<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));

        var result = AiDotNetEngine.Current.Add(_data, other._data);
        return new PrototypeVector<T>(result);
    }

    /// <summary>
    /// Subtracts another vector from this vector element-wise.
    /// </summary>
    public PrototypeVector<T> Subtract(PrototypeVector<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));

        var result = AiDotNetEngine.Current.Subtract(_data, other._data);
        return new PrototypeVector<T>(result);
    }

    /// <summary>
    /// Multiplies two vectors element-wise (Hadamard product).
    /// </summary>
    public PrototypeVector<T> Multiply(PrototypeVector<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));

        var result = AiDotNetEngine.Current.Multiply(_data, other._data);
        return new PrototypeVector<T>(result);
    }

    /// <summary>
    /// Multiplies the vector by a scalar.
    /// </summary>
    public PrototypeVector<T> Multiply(T scalar)
    {
        var result = AiDotNetEngine.Current.Multiply(_data, scalar);
        return new PrototypeVector<T>(result);
    }

    /// <summary>
    /// Divides this vector by another vector element-wise.
    /// </summary>
    public PrototypeVector<T> Divide(PrototypeVector<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));

        var result = AiDotNetEngine.Current.Divide(_data, other._data);
        return new PrototypeVector<T>(result);
    }

    /// <summary>
    /// Divides the vector by a scalar.
    /// </summary>
    public PrototypeVector<T> Divide(T scalar)
    {
        var result = AiDotNetEngine.Current.Divide(_data, scalar);
        return new PrototypeVector<T>(result);
    }

    /// <summary>
    /// Computes the square root of each element.
    /// </summary>
    public PrototypeVector<T> Sqrt()
    {
        var result = AiDotNetEngine.Current.Sqrt(_data);
        return new PrototypeVector<T>(result);
    }

    /// <summary>
    /// Raises each element to the specified power.
    /// </summary>
    public PrototypeVector<T> Power(T exponent)
    {
        var result = AiDotNetEngine.Current.Power(_data, exponent);
        return new PrototypeVector<T>(result);
    }

    #endregion

    #region Static Factory Methods

    /// <summary>
    /// Creates a vector filled with zeros.
    /// </summary>
    public static PrototypeVector<T> Zeros(int length)
    {
        return new PrototypeVector<T>(length);
    }

    /// <summary>
    /// Creates a vector filled with ones.
    /// </summary>
    public static PrototypeVector<T> Ones(int length)
    {
        var vec = new PrototypeVector<T>(length);
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < length; i++)
        {
            vec[i] = numOps.One;
        }
        return vec;
    }

    /// <summary>
    /// Creates a vector from an array.
    /// </summary>
    public static PrototypeVector<T> FromArray(T[] data)
    {
        return new PrototypeVector<T>(data);
    }

    #endregion

    /// <summary>
    /// Returns a string representation of the vector.
    /// </summary>
    public override string ToString()
    {
        if (Length <= 10)
        {
            return $"PrototypeVector<{typeof(T).Name}>[{string.Join(", ", ToArray())}]";
        }
        else
        {
            var first3 = new[] { this[0], this[1], this[2] };
            var last3 = new[] { this[Length - 3], this[Length - 2], this[Length - 1] };
            return $"PrototypeVector<{typeof(T).Name}>[{string.Join(", ", first3)}, ..., {string.Join(", ", last3)}] (length={Length})";
        }
    }
}
