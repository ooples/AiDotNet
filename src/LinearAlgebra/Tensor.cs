namespace AiDotNet.LinearAlgebra;

public class Tensor<T> : IEnumerable<T>
{
    private readonly Vector<T> _data;
    private readonly int[] _dimensions;
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public int[] Shape => _dimensions;
    public int Rank => _dimensions.Length;
    public int Length => _data.Length;

    public Tensor(int[] dimensions)
    {
        _dimensions = dimensions;
        int totalSize = dimensions.Aggregate(1, (a, b) => a * b);
        _data = new Vector<T>(totalSize);
    }

    public Tensor(int[] dimensions, Vector<T> data)
    {
        _dimensions = dimensions;
        int totalSize = dimensions.Aggregate(1, (a, b) => a * b);
        if (data.Length != totalSize)
            throw new ArgumentException("Data vector length must match the product of dimensions.");
        _data = data;
    }

    public IEnumerator<T> GetEnumerator()
    {
        return _data.GetEnumerator();
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }

    public T this[params int[] indices]
    {
        get => _data[GetFlatIndex(indices)];
        set => _data[GetFlatIndex(indices)] = value;
    }

    public Vector<T> GetSlice(int start, int length)
    {
        return _data.Slice(start, length);
    }

    public void SetSlice(int start, Vector<T> slice)
    {
        for (int i = 0; i < slice.Length; i++)
        {
            _data[start + i] = slice[i];
        }
    }

    public static Tensor<T> operator +(Tensor<T> left, Tensor<T> right)
    {
        return left.Add(right);
    }

    public static Tensor<T> operator +(Tensor<T> left, Vector<T> right)
    {
        return left.Add(right);
    }

    public static Tensor<T> operator *(Tensor<T> left, Tensor<T> right)
    {
        return left.Multiply(right);
    }

    public void SetSlice(int index, Tensor<T> slice)
    {
        if (index < 0 || index >= Shape[0])
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        if (!slice.Shape.SequenceEqual(Shape.Skip(1)))
        {
            throw new ArgumentException("Slice shape does not match tensor shape", nameof(slice));
        }

        int sliceSize = slice.Length;
        int offset = index * sliceSize;

        for (int i = 0; i < sliceSize; i++)
        {
            _data[offset + i] = slice._data[i];
        }
    }

    public void SetSlice(int dimension, int index, Tensor<T> slice)
    {
        if (dimension < 0 || dimension >= Rank)
            throw new ArgumentOutOfRangeException(nameof(dimension), "Dimension is out of range.");

        if (index < 0 || index >= Shape[dimension])
            throw new ArgumentOutOfRangeException(nameof(index), "Index is out of range for the specified dimension.");

        // Check if the slice shape matches the expected shape
        int[] expectedSliceShape = new int[Rank - 1];
        for (int i = 0, j = 0; i < Rank; i++)
        {
            if (i != dimension)
                expectedSliceShape[j++] = Shape[i];
        }

        if (!slice.Shape.SequenceEqual(expectedSliceShape))
            throw new ArgumentException("Slice shape does not match the expected shape for the given dimension.");

        // Calculate the stride for the specified dimension
        int stride = 1;
        for (int i = dimension + 1; i < Rank; i++)
            stride *= Shape[i];

        // Calculate the starting index in the flat array
        int startIndex = index * stride;
        for (int i = 0; i < dimension; i++)
            startIndex *= Shape[i];

        // Copy the slice data into the tensor
        for (int i = 0; i < slice.Length; i++)
        {
            int targetIndex = startIndex + (i % stride) + i / stride * stride * Shape[dimension];
            _data[targetIndex] = slice._data[i];
        }
    }

    public Tensor<T> Copy()
    {
        var newData = new Vector<T>(_data.Length);
        Array.Copy(_data, newData, _data.Length);

        return new Tensor<T>(Shape, newData);
    }

    public static Tensor<T> ElementwiseMultiply(Tensor<T> a, Tensor<T> b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
        {
            throw new ArgumentException("Tensors must have the same shape for element-wise multiplication.");
        }

        Tensor<T> result = new Tensor<T>(a.Shape);
        for (int i = 0; i < a.Length; i++)
        {
            result._data[i] = NumOps.Multiply(a._data[i], b._data[i]);
        }

        return result;
    }

    public void Fill(T value)
    {
        for (int i = 0; i < _data.Length; i++)
        {
            _data[i] = value;
        }
    }

    public void SetFlatIndex(int flatIndex, T value)
    {
        if (flatIndex < 0 || flatIndex >= _data.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(flatIndex), "Flat index is out of range.");
        }

        _data[flatIndex] = value;
    }

    /// <summary>
    /// Gets the complex value from a tensor at the specified index.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="tensor">The tensor.</param>
    /// <param name="index">The index.</param>
    /// <returns>The complex value at the specified index.</returns>
    public static Complex<T> GetComplex(Tensor<T> tensor, int index)
    {
        var value = tensor[index];
            
        // If the value is already a Complex<T>, return it
        if (value is Complex<T> complex)
        {
            return complex;
        }
            
        // Otherwise, create a new Complex<T> with the value as the real part
        // and zero as the imaginary part
        return new Complex<T>(value, NumOps.Zero);
    }

    public static Tensor<T> CreateRandom(params int[] dimensions)
    {
        if (dimensions == null || dimensions.Length == 0)
            throw new ArgumentException("Dimensions cannot be null or empty.", nameof(dimensions));

        var tensor = new Tensor<T>(dimensions);
        var random = new Random();
        var numOps = MathHelper.GetNumericOperations<T>();

        // Flatten the tensor into a 1D array for easier iteration
        var flattenedSize = dimensions.Aggregate(1, (a, b) => a * b);
        for (int i = 0; i < flattenedSize; i++)
        {
            // Generate a random value between 0 and 1
            var randomValue = numOps.FromDouble(random.NextDouble());
        
            // Calculate the multi-dimensional index
            var index = new int[dimensions.Length];
            var remaining = i;
            for (int j = dimensions.Length - 1; j >= 0; j--)
            {
                index[j] = remaining % dimensions[j];
                remaining /= dimensions[j];
            }

            // Set the random value in the tensor using the indexer
            tensor[index] = randomValue;
        }

        return tensor;
    }

    public Tensor<T> SubTensor(params int[] indices)
    {
        if (indices.Length > Shape.Length)
            throw new ArgumentException("Number of indices exceeds tensor dimensions.");

        int[] newShape = new int[Shape.Length - indices.Length];
        for (int i = 0; i < newShape.Length; i++)
        {
            newShape[i] = Shape[indices.Length + i];
        }

        Tensor<T> subTensor = new Tensor<T>(newShape);

        int[] currentIndices = new int[Shape.Length];
        Array.Copy(indices, currentIndices, indices.Length);

        CopySubTensorData(this, subTensor, currentIndices, indices.Length);

        return subTensor;
    }

    private void CopySubTensorData(Tensor<T> source, Tensor<T> destination, int[] currentIndices, int fixedDimensions)
    {
        if (fixedDimensions == source.Shape.Length)
        {
            destination[new int[0]] = source[currentIndices];
            return;
        }

        for (int i = 0; i < source.Shape[fixedDimensions]; i++)
        {
            currentIndices[fixedDimensions] = i;
            CopySubTensorData(source, destination, currentIndices, fixedDimensions + 1);
        }
    }

    public static Tensor<T> Empty()
    {
        return new Tensor<T>([]);
    }

    public Tensor<T> Slice(int startRow, int startCol, int endRow, int endCol)
    {
        if (this.Rank != 2)
        {
            throw new InvalidOperationException("This Slice method is only applicable for 2D tensors.");
        }

        if (startRow < 0 || startCol < 0 || endRow > this.Shape[0] || endCol > this.Shape[1] || startRow >= endRow || startCol >= endCol)
        {
            throw new ArgumentException("Invalid slice parameters.");
        }

        int newRows = endRow - startRow;
        int newCols = endCol - startCol;
        int[] newShape = new int[] { newRows, newCols };

        Tensor<T> result = new Tensor<T>(newShape);

        for (int i = 0; i < newRows; i++)
        {
            for (int j = 0; j < newCols; j++)
            {
                result[i, j] = this[startRow + i, startCol + j];
            }
        }

        return result;
    }

    public Tensor<T> Slice(int index)
    {
        if (index < 0 || index >= Shape[0])
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        int[] newShape = Shape.Skip(1).ToArray();
        int sliceSize = newShape.Aggregate(1, (a, b) => a * b);
        int offset = index * sliceSize;

        var sliceData = new Vector<T>(sliceSize);
        Array.Copy(_data, offset, sliceData, 0, sliceSize);

        return new Tensor<T>(newShape, sliceData);
    }

    public Tensor<T> Slice(int axis, int start, int? end = null)
    {
        if (axis < 0 || axis >= Rank)
            throw new ArgumentException($"Invalid axis. Must be between 0 and {Rank - 1}.");

        int axisSize = Shape[axis];
        int actualEnd = end ?? axisSize;
        if (start < 0 || start >= axisSize || actualEnd <= start || actualEnd > axisSize)
            throw new ArgumentException("Invalid start or end index for slicing.");

        int sliceSize = actualEnd - start;
        int[] newShape = new int[Rank];
        Array.Copy(Shape, newShape, Rank);
        newShape[axis] = sliceSize;

        Tensor<T> result = new Tensor<T>(newShape);

        int[] sourceIndices = new int[Rank];
        int[] destIndices = new int[Rank];

        void SliceRecursive(int depth)
        {
            if (depth == Rank)
            {
                result[destIndices] = this[sourceIndices];
                return;
            }

            int limit = depth == axis ? sliceSize : Shape[depth];
            for (int i = 0; i < limit; i++)
            {
                sourceIndices[depth] = depth == axis ? i + start : i;
                destIndices[depth] = i;
                SliceRecursive(depth + 1);
            }
        }

        SliceRecursive(0);
        return result;
    }

    public Tensor<T> MeanOverAxis(int axis)
    {
        if (axis < 0 || axis >= Rank)
            throw new ArgumentOutOfRangeException(nameof(axis));

        var newShape = Shape.ToList();
        newShape.RemoveAt(axis);
        var result = new Tensor<T>([.. newShape]);
        int axisSize = Shape[axis];

        // Iterate over all elements, grouping by the non-axis dimensions
        for (int i = 0; i < _data.Length; i += axisSize)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < axisSize; j++)
            {
                sum = NumOps.Add(sum, _data[i + j]);
            }

            result._data[i / axisSize] = NumOps.Divide(sum, NumOps.FromDouble(axisSize));
        }

        return result;
    }

    public Tensor<T> MaxOverAxis(int axis)
    {
        if (axis < 0 || axis >= Rank)
            throw new ArgumentOutOfRangeException(nameof(axis));

        var newShape = Shape.ToList();
        newShape.RemoveAt(axis);
        var result = new Tensor<T>([.. newShape]);
        int axisSize = Shape[axis];

        // Iterate over all elements, grouping by the non-axis dimensions
        for (int i = 0; i < _data.Length; i += axisSize)
        {
            T max = _data[i];
            for (int j = 1; j < axisSize; j++)
            {
                if (NumOps.GreaterThan(_data[i + j], max))
                    max = _data[i + j];
            }

            result._data[i / axisSize] = max;
        }

        return result;
    }

    public Tensor<T> SumOverAxis(int axis)
    {
        if (axis < 0 || axis >= Rank)
            throw new ArgumentOutOfRangeException(nameof(axis));

        var newShape = Shape.ToList();
        newShape.RemoveAt(axis);
        var result = new Tensor<T>([.. newShape]);
        int axisSize = Shape[axis];

        // Iterate over all elements, grouping by the non-axis dimensions
        for (int i = 0; i < _data.Length; i += axisSize)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < axisSize; j++)
            {
                sum = NumOps.Add(sum, _data[i + j]);
            }

            result._data[i / axisSize] = sum;
        }

        return result;
    }

    public void SetSubTensor(int[] indices, Tensor<T> subTensor)
    {
        if (indices.Length != subTensor.Rank)
            throw new ArgumentException("Number of indices must match the rank of the sub-tensor.");

        int[] currentIndices = new int[Rank];
        Array.Copy(indices, currentIndices, indices.Length);

        SetSubTensorRecursive(subTensor, currentIndices, 0);
    }

    private void SetSubTensorRecursive(Tensor<T> subTensor, int[] indices, int dimension)
    {
        if (dimension == subTensor.Rank)
        {
            this[indices] = subTensor._data[0];
            return;
        }

        for (int i = 0; i < subTensor.Shape[dimension]; i++)
        {
            indices[indices.Length - subTensor.Rank + dimension] = i;
            SetSubTensorRecursive(subTensor, indices, dimension + 1);
        }
    }

    public T DotProduct(Tensor<T> other)
    {
        if (!Shape.SequenceEqual(other.Shape))
            throw new ArgumentException("Tensors must have the same shape for dot product.");

        T result = NumOps.Zero;
        for (int i = 0; i < _data.Length; i++)
        {
            result = NumOps.Add(result, NumOps.Multiply(_data[i], other._data[i]));
        }

        return result;
    }

    public Tensor<T> Scale(T factor)
    {
        var result = new Tensor<T>(this.Shape);
    
        // Apply scaling to each element in the tensor
        for (int i = 0; i < this.Length; i++)
        {
            result[i] = NumOps.Multiply(this[i], factor);
        }
    
        return result;
    }

    public void ScaleInPlace(T factor)
    {
        for (int i = 0; i < this.Length; i++)
        {
            this[i] = NumOps.Multiply(this[i], factor);
        }
    }

    public static Tensor<T> Stack(Tensor<T>[] tensors, int axis = 0)
    {
        if (tensors == null || tensors.Length == 0)
            throw new ArgumentException("At least one tensor must be provided for stacking.");

        int rank = tensors[0].Rank;
        if (axis < 0 || axis > rank)
            throw new ArgumentException($"Invalid axis. Must be between 0 and {rank}.");

        // Validate that all tensors have the same shape
        for (int i = 1; i < tensors.Length; i++)
        {
            if (!tensors[i].Shape.SequenceEqual(tensors[0].Shape))
                throw new ArgumentException("All tensors must have the same shape for stacking.");
        }

        // Calculate the new shape
        int[] newShape = new int[rank + 1];
        int shapeIndex = 0;
        for (int i = 0; i <= rank; i++)
        {
            if (i == axis)
            {
                newShape[i] = tensors.Length;
            }
            else
            {
                newShape[i] = tensors[0].Shape[shapeIndex];
                shapeIndex++;
            }
        }

        // Create the new tensor
        Tensor<T> result = new Tensor<T>(newShape);

        // Copy data from input tensors to the result tensor
        int[] indices = new int[rank + 1];
        for (int i = 0; i < tensors.Length; i++)
        {
            indices[axis] = i;
            CopyTensorToStack(tensors[i], result, indices, axis);
        }

        return result;
    }

    private static void CopyTensorToStack(Tensor<T> source, Tensor<T> destination, int[] destIndices, int stackAxis)
    {
        int[] sourceIndices = new int[source.Rank];

        void CopyRecursive(int depth)
        {
            if (depth == source.Rank)
            {
                destination[destIndices] = source[sourceIndices];
                return;
            }

            int destDepth = depth < stackAxis ? depth : depth + 1;
            for (int i = 0; i < source.Shape[depth]; i++)
            {
                sourceIndices[depth] = i;
                destIndices[destDepth] = i;
                CopyRecursive(depth + 1);
            }
        }

        CopyRecursive(0);
    }

    public static Tensor<T> Concatenate(Tensor<T>[] tensors, int axis)
    {
        if (tensors == null || tensors.Length == 0)
            throw new ArgumentException("At least one tensor must be provided for concatenation.");

        int rank = tensors[0].Rank;
        if (axis < 0 || axis >= rank)
            throw new ArgumentException($"Invalid axis. Must be between 0 and {rank - 1}.");

        // Validate that all tensors have the same shape except for the concatenation axis
        for (int i = 1; i < tensors.Length; i++)
        {
            if (tensors[i].Rank != rank)
                throw new ArgumentException("All tensors must have the same rank.");

            for (int j = 0; j < rank; j++)
            {
                if (j != axis && tensors[i].Shape[j] != tensors[0].Shape[j])
                    throw new ArgumentException("All tensors must have the same shape except for the concatenation axis.");
            }
        }

        // Calculate the new shape
        int[] newShape = new int[rank];
        Array.Copy(tensors[0].Shape, newShape, rank);
        for (int i = 1; i < tensors.Length; i++)
        {
            newShape[axis] += tensors[i].Shape[axis];
        }

        // Create the new tensor
        Tensor<T> result = new Tensor<T>(newShape);

        // Copy data from input tensors to the result tensor
        int offset = 0;
        for (int i = 0; i < tensors.Length; i++)
        {
            CopyTensorSlice(tensors[i], result, axis, offset);
            offset += tensors[i].Shape[axis];
        }

        return result;
    }

    private static void CopyTensorSlice(Tensor<T> source, Tensor<T> destination, int axis, int destinationOffset)
    {
        int[] sourceIndices = new int[source.Rank];
        int[] destIndices = new int[destination.Rank];

        void CopyRecursive(int depth)
        {
            if (depth == source.Rank)
            {
                destination[destIndices] = source[sourceIndices];
                return;
            }

            int limit = depth == axis ? source.Shape[depth] : destination.Shape[depth];
            for (int i = 0; i < limit; i++)
            {
                sourceIndices[depth] = i;
                destIndices[depth] = depth == axis ? i + destinationOffset : i;
                CopyRecursive(depth + 1);
            }
        }

        CopyRecursive(0);
    }

    public Tensor<T> GetSlice(int batchIndex)
    {
        int[] newShape = new int[Shape.Length - 1];
        Array.Copy(Shape, 1, newShape, 0, Shape.Length - 1);
    
        Tensor<T> slice = new Tensor<T>(newShape);
    
        int sliceSize = slice.Length;
        Array.Copy(_data, batchIndex * sliceSize, slice._data, 0, sliceSize);
    
        return slice;
    }

    public static Tensor<T> CreateDefault(int[] shape, T value)
    {
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor._data[i] = value;
        }

        return tensor;
    }

    public Tensor<T> Subtract(Tensor<T> other)
    {
        if (!Shape.SequenceEqual(other.Shape))
            throw new ArgumentException("Tensors must have the same shape for subtraction.");

        var result = new Tensor<T>(Shape);
        var ops = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < _data.Length; i++)
        {
            result._data[i] = ops.Subtract(_data[i], other._data[i]);
        }

        return result;
    }

    public Tensor<T> ElementwiseMultiply(Tensor<T> other)
    {
        if (!_dimensions.SequenceEqual(other._dimensions))
            throw new ArgumentException("Tensors must have the same dimensions for element-wise multiplication.");

        Vector<T> result = _data.PointwiseMultiply(other._data);
        return new Tensor<T>(_dimensions, result);
    }

    public Tensor<T> PointwiseMultiply(Tensor<T> other)
    {
        if (this.Shape.SequenceEqual(other.Shape))
        {
            // Simple case: tensors have the same shape
            var result = new Tensor<T>(this.Shape);
            for (int i = 0; i < this.Length; i++)
            {
                result._data[i] = NumOps.Multiply(this._data[i], other._data[i]);
            }
            return result;
        }
        else
        {
            // Handle broadcasting
            return BroadcastPointwiseMultiply(other);
        }
    }

    private Tensor<T> BroadcastPointwiseMultiply(Tensor<T> other)
    {
        int[] broadcastShape = GetBroadcastShape(this.Shape, other.Shape);
        var result = new Tensor<T>(broadcastShape);

        // Create index arrays for both tensors
        int[] thisIndices = new int[this.Rank];
        int[] otherIndices = new int[other.Rank];

        // Iterate over the result tensor
        foreach (var index in result.GetIndices())
        {
            // Map result index to this tensor's index
            for (int i = 0; i < this.Rank; i++)
            {
                thisIndices[i] = this.Shape[i] == 1 ? 0 : index[i];
            }

            // Map result index to other tensor's index
            for (int i = 0; i < other.Rank; i++)
            {
                otherIndices[i] = other.Shape[i] == 1 ? 0 : index[i];
            }

            // Perform multiplication
            result[index] = NumOps.Multiply(this[thisIndices], other[otherIndices]);
        }

        return result;
    }

    private static int[] GetBroadcastShape(int[] shape1, int[] shape2)
    {
        int maxRank = Math.Max(shape1.Length, shape2.Length);
        var broadcastShape = new int[maxRank];

        for (int i = 0; i < maxRank; i++)
        {
            int dim1 = i < shape1.Length ? shape1[shape1.Length - 1 - i] : 1;
            int dim2 = i < shape2.Length ? shape2[shape2.Length - 1 - i] : 1;

            if (dim1 == dim2 || dim1 == 1 || dim2 == 1)
            {
                broadcastShape[maxRank - 1 - i] = Math.Max(dim1, dim2);
            }
            else
            {
                throw new ArgumentException("Tensors cannot be broadcast to a single shape.");
            }
        }

        return broadcastShape;
    }

    private IEnumerable<int[]> GetIndices()
    {
        int[] index = new int[this.Rank];
        int totalElements = this.Length;

        for (int i = 0; i < totalElements; i++)
        {
            yield return index;

            // Update index
            for (int j = this.Rank - 1; j >= 0; j--)
            {
                if (++index[j] < this.Shape[j])
                    break;
                index[j] = 0;
            }
        }
    }

    public Tensor<T> ElementwiseSubtract(Tensor<T> other)
    {
        if (!Shape.SequenceEqual(other.Shape))
            throw new ArgumentException("Tensors must have the same shape for elementwise subtraction.");

        var result = new Tensor<T>(Shape);
        for (int i = 0; i < _data.Length; i++)
        {
            result._data[i] = NumOps.Subtract(_data[i], other._data[i]);
        }

        return result;
    }

    public Tensor<T> ElementwiseSubtract(T scalar)
    {
        var result = new Tensor<T>(Shape);
        for (int i = 0; i < _data.Length; i++)
        {
            result._data[i] = NumOps.Subtract(_data[i], scalar);
        }

        return result;
    }

    public Tensor<T> Transform(Func<T, int, T> transformer)
    {
        var result = new Vector<T>(_data.Length);
        for (int i = 0; i < _data.Length; i++)
        {
            result[i] = transformer(_data[i], i);
        }

        return new Tensor<T>(Shape, result);
    }

    public Tensor<T> Add(Vector<T> vector)
    {
        if (this.Rank != 3 || this.Shape[2] != vector.Length)
            throw new ArgumentException("Vector length must match the last dimension of the tensor.");

        var result = new Tensor<T>(this.Shape);
        for (int i = 0; i < this.Shape[0]; i++)
        {
            for (int j = 0; j < this.Shape[1]; j++)
            {
                for (int k = 0; k < this.Shape[2]; k++)
                {
                    result[i, j, k] = NumOps.Add(this[i, j, k], vector[k]);
                }
            }
        }

        return result;
    }

    public Tensor<T> Multiply(Matrix<T> matrix)
    {
        if (this.Rank != 3 || this.Shape[2] != matrix.Rows)
            throw new ArgumentException("Matrix rows must match the last dimension of the tensor.");

        var result = new Tensor<T>([this.Shape[0], this.Shape[1], matrix.Columns]);
        for (int i = 0; i < this.Shape[0]; i++)
        {
            for (int j = 0; j < this.Shape[1]; j++)
            {
                for (int k = 0; k < matrix.Columns; k++)
                {
                    T sum = NumOps.Zero;
                    for (int l = 0; l < this.Shape[2]; l++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(this[i, j, l], matrix[l, k]));
                    }

                    result[i, j, k] = sum;
                }
            }
        }

        return result;
    }

    public Tensor<T> MatrixMultiply(Tensor<T> other)
    {
        if (this.Rank != 2 || other.Rank != 2)
        {
            throw new ArgumentException("MatMul is only defined for 2D tensors (matrices).");
        }

        if (this.Shape[1] != other.Shape[0])
        {
            throw new ArgumentException("Incompatible matrix dimensions for multiplication.");
        }

        return this.Multiply(other);
    }

    public Tensor<T> Multiply(Tensor<T> other)
    {
        // Check if shapes are compatible for multiplication
        if (!AreShapesMultiplicationCompatible(_dimensions, other._dimensions))
            throw new ArgumentException("Tensor shapes are not compatible for multiplication.");

        // Determine the output shape
        int[] outputShape = GetOutputShape(_dimensions, other._dimensions);

        // Create the result tensor
        Tensor<T> result = new Tensor<T>(outputShape);

        // Perform the multiplication
        MultiplyTensors(this, other, result);

        return result;
    }

    public Tensor<T> Transpose(int[] permutation)
    {
        if (permutation.Length != Rank)
            throw new ArgumentException("Permutation array length must match tensor rank.");

        if (!permutation.OrderBy(x => x).SequenceEqual(Enumerable.Range(0, Rank)))
            throw new ArgumentException("Invalid permutation array.");

        int[] newShape = new int[Rank];
        for (int i = 0; i < Rank; i++)
        {
            newShape[i] = Shape[permutation[i]];
        }

        Tensor<T> result = new Tensor<T>(newShape);

        int[] oldIndices = new int[Rank];
        int[] newIndices = new int[Rank];

        for (int i = 0; i < Length; i++)
        {
            GetIndicesFromFlatIndex(i, Shape, oldIndices);
            for (int j = 0; j < Rank; j++)
            {
                newIndices[j] = oldIndices[permutation[j]];
            }

            result[newIndices] = this[oldIndices];
        }

        return result;
    }

    private void GetIndicesFromFlatIndex(int flatIndex, int[] shape, int[] indices)
    {
        for (int i = Rank - 1; i >= 0; i--)
        {
            indices[i] = flatIndex % shape[i];
            flatIndex /= shape[i];
        }
    }

    public Matrix<T> ToMatrix()
    {
        if (Rank != 2)
        {
            throw new InvalidOperationException("Tensor must be 2-dimensional to convert to Matrix.");
        }

        var matrix = new Matrix<T>(Shape[0], Shape[1]);
        for (int i = 0; i < Shape[0]; i++)
        {
            for (int j = 0; j < Shape[1]; j++)
            {
                matrix[i, j] = this[i, j];
            }
        }

        return matrix;
    }

    public Tensor<T> Sum(int[]? axes = null)
    {
        if (axes == null || axes.Length == 0)
        {
            // Sum all elements
            T sum = NumOps.Zero;
            for (int i = 0; i < Length; i++)
            {
                sum = NumOps.Add(sum, _data[i]);
            }
            return new Tensor<T>([1], new Vector<T>(new[] { sum }));
        }

        axes = axes.OrderBy(x => x).ToArray();
        int[] newShape = new int[Rank - axes.Length];
        int newIndex = 0;

        for (int i = 0; i < Rank; i++)
        {
            if (!axes.Contains(i))
            {
                newShape[newIndex++] = Shape[i];
            }
        }

        Tensor<T> result = new Tensor<T>(newShape);
        int[] indices = new int[Rank];
        SumRecursive(this, result, axes, indices, 0, NumOps.Zero);

        return result;
    }

    private void SumRecursive(Tensor<T> input, Tensor<T> result, int[] axes, int[] indices, int depth, T currentSum)
    {
        if (depth == Rank)
        {
            int[] resultIndices = new int[result.Rank];
            int resultIndex = 0;
            for (int i = 0; i < Rank; i++)
            {
                if (!axes.Contains(i))
                {
                    resultIndices[resultIndex++] = indices[i];
                }
            }
            result[resultIndices] = NumOps.Add(result[resultIndices], currentSum);
            return;
        }

        if (axes.Contains(depth))
        {
            for (int i = 0; i < Shape[depth]; i++)
            {
                indices[depth] = i;
                SumRecursive(input, result, axes, indices, depth + 1, NumOps.Add(currentSum, this[indices]));
            }
        }
        else
        {
            for (int i = 0; i < Shape[depth]; i++)
            {
                indices[depth] = i;
                SumRecursive(input, result, axes, indices, depth + 1, currentSum);
            }
        }
    }

    private static bool AreShapesMultiplicationCompatible(int[] shape1, int[] shape2)
    {
        int rank1 = shape1.Length;
        int rank2 = shape2.Length;
        int maxRank = Math.Max(rank1, rank2);

        for (int i = 0; i < maxRank; i++)
        {
            int dim1 = i < rank1 ? shape1[rank1 - 1 - i] : 1;
            int dim2 = i < rank2 ? shape2[rank2 - 1 - i] : 1;

            if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
                return false;
        }

        return true;
    }

    private static int[] GetOutputShape(int[] shape1, int[] shape2)
    {
        int maxRank = Math.Max(shape1.Length, shape2.Length);
        int[] outputShape = new int[maxRank];

        for (int i = 0; i < maxRank; i++)
        {
            int dim1 = i < shape1.Length ? shape1[shape1.Length - 1 - i] : 1;
            int dim2 = i < shape2.Length ? shape2[shape2.Length - 1 - i] : 1;
            outputShape[maxRank - 1 - i] = Math.Max(dim1, dim2);
        }

        return outputShape;
    }

    public Vector<T> GetVector(int index)
    {
        if (Shape.Length < 2)
            throw new InvalidOperationException("Tensor must have at least 2 dimensions to get a vector.");

        int vectorSize = Shape[1];
        var vector = new Vector<T>(vectorSize);
        for (int i = 0; i < vectorSize; i++)
        {
            vector[i] = this[index, i];
        }

        return vector;
    }

    public void SetRow(int rowIndex, Vector<T> vector)
    {
        if (Shape.Length < 2)
            throw new InvalidOperationException("Tensor must have at least 2 dimensions to set a row.");

        if (vector.Length != Shape[1])
            throw new ArgumentException("Vector length must match the second dimension of the tensor.");

        for (int i = 0; i < vector.Length; i++)
        {
            this[rowIndex, i] = vector[i];
        }
    }

    public void SetColumn(int columnIndex, Vector<T> vector)
    {
        if (Shape.Length < 2)
            throw new InvalidOperationException("Tensor must have at least 2 dimensions to set a column.");

        if (vector.Length != Shape[0])
            throw new ArgumentException("Vector length must match the first dimension of the tensor.");

        for (int i = 0; i < vector.Length; i++)
        {
            this[i, columnIndex] = vector[i];
        }
    }

    public Vector<T> GetColumn(int columnIndex)
    {
        if (Shape.Length < 2)
            throw new InvalidOperationException("Tensor must have at least 2 dimensions to get a column.");

        if (columnIndex < 0 || columnIndex >= Shape[1])
            throw new ArgumentOutOfRangeException(nameof(columnIndex), "Column index is out of range.");

        int columnLength = Shape[0];
        Vector<T> column = new Vector<T>(columnLength);

        int stride = Shape[1];
        for (int i = 0; i < columnLength; i++)
        {
            column[i] = _data[i * stride + columnIndex];
        }

        return column;
    }

    public Vector<T> GetRow(int rowIndex)
    {
        if (rowIndex < 0 || rowIndex >= Shape[0])
        {
            throw new ArgumentOutOfRangeException(nameof(rowIndex), "Row index is out of range.");
        }

        int rowLength = 1;
        for (int i = 1; i < Shape.Length; i++)
        {
            rowLength *= Shape[i];
        }

        Vector<T> row = new Vector<T>(rowLength);
        int startIndex = rowIndex * rowLength;

        for (int i = 0; i < rowLength; i++)
        {
            row[i] = _data[startIndex + i];
        }

        return row;
    }

    public void SetVector(int index, Vector<T> vector)
    {
        if (Shape.Length < 2)
            throw new InvalidOperationException("Tensor must have at least 2 dimensions to set a vector.");

        if (vector.Length != Shape[1])
            throw new ArgumentException("Vector length must match the second dimension of the tensor.");

        for (int i = 0; i < vector.Length; i++)
        {
            this[index, i] = vector[i];
        }
    }

    public Tensor<T> Reshape(params int[] newShape)
    {
        if (newShape.Aggregate(1, (a, b) => a * b) != Length)
            throw new ArgumentException("New shape must have the same total number of elements as the original tensor.");

        var reshaped = new Tensor<T>(newShape);
        for (int i = 0; i < Length; i++)
        {
            reshaped._data[i] = _data[i];
        }

        return reshaped;
    }

    private static void MultiplyTensors(Tensor<T> a, Tensor<T> b, Tensor<T> result)
    {
        int[] indices = new int[result.Rank];
        MultiplyTensorsRecursive(a, b, result, indices, 0);
    }

    private static void MultiplyTensorsRecursive(Tensor<T> a, Tensor<T> b, Tensor<T> result, int[] indices, int depth)
    {
        if (depth == result.Rank)
        {
            result[indices] = NumOps.Multiply(a[indices], b[indices]);
            return;
        }

        for (int i = 0; i < result.Shape[depth]; i++)
        {
            indices[depth] = i;
            MultiplyTensorsRecursive(a, b, result, indices, depth + 1);
        }
    }

    public Tensor<T> GetSubTensor(int batch, int channel, int startHeight, int startWidth, int height, int width)
    {
        if (batch < 0 || batch >= Shape[0]) throw new ArgumentOutOfRangeException(nameof(batch));
        if (channel < 0 || channel >= Shape[1]) throw new ArgumentOutOfRangeException(nameof(channel));
        if (startHeight < 0 || startHeight + height > Shape[2]) throw new ArgumentOutOfRangeException(nameof(startHeight));
        if (startWidth < 0 || startWidth + width > Shape[3]) throw new ArgumentOutOfRangeException(nameof(startWidth));

        var subTensor = new Tensor<T>([1, 1, height, width]);

        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                subTensor[0, 0, h, w] = this[batch, channel, startHeight + h, startWidth + w];
            }
        }

        return subTensor;
    }

    public (T maxVal, int maxIndex) Max()
    {
        T maxVal = _data[0];
        int maxIndex = 0;

        for (int i = 1; i < _data.Length; i++)
        {
            if (NumOps.GreaterThan(_data[i], maxVal))
            {
                maxVal = _data[i];
                maxIndex = i;
            }
        }

        return (maxVal, maxIndex);
    }

    public T Mean()
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < _data.Length; i++)
        {
            sum = NumOps.Add(sum, _data[i]);
        }

        return NumOps.Divide(sum, NumOps.FromDouble(_data.Length));
    }

    public T GetFlatIndexValue(int flatIndex)
    {
        int[] indices = new int[Rank];
        GetIndicesFromFlatIndex(flatIndex, _dimensions, indices);
        return this[indices];
    }

    public void SetFlatIndexValue(int flatIndex, T value)
    {
        int[] indices = new int[Rank];
        GetIndicesFromFlatIndex(flatIndex, _dimensions, indices);
        this[indices] = value;
    }

    public int GetFlatIndex(int[] indices)
    {
        if (indices.Length != Rank)
            throw new ArgumentException("Number of indices must match tensor rank.");

        int flatIndex = 0;
        int stride = 1;

        for (int i = Rank - 1; i >= 0; i--)
        {
            if (indices[i] < 0 || indices[i] >= _dimensions[i])
                throw new IndexOutOfRangeException($"Index {indices[i]} is out of range for dimension {i}.");

            flatIndex += indices[i] * stride;
            stride *= _dimensions[i];
        }

        return flatIndex;
    }

    public Tensor<T> Add(Tensor<T> other)
    {
        if (!Shape.SequenceEqual(other.Shape))
            throw new ArgumentException("Tensors must have the same shape for addition.");

        return new Tensor<T>(Shape, _data.Add(other._data));
    }

    public Tensor<T> Multiply(T scalar)
    {
        return new Tensor<T>(Shape, _data.Multiply(scalar));
    }

    public Vector<T> ToVector()
    {
        if (Rank != 1)
            throw new InvalidOperationException("Can only convert rank-1 tensors to vectors.");

        return _data;
    }

    public static Tensor<T> FromVector(Vector<T> vector)
    {
        return new Tensor<T>([vector.Length], vector);
    }

    public static Tensor<T> FromMatrix(Matrix<T> matrix)
    {
        return new Tensor<T>([matrix.Rows, matrix.Columns], matrix.ToColumnVector());
    }
}