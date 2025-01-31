namespace AiDotNet.LinearAlgebra;

public class Tensor<T>
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

    public Tensor<T> ElementwiseMultiply(Tensor<T> other)
    {
        if (!_dimensions.SequenceEqual(other._dimensions))
            throw new ArgumentException("Tensors must have the same dimensions for element-wise multiplication.");

        Vector<T> result = _data.PointwiseMultiply(other._data);
        return new Tensor<T>(_dimensions, result);
    }

    public Tensor<T> Transform(Func<T, T> function)
    {
        Vector<T> transformedData = _data.Transform(function);
        return new Tensor<T>(_dimensions, transformedData);
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

    private int GetFlatIndex(int[] indices)
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

    public Matrix<T> ToMatrix()
    {
        if (Rank != 2)
            throw new InvalidOperationException("Can only convert rank-2 tensors to matrices.");

        return Matrix<T>.CreateFromVector(_data);
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