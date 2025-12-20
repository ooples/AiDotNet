using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Inference.Quantization;

internal static class Int8WeightOnlyQuantization
{
    internal readonly struct QuantizedWeights
    {
        public QuantizedWeights(sbyte[] weights, float[] scales, int rows, int cols)
        {
            Weights = weights;
            Scales = scales;
            Rows = rows;
            Cols = cols;
        }

        public sbyte[] Weights { get; }
        public float[] Scales { get; }
        public int Rows { get; }
        public int Cols { get; }
    }

    public static QuantizedWeights QuantizePerRow(Tensor<float> weights)
    {
        if (weights.Rank != 2)
            throw new ArgumentException("Expected 2D weight tensor.", nameof(weights));

        int rows = weights.Shape[0];
        int cols = weights.Shape[1];

        var q = new sbyte[rows * cols];
        var scales = new float[rows];

        for (int r = 0; r < rows; r++)
        {
            float maxAbs = 0f;
            int baseIdx = r * cols;
            for (int c = 0; c < cols; c++)
            {
                float v = weights[r, c];
                float av = MathF.Abs(v);
                if (av > maxAbs)
                    maxAbs = av;
            }

            float scale = maxAbs > 0f ? (maxAbs / 127f) : 1f;
            scales[r] = scale;

            float inv = 1f / scale;
            for (int c = 0; c < cols; c++)
            {
                float v = weights[r, c] * inv;
                int qi = (int)MathF.Round(v);
                if (qi > 127) qi = 127;
                if (qi < -127) qi = -127;
                q[baseIdx + c] = (sbyte)qi;
            }
        }

        return new QuantizedWeights(q, scales, rows, cols);
    }

    public static QuantizedWeights QuantizePerRow(ReadOnlySpan<float> weights, int rows, int cols)
    {
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (cols <= 0) throw new ArgumentOutOfRangeException(nameof(cols));
        if (weights.Length < rows * cols) throw new ArgumentException("Weight span too small for given dimensions.", nameof(weights));

        var q = new sbyte[rows * cols];
        var scales = new float[rows];

        for (int r = 0; r < rows; r++)
        {
            float maxAbs = 0f;
            int baseIdx = r * cols;
            for (int c = 0; c < cols; c++)
            {
                float v = weights[baseIdx + c];
                float av = MathF.Abs(v);
                if (av > maxAbs)
                    maxAbs = av;
            }

            float scale = maxAbs > 0f ? (maxAbs / 127f) : 1f;
            scales[r] = scale;

            float inv = 1f / scale;
            for (int c = 0; c < cols; c++)
            {
                float v = weights[baseIdx + c] * inv;
                int qi = (int)MathF.Round(v);
                if (qi > 127) qi = 127;
                if (qi < -127) qi = -127;
                q[baseIdx + c] = (sbyte)qi;
            }
        }

        return new QuantizedWeights(q, scales, rows, cols);
    }
}
