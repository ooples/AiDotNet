using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Helpers;

/// <summary>
/// Im2col / col2im helpers for 3D convolution. Reformulates Conv3D as a matrix
/// multiply, which is dramatically faster than the engine's per-output-element
/// kernel because the inner GEMM hits the heavily-optimized BLAS path in
/// <c>Engine.TensorMatMul</c> (SIMD, multi-threaded, cache-blocked) instead of
/// the seven-loop direct convolution. The tradeoff is one buffer allocation
/// of size <c>[B·OD·OH·OW, CI·KD·KH·KW]</c> per forward — for the VoxelCNN
/// shapes that is ~30 MB at fp64 worst case, well under memory limits.
/// </summary>
/// <remarks>
/// <para>
/// The im2col mapping for 3D conv with input <c>x[B, CI, ID, IH, IW]</c>,
/// kernel <c>k[CO, CI, KD, KH, KW]</c>, stride <c>s</c>, padding <c>p</c>:
/// </para>
/// <para>
/// For each output position <c>(b, od, oh, ow)</c>, the receptive field is the
/// <c>[CI, KD, KH, KW]</c> block of input starting at
/// <c>(b, :, od·s − p, oh·s − p, ow·s − p)</c>. Im2col flattens that block
/// into a row of the matrix <c>M</c>:
/// </para>
/// <para>
/// <c>M[row, col] = x[b, ci, od·s − p + kd, oh·s − p + kh, ow·s − p + kw]</c>
/// </para>
/// <para>
/// where <c>row = b·OD·OH·OW + od·OH·OW + oh·OW + ow</c> and
/// <c>col = ci·KD·KH·KW + kd·KH·KW + kh·KW + kw</c>. Out-of-bounds positions
/// (from padding) contribute zero.
/// </para>
/// <para>
/// The conv output <c>y[B, CO, OD, OH, OW]</c> is then
/// <c>Y_flat = M · K_flat_T</c> where <c>K_flat</c> is <c>k</c> reshaped to
/// <c>[CO, CI·KD·KH·KW]</c> — a standard GEMM.
/// </para>
/// <para>
/// Col2im is the transpose of im2col: given <c>∂L/∂M[row, col]</c>, scatter it
/// into <c>∂L/∂x</c> at the same input position. Multiple <c>(row, col)</c>
/// pairs map to the same input element when receptive fields overlap, so
/// col2im sums (scatter-add).
/// </para>
/// </remarks>
internal static class Im2Col3DHelper
{
    /// <summary>
    /// Fills <paramref name="m"/> from <paramref name="x"/> per the im2col 3D mapping
    /// described in the class summary. <paramref name="x"/> must be rank-5
    /// <c>[B, CI, ID, IH, IW]</c>; <paramref name="m"/> must be rank-2
    /// <c>[B·OD·OH·OW, CI·KD·KH·KW]</c>. Out-of-bounds positions (from padding)
    /// contribute zero.
    /// </summary>
    public static void Im2Col3D<T>(
        Tensor<T> x, Tensor<T> m,
        int kernelSize, int stride, int padding)
    {
        int b = x.Shape[0];
        int ci = x.Shape[1];
        int id = x.Shape[2];
        int ih = x.Shape[3];
        int iw = x.Shape[4];

        int kd = kernelSize, kh = kernelSize, kw = kernelSize;
        int od = (id + 2 * padding - kd) / stride + 1;
        int oh = (ih + 2 * padding - kh) / stride + 1;
        int ow = (iw + 2 * padding - kw) / stride + 1;

        int colsPerRow = ci * kd * kh * kw;
        int rowsTotal = b * od * oh * ow;

        if (m.Shape[0] != rowsTotal || m.Shape[1] != colsPerRow)
            throw new ArgumentException(
                $"Im2Col3D output shape mismatch: expected [{rowsTotal}, {colsPerRow}], got [{m.Shape[0]}, {m.Shape[1]}].");

        // Access the raw backing vectors through AiDotNet.Tensors' friend-assembly API.
        // Views share this storage, so all indexing below honors their physical strides
        // and offsets instead of using detached logical copies.
        var xData = x.DataVector.GetDataArray();
        var mData = m.DataVector.GetDataArray();
        int[] xStrides = x.Strides.ToArray();
        int[] mStrides = m.Strides.ToArray();
        long xOffset = x.LogicalToStorageIndex(0);
        long mOffset = m.LogicalToStorageIndex(0);

        // Zero-fill only logical destination elements so views cannot clear unrelated storage.
        for (int row = 0; row < rowsTotal; row++)
        {
            long mRow = mOffset + (long)row * mStrides[0];
            for (int col = 0; col < colsPerRow; col++)
                mData[mRow + (long)col * mStrides[1]] = default!;
        }

        // For each output spatial position, copy the receptive field into the
        // corresponding row of m. Parallelizing over the batch+depth axis
        // gives good multi-thread scaling without false-sharing on the row
        // writes (each iteration owns a contiguous oh·ow·colsPerRow chunk).
        Parallel.For(0, b * od, (bOdIdx) =>
        {
            int bi = bOdIdx / od;
            int ood = bOdIdx % od;

            long xBaseB = xOffset + (long)bi * xStrides[0];
            long rowBase = ((long)bi * od + ood) * oh * ow;
            int srcD = ood * stride - padding;

            for (int ooh = 0; ooh < oh; ooh++)
            {
                int srcH = ooh * stride - padding;
                for (int oow = 0; oow < ow; oow++)
                {
                    int srcW = oow * stride - padding;
                    long row = rowBase + (long)ooh * ow + oow;
                    long mRow = mOffset + row * mStrides[0];

                    // Walk the receptive field.
                    long col = 0;
                    for (int cci = 0; cci < ci; cci++)
                    {
                        long xBaseCi = xBaseB + (long)cci * xStrides[1];
                        for (int dd = 0; dd < kd; dd++)
                        {
                            int srcDd = srcD + dd;
                            if (srcDd < 0 || srcDd >= id)
                            {
                                // Whole [kh, kw] block is zero → advance col.
                                col += (long)kh * kw;
                                continue;
                            }
                            long xBaseD = xBaseCi + (long)srcDd * xStrides[2];
                            for (int hh = 0; hh < kh; hh++)
                            {
                                int srcHh = srcH + hh;
                                if (srcHh < 0 || srcHh >= ih)
                                {
                                    col += kw;
                                    continue;
                                }
                                long xBaseH = xBaseD + (long)srcHh * xStrides[3];
                                for (int ww = 0; ww < kw; ww++)
                                {
                                    int srcWw = srcW + ww;
                                    if (srcWw >= 0 && srcWw < iw)
                                    {
                                        long mIndex = mRow + col * mStrides[1];
                                        long xIndex = xBaseH + (long)srcWw * xStrides[4];
                                        mData[mIndex] = xData[xIndex];
                                    }
                                    col++;
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    /// <summary>
    /// Scatters <paramref name="m"/> into <paramref name="x"/> per the col2im 3D
    /// mapping — the transpose of Im2Col3D. Used in the backward pass: given
    /// <c>∂L/∂M</c>, produce <c>∂L/∂x</c>. Overlapping receptive fields
    /// accumulate (scatter-add).
    /// </summary>
    public static void Col2Im3D<T>(
        Tensor<T> m, Tensor<T> x,
        int kernelSize, int stride, int padding)
    {
        int b = x.Shape[0];
        int ci = x.Shape[1];
        int id = x.Shape[2];
        int ih = x.Shape[3];
        int iw = x.Shape[4];

        int kd = kernelSize, kh = kernelSize, kw = kernelSize;
        int od = (id + 2 * padding - kd) / stride + 1;
        int oh = (ih + 2 * padding - kh) / stride + 1;
        int ow = (iw + 2 * padding - kw) / stride + 1;

        int colsPerRow = ci * kd * kh * kw;
        int rowsTotal = b * od * oh * ow;

        if (m.Shape[0] != rowsTotal || m.Shape[1] != colsPerRow)
            throw new ArgumentException(
                $"Col2Im3D gradient shape mismatch: expected [{rowsTotal}, {colsPerRow}], got [{m.Shape[0]}, {m.Shape[1]}].");

        var xData = x.DataVector.GetDataArray();
        var mData = m.DataVector.GetDataArray();
        int[] xStrides = x.Strides.ToArray();
        int[] mStrides = m.Strides.ToArray();
        long xOffset = x.LogicalToStorageIndex(0);
        long mOffset = m.LogicalToStorageIndex(0);

        // Clear only logical destination elements; x may be a view over larger storage.
        for (int bi = 0; bi < b; bi++)
        {
            long xBaseB = xOffset + (long)bi * xStrides[0];
            for (int cci = 0; cci < ci; cci++)
            {
                long xBaseCi = xBaseB + (long)cci * xStrides[1];
                for (int dd = 0; dd < id; dd++)
                {
                    long xBaseD = xBaseCi + (long)dd * xStrides[2];
                    for (int hh = 0; hh < ih; hh++)
                    {
                        long xBaseH = xBaseD + (long)hh * xStrides[3];
                        for (int ww = 0; ww < iw; ww++)
                            xData[xBaseH + (long)ww * xStrides[4]] = default!;
                    }
                }
            }
        }

        var numOps = MathHelper.GetNumericOperations<T>();

        // Parallelism scheme: parallelize over batches. Within a batch, the
        // receptive fields can overlap so a parallel-over-output-positions
        // scheme would race on the input gradient. The per-batch slices are
        // disjoint in the input gradient buffer, so parallel-over-batch is
        // race-free without locks.
        Parallel.For(0, b, (bi) =>
        {
            long xBaseB = xOffset + (long)bi * xStrides[0];
            for (int ood = 0; ood < od; ood++)
            {
                int dstD = ood * stride - padding;
                for (int ooh = 0; ooh < oh; ooh++)
                {
                    int dstH = ooh * stride - padding;
                    for (int oow = 0; oow < ow; oow++)
                    {
                        int dstW = oow * stride - padding;
                        long row = (long)bi * od * oh * ow + (long)ood * oh * ow + (long)ooh * ow + oow;
                        long mRow = mOffset + row * mStrides[0];

                        long col = 0;
                        for (int cci = 0; cci < ci; cci++)
                        {
                            long xBaseCi = xBaseB + (long)cci * xStrides[1];
                            for (int dd = 0; dd < kd; dd++)
                            {
                                int srcDd = dstD + dd;
                                if (srcDd < 0 || srcDd >= id)
                                {
                                    col += (long)kh * kw;
                                    continue;
                                }
                                long xBaseD = xBaseCi + (long)srcDd * xStrides[2];
                                for (int hh = 0; hh < kh; hh++)
                                {
                                    int srcHh = dstH + hh;
                                    if (srcHh < 0 || srcHh >= ih)
                                    {
                                        col += kw;
                                        continue;
                                    }
                                    long xBaseH = xBaseD + (long)srcHh * xStrides[3];
                                    for (int ww = 0; ww < kw; ww++)
                                    {
                                        int srcWw = dstW + ww;
                                        if (srcWw >= 0 && srcWw < iw)
                                        {
                                            long xIdx = xBaseH + (long)srcWw * xStrides[4];
                                            long mIdx = mRow + col * mStrides[1];
                                            xData[xIdx] = numOps.Add(xData[xIdx], mData[mIdx]);
                                        }
                                        col++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
    }

}
