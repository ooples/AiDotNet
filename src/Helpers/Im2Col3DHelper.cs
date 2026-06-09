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

        // x is laid out [B, CI, ID, IH, IW] row-major (NCDHW). Stride per axis:
        //   stride_b   = CI · ID · IH · IW
        //   stride_ci  = ID · IH · IW
        //   stride_id  = IH · IW
        //   stride_ih  = IW
        //   stride_iw  = 1
        long stride_b = (long)ci * id * ih * iw;
        long stride_ci = (long)id * ih * iw;
        long stride_id = (long)ih * iw;
        long stride_ih = iw;

        // m is row-major [rowsTotal, colsPerRow]. Stride per axis:
        //   stride_row = colsPerRow
        //   stride_col = 1
        var xData = GetDataArray(x);
        var mData = GetDataArray(m);
        long mLen = (long)rowsTotal * colsPerRow;
        // Zero-fill m first — out-of-bounds positions stay zero.
        Array.Clear(mData, 0, (int)Math.Min(mLen, int.MaxValue));

        // For each output spatial position, copy the receptive field into the
        // corresponding row of m. Parallelizing over the batch+depth axis
        // gives good multi-thread scaling without false-sharing on the row
        // writes (each iteration owns a contiguous oh·ow·colsPerRow chunk).
        Parallel.For(0, b * od, (bOdIdx) =>
        {
            int bi = bOdIdx / od;
            int ood = bOdIdx % od;

            long xBaseB = bi * stride_b;
            long mBaseB = ((long)bi * od + ood) * (long)oh * ow * colsPerRow;
            int srcD = ood * stride - padding;

            for (int ooh = 0; ooh < oh; ooh++)
            {
                int srcH = ooh * stride - padding;
                for (int oow = 0; oow < ow; oow++)
                {
                    int srcW = oow * stride - padding;
                    long mRow = mBaseB + ((long)ooh * ow + oow) * colsPerRow;

                    // Walk the receptive field.
                    long col = 0;
                    for (int cci = 0; cci < ci; cci++)
                    {
                        long xBaseCi = xBaseB + cci * stride_ci;
                        for (int dd = 0; dd < kd; dd++)
                        {
                            int srcDd = srcD + dd;
                            if (srcDd < 0 || srcDd >= id)
                            {
                                // Whole [kh, kw] block is zero → advance col.
                                col += (long)kh * kw;
                                continue;
                            }
                            long xBaseD = xBaseCi + srcDd * stride_id;
                            for (int hh = 0; hh < kh; hh++)
                            {
                                int srcHh = srcH + hh;
                                if (srcHh < 0 || srcHh >= ih)
                                {
                                    col += kw;
                                    continue;
                                }
                                long xBaseH = xBaseD + srcHh * stride_ih;
                                for (int ww = 0; ww < kw; ww++)
                                {
                                    int srcWw = srcW + ww;
                                    if (srcWw >= 0 && srcWw < iw)
                                    {
                                        mData[mRow + col] = xData[xBaseH + srcWw];
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

        long stride_b = (long)ci * id * ih * iw;
        long stride_ci = (long)id * ih * iw;
        long stride_id = (long)ih * iw;
        long stride_ih = iw;

        var xData = GetDataArray(x);
        var mData = GetDataArray(m);
        long xLen = (long)b * stride_b;
        Array.Clear(xData, 0, (int)Math.Min(xLen, int.MaxValue));

        var numOps = MathHelper.GetNumericOperations<T>();

        // Parallelism scheme: parallelize over batches. Within a batch, the
        // receptive fields can overlap so a parallel-over-output-positions
        // scheme would race on the input gradient. The per-batch slices are
        // disjoint in the input gradient buffer, so parallel-over-batch is
        // race-free without locks.
        Parallel.For(0, b, (bi) =>
        {
            long xBaseB = bi * stride_b;
            for (int ood = 0; ood < od; ood++)
            {
                int dstD = ood * stride - padding;
                for (int ooh = 0; ooh < oh; ooh++)
                {
                    int dstH = ooh * stride - padding;
                    for (int oow = 0; oow < ow; oow++)
                    {
                        int dstW = oow * stride - padding;
                        long mRow = ((long)bi * od * oh * ow + (long)ood * oh * ow + (long)ooh * ow + oow) * colsPerRow;

                        long col = 0;
                        for (int cci = 0; cci < ci; cci++)
                        {
                            long xBaseCi = xBaseB + cci * stride_ci;
                            for (int dd = 0; dd < kd; dd++)
                            {
                                int srcDd = dstD + dd;
                                if (srcDd < 0 || srcDd >= id)
                                {
                                    col += (long)kh * kw;
                                    continue;
                                }
                                long xBaseD = xBaseCi + srcDd * stride_id;
                                for (int hh = 0; hh < kh; hh++)
                                {
                                    int srcHh = dstH + hh;
                                    if (srcHh < 0 || srcHh >= ih)
                                    {
                                        col += kw;
                                        continue;
                                    }
                                    long xBaseH = xBaseD + srcHh * stride_ih;
                                    for (int ww = 0; ww < kw; ww++)
                                    {
                                        int srcWw = dstW + ww;
                                        if (srcWw >= 0 && srcWw < iw)
                                        {
                                            long xIdx = xBaseH + srcWw;
                                            xData[xIdx] = numOps.Add(xData[xIdx], mData[mRow + col]);
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

    /// <summary>
    /// Extracts the contiguous backing array from a row-major Tensor. The
    /// AiDotNet.Tensors contract guarantees the storage is a contiguous
    /// T[] in row-major order; we use raw array indexing for the hot loops
    /// because the per-element <c>Tensor[i]</c> accessor goes through a
    /// virtual <c>NumericOperations</c> dispatch that is unacceptable in a
    /// 32k+ iteration inner loop.
    /// </summary>
    private static T[] GetDataArray<T>(Tensor<T> t)
    {
        var span = t.Data.Span;
        // Tensor<T>.Data is a Memory<T>; the backing storage is always a
        // managed T[] for AiDotNet.Tensors allocations. Recovering the
        // array via MemoryMarshal.TryGetArray is the documented
        // zero-copy path.
        if (System.Runtime.InteropServices.MemoryMarshal.TryGetArray<T>(t.Data, out var seg)
            && seg.Array is not null && seg.Offset == 0 && seg.Count == t.Length)
        {
            return seg.Array;
        }
        // Fallback: copy out. This shouldn't fire for AiDotNet.Tensors
        // allocations but it keeps the helper defensible against future
        // backing-store changes (e.g. native-buffer-backed tensors).
        var copy = new T[t.Length];
        span.CopyTo(copy);
        return copy;
    }
}
