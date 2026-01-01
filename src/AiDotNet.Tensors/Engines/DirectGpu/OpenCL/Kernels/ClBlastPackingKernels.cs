// Copyright (c) AiDotNet. All rights reserved.
// CLBlast packing kernels (Apache 2.0).
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    internal static class ClBlastPackingKernels
    {
        private const string CopyPadOpenCl = """
// =============================================================================
// Copies a matrix from source to destination. The output is padded with zero values in case the
// destination matrix dimensions are larger than the source matrix dimensions. Additionally, the ld
// value and offset can be different.
INLINE_FUNC void _CopyPadMatrix(const int src_one, const int src_two,
                                const int src_ld, const int src_offset,
                                __global const real* restrict src,
                                const int dest_one, const int dest_two,
                                const int dest_ld, const int dest_offset,
                                __global real* dest,
                                const real alpha,
                                const int do_conjugate) {

  // Loops over the work per thread in both dimensions
  #pragma unroll
  for (int _w_one = 0; _w_one < PAD_WPTX; _w_one += 1) {
    const int id_one = (get_group_id(0)*PAD_WPTX + _w_one) * PAD_DIMX + get_local_id(0);
    #pragma unroll
    for (int _w_two = 0; _w_two < PAD_WPTY; _w_two += 1) {
      const int id_two = (get_group_id(1)*PAD_WPTY + _w_two) * PAD_DIMY + get_local_id(1);
      if (id_two < dest_two && id_one < dest_one) {

        // Loads data if the thread IDs are within bounds of the source matrix. Otherwise, set the
        // value to be written to zero.
        real value;
        SetToZero(value);
        if (id_two < src_two && id_one < src_one) {
          value = src[id_two*src_ld + id_one + src_offset];
        }

        // Stores the value in the destination matrix
        if (do_conjugate == 1) { COMPLEX_CONJUGATE(value); }
        Multiply(dest[id_two*dest_ld + id_one + dest_offset], alpha, value);
      }
    }
  }
}

// Interface to the above function
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(PAD_DIMX, PAD_DIMY, 1)))
#endif
void CopyPadMatrix(const int src_one, const int src_two,
                   const int src_ld, const int src_offset,
                   __global const real* restrict src,
                   const int dest_one, const int dest_two,
                   const int dest_ld, const int dest_offset,
                   __global real* dest,
                   const real_arg arg_alpha,
                   const int do_conjugate) {
  const real alpha = GetRealArg(arg_alpha);
  _CopyPadMatrix(src_one, src_two, src_ld, src_offset, src,
                 dest_one, dest_two, dest_ld, dest_offset, dest,
                 alpha, do_conjugate);
}

// =============================================================================
// Same as above, but now un-pads a matrix. This kernel reads data from a padded source matrix, but
// writes only the actual data back to the destination matrix. Again, the ld value and offset can be
// different.
INLINE_FUNC void _CopyMatrix(const int src_one, const int src_two,
                             const int src_ld, const int src_offset,
                             __global const real* restrict src,
                             const int dest_one, const int dest_two,
                             const int dest_ld, const int dest_offset,
                             __global real* dest,
                             const real alpha,
                             const int upper, const int lower,
                             const int diagonal_imag_zero) {

  // Loops over the work per thread in both dimensions
  #pragma unroll
  for (int _w_one = 0; _w_one < PAD_WPTX; _w_one += 1) {
    const int id_one = (get_group_id(0)*PAD_WPTX + _w_one) * PAD_DIMX + get_local_id(0);
    #pragma unroll
    for (int _w_two = 0; _w_two < PAD_WPTY; _w_two += 1) {
      const int id_two = (get_group_id(1)*PAD_WPTY + _w_two) * PAD_DIMY + get_local_id(1);

      // Masking in case of triangular matrices: updates only the upper or lower part
      bool condition = true;
      #if defined(ROUTINE_SYRK) || defined(ROUTINE_HERK) || defined(ROUTINE_SYR2K) || defined(ROUTINE_HER2K)
        if (upper == 1) { condition = (id_two >= id_one); }
        else if (lower == 1) { condition = (id_two <= id_one); }
      #endif
      if (condition) {

        // Copies the value into the destination matrix. This is always within bounds of the source
        // matrix, as we know that the destination matrix is smaller or equal to the source.
        if (id_two < dest_two && id_one < dest_one) {
          real value = src[id_two*src_ld + id_one + src_offset];
          if (diagonal_imag_zero == 1 && id_one == id_two) { ImagToZero(value); }
          Multiply(dest[id_two*dest_ld + id_one + dest_offset], alpha, value);
        }
      }
    }
  }
}

// Interface to the above function
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(PAD_DIMX, PAD_DIMY, 1)))
#endif
void CopyMatrix(const int src_one, const int src_two,
                const int src_ld, const int src_offset,
                __global const real* restrict src,
                const int dest_one, const int dest_two,
                const int dest_ld, const int dest_offset,
                __global real* dest,
                const real_arg arg_alpha,
                const int upper, const int lower,
                const int diagonal_imag_zero) {
  const real alpha = GetRealArg(arg_alpha);
  _CopyMatrix(src_one, src_two, src_ld, src_offset, src,
              dest_one, dest_two, dest_ld, dest_offset, dest,
              alpha, upper, lower, diagonal_imag_zero);
}
""";

        private const string TransposePadOpenCl = """
// =============================================================================
// Transposes a matrix from source to destination. The output is padded with zero values in case the
// destination matrix dimensions are larger than the transposed source matrix dimensions.
INLINE_FUNC void _TransposePadMatrix(LOCAL_PTR real* tile,
                                     const int src_one, const int src_two,
                                     const int src_ld, const int src_offset,
                                     __global const real* restrict src,
                                     const int dest_one, const int dest_two,
                                     const int dest_ld, const int dest_offset,
                                     __global real* dest,
                                     const real alpha,
                                     const int do_conjugate) {

  // Loop over the work per thread
  #pragma unroll
  for (int _w_one = 0; _w_one < PADTRA_WPT; _w_one += 1) {
    #pragma unroll
    for (int _w_two = 0; _w_two < PADTRA_WPT; _w_two += 1) {

      // Computes the identifiers for the source matrix. Note that the local and global dimensions
      // do not correspond to each other!
      const int id_src_one = (get_group_id(1)*PADTRA_WPT + _w_two) * PADTRA_TILE + get_local_id(0);
      const int id_src_two = (get_group_id(0)*PADTRA_WPT + _w_one) * PADTRA_TILE + get_local_id(1);

      // Loads data into the local memory if the thread IDs are within bounds of the source matrix.
      // Otherwise, set the local memory value to zero.
      real value;
      SetToZero(value);
      if (id_src_two < src_two && id_src_one < src_one) {
        value = src[id_src_two*src_ld + id_src_one + src_offset];
      }
      const int tile_id0 = get_local_id(0)*PADTRA_WPT + _w_one;
      const int tile_id1 = get_local_id(1)*PADTRA_WPT + _w_two;
      tile[tile_id1 * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD) + tile_id0] = value;
    }
  }

  // Synchronizes all threads in a workgroup
  barrier(CLK_LOCAL_MEM_FENCE);

  // Loop over the work per thread
  #pragma unroll
  for (int _w_one = 0; _w_one < PADTRA_WPT; _w_one += 1) {
    #pragma unroll
    for (int _w_two = 0; _w_two < PADTRA_WPT; _w_two += 1) {

      // Computes the identifiers for the destination matrix
      const int id_dest_one = (get_group_id(0)*PADTRA_WPT + _w_one) * PADTRA_TILE + get_local_id(0);
      const int id_dest_two = (get_group_id(1)*PADTRA_WPT + _w_two) * PADTRA_TILE + get_local_id(1);

      // Stores the transposed value in the destination matrix
      if ((id_dest_one < dest_one) && (id_dest_two < dest_two)) {
        const int tile_id0 = get_local_id(1)*PADTRA_WPT + _w_one;
        const int tile_id1 = get_local_id(0)*PADTRA_WPT + _w_two;
        real value = tile[tile_id1 * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD) + tile_id0];
        if (do_conjugate == 1) { COMPLEX_CONJUGATE(value); }
        Multiply(dest[id_dest_two*dest_ld + id_dest_one + dest_offset], alpha, value);
      }
    }
  }
}

// Interface to the above function
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(PADTRA_TILE, PADTRA_TILE, 1)))
#endif
void TransposePadMatrix(const int src_one, const int src_two,
                        const int src_ld, const int src_offset,
                        __global const real* restrict src,
                        const int dest_one, const int dest_two,
                        const int dest_ld, const int dest_offset,
                        __global real* dest,
                        const real_arg arg_alpha,
                        const int do_conjugate) {
  const real alpha = GetRealArg(arg_alpha);
  __local real tile[(PADTRA_WPT*PADTRA_TILE) * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD)];
  _TransposePadMatrix(tile, src_one, src_two, src_ld, src_offset, src,
                      dest_one, dest_two, dest_ld, dest_offset, dest,
                      alpha, do_conjugate);
}

// =============================================================================
// Transposes a matrix, while considering possible padding in the source matrix. Data is read from a
// padded source matrix, but only the actual data is written back to the transposed destination
// matrix. This kernel optionally checks for upper/lower triangular matrices.
INLINE_FUNC void _TransposeMatrix(LOCAL_PTR real* tile,
                                  const int src_one, const int src_two,
                                  const int src_ld, const int src_offset,
                                  __global const real* restrict src,
                                  const int dest_one, const int dest_two,
                                  const int dest_ld, const int dest_offset,
                                  __global real* dest,
                                  const real alpha,
                                  const int upper, const int lower,
                                  const int diagonal_imag_zero) {

  // Loop over the work per thread
  #pragma unroll
  for (int _w_one = 0; _w_one < PADTRA_WPT; _w_one += 1) {
    #pragma unroll
    for (int _w_two = 0; _w_two < PADTRA_WPT; _w_two += 1) {

      // Computes the identifiers for the source matrix. Note that the local and global dimensions
      // do not correspond to each other!
      const int id_src_one = (get_group_id(1)*PADTRA_WPT + _w_two) * PADTRA_TILE + get_local_id(0);
      const int id_src_two = (get_group_id(0)*PADTRA_WPT + _w_one) * PADTRA_TILE + get_local_id(1);

      // Loads data into the local memory if the thread IDs are within bounds of the source matrix.
      if ((id_src_one < src_one) && (id_src_two < src_two)) {
        real value = src[id_src_two*src_ld + id_src_one + src_offset];
        const int tile_id0 = get_local_id(0)*PADTRA_WPT + _w_one;
        const int tile_id1 = get_local_id(1)*PADTRA_WPT + _w_two;
        tile[tile_id1 * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD) + tile_id0] = value;
      }
    }
  }

  // Synchronizes all threads in a workgroup
  barrier(CLK_LOCAL_MEM_FENCE);

  // Loop over the work per thread
  #pragma unroll
  for (int _w_one = 0; _w_one < PADTRA_WPT; _w_one += 1) {
    #pragma unroll
    for (int _w_two = 0; _w_two < PADTRA_WPT; _w_two += 1) {

      // Computes the identifiers for the destination matrix
      const int id_dest_one = (get_group_id(0)*PADTRA_WPT + _w_one) * PADTRA_TILE + get_local_id(0);
      const int id_dest_two = (get_group_id(1)*PADTRA_WPT + _w_two) * PADTRA_TILE + get_local_id(1);

      // Masking in case of triangular matrices: updates only the upper or lower part
      bool condition = true;
      #if defined(ROUTINE_SYRK) || defined(ROUTINE_HERK) || defined(ROUTINE_SYR2K) || defined(ROUTINE_HER2K)
        if (upper == 1) { condition = (id_dest_one >= id_dest_two); }
        else if (lower == 1) { condition = (id_dest_one <= id_dest_two); }
      #endif
      if (condition) {

        // Stores the transposed value in the destination matrix
        if ((id_dest_one < dest_one) && (id_dest_two < dest_two)) {
          const int tile_id0 = get_local_id(1)*PADTRA_WPT + _w_one;
          const int tile_id1 = get_local_id(0)*PADTRA_WPT + _w_two;
          real value = tile[tile_id1 * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD) + tile_id0];
          if (diagonal_imag_zero == 1 && id_dest_one == id_dest_two) { ImagToZero(value); }
          Multiply(dest[id_dest_two*dest_ld + id_dest_one + dest_offset], alpha, value);
        }
      }
    }
  }
}

// Interface to the above function
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(PADTRA_TILE, PADTRA_TILE, 1)))
#endif
void TransposeMatrix(const int src_one, const int src_two,
                     const int src_ld, const int src_offset,
                     __global const real* restrict src,
                     const int dest_one, const int dest_two,
                     const int dest_ld, const int dest_offset,
                     __global real* dest,
                     const real_arg arg_alpha,
                     const int upper, const int lower,
                     const int diagonal_imag_zero) {
  const real alpha = GetRealArg(arg_alpha);
  __local real tile[(PADTRA_WPT*PADTRA_TILE) * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD)];
  _TransposeMatrix(tile, src_one, src_two, src_ld, src_offset, src,
                   dest_one, dest_two, dest_ld, dest_offset, dest,
                   alpha, upper, lower, diagonal_imag_zero);
}
""";

        public static string BuildSource(ClBlastPadParameters padParams, ClBlastPadTransposeParameters padTranspose)
        {
            var sb = new StringBuilder();
            sb.AppendLine("// CLBlast packing kernels (Apache 2.0)");
            sb.AppendLine("#define PRECISION 32");
            sb.AppendLine($"#define PAD_DIMX {padParams.DimX}");
            sb.AppendLine($"#define PAD_DIMY {padParams.DimY}");
            sb.AppendLine($"#define PAD_WPTX {padParams.WorkPerThreadX}");
            sb.AppendLine($"#define PAD_WPTY {padParams.WorkPerThreadY}");
            sb.AppendLine($"#define PADTRA_PAD {padTranspose.Pad}");
            sb.AppendLine($"#define PADTRA_TILE {padTranspose.Tile}");
            sb.AppendLine($"#define PADTRA_WPT {padTranspose.WorkPerThread}");
            sb.AppendLine("#define RELAX_WORKGROUP_SIZE 0");
            sb.AppendLine(ClBlastXgemmKernel.CommonOpenCl);
            sb.AppendLine(ClBlastXgemmKernel.Level3OpenCl);
            sb.AppendLine(CopyPadOpenCl);
            sb.AppendLine(TransposePadOpenCl);
            return sb.ToString();
        }

        public static string[] GetKernelNames()
        {
            return new[]
            {
                "CopyPadMatrix",
                "CopyMatrix",
                "TransposePadMatrix",
                "TransposeMatrix"
            };
        }
    }
}
