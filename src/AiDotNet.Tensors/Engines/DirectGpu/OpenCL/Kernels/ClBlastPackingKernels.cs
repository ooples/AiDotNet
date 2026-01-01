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

        private const string CopyFastOpenCl = """


// =================================================================================================

// Data-widths
#if COPY_VW == 1
  typedef real realC;
#elif COPY_VW == 2
  typedef real2 realC;
#elif COPY_VW == 4
  typedef real4 realC;
#elif COPY_VW == 8
  typedef real8 realC;
#elif COPY_VW == 16
  typedef real16 realC;
#endif

// =================================================================================================

// Fast copy kernel. Requires 'ld' and the number of threads in dimension 0 to be a multiple of
// COPY_VW. Also requires both matrices to be of the same dimensions and without offset.
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))
#endif
void CopyMatrixFast(const int ld,
                    __global const realC* restrict src,
                    __global realC* dest,
                    const real_arg arg_alpha) {
#if __has_builtin(__builtin_assume)
  __builtin_assume(ld % COPY_VW == 0);
#endif

  const real alpha = GetRealArg(arg_alpha);
  #pragma unroll
  for (int _w_one = 0; _w_one < COPY_WPT; _w_one += 1) {
    const int id_one = get_global_id(0);
    const int id_two = (get_group_id(1)*COPY_WPT + _w_one) * COPY_DIMY + get_local_id(1);
    const int id = id_two*(ld/COPY_VW) + id_one;
    realC result;
    #if COPY_VW == 1
      Multiply(result, alpha, src[id]);
    #elif COPY_VW == 2
      Multiply(result.x, alpha, src[id].x);
      Multiply(result.y, alpha, src[id].y);
    #elif COPY_VW == 4
      Multiply(result.x, alpha, src[id].x);
      Multiply(result.y, alpha, src[id].y);
      Multiply(result.z, alpha, src[id].z);
      Multiply(result.w, alpha, src[id].w);
    #elif COPY_VW == 8
      Multiply(result.s0, alpha, src[id].s0);
      Multiply(result.s1, alpha, src[id].s1);
      Multiply(result.s2, alpha, src[id].s2);
      Multiply(result.s3, alpha, src[id].s3);
      Multiply(result.s4, alpha, src[id].s4);
      Multiply(result.s5, alpha, src[id].s5);
      Multiply(result.s6, alpha, src[id].s6);
      Multiply(result.s7, alpha, src[id].s7);
    #elif COPY_VW == 16
      Multiply(result.s0, alpha, src[id].s0);
      Multiply(result.s1, alpha, src[id].s1);
      Multiply(result.s2, alpha, src[id].s2);
      Multiply(result.s3, alpha, src[id].s3);
      Multiply(result.s4, alpha, src[id].s4);
      Multiply(result.s5, alpha, src[id].s5);
      Multiply(result.s6, alpha, src[id].s6);
      Multiply(result.s7, alpha, src[id].s7);
      Multiply(result.s8, alpha, src[id].s8);
      Multiply(result.s9, alpha, src[id].s9);
      Multiply(result.sA, alpha, src[id].sA);
      Multiply(result.sB, alpha, src[id].sB);
      Multiply(result.sC, alpha, src[id].sC);
      Multiply(result.sD, alpha, src[id].sD);
      Multiply(result.sE, alpha, src[id].sE);
      Multiply(result.sF, alpha, src[id].sF);
    #endif
    dest[id] = result;;
  }
}

// =================================================================================================

// End of the C++11 raw string literal
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

        private const string TransposeFastOpenCl = """


// =================================================================================================

// Data-widths
#if TRA_WPT == 1
  typedef real realT;
#elif TRA_WPT == 2
  typedef real2 realT;
#elif TRA_WPT == 4
  typedef real4 realT;
#elif TRA_WPT == 8
  typedef real8 realT;
#elif TRA_WPT == 16
  typedef real16 realT;
#endif

// =================================================================================================

// Transposes and copies a matrix. Requires both matrices to be of the same dimensions and without
// offset. A more general version is available in 'padtranspose.opencl'.
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(TRA_DIM, TRA_DIM, 1)))
#endif
void TransposeMatrixFast(const int ld,
                         __global const realT* restrict src,
                         __global realT* dest,
                         const real_arg arg_alpha) {
  const real alpha = GetRealArg(arg_alpha);

  // Sets the group identifiers. They might be 'shuffled' around to distribute work in a different
  // way over workgroups, breaking memory-bank dependencies.
  const int gid0 = get_group_id(0);
  #if TRA_SHUFFLE == 1
    const int gid1 = (get_group_id(0) + get_group_id(1)) % get_num_groups(0);
  #else
    const int gid1 = get_group_id(1);
  #endif

  // Local memory to store a tile of the matrix (for coalescing)
  __local realT tile[TRA_WPT*TRA_DIM][TRA_DIM + TRA_PAD];

  // Loops over the work per thread
  #pragma unroll
  for (int _w_one = 0; _w_one < TRA_WPT; _w_one += 1) {

    // Computes the identifiers for the source matrix. Note that the local and global dimensions
    // do not correspond to each other!
    const int id_one = gid1 * TRA_DIM + get_local_id(0);
    const int id_two = (gid0 * TRA_DIM + get_local_id(1))*TRA_WPT + _w_one;

    // Loads data into the local memory
    realT value = src[id_two*(ld/TRA_WPT) + id_one];
    tile[get_local_id(0)*TRA_WPT + _w_one][get_local_id(1)] = value;
  }

  // Synchronizes all threads in a workgroup
  barrier(CLK_LOCAL_MEM_FENCE);

  // Loads transposed data from the local memory
  #pragma promote_to_registers
  realT vpm[TRA_WPT];
  #pragma unroll
  for (int _w_one = 0; _w_one < TRA_WPT; _w_one += 1) {
    vpm[_w_one] = tile[get_local_id(1)*TRA_WPT + _w_one][get_local_id(0)];
  }

  // Performs the register-level transpose of the vectorized data
  #pragma promote_to_registers
  realT results[TRA_WPT];
  #if TRA_WPT == 1
    results[0] = vpm[0];
  #elif TRA_WPT == 2
    results[0].x = vpm[0].x; results[0].y = vpm[1].x;
    results[1].x = vpm[0].y; results[1].y = vpm[1].y;
  #elif TRA_WPT == 4
    results[0].x = vpm[0].x; results[0].y = vpm[1].x; results[0].z = vpm[2].x; results[0].w = vpm[3].x;
    results[1].x = vpm[0].y; results[1].y = vpm[1].y; results[1].z = vpm[2].y; results[1].w = vpm[3].y;
    results[2].x = vpm[0].z; results[2].y = vpm[1].z; results[2].z = vpm[2].z; results[2].w = vpm[3].z;
    results[3].x = vpm[0].w; results[3].y = vpm[1].w; results[3].z = vpm[2].w; results[3].w = vpm[3].w;
  #elif TRA_WPT == 8
    results[0].s0 = vpm[0].s0; results[0].s1 = vpm[1].s0; results[0].s2 = vpm[2].s0; results[0].s3 = vpm[3].s0; results[0].s4 = vpm[4].s0; results[0].s5 = vpm[5].s0; results[0].s6 = vpm[6].s0; results[0].s7 = vpm[7].s0;
    results[1].s0 = vpm[0].s1; results[1].s1 = vpm[1].s1; results[1].s2 = vpm[2].s1; results[1].s3 = vpm[3].s1; results[1].s4 = vpm[4].s1; results[1].s5 = vpm[5].s1; results[1].s6 = vpm[6].s1; results[1].s7 = vpm[7].s1;
    results[2].s0 = vpm[0].s2; results[2].s1 = vpm[1].s2; results[2].s2 = vpm[2].s2; results[2].s3 = vpm[3].s2; results[2].s4 = vpm[4].s2; results[2].s5 = vpm[5].s2; results[2].s6 = vpm[6].s2; results[2].s7 = vpm[7].s2;
    results[3].s0 = vpm[0].s3; results[3].s1 = vpm[1].s3; results[3].s2 = vpm[2].s3; results[3].s3 = vpm[3].s3; results[3].s4 = vpm[4].s3; results[3].s5 = vpm[5].s3; results[3].s6 = vpm[6].s3; results[3].s7 = vpm[7].s3;
    results[4].s0 = vpm[0].s4; results[4].s1 = vpm[1].s4; results[4].s2 = vpm[2].s4; results[4].s3 = vpm[3].s4; results[4].s4 = vpm[4].s4; results[4].s5 = vpm[5].s4; results[4].s6 = vpm[6].s4; results[4].s7 = vpm[7].s4;
    results[5].s0 = vpm[0].s5; results[5].s1 = vpm[1].s5; results[5].s2 = vpm[2].s5; results[5].s3 = vpm[3].s5; results[5].s4 = vpm[4].s5; results[5].s5 = vpm[5].s5; results[5].s6 = vpm[6].s5; results[5].s7 = vpm[7].s5;
    results[6].s0 = vpm[0].s6; results[6].s1 = vpm[1].s6; results[6].s2 = vpm[2].s6; results[6].s3 = vpm[3].s6; results[6].s4 = vpm[4].s6; results[6].s5 = vpm[5].s6; results[6].s6 = vpm[6].s6; results[6].s7 = vpm[7].s6;
    results[7].s0 = vpm[0].s7; results[7].s1 = vpm[1].s7; results[7].s2 = vpm[2].s7; results[7].s3 = vpm[3].s7; results[7].s4 = vpm[4].s7; results[7].s5 = vpm[5].s7; results[7].s6 = vpm[6].s7; results[7].s7 = vpm[7].s7;
  #elif TRA_WPT == 16
    results[ 0].s0 = vpm[0].s0; results[ 0].s1 = vpm[1].s0; results[ 0].s2 = vpm[2].s0; results[ 0].s3 = vpm[3].s0; results[ 0].s4 = vpm[4].s0; results[ 0].s5 = vpm[5].s0; results[ 0].s6 = vpm[6].s0; results[ 0].s7 = vpm[7].s0; results[ 0].s8 = vpm[8].s0; results[ 0].s9 = vpm[9].s0; results[ 0].sA = vpm[10].s0; results[ 0].sB = vpm[11].s0; results[ 0].sC = vpm[12].s0; results[ 0].sD = vpm[13].s0; results[ 0].sE = vpm[14].s0; results[ 0].sF = vpm[15].s0;
    results[ 1].s0 = vpm[0].s1; results[ 1].s1 = vpm[1].s1; results[ 1].s2 = vpm[2].s1; results[ 1].s3 = vpm[3].s1; results[ 1].s4 = vpm[4].s1; results[ 1].s5 = vpm[5].s1; results[ 1].s6 = vpm[6].s1; results[ 1].s7 = vpm[7].s1; results[ 1].s8 = vpm[8].s1; results[ 1].s9 = vpm[9].s1; results[ 1].sA = vpm[10].s1; results[ 1].sB = vpm[11].s1; results[ 1].sC = vpm[12].s1; results[ 1].sD = vpm[13].s1; results[ 1].sE = vpm[14].s1; results[ 1].sF = vpm[15].s1;
    results[ 2].s0 = vpm[0].s2; results[ 2].s1 = vpm[1].s2; results[ 2].s2 = vpm[2].s2; results[ 2].s3 = vpm[3].s2; results[ 2].s4 = vpm[4].s2; results[ 2].s5 = vpm[5].s2; results[ 2].s6 = vpm[6].s2; results[ 2].s7 = vpm[7].s2; results[ 2].s8 = vpm[8].s2; results[ 2].s9 = vpm[9].s2; results[ 2].sA = vpm[10].s2; results[ 2].sB = vpm[11].s2; results[ 2].sC = vpm[12].s2; results[ 2].sD = vpm[13].s2; results[ 2].sE = vpm[14].s2; results[ 2].sF = vpm[15].s2;
    results[ 3].s0 = vpm[0].s3; results[ 3].s1 = vpm[1].s3; results[ 3].s2 = vpm[2].s3; results[ 3].s3 = vpm[3].s3; results[ 3].s4 = vpm[4].s3; results[ 3].s5 = vpm[5].s3; results[ 3].s6 = vpm[6].s3; results[ 3].s7 = vpm[7].s3; results[ 3].s8 = vpm[8].s3; results[ 3].s9 = vpm[9].s3; results[ 3].sA = vpm[10].s3; results[ 3].sB = vpm[11].s3; results[ 3].sC = vpm[12].s3; results[ 3].sD = vpm[13].s3; results[ 3].sE = vpm[14].s3; results[ 3].sF = vpm[15].s3;
    results[ 4].s0 = vpm[0].s4; results[ 4].s1 = vpm[1].s4; results[ 4].s2 = vpm[2].s4; results[ 4].s3 = vpm[3].s4; results[ 4].s4 = vpm[4].s4; results[ 4].s5 = vpm[5].s4; results[ 4].s6 = vpm[6].s4; results[ 4].s7 = vpm[7].s4; results[ 4].s8 = vpm[8].s4; results[ 4].s9 = vpm[9].s4; results[ 4].sA = vpm[10].s4; results[ 4].sB = vpm[11].s4; results[ 4].sC = vpm[12].s4; results[ 4].sD = vpm[13].s4; results[ 4].sE = vpm[14].s4; results[ 4].sF = vpm[15].s4;
    results[ 5].s0 = vpm[0].s5; results[ 5].s1 = vpm[1].s5; results[ 5].s2 = vpm[2].s5; results[ 5].s3 = vpm[3].s5; results[ 5].s4 = vpm[4].s5; results[ 5].s5 = vpm[5].s5; results[ 5].s6 = vpm[6].s5; results[ 5].s7 = vpm[7].s5; results[ 5].s8 = vpm[8].s5; results[ 5].s9 = vpm[9].s5; results[ 5].sA = vpm[10].s5; results[ 5].sB = vpm[11].s5; results[ 5].sC = vpm[12].s5; results[ 5].sD = vpm[13].s5; results[ 5].sE = vpm[14].s5; results[ 5].sF = vpm[15].s5;
    results[ 6].s0 = vpm[0].s6; results[ 6].s1 = vpm[1].s6; results[ 6].s2 = vpm[2].s6; results[ 6].s3 = vpm[3].s6; results[ 6].s4 = vpm[4].s6; results[ 6].s5 = vpm[5].s6; results[ 6].s6 = vpm[6].s6; results[ 6].s7 = vpm[7].s6; results[ 6].s8 = vpm[8].s6; results[ 6].s9 = vpm[9].s6; results[ 6].sA = vpm[10].s6; results[ 6].sB = vpm[11].s6; results[ 6].sC = vpm[12].s6; results[ 6].sD = vpm[13].s6; results[ 6].sE = vpm[14].s6; results[ 6].sF = vpm[15].s6;
    results[ 7].s0 = vpm[0].s7; results[ 7].s1 = vpm[1].s7; results[ 7].s2 = vpm[2].s7; results[ 7].s3 = vpm[3].s7; results[ 7].s4 = vpm[4].s7; results[ 7].s5 = vpm[5].s7; results[ 7].s6 = vpm[6].s7; results[ 7].s7 = vpm[7].s7; results[ 7].s8 = vpm[8].s7; results[ 7].s9 = vpm[9].s7; results[ 7].sA = vpm[10].s7; results[ 7].sB = vpm[11].s7; results[ 7].sC = vpm[12].s7; results[ 7].sD = vpm[13].s7; results[ 7].sE = vpm[14].s7; results[ 7].sF = vpm[15].s7;
    results[ 8].s0 = vpm[0].s8; results[ 8].s1 = vpm[1].s8; results[ 8].s2 = vpm[2].s8; results[ 8].s3 = vpm[3].s8; results[ 8].s4 = vpm[4].s8; results[ 8].s5 = vpm[5].s8; results[ 8].s6 = vpm[6].s8; results[ 8].s7 = vpm[7].s8; results[ 8].s8 = vpm[8].s8; results[ 8].s9 = vpm[9].s8; results[ 8].sA = vpm[10].s8; results[ 8].sB = vpm[11].s8; results[ 8].sC = vpm[12].s8; results[ 8].sD = vpm[13].s8; results[ 8].sE = vpm[14].s8; results[ 8].sF = vpm[15].s8;
    results[ 9].s0 = vpm[0].s9; results[ 9].s1 = vpm[1].s9; results[ 9].s2 = vpm[2].s9; results[ 9].s3 = vpm[3].s9; results[ 9].s4 = vpm[4].s9; results[ 9].s5 = vpm[5].s9; results[ 9].s6 = vpm[6].s9; results[ 9].s7 = vpm[7].s9; results[ 9].s8 = vpm[8].s9; results[ 9].s9 = vpm[9].s9; results[ 9].sA = vpm[10].s9; results[ 9].sB = vpm[11].s9; results[ 9].sC = vpm[12].s9; results[ 9].sD = vpm[13].s9; results[ 9].sE = vpm[14].s9; results[ 9].sF = vpm[15].s9;
    results[10].s0 = vpm[0].sA; results[10].s1 = vpm[1].sA; results[10].s2 = vpm[2].sA; results[10].s3 = vpm[3].sA; results[10].s4 = vpm[4].sA; results[10].s5 = vpm[5].sA; results[10].s6 = vpm[6].sA; results[10].s7 = vpm[7].sA; results[10].s8 = vpm[8].sA; results[10].s9 = vpm[9].sA; results[10].sA = vpm[10].sA; results[10].sB = vpm[11].sA; results[10].sC = vpm[12].sA; results[10].sD = vpm[13].sA; results[10].sE = vpm[14].sA; results[10].sF = vpm[15].sA;
    results[11].s0 = vpm[0].sB; results[11].s1 = vpm[1].sB; results[11].s2 = vpm[2].sB; results[11].s3 = vpm[3].sB; results[11].s4 = vpm[4].sB; results[11].s5 = vpm[5].sB; results[11].s6 = vpm[6].sB; results[11].s7 = vpm[7].sB; results[11].s8 = vpm[8].sB; results[11].s9 = vpm[9].sB; results[11].sA = vpm[10].sB; results[11].sB = vpm[11].sB; results[11].sC = vpm[12].sB; results[11].sD = vpm[13].sB; results[11].sE = vpm[14].sB; results[11].sF = vpm[15].sB;
    results[12].s0 = vpm[0].sC; results[12].s1 = vpm[1].sC; results[12].s2 = vpm[2].sC; results[12].s3 = vpm[3].sC; results[12].s4 = vpm[4].sC; results[12].s5 = vpm[5].sC; results[12].s6 = vpm[6].sC; results[12].s7 = vpm[7].sC; results[12].s8 = vpm[8].sC; results[12].s9 = vpm[9].sC; results[12].sA = vpm[10].sC; results[12].sB = vpm[11].sC; results[12].sC = vpm[12].sC; results[12].sD = vpm[13].sC; results[12].sE = vpm[14].sC; results[12].sF = vpm[15].sC;
    results[13].s0 = vpm[0].sD; results[13].s1 = vpm[1].sD; results[13].s2 = vpm[2].sD; results[13].s3 = vpm[3].sD; results[13].s4 = vpm[4].sD; results[13].s5 = vpm[5].sD; results[13].s6 = vpm[6].sD; results[13].s7 = vpm[7].sD; results[13].s8 = vpm[8].sD; results[13].s9 = vpm[9].sD; results[13].sA = vpm[10].sD; results[13].sB = vpm[11].sD; results[13].sC = vpm[12].sD; results[13].sD = vpm[13].sD; results[13].sE = vpm[14].sD; results[13].sF = vpm[15].sD;
    results[14].s0 = vpm[0].sE; results[14].s1 = vpm[1].sE; results[14].s2 = vpm[2].sE; results[14].s3 = vpm[3].sE; results[14].s4 = vpm[4].sE; results[14].s5 = vpm[5].sE; results[14].s6 = vpm[6].sE; results[14].s7 = vpm[7].sE; results[14].s8 = vpm[8].sE; results[14].s9 = vpm[9].sE; results[14].sA = vpm[10].sE; results[14].sB = vpm[11].sE; results[14].sC = vpm[12].sE; results[14].sD = vpm[13].sE; results[14].sE = vpm[14].sE; results[14].sF = vpm[15].sE;
    results[15].s0 = vpm[0].sF; results[15].s1 = vpm[1].sF; results[15].s2 = vpm[2].sF; results[15].s3 = vpm[3].sF; results[15].s4 = vpm[4].sF; results[15].s5 = vpm[5].sF; results[15].s6 = vpm[6].sF; results[15].s7 = vpm[7].sF; results[15].s8 = vpm[8].sF; results[15].s9 = vpm[9].sF; results[15].sA = vpm[10].sF; results[15].sB = vpm[11].sF; results[15].sC = vpm[12].sF; results[15].sD = vpm[13].sF; results[15].sE = vpm[14].sF; results[15].sF = vpm[15].sF;
  #endif

  // Multiplies by alpha and then stores the results into the destination matrix
  #pragma unroll
  for (int _w_two = 0; _w_two < TRA_WPT; _w_two += 1) {
    realT result;
    #if TRA_WPT == 1
      Multiply(result, alpha, results[_w_two]);
    #elif TRA_WPT == 2
      Multiply(result.x, alpha, results[_w_two].x);
      Multiply(result.y, alpha, results[_w_two].y);
    #elif TRA_WPT == 4
      Multiply(result.x, alpha, results[_w_two].x);
      Multiply(result.y, alpha, results[_w_two].y);
      Multiply(result.z, alpha, results[_w_two].z);
      Multiply(result.w, alpha, results[_w_two].w);
    #elif TRA_WPT == 8
      Multiply(result.s0, alpha, results[_w_two].s0);
      Multiply(result.s1, alpha, results[_w_two].s1);
      Multiply(result.s2, alpha, results[_w_two].s2);
      Multiply(result.s3, alpha, results[_w_two].s3);
      Multiply(result.s4, alpha, results[_w_two].s4);
      Multiply(result.s5, alpha, results[_w_two].s5);
      Multiply(result.s6, alpha, results[_w_two].s6);
      Multiply(result.s7, alpha, results[_w_two].s7);
    #elif TRA_WPT == 16
      Multiply(result.s0, alpha, results[_w_two].s0);
      Multiply(result.s1, alpha, results[_w_two].s1);
      Multiply(result.s2, alpha, results[_w_two].s2);
      Multiply(result.s3, alpha, results[_w_two].s3);
      Multiply(result.s4, alpha, results[_w_two].s4);
      Multiply(result.s5, alpha, results[_w_two].s5);
      Multiply(result.s6, alpha, results[_w_two].s6);
      Multiply(result.s7, alpha, results[_w_two].s7);
      Multiply(result.s8, alpha, results[_w_two].s8);
      Multiply(result.s9, alpha, results[_w_two].s9);
      Multiply(result.sA, alpha, results[_w_two].sA);
      Multiply(result.sB, alpha, results[_w_two].sB);
      Multiply(result.sC, alpha, results[_w_two].sC);
      Multiply(result.sD, alpha, results[_w_two].sD);
      Multiply(result.sE, alpha, results[_w_two].sE);
      Multiply(result.sF, alpha, results[_w_two].sF);
    #endif
    const int id_one = gid0*TRA_DIM + get_local_id(0);
    const int id_two = (gid1*TRA_DIM + get_local_id(1))*TRA_WPT + _w_two;
    dest[id_two*(ld/TRA_WPT) + id_one] = result;
  }
}

// =================================================================================================

// End of the C++11 raw string literal
""";

        public static string BuildSource(
            ClBlastPadParameters padParams,
            ClBlastPadTransposeParameters padTranspose,
            ClBlastCopyParameters copyParams,
            ClBlastTransposeParameters transposeParams)
        {
            var sb = new StringBuilder();
            sb.AppendLine("// CLBlast packing kernels (Apache 2.0)");
            sb.AppendLine("#define PRECISION 32");
            sb.AppendLine($"#define COPY_DIMX {copyParams.DimX}");
            sb.AppendLine($"#define COPY_DIMY {copyParams.DimY}");
            sb.AppendLine($"#define COPY_WPT {copyParams.WorkPerThread}");
            sb.AppendLine($"#define COPY_VW {copyParams.VectorWidth}");
            sb.AppendLine($"#define PAD_DIMX {padParams.DimX}");
            sb.AppendLine($"#define PAD_DIMY {padParams.DimY}");
            sb.AppendLine($"#define PAD_WPTX {padParams.WorkPerThreadX}");
            sb.AppendLine($"#define PAD_WPTY {padParams.WorkPerThreadY}");
            sb.AppendLine($"#define TRA_DIM {transposeParams.Dim}");
            sb.AppendLine($"#define TRA_WPT {transposeParams.WorkPerThread}");
            sb.AppendLine($"#define TRA_PAD {transposeParams.Pad}");
            sb.AppendLine($"#define TRA_SHUFFLE {transposeParams.Shuffle}");
            sb.AppendLine($"#define PADTRA_PAD {padTranspose.Pad}");
            sb.AppendLine($"#define PADTRA_TILE {padTranspose.Tile}");
            sb.AppendLine($"#define PADTRA_WPT {padTranspose.WorkPerThread}");
            sb.AppendLine("#define RELAX_WORKGROUP_SIZE 0");
            sb.AppendLine(ClBlastXgemmKernel.CommonOpenCl);
            sb.AppendLine(ClBlastXgemmKernel.Level3OpenCl);
            sb.AppendLine(CopyFastOpenCl);
            sb.AppendLine(CopyPadOpenCl);
            sb.AppendLine(TransposeFastOpenCl);
            sb.AppendLine(TransposePadOpenCl);
            return sb.ToString();
        }

        public static string[] GetKernelNames()
        {
            return new[]
            {
                "CopyMatrixFast",
                "CopyPadMatrix",
                "CopyMatrix",
                "TransposeMatrixFast",
                "TransposePadMatrix",
                "TransposeMatrix"
            };
        }
    }
}


