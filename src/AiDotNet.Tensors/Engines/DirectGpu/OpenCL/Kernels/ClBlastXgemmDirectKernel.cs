// Copyright (c) AiDotNet. All rights reserved.
// Portions of this file are derived from the CLBlast project (Apache 2.0).
// See https://github.com/CNugteren/CLBlast for original sources.
using System.Text;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    internal static class ClBlastXgemmDirectKernel
    {
        private const string XgemmDirectPart1 = """


// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library. Note that all parameters here have a
// suffix 'D' to denote that they are for the 'direct' version of the GEMM kernel.
#ifndef WGD
  #define WGD 8      // Tile-size in dimension M, N, and K (e.g. 8, 16, 32, 64)
#endif
#ifndef MDIMCD
  #define MDIMCD 8    // Threads per workgroup in M-dimension (e.g. 8, 16, 32)
#endif
#ifndef NDIMCD
  #define NDIMCD 8    // Threads per workgroup in N-dimension (e.g. 8, 16, 32)
#endif
#ifndef MDIMAD
  #define MDIMAD 8    // Re-shaped tile dimension of matrix A: KDIMAD * MDIMAD
#endif
#ifndef NDIMBD
  #define NDIMBD 8    // Re-shaped tile dimension of matrix B: KDIMBD * NDIMBD
#endif
#ifndef KWID
  #define KWID 1      // Unroll factor of the WGD loop (smaller or equal than WGD)
#endif
#ifndef VWMD
  #define VWMD 1      // Vector width of matrices A and C
#endif
#ifndef VWND
  #define VWND 1      // Vector width of matrix B
#endif
#ifndef PADA
  #define PADA 1      // Local memory padding for matrix A
#endif
#ifndef PADB
  #define PADB 1      // Local memory padding for matrix B
#endif

// Helper parameters based on the above tuning parameters
#define MWID (WGD/MDIMCD)                // Work per work-item (M-dimension)
#define NWID (WGD/NDIMCD)                // Work per work-item (N-dimension)
#define KDIMAD ((MDIMCD*NDIMCD)/(MDIMAD)) // Re-shaped tile dimension of matrix A: KDIMAD * MDIMAD
#define KDIMBD ((MDIMCD*NDIMCD)/(NDIMBD)) // Re-shaped tile dimension of matrix B: KDIMBD * NDIMBD
#define MWAD (WGD/MDIMAD)                // Amount of loads-per-thread for matrix A (M-dimension)
#define KWAD (WGD/KDIMAD)                // Amount of loads-per-thread for matrix A (K-dimension)
#define KWBD (WGD/KDIMBD)                // Amount of loads-per-thread for matrix B (K-dimension)
#define NWBD (WGD/NDIMBD)                // Amount of loads-per-thread for matrix B (N-dimension)

// =================================================================================================

// Data-widths in dimension M
#if VWMD == 1
    typedef real realMD;
#elif VWMD == 2
    typedef real2 realMD;
#elif VWMD == 4
    typedef real4 realMD;
#elif VWMD == 8
    typedef real8 realMD;
#elif VWMD == 16
    typedef real16 realMD;
#endif

// Data-widths in dimension N
#if VWND == 1
    typedef real realND;
#elif VWND == 2
    typedef real2 realND;
#elif VWND == 4
    typedef real4 realND;
#elif VWND == 8
    typedef real8 realND;
#elif VWND == 16
    typedef real16 realND;
#endif

// =================================================================================================

// Loads global off-chip memory into thread-private register files. This function is specific for
// loading the A input matrix.
INLINE_FUNC real GlobalToPrivateDirectA(const __global real* restrict agms, const int _mi,
                                        const int a_ld, const int a_offset, const int idm, const int idk,
                                        const int a_transpose, const int a_conjugate) {
  const int a_index = (a_transpose) ? (idm + _mi)*a_ld + idk : idk*a_ld + (idm + _mi);
  real result = agms[a_index + a_offset];
  if (a_conjugate) { COMPLEX_CONJUGATE(result); }
  return result;
}

// Same as above, but now for the B input matrix
INLINE_FUNC real GlobalToPrivateDirectB(const __global real* restrict bgms, const int _ni,
                                        const int b_ld, const int b_offset, const int idn, const int idk,
                                        const int b_transpose, const int b_conjugate) {
  const int b_index = (b_transpose) ? (idn + _ni)*b_ld + idk : idk*b_ld + (idn + _ni);
  real result = bgms[b_index + b_offset];
  if (b_conjugate) { COMPLEX_CONJUGATE(result); }
  return result;
}

// Loads global off-chip memory into thread-private register files. This function is specific for
// loading the A input matrix. This is the same as above but now includes a bounds check.
INLINE_FUNC real GlobalToPrivateCheckedA(const __global real* restrict agms, const int _mi,
                                         const int a_ld, const int a_offset, const int idm, const int idk,
                                         const int a_transpose, const int a_conjugate,
                                         const int kSizeM) {
  real result;
  if (idm + _mi < kSizeM) {
    const int a_index = (a_transpose) ? (idm + _mi)*a_ld + idk : idk*a_ld + (idm + _mi);
    result = agms[a_index + a_offset];
    if (a_conjugate) { COMPLEX_CONJUGATE(result); }
  }
  else {
    SetToZero(result);
  }
  return result;
}

// Same as above, but now for the B input matrix
INLINE_FUNC real GlobalToPrivateCheckedB(const __global real* restrict bgms, const int _ni,
                                         const int b_ld, const int b_offset, const int idn, const int idk,
                                         const int b_transpose, const int b_conjugate,
                                         const int kSizeN) {
  real result;
  if (idn + _ni < kSizeN) {
    const int b_index = (b_transpose) ? (idn + _ni)*b_ld + idk : idk*b_ld + (idn + _ni);
    result = bgms[b_index + b_offset];
    if (b_conjugate) { COMPLEX_CONJUGATE(result); }
  }
  else {
    SetToZero(result);
  }
  return result;
}

// =================================================================================================

// Caches on-chip local memory into per-thread private memory (registers). This function is specific
// for caching the A input matrix.
INLINE_FUNC real LocalToPrivateDirectA(LOCAL_PTR real* alm, const int _mi, const int kg,
                                       const int a_transpose) {
  const int mg = _mi + get_local_id(0)*MWID;
  const int index = (a_transpose) ? mg*(WGD + PADA) + kg : kg*(WGD + PADA) + mg;
  return alm[index];
}

// Same as above, but now for the B input matrix
INLINE_FUNC real LocalToPrivateDirectB(LOCAL_PTR real* blm, const int _ni, const int kg,
                                       const int b_transpose) {
  const int ng = _ni + get_local_id(1)*NWID;
  const int index = (b_transpose) ? ng*(WGD + PADB) + kg : kg*(WGD + PADB) + ng;
  return blm[index];
}

// =================================================================================================

// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication
// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm
INLINE_FUNC void StoreResultsDirect(__global real* cgm, const real c_value,
                                    const int _mi, const int _ni, const int idm, const int idn,
                                    const real alpha, const real beta,
                                    const int c_ld, const int c_offset, const int c_transpose) {

  // Determines the destination index
  int c_index = (c_transpose) ? (idm + _mi)*c_ld + (idn + _ni) : (idn + _ni)*c_ld + (idm + _mi);

  // The final multiplication with alpha (in case beta == 0)
  real result;
  if (IsZero(beta)) {
    Multiply(result, alpha, c_value);
  }
  // The final multiplication with alpha and the addition with beta*C
  else {
    AXPBY(result, alpha, c_value, beta, cgm[c_index + c_offset]);
  }
  cgm[c_index + c_offset] = result;
}

// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication
// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm
INLINE_FUNC void StoreResultsChecked(__global real* cgm, const real c_value,
                                     const int _mi, const int _ni, const int idm, const int idn,
                                     const int kSizeM, const int kSizeN,
                                     const real alpha, const real beta,
                                     const int c_ld, const int c_offset, const int c_transpose) {
  if ((idm + _mi) < kSizeM && (idn + _ni) < kSizeN) {

    // Deter_mines the destination index
    int c_index = (c_transpose) ? (idm + _mi)*c_ld + (idn + _ni) : (idn + _ni)*c_ld + (idm + _mi);

    // The final multiplication with alpha (in case beta == 0)
    real result;
    if (IsZero(beta)) {
      Multiply(result, alpha, c_value);
    }
    // The final multiplication with alpha and the addition with beta*C
    else {
      AXPBY(result, alpha, c_value, beta, cgm[c_index + c_offset]);
    }
    cgm[c_index + c_offset] = result;
  }
}

// =================================================================================================

// End of the C++11 raw string literal
""";

        private const string XgemmDirectPart2 = """


// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix.
INLINE_FUNC void GlobalToLocalDirectA(const __global realMD* restrict agm, LOCAL_PTR real* alm,
                                      const int a_ld, const int a_offset, const int kwg,
                                      const int a_transpose, const int a_conjugate) {
  #if MDIMCD == MDIMAD
    const int la0 = get_local_id(0);
    const int la1 = get_local_id(1);
  #else
    const int tid = get_local_id(0) + MDIMCD*get_local_id(1);
    const int la0 = tid % MDIMAD;
    const int la1 = tid / MDIMAD;
  #endif
  #pragma unroll
  for (int _mia = 0; _mia < MWAD/VWMD; _mia += 1) {
    #pragma unroll
    for (int _kia = 0; _kia < KWAD; _kia += 1) {

      // Computes the indices for the global memory
      int mg = _mia + la0*(MWAD/VWMD);
      int kg = _kia + la1*KWAD;
      int idm = (a_transpose) ? mg + kwg/VWMD : mg + GetGroupID0()*(WGD/VWMD);
      int idk = (a_transpose) ? kg + GetGroupID0()*WGD : kg + kwg;

      // Loads the data from global memory into the local memory
      const realMD avec = agm[idk*(a_ld/VWMD) + idm + (a_offset/VWMD)];
      #if VWMD == 1
         alm[kg*(WGD + PADA) + mg] = avec;
      #elif VWMD == 2
         alm[kg*(WGD + PADA) + mg*VWMD + 0] = avec.x;
         alm[kg*(WGD + PADA) + mg*VWMD + 1] = avec.y;
      #elif VWMD == 4
         alm[kg*(WGD + PADA) + mg*VWMD + 0] = avec.x;
         alm[kg*(WGD + PADA) + mg*VWMD + 1] = avec.y;
         alm[kg*(WGD + PADA) + mg*VWMD + 2] = avec.z;
         alm[kg*(WGD + PADA) + mg*VWMD + 3] = avec.w;
      #elif VWMD == 8
         alm[kg*(WGD + PADA) + mg*VWMD + 0] = avec.s0;
         alm[kg*(WGD + PADA) + mg*VWMD + 1] = avec.s1;
         alm[kg*(WGD + PADA) + mg*VWMD + 2] = avec.s2;
         alm[kg*(WGD + PADA) + mg*VWMD + 3] = avec.s3;
         alm[kg*(WGD + PADA) + mg*VWMD + 4] = avec.s4;
         alm[kg*(WGD + PADA) + mg*VWMD + 5] = avec.s5;
         alm[kg*(WGD + PADA) + mg*VWMD + 6] = avec.s6;
         alm[kg*(WGD + PADA) + mg*VWMD + 7] = avec.s7;
      #elif VWMD == 16
         alm[kg*(WGD + PADA) + mg*VWMD + 0] = avec.s0;
         alm[kg*(WGD + PADA) + mg*VWMD + 1] = avec.s1;
         alm[kg*(WGD + PADA) + mg*VWMD + 2] = avec.s2;
         alm[kg*(WGD + PADA) + mg*VWMD + 3] = avec.s3;
         alm[kg*(WGD + PADA) + mg*VWMD + 4] = avec.s4;
         alm[kg*(WGD + PADA) + mg*VWMD + 5] = avec.s5;
         alm[kg*(WGD + PADA) + mg*VWMD + 6] = avec.s6;
         alm[kg*(WGD + PADA) + mg*VWMD + 7] = avec.s7;
         alm[kg*(WGD + PADA) + mg*VWMD + 8] = avec.s8;
         alm[kg*(WGD + PADA) + mg*VWMD + 9] = avec.s9;
         alm[kg*(WGD + PADA) + mg*VWMD + 10] = avec.sA;
         alm[kg*(WGD + PADA) + mg*VWMD + 11] = avec.sB;
         alm[kg*(WGD + PADA) + mg*VWMD + 12] = avec.sC;
         alm[kg*(WGD + PADA) + mg*VWMD + 13] = avec.sD;
         alm[kg*(WGD + PADA) + mg*VWMD + 14] = avec.sE;
         alm[kg*(WGD + PADA) + mg*VWMD + 15] = avec.sF;
      #endif
      if (a_conjugate) {
        for (int vm=0; vm<VWMD; ++vm) {
          COMPLEX_CONJUGATE(alm[kg*(WGD + PADA) + mg*VWMD + vm]);
        }
      }
    }
  }
}

// Same as above, but now for the B input matrix
INLINE_FUNC void GlobalToLocalDirectB(const __global realND* restrict bgm, LOCAL_PTR real* blm,
                                      const int b_ld, const int b_offset, const int kwg,
                                      const int b_transpose, const int b_conjugate) {
  #if MDIMCD == NDIMBD
    const int lb0 = get_local_id(0);
    const int lb1 = get_local_id(1);
  #else
    const int tid = get_local_id(0) + MDIMCD*get_local_id(1);
    const int lb0 = tid % NDIMBD;
    const int lb1 = tid / NDIMBD;
  #endif
  #pragma unroll
  for (int _kib = 0; _kib < KWBD; _kib += 1) {
    #pragma unroll
    for (int _nib = 0; _nib < NWBD/VWND; _nib += 1) {

      // Computes the indices for the global memory
      int ng = _nib + lb0*(NWBD/VWND);
      int kg = _kib + lb1*KWBD;
      int idn = (b_transpose) ? ng + kwg/VWND : ng + GetGroupID1()*(WGD/VWND);
      int idk = (b_transpose) ? kg + GetGroupID1()*WGD : kg + kwg;

      // Loads the data from global memory into the local memory
      const realND bvec = bgm[idk*(b_ld/VWND) + idn + (b_offset/VWND)];
      #if VWND == 1
         blm[kg*(WGD + PADB) + ng] = bvec;
      #elif VWND == 2
         blm[kg*(WGD + PADB) + ng*VWND + 0] = bvec.x;
         blm[kg*(WGD + PADB) + ng*VWND + 1] = bvec.y;
      #elif VWND == 4
         blm[kg*(WGD + PADB) + ng*VWND + 0] = bvec.x;
         blm[kg*(WGD + PADB) + ng*VWND + 1] = bvec.y;
         blm[kg*(WGD + PADB) + ng*VWND + 2] = bvec.z;
         blm[kg*(WGD + PADB) + ng*VWND + 3] = bvec.w;
      #elif VWND == 8
         blm[kg*(WGD + PADB) + ng*VWND + 0] = bvec.s0;
         blm[kg*(WGD + PADB) + ng*VWND + 1] = bvec.s1;
         blm[kg*(WGD + PADB) + ng*VWND + 2] = bvec.s2;
         blm[kg*(WGD + PADB) + ng*VWND + 3] = bvec.s3;
         blm[kg*(WGD + PADB) + ng*VWND + 4] = bvec.s4;
         blm[kg*(WGD + PADB) + ng*VWND + 5] = bvec.s5;
         blm[kg*(WGD + PADB) + ng*VWND + 6] = bvec.s6;
         blm[kg*(WGD + PADB) + ng*VWND + 7] = bvec.s7;
      #elif VWND == 16
         blm[kg*(WGD + PADB) + ng*VWND + 0] = bvec.s0;
         blm[kg*(WGD + PADB) + ng*VWND + 1] = bvec.s1;
         blm[kg*(WGD + PADB) + ng*VWND + 2] = bvec.s2;
         blm[kg*(WGD + PADB) + ng*VWND + 3] = bvec.s3;
         blm[kg*(WGD + PADB) + ng*VWND + 4] = bvec.s4;
         blm[kg*(WGD + PADB) + ng*VWND + 5] = bvec.s5;
         blm[kg*(WGD + PADB) + ng*VWND + 6] = bvec.s6;
         blm[kg*(WGD + PADB) + ng*VWND + 7] = bvec.s7;
         blm[kg*(WGD + PADB) + ng*VWND + 8] = bvec.s8;
         blm[kg*(WGD + PADB) + ng*VWND + 9] = bvec.s9;
         blm[kg*(WGD + PADB) + ng*VWND + 10] = bvec.sA;
         blm[kg*(WGD + PADB) + ng*VWND + 11] = bvec.sB;
         blm[kg*(WGD + PADB) + ng*VWND + 12] = bvec.sC;
         blm[kg*(WGD + PADB) + ng*VWND + 13] = bvec.sD;
         blm[kg*(WGD + PADB) + ng*VWND + 14] = bvec.sE;
         blm[kg*(WGD + PADB) + ng*VWND + 15] = bvec.sF;
      #endif
      if (b_conjugate) {
        #pragma unroll
        for (int _vn = 0; _vn < VWND; _vn += 1) {
          COMPLEX_CONJUGATE(blm[kg*(WGD + PADB) + ng*VWND + _vn]);
        }
      }
    }
  }
}

// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix. In contrast to the functions above, this function performs doesn't
// use the vector data-types.
INLINE_FUNC void GlobalToLocalScalarA(const __global real* restrict agms, LOCAL_PTR real* alm,
                                      const int a_ld, const int a_offset, const int kwg,
                                      const int a_transpose, const int a_conjugate) {
  #if MDIMCD == MDIMAD
    const int la0 = get_local_id(0);
    const int la1 = get_local_id(1);
  #else
    const int tid = get_local_id(0) + MDIMCD*get_local_id(1);
    const int la0 = tid % MDIMAD;
    const int la1 = tid / MDIMAD;
  #endif
  #pragma unroll
  for (int _mia = 0; _mia < MWAD; _mia += 1) {
    #pragma unroll
    for (int _kia = 0; _kia < KWAD; _kia += 1) {

      // Computes the indices for the global memory
      int mg = _mia + la0*MWAD;
      int kg = _kia + la1*KWAD;
      int idm = (a_transpose) ? mg + kwg : mg + GetGroupID0()*WGD;
      int idk = (a_transpose) ? kg + GetGroupID0()*WGD : kg + kwg;

      // Loads the data from global memory into the local memory
      real result = agms[idk*a_ld + idm + a_offset];
      if (a_conjugate) { COMPLEX_CONJUGATE(result); }
      alm[kg*(WGD + PADA) + mg] = result;
    }
  }
}

// Same as above, but now for the B input matrix
INLINE_FUNC void GlobalToLocalScalarB(const __global real* restrict bgms, LOCAL_PTR real* blm,
                                      const int b_ld, const int b_offset, const int kwg,
                                      const int b_transpose, const int b_conjugate) {
  #if MDIMCD == NDIMBD
    const int lb0 = get_local_id(0);
    const int lb1 = get_local_id(1);
  #else
    const int tid = get_local_id(0) + MDIMCD*get_local_id(1);
    const int lb0 = tid % NDIMBD;
    const int lb1 = tid / NDIMBD;
  #endif
  #pragma unroll
  for (int _kib = 0; _kib < KWBD; _kib += 1) {
    #pragma unroll
    for (int _nib = 0; _nib < NWBD; _nib += 1) {

      // Computes the indices for the global memory
      int ng = _nib + lb0*NWBD;
      int kg = _kib + lb1*KWBD;
      int idn = (b_transpose) ? ng + kwg : ng + GetGroupID1()*WGD;
      int idk = (b_transpose) ? kg + GetGroupID1()*WGD : kg + kwg;

      // Loads the data from global memory into the local memory
      real result = bgms[idk*b_ld + idn + b_offset];
      if (b_conjugate) { COMPLEX_CONJUGATE(result); }
      blm[kg*(WGD + PADB) + ng] = result;
    }
  }
}

// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix. In contrast to the functions above, this function performs bounds
// checks and doesn't use the vector data-types.
INLINE_FUNC void GlobalToLocalCheckedA(const __global real* restrict agms, LOCAL_PTR real* alm,
                                       const int a_ld, const int a_offset, const int kwg,
                                       const int a_transpose, const int a_conjugate,
                                       const int kSizeM, const int kSizeK) {
  #if MDIMCD == MDIMAD
    const int la0 = get_local_id(0);
    const int la1 = get_local_id(1);
  #else
    const int tid = get_local_id(0) + MDIMCD*get_local_id(1);
    const int la0 = tid % MDIMAD;
    const int la1 = tid / MDIMAD;
  #endif
  #pragma unroll
  for (int _mia = 0; _mia < MWAD; _mia += 1) {
    #pragma unroll
    for (int _kia = 0; _kia < KWAD; _kia += 1) {

      // Computes the indices for the global memory
      int mg = _mia + la0*MWAD;
      int kg = _kia + la1*KWAD;
      int idm = (a_transpose) ? mg + kwg : mg + GetGroupID0()*WGD;
      int idk = (a_transpose) ? kg + GetGroupID0()*WGD : kg + kwg;

      // Loads the data from global memory into the local memory
      int condition = (a_transpose) ? (idm < kSizeK) && (idk < kSizeM) :
                                      (idm < kSizeM) && (idk < kSizeK);
      if (condition) {
        real result = agms[idk*a_ld + idm + a_offset];
        if (a_conjugate) { COMPLEX_CONJUGATE(result); }
        alm[kg*(WGD + PADA) + mg] = result;
      }
      else {
        SetToZero(alm[kg*(WGD + PADA) + mg]);
      }
    }
  }
}

// Same as above, but now for the B input matrix
INLINE_FUNC void GlobalToLocalCheckedB(const __global real* restrict bgms, LOCAL_PTR real* blm,
                                       const int b_ld, const int b_offset, const int kwg,
                                       const int b_transpose, const int b_conjugate,
                                       const int kSizeN, const int kSizeK) {
  #if MDIMCD == NDIMBD
    const int lb0 = get_local_id(0);
    const int lb1 = get_local_id(1);
  #else
    const int tid = get_local_id(0) + MDIMCD*get_local_id(1);
    const int lb0 = tid % NDIMBD;
    const int lb1 = tid / NDIMBD;
  #endif
  #pragma unroll
  for (int _kib = 0; _kib < KWBD; _kib += 1) {
    #pragma unroll
    for (int _nib = 0; _nib < NWBD; _nib += 1) {

      // Computes the indices for the global memory
      int ng = _nib + lb0*NWBD;
      int kg = _kib + lb1*KWBD;
      int idn = (b_transpose) ? ng + kwg : ng + GetGroupID1()*WGD;
      int idk = (b_transpose) ? kg + GetGroupID1()*WGD : kg + kwg;

      // Loads the data from global memory into the local memory
      int condition = (b_transpose) ? (idn < kSizeK) && (idk < kSizeN) :
                                      (idn < kSizeN) && (idk < kSizeK);
      if (condition) {
        real result = bgms[idk*b_ld + idn + b_offset];
        if (b_conjugate) { COMPLEX_CONJUGATE(result); }
        blm[kg*(WGD + PADB) + ng] = result;
      }
      else {
        SetToZero(blm[kg*(WGD + PADB) + ng]);
      }
    }
  }
}

// =================================================================================================

// End of the C++11 raw string literal
""";

        private const string XgemmDirectPart3 = """


// =================================================================================================

// Main body of the kernel. This is the direct version without pre/post processing and restrictions.
INLINE_FUNC void XgemmDirect(const int kSizeM, const int kSizeN, const int kSizeK,
                             const real_arg arg_alpha,
                             const real_arg arg_beta,
                             const __global realMD* restrict agm, const int a_offset, const int a_ld,
                             const __global realND* restrict bgm, const int b_offset, const int b_ld,
                             __global real* cgm, const int c_offset, const int c_ld,
                             LOCAL_PTR real* alm, LOCAL_PTR real* blm,
                             const int a_transpose, const int b_transpose, const int c_transpose,
                             const int a_conjugate, const int b_conjugate) {
  const real alpha = GetRealArg(arg_alpha);
  const real beta = GetRealArg(arg_beta);

  // Extra pointers to scalar versions of global memory
  const __global real* restrict agms = (const __global real* restrict) agm;
  const __global real* restrict bgms = (const __global real* restrict) bgm;

  // Allocates workitem-private memory (registers)
  #pragma promote_to_registers
  real apd[MWID];
  #pragma promote_to_registers
  real bpd[NWID];
  #pragma promote_to_registers
  real cpd[NWID * MWID];

  // Initializes the accumulation registers
  #pragma unroll
  for (int _mi = 0; _mi < MWID; _mi += 1) {
    #pragma unroll
    for (int _ni = 0; _ni < NWID; _ni += 1) {
      SetToZero(cpd[_ni * MWID + _mi]);
    }
  }

  // The faster version of GEMM is not allowed on the (incomplete) borders. Therefore, this section
  // processes only the main parts: output blocks of WGD by WGD.
  const int idm = get_local_id(0) * MWID + GetGroupID0() * WGD;
  const int idn = get_local_id(1) * NWID + GetGroupID1() * WGD;
  if ((idm < (kSizeM/WGD)*WGD) && (idn < (kSizeN/WGD)*WGD)) {

    // Loops over all complete workgroup tiles (K-dimension)
    int kwg = 0;
    for (; kwg < (kSizeK/WGD) * WGD; kwg += WGD) {

      // Loads data: off-chip --> local (matrix A and B)
      if (a_ld % VWMD == 0 && a_offset % VWMD == 0) {
        GlobalToLocalDirectA(agm, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate);
      }
      else {
        GlobalToLocalScalarA(agms, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate);
      }
      if (b_ld % VWND == 0 && b_offset % VWND == 0) {
        GlobalToLocalDirectB(bgm, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate);
      }
      else {
        GlobalToLocalScalarB(bgms, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate);
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      // Loops over all workitem tiles, unrolled by a factor KWID
      for (int pwi = 0; pwi < WGD; pwi += KWID) {
        #pragma unroll
        for (int _pit = 0; _pit < KWID; _pit += 1) {
          int kg = pwi + _pit;

          // Loads data: local --> private (matrix A and B)
          #pragma unroll
          for (int _mi = 0; _mi < MWID; _mi += 1) {
            apd[_mi] = LocalToPrivateDirectA(alm, _mi, kg, a_transpose);
          }
          #pragma unroll
          for (int _ni = 0; _ni < NWID; _ni += 1) {
            bpd[_ni] = LocalToPrivateDirectB(blm, _ni, kg, b_transpose);
          }

          // Performs the accumulation (Cpmd += Apmd * Bpmd)
          #pragma unroll
          for (int _ni = 0; _ni < NWID; _ni += 1) {
            #pragma unroll
            for (int _mi = 0; _mi < MWID; _mi += 1) {
              MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Loop over the remaining part (incomplete tile in K-dimension)
    for (; kwg < kSizeK; ++kwg) {

      // Loads data: off-chip --> private (matrix A and B)
      #pragma unroll
      for (int _mi = 0; _mi < MWID; _mi += 1) {
        apd[_mi] = GlobalToPrivateDirectA(agms, _mi, a_ld, a_offset, idm, kwg, a_transpose, a_conjugate);
      }
      #pragma unroll
      for (int _ni = 0; _ni < NWID; _ni += 1) {
        bpd[_ni] = GlobalToPrivateDirectB(bgms, _ni, b_ld, b_offset, idn, kwg, b_transpose, b_conjugate);
      }

      // Performs the accumulation (Cpmd += Apmd * Bpmd)
      #pragma unroll
      for (int _ni = 0; _ni < NWID; _ni += 1) {
        #pragma unroll
        for (int _mi = 0; _mi < MWID; _mi += 1) {
          MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
        }
      }
    }

    // Stores a tile of results and performs the multiplication with alpha and beta
    #pragma unroll
    for (int _ni = 0; _ni < NWID; _ni += 1) {
      #pragma unroll
      for (int _mi = 0; _mi < MWID; _mi += 1) {
        StoreResultsDirect(cgm, cpd[_ni * MWID + _mi], _mi, _ni, idm, idn,
                           alpha, beta, c_ld, c_offset, c_transpose);
      }
    }
  }

  // Simple but slower version for the parts on the edge (incomplete tiles in M and N-dimensions)
  else {

    // Loops over all complete workgroup tiles (K-dimension)
    int kwg = 0;
    for (; kwg < (kSizeK/WGD) * WGD; kwg+=WGD) {

      // Loads data: off-chip --> local (matrix A and B)
      GlobalToLocalCheckedA(agms, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate, kSizeM, kSizeK);
      GlobalToLocalCheckedB(bgms, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate, kSizeN, kSizeK);
      barrier(CLK_LOCAL_MEM_FENCE);

      // Loops over all workitem tiles, unrolled by a factor KWID
      for (int pwi = 0; pwi < WGD; pwi += KWID) {
        #pragma unroll
        for (int _pit = 0; _pit < KWID; _pit += 1) {
          int kg = pwi + _pit;

          // Loads data: local --> private (matrix A and B)
          #pragma unroll
          for (int _mi = 0; _mi < MWID; _mi += 1) {
            apd[_mi] = LocalToPrivateDirectA(alm, _mi, kg, a_transpose);
          }
          #pragma unroll
          for (int _ni = 0; _ni < NWID; _ni += 1) {
            bpd[_ni] = LocalToPrivateDirectB(blm, _ni, kg, b_transpose);
          }

          // Performs the accumulation (C += A * B)
          #pragma unroll
          for (int _ni = 0; _ni < NWID; _ni += 1) {
            #pragma unroll
            for (int _mi = 0; _mi < MWID; _mi += 1) {
              MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Loop over the remaining part (incomplete tile in K-dimension)
    for (; kwg < kSizeK; ++kwg) {

      // Loads data: off-chip --> private (matrix A and B)
      #pragma unroll
      for (int _mi = 0; _mi < MWID; _mi += 1) {
        apd[_mi] = GlobalToPrivateCheckedA(agms, _mi, a_ld, a_offset, idm, kwg, a_transpose, a_conjugate, kSizeM);
      }
      #pragma unroll
      for (int _ni = 0; _ni < NWID; _ni += 1) {
        bpd[_ni] = GlobalToPrivateCheckedB(bgms, _ni, b_ld, b_offset, idn, kwg, b_transpose, b_conjugate, kSizeN);
      }

      // Performs the accumulation (C += A * B)
      #pragma unroll
      for (int _ni = 0; _ni < NWID; _ni += 1) {
        #pragma unroll
        for (int _mi = 0; _mi < MWID; _mi += 1) {
          MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
        }
      }
    }

    // Stores a tile of results and performs the multiplication with alpha and beta
    #pragma unroll
    for (int _ni = 0; _ni < NWID; _ni += 1) {
      #pragma unroll
      for (int _mi = 0; _mi < MWID; _mi += 1) {
        StoreResultsChecked(cgm, cpd[_ni * MWID + _mi], _mi, _ni, idm, idn, kSizeM, kSizeN,
                            alpha, beta, c_ld, c_offset, c_transpose);
      }
    }
  }
}

// =================================================================================================

// Direct version of the GEMM kernel with [A, B] = [non-transposed, non-transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void XgemmDirectNN(const int kSizeM, const int kSizeN, const int kSizeK,
                            const real_arg arg_alpha, const real_arg arg_beta,
                            const __global realMD* restrict agm, const int a_offset, const int a_ld,
                            const __global realND* restrict bgm, const int b_offset, const int b_ld,
                            __global real* cgm, const int c_offset, const int c_ld,
                            const int c_transpose, const int a_conjugate, const int b_conjugate) {
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 0, 0, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the GEMM kernel with [A, B] = [non-transposed, transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void XgemmDirectNT(const int kSizeM, const int kSizeN, const int kSizeK,
                            const real_arg arg_alpha, const real_arg arg_beta,
                            const __global realMD* restrict agm, const int a_offset, const int a_ld,
                            const __global realND* restrict bgm, const int b_offset, const int b_ld,
                            __global real* cgm, const int c_offset, const int c_ld,
                            const int c_transpose, const int a_conjugate, const int b_conjugate) {
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 0, 1, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the GEMM kernel with [A, B] = [transposed, non-transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void XgemmDirectTN(const int kSizeM, const int kSizeN, const int kSizeK,
                            const real_arg arg_alpha, const real_arg arg_beta,
                            const __global realMD* restrict agm, const int a_offset, const int a_ld,
                            const __global realND* restrict bgm, const int b_offset, const int b_ld,
                            __global real* cgm, const int c_offset, const int c_ld,
                            const int c_transpose, const int a_conjugate, const int b_conjugate) {
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 1, 0, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the GEMM kernel with [A, B] = [transposed, transposed]
#if RELAX_WORKGROUP_SIZE == 1
  __kernel
#else
  __kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
#endif
void XgemmDirectTT(const int kSizeM, const int kSizeN, const int kSizeK,
                            const real_arg arg_alpha, const real_arg arg_beta,
                            const __global realMD* restrict agm, const int a_offset, const int a_ld,
                            const __global realND* restrict bgm, const int b_offset, const int b_ld,
                            __global real* cgm, const int c_offset, const int c_ld,
                            const int c_transpose, const int a_conjugate, const int b_conjugate) {
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 1, 1, c_transpose, a_conjugate, b_conjugate);
}

// =================================================================================================

// End of the C++11 raw string literal
""";

        public static string BuildSource(ClBlastXgemmDirectParameters parameters)
        {
            var sb = new StringBuilder();
            sb.AppendLine("// CLBlast XGEMM direct kernel (Apache 2.0)");
            sb.AppendLine("#define PRECISION 32");
            sb.AppendLine($"#define WGD {parameters.Wgd}");
            sb.AppendLine($"#define MDIMCD {parameters.MdimCd}");
            sb.AppendLine($"#define NDIMCD {parameters.NdimCd}");
            sb.AppendLine($"#define MDIMAD {parameters.MdimAd}");
            sb.AppendLine($"#define NDIMBD {parameters.NdimBd}");
            sb.AppendLine($"#define KWID {parameters.Kwid}");
            sb.AppendLine($"#define VWMD {parameters.Vwm}");
            sb.AppendLine($"#define VWND {parameters.Vwn}");
            sb.AppendLine($"#define PADA {parameters.PadA}");
            sb.AppendLine($"#define PADB {parameters.PadB}");
            sb.AppendLine("#define RELAX_WORKGROUP_SIZE 0");
            sb.AppendLine(ClBlastXgemmKernel.CommonOpenCl);
            sb.AppendLine(ClBlastXgemmKernel.Level3OpenCl);
            sb.AppendLine(XgemmDirectPart1);
            sb.AppendLine(XgemmDirectPart2);
            sb.AppendLine(XgemmDirectPart3);
            return sb.ToString();
        }

        public static string[] GetKernelNames()
        {
            return new[]
            {
                "XgemmDirectNN",
                "XgemmDirectNT",
                "XgemmDirectTN",
                "XgemmDirectTT"
            };
        }
    }
}

