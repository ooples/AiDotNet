using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// OpenCL kernels for random number generation.
/// Implements XORSHIFT128+ for uniform distribution and Box-Muller transform for normal distribution.
/// </summary>
public static class RandomKernels
{
    public static string GetKernels()
    {
        var sb = new StringBuilder();

        // Helper: XORSHIFT128+ RNG
        // State is stored in a ulong2 (128 bits) per thread
        // We use a simple hash of global_id + seed to initialize the state
        sb.AppendLine(@"
            // Hash function for state initialization
            ulong hash(ulong x) {
                x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9UL;
                x = (x ^ (x >> 27)) * 0x94d049bb133111ebUL;
                x = x ^ (x >> 31);
                return x;
            }

            // XORSHIFT128+ implementation
            ulong next_rand(ulong2* state) {
                ulong s1 = state->x;
                ulong s0 = state->y;
                state->x = s0;
                s1 ^= s1 << 23;
                state->y = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
                return state->y + s0;
            }

            // Convert ulong to float [0, 1)
            float to_float_unorm(ulong x) {
                // Keep 23 bits of mantissa, divide by 2^23 to get [0, 1) exclusive
                return (x & 0x7FFFFF) * (1.0f / 8388608.0f);
            }
        ");

        // Kernel: Generate Uniform Random Numbers [min, max)
        sb.AppendLine(@"
            __kernel void GenerateRandomUniform(
                __global float* output,
                const int size,
                const float minVal,
                const float maxVal,
                const ulong seed)
            {
                int i = get_global_id(0);
                if (i >= size) return;

                // Initialize state unique to this thread
                ulong h1 = hash(i + seed);
                ulong h2 = hash(h1);
                ulong2 state = (ulong2)(h1, h2);

                // Generate random float [0, 1)
                float u = to_float_unorm(next_rand(&state));

                // Scale to [min, max)
                output[i] = minVal + u * (maxVal - minVal);
            }
        ");

        // Kernel: Generate Normal Random Numbers (Box-Muller)
        sb.AppendLine(@"
            __kernel void GenerateRandomNormal(
                __global float* output,
                const int size,
                const float mean,
                const float stdDev,
                const ulong seed)
            {
                // Box-Muller generates 2 values at once, so we map 1 thread to 2 outputs
                int i = get_global_id(0) * 2;
                if (i >= size) return;

                // Initialize state unique to this thread
                ulong h1 = hash(i + seed);
                ulong h2 = hash(h1);
                ulong2 state = (ulong2)(h1, h2);

                // Generate two uniform randoms u1, u2 in (0, 1]
                float u1 = to_float_unorm(next_rand(&state));
                float u2 = to_float_unorm(next_rand(&state));

                // Avoid log(0)
                if (u1 < 1e-6f) u1 = 1e-6f;

                // Box-Muller Transform
                float r = sqrt(-2.0f * log(u1));
                float theta = 2.0f * 3.14159265359f * u2;

                float z0 = r * cos(theta);
                float z1 = r * sin(theta);

                // Apply mean and stdDev
                output[i] = mean + z0 * stdDev;
                
                // Write second value if within bounds
                if (i + 1 < size) {
                    output[i + 1] = mean + z1 * stdDev;
                }
            }
        ");

        return sb.ToString();
    }
}
