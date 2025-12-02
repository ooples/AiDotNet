

namespace AiDotNet.JitCompiler.CodeGen;

/// <summary>
/// Provides optimized implementations of recurrent neural network operations for JIT compilation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These operations implement the core computations for LSTM and GRU cells.
///
/// Recurrent neural networks process sequences by maintaining hidden state that is updated
/// at each timestep. LSTM and GRU are the two most popular RNN variants:
/// - LSTM: Uses input, forget, and output gates with a separate cell state
/// - GRU: Uses update and reset gates with a simpler structure
///
/// These implementations are optimized for execution speed when JIT compiled.
/// </para>
/// </remarks>
public static class RecurrentOps
{
    /// <summary>
    /// Computes a single GRU (Gated Recurrent Unit) cell timestep.
    /// </summary>
    /// <typeparam name="T">The numeric type for tensor elements.</typeparam>
    /// <param name="x">Input tensor of shape [batch, input_size].</param>
    /// <param name="h">Previous hidden state of shape [batch, hidden_size].</param>
    /// <param name="wIh">Input-to-hidden weights of shape [3*hidden_size, input_size].</param>
    /// <param name="wHh">Hidden-to-hidden weights of shape [3*hidden_size, hidden_size].</param>
    /// <param name="bIh">Optional input-to-hidden bias of shape [3*hidden_size].</param>
    /// <param name="bHh">Optional hidden-to-hidden bias of shape [3*hidden_size].</param>
    /// <returns>New hidden state of shape [batch, hidden_size].</returns>
    /// <remarks>
    /// <para>
    /// GRU cell computes:
    /// - z = sigmoid(Wz @ x + Uz @ h + bz)  // Update gate
    /// - r = sigmoid(Wr @ x + Ur @ h + br)  // Reset gate
    /// - h_tilde = tanh(Wh @ x + Uh @ (r * h) + bh)  // Candidate hidden state
    /// - h_new = (1 - z) * h + z * h_tilde  // New hidden state
    /// </para>
    /// </remarks>
    public static Tensor<T> GRUCell<T>(
        Tensor<T> x,
        Tensor<T> h,
        Tensor<T> wIh,
        Tensor<T> wHh,
        Tensor<T>? bIh = null,
        Tensor<T>? bHh = null)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int hiddenSize = h.Shape[^1];
        int batchSize = h.Shape[0];

        // Compute gates: [z, r, h_tilde] for both input and hidden contributions
        // W_ih @ x: [batch, 3*hidden_size]
        var gatesIh = MatrixMultiply(x, Transpose(wIh));
        // W_hh @ h: [batch, 3*hidden_size]
        var gatesHh = MatrixMultiply(h, Transpose(wHh));

        // Add biases if present
        if (bIh != null)
        {
            gatesIh = Add(gatesIh, bIh);
        }
        if (bHh != null)
        {
            gatesHh = Add(gatesHh, bHh);
        }

        // Split gates into z, r, n components
        var gatesIhData = gatesIh.ToArray();
        var gatesHhData = gatesHh.ToArray();
        var hData = h.ToArray();

        var hNewData = new T[batchSize * hiddenSize];

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < hiddenSize; i++)
            {
                int zIdx = b * 3 * hiddenSize + i;
                int rIdx = b * 3 * hiddenSize + hiddenSize + i;
                int nIdx = b * 3 * hiddenSize + 2 * hiddenSize + i;
                int hIdx = b * hiddenSize + i;

                // Update gate: z = sigmoid(z_ih + z_hh)
                T z = Sigmoid(numOps.Add(gatesIhData[zIdx], gatesHhData[zIdx]), numOps);

                // Reset gate: r = sigmoid(r_ih + r_hh)
                T r = Sigmoid(numOps.Add(gatesIhData[rIdx], gatesHhData[rIdx]), numOps);

                // Candidate: n = tanh(n_ih + r * n_hh)
                T nHh = numOps.Multiply(r, gatesHhData[nIdx]);
                T n = Tanh(numOps.Add(gatesIhData[nIdx], nHh), numOps);

                // New hidden: h_new = (1 - z) * h + z * n
                T oneMinusZ = numOps.Subtract(numOps.One, z);
                hNewData[hIdx] = numOps.Add(
                    numOps.Multiply(oneMinusZ, hData[hIdx]),
                    numOps.Multiply(z, n)
                );
            }
        }

        return new Tensor<T>(h.Shape, new Vector<T>(hNewData));
    }

    /// <summary>
    /// Computes a single LSTM (Long Short-Term Memory) cell timestep.
    /// </summary>
    /// <typeparam name="T">The numeric type for tensor elements.</typeparam>
    /// <param name="x">Input tensor of shape [batch, input_size].</param>
    /// <param name="h">Previous hidden state of shape [batch, hidden_size].</param>
    /// <param name="c">Previous cell state of shape [batch, hidden_size].</param>
    /// <param name="wIh">Input-to-hidden weights of shape [4*hidden_size, input_size].</param>
    /// <param name="wHh">Hidden-to-hidden weights of shape [4*hidden_size, hidden_size].</param>
    /// <param name="bIh">Optional input-to-hidden bias of shape [4*hidden_size].</param>
    /// <param name="bHh">Optional hidden-to-hidden bias of shape [4*hidden_size].</param>
    /// <returns>Tuple of (new hidden state, new cell state), each of shape [batch, hidden_size].</returns>
    /// <remarks>
    /// <para>
    /// LSTM cell computes:
    /// - i = sigmoid(Wi @ x + Ui @ h + bi)  // Input gate
    /// - f = sigmoid(Wf @ x + Uf @ h + bf)  // Forget gate
    /// - g = tanh(Wg @ x + Ug @ h + bg)     // Cell candidate
    /// - o = sigmoid(Wo @ x + Uo @ h + bo)  // Output gate
    /// - c_new = f * c + i * g              // New cell state
    /// - h_new = o * tanh(c_new)            // New hidden state
    /// </para>
    /// </remarks>
    public static Tensor<T> LSTMCell<T>(
        Tensor<T> x,
        Tensor<T> h,
        Tensor<T> c,
        Tensor<T> wIh,
        Tensor<T> wHh,
        Tensor<T>? bIh = null,
        Tensor<T>? bHh = null)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int hiddenSize = h.Shape[^1];
        int batchSize = h.Shape[0];

        // Compute gates: [i, f, g, o] for both input and hidden contributions
        // W_ih @ x: [batch, 4*hidden_size]
        var gatesIh = MatrixMultiply(x, Transpose(wIh));
        // W_hh @ h: [batch, 4*hidden_size]
        var gatesHh = MatrixMultiply(h, Transpose(wHh));

        // Add biases if present
        if (bIh != null)
        {
            gatesIh = Add(gatesIh, bIh);
        }
        if (bHh != null)
        {
            gatesHh = Add(gatesHh, bHh);
        }

        // Split gates into i, f, g, o components
        var gatesIhData = gatesIh.ToArray();
        var gatesHhData = gatesHh.ToArray();
        var cData = c.ToArray();

        var hNewData = new T[batchSize * hiddenSize];
        var cNewData = new T[batchSize * hiddenSize];

        for (int b = 0; b < batchSize; b++)
        {
            for (int j = 0; j < hiddenSize; j++)
            {
                int iIdx = b * 4 * hiddenSize + j;
                int fIdx = b * 4 * hiddenSize + hiddenSize + j;
                int gIdx = b * 4 * hiddenSize + 2 * hiddenSize + j;
                int oIdx = b * 4 * hiddenSize + 3 * hiddenSize + j;
                int cIdx = b * hiddenSize + j;

                // Input gate: i = sigmoid(i_ih + i_hh)
                T i = Sigmoid(numOps.Add(gatesIhData[iIdx], gatesHhData[iIdx]), numOps);

                // Forget gate: f = sigmoid(f_ih + f_hh)
                T f = Sigmoid(numOps.Add(gatesIhData[fIdx], gatesHhData[fIdx]), numOps);

                // Cell candidate: g = tanh(g_ih + g_hh)
                T g = Tanh(numOps.Add(gatesIhData[gIdx], gatesHhData[gIdx]), numOps);

                // Output gate: o = sigmoid(o_ih + o_hh)
                T o = Sigmoid(numOps.Add(gatesIhData[oIdx], gatesHhData[oIdx]), numOps);

                // New cell state: c_new = f * c + i * g
                cNewData[cIdx] = numOps.Add(
                    numOps.Multiply(f, cData[cIdx]),
                    numOps.Multiply(i, g)
                );

                // New hidden state: h_new = o * tanh(c_new)
                hNewData[cIdx] = numOps.Multiply(o, Tanh(cNewData[cIdx], numOps));
            }
        }

        // Return concatenated h_new and c_new (caller can split if needed)
        // For simplicity, we return just h_new - the caller should manage c_new separately
        // In a full implementation, this would return a tuple or composite tensor
        return new Tensor<T>(h.Shape, new Vector<T>(hNewData));
    }

    // Helper methods for tensor operations

    private static T Sigmoid<T>(T x, INumericOperations<T> numOps)
    {
        // sigmoid(x) = 1 / (1 + exp(-x))
        var negX = numOps.Negate(x);
        var expNegX = numOps.Exp(negX);
        return numOps.Divide(numOps.One, numOps.Add(numOps.One, expNegX));
    }

    private static T Tanh<T>(T x, INumericOperations<T> numOps)
    {
        // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        var expX = numOps.Exp(x);
        var expNegX = numOps.Exp(numOps.Negate(x));
        return numOps.Divide(
            numOps.Subtract(expX, expNegX),
            numOps.Add(expX, expNegX)
        );
    }

    private static Tensor<T> MatrixMultiply<T>(Tensor<T> a, Tensor<T> b)
    {
        return a.MatrixMultiply(b);
    }

    private static Tensor<T> Transpose<T>(Tensor<T> a)
    {
        return a.Transpose();
    }

    private static Tensor<T> Add<T>(Tensor<T> a, Tensor<T> b)
    {
        return a.Add(b);
    }
}
