namespace AiDotNet.LinearAlgebra;

public readonly struct FastFourierTransform<T>
{
    private readonly INumericOperations<T> _numOps;

    public FastFourierTransform()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public Vector<Complex<T>> Forward(Vector<T> input)
    {
        int n = input.Length;
        var output = new Vector<Complex<T>>(n);

        for (int i = 0; i < n; i++)
        {
            output[i] = new Complex<T>(input[i], _numOps.Zero);
        }

        return FFTInternal(output, false);
    }

    public Vector<T> Inverse(Vector<Complex<T>> input)
    {
        int n = input.Length;
        var complexOutput = FFTInternal(input, true);
        var result = new Vector<T>(n, _numOps);

        for (int i = 0; i < n; i++)
        {
            result[i] = _numOps.Divide(complexOutput[i].Real, _numOps.FromDouble(n));
        }

        return result;
    }

    private Vector<Complex<T>> FFTInternal(Vector<Complex<T>> input, bool inverse)
    {
        int n = input.Length;
        if (n <= 1) return input;

        var even = new Vector<Complex<T>>(n / 2);
        var odd = new Vector<Complex<T>>(n / 2);

        for (int i = 0; i < n / 2; i++)
        {
            even[i] = input[2 * i];
            odd[i] = input[2 * i + 1];
        }

        even = FFTInternal(even, inverse);
        odd = FFTInternal(odd, inverse);

        var output = new Vector<Complex<T>>(n);
        T angleSign = inverse ? _numOps.One : _numOps.Negate(_numOps.One);
        var complexOps = MathHelper.GetNumericOperations<Complex<T>>();

        for (int k = 0; k < n / 2; k++)
        {
            T angle = _numOps.Multiply(angleSign, _numOps.Multiply(_numOps.FromDouble(2 * Math.PI * k), _numOps.FromDouble(1.0 / n)));
            var t = complexOps.Multiply(Complex<T>.FromPolarCoordinates(_numOps.One, angle), odd[k]);
            output[k] = complexOps.Add(even[k], t);
            output[k + n / 2] = complexOps.Subtract(even[k], t);
        }

        return output;
    }
}