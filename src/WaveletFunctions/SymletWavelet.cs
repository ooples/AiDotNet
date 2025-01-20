namespace AiDotNet.WaveletFunctions;

public class SymletWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _order;
    private readonly Vector<T> _lowDecomp;
    private readonly Vector<T> _highDecomp;
    private readonly Vector<T> _lowRecon;
    private readonly Vector<T> _highRecon;

    public SymletWavelet(int order = 4)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _order = order;
        (_lowDecomp, _highDecomp, _lowRecon, _highRecon) = GetSymletCoefficients(_order);
    }

    public T Calculate(T x)
    {
        // Approximate the wavelet function using the cascade algorithm
        int iterations = 8; // Number of iterations for approximation
        int points = 1024; // Number of points to evaluate

        var phi = new Vector<T>(points);
        phi[points / 2] = _numOps.One; // Initial delta function

        for (int i = 0; i < iterations; i++)
        {
            var newPhi = new Vector<T>(points * 2);
            for (int j = 0; j < points; j++)
            {
                for (int k = 0; k < _lowRecon.Length; k++)
                {
                    int ind = (2 * j + k) % (points * 2);
                    newPhi[ind] = _numOps.Add(newPhi[ind], _numOps.Multiply(_lowRecon[k], phi[j]));
                }
            }

            phi = new Vector<T>(newPhi.Take(points));
        }

        // Interpolate to find the value at x
        T xScaled = _numOps.Multiply(x, _numOps.FromDouble(points - 1));
        int index = (int)Convert.ToDouble(xScaled);
        T fraction = _numOps.Subtract(xScaled, _numOps.FromDouble(index));

        if (index >= points - 1)
            return phi[points - 1];

        return _numOps.Add(
            phi[index],
            _numOps.Multiply(fraction, _numOps.Subtract(phi[index + 1], phi[index]))
        );
    }

    public (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        int n = input.Length;
        var approximation = new Vector<T>(n / 2);
        var detail = new Vector<T>(n / 2);

        for (int i = 0; i < n / 2; i++)
        {
            T approx = _numOps.Zero;
            T det = _numOps.Zero;

            for (int j = 0; j < _lowDecomp.Length; j++)
            {
                int index = (2 * i + j) % n;
                approx = _numOps.Add(approx, _numOps.Multiply(_lowDecomp[j], input[index]));
                det = _numOps.Add(det, _numOps.Multiply(_highDecomp[j], input[index]));
            }

            approximation[i] = approx;
            detail[i] = det;
        }

        return (approximation, detail);
    }

    public Vector<T> GetScalingCoefficients()
    {
        return _lowRecon;
    }

    public Vector<T> GetWaveletCoefficients()
    {
        return _highRecon;
    }

    private (Vector<T> lowDecomp, Vector<T> highDecomp, Vector<T> lowRecon, Vector<T> highRecon) GetSymletCoefficients(int order)
    {
        return order switch
        {
            2 => (
                    new Vector<T>(new[] {
                        _numOps.FromDouble(-0.12940952255092145),
                        _numOps.FromDouble(0.22414386804185735),
                        _numOps.FromDouble(0.836516303737469),
                        _numOps.FromDouble(0.48296291314469025)
                                }),
                                new Vector<T>(new[] {
                        _numOps.FromDouble(-0.48296291314469025),
                        _numOps.FromDouble(0.836516303737469),
                        _numOps.FromDouble(-0.22414386804185735),
                        _numOps.FromDouble(-0.12940952255092145)
                                }),
                                new Vector<T>(new[] {
                        _numOps.FromDouble(0.48296291314469025),
                        _numOps.FromDouble(0.836516303737469),
                        _numOps.FromDouble(0.22414386804185735),
                        _numOps.FromDouble(-0.12940952255092145)
                                }),
                                new Vector<T>(new[] {
                        _numOps.FromDouble(-0.12940952255092145),
                        _numOps.FromDouble(-0.22414386804185735),
                        _numOps.FromDouble(0.836516303737469),
                        _numOps.FromDouble(-0.48296291314469025)
                            })
                        ),
            4 => (
                        new Vector<T>(new[] {
                        _numOps.FromDouble(-0.07576571), _numOps.FromDouble(-0.02963552),
                        _numOps.FromDouble(0.49761866), _numOps.FromDouble(0.80373875),
                        _numOps.FromDouble(0.29785779), _numOps.FromDouble(-0.09921954),
                        _numOps.FromDouble(-0.01260396), _numOps.FromDouble(0.03222310)
                                }),
                                new Vector<T>(new[] {
                        _numOps.FromDouble(-0.03222310), _numOps.FromDouble(-0.01260396),
                        _numOps.FromDouble(0.09921954), _numOps.FromDouble(0.29785779),
                        _numOps.FromDouble(-0.80373875), _numOps.FromDouble(0.49761866),
                        _numOps.FromDouble(0.02963552), _numOps.FromDouble(-0.07576571)
                                }),
                                new Vector<T>(new[] {
                        _numOps.FromDouble(0.03222310), _numOps.FromDouble(-0.01260396),
                        _numOps.FromDouble(-0.09921954), _numOps.FromDouble(0.29785779),
                        _numOps.FromDouble(0.80373875), _numOps.FromDouble(0.49761866),
                        _numOps.FromDouble(-0.02963552), _numOps.FromDouble(-0.07576571)
                                }),
                                new Vector<T>(new[] {
                        _numOps.FromDouble(-0.07576571), _numOps.FromDouble(0.02963552),
                        _numOps.FromDouble(0.49761866), _numOps.FromDouble(-0.80373875),
                        _numOps.FromDouble(0.29785779), _numOps.FromDouble(0.09921954),
                        _numOps.FromDouble(-0.01260396), _numOps.FromDouble(-0.03222310)
                            })
                        ),
            6 => (
                        new Vector<T>(new[] {
                        _numOps.FromDouble(0.015404109327027373),
                        _numOps.FromDouble(0.0034907120842174702),
                        _numOps.FromDouble(-0.11799011114819057),
                        _numOps.FromDouble(-0.048311742585633),
                        _numOps.FromDouble(0.4910559419267466),
                        _numOps.FromDouble(0.787641141030194),
                        _numOps.FromDouble(0.3379294217276218),
                        _numOps.FromDouble(-0.07263752278646252),
                        _numOps.FromDouble(-0.021060292512300564),
                        _numOps.FromDouble(0.04472490177066578),
                        _numOps.FromDouble(0.0017677118642428036),
                        _numOps.FromDouble(-0.007800708325034148)
                                }),
                                new Vector<T>(new[] {
                        _numOps.FromDouble(0.007800708325034148),
                        _numOps.FromDouble(-0.0017677118642428036),
                        _numOps.FromDouble(-0.04472490177066578),
                        _numOps.FromDouble(0.021060292512300564),
                        _numOps.FromDouble(0.07263752278646252),
                        _numOps.FromDouble(-0.3379294217276218),
                        _numOps.FromDouble(-0.787641141030194),
                        _numOps.FromDouble(0.4910559419267466),
                        _numOps.FromDouble(0.048311742585633),
                        _numOps.FromDouble(-0.11799011114819057),
                        _numOps.FromDouble(-0.0034907120842174702),
                        _numOps.FromDouble(0.015404109327027373)
                                }),
                                new Vector<T>(new[] {
                        _numOps.FromDouble(-0.007800708325034148),
                        _numOps.FromDouble(-0.0017677118642428036),
                        _numOps.FromDouble(0.04472490177066578),
                        _numOps.FromDouble(0.021060292512300564),
                        _numOps.FromDouble(-0.07263752278646252),
                        _numOps.FromDouble(-0.3379294217276218),
                        _numOps.FromDouble(0.787641141030194),
                        _numOps.FromDouble(0.4910559419267466),
                        _numOps.FromDouble(-0.048311742585633),
                        _numOps.FromDouble(-0.11799011114819057),
                        _numOps.FromDouble(0.0034907120842174702),
                        _numOps.FromDouble(0.015404109327027373)
                                }),
                                new Vector<T>(new[] {
                        _numOps.FromDouble(0.015404109327027373),
                        _numOps.FromDouble(-0.0034907120842174702),
                        _numOps.FromDouble(-0.11799011114819057),
                        _numOps.FromDouble(0.048311742585633),
                        _numOps.FromDouble(0.4910559419267466),
                        _numOps.FromDouble(-0.787641141030194),
                        _numOps.FromDouble(0.3379294217276218),
                        _numOps.FromDouble(0.07263752278646252),
                        _numOps.FromDouble(-0.021060292512300564),
                        _numOps.FromDouble(-0.04472490177066578),
                        _numOps.FromDouble(0.0017677118642428036),
                        _numOps.FromDouble(0.007800708325034148)
                        })
                    ),
            8 => (
                        new Vector<T>(new[] {
                        _numOps.FromDouble(-0.0033824159510061256),
                        _numOps.FromDouble(-0.0005421323317911481),
                        _numOps.FromDouble(0.03169508781149298),
                        _numOps.FromDouble(0.007607487324917605),
                        _numOps.FromDouble(-0.1432942383508097),
                        _numOps.FromDouble(-0.061273359067658524),
                        _numOps.FromDouble(0.4813596512583722),
                        _numOps.FromDouble(0.7771857517005235),
                        _numOps.FromDouble(0.3644418948353314),
                        _numOps.FromDouble(-0.05194583810770904),
                        _numOps.FromDouble(-0.027333068345077982),
                        _numOps.FromDouble(0.049137179673607506),
                        _numOps.FromDouble(0.003808752013890615),
                        _numOps.FromDouble(-0.01495225833704823),
                        _numOps.FromDouble(-0.0003029205147213668),
                        _numOps.FromDouble(0.0018899503327594609)
                                }),
                                new Vector<T>(new[] {
                        _numOps.FromDouble(-0.0018899503327594609),
                        _numOps.FromDouble(-0.0003029205147213668),
                        _numOps.FromDouble(0.01495225833704823),
                        _numOps.FromDouble(0.003808752013890615),
                        _numOps.FromDouble(-0.049137179673607506),
                        _numOps.FromDouble(-0.027333068345077982),
                        _numOps.FromDouble(0.05194583810770904),
                        _numOps.FromDouble(0.3644418948353314),
                        _numOps.FromDouble(-0.7771857517005235),
                        _numOps.FromDouble(0.4813596512583722),
                        _numOps.FromDouble(0.061273359067658524),
                        _numOps.FromDouble(-0.1432942383508097),
                        _numOps.FromDouble(-0.007607487324917605),
                        _numOps.FromDouble(0.03169508781149298),
                        _numOps.FromDouble(0.0005421323317911481),
                        _numOps.FromDouble(-0.0033824159510061256)
                                }),
                                new Vector<T>(new[] {
                        _numOps.FromDouble(0.0018899503327594609),
                        _numOps.FromDouble(-0.0003029205147213668),
                        _numOps.FromDouble(-0.01495225833704823),
                        _numOps.FromDouble(0.003808752013890615),
                        _numOps.FromDouble(0.049137179673607506),
                        _numOps.FromDouble(-0.027333068345077982),
                        _numOps.FromDouble(-0.05194583810770904),
                        _numOps.FromDouble(0.3644418948353314),
                        _numOps.FromDouble(0.7771857517005235),
                        _numOps.FromDouble(0.4813596512583722),
                        _numOps.FromDouble(-0.061273359067658524),
                        _numOps.FromDouble(-0.1432942383508097),
                        _numOps.FromDouble(0.007607487324917605),
                        _numOps.FromDouble(0.03169508781149298),
                        _numOps.FromDouble(-0.0005421323317911481),
                        _numOps.FromDouble(-0.0033824159510061256)
                                }),
                                new Vector<T>(new[] {
                        _numOps.FromDouble(-0.0033824159510061256),
                        _numOps.FromDouble(0.0005421323317911481),
                        _numOps.FromDouble(0.03169508781149298),
                        _numOps.FromDouble(-0.007607487324917605),
                        _numOps.FromDouble(-0.1432942383508097),
                        _numOps.FromDouble(0.061273359067658524),
                        _numOps.FromDouble(0.4813596512583722),
                        _numOps.FromDouble(-0.7771857517005235),
                        _numOps.FromDouble(0.3644418948353314),
                        _numOps.FromDouble(0.05194583810770904),
                        _numOps.FromDouble(-0.027333068345077982),
                        _numOps.FromDouble(-0.049137179673607506),
                        _numOps.FromDouble(0.003808752013890615),
                        _numOps.FromDouble(0.01495225833704823),
                        _numOps.FromDouble(-0.0003029205147213668),
                        _numOps.FromDouble(-0.0018899503327594609)
                    })
                ),
            _ => throw new ArgumentException($"Symlet wavelet of order {order} is not implemented or not supported. Please use a supported order (e.g., 2, 4, 6, or 8).", nameof(order)),
        };
    }
}