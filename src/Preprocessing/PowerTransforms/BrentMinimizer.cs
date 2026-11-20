namespace AiDotNet.Preprocessing.PowerTransforms;

/// <summary>
/// SciPy's scalar minimizer <c>scipy.optimize.brent(func, brack=(a, b))</c>: the downhill bracket search of
/// <c>scipy.optimize.bracket</c> followed by Brent's method (parabolic interpolation with golden-section
/// fallback, <c>tol = 1.48e-8</c>, at most 500 iterations), as used by scikit-learn's PowerTransformer and
/// <c>scipy.stats.boxcox</c> to pick the power parameter.
/// </summary>
internal static class BrentMinimizer
{
    private const double Gold = 1.618034;
    private const double VerySmall = 1e-21;
    private const double GrowLimit = 110.0;
    private const int MaxBracketIterations = 1000;
    private const double Tolerance = 1.48e-8;
    private const double MinTolerance = 1.0e-11;
    private const double GoldenSection = 0.3819660;
    private const int MaxIterations = 500;

    /// <summary>The minimizer of <paramref name="func"/> starting from the bracket (<paramref name="xa"/>, <paramref name="xb"/>).</summary>
    public static double Minimize(Func<double, double> func, double xa, double xb)
    {
        var (a0, b0, c0, fb0) = Bracket(func, xa, xb);

        double x = b0, w = b0, v = b0;
        double fx = fb0, fw = fb0, fv = fb0;
        double a = a0 < c0 ? a0 : c0;
        double b = a0 < c0 ? c0 : a0;
        double deltax = 0.0;
        double rat = 0.0;

        for (int iter = 0; iter < MaxIterations; iter++)
        {
            double tol1 = Tolerance * Math.Abs(x) + MinTolerance;
            double tol2 = 2.0 * tol1;
            double xmid = 0.5 * (a + b);
            if (Math.Abs(x - xmid) < tol2 - 0.5 * (b - a))
            {
                break;
            }

            if (Math.Abs(deltax) <= tol1)
            {
                deltax = x >= xmid ? a - x : b - x;
                rat = GoldenSection * deltax;
            }
            else
            {
                double tmp1 = (x - w) * (fx - fv);
                double tmp2 = (x - v) * (fx - fw);
                double p = (x - v) * tmp2 - (x - w) * tmp1;
                tmp2 = 2.0 * (tmp2 - tmp1);
                if (tmp2 > 0.0)
                {
                    p = -p;
                }

                tmp2 = Math.Abs(tmp2);
                double previous = deltax;
                deltax = rat;
                if (p > tmp2 * (a - x) && p < tmp2 * (b - x) && Math.Abs(p) < Math.Abs(0.5 * tmp2 * previous))
                {
                    rat = p / tmp2;
                    double trial = x + rat;
                    if (trial - a < tol2 || b - trial < tol2)
                    {
                        rat = xmid - x >= 0 ? tol1 : -tol1;
                    }
                }
                else
                {
                    deltax = x >= xmid ? a - x : b - x;
                    rat = GoldenSection * deltax;
                }
            }

            double u = Math.Abs(rat) < tol1 ? (rat >= 0 ? x + tol1 : x - tol1) : x + rat;
            double fu = func(u);

            if (fu > fx)
            {
                if (u < x)
                {
                    a = u;
                }
                else
                {
                    b = u;
                }

                if (fu <= fw || w == x)
                {
                    v = w;
                    w = u;
                    fv = fw;
                    fw = fu;
                }
                else if (fu <= fv || v == x || v == w)
                {
                    v = u;
                    fv = fu;
                }
            }
            else
            {
                if (u >= x)
                {
                    a = x;
                }
                else
                {
                    b = x;
                }

                v = w;
                w = x;
                x = u;
                fv = fw;
                fw = fx;
                fx = fu;
            }
        }

        return x;
    }

    /// <summary><c>scipy.optimize.bracket</c>: (xa, xb, xc, f(xb)) with f(xb) below f(xa) and f(xc).</summary>
    private static (double Xa, double Xb, double Xc, double Fb) Bracket(Func<double, double> func, double xa, double xb)
    {
        double fa = func(xa);
        double fb = func(xb);
        if (fa < fb)
        {
            (xa, xb) = (xb, xa);
            (fa, fb) = (fb, fa);
        }

        double xc = xb + Gold * (xb - xa);
        double fc = func(xc);
        int iter = 0;
        while (fc < fb)
        {
            double tmp1 = (xb - xa) * (fb - fc);
            double tmp2 = (xb - xc) * (fb - fa);
            double val = tmp2 - tmp1;
            double denom = Math.Abs(val) < VerySmall ? 2.0 * VerySmall : 2.0 * val;
            double w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom;
            double wlim = xb + GrowLimit * (xc - xb);
            if (iter > MaxBracketIterations)
            {
                throw new InvalidOperationException("Too many iterations while bracketing the minimum.");
            }

            iter++;
            double fw;
            if ((w - xc) * (xb - w) > 0.0)
            {
                fw = func(w);
                if (fw < fc)
                {
                    xa = xb;
                    xb = w;
                    fa = fb;
                    fb = fw;
                    break;
                }

                if (fw > fb)
                {
                    xc = w;
                    fc = fw;
                    break;
                }

                w = xc + Gold * (xc - xb);
                fw = func(w);
            }
            else if ((w - wlim) * (wlim - xc) >= 0.0)
            {
                w = wlim;
                fw = func(w);
            }
            else if ((w - wlim) * (xc - w) > 0.0)
            {
                fw = func(w);
                if (fw < fc)
                {
                    xb = xc;
                    xc = w;
                    w = xc + Gold * (xc - xb);
                    fb = fc;
                    fc = fw;
                    fw = func(w);
                }
            }
            else
            {
                w = xc + Gold * (xc - xb);
                fw = func(w);
            }

            xa = xb;
            xb = xc;
            xc = w;
            fa = fb;
            fb = fc;
            fc = fw;
        }

        return (xa, xb, xc, fb);
    }
}
