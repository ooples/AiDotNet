using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Codecs;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Audio;

/// <summary>
/// The MP3 decoder reproduces LAME-encoded test tones: an MPEG-2 (24 kHz, mono, 64 kb/s) 440 Hz tone and an MPEG-1
/// (44.1 kHz, joint stereo, 128 kb/s) file with 440 Hz on the left and 1 kHz on the right, both 0.4 amplitude and half
/// a second long. The decoded audio, aligned for the encoder's delay, matches the source tones closely.
/// </summary>
public class Mp3DecoderTests
{
    // lameenc (LAME 3.100), quality 2.
    /// <summary>The MPEG-2 mono test vector, shared with the WellSaid client test.</summary>
    internal const string Mono24kVector = Mono24k;

    private const string Mono24k = ""
        + "//OExAAUmELMf0YYAim5dtttvd3d3bEAGAwGAyZO3uggCCROD4P8McH36jlHn+BHaA/wIc0A/wIc0A/wI7n+j3wfB8CAgckAfD+UOd/R/+GAfB8HwfAgIAgCEeD4P8MKACEIhhli0GLiNsjKDAJmjC"
        + "MCkyiXOj3cSSMI5LM+kyPDFhSoPlpJ80LysjGPFsBwspWAiUCxhwEAGXFeBqRHgaKTZAiuXeBiEmAZNIwGRykBgoUmJ5JLwMVCcDFAxABC//OExGA5q45IB56oAIBhcLAYXDjorZXwDASBgoFgYMCI"
        + "C4AAwKAgMCghVf/BvKDZIAABDZgbiBtYP1//wuFC0ETcGKgxUK6IKh8Ix3//4oIUEOkUCLlIcLmGaJkc0c0pf///ighcxDRcpBSkOcTxiRU8cLyc1////9XOMoxSUTSKJMmReIEYl0gJeJogRMkVGV"
        + "//XTABwBwwA8AAAwAOYCsBCmAyAIZgQwDGYMGHPGG1Eqxg6QhGYYeKpH0bx5Rp//OExCweq14YAd8QABIKemGuhuRg9YOcYQOCPGA0AWJgMADCYDEAWmALgGRgDYAMLABTi/r/+n///9f+97L//J/b"
        + "///f+1l+//97p/X/73T///9P//vG//Hfp0J90vACDMYUNGzOp8MDbBkzFaSE0wbYA/MGTFezxKvz8xzkL9O+QI06ezJxEMWikw0GDCIPAQDQ0dewrbX6LPb3Rv6eixHbvd19H6SLOz1wJV+jkLN3U7"
        + "/RpZ7e66pMQU1FMy4x//OExGQXEFoYANf4JDAwqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqkDB0FAEPM0OOO1MC+AwDFBA84wc8DCFRTo12Ho6MNKC0jjf0yZ8MHRQITAUTAAIhos97L/Z+n/7v"
        + "//+71//Z7fx3/6P8Y7/6f+pMQU1FMy4xMDCqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq//OExJYSgFogANf2JKqqqqqqqq"
        + "qqqqqqqqqqqqqqqqqqqqqqqqqqfpqCZ4sMDG4DGGBigCZiqoEgYQsCTmCaC1Zg+/qKYEIGEmP4wYgRpg0nAgWkISEIBVjVpe692///3u/6f/f19f9n/1O/+3/Gf9On/31MQU1FMy4xMDBVVVVVVVVV"
        + "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV//OExJkTKFogANf4JFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVrrJk+RIYFUn6wYF"
        + "kBLmKHCKZhEILAYP6K9GxUc5Rh5IXUbqiZm9GGOieYbFRgIHqLN6r55b////ve7////1f2f+67+nt/1P/+n+5MQU1FMy4xMDBVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
        + "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV//OExJcSwFogAM/4JFVVVVVVVVVVVVVVVVVUyQpcIwAcyqA32MwLkFwMWBIvjCiAekwzwYdPJJ+IzHpgzs+fOjbSnM9l4yIMzEIQQNXkkSy6dUq/9H"
        + "2Mp1v/ToR+zOu9+n9rG7Hbtz3K/8n7uOf1/0o/fWpMQU1FMy4xMDBVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVMU//OExKYW"
        + "gFoYANf4JI4UAAxCAKAQA3MAVAijAYgdkw64qnMH5BxjDfREw/aCkAMlRCODiY4TQEdjJAQTFUHAELIsC6S6lTk2qe3r/2/T/29lZP+nb+3//73/7r9v/1p/W//2uvT/72/vr9/7x/++t7lfoQlMQU"
        + "1FMy4xMDBVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV//OExLUaQ14YAP9EaFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVWBK"
        + "okgIMioiNArDggHEMDHJ9jA9AT8wu4FhPn/HTzIsgJk/qSDcQlM+gwHJEmJ4KDCAhR+Ajun+/1/o/v3/9H2dn6v/pb/11P6PT/9f/0pMQU1FMy4xMDCqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq"
        + "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq//OExJsTqFYcAN/4JKqqqqqqqqqqqqpmKOghCQAOGGoJlU4YDyD+mFdlbJgHwHEYWyE3n+5Q6pku4K6cKCoZ2guZAAEAibML"
        + "ACMFQJAoBpUQGcZX//V0JIJ/d339WXR7O39TurRsS23/qq0u0E/+6u76aaZMQU1FMy4xMDCqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq"
        + "qqqqqqqqqq//OExKkXMFYYAN/6JKqqqqqqqqqqqqp6EqwQCGEi5jxwabDmBjA2xibhJAYLWABmD6Cbh68PKkY/oFUHrW+a2Mxl8bmNg8YhCphIHGAAAim7ltWz/9nvHPf9OjT7dj9yur9BBm1vqj36"
        + "vTbIe7jf/2/3Y99MQU1FMy4xMDBVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV//OExKkXOFoYAN/4JFVVVVVVVVVVVVVVVV"
        + "VVVVVVVVWLrELbmKDGhOnQnGBqgh5ivIrmYPQBXmBEi1Jwb38OYl2GIm4ogZRQ5iAlGDxWYGCxgUElx0+Hnvdv6P/ue//op///+hHYz7zT1f+i39f/Tt/vjVpMQU1FMy4xMDCqqqqqqqqqqqqqqqqq"
        + "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq//OExKAU6FocANf4JKqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqBWbpcBg8BNDjkzAxAFY"
        + "xUoFgMIFBCDA4RZYyb3z6MEoDABDEwSiAoTRkVDoTCgATUVrfK/2f//3/+j/vd//0f+97//kv8e6pMQU1FMy4xMDCqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq"
        + "qqqqqqqqqqqqqqqqqqqqqqqqqqqq//OExJQR0FogANf4JKqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqpxmaquIhIsuOAFMC7AYDFNw0owiQFRMHCFnTSm+60wuIL/NdxgysjzFpQMIC0EhBO9j6"
        + "vXmvf//////7v//R/71V/o7f97v///c6pMQU1FMy4xMDCqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq//OExJcSkFogANf4"
        + "JKqqqqqqqqqqqqqqqqqqqqqqqqqqXasZFEBBjJETdxDAsARUxTQbhMJRBqDCrxbo53zscMYFDCDt0lNWpgy8VzGYuMJA9MJrKdTTbXd+m39G93/p/7/r1/oQ3Y/7x3/7bf7v9Ohv/c5MQU1FMy4xMD"
        + "Cqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq//OExJ8UwFocANf4JKqqqqqqqqqqqqqqqlAkqRCCBhEwrA0n8wIMG8MRfKJj"
        + "CMgdkw3IT9PqdxATI3gps4RQIz2IkyLEwxQCcwpAQiAxOxL1r1lW//2W6dznddf2/2O6+r9BL3euOqq9PT7eMd/9DbfdddVMQU1FMy4xMDBVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
        + "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV//OExKcWmFoYANf6JFVVVVVVVVVVVVVVVVVVVVVVVVVVVVyqLEgGWBIqmYzBGAFA3pg3hO+YJ+CvGGBhJR8sSWYZFCCOH9DublHhn4Om"
        + "SQIPE0IC6E9bT6nv/d/6GI9lnd//t/9f/tT/+p6v0M/7sb+nShVMQU1FMy4xMDBVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
        + "VV//OExJ4UUFYcAN/4JFVVVVVVVVVVVVVVVVVVVVVVVVVVVWlKpigIFRUwA2MViTASAcQwZAnIMA2A8zCqgSo+SdTEMiIArj8YPHtYTOsWRAOJ5g4AFuFpyRun/9X6Uft7vq0/azd+vqX+khb+7V/2"
        + "f/u/0pVMQU1FMy4xMDBVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV//OExJ4UYFYcAN/4JFVVVVVVVVVVVVVVVXxSMBoOYM"
        + "MGNnpoUgYFiDqmIfk65giwC8YSaITnzqZPZkSYSSfWWxs0nmbBMY/C5iYHGFQaYBAKRDv4K21f/9z3u+r/+52+/V+xrNrPue7+iyy3Icc//To/61JMQU1FMy4xMDCqqqqqqqqqqqqqqqqqqqqqqqqq"
        + "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqBwB4//OExKcWqFoYAN/4JGAAgB5gEoCaNATRgNwFkYKAE/GHNBZBijw6EYiIOkGYFlDZ1mfioZy8csGSYCKxhEwK8Y"
        + "CeDFmC2gYZgO4DCGAEwoAAmAMgBpgBIAOx/s/T9v1/19P//fr/t9v3Adyv/R/vf/9H/31gBDoCD06zue3qcLESgt2CgWjAeAfMRIC0xXwMjJtEYM1B6kyYg4zKdfUMJYsgzSk/zAVAZVsMCICQwDwD"
        + "zM0AYMSsHe0qhyYdOCPc//OExLYaaFocAV8AAAewcB0VfgNCAxuNMLMAMSwNUCA0ZMBgnbh8ZsBYAKDJgDPBAMyRAOBgBGAMWH9cL2DNkUDjCDkUAx48DTFgNKjAwI0DKEAMqZ6uNA3Lg7DcuDsASJ"
        + "AxQwDFkQKhQMAIAwYsIQf75XIILYWyCDQPkUGgBgBwGFIhALAIGACiwSDgLCAEhwKBVb9XuaGjoGjoHnQPA3oGBwwKGOBxYYFDJA/MNlDvCXhjX1/7dms1ms1mjWH8//OExP9MhDpdv56gAFmjPEaM"
        + "qOohoyo5RMjKkCIqOaQL//9fZrNbbb/mpOl41LpeNS6XjxdNjxdNjxqbGhqbHgCWrclu+2kiamqzRECSGpBpq4oUPVQs5LVUJqJOwVAoalToKnYlWCsqdBU7Eqw1wVOxKVDW4FXaw1wVO6w16IiEoK"
        + "nSwMrBWDR1dR7iLlj3BrqPcRdR7iLqPcRdR7iJTEFNRTMuMTAwVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV//OExIAXsHqdn8kYAlVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
        + "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
        + "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV";

    private const string Stereo44k = ""
        + "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"
        + "BpmoSRr2nZ1lRhjFY+WdAR2ynOJj7pjHAXodqxOa3tqaaMwaS1yZam+YSAwXfYY7cMUlsM4X/0Weu//oyOKHdv7vdXUja1m6n14q77nb1a9KDz//KqOjpyY8ZIWmepRry6cpUmBpgkhgywWKYuPoUH"
        + "37j/phhYOQcSYJnkoGHgGDQQYSBoCCSloe7Vp/Vs3+rq0dv+x//q/u2f/9Nmv2b0xBTUUzLjEwMFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
        + "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVAAAEltkDaSaQxe0GlGJeYF4Yhv/7kkTXD/IKCUQDv+CQSeFoYG/7QgqUKQoO/6JBNQWhgb/xCDlNUHv2AYmE6NzJS6"
        + "RgQCpIyLkDToHZZobZYtb6Ep/2KZ9NP7f/u3er/9/Zr/0//7JTZvThhFJlXhsoJ6sZgWYEsYLwCwmK+tOx8j4gCYXOBrHCCWZ9AxjIBAwKCAHl7msv///+r9H/////+kDGIZtkObBJMdfzWYyEOyHR"
        + "V8Khz2Q7wYyaFsnbEPGtDbGlLPmiliGVRxGEYMl3Fzuw/8vsX7wBOf/7P7GV97l3/1ZvR9OtX+3uT92lltbNSpQQAAAIj//a17nkVikLQWaSnSLFB6JhuhbHOyAKRBpFAFiF7BHYidwX67Xd+6vrv+"
        + "/9V39H6v6On/+yr1P2dSpMQU1FMy4xMDCqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqog"
        + "BA67a7SRuLCpzLREZjAITTLfCzsEsAgQiBVJNNWMr6f/+5JEyIgCCwjM6z7gjD3haIBr/EIKGCsKDv+iQPyFZby8+Mx62sisCtex7/isDXMQrb+v2/9fi//d+7/3yYCAAABFG96gv5/uDWWMzWQ9LQ"
        + "Qws7GDDzQDOsEIYwdQCwfQOOmA1h25Rn1kv9VzUf+qVo7Wf3//7////06iI/zQQjDcNPzy6nDHPx/c7wDyNO3fIEzHRgyA9FmU2Zfk1IeI0tzcy5QswmCAtWuRrb+Sy3eaTNOf6rXCHTW3oqsapXih"
        + "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"
        + "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqoxhIIxSMMxdT4yUnMwV8eCMmw6EDJaR2QwRQJDMROg0gEDcYyOHJM1kIRIlKheaiti1X/+9//3Af/7kkTBAPHbCc/rumEsQa"
        + "FogHP8UAd4JUOubeSxIIWhwa/xCNEDqxMywWM9EwM6mDOJgKAJEYIqGKmJAdpx3xRIyYTMEJHy/JwUga6wmcl5lwgCidLh3SpG1PvZ/Vt7ez/Sz//d+j/q+mj+gwMIBHMFRAPTCKgOAw9kH5Mi0GDj"
        + "4abbQ9woZEMinCvz7ydjaiQjSGQTOb4DIJRDBULxAAiq7SH/A4I5JmY1GX/0bK9UrrdWGOU5Q9FSpnpKucllu4cWvAr1M/pZQElNImykUcLOWOf6XkXGw+tMUUSdr/98ZEHKWLCoHD0YkBCYfEKYEp"
        + "MYY4GYLVzVGgMEa5gYAQAaP0mYPxmaOZOKg5MGh9VnpV/Vs//2//3f0fv7tGj/9FFMQU1FMy4xMDBVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
        + "VVVVVVVVVVVVIAxeIExPMIxSUMyBnYwTQeTMil6fzIYB3UwOcJKMdvM0uHzcwnOIn41qGhYkqie2T2Aj9/5Cj///+5JE2w/xqQlEg7/gnEphaGBv+0IOSHkGD/SmwQyFocHf7UD///////////d/6x"
        + "TLBaMZUBmfjhqh6bNMmA0glhgUgX6YSLxqGvxkW5gkoPsalwmCuBg5iDi8aQRoeVa//9f9P6v2/9////Z/+z+ozLFQ1WCA4xME/eb4yMMVfPjKoaT3aRZsyLIKOPuZnNn4/NDJCMzfnMcVJMDguEYC"
        + "rANzfSWUd4+0ide3n5eiQSi2ng6tPakpawcaURY1V7GzWorwEVwEp/Fnlp/uMacwRrJWJnXqQr/7iOi50YwZUTmhoRsSychXmBggnBgtgX+YpHw/HqskVJhWYPgfdrmwsBhQmBgoxgHBQ0nz33/7q/"
        + "9qPV//V/7v1J1d/7KvV/WqTEFNRTMuMTAwqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqMWyAMSDDMSVCMbJ7"
        + "MEJHnjHiOtwx00eAMC2CUTKD6NOCw3WGDiRrNbBUaJSr//uSRNsN8gkJRIu/4JhCoWhwb/tCDPArBg7/okEjheGBv+0IXuk1sWq7v03Xs//ukIfMuiAwgLzEpGMiE80AgTZbMPOecxfGQWP3DFfzDJ"
        + "QTA48dzQogCDIIgoBAiWWYj7f/t9H/Z2f93//79n//RZ/aZpDGaugGcaE4fysCZGcJSHyOxaR73AnQZF2Eun2kqGyUeGe8imWH2GLilGBIXigCLBtwf+XzlvhYu0hP9U5Hr0DHqcXaQiiy6W2kuxk5"
        + "s2Dy1DfyDWvSRvm1xQPscoYbSzLvJXhEeqKGGQ7T/6ulU6ZE8FlZg3xonJ4pJgW4CsYMYBZmLpm0x+WIQKYYuAuHHAWDouZADRggPDITQ+eE7/+v//7P/7//93/R/+n11UxBTUUzLjEwMFVVVVVVVV"
        + "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVQAIFu2u0jbhakLsTdHXCAZTE/wTlYxkYIwafbHVfOde7B1HYCMun"
        + "e//7kkTVj/G2CUSDv+CcRCFocHP8UA2ALQYO/6JBCQWiAa/xCFT6apF8+x39f5Jd6Pu9P/du//3/v/WnxJABb3XSRqTmvoYkz0RJeUuKQYXled3gEPA20CLyx2InUF+Xofjuze9H2fV/06y2z7/+j/"
        + "/r+/9Zm2OAZYJxyPJ/UnxkbIfCfK45xnv7iFZkXwRUfYyGbDxyZwSMZJ/mYjqaYBBcKgKshub6SyfvWovZ2l0OIIIIa5vyKyeOYshSK2kGsP8W7UCyHZgw+bXaiqBb7o9HoKorZY+djGD0/+syA6eA"
        + "8ZFSZZgYMiaYsYEuBNmC+A8Bi58suflmLdGGLgpJyRNGnTGZeJxiwYGGwuYCADE8/d/6vu///9H///v3/t/v/WpMQU1FMy4xMDCqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq"
        + "qqqqqqqqqqqqqqqqqqqqqqqqqqqqpAAg67f7WyShxAjJRAhR0OjB6hDb6Q6poNMfNvWo3+PnJre9mhD1RVYpXSuqjT/7turR//gP3//Wj6dP//+5JE0wAB9glPax3QjDjhWf0F/ReM0CsGDv+iQRkF"
        + "4cGv8Qj+Y4KnSCxlY2ZwMCTUFWYwGEEVMEyC+jE4eY081kjSMKbCAz+PA42PNgYDODIy0TAxAn36Jj/s/9iv2f7T//tTu2v0//9NOr2mBwAP5gsoAsYR4AnGH2gfJkbwYYfMes7HwTBsBkYYPmfURc"
        + "a9RoZlyOY6foYXKYCQvHAEWW3CGwOCOWaiUbb/vOnSr9m00bZqaolLPP1o13vVUX/VK03qXr/6cWM3sXSgi9lJlFqbnbUm0rY0NyAXV3//OrAhKQEK5E8mijhm8AIgSEwFgL8MHt5LTVPyMQwQUH9N"
        + "e6DPX8zhIMoGAMniw4q//s6v6df////p/f36dH/Zop/tTEFNRTMuMTAwqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqkACDt//tbJKHCCiKk"
        + "EMQB2YH0OZnaBBNNk75Oc1K918pHYCOhuRUxKWJ9lf9f/FG6PX/r1u2ITyNv/03Vv4//uSRN0A8dkJ0On9yIxJQWhgb/tCDikXBg/0psENhaHBv+0Iv//1eASkgJWNAFzUzg2GWMBaBLzAhwwQwczl"
        + "rNMAIzDA4gf0zfiETmYcaAI0JkYeHVXc9zf7a/9v79f/R//////v/1mB0gRRgtABmYR+AQmH4AVRkcoRKfNcepnwihQpkYQN0fRw2a2xeZZSMYp/uYKqeIAwJQFWw6cQAgaONXe7ov9nVKabyrTr/9"
        + "mqd1/1RU39aU76/dujL9mb1d3Z+uv/Js7za+1x22kCtzNjJ1w4VUn/3kOCR1AqZaRGiH5sqscfYGBbAoJgqoYsYmR0vnjWEeRhQAQGe3wGqM5gI0GCJjoEDh1PpPIL/9u//Vp9P9rf6P/dt0/q/t1+"
        + "2kxBTUUzLjEwMKqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqogAgW7ba2SOLmTJTuKgHkozGBHkGHyEI0KBE1GPrhDaPiiof"
        + "uyxUxdhXcxyS2zit9zV/YtA//7kkTeAPH1CVDp/ciMQwF4cG/7Qg3tXwYP9KbBJgWhgb/tCK91KJ4VrsRqsPUin///TT/8yIBw2YfMVLDKEU0lOOBhjAxgNswZEIKMXAmxz7yheowwgFLOKoIz8Kxo"
        + "tkIRMAg4sixLv2/z39+z9//6f/s0/r7v/6zA7gJcwW0BKEhIEw/EBRMjmBEj5vxrw+FQGMMjBBfz5iBjV6KjJuRzCz+AspxCGJYARR94IbA4IPLNKu2//yk67/mLmXN36c2/0f3XRcv2T1uiR810aR"
        + "nGMbIvbFjWxMf3IclQeaHlkBRyk//6Khwzh8EqjEPzTujy0DAtgGkwYsDkMXEROz8FApIww4B/OMhwaiBj8KmBQ6SBBBl0tf/2///f/+j//9MCYKlyy2IAAI4SYAgC5gDYAuYV+KGmIFTH5ljh6uZI"
        + "OK4mAOgBqhpgAABgYAmBKmHvh6B9NwGQZIwPDGA1AFxgEgAg2VLM8UMgkzIvHGsOJBJdxOiMtbceaYYyySpiFvIFYe1+SKYK4esu4j5IX/i8kZQyyC0xHcvV6fUYll1/Icv/+5JE2YDyIwnPa7phLE"
        + "VheHBv/EINZPMGD/SmwPQF4gGv8QjZ9/DDdIIvgyJBgDA3gG4DAJpggoQWYR2EBmHKjUZiq2M+Yz8R1mJWhNJg4IE+YMaDcmDmhN5gtIVgYYAHxn0rgSZkhI7yYZKDYmC7gfY6AYC0A6881qkHN5yJ"
        + "u2/dZy4fypJZuVy+5DcP5Uljcrl9x/4f7Uw3T27kbt9w5+ef17b5MRh+XeB///s//V+/d//9tIiGQ8HY8Hg9Ho0GolBjQEmGdIQ3BSgX+ZEoOxgNDIMWWC/zOui1MLUCRH5Q3/NZ2k+bkWlP3/nJUS"
        + "ZOOZtNrtZlV//812XTMwNM0oEz+P31lUuq//+AmIYrJZlUUmVAoYBEs1Lu1qb///MahcyIGhQDmGwSY0DQ0AbU1GpdWjX///4KBZiQLBwQYmYSCAGCkHRrVNKtVpV////5fxAA6i0EHGSNkRUWp+sv"
        + "/L/y//////f5QBajnLkYI0NwGCMngjfNb5+8f3j///////2JfqxL+4U9v6e39en+KaC9NigAQAYSRhsRBsSBoGGg//uSZP+ABEskzW1/AA6JBBiFr+gAG+lfZbnuEBr7qedrOzAA7GhiNgELjFEKvy"
        + "NqwCMf0vMRixMEhP3s2r0AEAmYyCOVQ8/wDCAbECkRwYHwAWeA2MC/w5JI8LFBPwYgDlBSQ6SIlLw0MLoBfQMLiuDGi4TQhozPxcAnMpEEFxmxOlEc0ipiRX8nBO5E0Bc5B3Lqy6TMxLv8i5Pm5mRc"
        + "3Ok2bzEmVLLqTmP/RIuT54myfKhmT5UOpOYmRstI2pG3/zc0WmaM5ocBAocER4Gix4Rf8p/+pUxBTUUzLjEwMFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
        + "VVVVVVVVVVVVVVVVVVVVVVVUxBTUUzLjEwMFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
        + "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVf/7kmRAj/AAAGkHAAAIAAANIOAAAQAAAaQAAAAgAAA0gAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
        + "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
        + "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
        + "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
        + "VVU=";

    private static double[] Tone(int sampleRate, int n, double hz)
        => Enumerable.Range(0, n).Select(i => 0.4 * Math.Sin(2 * Math.PI * hz * i / sampleRate)).ToArray();

    // The best normalized correlation and the SNR (dB) of the decoded channel against the source over lags 0..3000.
    private static (double Correlation, double Snr) Compare(float[] interleaved, int channels, int channel, double[] source)
    {
        int frames = interleaved.Length / channels;
        double bestCorrelation = double.NegativeInfinity, bestSnr = 0;
        for (int lag = 0; lag <= 3000; lag++)
        {
            double dot = 0, ee = 0, ss = 0, err = 0;
            int n = Math.Min(source.Length - 2000, frames - lag);
            for (int i = 1000; i < n; i++)
            {
                double d = interleaved[(i + lag) * channels + channel], s = source[i];
                dot += d * s;
                ee += d * d;
                ss += s * s;
                err += (d - s) * (d - s);
            }
            double c = dot / Math.Sqrt(ee * ss + 1e-30);
            if (c > bestCorrelation)
            {
                bestCorrelation = c;
                bestSnr = 10 * Math.Log10(ss / Math.Max(err, 1e-30));
            }
        }
        return (bestCorrelation, bestSnr);
    }

    [Fact(Timeout = 60000)]
    public async Task Mpeg2MonoTone_DecodesToTheSourceTone()
    {
        await Task.Yield();
        var result = Mp3Decoder.Decode(Convert.FromBase64String(Mono24k));
        Assert.Equal(24000, result.SampleRate);
        Assert.Equal(1, result.Channels);
        Assert.InRange(result.Samples.Length, 12000, 12000 + 4 * 576 + 1152);
        var (correlation, snr) = Compare(result.Samples, 1, 0, Tone(24000, 12000, 440));
        Assert.True(correlation > 0.999, $"correlation {correlation}");
        Assert.True(snr > 25, $"SNR {snr} dB");
    }

    [Fact(Timeout = 60000)]
    public async Task Mpeg1JointStereo_DecodesEachChannelsTone()
    {
        await Task.Yield();
        var result = Mp3Decoder.Decode(Convert.FromBase64String(Stereo44k));
        Assert.Equal(44100, result.SampleRate);
        Assert.Equal(2, result.Channels);
        var (left, leftSnr) = Compare(result.Samples, 2, 0, Tone(44100, 22050, 440));
        var (right, rightSnr) = Compare(result.Samples, 2, 1, Tone(44100, 22050, 1000));
        Assert.True(left > 0.999 && leftSnr > 25, $"left: correlation {left}, SNR {leftSnr} dB");
        Assert.True(right > 0.999 && rightSnr > 25, $"right: correlation {right}, SNR {rightSnr} dB");
        // The channels are not swapped or mixed: the left channel does not carry the right channel's tone.
        var (cross, _) = Compare(result.Samples, 2, 0, Tone(44100, 22050, 1000));
        Assert.True(cross < 0.1, $"cross-channel correlation {cross}");
    }

    [Fact(Timeout = 60000)]
    public async Task DataWithoutFrames_IsRejected()
    {
        await Task.Yield();
        Assert.Throws<System.IO.InvalidDataException>(() => Mp3Decoder.Decode(new byte[1000]));
    }
}
