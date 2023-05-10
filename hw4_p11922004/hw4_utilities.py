import math

import numpy as np

from numpy.core import asarray, zeros, swapaxes, conjugate, take, sqrt
from . import _pocketfft_internal as pfi
from numpy.core.multiarray import normalize_axis_index


class FFT(object):
    def computeSinglePoint2DFT(self, img, u, v, N):
        result = 0 + 0j
        for x in range(N):
            for y in range(N):
                result += (img[x, y] * (math.cos((2 * math.pi * (u * x + v * y)) / N) -
                                         (1j * math.sin((2 * math.pi * (u * x + v * y)) / N))))
        return result

    @classmethod
    def computeForward2DDFTNoSeparability(self, img):
        N = img.shape[0]
        final2DDFT = np.zeros([N, N], dtype=np.complex)
        for u in range(N):
            for v in range(N):
                final2DDFT[u, v] = FFT.computeSinglePoint2DFT(img, u, v, N)
        return ((1.0 / (N ** 2)) * final2DDFT)

    @classmethod
    def computeCenteredImage(self, img):
        M, N = img.shape
        newImge = np.zeros([M, N], dtype=int)
        for x in range(M):
            for y in range(N):
                newImge[x, y] = img[x, y] * ((-1) ** (x + y))

        # newImge = img * centeringMatrix
        return newImge

    @classmethod
    def __computeSingleW(self, num, denom):
        """Computes one value of W from the given numerator and denominator values. """
        return math.cos((2 * math.pi * num) / denom) - (1j * math.sin((2 * math.pi * num) / denom))

    @classmethod
    def __computeW(self, val, denom, oneD=True):
        """Computes 1D or 2D values of W from the given numerator and denominator values."""
        if oneD:
            result = np.zeros([val, 1], dtype=np.complex)
            for i in range(val):
                result[i] = FFT.__computeSingleW(i, denom)
        else:
            result = np.zeros([val, val], dtype=np.complex)
            for i in range(val):
                for j in range(val):
                    result[i, j] = FFT.__computeSingleW((i + j), denom)
        return result

    @classmethod
    def computeFFT(self, img):
        """Computes the FFT of a given image.
        """

        # Compute size of the given image
        N = img.shape[0]

        # Compute the FFT for the base case (which uses the normal DFT)
        if N == 2:
            return FFT.computeForward2DDFTNoSeparability(img)

        # Otherwise compute FFT recursively

        # Divide the original image into even and odd
        imgEE = np.array([[img[i, j] for i in range(0, N, 2)] for j in range(0, N, 2)]).T
        imgEO = np.array([[img[i, j] for i in range(0, N, 2)] for j in range(1, N, 2)]).T
        imgOE = np.array([[img[i, j] for i in range(1, N, 2)] for j in range(0, N, 2)]).T
        imgOO = np.array([[img[i, j] for i in range(1, N, 2)] for j in range(1, N, 2)]).T

        # Compute FFT for each of the above divided images
        FeeUV = FFT.computeFFT(imgEE)
        FeoUV = FFT.computeFFT(imgEO)
        FoeUV = FFT.computeFFT(imgOE)
        FooUV = FFT.computeFFT(imgOO)

        # Compute also Ws
        Wu = FFT.__computeW(N / 2, N)
        Wv = Wu.T  # Transpose
        Wuv = FFT.__computeW(N / 2, N, oneD=False)

        # Compute F(u,v) for u,v = 0,1,2,...,N/2
        imgFuv = 0.25 * (FeeUV + (FeoUV * Wv) + (FoeUV * Wu) + (FooUV * Wuv))

        # Compute F(u, v+M) where M = N/2
        imgFuMv = 0.25 * (FeeUV + (FeoUV * Wv) - (FoeUV * Wu) - (FooUV * Wuv))

        # Compute F(u+M, v) where M = N/2
        imgFuvM = 0.25 * (FeeUV - (FeoUV * Wv) + (FoeUV * Wu) - (FooUV * Wuv))

        # Compute F(u+M, v+M) where M = N/2
        imgFuMvM = 0.25 * (FeeUV - (FeoUV * Wv) - (FoeUV * Wu) + (FooUV * Wuv))

        imgF1 = np.hstack((imgFuv, imgFuvM))
        imgF2 = np.hstack((imgFuMv, imgFuMvM))
        imgFFT = np.vstack((imgF1, imgF2))

        return imgFFT

    @classmethod
    def computeInverseFFT(self, imgFFT):
        N = imgFFT.shape[0]
        return np.real(np.conjugate(FFT.computeFFT(np.conjugate(imgFFT) * (N ** 2))) * (N ** 2))

def _raw_fft(a, n, axis, is_real, is_forward, inv_norm):
    axis = normalize_axis_index(axis, a.ndim)
    if n is None:
        n = a.shape[axis]

    fct = 1/inv_norm

    if a.shape[axis] != n:
        s = list(a.shape)
        index = [slice(None)]*len(s)
        if s[axis] > n:
            index[axis] = slice(0, n)
            a = a[tuple(index)]
        else:
            index[axis] = slice(0, s[axis])
            s[axis] = n
            z = zeros(s, a.dtype.char)
            z[tuple(index)] = a
            a = z

    if axis == a.ndim-1:
        r = pfi.execute(a, is_real, is_forward, fct)
    else:
        a = swapaxes(a, axis, -1)
        r = pfi.execute(a, is_real, is_forward, fct)
        r = swapaxes(r, axis, -1)
    return r


def _get_forward_norm(n, norm):
    if n < 1:
        raise ValueError(f"Invalid number of FFT data points ({n}) specified.")

    if norm is None or norm == "backward":
        return 1
    elif norm == "ortho":
        return sqrt(n)
    elif norm == "forward":
        return n
    raise ValueError(f'Invalid norm value {norm}; should be "backward",'
                     '"ortho" or "forward".')


def _get_backward_norm(n, norm):
    if n < 1:
        raise ValueError(f"Invalid number of FFT data points ({n}) specified.")

    if norm is None or norm == "backward":
        return n
    elif norm == "ortho":
        return sqrt(n)
    elif norm == "forward":
        return 1
    raise ValueError(f'Invalid norm value {norm}; should be "backward", '
                     '"ortho" or "forward".')


_SWAP_DIRECTION_MAP = {"backward": "forward", None: "forward",
                       "ortho": "ortho", "forward": "backward"}


def _swap_direction(norm):
    try:
        return _SWAP_DIRECTION_MAP[norm]
    except KeyError:
        raise ValueError(f'Invalid norm value {norm}; should be "backward", '
                         '"ortho" or "forward".') from None


def _fft_dispatcher(a, n=None, axis=None, norm=None):
    return (a,)


def fft(a, n=None, axis=-1, norm=None):
    a = asarray(a)
    if n is None:
        n = a.shape[axis]
    inv_norm = _get_forward_norm(n, norm)
    output = _raw_fft(a, n, axis, False, True, inv_norm)
    return output


def ifft(a, n=None, axis=-1, norm=None):
    a = asarray(a)
    if n is None:
        n = a.shape[axis]
    inv_norm = _get_backward_norm(n, norm)
    output = _raw_fft(a, n, axis, False, False, inv_norm)
    return output


def rfft(a, n=None, axis=-1, norm=None):
    a = asarray(a)
    if n is None:
        n = a.shape[axis]
    inv_norm = _get_forward_norm(n, norm)
    output = _raw_fft(a, n, axis, True, True, inv_norm)
    return output


def irfft(a, n=None, axis=-1, norm=None):
    a = asarray(a)
    if n is None:
        n = (a.shape[axis] - 1) * 2
    inv_norm = _get_backward_norm(n, norm)
    output = _raw_fft(a, n, axis, True, False, inv_norm)
    return output


def hfft(a, n=None, axis=-1, norm=None):
    a = asarray(a)
    if n is None:
        n = (a.shape[axis] - 1) * 2
    new_norm = _swap_direction(norm)
    output = irfft(conjugate(a), n, axis, norm=new_norm)
    return output


def ihfft(a, n=None, axis=-1, norm=None):
    a = asarray(a)
    if n is None:
        n = a.shape[axis]
    new_norm = _swap_direction(norm)
    output = conjugate(rfft(a, n, axis, norm=new_norm))
    return output


def _cook_nd_args(a, s=None, axes=None, invreal=0):
    if s is None:
        shapeless = 1
        if axes is None:
            s = list(a.shape)
        else:
            s = take(a.shape, axes)
    else:
        shapeless = 0
    s = list(s)
    if axes is None:
        axes = list(range(-len(s), 0))
    if len(s) != len(axes):
        raise ValueError("Shape and axes have different lengths.")
    if invreal and shapeless:
        s[-1] = (a.shape[axes[-1]] - 1) * 2
    return s, axes


def _raw_fftnd(a, s=None, axes=None, function=fft, norm=None):
    a = asarray(a)
    s, axes = _cook_nd_args(a, s, axes)
    itl = list(range(len(axes)))
    itl.reverse()
    for ii in itl:
        a = function(a, n=s[ii], axis=axes[ii], norm=norm)
    return a


def _fftn_dispatcher(a, s=None, axes=None, norm=None):
    return (a,)


def fftn(a, s=None, axes=None, norm=None):
    return _raw_fftnd(a, s, axes, fft, norm)


def ifftn(a, s=None, axes=None, norm=None):
    return _raw_fftnd(a, s, axes, ifft, norm)


def fft2(a, s=None, axes=(-2, -1), norm=None):
    return _raw_fftnd(a, s, axes, fft, norm)


def ifft2(a, s=None, axes=(-2, -1), norm=None):
    return _raw_fftnd(a, s, axes, ifft, norm)


def rfftn(a, s=None, axes=None, norm=None):
    a = asarray(a)
    s, axes = _cook_nd_args(a, s, axes)
    a = rfft(a, s[-1], axes[-1], norm)
    for ii in range(len(axes)-1):
        a = fft(a, s[ii], axes[ii], norm)
    return a


def rfft2(a, s=None, axes=(-2, -1), norm=None):
    return rfftn(a, s, axes, norm)


def irfftn(a, s=None, axes=None, norm=None):
    a = asarray(a)
    s, axes = _cook_nd_args(a, s, axes, invreal=1)
    for ii in range(len(axes)-1):
        a = ifft(a, s[ii], axes[ii], norm)
    a = irfft(a, s[-1], axes[-1], norm)
    return a


def irfft2(a, s=None, axes=(-2, -1), norm=None):
    return irfftn(a, s, axes, norm)