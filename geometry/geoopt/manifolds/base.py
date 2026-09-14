import abc
import torch.nn
import itertools
from typing import Optional, Tuple, Union

__all__ = ["Manifold", "ScalingInfo"]


class ScalingInfo(object):














    
    NotCompatible = object()
    __slots__ = ["kwargs", "results"]

    def __init__(self, *results: float, **kwargs: float):
        self.results = results
        self.kwargs = kwargs


class ScalingStorage(dict):
















































    def __call__(self, scaling_info: ScalingInfo, *aliases):
        def register(fn):
            self[fn.__name__] = scaling_info
            for alias in aliases:
                self[alias] = scaling_info
            return fn

        return register

    def copy(self):
        return self.__class__(self)


class Manifold(torch.nn.Module, metaclass=abc.ABCMeta):
    __scaling__ = ScalingStorage()  
    name = None
    ndim = None
    reversible = None

    forward = NotImplemented

    def __init__(self, **kwargs):
        super().__init__()

    @property
    def device(self) -> Optional[torch.device]:






        p = next(itertools.chain(self.buffers(), self.parameters()), None)
        if p is not None:
            return p.device
        else:
            return None

    @property
    def dtype(self) -> Optional[torch.dtype]:







        p = next(itertools.chain(self.buffers(), self.parameters()), None)
        if p is not None:
            return p.dtype
        else:
            return None

    def check_point(
        self, x: torch.Tensor, *, explain=False
    ) -> Union[Tuple[bool, Optional[str]], bool]:


















        ok, reason = self._check_shape(x.shape, "x")
        if explain:
            return ok, reason
        else:
            return ok

    def assert_check_point(self, x: torch.Tensor):












        ok, reason = self._check_shape(x.shape, "x")
        if not ok:
            raise ValueError(
                "`x` seems to be not valid "
                "tensor for {} manifold.\nerror: {}".format(self.name, reason)
            )

    def check_vector(self, u: torch.Tensor, *, explain=False):


















        ok, reason = self._check_shape(u.shape, "u")
        if explain:
            return ok, reason
        else:
            return ok

    def assert_check_vector(self, u: torch.Tensor):












        ok, reason = self._check_shape(u.shape, "u")
        if not ok:
            raise ValueError(
                "`u` seems to be not valid "
                "tensor for {} manifold.\nerror: {}".format(self.name, reason)
            )

    def check_point_on_manifold(
        self, x: torch.Tensor, *, explain=False, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:






















        ok, reason = self._check_shape(x.shape, "x")
        if ok:
            ok, reason = self._check_point_on_manifold(x, atol=atol, rtol=rtol)
        if explain:
            return ok, reason
        else:
            return ok

    def assert_check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5):











        self.assert_check_point(x)
        ok, reason = self._check_point_on_manifold(x, atol=atol, rtol=rtol)
        if not ok:
            raise ValueError(
                "`x` seems to be a tensor "
                "not lying on {} manifold.\nerror: {}".format(self.name, reason)
            )

    def check_vector_on_tangent(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        *,
        ok_point=False,
        explain=False,
        atol=1e-5,
        rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:






















        if not ok_point:
            ok, reason = self._check_shape(x.shape, "x")
            if ok:
                ok, reason = self._check_shape(u.shape, "u")
            if ok:
                ok, reason = self._check_point_on_manifold(x, atol=atol, rtol=rtol)
        else:
            ok = True
            reason = None
        if ok:
            ok, reason = self._check_vector_on_tangent(x, u, atol=atol, rtol=rtol)
        if explain:
            return ok, reason
        else:
            return ok

    def assert_check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, ok_point=False, atol=1e-5, rtol=1e-5
    ):















        if not ok_point:
            ok, reason = self._check_shape(x.shape, "x")
            if ok:
                ok, reason = self._check_shape(u.shape, "u")
            if ok:
                ok, reason = self._check_point_on_manifold(x, atol=atol, rtol=rtol)
        else:
            ok = True
            reason = None
        if ok:
            ok, reason = self._check_vector_on_tangent(x, u, atol=atol, rtol=rtol)
        if not ok:
            raise ValueError(
                "`u` seems to be a tensor "
                "not lying on tangent space to `x` for {} manifold.\nerror: {}".format(
                    self.name, reason
                )
            )

    @__scaling__(ScalingInfo(1))
    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) -> torch.Tensor:
















        raise NotImplementedError

    @__scaling__(ScalingInfo(2))
    def dist2(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) -> torch.Tensor:
















        return self.dist(x, y, keepdim=keepdim) ** 2

    @abc.abstractmethod
    @__scaling__(ScalingInfo(u=-1))
    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:














        raise NotImplementedError

    @abc.abstractmethod
    @__scaling__(ScalingInfo(u=-1))
    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:














        raise NotImplementedError

    @__scaling__(ScalingInfo(1))
    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:














        raise NotImplementedError

    @__scaling__(ScalingInfo(u=-1))
    def expmap_transp(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
















        y = self.expmap(x, u)
        v_transp = self.transp(x, y, v)
        return y, v_transp

    @__scaling__(ScalingInfo(u=-1))
    def retr_transp(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:




















        y = self.retr(x, u)
        v_transp = self.transp(x, y, v)
        return y, v_transp

    @__scaling__(ScalingInfo(u=-1))
    def transp_follow_retr(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:


















        y = self.retr(x, u)
        return self.transp(x, y, v)

    @__scaling__(ScalingInfo(u=-1))
    def transp_follow_expmap(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:




















        y = self.expmap(x, u)
        return self.transp(x, y, v)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
















        raise NotImplementedError

    @abc.abstractmethod
    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:


















        raise NotImplementedError

    def component_inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None
    ) -> torch.Tensor:


























        return self.inner(x, u, v, keepdim=True)

    def norm(self, x: torch.Tensor, u: torch.Tensor, *, keepdim=False) -> torch.Tensor:
















        return self.inner(x, u, keepdim=keepdim) ** 0.5

    @abc.abstractmethod
    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:














        raise NotImplementedError

    @abc.abstractmethod
    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:














        raise NotImplementedError

    @abc.abstractmethod
    def projx(self, x: torch.Tensor) -> torch.Tensor:












        raise NotImplementedError

    def _check_shape(
        self, shape: Tuple[int], name: str
    ) -> Union[Tuple[bool, Optional[str]], bool]:



















        ok = len(shape) >= self.ndim
        if not ok:
            reason = "'{}' on the {} requires more than {} dim".format(
                name, self, self.ndim
            )
        else:
            reason = None
        return ok, reason

    def _assert_check_shape(self, shape: Tuple[int], name: str):

















        ok, reason = self._check_shape(shape, name)
        if not ok:
            raise ValueError(reason)

    @abc.abstractmethod
    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:























        
        raise NotImplementedError

    @abc.abstractmethod
    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
























        
        raise NotImplementedError

    def extra_repr(self):
        return ""

    def __repr__(self):
        extra = self.extra_repr()
        if extra:
            return self.name + "({}) manifold".format(extra)
        else:
            return self.name + " manifold"

    def unpack_tensor(self, tensor: torch.Tensor) -> torch.Tensor:















        return tensor

    def pack_point(self, *tensors: torch.Tensor) -> torch.Tensor:













        if len(tensors) != 1:
            raise ValueError("1 tensor expected, got {}".format(len(tensors)))
        return tensors[0]

    def random(self, *size, dtype=None, device=None, **kwargs) -> torch.Tensor:





        raise NotImplementedError

    def origin(
        self,
        *size: Union[int, Tuple[int]],
        dtype=None,
        device=None,
        seed: Optional[int] = 42
    ) -> torch.Tensor:






















        if seed is not None:
            
            state = torch.random.get_rng_state()
            torch.random.manual_seed(seed)
            try:
                return self.random(*size, dtype=dtype, device=device)
            finally:
                torch.random.set_rng_state(state)
        else:
            return self.random(*size, dtype=dtype, device=device)
