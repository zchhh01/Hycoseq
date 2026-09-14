import torch.optim

from .mixin import OptimMixin
from ..tensor import ManifoldParameter, ManifoldTensor


__all__ = ["RiemannianAdam"]


class RiemannianAdam(OptimMixin, torch.optim.Adam):


































    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                betas = group["betas"]
                weight_decay = group["weight_decay"]
                eps = group["eps"]
                learning_rate = group["lr"]
                amsgrad = group["amsgrad"]
                stablilize = False
                for point in group["params"]:
                    grad = point.grad
                    if grad is None:
                        continue
                    if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                        manifold = point.manifold
                    else:
                        manifold = self._default_manifold

                    if grad.is_sparse:
                        raise RuntimeError(
                            "RiemannianAdam does not support sparse gradients, use SparseRiemannianAdam instead"
                        )

                    state = self.state[point]

                    
                    if len(state) == 0:
                        state["step"] = 0
                        
                        state["exp_avg"] = torch.zeros_like(point)
                        
                        state["exp_avg_sq"] = torch.zeros_like(point)
                        if amsgrad:
                            
                            state["max_exp_avg_sq"] = torch.zeros_like(point)
                    state["step"] += 1
                    
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    
                    grad.add_(point, alpha=weight_decay)
                    grad = manifold.egrad2rgrad(point, grad)
                    exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                    exp_avg_sq.mul_(betas[1]).add_(
                        manifold.component_inner(point, grad), alpha=1 - betas[1]
                    )
                    bias_correction1 = 1 - betas[0] ** state["step"]
                    bias_correction2 = 1 - betas[1] ** state["step"]
                    if amsgrad:
                        max_exp_avg_sq = state["max_exp_avg_sq"]
                        
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        
                        denom = max_exp_avg_sq.div(bias_correction2).sqrt_()
                    else:
                        denom = exp_avg_sq.div(bias_correction2).sqrt_()
                    
                    
                    direction = exp_avg.div(bias_correction1) / denom.add_(eps)
                    
                    new_point, exp_avg_new = manifold.retr_transp(
                        point, -learning_rate * direction, exp_avg
                    )
                    
                    point.copy_(new_point)
                    exp_avg.copy_(exp_avg_new)

                    if (
                        group["stabilize"] is not None
                        and state["step"] % group["stabilize"] == 0
                    ):
                        stablilize = True
                if stablilize:
                    self.stabilize_group(group)
        return loss

    @torch.no_grad()
    def stabilize_group(self, group):
        for p in group["params"]:
            if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                continue
            state = self.state[p]
            if not state:  
                continue
            manifold = p.manifold
            exp_avg = state["exp_avg"]
            p.copy_(manifold.projx(p))
            exp_avg.copy_(manifold.proju(p, exp_avg))
