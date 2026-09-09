import torch
import torch.nn as nn

class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class SteeringHookHandler:
    def __init__(self, v_steer, alpha_0=18.0, K=16, schedule="linear_decay"):
        self.v_steer = v_steer
        self.alpha_0 = alpha_0
        self.K = K
        self.schedule = schedule
        self.t = 0
        self.injection_log = []

    def reset(self):
        self.t = 0
        self.injection_log = []

    def hook_fn(self, module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        # If sequence length > 1, this is prefill (t=0)
        if out.shape[1] > 1:
            self.t = 0
            self.injection_log.append((self.t, 0.0, "prefill_unsteered"))
            return output
        # Single token decoding (t >= 1)
        self.t += 1
        if self.schedule == "linear_decay":
            a_t = self.alpha_0 * max(0.0, 1.0 - (self.t - 1) / self.K)
        elif self.schedule == "hard_cutoff":
            a_t = self.alpha_0 if self.t <= self.K else 0.0
        elif self.schedule == "continuous":
            a_t = self.alpha_0
        else:
            a_t = 0.0

        if a_t != 0.0:
            steer = self.v_steer.to(device=out.device, dtype=out.dtype)
            if len(steer.shape) == 1:
                out[:, -1, :] += a_t * steer
            elif len(steer.shape) == 2:
                out[:, -1, :] += a_t * steer
            elif len(steer.shape) == 3:
                out[:, -1, :] += a_t * steer.squeeze(1)

        self.injection_log.append((self.t, a_t, f"step_{self.t}"))
        return output

def run_sanity_check():
    print("=== SANITY CHECK: HOOK INJECTION LOGIC ===")
    v_steer = torch.randn(3584)
    handler = SteeringHookHandler(v_steer, alpha_0=18.0, K=16, schedule="linear_decay")
    layer = DummyLayer()
    handle = layer.register_forward_hook(handler.hook_fn)

    # 1. Simulate prefill (seq_len = 75)
    x_prefill = torch.randn(1, 75, 3584)
    _ = layer(x_prefill)

    # 2. Simulate 20 decoding steps (seq_len = 1)
    for step in range(1, 21):
        x_dec = torch.randn(1, 1, 3584)
        _ = layer(x_dec)

    handle.remove()

    print("Step-by-step Hook Log:")
    for t, a_t, status in handler.injection_log:
        print(f"  [Step t={t:02d}] status={status:<20} alpha(t)={a_t:6.3f}")

    print("\nSanity Check Verified:")
    print(f"  - Prefill t=0 unsteered: {handler.injection_log[0][1] == 0.0}")
    print(f"  - Step t=1 alpha: {handler.injection_log[1][1]:.3f} (target 18.000)")
    print(f"  - Step t=16 alpha: {handler.injection_log[16][1]:.3f} (target 1.125)")
    print(f"  - Step t=17 alpha: {handler.injection_log[17][1]:.3f} (target 0.000, unsteered past K=16)")

if __name__ == "__main__":
    run_sanity_check()
