"""
  loss functions
"""
import torch
import numpy as np


class LossMSE():
    """ mse loss """
    def __init__(self, params, model):
        self.params = params
        self.model = model
        self.device = params.device
        self.system = str(getattr(params, 'system', '')).lower()
        self.phase = 'eval'
        self.epoch = 0
        self.max_epochs = max(1, int(getattr(params, 'max_epochs', 1)))

        # PDE residual settings
        self.constraint_pde_enable = bool(getattr(params, 'constraint_pde_enable', False))
        self.constraint_pde_weight = float(getattr(params, 'constraint_pde_weight', 0.0))
        self.constraint_pde_warmup_fraction = float(getattr(params, 'constraint_pde_warmup_fraction', 0.0))
        self.constraint_pde_eps = float(getattr(params, 'constraint_pde_eps', 1.0e-8))
        self.constraint_pde_relative_norm = bool(getattr(params, 'constraint_pde_relative_norm', True))
        self.constraint_pde_method = str(getattr(params, 'constraint_pde_method', 'penalty')).lower()
        self.constraint_pde_al_rho = float(getattr(params, 'constraint_pde_al_rho', 1.0))
        self.constraint_pde_al_lambda0 = float(getattr(params, 'constraint_pde_al_lambda0', 0.0))
        self.constraint_pde_al_dual_clip = float(getattr(params, 'constraint_pde_al_dual_clip', 1.0e6))
        self.constraint_pde_discretization = str(
            getattr(params, 'constraint_pde_discretization', 'spectral')
        ).lower()
        if self.constraint_pde_discretization not in ['spectral', 'fd']:
            raise ValueError("constraint_pde_discretization must be one of ['spectral', 'fd']")

        # Boundary-condition settings
        self.use_bc_channels = self._coerce_bool(getattr(params, 'use_bc_channels', False))
        in_dim = int(getattr(self.params, 'in_dim', 1))
        self.bc_value_channel_idx = int(getattr(params, 'bc_value_channel_idx', max(0, in_dim - 2)))
        self.bc_mask_channel_idx = int(getattr(params, 'bc_mask_channel_idx', max(0, in_dim - 1)))
        self.constraint_bc_enforcement = self._resolve_bc_enforcement()
        self.constraint_bc_weight = float(getattr(params, 'constraint_bc_weight', 0.0))
        self.constraint_bc_warmup_fraction = float(getattr(params, 'constraint_bc_warmup_fraction', 0.0))
        self.constraint_bc_eps = float(getattr(params, 'constraint_bc_eps', 1.0e-8))
        self.constraint_bc_loss_norm = str(getattr(params, 'constraint_bc_loss_norm', 'l2')).lower()
        if self.constraint_bc_loss_norm not in ['l2', 'l1']:
            raise ValueError("constraint_bc_loss_norm must be one of ['l2', 'l1']")

        # Zero-mode settings (hard/soft use same masking semantics)
        self.constraint_zero_mode_enforcement = self._resolve_zero_mode_enforcement()
        self.constraint_zero_mode_weight = float(getattr(params, 'constraint_zero_mode_weight', 0.0))
        self.constraint_zero_mode_warmup_fraction = float(
            getattr(params, 'constraint_zero_mode_warmup_fraction', 0.0)
        )
        self.constraint_zero_mode_mode = str(getattr(params, 'constraint_zero_mode_mode', 'all')).lower()
        self.constraint_zero_mode_omega_tol = float(getattr(params, 'constraint_zero_mode_omega_tol', 1.0e-8))

        self.constraint_diffusion_tensor_order = str(
            getattr(params, 'constraint_diffusion_tensor_order', 'k11_k22_k12')
        ).lower()

        self._layout = self._infer_layout()
        self._coeff_scales = self._load_coeff_scales()
        self._fft_cache = {}

        self._al_lambda = self.constraint_pde_al_lambda0
        self._epoch_constraint_sum = 0.0
        self._epoch_constraint_count = 0

        self._last_pde_residual_norm = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self._last_pde_constraint = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self._last_pde_residual_map = None
        self._last_zero_mode_constraint_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self._last_bc_constraint_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self._last_bc_violation_raw = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self._last_bc_violation_final = torch.tensor(0.0, device=self.device, dtype=torch.float32)

    @staticmethod
    def _coerce_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in ['1', 'true', 'yes', 'y', 'on']
        return bool(val)

    def _resolve_bc_enforcement(self):
        enforcement = getattr(self.params, 'constraint_bc_enforcement', None)
        if enforcement is None:
            return 'off'

        if isinstance(enforcement, bool):
            return 'hard' if enforcement else 'off'

        enforcement = str(enforcement).strip().lower()
        if enforcement in ['true', 'false']:
            return 'hard' if enforcement == 'true' else 'off'

        valid = ['off', 'soft', 'hard', 'hard+soft']
        if enforcement not in valid:
            raise ValueError("constraint_bc_enforcement must be one of ['off', 'soft', 'hard', 'hard+soft']")
        return enforcement

    def _resolve_zero_mode_enforcement(self):
        enforcement = getattr(self.params, 'constraint_zero_mode_enforcement', None)
        if enforcement is not None:
            if isinstance(enforcement, bool):
                return 'hard' if enforcement else 'off'
            enforcement = str(enforcement).strip().lower()
            if enforcement in ['true', 'false']:
                return 'hard' if enforcement == 'true' else 'off'
            if enforcement not in ['off', 'hard', 'soft']:
                raise ValueError(
                    "constraint_zero_mode_enforcement must be one of ['off', 'hard', 'soft']"
                )
            return enforcement

        return 'hard' if self._coerce_bool(getattr(self.params, 'constraint_zero_mode_enable', False)) else 'off'

    def _load_coeff_scales(self):
        if not hasattr(self.params, 'scales_path'):
            return None
        try:
            scales = np.load(self.params.scales_path).astype('float32')
        except Exception:
            return None

        in_dim = int(getattr(self.params, 'in_dim', 1))
        n_tensor_channels = max(0, in_dim - 1 - (2 if self.use_bc_channels else 0))
        if n_tensor_channels == 0:
            return None

        coeff_scales = scales[1:1+n_tensor_channels]
        if len(coeff_scales) == 0:
            return None
        return torch.tensor(coeff_scales, device=self.device, dtype=torch.float32)

    def _infer_layout(self):
        in_dim = int(getattr(self.params, 'in_dim', 1))
        layout = {
            'source_idx': 0,
            'k11_idx': None,
            'k12_idx': None,
            'k22_idx': None,
            'vx_idx': None,
            'vy_idx': None,
            'omega_idx': None,
        }

        # Mixed-format layout is canonical and reused by some single-system configs.
        if in_dim >= 7:
            layout.update({
                'k11_idx': 1,
                'k12_idx': 2,
                'k22_idx': 3,
                'vx_idx': 4,
                'vy_idx': 5,
                'omega_idx': 6,
            })
            return layout

        diffusion_triplets = {
            'k11_k22_k12': (1, 3, 2),
            'k11_k12_k22': (1, 2, 3),
        }
        if self.constraint_diffusion_tensor_order not in diffusion_triplets:
            raise ValueError(
                "constraint_diffusion_tensor_order must be one of "
                "['k11_k22_k12', 'k11_k12_k22']"
            )

        k11_idx, k12_idx, k22_idx = diffusion_triplets[self.constraint_diffusion_tensor_order]

        if self.system in ['poisson'] and in_dim >= 4:
            layout.update({'k11_idx': k11_idx, 'k12_idx': k12_idx, 'k22_idx': k22_idx})
        elif self.system in ['advection-diffusion', 'advdiff', 'advection_diffusion'] and in_dim >= 6:
            layout.update({
                'k11_idx': k11_idx,
                'k12_idx': k12_idx,
                'k22_idx': k22_idx,
                'vx_idx': 4,
                'vy_idx': 5,
            })
        elif self.system in ['helmholtz'] and in_dim >= 3:
            # Helmholtz legacy format uses [source, diff_coef, omega].
            layout.update({
                'k11_idx': 1,
                'k12_idx': None,
                'k22_idx': 1,
                'omega_idx': 2,
            })
        elif in_dim >= 4:
            layout.update({'k11_idx': k11_idx, 'k12_idx': k12_idx, 'k22_idx': k22_idx})

        return layout

    def _get_fft_factors(self, nx, ny, dtype, device):
        complex_dtype = torch.complex64 if dtype in [torch.float16, torch.bfloat16, torch.float32] else torch.complex128
        key = (nx, ny, complex_dtype, device)
        if key in self._fft_cache:
            return self._fft_cache[key]

        kx = torch.fft.fftfreq(nx, d=1.0 / nx, device=device).view(1, nx, 1)
        ky = torch.fft.fftfreq(ny, d=1.0 / ny, device=device).view(1, 1, ny)

        factor_x = (2.0 * np.pi / float(self.params.Lx))
        factor_y = (2.0 * np.pi / float(self.params.Ly))
        ikx = (1j * kx * factor_x).to(complex_dtype)
        iky = (1j * ky * factor_y).to(complex_dtype)

        self._fft_cache[key] = (ikx, iky)
        return ikx, iky

    def _grad(self, u):
        # u is [B, nx, ny]
        nx, ny = u.shape[-2], u.shape[-1]
        ikx, iky = self._get_fft_factors(nx, ny, u.dtype, u.device)
        u_hat = torch.fft.fft2(u, dim=(-2, -1))
        ux = torch.fft.ifft2(u_hat * ikx, dim=(-2, -1)).real
        uy = torch.fft.ifft2(u_hat * iky, dim=(-2, -1)).real
        return ux, uy

    def _div(self, ux, uy):
        # ux, uy are [B, nx, ny]
        nx, ny = ux.shape[-2], ux.shape[-1]
        ikx, iky = self._get_fft_factors(nx, ny, ux.dtype, ux.device)
        ux_hat = torch.fft.fft2(ux, dim=(-2, -1))
        uy_hat = torch.fft.fft2(uy, dim=(-2, -1))
        div_hat = ux_hat * ikx + uy_hat * iky
        return torch.fft.ifft2(div_hat, dim=(-2, -1)).real

    def _get_channel(self, inputs, idx):
        if idx is None or idx >= inputs.shape[1]:
            return torch.zeros_like(inputs[:, 0])
        return inputs[:, idx]

    def _get_coeff_channel(self, inputs, idx):
        coeff = self._get_channel(inputs, idx)
        if self._coeff_scales is None or idx is None:
            return coeff
        coeff_idx = idx - 1
        if coeff_idx < 0 or coeff_idx >= len(self._coeff_scales):
            return coeff
        return coeff * self._coeff_scales[coeff_idx].to(coeff.dtype)

    def _extract_bc(self, inputs):
        if not self.use_bc_channels:
            return None, None
        if self.bc_value_channel_idx >= inputs.shape[1] or self.bc_mask_channel_idx >= inputs.shape[1]:
            return None, None
        g = self._get_channel(inputs, self.bc_value_channel_idx)
        m = self._get_channel(inputs, self.bc_mask_channel_idx)
        m = torch.clamp(m, min=0.0, max=1.0)
        return g, m

    def _current_pde_weight(self):
        warmup_epochs = int(self.constraint_pde_warmup_fraction * self.max_epochs)
        if warmup_epochs <= 0:
            return self.constraint_pde_weight
        warmup_scale = min(1.0, float(self.epoch + 1) / float(warmup_epochs))
        return self.constraint_pde_weight * warmup_scale

    def _current_zero_mode_weight(self):
        warmup_epochs = int(self.constraint_zero_mode_warmup_fraction * self.max_epochs)
        if warmup_epochs <= 0:
            return self.constraint_zero_mode_weight
        warmup_scale = min(1.0, float(self.epoch + 1) / float(warmup_epochs))
        return self.constraint_zero_mode_weight * warmup_scale

    def _current_bc_weight(self):
        warmup_epochs = int(self.constraint_bc_warmup_fraction * self.max_epochs)
        if warmup_epochs <= 0:
            return self.constraint_bc_weight
        warmup_scale = min(1.0, float(self.epoch + 1) / float(max(1, warmup_epochs)))
        return self.constraint_bc_weight * warmup_scale

    def set_phase(self, phase):
        self.phase = phase

    def set_epoch(self, epoch, max_epochs):
        self.epoch = int(epoch)
        self.max_epochs = max(1, int(max_epochs))

        if self.constraint_pde_method != 'augmented_lagrangian' or not self.constraint_pde_enable:
            self._epoch_constraint_sum = 0.0
            self._epoch_constraint_count = 0
            return

        if self.epoch > 0 and self._epoch_constraint_count > 0:
            avg_constraint = self._epoch_constraint_sum / float(self._epoch_constraint_count)
            self._al_lambda += self.constraint_pde_al_rho * avg_constraint
            self._al_lambda = max(0.0, min(self._al_lambda, self.constraint_pde_al_dual_clip))

        self._epoch_constraint_sum = 0.0
        self._epoch_constraint_count = 0

    def data(self, inputs, pred, target):
        if self.params.loss_style == 'mean':
            loss = torch.mean((target - pred)**2)
        elif self.params.loss_style == 'sum':
            loss = torch.sum((target - pred)**2)/pred.shape[0]
        return loss

    def _zero_mode_mask(self, inputs):
        if self.constraint_zero_mode_mode == 'all':
            return None

        if self.constraint_zero_mode_mode == 'gauge_aware':
            omega_idx = self._layout.get('omega_idx')
            if omega_idx is None:
                if self.system == 'helmholtz':
                    return torch.zeros((inputs.shape[0],), dtype=torch.bool, device=inputs.device)
                return None
            omega = self._get_channel(inputs, omega_idx)
            omega_sample = torch.mean(torch.abs(omega), dim=(-2, -1))
            return omega_sample <= self.constraint_zero_mode_omega_tol

        return None

    def _zero_mode_dc(self, inputs, pred):
        # Returns per-sample DC component for selected samples, shape [N, C].
        dc = torch.mean(pred, dim=(-2, -1))
        mask = self._zero_mode_mask(inputs)
        if mask is None:
            return dc
        if torch.any(mask):
            return dc[mask]
        return None

    def zero_mode_violation(self, inputs, pred):
        dc = self._zero_mode_dc(inputs, pred)
        if dc is None:
            return torch.zeros((), dtype=pred.dtype, device=pred.device)
        return torch.mean(torch.abs(dc))

    def zero_mode_constraint(self, inputs, pred, targets=None):
        zero = torch.zeros((), dtype=pred.dtype, device=pred.device)
        self._last_zero_mode_constraint_loss = zero.detach()

        if self.constraint_zero_mode_enforcement != 'soft':
            return zero
        if self.constraint_zero_mode_weight <= 0:
            return zero

        dc = self._zero_mode_dc(inputs, pred)
        if dc is None:
            return zero

        raw_loss = torch.mean(dc**2)
        loss = self._current_zero_mode_weight() * raw_loss
        self._last_zero_mode_constraint_loss = loss.detach()
        return loss

    def project_bc(self, inputs, pred):
        if self.constraint_bc_enforcement not in ['hard', 'hard+soft']:
            return pred
        g, m = self._extract_bc(inputs)
        if g is None or m is None:
            return pred

        projected = pred.clone()
        projected[:, 0] = pred[:, 0] * (1.0 - m) + g * m
        return projected

    def _bc_raw_loss(self, inputs, pred):
        g, m = self._extract_bc(inputs)
        if g is None or m is None:
            return torch.zeros((), dtype=pred.dtype, device=pred.device)

        residual = (pred[:, 0] - g) * m
        denom = torch.mean(m) + self.constraint_bc_eps
        if self.constraint_bc_loss_norm == 'l1':
            return torch.mean(torch.abs(residual)) / denom
        return torch.mean(residual**2) / denom

    def bc_violation(self, inputs, pred):
        g, m = self._extract_bc(inputs)
        if g is None or m is None:
            return torch.zeros((), dtype=pred.dtype, device=pred.device)

        residual = (pred[:, 0] - g) * m
        denom = torch.mean(m) + self.constraint_bc_eps
        if self.constraint_bc_loss_norm == 'l1':
            return torch.mean(torch.abs(residual)) / denom
        return torch.sqrt(torch.mean(residual**2) / denom)

    def interior_relative_l2(self, inputs, pred, targets):
        g, m = self._extract_bc(inputs)
        if g is None or m is None:
            diff = pred - targets
            target_masked = targets
        else:
            interior = (1.0 - m).unsqueeze(1)
            diff = (pred - targets) * interior
            target_masked = targets * interior

        diff_norm = torch.sqrt(torch.sum(diff**2, dim=(-2, -1)))
        target_norm = torch.sqrt(torch.sum(target_masked**2, dim=(-2, -1)) + self.constraint_bc_eps)
        rel = diff_norm / target_norm
        return torch.mean(rel)

    def bc(self, inputs, pred, targets):
        zero = torch.zeros((), dtype=pred.dtype, device=pred.device)
        self._last_bc_constraint_loss = zero.detach()

        if self.constraint_bc_enforcement not in ['soft', 'hard+soft']:
            return zero
        if self.constraint_bc_weight <= 0:
            return zero

        raw_loss = self._bc_raw_loss(inputs, pred)
        loss = self._current_bc_weight() * raw_loss
        self._last_bc_constraint_loss = loss.detach()
        return loss

    def update_bc_metrics(self, inputs, pred_raw, pred_final):
        self._last_bc_violation_raw = self.bc_violation(inputs, pred_raw).detach()
        self._last_bc_violation_final = self.bc_violation(inputs, pred_final).detach()

    def _compute_residual_spectral(self, inputs, pred):
        source = self._get_channel(inputs, self._layout['source_idx'])
        u = pred[:, 0]

        k11 = self._get_coeff_channel(inputs, self._layout.get('k11_idx'))
        k12 = self._get_coeff_channel(inputs, self._layout.get('k12_idx'))
        k22 = self._get_coeff_channel(inputs, self._layout.get('k22_idx'))
        vx = self._get_coeff_channel(inputs, self._layout.get('vx_idx'))
        vy = self._get_coeff_channel(inputs, self._layout.get('vy_idx'))
        omega = self._get_coeff_channel(inputs, self._layout.get('omega_idx'))

        ux, uy = self._grad(u)
        flux_x = k11 * ux + k12 * uy
        flux_y = k12 * ux + k22 * uy
        diffusion_term = self._div(flux_x, flux_y)
        advection_term = vx * ux + vy * uy
        helmholtz_term = omega * u

        residual = torch.zeros_like(source)

        if self.system == 'mixed':
            omega_abs = torch.mean(torch.abs(omega), dim=(-2, -1))
            vel_norm = torch.sqrt(torch.mean(vx**2 + vy**2, dim=(-2, -1)))
            helm_mask = omega_abs > self.constraint_zero_mode_omega_tol
            adv_mask = (~helm_mask) & (vel_norm > self.constraint_zero_mode_omega_tol)
            poisson_mask = ~(helm_mask | adv_mask)

            if torch.any(poisson_mask):
                residual[poisson_mask] = diffusion_term[poisson_mask] + source[poisson_mask]
            if torch.any(adv_mask):
                residual[adv_mask] = diffusion_term[adv_mask] - advection_term[adv_mask] + source[adv_mask]
            if torch.any(helm_mask):
                residual[helm_mask] = diffusion_term[helm_mask] + helmholtz_term[helm_mask] + source[helm_mask]

        elif self.system in ['advection-diffusion', 'advdiff', 'advection_diffusion']:
            residual = diffusion_term - advection_term + source
        elif self.system in ['helmholtz']:
            residual = diffusion_term + helmholtz_term + source
        else:
            residual = diffusion_term + source

        return residual, source

    def _compute_residual_fd(self, inputs, pred):
        # Phase-2 scaffold: non-periodic FD residual path will be implemented here.
        # For Phase 1 we keep spectral behavior to preserve compatibility.
        return self._compute_residual_spectral(inputs, pred)

    def _compute_residual(self, inputs, pred):
        if self.constraint_pde_discretization == 'fd':
            return self._compute_residual_fd(inputs, pred)
        return self._compute_residual_spectral(inputs, pred)

    def _pde_metric(self, residual, source):
        res_norm = torch.sqrt(torch.sum(residual**2, dim=(-2, -1)))
        src_norm = torch.sqrt(torch.sum(source**2, dim=(-2, -1)))
        if self.constraint_pde_relative_norm:
            metric = res_norm / (src_norm + self.constraint_pde_eps)
        else:
            metric = res_norm

        raw_loss = torch.mean(metric**2)
        constraint_value = torch.mean(metric)
        residual_norm = constraint_value
        return raw_loss, constraint_value, residual_norm

    def pde(self, inputs, pred, targets):
        residual, source = self._compute_residual(inputs, pred)
        raw_loss, constraint_value, residual_norm = self._pde_metric(residual, source)

        self._last_pde_residual_norm = residual_norm.detach()
        self._last_pde_constraint = constraint_value.detach()
        self._last_pde_residual_map = residual.detach()

        if self.constraint_pde_enable and self.phase == 'train' and torch.is_grad_enabled():
            self._epoch_constraint_sum += float(constraint_value.detach())
            self._epoch_constraint_count += 1

        if not self.constraint_pde_enable or self.constraint_pde_weight <= 0:
            return torch.zeros((), dtype=pred.dtype, device=pred.device)

        pde_weight = self._current_pde_weight()
        if self.constraint_pde_method == 'augmented_lagrangian':
            lagrange_term = self._al_lambda * constraint_value
            penalty_term = 0.5 * self.constraint_pde_al_rho * (constraint_value**2)
            return pde_weight * (lagrange_term + penalty_term)

        return pde_weight * raw_loss

    def get_last_pde_residual_norm(self):
        return self._last_pde_residual_norm

    def get_last_pde_residual_map(self):
        return self._last_pde_residual_map

    def get_aug_lagrange_multiplier(self):
        return self._al_lambda

    def get_last_zero_mode_constraint_loss(self):
        return self._last_zero_mode_constraint_loss

    def get_last_bc_constraint_loss(self):
        return self._last_bc_constraint_loss

    def get_last_bc_violation_raw(self):
        return self._last_bc_violation_raw

    def get_last_bc_violation_final(self):
        return self._last_bc_violation_final
