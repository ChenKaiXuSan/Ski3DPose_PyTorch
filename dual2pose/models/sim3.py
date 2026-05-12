#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/dual2pose/models/sim3.py
Project: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/dual2pose/models
Created Date: Monday May 11th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Monday May 11th 2026 9:07:00 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

from __future__ import annotations

from typing import Dict, Tuple

import torch


def _as_batched_points(
	x: torch.Tensor,
	name: str,
) -> Tuple[torch.Tensor, bool, Tuple[int, ...]]:
	"""Normalize points to [B, N, 3] and keep original point shape metadata.

	Supported:
	- [N,3]
	- [B,N,3]
	- [B,*,3] (e.g. [B,T,J,3])
	"""
	if x.ndim == 2 and x.shape[-1] == 3:
		return x.unsqueeze(0), True, (int(x.shape[0]),)

	if x.ndim >= 3 and x.shape[-1] == 3:
		bsz = int(x.shape[0])
		point_shape = tuple(int(v) for v in x.shape[1:-1])
		if len(point_shape) == 1:
			return x, False, point_shape
		return x.reshape(bsz, -1, 3), False, point_shape

	raise ValueError(
		f"Expected {name} shape [N,3], [B,N,3], or [B,*,3], got {tuple(x.shape)}"
	)


def _restore_batched_points(
	x_bn3: torch.Tensor,
	squeeze_back: bool,
	point_shape: Tuple[int, ...],
) -> torch.Tensor:
	"""Restore [B,N,3] back to original point layout."""
	if squeeze_back:
		return x_bn3.squeeze(0)

	if len(point_shape) == 1:
		return x_bn3

	b = int(x_bn3.shape[0])
	return x_bn3.reshape(b, *point_shape, 3)


def _as_batched_weights(w: torch.Tensor, bsz: int, num_pts: int) -> torch.Tensor:
	"""Normalize weights to [B, N]."""
	if w.ndim == 1:
		if w.shape[0] != num_pts:
			raise ValueError(f"Expected weights length {num_pts}, got {w.shape[0]}")
		w = w.unsqueeze(0).expand(bsz, -1)
	elif w.ndim == 2:
		if w.shape != (bsz, num_pts):
			raise ValueError(
				f"Expected weights shape [{bsz},{num_pts}], got {tuple(w.shape)}"
			)
	else:
		raise ValueError(f"Expected weights shape [N] or [B,N], got {tuple(w.shape)}")
	return w


def apply_sim3(
	points: torch.Tensor,
	scale: torch.Tensor,
	rotation: torch.Tensor,
	translation: torch.Tensor,
) -> torch.Tensor:
	"""Apply Sim(3) transform.

	Convention:
	  y = s * (x @ R^T) + t

	Args:
	  points: [N,3] or [B,N,3]
	  scale: scalar [] or [B]
	  rotation: [3,3] or [B,3,3]
	  translation: [3] or [B,3]
	"""
	pts, squeeze_back, point_shape = _as_batched_points(points, "points")
	bsz, _, _ = pts.shape

	if scale.ndim == 0:
		scale = scale.view(1).expand(bsz)
	elif scale.ndim == 1 and scale.shape[0] == 1 and bsz > 1:
		scale = scale.expand(bsz)
	elif scale.ndim != 1 or scale.shape[0] != bsz:
		raise ValueError(f"Expected scale shape [] or [B], got {tuple(scale.shape)}")

	if rotation.ndim == 2:
		rotation = rotation.unsqueeze(0).expand(bsz, -1, -1)
	elif rotation.ndim != 3 or rotation.shape[0] != bsz:
		raise ValueError(
			f"Expected rotation shape [3,3] or [B,3,3], got {tuple(rotation.shape)}"
		)

	if translation.ndim == 1:
		translation = translation.unsqueeze(0).expand(bsz, -1)
	elif translation.ndim != 2 or translation.shape != (bsz, 3):
		raise ValueError(
			f"Expected translation shape [3] or [B,3], got {tuple(translation.shape)}"
		)

	out = scale.view(bsz, 1, 1) * (pts @ rotation.transpose(-1, -2)) + translation.view(
		bsz, 1, 3
	)
	return _restore_batched_points(out, squeeze_back, point_shape)


def invert_sim3(
	scale: torch.Tensor,
	rotation: torch.Tensor,
	translation: torch.Tensor,
	eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Compute inverse Sim(3) under y = s * (x @ R^T) + t."""
	if scale.ndim == 0:
		scale = scale.view(1)
	if rotation.ndim == 2:
		rotation = rotation.unsqueeze(0)
	if translation.ndim == 1:
		translation = translation.unsqueeze(0)

	if scale.ndim != 1:
		raise ValueError(f"Expected scale [] or [B], got {tuple(scale.shape)}")
	if rotation.ndim != 3 or rotation.shape[-2:] != (3, 3):
		raise ValueError(f"Expected rotation [3,3] or [B,3,3], got {tuple(rotation.shape)}")
	if translation.ndim != 2 or translation.shape[-1] != 3:
		raise ValueError(
			f"Expected translation [3] or [B,3], got {tuple(translation.shape)}"
		)

	if rotation.shape[0] != scale.shape[0] or translation.shape[0] != scale.shape[0]:
		raise ValueError("Batch size mismatch among scale/rotation/translation")

	inv_s = 1.0 / torch.clamp(scale, min=eps)
	inv_r = rotation.transpose(-1, -2)
	# x = (1/s) * (y - t) @ R
	inv_t = -inv_s.view(-1, 1) * (translation @ rotation)

	return inv_s.squeeze(0), inv_r.squeeze(0), inv_t.squeeze(0)


def estimate_sim3(
	src: torch.Tensor,
	dst: torch.Tensor,
	weights: torch.Tensor | None = None,
	allow_reflection: bool = False,
	eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
	"""Estimate Sim(3) mapping src -> dst with Umeyama-style closed form.

	Args:
	  src: [N,3] or [B,N,3]
	  dst: [N,3] or [B,N,3]
	  weights: optional [N] or [B,N]
	  allow_reflection: whether det(R) < 0 is allowed

	Returns:
	  (scale, rotation, translation, stats)
	  - scale: [] or [B]
	  - rotation: [3,3] or [B,3,3]
	  - translation: [3] or [B,3]
	"""
	x, squeeze_back, point_shape_x = _as_batched_points(src, "src")
	y, _, point_shape_y = _as_batched_points(dst, "dst")

	if x.shape != y.shape or point_shape_x != point_shape_y:
		raise ValueError(f"src/dst shape mismatch: {tuple(src.shape)} vs {tuple(dst.shape)}")

	bsz, num_pts, _ = x.shape
	if num_pts < 3:
		raise ValueError(f"At least 3 points required, got {num_pts}")

	if weights is None:
		w = torch.ones((bsz, num_pts), device=x.device, dtype=x.dtype)
	else:
		w = _as_batched_weights(weights.to(device=x.device, dtype=x.dtype), bsz, num_pts)

	w = torch.clamp(w, min=0.0)
	wsum = torch.clamp(w.sum(dim=1, keepdim=True), min=eps)  # [B,1]
	wn = w / wsum

	mu_x = (wn.unsqueeze(-1) * x).sum(dim=1)  # [B,3]
	mu_y = (wn.unsqueeze(-1) * y).sum(dim=1)  # [B,3]

	x_c = x - mu_x.unsqueeze(1)
	y_c = y - mu_y.unsqueeze(1)

	# Cross-covariance: H = X^T W Y, shape [B,3,3]
	h = torch.einsum("bnc,bnd,bn->bcd", x_c, y_c, wn)

	u, svals, vh = torch.linalg.svd(h, full_matrices=False)
	v = vh.transpose(-1, -2)
	ut = u.transpose(-1, -2)

	# Reflection handling
	d = torch.ones((bsz, 3), device=x.device, dtype=x.dtype)
	det_r = torch.det(v @ ut)
	if not allow_reflection:
		reflect_mask = det_r < 0
		d[reflect_mask, -1] = -1.0

	dmat = torch.diag_embed(d)
	r = v @ dmat @ ut

	var_x = (wn.unsqueeze(-1) * (x_c * x_c)).sum(dim=(1, 2))  # [B]
	var_x = torch.clamp(var_x, min=eps)
	trace_term = (svals * d).sum(dim=1)  # [B]
	scale = trace_term / var_x

	t = mu_y - scale.unsqueeze(-1) * torch.bmm(mu_x.unsqueeze(1), r.transpose(-1, -2)).squeeze(1)

	# Fit quality
	y_hat = apply_sim3(x, scale, r, t)
	rmse = torch.sqrt(torch.mean((y_hat - y) ** 2, dim=(1, 2)))

	stats = {
		"rmse": rmse,
		"det_r": torch.det(r),
		"var_src": var_x,
	}

	if squeeze_back:
		return scale[0], r[0], t[0], {k: v[0] for k, v in stats.items()}
	return scale, r, t, stats


def align_points_sim3(
	src: torch.Tensor,
	dst: torch.Tensor,
	weights: torch.Tensor | None = None,
	allow_reflection: bool = False,
	eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
	"""Estimate Sim(3) from src->dst and return transformed src plus parameters."""
	scale, rotation, translation, stats = estimate_sim3(
		src=src,
		dst=dst,
		weights=weights,
		allow_reflection=allow_reflection,
		eps=eps,
	)
	aligned = apply_sim3(src, scale, rotation, translation)
	return aligned, {
		"scale": scale,
		"rotation": rotation,
		"translation": translation,
		**stats,
	}

