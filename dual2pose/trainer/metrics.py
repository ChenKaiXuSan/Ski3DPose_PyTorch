#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/dual2pose/trainer/metrics.py
Project: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/dual2pose/trainer
Created Date: Wednesday May 13th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday May 12th 2026 11:38:14 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

def mpjpe(preds, targets):
    """Mean Per Joint Position Error (MPJPE)

    Args:
        preds: [B, J, 3]
        targets: [B, J, 3]

    Returns:
        mpjpe: scalar
    """
    return ((preds - targets) ** 2).sum(dim=-1).sqrt().mean()


