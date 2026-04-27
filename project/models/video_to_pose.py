#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Simple video-to-pose CNN model for pole and ski keypoint estimation."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleVideo2Pose(nn.Module):
    """Simple CNN to estimate 3D keypoints from video frames.
    
    Input: Video frames (B, T, H, W, 3) or (B, C, T, H, W)
    Output: 3D keypoints (B, T, J, 3)
    """

    def __init__(self, num_joints: int, hidden_dim: int = 128) -> None:
        """Initialize Video2Pose model.
        
        Args:
            num_joints: Number of joints to predict (4 for pole, 6 for ski)
            hidden_dim: Hidden dimension for CNN layers
        """
        super().__init__()
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim

        # Simple CNN backbone: extract spatial features from each frame
        self.conv1 = nn.Conv2d(3, hidden_dim // 2, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(hidden_dim // 2)
        self.conv2 = nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim)

        # Global average pooling + FC regression head
        # After 3 stride-2 layers: H//8, W//8
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Temporal aggregation: simple LSTM or temporal pooling
        self.temporal_fc = nn.Linear(hidden_dim, hidden_dim)
        
        # Regression head: predict (J, 3) keypoints per frame
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_joints * 3),  # (x, y, z) per joint
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            video: (B*T, 3, H, W) or (B, T, 3, H, W)
        
        Returns:
            keypoints: (B*T, 1, J, 3) - matches FusionSSM output format
        """
        # Handle input format
        if video.ndim == 5:
            # (B, T, 3, H, W) -> (B*T, 3, H, W)
            B, T, C, H, W = video.shape
            video = video.view(B * T, C, H, W)
        else:
            B_T = video.shape[0]
            T = 1  # Single frame

        # Extract spatial features
        x = self.conv1(video)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # Global average pooling: (B*T, hidden_dim, H//8, W//8) -> (B*T, hidden_dim)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)  # Flatten

        # Temporal processing
        x = self.temporal_fc(x)
        x = F.relu(x)

        # Regression: predict keypoints
        keypoints = self.regression_head(x)  # (B*T, J*3)
        keypoints = keypoints.view(-1, 1, self.num_joints, 3)  # (B*T, 1, J, 3)

        return keypoints


class Video2PoseWithConfidence(nn.Module):
    """Video2Pose model with confidence scores for each joint."""

    def __init__(self, num_joints: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.base_model = SimpleVideo2Pose(num_joints, hidden_dim)
        
        # Additional head for confidence/uncertainty
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_joints),
            nn.Sigmoid(),  # confidence in [0, 1]
        )

    def forward(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning keypoints and confidence.
        
        Args:
            video: (B*T, 3, H, W) or (B, T, 3, H, W)
        
        Returns:
            Dict with keys:
                - "keypoints": (B*T, 1, J, 3)
                - "confidence": (B*T, 1, J, 1)
        """
        # Handle input format
        if video.ndim == 5:
            B, T, C, H, W = video.shape
            video_2d = video.view(B * T, C, H, W)
        else:
            video_2d = video

        # Extract features for keypoints
        keypoints = self.base_model(video_2d)

        # Extract confidence scores (reuse base model's intermediate features)
        # For now, just return keypoints without confidence
        out = {"keypoints": keypoints}

        return out
