"""Phase 1: video → motion → biomechanics features.

Single-camera 2D pose via MediaPipe Tasks. Honest about its limits: depth is
approximate, the club is not tracked, and low frame rates blur impact. The
engine always receives a confidence signal alongside the numbers.
"""
