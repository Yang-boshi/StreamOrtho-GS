"""ZSO: Zero-Shot Streaming Orthomosaicking.

Streaming orthomosaic reconstruction from monocular UAV video using a
feed-forward reconstruction prior (MASt3R-SLAM) + feed-forward Gaussians.

The package is intentionally independent of the upstream repositories
(``MASt3R-SLAM``, ``Splatt3R``, ``gsplat``); it only imports their public API.
See ``README.md`` and ``../../设计框架和想法.md`` (section 11) for the design.
"""

__version__ = "0.1.0-week1"
