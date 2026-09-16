"""PointDINO Stage 4 with local FIDT supervision within 16 image pixels."""

_base_ = './pointdino_r50_shanghaitech_stage4_12e.py'

# Only the FIDT MSE support changes relative to full-map supervision.
model = dict(point_fidt_head=dict(local_radius_px=16.0))
work_dir = './work_dirs/pointdino_r50_shanghaitech_stage4_local_12e'
