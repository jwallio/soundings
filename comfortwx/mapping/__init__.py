"""Mapping helpers for ComfortWx."""

__all__ = ["render_daily_maps"]


def render_daily_maps(*args, **kwargs):
    """Import plotting dependencies only when map rendering is actually requested."""
    from comfortwx.mapping.plotting import render_daily_maps as _render_daily_maps

    return _render_daily_maps(*args, **kwargs)
