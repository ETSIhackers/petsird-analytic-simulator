"""minimal interactive 3-axis viewer for 3D (and 4D) volumes based on matplotlib

`show_vol_cuts` is copied from the (private) examples helper module of parallelproj
(parallelproj/_examples_utils.py, parallelproj v2.0.2, Apache-2.0 license,
https://github.com/gschramm/parallelproj), since that module is not part of the
public parallelproj API.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, TextBox
from parallelproj import Array, to_numpy_array


def show_vol_cuts(
    vol: "np.ndarray | Array",
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    axis_labels: tuple[str, ...] | None = None,
    fig_title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "Greys",
    origin: str | tuple[str, str, str] = ("lower", "lower", "upper"),
    figsize: tuple[int, int] = (12, 6),
) -> tuple:
    """Display interactive 2D cuts through a 3D or 4D array.

    For a 3D array the three panels show cuts perpendicular to each of the
    three axes.  For a 4D array the leading axis (e.g. time or TOF bin) gets
    an additional full-width slider; the three panels still show spatial cuts
    at the currently selected leading-axis index.

    Parameters
    ----------
    vol : array, shape (nx, ny, nz) or (n0, nx, ny, nz)
        3-D or 4-D array to visualise.  May be a NumPy array or any array-API
        array (NumPy, CuPy, PyTorch, ...), including one on a GPU -- it is
        converted to a host NumPy array via ``parallelproj.to_numpy_array``.
    voxel_size : tuple of three floats, optional
        Physical size ``(dx, dy, dz)`` of one voxel along the last three axes.
        Default ``(1, 1, 1)``.
    axis_labels : tuple of str or None, optional
        Label for each axis, e.g. ``("x", "y", "z")`` (3-D) or
        ``("t", "x", "y", "z")`` (4-D).
    fig_title : str or None, optional
        Overall figure title.
    vmin, vmax : float or None, optional
        Initial colour scale limits.  Defaults to the global min / max of
        ``vol``.
    cmap : str, optional
        Initial matplotlib colormap name.  Default ``"Greys"``.
    origin : str or tuple of three str, optional
        ``imshow`` origin for each of the three panels.  A single string is
        applied to all panels; a 3-tuple sets them individually.  Default
        ``("lower", "lower", "upper")`` -- the right-most (transaxial ``x``-``y``)
        panel uses ``"upper"`` so the vertical axis ``x1``/``y`` runs top to
        bottom, matching the scanner coordinate convention.
    figsize : tuple, optional
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : list of three Axes (image panels)
    widgets : dict
        All interactive widgets keyed by name.  Keep a reference to prevent
        garbage collection of the callbacks.
    """
    # accept any array-API array (numpy / cupy / torch, incl. on GPU): move it
    # to a host numpy array once, up front, so all downstream slicing / imshow
    # works regardless of the input backend or device.
    if not isinstance(vol, np.ndarray):
        vol = np.asarray(to_numpy_array(vol))

    if vol.ndim not in (3, 4):
        raise ValueError("vol must be a 3-D or 4-D array")

    is_4d = vol.ndim == 4

    if axis_labels is None:
        axis_labels = ("t", "x", "y", "z") if is_4d else ("x", "y", "z")
    elif len(axis_labels) != vol.ndim:
        raise ValueError(
            f"axis_labels must have {vol.ndim} entries for a {vol.ndim}-D array"
        )

    if is_4d:
        n0, nx, ny, nz = vol.shape
        l0, lx, ly, lz = axis_labels
        i00 = n0 // 2
    else:
        nx, ny, nz = vol.shape
        lx, ly, lz = axis_labels
        n0, l0, i00 = None, None, None

    dx, dy, dz = voxel_size

    if vmin is None:
        vmin = float(vol.min())
    if vmax is None:
        vmax = float(vol.max())

    ix0, iy0, iz0 = nx // 2, ny // 2, nz // 2

    fig = plt.figure(figsize=figsize)
    if fig_title is not None:
        fig.suptitle(fig_title)

    # Some matplotlib versions ship a ``TextBox`` ``resize_event`` handler that
    # assumes a mouse event and reads ``event.inaxes`` -- which a ``ResizeEvent``
    # does not have.  This raises a harmless ``AttributeError`` (printed once per
    # TextBox) on every figure resize.  Filter just that error so it does not
    # spam the console; everything else propagates normally.
    _cb = fig.canvas.callbacks
    _orig_exc_handler = getattr(_cb, "exception_handler", None)

    def _filter_resize_bug(exc, _orig=_orig_exc_handler):
        if isinstance(exc, AttributeError) and "inaxes" in str(exc):
            return
        if _orig is not None:
            return _orig(exc)
        raise exc

    if hasattr(_cb, "exception_handler"):
        _cb.exception_handler = _filter_resize_bug

    if is_4d:
        # rows: images | leading slider (+ its slice-entry) | 3 sliders |
        #       3 slice-entry boxes | vmin/vmax/cmap | colorbar
        gs = fig.add_gridspec(
            6, 3, height_ratios=[8, 1, 1, 1, 1, 0.5], hspace=0.6, wspace=0.35
        )
        ax_s0 = fig.add_subplot(gs[1, 0:2])
        ax_ntb0 = fig.add_subplot(gs[1, 2])  # leading-axis slice entry
        ax_sl = [fig.add_subplot(gs[2, c]) for c in range(3)]
        ax_ntb = [fig.add_subplot(gs[3, c]) for c in range(3)]
        ax_tb = [fig.add_subplot(gs[4, c]) for c in range(3)]
        ax_cb = fig.add_subplot(gs[5, :])
    else:
        # rows: images | 3 sliders | 3 slice-entry boxes | vmin/vmax/cmap | colorbar
        gs = fig.add_gridspec(
            5, 3, height_ratios=[8, 1, 1, 1, 0.5], hspace=0.6, wspace=0.35
        )
        ax_s0 = None
        ax_ntb0 = None
        ax_sl = [fig.add_subplot(gs[1, c]) for c in range(3)]
        ax_ntb = [fig.add_subplot(gs[2, c]) for c in range(3)]
        ax_tb = [fig.add_subplot(gs[3, c]) for c in range(3)]
        ax_cb = fig.add_subplot(gs[4, :])

    ax_im = [fig.add_subplot(gs[0, c]) for c in range(3)]

    if isinstance(origin, str):
        origins = [origin, origin, origin]
    else:
        origins = list(origin)
    if len(origins) != 3:
        raise ValueError("origin must be a string or a sequence of three strings")

    aspects = [dz / dy, dz / dx, dy / dx]
    panel_xlabels = [ly, lx, lx]
    panel_ylabels = [lz, lz, ly]

    # per-panel cut extractor: panel 0 depends on ix, panel 1 on iy, panel 2 on iz
    def _panel_cut(c, i0, ix, iy, iz):
        if is_4d:
            return (vol[i0, ix, :, :], vol[i0, :, iy, :], vol[i0, :, :, iz])[c].T
        return (vol[ix, :, :], vol[:, iy, :], vol[:, :, iz])[c].T

    ims = [
        ax_im[c].imshow(
            _panel_cut(c, i00, ix0, iy0, iz0),
            aspect=aspects[c],
            origin=origins[c],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        for c in range(3)
    ]

    for c, ax in enumerate(ax_im):
        ax.set_xlabel(panel_xlabels[c])
        ax.set_ylabel(panel_ylabels[c])

    sx = Slider(ax_sl[0], lx, 0, nx - 1, valinit=ix0, valstep=1)
    sy = Slider(ax_sl[1], ly, 0, ny - 1, valinit=iy0, valstep=1)
    sz = Slider(ax_sl[2], lz, 0, nz - 1, valinit=iz0, valstep=1)
    s0 = Slider(ax_s0, l0, 0, n0 - 1, valinit=i00, valstep=1) if is_4d else None

    def _current_indices():
        i0 = int(s0.val) if s0 is not None else None
        return i0, int(sx.val), int(sy.val), int(sz.val)

    # --- fast (blitted) rendering -------------------------------------------
    # Moving a slider re-renders only the affected panel(s), and (when the
    # backend supports it) blits just that panel instead of redrawing the whole
    # figure.  Backgrounds are captured on every full draw (covers resize and
    # colour-scale / colormap changes).
    _use_blit = bool(getattr(fig.canvas, "supports_blit", False))
    _bg = [None, None, None]

    def _capture_backgrounds(_event=None):
        if not _use_blit:
            return
        for c in range(3):
            _bg[c] = fig.canvas.copy_from_bbox(ax_im[c].bbox)

    fig.canvas.mpl_connect("draw_event", _capture_backgrounds)

    def _render(panels):
        i0, ix, iy, iz = _current_indices()
        for c in panels:
            ims[c].set_data(_panel_cut(c, i0, ix, iy, iz))
        if _use_blit and all(_bg[c] is not None for c in panels):
            for c in panels:
                fig.canvas.restore_region(_bg[c])
                ax_im[c].draw_artist(ims[c])
                fig.canvas.blit(ax_im[c].bbox)
        else:
            fig.canvas.draw_idle()

    # axis -> (entry box, slider); filled in when the entry boxes are built
    # below.  Lets slider drags / clicks push the current index into the box.
    _entry_for_axis = {}

    def _sync_entry(axis):
        pair = _entry_for_axis.get(axis)
        if pair is None:
            return
        entry, slider = pair
        txt = str(int(slider.val))
        if entry.text_disp.get_text() != txt:
            # update the text artist directly (no submit, no extra draw); it
            # refreshes on the redraw the slider already schedules
            entry.text_disp.set_text(txt)

    # each spatial slider affects only its own panel (and its own entry box);
    # the leading-axis slider (4-D only) affects all three panels
    def _slider_cb(axis, panels):
        def _cb(_v):
            _render(panels)
            _sync_entry(axis)

        return _cb

    sx.on_changed(_slider_cb(0, [0]))
    sy.on_changed(_slider_cb(1, [1]))
    sz.on_changed(_slider_cb(2, [2]))
    if s0 is not None:
        s0.on_changed(_slider_cb("t", [0, 1, 2]))

    # --- click-to-locate: clicking a panel moves the two orthogonal panels ---
    # index-axis a (0=x, 1=y, 2=z) is the slice axis of panel a, so re-rendering
    # panel a exactly follows a change of index-axis a.
    _slider_for_axis = {0: sx, 1: sy, 2: sz}
    _axis_extent = {0: nx, 1: ny, 2: nz}
    # per panel: (horizontal index-axis, vertical index-axis) of the displayed cut
    _panel_click_axes = {0: (1, 2), 1: (0, 2), 2: (0, 1)}

    def _on_click(event):
        if event.button != 1 or event.inaxes not in ax_im:
            return
        if event.xdata is None or event.ydata is None:
            return
        toolbar = getattr(fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return  # ignore clicks while the zoom / pan tool is active
        c = ax_im.index(event.inaxes)
        col_axis, row_axis = _panel_click_axes[c]
        panels = []
        for axis, coord in ((col_axis, event.xdata), (row_axis, event.ydata)):
            value = max(0, min(_axis_extent[axis] - 1, int(round(coord))))
            slider = _slider_for_axis[axis]
            slider.eventson = False  # set both, then render once
            slider.set_val(value)
            slider.eventson = True
            panels.append(axis)  # index-axis a <-> panel a
        _render(sorted(panels))
        for axis in panels:
            _sync_entry(axis)  # click used eventson=False, so sync boxes here

    fig.canvas.mpl_connect("button_press_event", _on_click)

    # --- direct slice-number entry: type an index and press Enter to jump ----
    def _attach_slice_entry(ax_box, label, slider, n, axis):
        entry = TextBox(ax_box, label, initial=str(int(slider.val)))

        def _submit(text, _entry=entry, _slider=slider, _n=n):
            try:
                v = int(round(float(text)))
            except (ValueError, TypeError):
                v = int(_slider.val)
            v = max(0, min(_n - 1, v))
            _entry.text_disp.set_text(str(v))  # show the clamped value
            # set_val fires the slider callback -> _render + _sync_entry; no
            # entry.set_val() here, so there is no submit-recursion
            _slider.set_val(v)

        entry.on_submit(_submit)
        _entry_for_axis[axis] = (entry, slider)
        return entry

    slice_entries = {
        lx: _attach_slice_entry(ax_ntb[0], lx, sx, nx, 0),
        ly: _attach_slice_entry(ax_ntb[1], ly, sy, ny, 1),
        lz: _attach_slice_entry(ax_ntb[2], lz, sz, nz, 2),
    }
    if is_4d:
        slice_entries[l0] = _attach_slice_entry(ax_ntb0, l0, s0, n0, "t")

    cb = fig.colorbar(ims[0], cax=ax_cb, orientation="horizontal")

    tb_vmin = TextBox(ax_tb[0], "vmin", initial=f"{vmin:.4g}")
    tb_vmax = TextBox(ax_tb[1], "vmax", initial=f"{vmax:.4g}")

    def update_clim(_):
        try:
            lo = float(tb_vmin.text)
            hi = float(tb_vmax.text)
        except ValueError:
            return
        for im in ims:
            im.set_clim(lo, hi)
        cb.update_normal(ims[0])
        fig.canvas.draw_idle()

    tb_vmin.on_submit(update_clim)
    tb_vmax.on_submit(update_clim)

    tb_cmap = TextBox(ax_tb[2], "cmap", initial=cmap)

    def update_cmap(name):
        try:
            cm = plt.get_cmap(name.strip())
        except (ValueError, KeyError):
            tb_cmap.set_val(ims[0].cmap.name)
            return
        for im in ims:
            im.set_cmap(cm)
        cb.update_normal(ims[0])
        fig.canvas.draw_idle()

    tb_cmap.on_submit(update_cmap)

    widgets = {
        f"s_{lx}": sx,
        f"s_{ly}": sy,
        f"s_{lz}": sz,
        "tb_vmin": tb_vmin,
        "tb_vmax": tb_vmax,
        "tb_cmap": tb_cmap,
        # keep a strong ref to the click callback (also handy for tests)
        "on_click": _on_click,
    }
    for _lab, _entry in slice_entries.items():
        widgets[f"e_{_lab}"] = _entry
    if s0 is not None:
        widgets[f"s_{l0}"] = s0

    return fig, ax_im, widgets


