"""visualization of the PETSIRD scanner geometry and of coincidence events

Everything shown is computed directly from the PETSIRD scanner information and the
(raw) events following the PETSIRD definitions:

- detecting element boxes: module transform x detecting element transform x box corners
- front face (first 4 corners) and box axis (center of front face -> center of back face)
- average detection point (ADP): (0,0,0) of the box coordinate system
- detection bins: energy_idx + (element_idx + module_idx * num_det_els) * num_energy_bins
- TOF: tof_idx refers to (t1 - t2) * c / 2 (positive = emission closer to the second
  detection bin), shown as the center of the TOF bin +/- FWHM / 2 of the TOF resolution
"""

import matplotlib.pyplot as plt
import numpy as np
import petsird
from matplotlib.collections import LineCollection, PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

# faces of a PETSIRD box shape (front face, back face, 4 side faces)
_BOX_FACES = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
]


def _affine(transform: petsird.RigidTransformation) -> np.ndarray:
    return np.vstack([np.asarray(transform.matrix, dtype="float64"), [0, 0, 0, 1]])


def get_detecting_element_geometry(
    scanner: petsird.ScannerInformation,
) -> list[dict[str, np.ndarray]]:
    """corners, ADPs and box axes of all detecting elements (in scanner coordinates)

    Returns a list (one entry per module type) of dictionaries with the arrays
    - corners: (num_modules, num_det_els, 8, 3)
    - adp: (num_modules, num_det_els, 3)
    - box_axis: (num_modules, num_det_els, 3) unit vectors (front -> back face)
    """
    geom = []
    for rep_module in scanner.scanner_geometry.replicated_modules:
        det_els = rep_module.object.detecting_elements
        mod = np.array([_affine(t) for t in rep_module.transforms])
        el = np.array([_affine(t) for t in det_els.transforms])
        # (num_modules, num_det_els, 4, 4) combined transforms
        full = np.einsum("mij,ejk->meik", mod, el)

        corners_local = np.array([c.c for c in det_els.object.shape.corners], dtype="float64")
        corners = (
            np.einsum("meij,cj->meci", full[..., :3, :3], corners_local)
            + full[:, :, None, :3, 3]
        )
        adp = full[..., :3, 3]
        box_axis = corners[..., 4:, :].mean(-2) - corners[..., :4, :].mean(-2)
        box_axis /= np.linalg.norm(box_axis, axis=-1, keepdims=True)
        geom.append({"corners": corners, "adp": adp, "box_axis": box_axis})
    return geom


def read_first_events(
    reader: petsird.BinaryPETSIRDReader, num_events: int
) -> np.ndarray:
    """read the first prompt events of a PETSIRD file

    The header must have been read already. Note that the yardl-generated Python reader
    deserializes the complete first event time block (slow for large time blocks); if
    the events were read already, use petsird_reader.first_events() instead. Returns an array of shape (num_events, 5)
    with the columns (module type 0, module type 1, detection bin 0, detection bin 1, tof_idx).
    """
    events = []
    for time_block in reader.read_time_blocks():
        if not isinstance(time_block, petsird.TimeBlock.EventTimeBlock):
            continue
        for m0, row in enumerate(time_block.value.prompt_events):
            for m1, event_list in enumerate(row):
                for e in event_list:
                    events.append((m0, m1, e.detection_bins[0], e.detection_bins[1], e.tof_idx))
                    if len(events) >= num_events:
                        return np.array(events, dtype=np.int64)
    return np.array(events, dtype=np.int64).reshape(-1, 5)


def _decode_events(
    scanner: petsird.ScannerInformation,
    geom: list[dict[str, np.ndarray]],
    events: np.ndarray,
) -> list[dict[str, np.ndarray]]:
    """ADPs of both detection bins and TOF position / FWHM interval of events"""
    decoded = []
    for m0, m1, b0, b1, tof_idx in events:
        p = []
        for m, b in ((m0, b0), (m1, b1)):
            rep_module = scanner.scanner_geometry.replicated_modules[m]
            num_det_els = len(rep_module.object.detecting_elements.transforms)
            num_energy_bins = scanner.event_energy_bin_edges[m].number_of_bins()
            i_mod = b // (num_det_els * num_energy_bins)
            i_el = (b // num_energy_bins) % num_det_els
            p.append(geom[m]["adp"][i_mod, i_el])
        p0, p1 = p
        u = (p1 - p0) / np.linalg.norm(p1 - p0)
        edges = np.asarray(scanner.tof_bin_edges[m0][m1].edges, dtype="float64")
        d = 0.5 * (edges[tof_idx] + edges[tof_idx + 1])
        half_fwhm = 0.5 * float(scanner.tof_resolution[m0][m1])
        center = 0.5 * (p0 + p1)
        decoded.append(
            {
                "p0": p0,
                "p1": p1,
                "tof_pos": center + d * u,
                "tof_interval": np.array(
                    [center + (d - half_fwhm) * u, center + (d + half_fwhm) * u]
                ),
            }
        )
    return decoded


def _draw_boxes(
    ax,
    g: dict[str, np.ndarray],
    i_mod: int,
    els: np.ndarray,
    axes: tuple[int, int],
    color,
    arrow_width: float = 0.002,
) -> None:
    """draw detecting element boxes projected onto 2 coordinate axes

    boxes (convex hull of the projected corners), front faces (black lines),
    ADPs (dots) and box axes (arrows)
    """
    polys, front_faces = [], []
    for i_el in els:
        c = g["corners"][i_mod, i_el][:, axes]
        polys.append(c[ConvexHull(c).vertices])
        # projected front face (first 4 corners): its 2 most distant points
        f = c[:4]
        dist = np.linalg.norm(f[:, None] - f[None], axis=-1)
        i, j = np.unravel_index(np.argmax(dist), dist.shape)
        front_faces.append([f[i], f[j]])
    ax.add_collection(
        PolyCollection(polys, facecolors=[(*color[:3], 0.25)], edgecolors=[color], linewidths=0.6)
    )
    ax.add_collection(LineCollection(front_faces, colors="k", linewidths=1.2))
    adp = g["adp"][i_mod, els][:, axes]
    box_axis = g["box_axis"][i_mod, els][:, axes]
    ax.plot(adp[:, 0], adp[:, 1], ".", color="k", ms=2)
    ax.quiver(
        adp[:, 0],
        adp[:, 1],
        box_axis[:, 0],
        box_axis[:, 1],
        angles="xy",
        scale_units="xy",
        scale=0.25,
        width=arrow_width,
        color="k",
    )


def _plot_plane_view(
    fig,
    ax,
    geom: list[dict[str, np.ndarray]],
    decoded: list[dict[str, np.ndarray]],
    normal: int,
    background,
    background_label: str,
    cmap,
) -> float:
    """2D view of the detecting elements whose ADPs lie in a plane perpendicular to
    the coordinate axis `normal` (the plane through the ADP closest to 0), of the
    (projected) events and of the corresponding slice of a background image

    Returns the position of the plane along the normal axis.
    """
    h, v = [a for a in range(3) if a != normal]
    names = "xyz"
    all_adps = np.concatenate([g["adp"].reshape(-1, 3) for g in geom])
    plane = all_adps[np.argmin(np.abs(all_adps[:, normal])), normal]

    if background is not None:
        bg_img, bg_voxel_size, bg_origin = background
        bg_img = np.asarray(bg_img)
        bg_origin = np.asarray(bg_origin, dtype="float64")
        idx = int(round((plane - bg_origin[normal]) / bg_voxel_size[normal]))
        idx = min(max(idx, 0), bg_img.shape[normal] - 1)
        extent = [
            bg_origin[h] - 0.5 * bg_voxel_size[h],
            bg_origin[h] + (bg_img.shape[h] - 0.5) * bg_voxel_size[h],
            bg_origin[v] - 0.5 * bg_voxel_size[v],
            bg_origin[v] + (bg_img.shape[v] - 0.5) * bg_voxel_size[v],
        ]
        im = ax.imshow(
            np.take(bg_img, idx, axis=normal).T,
            origin="lower",
            extent=extent,
            cmap="Greys",
            alpha=0.8,
        )
        fig.colorbar(
            im, ax=ax, shrink=0.6, label=f"{background_label}\n({names[normal]} = {plane:.1f} mm)"
        )

    for i_type, g in enumerate(geom):
        in_plane = np.isclose(g["adp"][..., normal], plane, atol=1e-3)
        for i_mod in range(g["adp"].shape[0]):
            els = np.nonzero(in_plane[i_mod])[0]
            if els.size == 0:
                continue
            color = cmap((i_mod + 3 * i_type) % 20)
            _draw_boxes(ax, g, i_mod, els, (h, v), color)
            # module number outside of the module (in-plane direction from the center)
            mc = g["adp"][i_mod, els][:, (h, v)].mean(0)
            pos = mc * (1 + 22 / np.linalg.norm(mc)) if np.linalg.norm(mc) > 0 else mc
            ax.text(*pos, f"m{i_mod}", ha="center", va="center", fontsize="small", color=color, fontweight="bold")

    for i, ev in enumerate(decoded):
        color = plt.get_cmap("tab10")(i % 10)
        ax.plot([ev["p0"][h], ev["p1"][h]], [ev["p0"][v], ev["p1"][v]], "-", color=color, lw=0.8)
        ax.plot(ev["p0"][h], ev["p0"][v], "o", color=color, ms=5, mfc="none")
        ax.plot(ev["tof_interval"][:, h], ev["tof_interval"][:, v], "-", color=color, lw=3, alpha=0.6)
        ax.plot(ev["tof_pos"][h], ev["tof_pos"][v], "o", color=color, ms=4)

    all_corners = np.concatenate([g["corners"].reshape(-1, 3) for g in geom])
    lim_h = 1.25 * np.abs(all_corners[:, h]).max()
    lim_v = np.abs(all_corners[:, v]).max() + 0.1 * lim_h
    ax.set_xlim(-lim_h, lim_h)
    ax.set_ylim(-lim_v, lim_v)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{names[h]} (mm)")
    ax.set_ylabel(f"{names[v]} (mm)")
    return float(plane)


def plot_petsird_geometry(
    scanner: petsird.ScannerInformation,
    events: np.ndarray | None = None,
    background: tuple[np.ndarray, tuple[float, float, float], np.ndarray] | None = None,
    background_label: str = "",
    title: str | None = None,
    figsize: tuple[float, float] = (15, 15),
):
    """transaxial, sagittal and 3D view of the PETSIRD scanner geometry and of coincidence events

    Parameters
    ----------
    scanner : petsird.ScannerInformation
    events : array of shape (num_events, 5), optional
        (module type 0, module type 1, detection bin 0, detection bin 1, tof_idx)
        e.g. from read_first_events()
    background : (image, voxel_size, image_origin), optional
        3D image (e.g. activity) shown in the transaxial and sagittal views (slices
        through the planes of the shown detecting elements); image_origin is the
        center of voxel [0,0,0]
    background_label : str
        label of the background image
    """
    geom = get_detecting_element_geometry(scanner)
    decoded = _decode_events(scanner, geom, events) if events is not None else []

    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.7, 0.75, 1])
    ax_trans = fig.add_subplot(gs[0, 0])
    ax3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax_sag = fig.add_subplot(gs[1, :])
    ax_zoom = fig.add_subplot(gs[2, 0])
    gs_layers = gs[2, 1]
    if title is not None:
        fig.suptitle(title)

    cmap = plt.get_cmap("tab20")
    all_adps = np.concatenate([g["adp"].reshape(-1, 3) for g in geom])

    # ---------------------------------------------------------------------------
    # transaxial view (plane z = const) and sagittal view (plane x = const)
    events_text = (
        "events (3D, projected): line from detection bin 0 (open circle) to 1, "
        "filled circle + thick line: center of TOF bin +/- FWHM/2"
    )
    z_plane = _plot_plane_view(
        fig, ax_trans, geom, decoded, 2, background, background_label, cmap
    )
    ax_trans.set_title(
        f"transaxial view of the detecting elements with ADPs at z = {z_plane:.1f} mm\n"
        "boxes (black line: front face, arrow: box axis), dots: ADPs, m<i>: module index\n"
        + events_text.replace(", filled", ",\nfilled"),
        fontsize="small",
    )
    x_plane = _plot_plane_view(
        fig, ax_sag, geom, decoded, 0, background, background_label, cmap
    )
    ax_sag.set_title(
        f"sagittal view of the detecting elements with ADPs at x = {x_plane:.1f} mm "
        "(same symbols as in the transaxial view)",
        fontsize="small",
    )

    # ---------------------------------------------------------------------------
    # transaxial zoom on module 0 (of the first module type) with element indices
    g = geom[0]
    # plane through the ADPs of module 0 closest to its axial center (module 0 is not
    # necessarily at the z of the transaxial view)
    z_mod = g["adp"][0, :, 2]
    z_zoom = z_mod[np.argmin(np.abs(z_mod - 0.5 * (z_mod.min() + z_mod.max())))]
    els = np.nonzero(np.isclose(z_mod, z_zoom, atol=1e-3))[0]
    if els.size > 0:
        _draw_boxes(ax_zoom, g, 0, els, (0, 1), cmap(0), arrow_width=0.004)
        for i_el in els:
            p = g["adp"][0, i_el, :2] - 1.5 * g["box_axis"][0, i_el, :2]
            ax_zoom.text(*p, str(i_el), ha="center", va="center", fontsize=7)
        c = g["corners"][0][els][..., :2].reshape(-1, 2)
        margin = 3
        ax_zoom.set_xlim(c[:, 0].min() - margin, c[:, 0].max() + margin)
        ax_zoom.set_ylim(c[:, 1].min() - margin, c[:, 1].max() + margin)
    ax_zoom.set_aspect("equal")
    ax_zoom.set_xlabel("x (mm)")
    ax_zoom.set_ylabel("y (mm)")
    ax_zoom.set_title(
        f"module 0 (transaxial, z = {z_zoom:.1f} mm): detecting element indices,\n"
        "front faces (black), box axes (arrows), ADPs (dots)",
        fontsize="small",
    )

    # ---------------------------------------------------------------------------
    # 3D view of all detecting elements
    for i_type, g in enumerate(geom):
        num_modules = g["corners"].shape[0]
        faces, colors = [], []
        for i_mod in range(num_modules):
            color = cmap((i_mod + 3 * i_type) % 20)
            for corners in g["corners"][i_mod]:
                faces.extend(corners[f] for f in _BOX_FACES)
                colors.extend([(*color[:3], 0.12)] * len(_BOX_FACES))
        ax3d.add_collection3d(Poly3DCollection(faces, facecolors=colors, edgecolors="none"))

    ax3d.scatter(*all_adps.T, s=0.5, c="k", alpha=0.4, depthshade=False)
    for i, ev in enumerate(decoded):
        color = plt.get_cmap("tab10")(i % 10)
        ax3d.plot(*np.array([ev["p0"], ev["p1"]]).T, "-", color=color, lw=0.8)
        ax3d.plot(*ev["tof_interval"].T, "-", color=color, lw=3)
        ax3d.scatter(*ev["p0"], s=20, facecolors="none", edgecolors=[color])

    all_corners = np.concatenate([g["corners"].reshape(-1, 3) for g in geom])
    lo, hi = all_corners.min(0), all_corners.max(0)
    ax3d.set_xlim(lo[0], hi[0])
    ax3d.set_ylim(lo[1], hi[1])
    ax3d.set_zlim(lo[2], hi[2])
    ax3d.set_box_aspect(hi - lo)
    ax3d.set_xlabel("x (mm)")
    ax3d.set_ylabel("y (mm)")
    ax3d.set_zlabel("z (mm)")
    ax3d.set_zticks(np.round(np.linspace(lo[2], hi[2], 3)))
    ax3d.view_init(elev=30, azim=-60)
    ax3d.set_title(
        "3D view: detecting element boxes (color: module), ADPs and events", fontsize="small"
    )

    _plot_module_layers(fig, gs_layers, geom[0], i_mod=0)

    return fig


def _plot_module_layers(fig, subplot_spec, g: dict[str, np.ndarray], i_mod: int = 0):
    """detecting element indices of a module, seen along the (mean) box axis

    One panel per layer of detecting elements (distinct ADP positions along the box
    axis), showing the elements in the plane perpendicular to the box axis.
    """
    adp = g["adp"][i_mod]
    corners = g["corners"][i_mod]
    axis = g["box_axis"][i_mod].mean(0)
    axis /= np.linalg.norm(axis)

    # in-plane directions as seen from the scanner center (looking along the box axis):
    # up = scanner z axis (projected), right = box axis x up
    e_v = np.array([0.0, 0.0, 1.0]) - axis[2] * axis
    e_v /= np.linalg.norm(e_v)
    e_h = np.cross(axis, e_v)

    depth = adp @ axis
    layers = np.unique(np.round(depth, 3))

    sub = subplot_spec.subgridspec(1, len(layers))
    for i_layer, layer_depth in enumerate(layers):
        ax = fig.add_subplot(sub[0, i_layer])
        els = np.nonzero(np.isclose(depth, layer_depth, atol=1e-3))[0]
        polys = []
        for i_el in els:
            c = np.stack([corners[i_el] @ e_h, corners[i_el] @ e_v], axis=-1)
            polys.append(c[ConvexHull(c).vertices])
            ax.text(adp[i_el] @ e_h, adp[i_el] @ e_v, str(i_el), ha="center", va="center", fontsize=6)
        ax.add_collection(PolyCollection(polys, facecolors="none", edgecolors="tab:blue", linewidths=0.5))
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set_xlabel("horizontal (mm), seen from center")
        ax.set_ylabel("z (mm)")
        ax.set_title(
            f"module {i_mod}, layer {i_layer} (ADPs at {layer_depth:.1f} mm\n"
            "along box axis): element indices",
            fontsize="small",
        )
