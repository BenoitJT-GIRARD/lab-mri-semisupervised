# Figure style v1 — generated 2026-09-16. Copy: edit the source, not this file.

"""Figure style: one palette, one writer, and a refusal to publish an unreadable figure.

Every figure of this project is written by :func:`save_figure`. It fixes the size, the
resolution and the background, stamps the sample size into the image, and records what it
wrote in ``reports/figures/MANIFEST.json``.

It also refuses. A figure whose axes carry data without saying what they measure, or whose
sample size is not declared, is not written at all — a reader cannot check a number they
cannot read, and a figure that omits its N is a claim without an effective.

Colours are taken by ROLE, never by position. ``series_colours`` assigns by name, so an
entity keeps its colour across every figure; a control or a baseline is grey by contract, a
reference line is dashed. Nothing beyond matplotlib is needed to regenerate every figure, and
nothing at all to read the palette: matplotlib is imported by the functions that plot, so a
project that builds its figures by hand — :func:`save_svg` — carries this module without
carrying a plotting dependency it never calls.

    from .figure_style import apply_style, save_figure, series_colours

    apply_style()
    fig, ax = plt.subplots()
    colours = series_colours(["supervised", "permuted control"], control=["permuted control"])
    ...
    save_figure(fig, "reports/figures/roc_arms.png", n=99,
                dispersion="95% bootstrap CI, 2000 resamples",
                source="scripts/build_figures.py")
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # the only place matplotlib is named outside a function body
    from matplotlib.colors import LinearSegmentedColormap

# --- palette (generated) ---
# Generated from the palette source. Do not edit by hand: edit the source and sync.

STYLE_VERSION = 1

PALETTE: dict[str, str] = {
    "primary": "#123F5A",
    "quaternary": "#6F3A55",
    "quinary": "#745913",
    "control": "#66707A",
    "secondary": "#BC6A24",
    "tertiary": "#4E9DB5",
    "reference": "#B0342C",
    "ink": "#1B1F23",
    "muted": "#5B646D",
    "grid": "#DDE1E4",
    "surface": "#F3F5F6",
    "paper": "#FFFFFF",
}

#: Series colours in order of lightness. A control is not in this cycle: it is grey.
SERIES: tuple[str, ...] = (
    PALETTE["primary"],
    PALETTE["quaternary"],
    PALETTE["quinary"],
    PALETTE["secondary"],
    PALETTE["tertiary"],
)

STATE: dict[str, str] = {
    "ok": "#2E6B4F",
    "warning": "#A8621B",
    "danger": "#8F2A22",
    "neutral": "#8A939B",
}

SEQUENTIAL: tuple[str, ...] = ("#F4EAD9", "#E3C46A", "#BC6A24", "#6A3516")
DIVERGING: tuple[str, ...] = ("#123F5A", "#7FA8B8", "#DDE1E4", "#DFA96B", "#BC6A24")

FONT_STACK: tuple[str, ...] = ("Source Sans 3", "Inter", "DejaVu Sans")
FIGURE_SIZE: tuple[float, float] = (7.0, 4.4)
DPI = 200

# --- end palette (generated) ------------------------------------------------


def apply_style() -> None:
    """Install the style. Call once, before plotting."""

    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.figsize": FIGURE_SIZE,
            "figure.dpi": 110,
            "savefig.dpi": DPI,
            "figure.facecolor": PALETTE["paper"],
            "savefig.facecolor": PALETTE["paper"],
            "axes.facecolor": PALETTE["paper"],
            "axes.edgecolor": PALETTE["muted"],
            "axes.labelcolor": PALETTE["ink"],
            "axes.titlecolor": PALETTE["ink"],
            "axes.titlesize": 12,
            # "bold", not "semibold": the fallback of the stack is DejaVu Sans, which has no
            # semibold face, and matplotlib logs a substitution line into every figure build
            # and every notebook output that draws one.
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 11,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": mpl.cycler(color=list(SERIES)),
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.7,
            "text.color": PALETTE["ink"],
            "font.family": "sans-serif",
            "font.sans-serif": list(FONT_STACK),
            "font.size": 10,
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
        }
    )


def series_colours(
    names: Sequence[str],
    *,
    control: Iterable[str] = (),
    reference: Iterable[str] = (),
) -> dict[str, str]:
    """Map entity names to colours, stably and by role.

    Sorting is by name, not by call order, so the same entity keeps its colour in every
    figure — including figures drawn by different scripts. Controls and baselines are grey,
    reference lines get the reserved colour, and neither consumes a series slot.
    """

    controls, references = set(control), set(reference)
    unknown = (controls | references) - set(names)
    if unknown:
        raise ValueError(f"named as control or reference but never plotted: {sorted(unknown)}")

    colours: dict[str, str] = {}
    slot = 0
    for name in sorted(names):
        if name in references:
            colours[name] = PALETTE["reference"]
        elif name in controls:
            colours[name] = PALETTE["control"]
        else:
            if slot >= len(SERIES):
                raise ValueError(
                    f"{len(names)} series for {len(SERIES)} distinguishable colours: "
                    "split the figure; no hue is reused"
                )
            colours[name] = SERIES[slot]
            slot += 1
    return colours


#: Below this many values a cloud of points reads as noise rather than as a distribution.
DOTS_NEED = 6


def distribution(
    ax: Any,
    positions: Sequence[float],
    groups: Sequence[Sequence[float]],
    *,
    orient: str = "v",
    colours: Sequence[str] | None = None,
    width: float = 0.62,
    jitter: float = 0.16,
    seed: int = 20260916,
    dot_size: float = 15.0,
    dot_alpha: float = 0.55,
    label: str | None = None,
) -> dict[str, list[float]]:
    """Every individual as a point, the mean as a rule, and the standard error around it.

    A mean drawn alone hides the sample it came from: two groups with the same mean and
    opposite spreads become the same picture. Here every value is drawn, and the reader can
    see the sample before reading the summary of it.

    Above ``DOTS_NEED`` values the mean is a short horizontal rule and nothing is filled: the
    points carry the comparison, so the axis is free to start where the data does. Below it,
    four points read as noise rather than as a distribution, so the mean becomes a bar from
    zero — the bar is what makes the comparison legible — and the points sit on top of it, so
    that no individual is hidden by the summary.

    The error bar is the **standard error of the mean**, which is what a reader comparing two
    means needs. A standard deviation describes the sample, not the precision of its mean;
    when that is the intended message, draw it yourself and name it.

    The jitter comes from a seeded generator, so the same values give the same picture. A
    figure that moved on every regeneration could not be compared with the one it replaces.
    """

    import numpy as np

    if orient not in {"v", "h"}:
        raise ValueError("orient is 'v' or 'h'")
    if len(positions) != len(groups):
        raise ValueError(f"{len(positions)} positions for {len(groups)} groups")

    scatterer = np.random.default_rng(seed)
    means: list[float] = []
    sems: list[float] = []
    counts: list[int] = []

    for index, (centre, values) in enumerate(zip(positions, groups, strict=True)):
        sample = np.asarray(list(values), dtype=float)
        sample = sample[~np.isnan(sample)]
        n = int(sample.size)
        counts.append(n)
        if n == 0:
            means.append(float("nan"))
            sems.append(float("nan"))
            continue
        mean = float(sample.mean())
        sem = float(sample.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        means.append(mean)
        sems.append(sem)

        colour = colours[index] if colours is not None else SERIES[index % len(SERIES)]
        spread = centre + scatterer.uniform(-jitter, jitter, size=n)
        first = label if index == 0 else None

        if n >= DOTS_NEED:
            left, right = centre - width / 2, centre + width / 2
            if orient == "v":
                ax.scatter(
                    spread,
                    sample,
                    s=dot_size,
                    color=colour,
                    alpha=dot_alpha,
                    linewidths=0,
                    zorder=2,
                )
                ax.hlines(mean, left, right, color=colour, linewidth=2.2, zorder=3)
                if sem:
                    ax.errorbar(
                        centre,
                        mean,
                        yerr=sem,
                        fmt="none",
                        ecolor=colour,
                        elinewidth=1.6,
                        capsize=5,
                        zorder=4,
                        label=first,
                    )
            else:
                ax.scatter(
                    sample,
                    spread,
                    s=dot_size,
                    color=colour,
                    alpha=dot_alpha,
                    linewidths=0,
                    zorder=2,
                )
                ax.vlines(mean, left, right, color=colour, linewidth=2.2, zorder=3)
                if sem:
                    ax.errorbar(
                        mean,
                        centre,
                        xerr=sem,
                        fmt="none",
                        ecolor=colour,
                        elinewidth=1.6,
                        capsize=5,
                        zorder=4,
                        label=first,
                    )
        elif orient == "v":
            ax.bar(centre, mean, width=width, color=colour, alpha=0.85, zorder=1)
            if sem:
                ax.errorbar(
                    centre,
                    mean,
                    yerr=sem,
                    fmt="none",
                    ecolor=PALETTE["ink"],
                    elinewidth=1.4,
                    capsize=5,
                    zorder=3,
                    label=first,
                )
            ax.scatter(
                spread,
                sample,
                s=dot_size + 6,
                color=PALETTE["ink"],
                alpha=0.8,
                linewidths=0,
                zorder=4,
            )
        else:
            ax.barh(centre, mean, height=width, color=colour, alpha=0.85, zorder=1)
            if sem:
                ax.errorbar(
                    mean,
                    centre,
                    xerr=sem,
                    fmt="none",
                    ecolor=PALETTE["ink"],
                    elinewidth=1.4,
                    capsize=5,
                    zorder=3,
                    label=first,
                )
            ax.scatter(
                sample,
                spread,
                s=dot_size + 6,
                color=PALETTE["ink"],
                alpha=0.8,
                linewidths=0,
                zorder=4,
            )

    return {"mean": means, "sem": sems, "n": counts}


def reference_line(
    ax: Any,
    *,
    y: float | None = None,
    x: float | None = None,
    diagonal: bool = False,
    label: str | None = None,
) -> None:
    """Draw a reference — zero, chance, perfect calibration — dashed and in the reserved colour."""

    kwargs = {"color": PALETTE["reference"], "linestyle": "--", "linewidth": 1.2, "label": label}
    if diagonal:
        ax.plot([0, 1], [0, 1], **kwargs)
    elif y is not None:
        ax.axhline(y, **kwargs)
    elif x is not None:
        ax.axvline(x, **kwargs)
    else:
        raise ValueError("reference_line needs y=, x= or diagonal=True")


def sequential_cmap() -> LinearSegmentedColormap:
    """Ordered quantities. Pass this to any third-party plot that takes a colormap."""

    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("seq", list(SEQUENTIAL))


def diverging_cmap() -> LinearSegmentedColormap:
    """Signed quantities around a neutral point."""

    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("div", list(DIVERGING))


# ---------------------------------------------------------------------------
# Writing, and refusing to write
# ---------------------------------------------------------------------------


def _title_of(ax: Any) -> str:
    """The title as it was set, wherever the style put it.

    ``set_title`` honours ``axes.titlelocation``, and this style left-aligns titles, so the
    text goes into the left slot. ``get_title()`` reads the centre slot and answers with an
    empty string: the manifest of a figure that carries a title would record none.
    """
    import matplotlib as mpl

    for loc in (mpl.rcParams.get("axes.titlelocation", "center"), "center", "left", "right"):
        title = ax.get_title(loc=loc)
        if title.strip():
            return title
    return ""


def _axes_carrying_data(fig: Any) -> list[Any]:
    return [ax for ax in fig.axes if ax.has_data() and ax.get_label() != "<colorbar>"]


def _unlabelled(fig: Any) -> list[str]:
    """Axes that plot something without saying what it measures."""

    problems = []
    for index, ax in enumerate(_axes_carrying_data(fig)):
        missing = [
            name
            for name, value in (("x", ax.get_xlabel()), ("y", ax.get_ylabel()))
            if not value.strip()
        ]
        if missing:
            title = ax.get_title() or f"axes {index}"
            problems.append(f"{title}: no {' or '.join(missing)} label")
    return problems


def _shows_dispersion(fig: Any) -> bool:
    """Does the figure draw error bars, a band, or a confidence interval?"""

    for ax in fig.axes:
        for container in getattr(ax, "containers", []):
            if type(container).__name__ == "ErrorbarContainer":
                return True
        for collection in ax.collections:
            if type(collection).__name__ in {"PolyCollection", "FillBetweenPolyCollection"}:
                return True
    return False


def _format_n(n: int | Mapping[str, int] | str) -> str:
    if isinstance(n, Mapping):
        return ", ".join(f"n({key}) = {value}" for key, value in n.items())
    if isinstance(n, int):
        return f"n = {n}"
    return str(n).strip()


def _stamp(fig: Any, parts: Sequence[str]) -> None:
    """Write the effective and the dispersion into the image, not only into the caption."""

    fig.text(
        0.005,
        0.005,
        "  ·  ".join(p for p in parts if p),
        fontsize=8,
        color=PALETTE["muted"],
        ha="left",
        va="bottom",
    )


def _manifest_path(path: Path) -> Path:
    for parent in path.parents:
        if parent.name == "figures":
            return parent / "MANIFEST.json"
    return path.parent / "MANIFEST.json"


def _record(path: Path, entry: dict[str, Any]) -> None:
    manifest = _manifest_path(path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"schema": "image-manifest/1", "images": {}}
    if manifest.exists():
        with contextlib.suppress(json.JSONDecodeError):
            payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.setdefault("images", {})
    key = path.name
    with contextlib.suppress(ValueError):
        key = str(path.relative_to(manifest.parent)).replace("\\", "/")
    payload["images"][key] = entry
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def save_figure(
    fig: Any,
    path: str | Path,
    *,
    n: int | Mapping[str, int] | str,
    dispersion: str | None = None,
    source: str | None = None,
    note: str | None = None,
) -> Path:
    """Write a figure, or refuse it.

    ``n`` is the effective the figure rests on, and it is mandatory. ``dispersion`` names what
    an error bar or a band means — "±1 SD across 25 folds", "95% bootstrap CI, 2000 resamples" —
    and becomes mandatory as soon as the figure draws one: a spread nobody can name is a spread
    nobody can read. ``source`` is the script that produced the figure.
    """

    target = Path(path)
    refusals = _unlabelled(fig)
    if not str(n).strip():
        refusals.append("the sample size is not declared")
    if _shows_dispersion(fig) and not (dispersion or "").strip():
        refusals.append("error bars or a band are drawn without being named")
    if refusals:
        raise ValueError(f"{target.name} was not written — " + "; ".join(refusals))

    stamp = _format_n(n)
    _stamp(fig, [stamp, dispersion or "", note or ""])

    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=DPI, bbox_inches="tight", facecolor=PALETTE["paper"])

    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    _record(
        target,
        {
            "n": stamp,
            "dispersion": dispersion,
            "source": source,
            "axes": [
                {"x": ax.get_xlabel(), "y": ax.get_ylabel(), "title": _title_of(ax)}
                for ax in _axes_carrying_data(fig)
            ],
            "dpi": DPI,
            "style_version": STYLE_VERSION,
            "sha256": digest,
            "written": datetime.now(tz=UTC).date().isoformat(),
        },
    )
    return target


def save_svg(
    svg: str,
    path: str | Path,
    *,
    n: int | Mapping[str, int] | str,
    axes: Mapping[str, str],
    dispersion: str | None = None,
    source: str | None = None,
) -> Path:
    """Write a hand-built SVG under the same contract as a plotted figure.

    A figure written by hand escapes every check matplotlib allows, so the axis labels and the
    effective are passed explicitly and recorded the same way.
    """

    target = Path(path)
    missing = [key for key in ("x", "y") if not axes.get(key, "").strip()]
    if missing or not str(n).strip():
        raise ValueError(
            f"{target.name} was not written — "
            + "; ".join(
                ([f"no {' or '.join(missing)} label"] if missing else [])
                + ([] if str(n).strip() else ["the sample size is not declared"])
            )
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(svg, encoding="utf-8", newline="")
    _record(
        target,
        {
            "n": _format_n(n),
            "dispersion": dispersion,
            "source": source,
            "axes": [{"x": axes["x"], "y": axes["y"], "title": axes.get("title", "")}],
            "dpi": None,
            "style_version": STYLE_VERSION,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "written": datetime.now(tz=UTC).date().isoformat(),
        },
    )
    return target


def close(fig: Any) -> None:
    """Close a figure. Kept here so a script imports one module, not two."""

    import matplotlib.pyplot as plt

    plt.close(fig)
