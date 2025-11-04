import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path


def make_location_map(
    inferred_home_locations: gpd.GeoDataFrame,
    column_to_plot_label: str,
    column_to_plot_markersize: str | None = None,
    boundaries_shapefile: gpd.GeoDataFrame | None = None,
    plot_title: str | None = None,
    **plotting_kwargs,
) -> gpd.GeoDataFrame:
    """
    Plot a map of inferred home locations

    Args:
        inferred_home_locations: geopandas df with inferred home locations
        column_to_plot_label: column name in inferred_home_locations to plot as labels
        column_to_plot_markersize: column name in inferred_home_locations to plot as marker sizes. If None, all markers will be the same size
        boundaries_shapefile: geopandas df with boundaries polygons. If None, no boundaries will be plotted
        plot_title: title for the plot. If None, the column_to_plot_label will be used as title
        **plotting_kwargs: additional keyword arguments for geopandas plot method (e.g. color, cmap, etc.)
    """
    if column_to_plot_label not in inferred_home_locations.columns:
        raise ValueError(
            f"Inferred home locations df must contain  `{column_to_plot_label}` columns"
        )

    if (column_to_plot_markersize is not None) and (
        column_to_plot_markersize not in inferred_home_locations.columns
    ):
        raise ValueError(
            f"Inferred home locations df does not contain  `{column_to_plot_markersize}` columns"
        )

    with mpl.rc_context(fname=Path(__file__).parent / "../matplotlibrc"):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.axis("off")
        ax.set_title(
            plot_title if plot_title else column_to_plot_label.replace("_", " ").title()
        )

        markersizes: gpd.GeoSeries = None  # Default marker size
        if column_to_plot_markersize:
            markersizes = inferred_home_locations[column_to_plot_markersize].values
            markersizes = (
                (markersizes - markersizes.min())
                * 50
                / (markersizes.max() - markersizes.min())
            )

        inferred_home_locations.plot(
            column=column_to_plot_label,
            ax=ax,
            markersize=markersizes,
            **plotting_kwargs,
        )

        if boundaries_shapefile is not None:
            boundaries_shapefile.plot(
                ax=ax, linewidth=1, facecolor="none", edgecolor="black"
            )

        fig.tight_layout()

    return fig
