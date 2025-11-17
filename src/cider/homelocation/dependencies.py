# Copyright ©2022-2023. The Regents of the University of California
# (Regents). All Rights Reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the
# distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import geovoronoi
import pandas as pd
import geopandas as gpd


def _deduplicate_points_within_buffer(
    xy_points: gpd.GeoDataFrame,
    points_id_col: str,
    buffer_distance: float = 0.0001,
) -> gpd.GeoDataFrame:
    """
    Deduplicate points that are within a certain buffer distance of each other.

    Args:
        xy_points: geopandas df with point geometry column
        buffer_distance: distance threshold to consider points as duplicates

    Returns:
        geopandas df with duplicates removed
    """
    xy_points.to_crs(epsg=3857, inplace=True)
    xy_points_buffered = xy_points.copy()
    xy_points_buffered["geometry"] = xy_points.geometry.buffer(buffer_distance)

    # Do an sjoin of buffered points with original points to find overlaps
    joined = gpd.sjoin(
        xy_points_buffered,
        xy_points,
        predicate="intersects",
    )

    # De-duplicate points:
    # 1. Keep only rows where points intersect with other points (drop self-joins)
    # 2. For each group of intersecting points, keep only one representative (ids_left)
    # 3. For each point, keep it if it's the first time we encounter it. Drop others that it intersects with.
    joined = joined.loc[
        joined[points_id_col + "_left"] != joined[points_id_col + "_right"]
    ]
    deduplicated_points = xy_points.copy()
    groups = joined.groupby(points_id_col + "_left")
    ids_to_drop = joined[points_id_col + "_right"].unique().tolist()

    for _, group in groups:
        ids_to_drop = [
            id for id in ids_to_drop if id != group[points_id_col + "_left"].iloc[0]
        ]
        deduplicated_points = deduplicated_points.loc[
            ~deduplicated_points[points_id_col].isin(ids_to_drop)
        ]

    deduplicated_points.to_crs(epsg=4326, inplace=True)
    return deduplicated_points


def get_voronoi_tessellation(
    xy_points: gpd.GeoDataFrame,
    boundary_shapefile: gpd.GeoDataFrame,
    points_id_col: str,
    buffer_distance_for_deduplication: float = 1e-4,
) -> gpd.GeoDataFrame:
    """
    Create voronoi tessellation starting from points and a shapefile to define country boundaries

    Args:
        xy_points: geopandas df with point geometry column
        boundary_shapefile: geopandas df with external boundaries
        points_id_col: string point identifier for points
        buffer_distance_for_deduplication: distance threshold to consider points as duplicates

    Returns:
        voronoi_regions: geopandas df with geometry column containing voronoi tessellation polygons
    """
    if points_id_col not in xy_points.columns:
        raise ValueError(f"'{points_id_col}' not found in xy_points columns")

    xy_points = _deduplicate_points_within_buffer(
        xy_points=xy_points,
        points_id_col=points_id_col,
        buffer_distance=buffer_distance_for_deduplication,
    )
    boundary = boundary_shapefile.to_crs(epsg=4326).union_all()
    # Filter out points outside boundary
    points_within_boundary = xy_points[xy_points.within(boundary)].reset_index(
        drop=True
    )
    coords = points_within_boundary.geometry

    # geovoronoi returns two dictionaries. Both have arbitrary indices as keys.
    # One maps these to regions, the other to one or more indices into the list
    # of labels, representing towers in that given region.
    regions, label_indices = geovoronoi.voronoi_regions_from_coords(coords, boundary)

    # Merge regions and labels into a single gdf, then merge with points
    regions = gpd.GeoDataFrame({"ids": regions.keys(), "geometry": regions.values()})
    label_indices = pd.DataFrame(
        {
            "ids": label_indices.keys(),
            "point_index": [v[0] for v in label_indices.values()],
        }
    )

    regions_and_labels = regions.merge(label_indices, on="ids", how="inner")
    points_within_boundary["point_index"] = points_within_boundary.index
    points_within_boundary.drop(columns=["geometry"], inplace=True)
    points_to_regions = points_within_boundary.merge(
        regions_and_labels, on="point_index", how="inner"
    )

    # Get final voronoi regions with point ids
    voronoi_regions = gpd.GeoDataFrame(
        points_to_regions[[points_id_col, "geometry"]], geometry="geometry"
    )

    return voronoi_regions
