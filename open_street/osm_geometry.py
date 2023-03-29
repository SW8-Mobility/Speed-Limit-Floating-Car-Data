import osmium
from osmium.geom import GeoJSONFactory
from typing import Any


class GeometryRoadHandler(osmium.SimpleHandler):
    """This class is responsible for generating a dictionary between OSM ids and their respective LineString through .apply_file"""

    def __init__(self) -> None:
        osmium.SimpleHandler.__init__(self)
        self.geometryDictionary: dict[str, str] = {}

    def way(self, w: Any) -> None:
        """This function gets called upon .apply_file for every road (OSM Way) that is in the file that apply_file is called with.

        Args:
            w (Any): OSM Way
        """
        if w.tags.get("highway") is not None:
            try:
                geo = GeoJSONFactory().create_linestring(w)  # Get the road linestring
                self.geometryDictionary[w.id] = geo
            except Exception as e:
                print("error", e)
                return


def geometry_dictionary_to_geojson(geoDict: dict[str, str]) -> str:
    """outputs geoJson formatted string from at osm_id -> linestring dictionary

    Args:
        geoDict (dict[str, str]): A dictionary that maps osm_ids to their geojson dictionary

    Returns:
        str: the entire geoJson formatted string.
    """
    # Start geoJson string
    featureCollecton = '{"type": "FeatureCollection","features": ['

    # Loop over geometry to build each LineString and give it an osm_id property
    for osm_id, geometry in geoDict.items():
        featureCollecton += '{"type": "Feature","geometry":'
        featureCollecton += geometry + ',"properties": {"osm_id":' + str(osm_id) + "}},"

    # Remove last comma since we are finished with the array
    featureCollecton = featureCollecton.rstrip(featureCollecton[-1])

    # End the geoJson string
    featureCollecton += "]}"

    return featureCollecton


def get_osmid_to_linestring_dictionary(OSMFilePath: str) -> dict[str, str]:
    """Get the dictionary that maps osm_id to a geojson linestring

    Args:
        OSMFilePath (str): File path to the .osm.pbf

    Returns:
        dict[str, str]: osm_id -> geojson linestring dictionary
    """
    geometryHandler = GeometryRoadHandler()
    geometryHandler.apply_file(OSMFilePath, locations=True)

    return geometryHandler.geometryDictionary


def main() -> None:
    filename_latest = "wkd/denmark-latest.osm.pbf"
    geoDict = get_osmid_to_linestring_dictionary(filename_latest)

    with open("denmark-latest-geometry.json", "w") as f:
        f.write(geometry_dictionary_to_geojson(geoDict))


if __name__ == "__main__":
    main()
