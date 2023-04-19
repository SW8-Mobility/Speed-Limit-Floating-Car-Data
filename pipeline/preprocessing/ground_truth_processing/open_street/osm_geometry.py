import osmium
from osmium.geom import GeoJSONFactory
from typing import Any
import json


class GeometryRoadHandler(osmium.SimpleHandler):
    """This class is responsible for generating a dictionary between OSM ids and their respective LineString through .apply_file"""

    def __init__(self) -> None:
        osmium.SimpleHandler.__init__(self)
        self.geometry_dictionary: dict[str, tuple[str, str]] = {}

    def way(self, w: Any) -> None:
        """Looks for all ways with the tag "highway" and adds their osm id and linestring to the self.geometryDictionary.
        This function is used by apply_file(), which is method inherited by osmium.SimpleHandler.
        apply_file will look for a function called way() and apply the function on each OSM way (road) in a given .pbf file

        Args:
            w (Any): OSM Way
        """
        if (
            w.tags.get("highway") is not None and w.tags.get("name") is not None
        ):  # The highway tag annotates the type of road, e.g. 'path' or 'motorway'
            try:
                geo = GeoJSONFactory().create_linestring(w)  # Get the road linestring
                self.geometry_dictionary[w.id] = (
                    geo,
                    w.tags.get("name"),
                )  # geo is also a string since the format is json, w.id is an int
            except Exception as e:
                print("error", e)
                return


def geometry_dictionary_to_geojson(
    geo_dict: dict[str, tuple[str, str]],
) -> str:  # TODO: is the type of dict correct?
    """outputs geoJson formatted string from at osm_id -> linestring dictionary

    Args:
        geoDict ('dict[str, tuple[str, str]]'): A dictionary that maps osm_ids to their geojson dictionary

    Returns:
        str: the entire geoJson formatted string.
    """
    # Start geoJson string
    feature_collecton: str = '{"type":"FeatureCollection","features":['

    # Loop over geometry to build each LineString and give it an osm_id property
    for osm_id, (geometry, name) in geo_dict.items():
        feature_collecton += '{"type":"Feature","geometry":' + geometry
        feature_collecton += (
            ',"properties":{"osm_id":'
            + str(osm_id)
            + ',"osm_name":'
            + json.dumps(name)
            + "}},"
        )

    # Remove last comma since we are finished with the array
    feature_collecton = feature_collecton.rstrip(feature_collecton[-1])

    # End the geoJson string
    feature_collecton += "]}"

    return feature_collecton


def get_osmid_to_linestring_dictionary(
    osm_file_path: str,
) -> dict[str, tuple[str, str]]:
    """Get the dictionary that maps osm_id to a geojson linestring

    Args:
        OSMFilePath (str): File path to the .osm.pbf

    Returns:
        'dict[str, tuple[str, str]]': osm_id -> geojson LineString
    """
    geometry_handler = GeometryRoadHandler()
    geometry_handler.apply_file(osm_file_path, locations=True)

    return geometry_handler.geometry_dictionary


def main() -> None:
    filename_latest = "wkd/denmark-latest.osm.pbf"
    geo_dict = get_osmid_to_linestring_dictionary(filename_latest)

    with open("denmark-latest-geometry.json", "w") as f:
        f.write(geometry_dictionary_to_geojson(geo_dict))


if __name__ == "__main__":
    main()
