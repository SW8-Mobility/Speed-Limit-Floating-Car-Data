import sys

import osmium  # type: ignore
from osmium.osm.types import Way, Relation, Node, Area, Changeset  # type: ignore
import dataclasses
import json

# find OSM files here: https://drive.google.com/drive/folders/108HeqS1M9NDRV1RMFQHvuBL-OzHGuJRW?fbclid=IwAR0v3E1tr8Jet6XI3VwNnBakZVHcjkTUZ5zJS4VOqBArosg_hXaSb_grjUY
# use osm.pbf file


@dataclasses.dataclass
class Road:
    """Used to represent road in json file"""

    osm_id: int
    name: str
    speed_limit: int
    # should also have road type: landevej, hovedvej, motervej...


class RoadHandler(osmium.SimpleHandler):
    """Used for reading osm file. The corrensponding methods in
    this class will be called, when a node of that type is discovered.
    A road is found -> call way.

    Args:
        osmium (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        self.road_dict = dict()  # dict of all roads found, maps id to road
        self.tag_set = set()
        self.highway_value_set = set()
        # self.i = 0

    def way(self, w: Way):
        """Called when a road is found

        Args:
            w (Way): The way object
        """

        try:
            name = w.tags.get("name", default="na")
            id = int(w.id)

            if w.tags["highway"]:
                self.highway_value_set.add(w.tags["highway"])

            for entry in w.tags:
                self.tag_set.add(
                    entry[0]
                )  # Found useful so far: maxspeed, highway, traffic_calming(?)

            # self.tag_set.add(w.tags) # Add tags (no duplicates because it is a set)
            speed_limit = int(w.tags["maxspeed"])  # cause exception if not present
            # print(f'{name} ({id}): {speed_limit} km/h')
            self.road_dict[id] = Road(
                id, str(name), speed_limit
            ).__dict__  # save object as dict, used for json serializing
            # self.i += 1
        except ValueError as e:
            print("maxspeed not int: ", e)
        except:
            "Road without speedlimit"
        # if self.i > 20:
        #     self.write_to_json('new_test.json')
        #     exit()

    def write_to_json(self, filename: str):
        """output

        Args:
            filename (str): filename of output file
        """
        j = json.dumps(self.road_dict, indent=4)
        with open(filename, "w+") as outfile:
            outfile.write(j)

    # def relation(self, r: Relation):
    #     pass

    # def area(self, a):
    #     pass

    # def changeset(self, c):
    #     pass


def main():
    filename_2014 = "DK_2014.osm.pbf"
    filename_2023 = "denmark-latest.osm.pbf"
    rh = RoadHandler()
    rh.apply_file(filename_2014)
    sorted_set = sorted(rh.tag_set)

    print(f"Highway values: {rh.highway_value_set}")

    # sourceFile = open('tags.txt', 'w')
    # print(f"Tags: {sorted_set}", file=sourceFile)
    # sourceFile.close()

    # outfile_2014 = 'dk_roads_2014.json'
    # outfile_2023 = 'dk_roads_2023.json'
    # rh.write_to_json(outfile_2014)
    # print("number of roads ", len(rh.road_dict))


if __name__ == "__main__":
    main()
