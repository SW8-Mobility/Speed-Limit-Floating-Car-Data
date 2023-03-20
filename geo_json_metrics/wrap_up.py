from typing import Any, Callable

coordinates = [1, 2, 3, 4, 5, 6, 7, 8, 9]
segments = [1, 1, 1, 2, 2, 2, 3, 3, 3]


def take_while(l: list, pred: Callable):
    if len(l) == 0 or not pred(l[0]):
        return []
    else:
        return [l[0]] + take_while(l[1:], pred)


def drop_while(l: list, pred: Callable):
    if len(l) == 0 or not pred(l[0]):
        return l
    else:
        return drop_while(l[1:], pred)


def wrap_up(l: list):
    def __wrap_up(first_elem: Any, rest: list[Any]):
        if len(rest) == 0:
            return [first_elem]

        l = [first_elem] + take_while(rest, lambda x: x == first_elem)
        rest = drop_while(rest, lambda x: x == first_elem)
        return [l] + wrap_up(rest)

    if len(l) == 0:
        return []
    else:
        return __wrap_up(l[0], l[1:])


def segment_wrap_up(combined: list[tuple]) -> list[list[tuple]]:
    def __wrap_up(first_elem: Any, rest: list[Any]):
        if len(rest) == 0:
            return [first_elem]

        l = [first_elem] + take_while(rest, lambda x: x[0] == first_elem[0])
        rest = drop_while(rest, lambda x: x[0] == first_elem[0])
        return [l] + segment_wrap_up(rest)

    if len(combined) == 0:
        return []
    else:
        return __wrap_up(combined[0], combined[1:])


def untangle_wrapup(wrapup_out: list[list[tuple]]) -> list[tuple]:
    result = []
    for segment in wrapup_out:
        temp = (segment[0][0], [])
        for _, coordinate in segment:
            temp[1].append(coordinate)
        result.append(temp)
    return result


def map_segments_to_coordinates(segments: list, coordinates: list) -> list[tuple]:
    temp = segment_wrap_up(list(zip(segments, coordinates)))
    return untangle_wrapup(temp)


# zipped = list(zip(coordinates, segments))
res = segment_wrap_up(list(zip(segments, coordinates)))
print(res)
print(untangle_wrapup(res))
# print(take_while(segments, lambda x: x==segments[0]))
# print(drop_while(segments, lambda x: x==segments[0]))
