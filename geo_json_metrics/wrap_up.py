
segment_to_coordinates_list = list[tuple[int, list[int]]]

def wrap_up_(segments: list, coordinates: list) -> segment_to_coordinates_list:
    if len(segments) == 0:
        return []

    result = []
    current_seg = (segments[0], [])
    for seg, cor in zip(segments, coordinates):
        if seg != current_seg[0]: # new segment starts
            result.append(current_seg)
            current_seg = (seg, [cor])
        else:
            current_seg[1].append(cor)
    
    result.append(current_seg)

    return result

if __name__ == "__main__":
    coordinates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    segments = [1, 1, 1, 2, 2, 2, 3, 3, 3, None, None]
    c1 = [1]
    s1 = [1]

    print(wrap_up_(segments, coordinates))
    print(wrap_up_(s1, c1))