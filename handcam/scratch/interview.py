#(1, 1)(3, 4)(5, 7)(4, 4)

#(1, 1) - (3, 4) - (5, 7)

# return list of connected points

points = [(1, 1), (3, 4), (5, 7), (4, 4)]


def find_max_points(points):
    line_dict = {}  # {(3/2,1): {(1,1): None, (3,4): None,...]}}


    for point_1 in points:
        for point_2 in points:
            if point_1 is point_2:
                continue

            new_line = get_line(point_1, point_2)

            try:
                line_dict[new_line][point_1] = None

            except KeyError as e:
                line_dict[new_line] = {}
                line_dict[new_line][point_1] = None

            line_dict[new_line][point_2] = None

    max_line = None
    max_points = 0

    for line in line_dict.keys():
        if len(line_dict[line]) > max_points:
            max_points = len(line_dict[line])
            max_line = line

    return max_line, max_points, line_dict[max_line]


def get_line(point_1, point_2):
    x1 = point_1[0]
    x2 = point_2[0]
    y1 = point_1[1]
    y2 = point_2[1]

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    return (slope, intercept)

max_line, max_points, out_points = find_max_points(points)

print(max_line)
print(max_points)
print(out_points.keys())
