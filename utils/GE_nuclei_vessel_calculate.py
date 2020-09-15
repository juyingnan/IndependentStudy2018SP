from skimage import io
import math
import csv
import os


def write_csv(path, list_of_columns, list_of_names=None):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if list_of_names is not None:
            writer.writerow(list_of_names)
        lines = []
        for _i in range(len(list_of_columns[0])):
            lines.append([column[_i] for column in list_of_columns])
        writer.writerows(lines)


def read_csv(path):
    with open(path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        _headers = next(reader, None)

        # get column
        _columns = {}
        for h in _headers:
            _columns[h] = []
        for row in reader:
            for h, v in zip(_headers, row):
                _columns[h].append(v)

        for item in _headers:
            print(item)

        return _headers, _columns


# prs = Presentation(r'C:\Users\bunny\Desktop\KidneyAnnotated-GW.pptx')
# print(prs.slide_height, prs.slide_width)
# slide = prs.slides[0]
#
# shapes = slide.shapes
#
# print(len(shapes))
#
# unit_per_um = 1024646 / 50
# unit_per_pixel = 28346400 / 2232
# pixel_per_um = unit_per_um / unit_per_pixel
#
nuclei_id_list = list()
nuclei_x_list = list()
nuclei_y_list = list()
nuclei_distance_list = list()
nuclei_nearest_vessel_x_list = list()
nuclei_nearest_vessel_y_list = list()

root_path = r'C:\Users\bunny\Desktop\Region_6'
nuclei_file_name = 'quant_slide77_region6.csv'
vessel_file_name = 'vessel.csv'
nuclei_file_path = os.path.join(root_path, nuclei_file_name)

n_headers, n_columns = read_csv(nuclei_file_path)
for i in range(0, 3):  # len(n_headers)):
    n_columns[n_headers[i]] = [float(value) for value in n_columns[n_headers[i]]]
n_columns['Y'] = [float(value) for value in n_columns['Y']]

nuclei_x_list = [float(value / 2) for value in n_columns['Y']]
nuclei_y_list = [float(value / 2) for value in n_columns['X']]
nuclei_id_list = [int(value) for value in n_columns['Cell ID']]
nuclei_class_list = [int(value) for value in n_columns['Class']]

# nuclei_image = io.imread('../images/nuclei_ml.png')  # [::-1, :]
# _nid = 0
# for i in range(len(nuclei_image)):
#     row = nuclei_image[i]
#     for j in range(len(row)):
#         pixel = row[j]
#         if pixel[0] > 200:
#             # each nuclear is 2x2 pixels
#             # get the average (+0.5)
#             # and erase other 3 points
#             nuclei_y_list.append((i + 0.5) / pixel_per_um)
#             nuclei_x_list.append((j + 0.5) / pixel_per_um)
#             nuclei_image[i][j + 1] *= 0
#             nuclei_image[i + 1][j] *= 0
#             nuclei_image[i + 1][j + 1] *= 0
#             nuclei_id_list.append(_nid)
#             _nid += 1
print(len(nuclei_x_list))

vessel_x_list = list()
vessel_y_list = list()

vessel_image = io.imread(r'C:\Users\bunny\Desktop\Region_6\Vessels_S77R6_mask_05.png')  # [::-1, :]
for i in range(len(vessel_image)):
    row = vessel_image[i]
    for j in range(len(row)):
        pixel = row[j]
        if pixel > 200:  # if pixel[0] > 200:
            vessel_y_list.append(i)
            vessel_x_list.append(j)
print(len(vessel_x_list))

write_csv('../csv/nuclei_GE.csv',
          [nuclei_id_list,
           nuclei_x_list,
           nuclei_y_list],
          ['id', 'x', 'y'])

write_csv('../csv/vessel_GE.csv',
          [vessel_x_list, vessel_y_list],
          ['x', 'y'])

for nid in range(len(nuclei_id_list)):
    _min_dist = 600
    _min_vessel_x = 0
    _min_vessel_y = 0
    _nx = nuclei_x_list[nid]
    _ny = nuclei_y_list[nid]
    _has_near = False
    for v in range(len(vessel_x_list)):
        _vx = vessel_x_list[v]
        _vy = vessel_y_list[v]
        if abs(_nx - _vx) < _min_dist and abs(_ny - _vy) < _min_dist:
            _dist = math.sqrt((_nx - _vx) ** 2 + (_ny - _vy) ** 2)
            if _dist < _min_dist:
                _has_near = True
                _min_dist = _dist
                _min_vessel_x = _vx
                _min_vessel_y = _vy
    if not _has_near:
        print("NO NEAR")
    nuclei_distance_list.append(_min_dist)
    nuclei_nearest_vessel_x_list.append(_min_vessel_x)
    nuclei_nearest_vessel_y_list.append(_min_vessel_y)
    if nid % 100 == 0:
        print('\r' + str(nid), end='')

print(len(nuclei_id_list), len(nuclei_x_list), len(nuclei_y_list), len(nuclei_distance_list),
      len(nuclei_nearest_vessel_x_list), len(nuclei_nearest_vessel_y_list))

write_csv('../csv/nuclei_GE.csv',
          [nuclei_id_list,
           nuclei_x_list,
           nuclei_y_list,
           nuclei_class_list,
           nuclei_distance_list,
           nuclei_nearest_vessel_x_list,
           nuclei_nearest_vessel_y_list],
          ['id', 'x', 'y', 'class', 'distance', 'vx', 'vy'])

write_csv('../csv/vessel_GE.csv',
          [vessel_x_list, vessel_y_list],
          ['x', 'y'])

import matplotlib.pyplot as plt

# plt.scatter(vessel_x_list, vessel_y_list, c='r')
# plt.scatter(nuclei_x_list, nuclei_y_list, c='b')
plt.hist(nuclei_distance_list, bins=100)
plt.show()
