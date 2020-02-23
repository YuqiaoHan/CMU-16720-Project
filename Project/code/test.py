import numpy as np

test_points = [[197,65],
[168,62],
[167,88],
[163,113],
[164,141],
[166,169],
[167,197],
[167,224],
[167,257],
[172,283],
[172,310],
[172,323],
[214,320],
[231,285],
[254,260],
[270,257],
[274,280],
[276,293],
[270,307],
[263,328],
[245,345],
[228,357],
[214,384],
[214,400],
[206,420],
[222,429],
[245,419],
[268,411],
[286,391],
[304,375],
[317,388],
[334,398],
[356,390],
[379,386],
[396,373],
[402,346],
[427,353],
[449,357],
[465,349],
[474,331],
[483,321],
[491,309],
[481,282],
[500,270],
[522,261],
[536,248],
[546,230],
[542,218],
[558,206],
[579,195],
[592,186],
[600,172],
[587,158],
[589,141],
[594,126],
[583,116],
[566,108],
[548,96],
[563,80],
[567,66],
[552,58],
[529,57],
[512,57],
[495,54],
[485,41],
[474,37],
[459,44],
[448,47],
[441,37],
[435,24],
[425,17],
[413,23],
[401,30],
[384,34],
[369,36],
[355,41],
[344,36],
[318,41],
[299,50],
[264,65],
[276,49],
[246,71],
[235,82],
[218,96],
[206,77],
[333,55],
[371,65],
[402,71],
[356,59],
[388,67],
[421,78],
[437,88],
[448,95],
[459,109],
[468,123],
[476,143],
[479,164],
[477,179],
[471,207],
[461,218],
[459,238],
[450,254],
[434,280],
[416,296],
[393,302],
[388,326],
[365,331],
[343,349],
[320,363],
[501,203],
[518,218],
[534,230],
[344,373],
[451,279],
[496,136],
[516,131],
[538,142],
[561,151],
[530,120],
[475,55],
[457,62],
[444,77],
[427,56],
[539,103],
[300,112],
[278,108],
[260,115],
[239,115],
[222,110],
[226,193],
[235,176],
[259,165],
[293,166],
[312,167],
[327,167],
[336,148],
[356,135],
[376,121],
[308,201],
[293,226],
[289,246],
[215,128],
[212,151],
[208,179],
[201,204],
[249,397],
[309,84],
[324,72],
[487,207]
]
np.savez("../results/test_points.npz", pts = test_points)