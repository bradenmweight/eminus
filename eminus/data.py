#!/usr/bin/env python3
"""Atomic data collections."""

# Map atom symbols and atom numbers
SYMBOL2NUMBER = {
    'X': 0,
    'H': 1,
    'He': 2,
    'Li': 3,
    'Be': 4,
    'B': 5,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
    'Ne': 10,
    'Na': 11,
    'Mg': 12,
    'Al': 13,
    'Si': 14,
    'P': 15,
    'S': 16,
    'Cl': 17,
    'Ar': 18,
    'K': 19,
    'Ca': 20,
    'Sc': 21,
    'Ti': 22,
    'V': 23,
    'Cr': 24,
    'Mn': 25,
    'Fe': 26,
    'Co': 27,
    'Ni': 28,
    'Cu': 29,
    'Zn': 30,
    'Ga': 31,
    'Ge': 32,
    'As': 33,
    'Se': 34,
    'Br': 35,
    'Kr': 36,
    'Rb': 37,
    'Sr': 38,
    'Y': 39,
    'Zr': 40,
    'Nb': 41,
    'Mo': 42,
    'Tc': 43,
    'Ru': 44,
    'Rh': 45,
    'Pd': 46,
    'Ag': 47,
    'Cd': 48,
    'In': 49,
    'Sn': 50,
    'Sb': 51,
    'Te': 52,
    'I': 53,
    'Xe': 54,
    'Cs': 55,
    'Ba': 56,
    'La': 57,
    'Ce': 58,
    'Pr': 59,
    'Nd': 60,
    'Pm': 61,
    'Sm': 62,
    'Eu': 63,
    'Gd': 64,
    'Tb': 65,
    'Dy': 66,
    'Ho': 67,
    'Er': 68,
    'Tm': 69,
    'Yb': 70,
    'Lu': 71,
    'Hf': 72,
    'Ta': 73,
    'W': 74,
    'Re': 75,
    'Os': 76,
    'Ir': 77,
    'Pt': 78,
    'Au': 79,
    'Hg': 80,
    'Tl': 81,
    'Pb': 82,
    'Bi': 83,
    'Po': 84,
    'At': 85,
    'Rn': 86,
    # 'Fr': 87,
    # 'Ra': 88,
    'Ac': 89,
    'Th': 90,
    'Pa': 91,
    'U': 92,
    'Np': 93,
    'Pu': 94,
    'Am': 95,
    'Cm': 96,
    'Bk': 97,
    'Cf': 98,
    'Es': 99,
    'Fm': 100,
    'Md': 101,
    'No': 102,
    'Lr': 103
}

# Map atom numbers and atom symbols
NUMBER2SYMBOL = {value: key for key, value in SYMBOL2NUMBER.items()}

# Adapted from https://gitlab.com/ase/ase/-/blob/master/ase/data/__init__.py
COVALENT_RADII = {
    'X': 0.3779452249251541,
    'H': 0.5858150986339887,
    'He': 0.5291233148952157,
    'Li': 2.418849439520986,
    'Be': 1.8141370796407394,
    'B': 1.5873699446856468,
    'C': 1.4361918547155854,
    'N': 1.3417055484842968,
    'O': 1.2472192422530084,
    'F': 1.0771438910366888,
    'Ne': 1.0960411522829467,
    'Na': 3.1369453668787783,
    'Mg': 2.6645138357223357,
    'Al': 2.286568610797182,
    'Si': 2.0975959983346053,
    'P': 2.022006953349574,
    'S': 1.9842124308570588,
    'Cl': 1.9275206471182857,
    'Ar': 2.0031096921033167,
    'K': 3.836144032990313,
    'Ca': 3.3259179793413556,
    'Sc': 3.2125344118638095,
    'Ti': 3.0235617994012327,
    'V': 2.8912809706774283,
    'Cr': 2.6267193132298203,
    'Mn': 2.6267193132298203,
    'Fe': 2.494438484506017,
    'Co': 2.3810549170284707,
    'Ni': 2.343260394535955,
    'Cu': 2.494438484506017,
    'Zn': 2.3054658720434396,
    'Ga': 2.3054658720434396,
    'Ge': 2.267671349550924,
    'As': 2.2487740883046663,
    'Se': 2.267671349550924,
    'Br': 2.267671349550924,
    'Kr': 2.1920823045658935,
    'Rb': 4.157397474176695,
    'Sr': 3.6849659430202517,
    'Y': 3.590479636788963,
    'Zr': 3.3070207180950977,
    'Nb': 3.099150844386263,
    'Mo': 2.910178231923686,
    'Tc': 2.777897403199882,
    'Ru': 2.7590001419536243,
    'Rh': 2.6834110969685936,
    'Pd': 2.6267193132298203,
    'Ag': 2.740102880707367,
    'Cd': 2.721205619461109,
    'In': 2.6834110969685936,
    'Sn': 2.6267193132298203,
    'Sb': 2.6267193132298203,
    'Te': 2.607822051983563,
    'I': 2.6267193132298203,
    'Xe': 2.6456165744760782,
    'Cs': 4.610931744086879,
    'Ba': 4.062911167945406,
    'La': 3.911733077975344,
    'Ce': 3.8550412942365715,
    'Pr': 3.836144032990313,
    'Nd': 3.798349510497798,
    'Pm': 3.760554988005283,
    'Sm': 3.741657726759025,
    'Eu': 3.741657726759025,
    'Gd': 3.7038632042665096,
    'Tb': 3.6660686817739943,
    'Dy': 3.628274159281479,
    'Ho': 3.628274159281479,
    'Er': 3.5715823755427056,
    'Tm': 3.590479636788963,
    'Yb': 3.5337878530501907,
    'Lu': 3.5337878530501907,
    'Hf': 3.3070207180950977,
    'Ta': 3.2125344118638095,
    'W': 3.061356321893748,
    'Re': 2.853486448184913,
    'Os': 2.721205619461109,
    'Ir': 2.6645138357223357,
    'Pt': 2.5700275294910475,
    'Au': 2.5700275294910475,
    'Hg': 2.494438484506017,
    'Tl': 2.740102880707367,
    'Pb': 2.7590001419536243,
    'Bi': 2.79679466444614,
    'Po': 2.6456165744760782,
    'At': 2.8345891869386555,
    'Rn': 2.8345891869386555,
    'Ac': 4.913287924027003,
    'Th': 4.176294735422952,
    'Pa': 4.062911167945406,
    'U': 3.892835816729087,
    'Np': 3.7794522492515403,
    'Pu': 3.7038632042665096,
    'Am': 3.590479636788963,
    'Cm': 3.5337878530501907,
    'Bk': 3.4015070243263863,
    'Cf': 3.1936371506175516,
    'Es': 0.3779452249251541,
    'Fm': 0.3779452249251541,
    'Md': 0.3779452249251541,
    'No': 0.3779452249251541,
    'Lr': 0.3779452249251541,
}

# Adapted from https://gitlab.com/ase/ase/-/blob/master/ase/data/colors.py
CPK_COLORS = {
    'X': '#ff0000',
    'H': '#ffffff',
    'He': '#ffc0ca',
    'Li': '#b12121',
    'Be': '#ff1392',
    'B': '#00ff00',
    'C': '#c7c7c7',
    'N': '#8f8fff',
    'O': '#ef0000',
    'F': '#daa41f',
    'Ne': '#ff1392',
    'Na': '#0000ff',
    'Mg': '#218a21',
    'Al': '#808090',
    'Si': '#daa41f',
    'P': '#ffa400',
    'S': '#ffc731',
    'Cl': '#00ff00',
    'Ar': '#ff1392',
    'K': '#ff1392',
    'Ca': '#808090',
    'Sc': '#ff1392',
    'Ti': '#808090',
    'V': '#ff1392',
    'Cr': '#808090',
    'Mn': '#808090',
    'Fe': '#ffa400',
    'Co': '#ff1392',
    'Ni': '#a42a2a',
    'Cu': '#a42a2a',
    'Zn': '#a42a2a',
    'Ga': '#ff1392',
    'Ge': '#ff1392',
    'As': '#ff1392',
    'Se': '#ff1392',
    'Br': '#a42a2a',
    'Kr': '#ff1392',
    'Rb': '#ff1392',
    'Sr': '#ff1392',
    'Y': '#ff1392',
    'Zr': '#ff1392',
    'Nb': '#ff1392',
    'Mo': '#ff1392',
    'Tc': '#ff1392',
    'Ru': '#ff1392',
    'Rh': '#ff1392',
    'Pd': '#ff1392',
    'Ag': '#808090',
    'Cd': '#ff1392',
    'In': '#ff1392',
    'Sn': '#ff1392',
    'Sb': '#ff1392',
    'Te': '#ff1392',
    'I': '#9f1fef',
    'Xe': '#ff1392',
    'Cs': '#ff1392',
    'Ba': '#ffa400',
    'La': '#ff1392',
    'Ce': '#ff1392',
    'Pr': '#ff1392',
    'Nd': '#ff1392',
    'Pm': '#ff1392',
    'Sm': '#ff1392',
    'Eu': '#ff1392',
    'Gd': '#ff1392',
    'Tb': '#ff1392',
    'Dy': '#ff1392',
    'Ho': '#ff1392',
    'Er': '#ff1392',
    'Tm': '#ff1392',
    'Yb': '#ff1392',
    'Lu': '#ff1392',
    'Hf': '#ff1392',
    'Ta': '#ff1392',
    'W': '#ff1392',
    'Re': '#ff1392',
    'Os': '#ff1392',
    'Ir': '#ff1392',
    'Pt': '#ff1392',
    'Au': '#daa41f',
    'Hg': '#ff1392',
    'Tl': '#ff1392',
    'Pb': '#ff1392',
    'Bi': '#ff1392',
    'Po': '#ff1392',
    'At': '#ff1392',
    'Rn': '#ffffff',
    'Ac': '#ffffff',
    'Th': '#ffffff',
    'Pa': '#ffffff',
    'U': '#ff1392',
    'Np': '#ffffff',
    'Pu': '#ff1392',
    'Am': '#ffffff',
    'Cm': '#ffffff',
    'Bk': '#ffffff',
    'Cf': '#ffffff',
    'Es': '#ffffff',
    'Fm': '#ffffff',
    'Md': '#ffffff',
    'No': '#ffffff',
    'Lr': '#ffffff'
}
