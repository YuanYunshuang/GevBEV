
LABEL_COLORS_carla = {
    'Unlabeled': (0, 0, 0),  # 0 Unlabeled
    'Buildings': (70, 70, 70),  # 1 Buildings
    'Fences': (100, 40, 40),  # 2 Fences
    'Other': (55, 90, 80),  # 3 Other
    'Pedestrians': (220, 20, 60),  # 4 Pedestrians
    'Poles': (153, 153, 153),  # 5 Poles
    'RoadLines': (157, 234, 50),  # 6 RoadLines
    'Roads': (128, 64, 128),  # 7 Roads
    'Sidewalks': (244, 35, 232),  # 8 Sidewalks
    'Vegetation': (107, 142, 35),  # 9 Vegetation
    'Vehicles': (0, 0, 142),  # 10 Vehicles
    'Walls': (102, 102, 156),  # 11 Walls
    'TrafficSign': (220, 220, 0),  # 12 TrafficSign
    'Sky': (70, 130, 180),  # 13 Sky
    'Ground': (81, 0, 81),  # 14 Ground
    'Bridge': (150, 100, 100),  # 15 Bridge
    'Railtrack': (230, 150, 140),  # 16 Railtrack
    'GuardRail': (180, 165, 180),  # 17 GuardRail
    'TrafficLight': (250, 170, 30),  # 18 TrafficLight
    'Static': (110, 190, 160),  # 19 Static
    'Dynamic': (170, 120, 50),  # 20 Dynamic
    'Water': (45, 60, 150),  # 21 Water
    'Terrain': (145, 170, 100)  # 22 Terrain
}

VALID_CLS_carla = [
    [10],  # 1 vehicles
    [6, 7], # 2 road
    [8],  # 3 sidewalks
    [1, 11],  # 4 buildings, walls
    [2],  # 5 fences
    [5],  # 6 poles
    [9],  # 7 vegetation
    [14],  # 8 ground
    [16],  # 9 guardrail
]

VALID_CLS_nuscenes = [
    [24],  # 1 drivable surface
    [17, 19, 20],  # 2 car
    [15, 16],  # 3 bus
    [18],  # 4 construction_vehicle
    [21],  # 5 motorcycle
    [14],  # 6 bicycle
    [22],  # 7 trailer
    [23],  # 8 truck
    [2, 3, 4, 5, 6, 7, 8],  # 9 pedestrian
    [12],  # 10 traffic_cone
    [25],  # 11 other_flat
    [26],  # 12 sidewalk
    [27],  # 13 terrain
    [28],  # 14 manmade
    [30],  # 15 vegetation
    [9],  # 16 barrier
]