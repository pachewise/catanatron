from catanatron.models.map import Resource
from catanatron.models.board import BuildingType


def number_probability(number):
    return {
        2: 2.778,
        3: 5.556,
        4: 8.333,
        5: 11.111,
        6: 13.889,
        7: 16.667,
        8: 13.889,
        9: 11.111,
        10: 8.333,
        11: 5.556,
        12: 2.778,
    }[number] / 100


def public_victory_points(game, player):
    return player.public_victory_points


def build_resource_production_feature(resource):
    def production_feature(game, player):
        production = 0
        settlements = game.board.get_player_buildings(
            player.color, building_type=BuildingType.SETTLEMENT
        )
        for (node, settlement) in settlements:
            tiles = game.board.get_adjacent_tiles(node)
            production += sum(
                [number_probability(t.number) for t in tiles if t.resource == resource]
            )
        cities = game.board.get_player_buildings(
            player.color, building_type=BuildingType.CITY
        )
        for (node, city) in cities:
            tiles = game.board.get_adjacent_tiles(node)
            production += sum(
                [
                    number_probability(t.number) * 2
                    for t in tiles
                    if t.resource == resource
                ]
            )
        return production

    production_feature.__name__ = resource.value.lower() + "_production"
    return production_feature


features = [
    public_victory_points,
    build_resource_production_feature(Resource.WOOD),
    build_resource_production_feature(Resource.BRICK),
    build_resource_production_feature(Resource.SHEEP),
    build_resource_production_feature(Resource.WHEAT),
    build_resource_production_feature(Resource.ORE),
]


def create_sample(game, p0, p1, p2, p3):
    record = {}
    for i, p in enumerate([p0, p1, p2, p3]):
        for feature in features:
            feature_name = feature.__name__ + "_p" + str(i)
            record[feature_name] = feature(game, p)
    return record
