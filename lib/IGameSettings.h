/*
 * IIGameSettings.h, part of VCMI engine
 *
 * Authors: listed in file AUTHORS in main folder
 *
 * License: GNU General Public License v2.0 or later
 * Full text of license available in license.txt file, in main folder
 *
 */
#pragma once

VCMI_LIB_NAMESPACE_BEGIN

class JsonNode;

enum class EGameSettings
{
	BANKS_SHOW_GUARDS_COMPOSITION,
	BONUSES_GLOBAL,
	BONUSES_PER_HERO,
	COMBAT_ABILITY_BIAS,
	COMBAT_AREA_SHOT_CAN_TARGET_EMPTY_HEX,
	COMBAT_ATTACK_POINT_DAMAGE_FACTOR,
	COMBAT_ATTACK_POINT_DAMAGE_FACTOR_CAP,
	COMBAT_DEFENSE_POINT_DAMAGE_FACTOR,
	COMBAT_DEFENSE_POINT_DAMAGE_FACTOR_CAP,
	COMBAT_GOOD_MORALE_CHANCE, 
	COMBAT_BAD_MORALE_CHANCE, 
	COMBAT_MORALE_DICE_SIZE,
	COMBAT_MORALE_BIAS,
	COMBAT_GOOD_LUCK_CHANCE,
	COMBAT_BAD_LUCK_CHANCE,
	COMBAT_LUCK_DICE_SIZE,
	COMBAT_LUCK_BIAS,
	COMBAT_LAYOUTS,
	COMBAT_ONE_HEX_TRIGGERS_OBSTACLES,
	CREATURES_ALLOW_ALL_FOR_DOUBLE_MONTH,
	CREATURES_ALLOW_RANDOM_SPECIAL_WEEKS,
	CREATURES_DAILY_STACK_EXPERIENCE,
	CREATURES_ALLOW_JOINING_FOR_FREE,
	CREATURES_JOINING_PERCENTAGE,
	CREATURES_WEEKLY_GROWTH_CAP,
	CREATURES_WEEKLY_GROWTH_PERCENT,
	DIMENSION_DOOR_EXPOSES_TERRAIN_TYPE,
	DIMENSION_DOOR_FAILURE_SPENDS_POINTS,
	DIMENSION_DOOR_ONLY_TO_UNCOVERED_TILES,
	DIMENSION_DOOR_TOURNAMENT_RULES_LIMIT,
	DIMENSION_DOOR_TRIGGERS_GUARDS,
	DWELLINGS_ACCUMULATE_WHEN_NEUTRAL,
	DWELLINGS_ACCUMULATE_WHEN_OWNED,
	DWELLINGS_MERGE_ON_RECRUIT,
	HEROES_BACKPACK_CAP,
	HEROES_MINIMAL_PRIMARY_SKILLS,
	HEROES_PER_PLAYER_ON_MAP_CAP,
	HEROES_PER_PLAYER_TOTAL_CAP,
	HEROES_RETREAT_ON_WIN_WITHOUT_TROOPS,
	HEROES_STARTING_STACKS_CHANCES,
	HEROES_TAVERN_INVITE,
	HEROES_MOVEMENT_COST_BASE,
	HEROES_MOVEMENT_POINTS_LAND,
	HEROES_MOVEMENT_POINTS_SEA,
	MAP_FORMAT_ARMAGEDDONS_BLADE,
	MAP_FORMAT_CHRONICLES,
	MAP_FORMAT_HORN_OF_THE_ABYSS,
	MAP_FORMAT_IN_THE_WAKE_OF_GODS,
	MAP_FORMAT_JSON_VCMI,
	MAP_FORMAT_RESTORATION_OF_ERATHIA,
	MAP_FORMAT_SHADOW_OF_DEATH,
	MAP_OBJECTS_H3_BUG_QUEST_TAKES_ENTIRE_ARMY,
	MARKETS_BLACK_MARKET_RESTOCK_PERIOD,
	MODULE_COMMANDERS,
	MODULE_STACK_ARTIFACT,
	MODULE_STACK_EXPERIENCE,
	PATHFINDER_IGNORE_GUARDS,
	PATHFINDER_ORIGINAL_FLY_RULES,
	PATHFINDER_USE_BOAT,
	PATHFINDER_USE_MONOLITH_ONE_WAY_RANDOM,
	PATHFINDER_USE_MONOLITH_ONE_WAY_UNIQUE,
	PATHFINDER_USE_MONOLITH_TWO_WAY,
	PATHFINDER_USE_WHIRLPOOL,
	RESOURCES_WEEKLY_BONUSES_AI,
	TEXTS_ARTIFACT,
	TEXTS_CREATURE,
	TEXTS_FACTION,
	TEXTS_HERO,
	TEXTS_HERO_CLASS,
	TEXTS_OBJECT,
	TEXTS_RIVER,
	TEXTS_ROAD,
	TEXTS_SPELL,
	TEXTS_TERRAIN,
	TOWNS_BUILDINGS_PER_TURN_CAP,
	TOWNS_STARTING_DWELLING_CHANCES,
	INTERFACE_PLAYER_COLORED_BACKGROUND,
	TOWNS_SPELL_RESEARCH,
	TOWNS_SPELL_RESEARCH_COST,
	TOWNS_SPELL_RESEARCH_PER_DAY,
	TOWNS_SPELL_RESEARCH_COST_EXPONENT_PER_RESEARCH,

	OPTIONS_COUNT,
	OPTIONS_BEGIN = BONUSES_GLOBAL
};

class DLL_LINKAGE IGameSettings
{
public:
	virtual JsonNode getFullConfig() const = 0;
	virtual const JsonNode & getValue(EGameSettings option) const = 0;
	virtual ~IGameSettings() = default;

	bool getBoolean(EGameSettings option) const;
	int64_t getInteger(EGameSettings option) const;
	double getDouble(EGameSettings option) const;
	std::vector<int> getVector(EGameSettings option) const;
	int getVectorValue(EGameSettings option, size_t index) const;
};

VCMI_LIB_NAMESPACE_END
