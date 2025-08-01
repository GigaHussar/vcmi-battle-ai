/*
* BuildingManager.cpp, part of VCMI engine
*
* Authors: listed in file AUTHORS in main folder
*
* License: GNU General Public License v2.0 or later
* Full text of license available in license.txt file, in main folder
*
*/

#include "StdInc.h"
#include "BuildingManager.h"

#include "../../lib/mapObjects/MapObjects.h"
#include "../../lib/entities/building/CBuilding.h"

bool BuildingManager::tryBuildThisStructure(const CGTownInstance * t, BuildingID building, unsigned int maxDays)
{
	if (maxDays == 0)
	{
		logAi->warn("Request to build building %d in 0 days!", building.toEnum());
		return false;
	}

	if (!vstd::contains(t->getTown()->buildings, building))
		return false; // no such building in town

	if (t->hasBuilt(building)) //Already built? Shouldn't happen in general
		return true;

	const auto & buildPtr = t->getTown()->buildings.at(building);

	auto toBuild = buildPtr->requirements.getFulfillmentCandidates([&](const BuildingID & buildID)
	{
		return t->hasBuilt(buildID);
	});
	toBuild.push_back(building);

	for (BuildingID buildID : toBuild)
	{
		EBuildingState canBuild = cb->canBuildStructure(t, buildID);
		if (canBuild == EBuildingState::HAVE_CAPITAL || canBuild == EBuildingState::FORBIDDEN || canBuild == EBuildingState::NO_WATER)
			return false; //we won't be able to build this
	}

	if (maxDays && toBuild.size() > maxDays)
		return false;

	//TODO: calculate if we have enough resources to build it in maxDays?

	for (const auto & buildID : toBuild)
	{
		const auto & b = t->getTown()->buildings.at(buildID);

		EBuildingState canBuild = cb->canBuildStructure(t, buildID);
		if (canBuild == EBuildingState::ALLOWED)
		{
			PotentialBuilding pb;
			pb.bid = buildID;
			pb.price = t->getBuildingCost(buildID);
			immediateBuildings.push_back(pb); //these are checked again in try
			return true;
		}
		else if (canBuild == EBuildingState::PREREQUIRES)
		{
			// can happen when dependencies have their own missing dependencies
			if (tryBuildThisStructure(t, buildID, maxDays - 1))
				return true;
		}
		else if (canBuild == EBuildingState::MISSING_BASE)
		{
			if (tryBuildThisStructure(t, b->upgrade, maxDays - 1))
				return true;
		}
		else if (canBuild == EBuildingState::NO_RESOURCES)
		{
			//we may need to gather resources for those
			PotentialBuilding pb;
			pb.bid = buildID;
			pb.price = t->getBuildingCost(buildID);
			expensiveBuildings.push_back(pb); //these are checked again in try
			return false;
		}
		else
			return false;
	}
	return false;
}

bool BuildingManager::tryBuildAnyStructure(const CGTownInstance * t, std::vector<BuildingID> buildList, unsigned int maxDays)
{
	for (const auto & building : buildList)
	{
		if (t->hasBuilt(building))
			continue;
		return tryBuildThisStructure(t, building, maxDays);

	}
	return false; //Can't build anything
}

std::optional<BuildingID> BuildingManager::canBuildAnyStructure(const CGTownInstance * t, const std::vector<BuildingID> & buildList, unsigned int maxDays) const
{
	for (const auto & building : buildList)
	{
		if (t->hasBuilt(building))
			continue;
		switch (cb->canBuildStructure(t, building))
		{
			case EBuildingState::ALLOWED:
			case EBuildingState::NO_RESOURCES: //TODO: allow this via optional parameter?
				return std::optional<BuildingID>(building);
				break;
		}
	}
	return std::optional<BuildingID>(); //Can't build anything
}

bool BuildingManager::tryBuildNextStructure(const CGTownInstance * t, std::vector<BuildingID> buildList, unsigned int maxDays)
{
	for (const auto & building : buildList)
	{
		if (t->hasBuilt(building))
			continue;
		return tryBuildThisStructure(t, building, maxDays);
	}
	return false; //Nothing to build
}

void BuildingManager::init(CPlayerSpecificInfoCallback * CB)
{
	cb = CB;
}

void BuildingManager::setAI(VCAI * AI)
{
	ai = AI;
}
//Set of buildings for different goals. Does not include any prerequisites.
static const std::vector<BuildingID> essential = { BuildingID::TAVERN, BuildingID::TOWN_HALL };
static const std::vector<BuildingID> basicGoldSource = { BuildingID::TOWN_HALL, BuildingID::CITY_HALL };
static const std::vector<BuildingID> defence = { BuildingID::FORT, BuildingID::CITADEL, BuildingID::CASTLE };
static const std::vector<BuildingID> capitolAndRequirements = { BuildingID::FORT, BuildingID::CITADEL, BuildingID::CASTLE, BuildingID::CAPITOL };
static const std::vector<BuildingID> unitsSource = { BuildingID::DWELL_LVL_1, BuildingID::DWELL_LVL_2, BuildingID::DWELL_LVL_3,
BuildingID::DWELL_LVL_4, BuildingID::DWELL_LVL_5, BuildingID::DWELL_LVL_6, BuildingID::DWELL_LVL_7, BuildingID::DWELL_LVL_8 };
static const std::vector<BuildingID> unitsUpgrade = { BuildingID::DWELL_LVL_1_UP, BuildingID::DWELL_LVL_2_UP, BuildingID::DWELL_LVL_3_UP,
BuildingID::DWELL_LVL_4_UP, BuildingID::DWELL_LVL_5_UP, BuildingID::DWELL_LVL_6_UP, BuildingID::DWELL_LVL_7_UP, BuildingID::DWELL_LVL_8_UP };
static const std::vector<BuildingID> unitGrowth = { BuildingID::HORDE_1, BuildingID::HORDE_1_UPGR, BuildingID::HORDE_2, BuildingID::HORDE_2_UPGR };
static const std::vector<BuildingID> _spells = { BuildingID::MAGES_GUILD_1, BuildingID::MAGES_GUILD_2, BuildingID::MAGES_GUILD_3,
BuildingID::MAGES_GUILD_4, BuildingID::MAGES_GUILD_5 };
static const std::vector<BuildingID> extra = { BuildingID::MARKETPLACE, BuildingID::BLACKSMITH, BuildingID::RESOURCE_SILO, BuildingID::SPECIAL_1, BuildingID::SPECIAL_2,
BuildingID::SPECIAL_3, BuildingID::SPECIAL_4, BuildingID::SHIPYARD }; // all remaining buildings

bool BuildingManager::getBuildingOptions(const CGTownInstance * t)
{
	//TODO make *real* town development system
	//TODO: faction-specific development: use special buildings, build dwellings in better order, etc
	//TODO: build resource silo, defences when needed
	//Possible - allow "locking" on specific building (build prerequisites and then building itself)
	
	//TODO: There is some disabled building code in GatherTroops and GatherArmy - take it into account when enhancing building. For now AI works best with building only via Build goal.

	immediateBuildings.clear();
	expensiveBuildings.clear();

	//below algorithm focuses on economy growth at start of the game, saving money instead of build rushing is handled by Build goal
	//changing code blocks order will alter behavior by changing order of adding elements to immediateBuildings / expensiveBuildings

	// TResources currentRes = cb->getResourceAmount();
	// TResources currentIncome = t->dailyIncome();

	if(tryBuildAnyStructure(t, essential))
		return true;

	if (cb->getDate(Date::DAY_OF_WEEK) < 5) // first 4 days of week - try to focus on dwellings
	{
		if (tryBuildNextStructure(t, unitsSource, 4))
			return true;
	}

	if (cb->getDate(Date::DAY_OF_WEEK) > 4) // last 3 days of week - try to focus on growth by building Fort/Citadel/Castle
	{
		if (tryBuildNextStructure(t, defence, 3))
			return true;
	}

	if (t->hasBuilt(BuildingID::CASTLE))
	{
		if (tryBuildAnyStructure(t, unitGrowth))
			return true;
	}

	//try to make City Hall
	if (tryBuildNextStructure(t, basicGoldSource))
		return true;

	//workaround for mantis #2696 - build capitol with separate algorithm if it is available
	if(t->hasBuilt(BuildingID::CITY_HALL) && getMaxPossibleGoldBuilding(t) == BuildingID::CAPITOL)
	{
		if(tryBuildNextStructure(t, capitolAndRequirements))
			return true;
	}

	//try to upgrade dwelling
	for (int i = 0; i < unitsUpgrade.size(); i++)
	{
		if (t->hasBuilt(unitsSource[i]) && !t->hasBuilt(unitsUpgrade[i]) && t->hasBuilt(BuildingID::FORT))
		{
			if (tryBuildThisStructure(t, unitsUpgrade[i]))
				return true;
		}
	}

	//remaining tasks
	if (tryBuildNextStructure(t, _spells))
		return true;
	if (tryBuildAnyStructure(t, extra))
		return true;

	//at the end, try to get and build any extra buildings with nonstandard slots (for example HotA 3rd level dwelling)
	std::vector<BuildingID> extraBuildings;
	for (const auto & buildingInfo : t->getTown()->buildings)
	{
		if (buildingInfo.first.isDwelling() && BuildingID::getUpgradedFromDwelling(buildingInfo.first) > 1)
			extraBuildings.push_back(buildingInfo.first);
	}
	return tryBuildAnyStructure(t, extraBuildings);
}

BuildingID BuildingManager::getMaxPossibleGoldBuilding(const CGTownInstance * t)
{
	if(cb->canBuildStructure(t, BuildingID::CAPITOL) != EBuildingState::HAVE_CAPITAL && cb->canBuildStructure(t, BuildingID::CAPITOL) != EBuildingState::FORBIDDEN)
		return BuildingID::CAPITOL;
	else if(cb->canBuildStructure(t, BuildingID::CITY_HALL) != EBuildingState::FORBIDDEN)
		return BuildingID::CITY_HALL;
	else if(cb->canBuildStructure(t, BuildingID::TOWN_HALL) != EBuildingState::FORBIDDEN)
		return BuildingID::TOWN_HALL;
	else
		return BuildingID::VILLAGE_HALL;
}

std::optional<PotentialBuilding> BuildingManager::immediateBuilding() const
{
	if (immediateBuildings.size())
		return std::optional<PotentialBuilding>(immediateBuildings.front()); //back? whatever
	else
		return std::optional<PotentialBuilding>();
}

std::optional<PotentialBuilding> BuildingManager::expensiveBuilding() const
{
	if (expensiveBuildings.size())
		return std::optional<PotentialBuilding>(expensiveBuildings.front());
	else
		return std::optional<PotentialBuilding>();
}
