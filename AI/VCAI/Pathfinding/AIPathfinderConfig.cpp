/*
* AIPathfinderConfig.cpp, part of VCMI engine
*
* Authors: listed in file AUTHORS in main folder
*
* License: GNU General Public License v2.0 or later
* Full text of license available in license.txt file, in main folder
*
*/
#include "StdInc.h"
#include "AIPathfinderConfig.h"
#include "Rules/AILayerTransitionRule.h"
#include "Rules/AIMovementAfterDestinationRule.h"
#include "Rules/AIMovementToDestinationRule.h"
#include "Rules/AIPreviousNodeRule.h"

#include "../../../lib/pathfinder/CPathfinder.h"

namespace AIPathfinding
{
	std::vector<std::shared_ptr<IPathfindingRule>> makeRuleset(
		CPlayerSpecificInfoCallback * cb,
		VCAI * ai,
		std::shared_ptr<AINodeStorage> nodeStorage)
	{
		std::vector<std::shared_ptr<IPathfindingRule>> rules = {
			std::make_shared<AILayerTransitionRule>(cb, ai, nodeStorage),
			std::make_shared<DestinationActionRule>(),
			std::make_shared<AIMovementToDestinationRule>(nodeStorage),
			std::make_shared<MovementCostRule>(),
			std::make_shared<AIPreviousNodeRule>(nodeStorage),
			std::make_shared<AIMovementAfterDestinationRule>(cb, nodeStorage)
		};

		return rules;
	}

	AIPathfinderConfig::AIPathfinderConfig(
		CPlayerSpecificInfoCallback * cb,
		VCAI * ai,
		std::shared_ptr<AINodeStorage> nodeStorage)
		:PathfinderConfig(nodeStorage, *cb, makeRuleset(cb, ai, nodeStorage)), hero(nodeStorage->getHero())
	{
		options.ignoreGuards = false;
		options.useEmbarkAndDisembark = true;
		options.useTeleportTwoWay = true;
		options.useTeleportOneWay = true;
		options.useTeleportOneWayRandom = true;
		options.useTeleportWhirlpool = true;
	}

	AIPathfinderConfig::~AIPathfinderConfig() = default;

	CPathfinderHelper * AIPathfinderConfig::getOrCreatePathfinderHelper(const PathNodeInfo & source, const IGameInfoCallback & gameInfo)
	{
		if(!helper)
		{
			helper.reset(new CPathfinderHelper(gameInfo, hero, options));
		}

		return helper.get();
	}
}
