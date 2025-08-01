/*
 * CGPathNode.cpp, part of VCMI engine
 *
 * Authors: listed in file AUTHORS in main folder
 *
 * License: GNU General Public License v2.0 or later
 * Full text of license available in license.txt file, in main folder
 *
 */
#include "StdInc.h"
#include "CGPathNode.h"

#include "CPathfinder.h"

#include "../callback/IGameInfoCallback.h"
#include "../mapObjects/CGHeroInstance.h"
#include "../mapping/CMapDefines.h"

VCMI_LIB_NAMESPACE_BEGIN

static bool canSeeObj(const CGObjectInstance * obj)
{
	/// Pathfinder should ignore placed events
	return obj != nullptr && obj->ID != Obj::EVENT;
}

const CGPathNode & CGPath::currNode() const
{
	assert(nodes.size() > 1);
	return nodes[nodes.size()-1];
}

const CGPathNode & CGPath::nextNode() const
{
	assert(nodes.size() > 1);
	return nodes[nodes.size()-2];
}

const CGPathNode & CGPath::lastNode() const
{
	assert(nodes.size() > 1);
	return nodes[0];
}

int3 CGPath::startPos() const
{
	return nodes[nodes.size()-1].coord;
}

int3 CGPath::endPos() const
{
	return nodes[0].coord;
}

CPathsInfo::CPathsInfo(const int3 & Sizes, const CGHeroInstance * hero_)
	: sizes(Sizes), hero(hero_)
{
	nodes.resize(boost::extents[ELayer::NUM_LAYERS][sizes.z][sizes.x][sizes.y]);
	heroBonusTreeVersion = hero->getTreeVersion();
}

CPathsInfo::~CPathsInfo() = default;

const CGPathNode * CPathsInfo::getPathInfo(const int3 & tile) const
{
	assert(vstd::iswithin(tile.x, 0, sizes.x));
	assert(vstd::iswithin(tile.y, 0, sizes.y));
	assert(vstd::iswithin(tile.z, 0, sizes.z));

	return getNode(tile);
}

bool CPathsInfo::getPath(CGPath & out, const int3 & dst) const
{
	out.nodes.clear();
	const CGPathNode * curnode = getNode(dst);
	if(!curnode->theNodeBefore)
		return false;

	while(curnode)
	{
		const CGPathNode cpn = * curnode;
		curnode = curnode->theNodeBefore;
		out.nodes.push_back(cpn);
	}
	return true;
}

const CGPathNode * CPathsInfo::getNode(const int3 & coord) const
{
	const auto * landNode = &nodes[ELayer::LAND][coord.z][coord.x][coord.y];
	if(landNode->reachable())
		return landNode;
	else
		return &nodes[ELayer::SAIL][coord.z][coord.x][coord.y];
}

PathNodeInfo::PathNodeInfo()
	: node(nullptr), nodeObject(nullptr), tile(nullptr), coord(-1, -1, -1), guarded(false),	isInitialPosition(false)
{
}

void PathNodeInfo::setNode(const IGameInfoCallback & gameInfo, CGPathNode * n)
{
	node = n;
	guarded = false;

	if(coord != node->coord)
	{
		assert(node->coord.isValid());

		coord = node->coord;
		tile = gameInfo.getTile(coord);
		nodeObject = nullptr;
		nodeHero = nullptr;

		ObjectInstanceID topObjectID = tile->topVisitableObj();
		if (topObjectID.hasValue())
		{
			nodeObject = gameInfo.getObjInstance(topObjectID);

			if (nodeObject->ID == Obj::HERO)
			{
				nodeHero = dynamic_cast<const CGHeroInstance *>(nodeObject);
				ObjectInstanceID bottomObjectID = tile->topVisitableObj(true);

				if (bottomObjectID.hasValue())
					nodeObject = gameInfo.getObjInstance(bottomObjectID);
			}
		}
	}

}

void PathNodeInfo::updateInfo(CPathfinderHelper * hlp, const IGameInfoCallback & gameInfo)
{
	if(gameInfo.guardingCreaturePosition(node->coord).isValid() && !isInitialPosition)
	{
		guarded = true;
	}

	if(nodeObject)
	{
		objectRelations = gameInfo.getPlayerRelations(hlp->owner, nodeObject->tempOwner);
	}

	if(nodeHero)
	{
		heroRelations = gameInfo.getPlayerRelations(hlp->owner, nodeHero->tempOwner);
	}
}

bool PathNodeInfo::isNodeObjectVisitable() const
{
	/// Hero can't visit objects while walking on water or flying
	return (node->layer == EPathfindingLayer::LAND || node->layer == EPathfindingLayer::SAIL)
		&& (canSeeObj(nodeObject) || canSeeObj(nodeHero));
}


CDestinationNodeInfo::CDestinationNodeInfo():
	blocked(false),
	action(EPathNodeAction::UNKNOWN)
{
}

void CDestinationNodeInfo::setNode(const IGameInfoCallback & gameInfo, CGPathNode * n)
{
	PathNodeInfo::setNode(gameInfo, n);

	blocked = false;
	action = EPathNodeAction::UNKNOWN;
}

bool CDestinationNodeInfo::isBetterWay() const
{
	if(node->turns == 0xff) //we haven't been here before
		return true;
	else
		return cost < node->getCost(); //this route is faster
}

VCMI_LIB_NAMESPACE_END
