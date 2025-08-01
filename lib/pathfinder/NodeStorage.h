/*
 * NodeStorage.h, part of VCMI engine
 *
 * Authors: listed in file AUTHORS in main folder
 *
 * License: GNU General Public License v2.0 or later
 * Full text of license available in license.txt file, in main folder
 *
 */
#pragma once

#include "INodeStorage.h"
#include "CGPathNode.h"

VCMI_LIB_NAMESPACE_BEGIN

class DLL_LINKAGE NodeStorage : public INodeStorage
{
private:
	CPathsInfo & out;

	STRONG_INLINE
	void resetTile(const int3 & tile, const EPathfindingLayer & layer, EPathAccessibility accessibility);

public:
	NodeStorage(CPathsInfo & pathsInfo, const CGHeroInstance * hero);

	STRONG_INLINE
	CGPathNode * getNode(const int3 & coord, const EPathfindingLayer layer)
	{
		return out.getNode(coord, layer);
	}

	void initialize(const PathfinderOptions & options, const IGameInfoCallback & gameInfo) override;
	virtual ~NodeStorage() = default;

	std::vector<CGPathNode *> getInitialNodes() override;

	virtual void calculateNeighbours(
		std::vector<CGPathNode *> & result,
		const PathNodeInfo & source,
		EPathfindingLayer layer,
		const PathfinderConfig * pathfinderConfig,
		const CPathfinderHelper * pathfinderHelper) override;

	virtual std::vector<CGPathNode *> calculateTeleportations(
		const PathNodeInfo & source,
		const PathfinderConfig * pathfinderConfig,
		const CPathfinderHelper * pathfinderHelper) override;

	void commit(CDestinationNodeInfo & destination, const PathNodeInfo & source) override;
};

VCMI_LIB_NAMESPACE_END
