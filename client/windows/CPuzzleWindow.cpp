/*
 * CPuzzleWindow.cpp, part of VCMI engine
 *
 * Authors: listed in file AUTHORS in main folder
 *
 * License: GNU General Public License v2.0 or later
 * Full text of license available in license.txt file, in main folder
 *
 */
#include "StdInc.h"
#include "CPuzzleWindow.h"

#include "../CPlayerInterface.h"
#include "../adventureMap/CResDataBar.h"
#include "../GameEngine.h"
#include "../GameInstance.h"
#include "../gui/TextAlignment.h"
#include "../gui/Shortcut.h"
#include "../mapView/MapView.h"
#include "../media/ISoundPlayer.h"
#include "../widgets/Buttons.h"
#include "../widgets/Images.h"
#include "../widgets/TextControls.h"

#include "../../lib/callback/CCallback.h"
#include "../../lib/entities/faction/CFaction.h"
#include "../../lib/entities/faction/CTownHandler.h"
#include "../../lib/texts/CGeneralTextHandler.h"
#include "../../lib/StartInfo.h"
#include "../../lib/GameLibrary.h"


CPuzzleWindow::CPuzzleWindow(const int3 & GrailPos, double discoveredRatio)
	: CWindowObject(PLAYER_COLORED | BORDERED, ImagePath::builtin("PUZZLE")),
	grailPos(GrailPos),
	currentAlpha(ColorRGBA::ALPHA_OPAQUE)
{
	OBJECT_CONSTRUCTION;

	ENGINE->sound().playSound(soundBase::OBELISK);

	quitb = std::make_shared<CButton>(Point(670, 538), AnimationPath::builtin("IOK6432.DEF"), CButton::tooltip(LIBRARY->generaltexth->allTexts[599]), std::bind(&CPuzzleWindow::close, this), EShortcut::GLOBAL_RETURN);
	quitb->setBorderColor(Colors::METALLIC_GOLD);

	mapView = std::make_shared<PuzzleMapView>(Point(8,9), Point(591, 544), grailPos);
	mapView->needFullUpdate = true;

	logo = std::make_shared<CPicture>(ImagePath::builtin("PUZZLOGO"), 607, 3);
	title = std::make_shared<CLabel>(700, 95, FONT_BIG, ETextAlignment::CENTER, Colors::YELLOW, LIBRARY->generaltexth->allTexts[463]);
	resDataBar = std::make_shared<CResDataBar>(ImagePath::builtin("ARESBAR.bmp"), 3, 575, 32, 2, 85, 85);

	FactionID faction = GAME->interface()->cb->getStartInfo()->playerInfos.find(GAME->interface()->playerID)->second.castle;

	auto & puzzleMap = faction.toFaction()->puzzleMap;

	for(auto & elem : puzzleMap)
	{
		const SPuzzleInfo & info = elem;

		auto piece = std::make_shared<CPicture>(info.filename, info.position.x, info.position.y);
		piece->needRefresh = true;

		//piece that will slowly disappear
		if(info.whenUncovered <= GameConstants::PUZZLE_MAP_PIECES * discoveredRatio)
		{
			piecesToRemove.push_back(piece);
			piece->recActions = piece->recActions & ~SHOWALL;
		}
		else
		{
			visiblePieces.push_back(piece);
		}
	}
}

void CPuzzleWindow::showAll(Canvas & to)
{
	CWindowObject::showAll(to);
}

void CPuzzleWindow::show(Canvas & to)
{
	constexpr int animSpeed = 2;

	if(currentAlpha < animSpeed)
	{
		piecesToRemove.clear();
	}
	else
	{
		//update disappearing puzzles
		for(auto & piece : piecesToRemove)
			piece->setAlpha(currentAlpha);
		currentAlpha -= animSpeed;
	}
	CWindowObject::show(to);

	if(mapView->needFullUpdate && piecesToRemove.empty())
		mapView->needFullUpdate = false;
}
