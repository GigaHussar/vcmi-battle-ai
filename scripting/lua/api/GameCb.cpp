/*
 * GameCb.cpp, part of VCMI engine
 *
 * Authors: listed in file AUTHORS in main folder
 *
 * License: GNU General Public License v2.0 or later
 * Full text of license available in license.txt file, in main folder
 *
 */
#include "StdInc.h"

#include "GameCb.h"

#include <vcmi/Player.h>

#include "../LuaCallWrapper.h"

#include "../../../lib/callback/IGameInfoCallback.h"
#include "../../../lib/mapObjects/CGHeroInstance.h"

VCMI_LIB_NAMESPACE_BEGIN

namespace scripting
{
namespace api
{

VCMI_REGISTER_CORE_SCRIPT_API(GameCbProxy, "Game");

const std::vector<GameCbProxy::CustomRegType> GameCbProxy::REGISTER_CUSTOM =
{
	{"getDate", LuaMethodWrapper<GameCb, decltype(&GameCb::getDate), &GameCb::getDate>::invoke, false},

	{"getHero", LuaMethodWrapper<GameCb, decltype(&GameCb::getHero), &GameCb::getHero>::invoke, false},

	{"getObj", LuaMethodWrapper<GameCb, decltype(&GameCb::getObj), &GameCb::getObj>::invoke, false},
};

}
}

VCMI_LIB_NAMESPACE_END
