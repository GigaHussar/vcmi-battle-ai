/*
 * QuestInfo.cpp, part of VCMI engine
 *
 * Authors: listed in file AUTHORS in main folder
 *
 * License: GNU General Public License v2.0 or later
 * Full text of license available in license.txt file, in main folder
 *
 */
#include "StdInc.h"
#include "QuestInfo.h"

#include "../callback/IGameInfoCallback.h"
#include "../mapObjects/CQuest.h"

VCMI_LIB_NAMESPACE_BEGIN

const CQuest * QuestInfo::getQuest(IGameInfoCallback *cb) const
{
	auto questObject = dynamic_cast<const IQuestObject*>(getObject(cb));
	assert(questObject);

	return &questObject->getQuest();
}

const CGObjectInstance * QuestInfo::getObject(IGameInfoCallback *cb) const
{
	return cb->getObjInstance(obj);
}

int3 QuestInfo::getPosition(IGameInfoCallback *cb) const
{
	return getObject(cb)->visitablePos();
}

VCMI_LIB_NAMESPACE_END
