/*
 * BattleObstacleController.cpp, part of VCMI engine
 *
 * Authors: listed in file AUTHORS in main folder
 *
 * License: GNU General Public License v2.0 or later
 * Full text of license available in license.txt file, in main folder
 *
 */
#include "StdInc.h"
#include "BattleObstacleController.h"

#include "BattleInterface.h"
#include "BattleFieldController.h"
#include "BattleAnimationClasses.h"
#include "BattleStacksController.h"
#include "BattleRenderer.h"
#include "CreatureAnimation.h"

#include "../CPlayerInterface.h"
#include "../GameEngine.h"
#include "../media/ISoundPlayer.h"
#include "../render/CAnimation.h"
#include "../render/Canvas.h"
#include "../render/IRenderHandler.h"

#include "../../lib/battle/CObstacleInstance.h"
#include "../../lib/ObstacleHandler.h"
#include "../../lib/battle/CPlayerBattleCallback.h"
#include "../../lib/serializer/JsonDeserializer.h"

BattleObstacleController::BattleObstacleController(BattleInterface & owner):
	owner(owner),
	timePassed(0.f)
{
	auto obst = owner.getBattle()->battleGetAllObstacles();
	for(auto & elem : obst)
	{
		if ( elem->obstacleType == CObstacleInstance::MOAT )
			continue; // handled by siege controller;
		loadObstacleImage(*elem);
	}
}

void BattleObstacleController::loadObstacleImage(const CObstacleInstance & oi)
{
	AnimationPath animationName = oi.getAnimation();

	if (oi.obstacleType == CObstacleInstance::ABSOLUTE_OBSTACLE)
	{
		// obstacle uses single bitmap image for animations
		obstacleImages[oi.uniqueID] = ENGINE->renderHandler().loadImage(animationName.toType<EResType::IMAGE>(), EImageBlitMode::SIMPLE);
	}
	else
	{
		obstacleAnimations[oi.uniqueID] = ENGINE->renderHandler().loadAnimation(animationName, EImageBlitMode::SIMPLE);
		obstacleImages[oi.uniqueID] = obstacleAnimations[oi.uniqueID]->getImage(0);
	}
}

void BattleObstacleController::obstacleRemoved(const std::vector<ObstacleChanges> & obstacles)
{
	for(const auto & oi : obstacles)
	{
		auto & obstacle = oi.data["obstacle"];

		if (!obstacle.isStruct())
		{
			logGlobal->error("I don't know how to animate removal of this obstacle");
			continue;
		}

		AnimationPath animationPath;
		JsonDeserializer ser(nullptr, obstacle);
		ser.serializeStruct("appearAnimation", animationPath);

		if(animationPath.empty())
			continue;

		auto animation = ENGINE->renderHandler().loadAnimation(animationPath, EImageBlitMode::SIMPLE);
		auto first = animation->getImage(0, 0);
		if(!first)
			continue;

		//we assume here that effect graphics have the same size as the usual obstacle image
		// -> if we know how to blit obstacle, let's blit the effect in the same place
		Point whereTo = getObstaclePosition(first, obstacle);
		//AFAIK, in H3 there is no sound of obstacle removal
		owner.stacksController->addNewAnim(new EffectAnimation(owner, animationPath, whereTo, obstacle["position"].Integer(), 0, true));

		obstacleAnimations.erase(oi.id);
		obstacleImages.erase(oi.id);
		//so when multiple obstacles are removed, they show up one after another
		owner.waitForAnimations();
	}
}

void BattleObstacleController::obstaclePlaced(const std::vector<std::shared_ptr<const CObstacleInstance>> & obstacles)
{
	for(const auto & oi : obstacles)
	{
		auto side = owner.getBattle()->playerToSide(owner.curInt->playerID);

		if(!oi->visibleForSide(side, owner.getBattle()->battleHasNativeStack(side)))
			continue;

		auto animation = ENGINE->renderHandler().loadAnimation(oi->getAppearAnimation(), EImageBlitMode::SIMPLE);
		auto first = animation->getImage(0, 0);
		if(!first)
			continue;

		//we assume here that effect graphics have the same size as the usual obstacle image
		// -> if we know how to blit obstacle, let's blit the effect in the same place
		Point whereTo = getObstaclePosition(first, *oi);
		ENGINE->sound().playSound( oi->getAppearSound() );
		owner.stacksController->addNewAnim(new EffectAnimation(owner, oi->getAppearAnimation(), whereTo, oi->pos));

		//so when multiple obstacles are added, they show up one after another
		owner.waitForAnimations();

		loadObstacleImage(*oi);
	}
}

void BattleObstacleController::showAbsoluteObstacles(Canvas & canvas)
{
	//Blit absolute obstacles
	for(auto & obstacle : owner.getBattle()->battleGetAllObstacles())
	{
		if(obstacle->obstacleType == CObstacleInstance::ABSOLUTE_OBSTACLE)
		{
			auto img = getObstacleImage(*obstacle);
			if(img)
				canvas.draw(img, Point(obstacle->getInfo().width, obstacle->getInfo().height));
		}

		if (obstacle->obstacleType == CObstacleInstance::USUAL)
		{
			if (obstacle->getInfo().isForegroundObstacle)
				continue;

			auto img = getObstacleImage(*obstacle);
			if(img)
			{
				Point p = getObstaclePosition(img, *obstacle);
				canvas.draw(img, p);
			}
		}
	}
}

void BattleObstacleController::collectRenderableObjects(BattleRenderer & renderer)
{
	for (auto obstacle : owner.getBattle()->battleGetAllObstacles())
	{
		if (obstacle->obstacleType == CObstacleInstance::ABSOLUTE_OBSTACLE)
			continue;

		if (obstacle->obstacleType == CObstacleInstance::MOAT)
			continue;

		if (obstacle->obstacleType == CObstacleInstance::USUAL && !obstacle->getInfo().isForegroundObstacle)
			continue;

		renderer.insert(EBattleFieldLayer::OBSTACLES, obstacle->pos, [this, obstacle]( BattleRenderer::RendererRef canvas ){
			auto img = getObstacleImage(*obstacle);
			if(img)
			{
				Point p = getObstaclePosition(img, *obstacle);
				canvas.draw(img, p);
			}
		});
	}
}

void BattleObstacleController::tick(uint32_t msPassed)
{
	timePassed += msPassed / 1000.f;
	int framesCount = timePassed * AnimationControls::getObstaclesSpeed();

	for(auto & animation : obstacleAnimations)
	{
		int frameIndex = framesCount % animation.second->size(0);
		obstacleImages[animation.first] = animation.second->getImage(frameIndex, 0);
	}
}

std::shared_ptr<IImage> BattleObstacleController::getObstacleImage(const CObstacleInstance & oi)
{
	// obstacle is not loaded yet, don't show anything
	if (obstacleImages.count(oi.uniqueID) == 0)
		return nullptr;

	return obstacleImages[oi.uniqueID];
}

Point BattleObstacleController::getObstaclePosition(std::shared_ptr<IImage> image, const CObstacleInstance & obstacle)
{
	int offset = obstacle.getAnimationYOffset(image->height());

	Rect r = owner.fieldController->hexPositionLocal(obstacle.pos);
	r.y += 42 - image->height() + offset;

	return r.topLeft();
}

Point BattleObstacleController::getObstaclePosition(std::shared_ptr<IImage> image, const JsonNode & obstacle)
{
	auto animationYOffset = obstacle["animationYOffset"].Integer();
	auto offset = image->height() % 42;

	if(obstacle["needAnimationOffsetFix"].Bool() && offset > 37)
		animationYOffset -= 42;

	offset += animationYOffset;

	Rect r = owner.fieldController->hexPositionLocal(obstacle["position"].Integer());
	r.y += 42 - image->height() + offset;

	return r.topLeft();
}
