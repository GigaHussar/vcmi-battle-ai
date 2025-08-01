/*
 * BattleActionsController.cpp, part of VCMI engine
 *
 * Authors: listed in file AUTHORS in main folder
 *
 * License: GNU General Public License v2.0 or later
 * Full text of license available in license.txt file, in main folder
 *
 */
#include <fstream>
#include "../../external/json/json.hpp"

#include "StdInc.h"
#include "BattleActionsController.h"
BattleActionsController* GLOBAL_SOCKET_ACTION_CONTROLLER = nullptr;
#include "../../lib/battle/BattleInfo.h"

#include "BattleWindow.h"
#include "BattleStacksController.h"
#include "BattleInterface.h"
#include "BattleFieldController.h"
#include "BattleSiegeController.h"
#include "BattleInterfaceClasses.h"

#include "../CPlayerInterface.h"
#include "../gui/CursorHandler.h"
#include "../GameEngine.h"
#include "../GameInstance.h"
#include "../gui/CIntObject.h"
#include "../gui/WindowHandler.h"
#include "../windows/CCreatureWindow.h"
#include "../windows/InfoWindows.h"

#include "../../lib/CConfigHandler.h"
#include "../../lib/GameLibrary.h"
#include "../../lib/texts/CGeneralTextHandler.h"
#include "../../lib/CRandomGenerator.h"
#include "../../lib/CStack.h"
#include "../../lib/battle/BattleAction.h"
#include "../../lib/battle/CPlayerBattleCallback.h"
#include "../../lib/callback/CCallback.h"
#include "../../lib/spells/CSpellHandler.h"
#include "../../lib/spells/ISpellMechanics.h"
#include "../../lib/spells/Problem.h"

struct TextReplacement
{
	std::string placeholder;
	std::string replacement;
};

using TextReplacementList = std::vector<TextReplacement>;

static std::string replacePlaceholders(std::string input, const TextReplacementList & format )
{
	for(const auto & entry : format)
		boost::replace_all(input, entry.placeholder, entry.replacement);

	return input;
}

static std::string translatePlural(int amount, const std::string& baseTextID)
{
	if(amount == 1)
		return LIBRARY->generaltexth->translate(baseTextID + ".1");
	return LIBRARY->generaltexth->translate(baseTextID);
}

static std::string formatPluralImpl(int amount, const std::string & amountString, const std::string & baseTextID)
{
	std::string baseString = translatePlural(amount, baseTextID);
	TextReplacementList replacements {
		{ "%d", amountString }
	};

	return replacePlaceholders(baseString, replacements);
}

static std::string formatPlural(int amount, const std::string & baseTextID)
{
	return formatPluralImpl(amount, std::to_string(amount), baseTextID);
}

static std::string formatPlural(DamageRange range, const std::string & baseTextID)
{
	if (range.min == range.max)
		return formatPlural(range.min, baseTextID);

	std::string rangeString = std::to_string(range.min) + " - " + std::to_string(range.max);

	return formatPluralImpl(range.max, rangeString, baseTextID);
}

static std::string formatAttack(const DamageEstimation & estimation, const std::string & creatureName, const std::string & baseTextID, int shotsLeft)
{
	TextReplacementList replacements = {
		{ "%CREATURE", creatureName },
		{ "%DAMAGE", formatPlural(estimation.damage, "vcmi.battleWindow.damageEstimation.damage") },
		{ "%SHOTS", formatPlural(shotsLeft, "vcmi.battleWindow.damageEstimation.shots") },
		{ "%KILLS", formatPlural(estimation.kills, "vcmi.battleWindow.damageEstimation.kills") },
	};

	return replacePlaceholders(LIBRARY->generaltexth->translate(baseTextID), replacements);
}

static std::string formatMeleeAttack(const DamageEstimation & estimation, const std::string & creatureName)
{
	std::string baseTextID = estimation.kills.max == 0 ?
		"vcmi.battleWindow.damageEstimation.melee" :
		"vcmi.battleWindow.damageEstimation.meleeKills";

	return formatAttack(estimation, creatureName, baseTextID, 0);
}

static std::string formatRangedAttack(const DamageEstimation & estimation, const std::string & creatureName, int shotsLeft)
{
	std::string baseTextID = estimation.kills.max == 0 ?
		"vcmi.battleWindow.damageEstimation.ranged" :
		"vcmi.battleWindow.damageEstimation.rangedKills";

	return formatAttack(estimation, creatureName, baseTextID, shotsLeft);
}

static std::string formatRetaliation(const DamageEstimation & estimation, bool mayBeKilled)
{
	if (estimation.damage.max == 0)
		return LIBRARY->generaltexth->translate("vcmi.battleWindow.damageRetaliation.never");

	std::string baseTextID = estimation.kills.max == 0 ?
								 "vcmi.battleWindow.damageRetaliation.damage" :
								 "vcmi.battleWindow.damageRetaliation.damageKills";

	std::string prefixTextID = mayBeKilled ?
		"vcmi.battleWindow.damageRetaliation.may" :
		"vcmi.battleWindow.damageRetaliation.will";

	return LIBRARY->generaltexth->translate(prefixTextID) + formatAttack(estimation, "", baseTextID, 0);
}

BattleActionsController::BattleActionsController(BattleInterface & owner):
	owner(owner),
	selectedStack(nullptr),
	heroSpellToCast(nullptr)
{
	GLOBAL_SOCKET_ACTION_CONTROLLER = this;
}

void BattleActionsController::endCastingSpell()
{
	if(heroSpellToCast)
	{
		heroSpellToCast.reset();
		owner.windowObject->blockUI(false);
	}

	if(owner.stacksController->getActiveStack())
		possibleActions = getPossibleActionsForStack(owner.stacksController->getActiveStack()); //restore actions after they were cleared

	selectedStack = nullptr;
	ENGINE->fakeMouseMove();
}

bool BattleActionsController::isActiveStackSpellcaster() const
{
	const CStack * casterStack = owner.stacksController->getActiveStack();
	if (!casterStack)
		return false;

	bool spellcaster = casterStack->hasBonusOfType(BonusType::SPELLCASTER);
	return (spellcaster && casterStack->canCast());
}

void BattleActionsController::enterCreatureCastingMode()
{
	//silently check for possible errors
	if (owner.tacticsMode)
		return;

	//hero is casting a spell
	if (heroSpellToCast)
		return;

	if (!owner.stacksController->getActiveStack())
		return;

	if(owner.getBattle()->battleCanTargetEmptyHex(owner.stacksController->getActiveStack()))
	{
		auto actionFilterPredicate = [](const PossiblePlayerBattleAction x)
		{
			return x.get() != PossiblePlayerBattleAction::SHOOT;
		};

		vstd::erase_if(possibleActions, actionFilterPredicate);
		ENGINE->fakeMouseMove();
		return;
	}

	if (!isActiveStackSpellcaster())
		return;

	for(const auto & action : possibleActions)
	{
		if (action.get() != PossiblePlayerBattleAction::NO_LOCATION)
			continue;

		const spells::Caster * caster = owner.stacksController->getActiveStack();
		const CSpell * spell = action.spell().toSpell();

		spells::Target target;
		target.emplace_back();

		spells::BattleCast cast(owner.getBattle().get(), caster, spells::Mode::CREATURE_ACTIVE, spell);

		auto m = spell->battleMechanics(&cast);
		spells::detail::ProblemImpl ignored;

		const bool isCastingPossible = m->canBeCastAt(target, ignored);

		if (isCastingPossible)
		{
			owner.giveCommand(EActionType::MONSTER_SPELL, BattleHex::INVALID, spell->getId());
			selectedStack = nullptr;

			ENGINE->cursor().set(Cursor::Combat::POINTER);
		}
		return;
	}

	possibleActions = getPossibleActionsForStack(owner.stacksController->getActiveStack());

	auto actionFilterPredicate = [](const PossiblePlayerBattleAction x)
	{
		return !x.spellcast();
	};

	vstd::erase_if(possibleActions, actionFilterPredicate);
	ENGINE->fakeMouseMove();
}

std::vector<PossiblePlayerBattleAction> BattleActionsController::getPossibleActionsForStack(const CStack *stack) const
{
	BattleClientInterfaceData data; //hard to get rid of these things so for now they're required data to pass

	for(const auto & spell : creatureSpells)
		data.creatureSpellsToCast.push_back(spell->id);

	data.tacticsMode = owner.tacticsMode;
	auto allActions = owner.getBattle()->getClientActionsForStack(stack, data);

	allActions.push_back(PossiblePlayerBattleAction::HERO_INFO);
	allActions.push_back(PossiblePlayerBattleAction::CREATURE_INFO);

	return std::vector<PossiblePlayerBattleAction>(allActions);
}

void BattleActionsController::reorderPossibleActionsPriority(const CStack * stack, const CStack * targetStack)
{
	if(owner.tacticsMode || possibleActions.empty()) return; //this function is not supposed to be called in tactics mode or before getPossibleActionsForStack

	auto assignPriority = [&](const PossiblePlayerBattleAction & item
						  ) -> uint8_t //large lambda assigning priority which would have to be part of possibleActions without it
	{
		switch(item.get())
		{
			case PossiblePlayerBattleAction::AIMED_SPELL_CREATURE:
			case PossiblePlayerBattleAction::ANY_LOCATION:
			case PossiblePlayerBattleAction::NO_LOCATION:
			case PossiblePlayerBattleAction::FREE_LOCATION:
			case PossiblePlayerBattleAction::OBSTACLE:
				if(!stack->hasBonusOfType(BonusType::NO_SPELLCAST_BY_DEFAULT) && targetStack != nullptr)
				{
					PlayerColor stackOwner = owner.getBattle()->battleGetOwner(targetStack);
					bool enemyTargetingPositiveSpellcast = item.spell().toSpell()->isPositive() && stackOwner != owner.curInt->playerID;
					bool friendTargetingNegativeSpellcast = item.spell().toSpell()->isNegative() && stackOwner == owner.curInt->playerID;

					if(!enemyTargetingPositiveSpellcast && !friendTargetingNegativeSpellcast)
						return 1;
				}
				return 100; //bottom priority

				break;
			case PossiblePlayerBattleAction::RANDOM_GENIE_SPELL:
				return 2;
				break;
			case PossiblePlayerBattleAction::SHOOT:
				if(targetStack == nullptr || targetStack->unitSide() == stack->unitSide() || !targetStack->alive())
					return 100; //bottom priority

				return 4;
				break;
			case PossiblePlayerBattleAction::ATTACK_AND_RETURN:
				return 5;
				break;
			case PossiblePlayerBattleAction::ATTACK:
				return 6;
				break;
			case PossiblePlayerBattleAction::WALK_AND_ATTACK:
				return 7;
				break;
			case PossiblePlayerBattleAction::MOVE_STACK:
				return 8;
				break;
			case PossiblePlayerBattleAction::CATAPULT:
				return 9;
				break;
			case PossiblePlayerBattleAction::HEAL:
				return 10;
				break;
			case PossiblePlayerBattleAction::CREATURE_INFO:
				return 11;
				break;
			case PossiblePlayerBattleAction::HERO_INFO:
				return 12;
				break;
			case PossiblePlayerBattleAction::TELEPORT:
				return 13;
				break;
			default:
				assert(0);
				return 200;
				break;
		}
	};

	auto comparer = [&](const PossiblePlayerBattleAction & lhs, const PossiblePlayerBattleAction & rhs)
	{
		return assignPriority(lhs) < assignPriority(rhs);
	};

	std::sort(possibleActions.begin(), possibleActions.end(), comparer);
}

void BattleActionsController::castThisSpell(SpellID spellID)
{
	heroSpellToCast = std::make_shared<BattleAction>();
	heroSpellToCast->actionType = EActionType::HERO_SPELL;
	heroSpellToCast->spell = spellID;
	heroSpellToCast->stackNumber = -1;
	heroSpellToCast->side = owner.curInt->cb->getBattle(owner.getBattleID())->battleGetMySide();

	//choosing possible targets
	const CGHeroInstance *castingHero = (owner.attackingHeroInstance->tempOwner == owner.curInt->playerID) ? owner.attackingHeroInstance : owner.defendingHeroInstance;
	assert(castingHero); // code below assumes non-null hero
	PossiblePlayerBattleAction spellSelMode = owner.getBattle()->getCasterAction(spellID.toSpell(), castingHero, spells::Mode::HERO);

	if (spellSelMode.get() == PossiblePlayerBattleAction::NO_LOCATION) //user does not have to select location
	{
		heroSpellToCast->aimToHex(BattleHex::INVALID);
		owner.curInt->cb->battleMakeSpellAction(owner.getBattleID(), *heroSpellToCast);
		endCastingSpell();
	}
	else
	{
		possibleActions.clear();
		possibleActions.push_back (spellSelMode); //only this one action can be performed at the moment
		ENGINE->fakeMouseMove();//update cursor
	}

	owner.windowObject->blockUI(true);
}

const CSpell * BattleActionsController::getHeroSpellToCast( ) const
{
	if (heroSpellToCast)
		return heroSpellToCast->spell.toSpell();
	return nullptr;
}

const CSpell * BattleActionsController::getStackSpellToCast(const BattleHex & hoveredHex)
{
	if (heroSpellToCast)
		return nullptr;

	if (!owner.stacksController->getActiveStack())
		return nullptr;

	if (!hoveredHex.isValid())
		return nullptr;

	auto action = selectAction(hoveredHex);

	if(owner.stacksController->getActiveStack()->hasBonusOfType(BonusType::SPELL_LIKE_ATTACK))
	{
		auto bonus = owner.stacksController->getActiveStack()->getBonus(Selector::type()(BonusType::SPELL_LIKE_ATTACK));
		return bonus->subtype.as<SpellID>().toSpell();
	}

	if (action.spell() == SpellID::NONE)
		return nullptr;

	return action.spell().toSpell();
}

const CSpell * BattleActionsController::getCurrentSpell(const BattleHex & hoveredHex)
{
	if (getHeroSpellToCast())
		return getHeroSpellToCast();
	return getStackSpellToCast(hoveredHex);
}

const CStack * BattleActionsController::getStackForHex(const BattleHex & hoveredHex)
{
	const CStack * shere = owner.getBattle()->battleGetStackByPos(hoveredHex, true);
	if(shere)
		return shere;
	return owner.getBattle()->battleGetStackByPos(hoveredHex, false);
}

void BattleActionsController::actionSetCursor(PossiblePlayerBattleAction action, const BattleHex & targetHex)
{
	switch (action.get())
	{
		case PossiblePlayerBattleAction::CHOOSE_TACTICS_STACK:
			ENGINE->cursor().set(Cursor::Combat::POINTER);
			return;

		case PossiblePlayerBattleAction::MOVE_TACTICS:
		case PossiblePlayerBattleAction::MOVE_STACK:
			if (owner.stacksController->getActiveStack()->hasBonusOfType(BonusType::FLYING))
				ENGINE->cursor().set(Cursor::Combat::FLY);
			else
				ENGINE->cursor().set(Cursor::Combat::MOVE);
			return;

		case PossiblePlayerBattleAction::ATTACK:
		case PossiblePlayerBattleAction::WALK_AND_ATTACK:
		case PossiblePlayerBattleAction::ATTACK_AND_RETURN:
		{
			static const std::map<BattleHex::EDir, Cursor::Combat> sectorCursor = {
				{BattleHex::TOP_LEFT,     Cursor::Combat::HIT_SOUTHEAST},
				{BattleHex::TOP_RIGHT,    Cursor::Combat::HIT_SOUTHWEST},
				{BattleHex::RIGHT,        Cursor::Combat::HIT_WEST     },
				{BattleHex::BOTTOM_RIGHT, Cursor::Combat::HIT_NORTHWEST},
				{BattleHex::BOTTOM_LEFT,  Cursor::Combat::HIT_NORTHEAST},
				{BattleHex::LEFT,         Cursor::Combat::HIT_EAST     },
				{BattleHex::TOP,          Cursor::Combat::HIT_SOUTH    },
				{BattleHex::BOTTOM,       Cursor::Combat::HIT_NORTH    }
			};

			auto direction = owner.fieldController->selectAttackDirection(targetHex);

			assert(sectorCursor.count(direction) > 0);
			if (sectorCursor.count(direction))
				ENGINE->cursor().set(sectorCursor.at(direction));

			return;
		}

		case PossiblePlayerBattleAction::SHOOT:
			if (owner.getBattle()->battleHasShootingPenalty(owner.stacksController->getActiveStack(), targetHex))
				ENGINE->cursor().set(Cursor::Combat::SHOOT_PENALTY);
			else
				ENGINE->cursor().set(Cursor::Combat::SHOOT);
			return;

		case PossiblePlayerBattleAction::AIMED_SPELL_CREATURE:
		case PossiblePlayerBattleAction::ANY_LOCATION:
		case PossiblePlayerBattleAction::RANDOM_GENIE_SPELL:
		case PossiblePlayerBattleAction::FREE_LOCATION:
		case PossiblePlayerBattleAction::OBSTACLE:
			ENGINE->cursor().set(Cursor::Spellcast::SPELL);
			return;

		case PossiblePlayerBattleAction::TELEPORT:
			ENGINE->cursor().set(Cursor::Combat::TELEPORT);
			return;

		case PossiblePlayerBattleAction::SACRIFICE:
			ENGINE->cursor().set(Cursor::Combat::SACRIFICE);
			return;

		case PossiblePlayerBattleAction::HEAL:
			ENGINE->cursor().set(Cursor::Combat::HEAL);
			return;

		case PossiblePlayerBattleAction::CATAPULT:
			ENGINE->cursor().set(Cursor::Combat::SHOOT_CATAPULT);
			return;

		case PossiblePlayerBattleAction::CREATURE_INFO:
			ENGINE->cursor().set(Cursor::Combat::QUERY);
			return;
		case PossiblePlayerBattleAction::HERO_INFO:
			ENGINE->cursor().set(Cursor::Combat::HERO);
			return;
	}
	assert(0);
}

void BattleActionsController::actionSetCursorBlocked(PossiblePlayerBattleAction action, const BattleHex & targetHex)
{
	switch (action.get())
	{
		case PossiblePlayerBattleAction::AIMED_SPELL_CREATURE:
		case PossiblePlayerBattleAction::RANDOM_GENIE_SPELL:
		case PossiblePlayerBattleAction::TELEPORT:
		case PossiblePlayerBattleAction::SACRIFICE:
		case PossiblePlayerBattleAction::FREE_LOCATION:
			ENGINE->cursor().set(Cursor::Combat::BLOCKED);
			return;
		default:
			if (targetHex == -1)
				ENGINE->cursor().set(Cursor::Combat::POINTER);
			else
				ENGINE->cursor().set(Cursor::Combat::BLOCKED);
			return;
	}
	assert(0);
}

std::string BattleActionsController::actionGetStatusMessage(PossiblePlayerBattleAction action, const BattleHex & targetHex)
{
	const CStack * targetStack = getStackForHex(targetHex);

	switch (action.get()) //display console message, realize selected action
	{
		case PossiblePlayerBattleAction::CHOOSE_TACTICS_STACK:
			return (boost::format(LIBRARY->generaltexth->allTexts[481]) % targetStack->getName()).str(); //Select %s

		case PossiblePlayerBattleAction::MOVE_TACTICS:
		case PossiblePlayerBattleAction::MOVE_STACK:
			if (owner.stacksController->getActiveStack()->hasBonusOfType(BonusType::FLYING))
				return (boost::format(LIBRARY->generaltexth->allTexts[295]) % owner.stacksController->getActiveStack()->getName()).str(); //Fly %s here
			else
				return (boost::format(LIBRARY->generaltexth->allTexts[294]) % owner.stacksController->getActiveStack()->getName()).str(); //Move %s here

		case PossiblePlayerBattleAction::ATTACK:
		case PossiblePlayerBattleAction::WALK_AND_ATTACK:
		case PossiblePlayerBattleAction::ATTACK_AND_RETURN: //TODO: allow to disable return
			{
				const auto * attacker = owner.stacksController->getActiveStack();
				BattleHex attackFromHex = owner.fieldController->fromWhichHexAttack(targetHex);
				int distance = attacker->position.isValid() ? owner.getBattle()->battleGetDistances(attacker, attacker->getPosition())[attackFromHex.toInt()] : 0;
				DamageEstimation retaliation;
				BattleAttackInfo attackInfo(attacker, targetStack, distance, false );
				DamageEstimation estimation = owner.getBattle()->battleEstimateDamage(attackInfo, &retaliation);
				estimation.kills.max = std::min<int64_t>(estimation.kills.max, targetStack->getCount());
				estimation.kills.min = std::min<int64_t>(estimation.kills.min, targetStack->getCount());
				bool enemyMayBeKilled = estimation.kills.max == targetStack->getCount();

				return formatMeleeAttack(estimation, targetStack->getName()) + "\n" + formatRetaliation(retaliation, enemyMayBeKilled);
			}

		case PossiblePlayerBattleAction::SHOOT:
		{
			if(targetStack == nullptr) //should be true only for spell-like attack
			{
				auto spellLikeAttackBonus = owner.stacksController->getActiveStack()->getBonus(Selector::type()(BonusType::SPELL_LIKE_ATTACK));
				assert(spellLikeAttackBonus != nullptr);
				return boost::str(boost::format(LIBRARY->generaltexth->allTexts[26]) % spellLikeAttackBonus->subtype.as<SpellID>().toSpell()->getNameTranslated());
			}

			const auto * shooter = owner.stacksController->getActiveStack();

			DamageEstimation retaliation;
			BattleAttackInfo attackInfo(shooter, targetStack, 0, true );
			DamageEstimation estimation = owner.getBattle()->battleEstimateDamage(attackInfo, &retaliation);
			estimation.kills.max = std::min<int64_t>(estimation.kills.max, targetStack->getCount());
			estimation.kills.min = std::min<int64_t>(estimation.kills.min, targetStack->getCount());
			return formatRangedAttack(estimation, targetStack->getName(), shooter->shots.available());
		}

		case PossiblePlayerBattleAction::AIMED_SPELL_CREATURE:
			return boost::str(boost::format(LIBRARY->generaltexth->allTexts[27]) % action.spell().toSpell()->getNameTranslated() % targetStack->getName()); //Cast %s on %s

		case PossiblePlayerBattleAction::ANY_LOCATION:
			return boost::str(boost::format(LIBRARY->generaltexth->allTexts[26]) % action.spell().toSpell()->getNameTranslated()); //Cast %s

		case PossiblePlayerBattleAction::RANDOM_GENIE_SPELL: //we assume that teleport / sacrifice will never be available as random spell
			return boost::str(boost::format(LIBRARY->generaltexth->allTexts[301]) % targetStack->getName()); //Cast a spell on %

		case PossiblePlayerBattleAction::TELEPORT:
			return LIBRARY->generaltexth->allTexts[25]; //Teleport Here

		case PossiblePlayerBattleAction::OBSTACLE:
			return LIBRARY->generaltexth->allTexts[550];

		case PossiblePlayerBattleAction::SACRIFICE:
			return (boost::format(LIBRARY->generaltexth->allTexts[549]) % targetStack->getName()).str(); //sacrifice the %s

		case PossiblePlayerBattleAction::FREE_LOCATION:
			return boost::str(boost::format(LIBRARY->generaltexth->allTexts[26]) % action.spell().toSpell()->getNameTranslated()); //Cast %s

		case PossiblePlayerBattleAction::HEAL:
			return (boost::format(LIBRARY->generaltexth->allTexts[419]) % targetStack->getName()).str(); //Apply first aid to the %s

		case PossiblePlayerBattleAction::CATAPULT:
			return ""; // TODO

		case PossiblePlayerBattleAction::CREATURE_INFO:
			return (boost::format(LIBRARY->generaltexth->allTexts[297]) % targetStack->getName()).str();

		case PossiblePlayerBattleAction::HERO_INFO:
			return  LIBRARY->generaltexth->translate("core.genrltxt.417"); // "View Hero Stats"
	}
	assert(0);
	return "";
}

std::string BattleActionsController::actionGetStatusMessageBlocked(PossiblePlayerBattleAction action, const BattleHex & targetHex)
{
	switch (action.get())
	{
		case PossiblePlayerBattleAction::AIMED_SPELL_CREATURE:
		case PossiblePlayerBattleAction::RANDOM_GENIE_SPELL:
			return LIBRARY->generaltexth->allTexts[23];
			break;
		case PossiblePlayerBattleAction::TELEPORT:
			return LIBRARY->generaltexth->allTexts[24]; //Invalid Teleport Destination
			break;
		case PossiblePlayerBattleAction::SACRIFICE:
			return LIBRARY->generaltexth->allTexts[543]; //choose army to sacrifice
			break;
		case PossiblePlayerBattleAction::FREE_LOCATION:
			return boost::str(boost::format(LIBRARY->generaltexth->allTexts[181]) % action.spell().toSpell()->getNameTranslated()); //No room to place %s here
			break;
		default:
			return "";
	}
}

bool BattleActionsController::actionIsLegal(PossiblePlayerBattleAction action, const BattleHex & targetHex)
{
	const CStack * targetStack = getStackForHex(targetHex);
	bool targetStackOwned = targetStack && targetStack->unitOwner() == owner.curInt->playerID;

	switch (action.get())
	{
		case PossiblePlayerBattleAction::CHOOSE_TACTICS_STACK:
			return (targetStack && targetStackOwned && targetStack->getMovementRange() > 0);

		case PossiblePlayerBattleAction::CREATURE_INFO:
			return (targetStack && targetStackOwned && targetStack->alive());

		case PossiblePlayerBattleAction::HERO_INFO:
			if (targetHex == BattleHex::HERO_ATTACKER)
				return owner.attackingHero != nullptr;

			if (targetHex == BattleHex::HERO_DEFENDER)
				return owner.defendingHero != nullptr;

			return false;

		case PossiblePlayerBattleAction::MOVE_TACTICS:
		case PossiblePlayerBattleAction::MOVE_STACK:
			if (!(targetStack && targetStack->alive())) //we can walk on dead stacks
			{
				if(canStackMoveHere(owner.stacksController->getActiveStack(), targetHex))
					return true;
			}
			return false;

		case PossiblePlayerBattleAction::ATTACK:
		case PossiblePlayerBattleAction::WALK_AND_ATTACK:
		case PossiblePlayerBattleAction::ATTACK_AND_RETURN:
			{
				if (owner.fieldController->isTileAttackable(targetHex)) // move isTileAttackable to be part of battleCanAttack?
				{
					BattleHex attackFromHex = owner.fieldController->fromWhichHexAttack(targetHex);
					if(owner.getBattle()->battleCanAttack(owner.stacksController->getActiveStack(), targetStack, attackFromHex))
						return true;
				}
				return false;
			}

		case PossiblePlayerBattleAction::SHOOT:
			{
				auto currentStack = owner.stacksController->getActiveStack();
				if(!owner.getBattle()->battleCanShoot(currentStack, targetHex))
					return false;

				if(targetStack == nullptr && owner.getBattle()->battleCanTargetEmptyHex(currentStack))
				{
					auto spellLikeAttackBonus = currentStack->getBonus(Selector::type()(BonusType::SPELL_LIKE_ATTACK));
					const CSpell * spellDataToCheck = spellLikeAttackBonus->subtype.as<SpellID>().toSpell();
					return isCastingPossibleHere(spellDataToCheck, nullptr, targetHex);
				}

				return true;
			}

		case PossiblePlayerBattleAction::NO_LOCATION:
			return false;

		case PossiblePlayerBattleAction::ANY_LOCATION:
			return isCastingPossibleHere(action.spell().toSpell(), nullptr, targetHex);

		case PossiblePlayerBattleAction::AIMED_SPELL_CREATURE:
			return !selectedStack && targetStack && isCastingPossibleHere(action.spell().toSpell(), nullptr, targetHex);

		case PossiblePlayerBattleAction::RANDOM_GENIE_SPELL:
			if(targetStack && targetStackOwned && targetStack != owner.stacksController->getActiveStack() && targetStack->alive()) //only positive spells for other allied creatures
			{
				SpellID spellID = owner.getBattle()->getRandomBeneficialSpell(CRandomGenerator::getDefault(), owner.stacksController->getActiveStack(), targetStack);
				return spellID != SpellID::NONE;
			}
			return false;

		case PossiblePlayerBattleAction::TELEPORT:
			return selectedStack && isCastingPossibleHere(action.spell().toSpell(), selectedStack, targetHex);

		case PossiblePlayerBattleAction::SACRIFICE: //choose our living stack to sacrifice
			return targetStack && targetStack != selectedStack && targetStackOwned && targetStack->alive();

		case PossiblePlayerBattleAction::OBSTACLE:
		case PossiblePlayerBattleAction::FREE_LOCATION:
			return isCastingPossibleHere(action.spell().toSpell(), nullptr, targetHex);

		case PossiblePlayerBattleAction::CATAPULT:
			return owner.siegeController && owner.siegeController->isAttackableByCatapult(targetHex);

		case PossiblePlayerBattleAction::HEAL:
			return targetStack && targetStackOwned && targetStack->canBeHealed();
	}

	assert(0);
	return false;
}

void BattleActionsController::actionRealize(PossiblePlayerBattleAction action, const BattleHex & targetHex)
{
	const CStack * targetStack = getStackForHex(targetHex);

	switch (action.get()) //display console message, realize selected action
	{
		case PossiblePlayerBattleAction::CHOOSE_TACTICS_STACK:
		{
			owner.stackActivated(targetStack);
			return;
		}

		case PossiblePlayerBattleAction::MOVE_TACTICS:
		case PossiblePlayerBattleAction::MOVE_STACK:
		{
			const auto * activeStack = owner.stacksController->getActiveStack();
			const bool backwardsMove = activeStack->unitSide() == BattleSide::ATTACKER ?
				targetHex.getX() < activeStack->getPosition().getX():
				targetHex.getX() > activeStack->getPosition().getX();

			if(activeStack->doubleWide() && backwardsMove)
			{
				BattleHexArray acc = owner.getBattle()->battleGetAvailableHexes(activeStack, false);
				BattleHex shiftedDest = targetHex.cloneInDirection(activeStack->destShiftDir(), false);
				if(acc.contains(shiftedDest))
					owner.giveCommand(EActionType::WALK, shiftedDest);
				else
					owner.giveCommand(EActionType::WALK, targetHex);
			}
			else
			{
				owner.giveCommand(EActionType::WALK, targetHex);
			}
			return;
		}

		case PossiblePlayerBattleAction::ATTACK:
		case PossiblePlayerBattleAction::WALK_AND_ATTACK:
		case PossiblePlayerBattleAction::ATTACK_AND_RETURN: //TODO: allow to disable return
		{
			bool returnAfterAttack = action.get() == PossiblePlayerBattleAction::ATTACK_AND_RETURN;
			BattleHex attackFromHex = owner.fieldController->fromWhichHexAttack(targetHex);
			if(attackFromHex.isValid()) //we can be in this line when unreachable creature is L - clicked (as of revision 1308)
			{
				BattleAction command = BattleAction::makeMeleeAttack(owner.stacksController->getActiveStack(), targetHex, attackFromHex, returnAfterAttack);
				owner.sendCommand(command, owner.stacksController->getActiveStack());
			}
			return;
		}

		case PossiblePlayerBattleAction::SHOOT:
		{
			owner.giveCommand(EActionType::SHOOT, targetHex);
			return;
		}

		case PossiblePlayerBattleAction::HEAL:
		{
			owner.giveCommand(EActionType::STACK_HEAL, targetHex);
			return;
		};

		case PossiblePlayerBattleAction::CATAPULT:
		{
			owner.giveCommand(EActionType::CATAPULT, targetHex);
			return;
		}

		case PossiblePlayerBattleAction::CREATURE_INFO:
		{
			ENGINE->windows().createAndPushWindow<CStackWindow>(targetStack, false);
			return;
		}

		case PossiblePlayerBattleAction::HERO_INFO:
		{
			if (targetHex == BattleHex::HERO_ATTACKER)
				owner.attackingHero->heroLeftClicked();

			if (targetHex == BattleHex::HERO_DEFENDER)
				owner.defendingHero->heroLeftClicked();

			return;
		}

		case PossiblePlayerBattleAction::AIMED_SPELL_CREATURE:
		case PossiblePlayerBattleAction::ANY_LOCATION:
		case PossiblePlayerBattleAction::RANDOM_GENIE_SPELL: //we assume that teleport / sacrifice will never be available as random spell
		case PossiblePlayerBattleAction::TELEPORT:
		case PossiblePlayerBattleAction::OBSTACLE:
		case PossiblePlayerBattleAction::SACRIFICE:
		case PossiblePlayerBattleAction::FREE_LOCATION:
		{
			if (action.get() == PossiblePlayerBattleAction::AIMED_SPELL_CREATURE )
			{
				if (action.spell() == SpellID::SACRIFICE)
				{
					heroSpellToCast->aimToHex(targetHex);
					possibleActions.push_back({PossiblePlayerBattleAction::SACRIFICE, action.spell()});
					selectedStack = targetStack;
					return;
				}
				if (action.spell() == SpellID::TELEPORT)
				{
					heroSpellToCast->aimToUnit(targetStack);
					possibleActions.push_back({PossiblePlayerBattleAction::TELEPORT, action.spell()});
					selectedStack = targetStack;
					return;
				}
			}

			if (!heroSpellcastingModeActive())
			{
				if (action.spell().hasValue())
				{
					owner.giveCommand(EActionType::MONSTER_SPELL, targetHex, action.spell());
				}
				else //unknown random spell
				{
					owner.giveCommand(EActionType::MONSTER_SPELL, targetHex);
				}
			}
			else
			{
				assert(getHeroSpellToCast());
				switch (getHeroSpellToCast()->id.toEnum())
				{
					case SpellID::SACRIFICE:
						heroSpellToCast->aimToUnit(targetStack);//victim
						break;
					default:
						heroSpellToCast->aimToHex(targetHex);
						break;
				}
				owner.curInt->cb->battleMakeSpellAction(owner.getBattleID(), *heroSpellToCast);
				endCastingSpell();
			}
			selectedStack = nullptr;
			return;
		}
	}
	assert(0);
	return;
}

PossiblePlayerBattleAction BattleActionsController::selectAction(const BattleHex & targetHex)
{
	assert(owner.stacksController->getActiveStack() != nullptr);
	assert(!possibleActions.empty());
	assert(targetHex.isValid());

	if (owner.stacksController->getActiveStack() == nullptr)
		return PossiblePlayerBattleAction::INVALID;

	if (possibleActions.empty())
		return PossiblePlayerBattleAction::INVALID;

	const CStack * targetStack = getStackForHex(targetHex);

	reorderPossibleActionsPriority(owner.stacksController->getActiveStack(), targetStack);

	for (PossiblePlayerBattleAction action : possibleActions)
	{
		if (actionIsLegal(action, targetHex))
			return action;
	}
	return possibleActions.front();
}

void BattleActionsController::onHexHovered(const BattleHex & hoveredHex)
{
	if (owner.openingPlaying())
	{
		currentConsoleMsg = LIBRARY->generaltexth->translate("vcmi.battleWindow.pressKeyToSkipIntro");
		ENGINE->statusbar()->write(currentConsoleMsg);
		return;
	}

	if (owner.stacksController->getActiveStack() == nullptr)
		return;

	if (hoveredHex == BattleHex::INVALID)
	{
		if (!currentConsoleMsg.empty())
			ENGINE->statusbar()->clearIfMatching(currentConsoleMsg);

		currentConsoleMsg.clear();
		ENGINE->cursor().set(Cursor::Combat::BLOCKED);
		return;
	}

	auto action = selectAction(hoveredHex);

	std::string newConsoleMsg;

	if (actionIsLegal(action, hoveredHex))
	{
		actionSetCursor(action, hoveredHex);
		newConsoleMsg = actionGetStatusMessage(action, hoveredHex);
	}
	else
	{
		actionSetCursorBlocked(action, hoveredHex);
		newConsoleMsg = actionGetStatusMessageBlocked(action, hoveredHex);
	}

	if (!currentConsoleMsg.empty())
		ENGINE->statusbar()->clearIfMatching(currentConsoleMsg);

	if (!newConsoleMsg.empty())
		ENGINE->statusbar()->write(newConsoleMsg);

	currentConsoleMsg = newConsoleMsg;
}

void BattleActionsController::onHoverEnded()
{
	ENGINE->cursor().set(Cursor::Combat::POINTER);

	if (!currentConsoleMsg.empty())
		ENGINE->statusbar()->clearIfMatching(currentConsoleMsg);

	currentConsoleMsg.clear();
}

void BattleActionsController::onHexLeftClicked(const BattleHex & clickedHex)
{
	if (owner.stacksController->getActiveStack() == nullptr)
		return;

	auto action = selectAction(clickedHex);

	std::string newConsoleMsg;

	if (!actionIsLegal(action, clickedHex))
		return;
	
	actionRealize(action, clickedHex);
	ENGINE->statusbar()->clear();
}

void BattleActionsController::tryActivateStackSpellcasting(const CStack *casterStack)
{
	creatureSpells.clear();

	bool spellcaster = casterStack->hasBonusOfType(BonusType::SPELLCASTER);
	if(casterStack->canCast() && spellcaster)
	{
		// faerie dragon can cast only one, randomly selected spell until their next move
		//TODO: faerie dragon type spell should be selected by server
		const auto spellToCast = owner.getBattle()->getRandomCastedSpell(CRandomGenerator::getDefault(), casterStack);

		if (spellToCast.hasValue())
			creatureSpells.push_back(spellToCast.toSpell());
	}

	TConstBonusListPtr bl = casterStack->getBonusesOfType(BonusType::SPELLCASTER);

	for(const auto & bonus : *bl)
	{
		if (bonus->additionalInfo[0] <= 0 && bonus->subtype.as<SpellID>().hasValue())
			creatureSpells.push_back(bonus->subtype.as<SpellID>().toSpell());
	}
}

const spells::Caster * BattleActionsController::getCurrentSpellcaster() const
{
	if (heroSpellToCast)
		return owner.currentHero();
	else
		return owner.stacksController->getActiveStack();
}

spells::Mode BattleActionsController::getCurrentCastMode() const
{
	if (heroSpellToCast)
		return spells::Mode::HERO;
	else
		return spells::Mode::CREATURE_ACTIVE;

}

bool BattleActionsController::isCastingPossibleHere(const CSpell * currentSpell, const CStack *targetStack, const BattleHex & targetHex)
{
	assert(currentSpell);
	if (!currentSpell)
		return false;

	auto caster = getCurrentSpellcaster();

	const spells::Mode mode = heroSpellToCast ? spells::Mode::HERO : spells::Mode::CREATURE_ACTIVE;

	spells::Target target;
	if(targetStack)
		target.emplace_back(targetStack);
	target.emplace_back(targetHex);

	spells::BattleCast cast(owner.getBattle().get(), caster, mode, currentSpell);

	auto m = currentSpell->battleMechanics(&cast);
	spells::detail::ProblemImpl problem; //todo: display problem in status bar

	return m->canBeCastAt(target, problem);
}

bool BattleActionsController::canStackMoveHere(const CStack * stackToMove, const BattleHex & myNumber) const
{
	BattleHexArray acc = owner.getBattle()->battleGetAvailableHexes(stackToMove, false);
	BattleHex shiftedDest = myNumber.cloneInDirection(stackToMove->destShiftDir(), false);

	if (acc.contains(myNumber))
		return true;
	else if (stackToMove->doubleWide() && acc.contains(shiftedDest))
		return true;
	else
		return false;
}

void BattleActionsController::generateClientExportFileName()
{
	if (!exportFileNameAction.empty())
		return;
	
	using namespace std::chrono;

	auto now = system_clock::now();
	auto now_ns = time_point_cast<nanoseconds>(now);
	auto epoch = now_ns.time_since_epoch();

	auto seconds_since_epoch = duration_cast<seconds>(epoch);
	auto ms_part = duration_cast<milliseconds>(epoch - seconds_since_epoch).count();
	auto ns_part = duration_cast<nanoseconds>(epoch - seconds_since_epoch).count() % 1000000;

	auto t = system_clock::to_time_t(now);
	std::tm tm = *std::localtime(&t);

	std::ostringstream timestampStream;
	timestampStream << std::put_time(&tm, "%Y%m%d_%H%M%S");
	timestampStream << "_" << std::setw(3) << std::setfill('0') << ms_part;
	timestampStream << "_" << std::setw(6) << std::setfill('0') << ns_part;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(0, 0xFFFFFF);
	std::ostringstream randomIdStream;
	randomIdStream << std::hex << std::setw(6) << std::setfill('0') << dis(gen);

	std::string exportIdAction = timestampStream.str() + "_" + randomIdStream.str();
	exportFileNameAction = "battle_" + exportIdAction + "_available_actions.json";
}

void BattleActionsController::exportPossibleActionsToJson(const CStack *stack, const std::vector<PossiblePlayerBattleAction> &actions)
{	
	//generateClientExportFileName();
	// Set up log file path
	std::filesystem::path exportPath = "../../export";
	std::filesystem::create_directories(exportPath);
	std::filesystem::path logFilePath = exportPath / "possible_actions.json";


	// Static turn counter, incremented only when this function is executed
	static int turnCounter = 0;

	// Prepare the data for this stack for this turn
	nlohmann::json j;
	j["stack_id"] = stack->unitId();
	const BattleHex &pos = stack->position;
	j["origin"] = {
		{"x", pos.getX()},
		{"y", pos.getY()},
		{"hex", pos.toInt()}
	};
	j["actions"] = nlohmann::json::array();

	// Add spellcaster flag
	bool isSpellcaster = stack->hasBonusOfType(BonusType::SPELLCASTER) && stack->canCast();
	j["is_spellcaster"] = isSpellcaster;

	bool canWait = !stack->waitedThisTurn;      // legal ⇔ unit has not waited yet
	j["can_wait"] = canWait;                    // write to JSON

	// Export known creature spells
	nlohmann::json creatureSpellsJson = nlohmann::json::array();
	TConstBonusListPtr bl = stack->getBonusesOfType(BonusType::SPELLCASTER);
	for (const auto &bonus : *bl)
	{
		if (bonus->additionalInfo[0] <= 0 && bonus->subtype.as<SpellID>().hasValue())
		{
			const CSpell *s = bonus->subtype.as<SpellID>().toSpell();
			if (s)
			{
				creatureSpellsJson.push_back({
					{"spell_id", s->id.getNum()},
					{"name", s->getNameTranslated()}
				});
			}
		}
	}
	j["creature_spells"] = creatureSpellsJson;

	// Determine creature spellcasting targets
	nlohmann::json spellcastTargets = nlohmann::json::array();

	if (isSpellcaster)
	{
		for (const auto &action : actions)
		{
			if (action.get() != PossiblePlayerBattleAction::NO_LOCATION)
				continue;

			const spells::Caster *caster = stack;
			const CSpell *spell = action.spell().toSpell();
			if (!spell)
				continue;

			spells::Target target;
			target.emplace_back();
			spells::BattleCast cast(owner.getBattle().get(), caster, spells::Mode::CREATURE_ACTIVE, spell);
			auto m = spell->battleMechanics(&cast);
			spells::detail::ProblemImpl ignored;

			if (m->canBeCastAt(target, ignored))
			{
				for (const CStack *unit : owner.getBattle()->battleGetAllStacks())
				{
					if (!unit->alive())
						continue;
					if (!isCastingPossibleHere(spell, unit, unit->getPosition()))
						continue;

					spellcastTargets.push_back({
						{"stack_id", unit->unitId()},
						{"x", unit->position.getX()},
						{"y", unit->position.getY()},
						{"hex", unit->position.toInt()}
					});
				}
				break;
			}
		}
	}
	j["creature_spellcast_possible"] = !spellcastTargets.empty();
	j["creature_spellcast_targets"] = spellcastTargets;

	// Hero spellcasting info
	const CGHeroInstance *castingHero = (owner.attackingHeroInstance && owner.attackingHeroInstance->tempOwner == owner.curInt->playerID)
										? owner.attackingHeroInstance
										: owner.defendingHeroInstance;

	if (castingHero && castingHero->hasSpellbook())
	{
		j["hero_spellcasting_available"] = true;
		j["hero_can_still_cast_this_round"] = owner.getBattle()->battleCanCastSpell(castingHero, spells::Mode::HERO) == ESpellCastProblem::OK;

		nlohmann::json heroSpells = nlohmann::json::array();
		for (const auto &spell : castingHero->getSpellsInSpellbook())
		{
			nlohmann::json spellJson;
			spellJson["spell_id"] = spell.getNum();
			spellJson["targets"] = nlohmann::json::array();

			const CSpell *spellPtr = spell.toSpell();
			if (spellPtr)
			{   
				spellJson["can_cast_now"] = spellPtr->canBeCast(owner.getBattle().get(), spells::Mode::HERO, castingHero);

				for (const CStack *unit : owner.getBattle()->battleGetAllStacks())
				{
					if (!unit->alive()) continue;
					if (!isCastingPossibleHere(spellPtr, unit, unit->getPosition())) continue;

					spellJson["targets"].push_back({
						{"stack_id", unit->unitId()},
						{"x", unit->position.getX()},
						{"y", unit->position.getY()},
						{"hex", unit->position.toInt()}
					});
				}
			}
			heroSpells.push_back(spellJson);
		}
		j["hero_spells"] = heroSpells;
	}
	else
	{
		j["hero_spellcasting_available"] = false;
		j["hero_spells"] = nlohmann::json::array();
	}

	for (const auto &action : actions)
	{
		nlohmann::json a;
		a["type"] = static_cast<int>(action.get());

		switch (action.get())
		{
			case PossiblePlayerBattleAction::MOVE_STACK:
			{
				a["reachable_tiles"] = nlohmann::json::array();
				BattleHexArray reachable = owner.getBattle()->battleGetAvailableHexes(stack, false);
				for (const BattleHex &hex : reachable)
				{
					a["reachable_tiles"].push_back({
						{"x", hex.getX()},
						{"y", hex.getY()},
						{"hex", hex.toInt()}
					});
				}
				break;
			}
			case PossiblePlayerBattleAction::ATTACK:
			case PossiblePlayerBattleAction::ATTACK_AND_RETURN:
			case PossiblePlayerBattleAction::WALK_AND_ATTACK:
			{
				a["melee_targets"] = nlohmann::json::array();
				for (const CStack *target : owner.getBattle()->battleGetAllStacks())
				{
					if (target != stack && target->alive() && target->unitSide() != stack->unitSide())
					{
						BattleHex fromHex;
						if (owner.fieldController)
							fromHex = owner.fieldController->fromWhichHexAttack(target->getPosition());

						nlohmann::json targetEntry = {
							{"stack_id", target->unitId()},
							{"x", target->position.getX()},
							{"y", target->position.getY()},
							{"hex", target->position.toInt()}
						};

						if (fromHex.isValid())
						{
							targetEntry["attack_from"] = {
								{"x", fromHex.getX()},
								{"y", fromHex.getY()},
								{"hex", fromHex.toInt()}
							};
						}

						// Add can_melee_attack field
						bool canMeleeAttack = false;
						if (fromHex.isValid() && fromHex.toInt() != target->position.toInt())
							canMeleeAttack = true;
						targetEntry["can_melee_attack"] = canMeleeAttack;

						a["melee_targets"].push_back(targetEntry);
					}
				}
				break;
			}
			case PossiblePlayerBattleAction::SHOOT:
			{
				if (stack->canShoot())
				{
					a["ranged_targets"] = nlohmann::json::array();
					for (const CStack *target : owner.getBattle()->battleGetAllStacks())
					{
						if (target != stack && target->alive() && target->unitSide() != stack->unitSide())
						{
							nlohmann::json targetEntry = {
								{"stack_id", target->unitId()},
								{"x", target->position.getX()},
								{"y", target->position.getY()},
								{"hex", target->position.toInt()}
							};
							a["ranged_targets"].push_back(targetEntry);
						}
					}
				}
				break;
			}
			case PossiblePlayerBattleAction::AIMED_SPELL_CREATURE:
			case PossiblePlayerBattleAction::ANY_LOCATION:
			case PossiblePlayerBattleAction::NO_LOCATION:
			case PossiblePlayerBattleAction::FREE_LOCATION:
			case PossiblePlayerBattleAction::OBSTACLE:
			case PossiblePlayerBattleAction::TELEPORT:
			case PossiblePlayerBattleAction::RANDOM_GENIE_SPELL:
			{
				a["spell_targeting"] = true;
				if (action.spell().hasValue())
					a["spell_id"] = static_cast<int>(action.spell().toEnum());

				const CSpell *spellPtr = action.spell().toSpell();
				a["spell_targets"] = nlohmann::json::array();
				for (const CStack *target : owner.getBattle()->battleGetAllStacks())
				{
					if (!target->alive()) continue;
					if (!spellPtr) continue;
					if (!isCastingPossibleHere(spellPtr, target, target->getPosition())) continue;

					a["spell_targets"].push_back({
						{"stack_id", target->unitId()},
						{"x", target->position.getX()},
						{"y", target->position.getY()},
						{"hex", target->position.toInt()}
					});
				}
				break;
			}
			case PossiblePlayerBattleAction::CREATURE_INFO:
			case PossiblePlayerBattleAction::HERO_INFO:
			case PossiblePlayerBattleAction::CATAPULT:
			case PossiblePlayerBattleAction::HEAL:
			{
				a["misc"] = true;
				break;
			}
			default:
				a["note"] = "Unhandled or unknown action type";
				break;
		}

		j["actions"].push_back(a);
	}

	// Add turn number
	j["turn"] = turnCounter;

	// Increment turn counter only when this function is executed
	++turnCounter;

	std::ofstream outFile(logFilePath, std::ios::trunc);
	outFile << j.dump(2); // pretty format
	outFile.close();
}


void BattleActionsController::activateStack()
{
	const CStack * s = owner.stacksController->getActiveStack();
	if(s)
	{
		tryActivateStackSpellcasting(s);

		possibleActions = getPossibleActionsForStack(s);
		exportPossibleActionsToJson(s, possibleActions);
		std::list<PossiblePlayerBattleAction> actionsToSelect;
		if(!possibleActions.empty())
		{
			auto primaryAction = possibleActions.front().get();

			if(primaryAction == PossiblePlayerBattleAction::SHOOT || primaryAction == PossiblePlayerBattleAction::AIMED_SPELL_CREATURE
				|| primaryAction == PossiblePlayerBattleAction::ANY_LOCATION || primaryAction == PossiblePlayerBattleAction::ATTACK_AND_RETURN)
			{
				actionsToSelect.push_back(possibleActions.front());

				auto shootActionPredicate = [](const PossiblePlayerBattleAction& action)
				{
					return action.get() == PossiblePlayerBattleAction::SHOOT;
				};
				bool hasShootSecondaryAction = std::any_of(possibleActions.begin() + 1, possibleActions.end(), shootActionPredicate);

				if(hasShootSecondaryAction) //casters may have shooting capabilities, for example storm elementals
					actionsToSelect.emplace_back(PossiblePlayerBattleAction::SHOOT);

				/* TODO: maybe it would also make sense to check spellcast as non-top priority action ("NO_SPELLCAST_BY_DEFAULT" bonus)?
				 * it would require going beyond this "if" block for melee casters
				 * F button helps, but some mod creatures may have that bonus and more than 1 castable spell */

				actionsToSelect.emplace_back(PossiblePlayerBattleAction::ATTACK); //always allow melee attack as last option
			}
		}
		owner.windowObject->setAlternativeActions(actionsToSelect);
	}
}

void BattleActionsController::onHexRightClicked(const BattleHex & clickedHex)
{
	bool isCurrentStackInSpellcastMode = creatureSpellcastingModeActive();

	if (heroSpellcastingModeActive() || isCurrentStackInSpellcastMode)
	{
		endCastingSpell();
		CRClickPopup::createAndPush(LIBRARY->generaltexth->translate("core.genrltxt.731")); // spell cancelled
		return;
	}

	auto selectedStack = owner.getBattle()->battleGetStackByPos(clickedHex, true);

	if (selectedStack != nullptr)
		ENGINE->windows().createAndPushWindow<CStackWindow>(selectedStack, true);

	if (clickedHex == BattleHex::HERO_ATTACKER && owner.attackingHero)
		owner.attackingHero->heroRightClicked();

	if (clickedHex == BattleHex::HERO_DEFENDER && owner.defendingHero)
		owner.defendingHero->heroRightClicked();
}

bool BattleActionsController::heroSpellcastingModeActive() const
{
	return heroSpellToCast != nullptr;
}

bool BattleActionsController::creatureSpellcastingModeActive() const
{
	auto spellcastModePredicate = [](const PossiblePlayerBattleAction & action)
	{
		return action.spellcast() || action.get() == PossiblePlayerBattleAction::SHOOT; //for hotkey-eligible SPELL_LIKE_ATTACK creature should have only SHOOT action
	};

	return !possibleActions.empty() && std::all_of(possibleActions.begin(), possibleActions.end(), spellcastModePredicate);
}

bool BattleActionsController::currentActionSpellcasting(const BattleHex & hoveredHex)
{
	if (heroSpellToCast)
		return true;

	if (!owner.stacksController->getActiveStack())
		return false;

	auto action = selectAction(hoveredHex);

	return action.spellcast();
}

const std::vector<PossiblePlayerBattleAction> & BattleActionsController::getPossibleActions() const
{
	return possibleActions;
}

void BattleActionsController::removePossibleAction(PossiblePlayerBattleAction action)
{
	vstd::erase(possibleActions, action);
}

void BattleActionsController::pushFrontPossibleAction(PossiblePlayerBattleAction action)
{
	possibleActions.insert(possibleActions.begin(), action);
}

void BattleActionsController::resetCurrentStackPossibleActions()
{
	possibleActions = getPossibleActionsForStack(owner.stacksController->getActiveStack());
}

// This function handles socket commands to control battle actions for the active stack.
// Available commands:
// - "move <hex>": Move to the specified hex tile.
// - "melee <target_hex> <from_hex>": Perform a melee attack on target_hex from from_hex.
// - "heal <target_id>": Heal the stack with the specified ID.
// - "shoot <target_id>": Shoot at the stack with the specified ID. (battle id like 0 or 1 or 2)
// - "cast <target_id> <spell_id>": Cast a creature spell at the specified target ID with the given spell ID.
// - "wait": Perform a wait action.
// - "defend": Perform a defend action.
// - "endtactic <side>": Ends the tactic phase for the given side (0 for attacker, 1 for defender).
// - "surrender <side>": Makes the specified side surrender (0 for attacker, 1 for defender).
// - "retreat <side>": Makes the specified side retreat (0 for attacker, 1 for defender).

void BattleActionsController::performSocketCommand(const std::string &cmd)
{
	const CStack* stack = owner.stacksController->getActiveStack();
	if (!stack)
	{
		logGlobal->warn("No active stack available.");
		return;
	}

	// move <hex>: Moves the active stack to the specified hex tile
	if (cmd.rfind("move ", 0) == 0)
	{
		std::string hexStr = cmd.substr(5);
		int hex = std::stoi(hexStr);

		logGlobal->info("Move command: stack %d -> hex %d", stack->unitId(), hex);

		BattleHex dest(hex);
		BattleAction action = BattleAction::makeMove(stack, dest);

		owner.curInt->cb->battleMakeUnitAction(owner.getBattleID(), action);
	}
	// wait: Makes the active stack wait
	else if (cmd == "wait")
	{
		logGlobal->info("Wait command: stack %d waits", stack->unitId());

		BattleAction action = BattleAction::makeWait(stack);
		owner.curInt->cb->battleMakeUnitAction(owner.getBattleID(), action);
	}
	// defend: Makes the active stack defend
	else if (cmd == "defend")
	{
		logGlobal->info("Defend command: stack %d defends", stack->unitId());

		BattleAction action = BattleAction::makeDefend(stack);
		owner.curInt->cb->battleMakeUnitAction(owner.getBattleID(), action);
	}
	// melee <target_hex> <from_hex>: Perform a melee attack
	else if (cmd.rfind("melee ", 0) == 0)
	{
		std::istringstream iss(cmd.substr(6));
		int targetHexInt, fromHexInt;
		iss >> targetHexInt >> fromHexInt;

		BattleHex targetHex(targetHexInt);
		BattleHex fromHex(fromHexInt);

		logGlobal->info("Melee command: stack %d attacks hex %d from hex %d", stack->unitId(), targetHexInt, fromHexInt);

		BattleAction action = BattleAction::makeMeleeAttack(stack, targetHex, fromHex, /*returnAfterAttack*/false);
		owner.curInt->cb->battleMakeUnitAction(owner.getBattleID(), action);
	}
	// shoot <target_id>: Shoots at the specified target stack
	else if (cmd.rfind("shoot ", 0) == 0)
	{
		int targetId = std::stoi(cmd.substr(6));

		const CStack* target = nullptr;
		for (const auto& s : owner.getBattle()->battleGetAllStacks())
		{
			if (s->unitId() == targetId)
			{
				target = s;
				break;
			}
		}

		if (target)
		{
			logGlobal->info("Shoot command: stack %d shoots at stack %d", stack->unitId(), target->unitId());
			BattleAction action = BattleAction::makeShotAttack(stack, target);
			owner.curInt->cb->battleMakeUnitAction(owner.getBattleID(), action);
		}
		else
		{
			logGlobal->warn("Shoot target with ID %d not found", targetId);
		}
	}	else
	{
		logGlobal->warn("Unknown command: %s", cmd.c_str());
	}
}
