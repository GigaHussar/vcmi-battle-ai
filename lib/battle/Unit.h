/*
 * Unit.h, part of VCMI engine
 *
 * Authors: listed in file AUTHORS in main folder
 *
 * License: GNU General Public License v2.0 or later
 * Full text of license available in license.txt file, in main folder
 *
 */

#pragma once

#include <vcmi/Creature.h>
#include <vcmi/spells/Caster.h>

#include "../bonuses/Bonus.h"
#include "../bonuses/IBonusBearer.h"

#include "IUnitInfo.h"
#include "BattleHexArray.h"

VCMI_LIB_NAMESPACE_BEGIN

enum class EMetaText : uint8_t;
class MetaString;
class JsonNode;
class JsonSerializeFormat;

namespace battle
{

namespace BattlePhases
{
	enum Type
	{
		SIEGE, // turrets/catapult,
		NORMAL, // normal (unmoved) creatures, other war machines,
		WAIT_MORALE, // waited creatures that had morale,
		WAIT, // rest of waited creatures
		NUMBER_OF_PHASES // number of phases.
	};
}

// Healed HP (also drained life) and resurrected units info
struct HealInfo
{
	HealInfo() = default;
	HealInfo(int64_t healedHP, int32_t resurrected)
		: healedHealthPoints(healedHP), resurrectedCount(resurrected)
	{ }

	int64_t healedHealthPoints = 0;
	int32_t resurrectedCount = 0;

	HealInfo & operator+=(const HealInfo & other)
	{
		healedHealthPoints += other.healedHealthPoints;
		resurrectedCount += other.resurrectedCount;
		return *this;
	}
};

class CUnitState;

class DLL_LINKAGE Unit : public IUnitInfo, public spells::Caster, public virtual IBonusBearer, public ACreature
{
	static BattleHexArray::ArrayOfBattleHexArrays precomputeUnitHexes(BattleSide side, bool twoHex);

public:
	virtual ~Unit();

	virtual bool doubleWide() const = 0;

	virtual int32_t creatureIndex() const = 0;
	virtual CreatureID creatureId() const = 0;
	virtual int32_t creatureLevel() const = 0;
	virtual int32_t creatureCost() const = 0;
	virtual int32_t creatureIconIndex() const = 0;

	virtual bool ableToRetaliate() const = 0;
	virtual bool alive() const = 0;
	virtual bool isGhost() const = 0;
	virtual bool isFrozen() const = 0;

	bool isDead() const;
	bool isTurret() const;
	virtual bool isValidTarget(bool allowDead = false) const = 0; //non-turret non-ghost stacks (can be attacked or be object of magic effect)

	virtual bool isHypnotized() const = 0;
	virtual bool isInvincible() const = 0;

	virtual bool isClone() const = 0;
	virtual bool hasClone() const = 0;

	virtual bool canCast() const = 0;
	virtual bool isCaster() const = 0;
	virtual bool canShootBlocked() const = 0;
	virtual bool canShoot() const = 0;
	virtual bool isShooter() const = 0;

	/// returns initial size of this unit
	virtual int32_t getCount() const = 0;

	/// returns remaining health of first unit
	virtual int32_t getFirstHPleft() const = 0;

	/// returns total amount of killed in this unit
	virtual int32_t getKilled() const = 0;

	/// returns total health that unit still has
	virtual int64_t getAvailableHealth() const = 0;

	/// returns total health that unit had initially
	virtual int64_t getTotalHealth() const = 0;

	virtual int getTotalAttacks(bool ranged) const = 0;

	virtual BattleHex getPosition() const = 0;
	virtual void setPosition(const BattleHex & hex) = 0;

	virtual bool canMove(int turn = 0) const = 0; //if stack can move
	virtual bool defended(int turn = 0) const = 0;
	virtual bool moved(int turn = 0) const = 0; //if stack was already moved this turn
	virtual bool willMove(int turn = 0) const = 0; //if stack has remaining move this turn
	virtual bool waited(int turn = 0) const = 0;

	virtual std::shared_ptr<Unit> acquire() const = 0;
	virtual std::shared_ptr<CUnitState> acquireState() const = 0;

	virtual BattlePhases::Type battleQueuePhase(int turn) const = 0;

	virtual std::string getDescription() const;

	const BattleHexArray & getSurroundingHexes(const BattleHex & assumedPosition = BattleHex::INVALID) const; // get six or 8 surrounding hexes depending on creature size

	/// Returns list of hexes from which attacker can attack this unit
	BattleHexArray getAttackableHexes(const Unit * attacker) const;
	static const BattleHexArray & getSurroundingHexes(const BattleHex & position, bool twoHex, BattleSide side);

	bool coversPos(const BattleHex & position) const; //checks also if unit is double-wide

	const BattleHexArray & getHexes() const; //up to two occupied hexes, starting from front
	const BattleHexArray & getHexes(const BattleHex & assumedPos) const; //up to two occupied hexes, starting from front
	static const BattleHexArray & getHexes(const BattleHex & assumedPos, bool twoHex, BattleSide side);

	BattleHex occupiedHex() const; //returns number of occupied hex (not the position) if stack is double wide; otherwise -1
	BattleHex occupiedHex(const BattleHex & assumedPos) const; //returns number of occupied hex (not the position) if stack is double wide and would stand on assumedPos; otherwise -1
	static BattleHex occupiedHex(const BattleHex & assumedPos, bool twoHex, BattleSide side);

	///MetaStrings
	void addText(MetaString & text, EMetaText type, int32_t serial, const boost::logic::tribool & plural = boost::logic::indeterminate) const;
	void addNameReplacement(MetaString & text, const boost::logic::tribool & plural = boost::logic::indeterminate) const;
	std::string formatGeneralMessage(const int32_t baseTextId) const;

	int getRawSurrenderCost() const;

	//IConstBonusProvider
	const IBonusBearer* getBonusBearer() const override;

	//NOTE: save could possibly be const, but this requires heavy changes to Json serialization,
	//also this method should be called only after modifying object
	virtual void save(JsonNode & data) = 0;
	virtual void load(const JsonNode & data) = 0;

	virtual void damage(int64_t & amount) = 0;
	virtual HealInfo heal(int64_t & amount, EHealLevel level, EHealPower power) = 0;
};

class DLL_LINKAGE UnitInfo
{
public:
    uint32_t id = 0;
	TQuantity count = 0;
	CreatureID type;
	BattleSide side = BattleSide::NONE;
	BattleHex position;
	bool summoned = false;

	void serializeJson(JsonSerializeFormat & handler);

	void save(JsonNode & data);
	void load(uint32_t id_, const JsonNode & data);
};

}

VCMI_LIB_NAMESPACE_END
