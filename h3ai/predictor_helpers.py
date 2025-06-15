def extract_all_possible_commands(actions_data):
    """
    Parses possible_actions.json and returns a flat list of all concrete commands
    like: ["move 45", "melee 17 11", "defend", "wait"]
    """
    commands = []

    for action in actions_data.get("actions", []):
        type_id = action.get("type")

        # Defend
        if type_id == 0:
            commands.append("defend")

        # Wait
        elif type_id == 1:
            commands.append("wait")

        # Move: one command per reachable tile
        elif type_id == 4:
            for tile in action.get("reachable_tiles", []):
                commands.append(f"move {tile['hex']}")

        # Melee: one command per melee target
        elif type_id in (5, 6):
            for target in action.get("melee_targets", []):
                if target.get("can_melee_attack", False):
                    target_hex = target["hex"]
                    from_hex = target["attack_from"]["hex"]
                    commands.append(f"melee {target_hex} {from_hex}")

        # (Optional) Shoot or Cast Spell can go here later if added

    return commands
'''
Test with a loaded possible_actions.json (requires live input on user's machine)
Example:
with open("/Users/syntaxerror/vcmi/export/possible_actions.json") as f:
    actions_data = json.load(f)
commands = extract_all_possible_commands(actions_data)
print(commands)
'''