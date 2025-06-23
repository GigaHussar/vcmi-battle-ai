ACTION_TYPE_MAP = {
    0: "defend",
    1: "wait",
    4: "move",
    5: "melee",
    # 6 is now ignored
}

def extract_all_possible_commands(actions_data):
    """
    Parses possible_actions.json and returns a list of structured commands:
      [
        {"type":"defend"},
        {"type":"wait"},
        {"type":"move",  "hex1": 45},
        {"type":"melee", "hex1": from_hex, "hex2": target_hex},
        ...
      ]
    """
    commands = []

    for action in actions_data.get("actions", []):
        type_id = action.get("type")
        action_type = ACTION_TYPE_MAP.get(type_id)
        if action_type is None:
            # skip unhandled types (including type 6)
            continue

        if action_type in ("defend", "wait"):
            # For "defend" and "wait", we just append the type
            commands.append({"type": action_type})

        elif action_type == "move":
            # For "move", we need to extract reachable tiles
            for tile in action.get("reachable_tiles", []):
                commands.append({
                    "type": "move",
                    "hex1": tile["hex"]
                })

        elif action_type == "melee":
            # For "melee", we need to extract hexes to move to and targets
            for target in action.get("melee_targets", []):
                if target.get("can_melee_attack", False):
                    commands.append({
                        "type": "melee",
                        "hex1": target["attack_from"]["hex"],
                        "hex2": target["hex"]
                    })

    return commands

'''
Test with a loaded possible_actions.json (requires live input on user's machine)
Example:
with open("/Users/syntaxerror/vcmi/export/possible_actions.json") as f:
    actions_data = json.load(f)
commands = extract_all_possible_commands(actions_data)
print(commands)
'''

def format_command_for_vcmi(action: dict) -> str:
    """
    Turn one of your action dicts into the VCMI CLI string.
    """
    t = action["type"]
    if t == "wait" or t == "defend":
        # VCMI accepts the same keyword
        return t
    elif t == "move":
        # e.g. "move 42"
        return f"move {action['hex1']}"
    elif t == "melee":
        # note: original JSON -> dict stored {"hex1":attack_from, "hex2":target}
        # but VCMI wants "melee <target_hex> <from_hex>"
        return f"melee {action['hex2']} {action['hex1']}"
    else:
        raise ValueError(f"Unknown action type `{t}`")