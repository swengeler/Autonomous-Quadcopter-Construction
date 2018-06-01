target_map_name = input("Specify the target map name (no spaces): ")
agent_types = str.split(input("Specify agent types to exclude: "))
excluded = []
if not (len(agent_types) == 0 or len(agent_types) == 1 and agent_types[0] == ""):
    excluded.extend(agent_types)




